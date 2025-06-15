#!/usr/bin/env python
"""
train_ae.py  –  train AE on pure‐noise windows from a 60/20/20 split of ALL windows
"""
import argparse, glob, numpy as np, torch, pytorch_lightning as pl, shutil
from pathlib import Path
from torch.utils.data           import DataLoader, TensorDataset
from torchmetrics               import MeanAbsoluteError
from torch.nn                   import L1Loss
from leela_ml.datamodules_npy   import StrikeDataset
from leela_ml.models.dae_unet_baseline   import UNet1D

p = argparse.ArgumentParser()
p.add_argument("--npy",     required=True)
p.add_argument("--meta",    required=True)
p.add_argument("--chunk",   type=int,   default=4096)
p.add_argument("--overlap", type=float, default=0.5)
p.add_argument("--bs",      type=int,   default=64)
p.add_argument("--epochs",  type=int,   default=20)
p.add_argument("--ckpt",    default="lightning_logs/ae_best.ckpt")
args = p.parse_args()
Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

# alias fallback
if args.npy.endswith("_wave.npy") and not Path(args.npy).exists():
    cand = sorted(glob.glob(args.npy.replace("_wave","_*.npy")))
    if not cand: raise FileNotFoundError()
    args.npy = cand[0]

# load full dataset (all windows, all labels)
ds       = StrikeDataset(args.npy, args.meta,
                         chunk_size=args.chunk,
                         overlap=args.overlap)
windows  = ds._windows    # shape (n_win, chunk)
labels   = ds.labels      # 0=noise, 1=burst
n_win    = len(ds)

# 60/20/20 sequential split on full windows
i1 = int(0.6 * n_win)
i2 = int(0.8 * n_win)
idx_tr_full = np.arange(0, i1)
idx_val_full= np.arange(i1, i2)
idx_te_full = np.arange(i2, n_win)
print(f"[info] full windows tr/va/te = {len(idx_tr_full)}/{len(idx_val_full)}/{len(idx_te_full)}")

# within train/val, pick only noise windows for training & validation
noise = np.where(labels==0)[0]
# intersect noise with each split
idx_tr_noise = np.intersect1d(idx_tr_full, noise)
idx_val_noise= np.intersect1d(idx_val_full, noise)
print(f"[info] noise windows tr/va = {len(idx_tr_noise)}/{len(idx_val_noise)}")

# build TensorDatasets for AE training
X_tr = torch.from_numpy(windows[idx_tr_noise]).unsqueeze(1)
X_va = torch.from_numpy(windows[idx_val_noise]).unsqueeze(1)
tr_ds = TensorDataset(X_tr)
va_ds = TensorDataset(X_va)

dl_tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=0)
dl_va = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=0)

# LightningModule -------------------------------------------------------------
class LitAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = UNet1D()
        self.loss = L1Loss()
        self.mae  = MeanAbsoluteError()
    def forward(self, x): return self.net(x)
    def training_step(self, batch, _):
        (x,) = batch
        rec = self(x); l = self.loss(rec, x)
        self.log("train_l1", l); return l
    def validation_step(self, batch, _):
        (x,) = batch
        rec = self(x)
        self.log("val_l1", self.loss(rec, x))
        self.mae.update(rec, x)
    def on_validation_epoch_end(self):
        self.log("val_mae", self.mae.compute(), prog_bar=True)
        self.mae.reset()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

ckpt_cb = pl.callbacks.ModelCheckpoint(
    dirpath=Path(args.ckpt).parent,
    filename=Path(args.ckpt).stem,
    monitor="val_mae", mode="min", save_top_k=1
)
trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    logger=False,
    callbacks=[ckpt_cb],
    num_sanity_val_steps=0
)
trainer.fit(LitAE(), dl_tr, dl_va)

# copy best ckpt safely
best = Path(ckpt_cb.best_model_path).resolve()
dst  = Path(args.ckpt).resolve()
if best.exists() and best != dst:
    shutil.copy(best, dst)
    print(f"✓ AE saved → {dst}")
else:
    print(f"✓ AE checkpoint at → {dst}")

# save full tr/va/te splits for eval
split_path = dst.with_suffix("").with_name(dst.stem + ".split.npz")
np.savez(split_path,
         idx_tr=idx_tr_full.astype(int),
         idx_val=idx_val_full.astype(int),
         idx_test=idx_te_full.astype(int))
print(f"✓ splits saved → {split_path}")
