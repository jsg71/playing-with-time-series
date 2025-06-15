#!/usr/bin/env python
"""
train_resnet.py – RawResNet1D with event-level 60/20/20 split.
(unchanged logic; just uses chunk_size & overlap expected by StrikeDataset)
"""
import argparse, glob, shutil
from pathlib import Path
import numpy as np, torch, pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import GroupShuffleSplit
from leela_ml.datamodules_npy  import StrikeDataset   # signature uses chunk_size, overlap
from leela_ml.models.raw_resnet import RawResNet1D

# ─── CLI ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--npy",    required=True)
ap.add_argument("--meta",   required=True)
ap.add_argument("--chunk",  type=int,  default=8192, help="window length")
ap.add_argument("--overlap",type=float,default=0.75, help="fractional overlap 0–1")
ap.add_argument("--bs",     type=int,  default=64)
ap.add_argument("--epochs", type=int,  default=40)
ap.add_argument("--ckpt",   default="lightning_logs/raw_best.ckpt")
args = ap.parse_args()
log_dir = Path(args.ckpt).parent; log_dir.mkdir(parents=True, exist_ok=True)

# alias fallback ---------------------------------------------------------------
if args.npy.endswith("_wave.npy") and not Path(args.npy).is_file():
    alt = sorted(glob.glob(args.npy.replace("_wave", "_*.npy"))); args.npy = alt[0]

# dataset (note the right kwargs) ---------------------------------------------
ds = StrikeDataset(args.npy, args.meta,
                   chunk_size=args.chunk,
                   overlap=args.overlap)

labels = ds.labels.astype(int)

# build event-id for every chunk (noise chunks keep unique ids) ---------------
groups = np.arange(len(ds), dtype=int)   # each chunk its own group
evt = max(groups) + 1
i = 0
while i < len(ds):
    if labels[i]:
        j = i
        while j < len(ds) and labels[j]:
            groups[j] = evt
            j += 1
        evt += 1
        i = j
    else:
        i += 1

# GroupShuffleSplit 60/20/20 ---------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
tmp_tr, tmp_rest = next(gss.split(np.zeros(len(ds)), labels, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
idx_val, idx_test = next(gss2.split(np.zeros(len(tmp_rest)),
                                    labels[tmp_rest], groups[tmp_rest]))
idx_tr   = np.sort(tmp_tr)
idx_val  = np.sort(tmp_rest[idx_val])
idx_test = np.sort(tmp_rest[idx_test])

np.savez(log_dir/"split.npz", idx_tr=idx_tr, idx_val=idx_val, idx_test=idx_test)
print(f"[info] chunks   train/val/test = {len(idx_tr)}/{len(idx_val)}/{len(idx_test)}")
print(f"[info] strikes  train/val/test = {labels[idx_tr].sum()}/"
      f"{labels[idx_val].sum()}/{labels[idx_test].sum()}")

# DataLoaders ------------------------------------------------------------------
pos = labels[idx_tr].sum(); neg = len(idx_tr) - pos
w_pos = neg/pos if pos else 1.0
weights = [w_pos if labels[i] else 1.0 for i in idx_tr]
dl_tr  = DataLoader(Subset(ds, idx_tr), args.bs,
                    sampler=WeightedRandomSampler(weights, len(idx_tr), True),
                    num_workers=0)
dl_val = DataLoader(Subset(ds, idx_val), args.bs, shuffle=False, num_workers=0)

# Lightning module ------------------------------------------------------------
class Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = RawResNet1D()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.auc  = BinaryAUROC()
    def forward(self,x): return self.net(x)
    def training_step(self,batch,_):
        x,y=batch; loss=self.loss(self(x),y); self.log("train_loss",loss); return loss
    def validation_step(self,batch,_):
        x,y=batch; out=self(x); self.log("val_loss",self.loss(out,y))
        self.auc.update(torch.sigmoid(out), y.int())
    def on_validation_epoch_end(self):
        self.log("val_auc", self.auc.compute(), prog_bar=True); self.auc.reset()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=log_dir, filename="tmp",
                                       monitor="val_auc", mode="max",
                                       save_top_k=1, auto_insert_metric_name=False)
pl.Trainer(max_epochs=args.epochs, accelerator="auto", logger=False,
           callbacks=[ckpt_cb], num_sanity_val_steps=0).fit(Lit(), dl_tr, dl_val)

best = Path(ckpt_cb.best_model_path)
if best.is_file():
    shutil.copy(best, args.ckpt)
    shutil.copy(best, log_dir/"raw.ckpt")
    print("✓ best model →", args.ckpt)
