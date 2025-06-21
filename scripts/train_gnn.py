#!/usr/bin/env python
"""
train_gnn.py -- Train the GNN lightning location model proposed by Tian et al.
"""

import argparse
import shutil
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import torch

from leela_ml.gnn_lightning.dataset import LightningGraphDataset
from leela_ml.gnn_lightning.model import LightningGNN


ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True, help="dataset file prefix")
ap.add_argument("--window_ms", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--ckpt", default="lightning_logs/gnn_best.ckpt")
args = ap.parse_args()
log_dir = Path(args.ckpt).parent
log_dir.mkdir(parents=True, exist_ok=True)

train_ds = LightningGraphDataset(args.prefix, split="train", window_ms=args.window_ms)
val_ds = LightningGraphDataset(args.prefix, split="val", window_ms=args.window_ms)
train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=args.bs)


class Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LightningGNN(train_ds[0].x.shape[1])
        self.loss = torch.nn.MSELoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, _):
        out = self(batch)
        target = batch.y.view(-1, 2)
        loss = self.loss(out, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        out = self(batch)
        target = batch.y.view(-1, 2)
        loss = self.loss(out, target)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


ckpt_cb = pl.callbacks.ModelCheckpoint(
    dirpath=log_dir,
    filename="tmp",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    auto_insert_metric_name=False,
)
logger = pl.loggers.CSVLogger(log_dir, name="gnn")
pl.Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    logger=logger,
    callbacks=[ckpt_cb],
    num_sanity_val_steps=0,
).fit(Lit(), train_dl, val_dl)

best = Path(ckpt_cb.best_model_path)
if best.is_file():
    shutil.copy(best, args.ckpt)
    shutil.copy(best, log_dir / "gnn.ckpt")
    print("\u2713 best model \u2192", args.ckpt)

# plot learning curve ----------------------------------------------------------
try:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
    fig, ax = plt.subplots()
    ax.plot(df["step"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        ax.plot(df["step"], df["val_loss"], label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(log_dir / "gnn_training.png")
except Exception as e:  # pragma: no cover - plotting optional
    print("[warn] failed to plot training curve:", e)
