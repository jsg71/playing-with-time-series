#!/usr/bin/env python
"""Evaluate a trained GNN lightning locator."""

import argparse

import torch
from torch_geometric.loader import DataLoader

from leela_ml.gnn_lightning.dataset import LightningGraphDataset
from leela_ml.gnn_lightning.model import LightningGNN

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--window_ms", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--split", default="test", choices=["train", "val", "test"])
ap.add_argument("--plot", action="store_true")
ap.add_argument("--no_denoise", action="store_true", help="disable band-pass denoising")
args = ap.parse_args()

ds = LightningGraphDataset(
    args.prefix, split=args.split, window_ms=args.window_ms, denoise=not args.no_denoise
)
dl = DataLoader(ds, batch_size=args.bs)

state = torch.load(args.ckpt, map_location="cpu")
sd = state.get("state_dict", state)
if any(k.startswith("model.") for k in sd):
    sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
model = LightningGNN()
model.load_state_dict(sd)
model.eval()

err = 0.0
n = 0
preds = []
truth = []
with torch.no_grad():
    for data in dl:
        out = model(data)
        target = data.y.view(-1, 2)
        d = ((out - target) ** 2).sum(dim=1).sqrt() * 111.0
        err += d.sum().item()
        n += d.numel()
        preds.append(out)
        truth.append(target)

print(f"mean location error: {err / n:.2f} km")

if args.plot:
    import matplotlib.pyplot as plt
    from pathlib import Path

    p = torch.cat(preds).numpy()
    t = torch.cat(truth).numpy()
    fig, ax = plt.subplots()
    ax.scatter(t[:, 1], t[:, 0], s=10, label="true")
    ax.scatter(p[:, 1], p[:, 0], s=10, alpha=0.7, label="pred")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend()
    Path("reports").mkdir(exist_ok=True)
    fig.savefig(f"reports/gnn_pred_vs_true_{args.split}.png")
