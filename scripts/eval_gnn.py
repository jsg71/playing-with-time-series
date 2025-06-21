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
args = ap.parse_args()

ds = LightningGraphDataset(args.prefix, window_ms=args.window_ms)
dl = DataLoader(ds, batch_size=args.bs)

state = torch.load(args.ckpt, map_location="cpu")
model = LightningGNN(ds[0].x.shape[1])
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()

err = 0.0
n = 0
with torch.no_grad():
    for data in dl:
        out = model(data)
        d = ((out - data.y) ** 2).sum(dim=1).sqrt() * 111.0
        err += d.sum().item()
        n += d.numel()

print(f"mean location error: {err / n:.2f} km")
