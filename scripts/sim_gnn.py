#!/usr/bin/env python
"""Generate multi-station dataset for GNN lightning locator."""
import argparse
import json
from pathlib import Path

from leela_ml.gnn_lightning.simulator import simulate_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=int, default=1)
ap.add_argument("--out", required=True, help="output file prefix")
ap.add_argument(
    "--stations", default="data/synthetic/stations.json", help="stations JSON"
)
ap.add_argument("--fs", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--test_frac", type=float, default=0.1)
args = ap.parse_args()

if Path(args.stations).is_file():
    stations = json.load(open(args.stations))["stations"]
else:
    stations = json.loads(args.stations)

simulate_dataset(
    args.minutes,
    args.out,
    stations,
    fs=args.fs,
    seed=args.seed,
    val_frac=args.val_frac,
    test_frac=args.test_frac,
)
