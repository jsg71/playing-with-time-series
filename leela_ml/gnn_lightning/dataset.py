import json
from pathlib import Path
from math import radians, sin, cos, asin, sqrt
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

C = 3.0e5  # km/s ground wave


def hav_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat, dlon = map(radians, (lat2 - lat1, lon2 - lon1))
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R * asin(sqrt(a))


class LightningGraphDataset(Dataset):
    """Event-centric graphs from multi-station waveforms."""

    def __init__(self, prefix: str, window_ms: float = 2.0, ds: int = 16) -> None:
        self.prefix = prefix
        meta_path = Path(f"{prefix}_meta.json")
        self.meta = json.load(open(meta_path))
        self.fs = self.meta["fs"]
        self.window = int(window_ms * self.fs / 1000)
        self.ds = int(ds)
        self.half = self.window // 2
        self.waves = {
            s["id"]: np.load(f"{prefix}_{s['id']}.npy", mmap_mode="r")
            for s in self.meta["stations"]
        }
        self.events = self.meta["events"]
        n = len(self.meta["stations"])
        src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Data:
        ev = self.events[idx]
        node_feat: List[np.ndarray] = []
        pos: List[List[float]] = []
        for s in self.meta["stations"]:
            dist = hav_km(ev["lat"], ev["lon"], s["lat"], s["lon"])
            delay = dist / C
            i0 = int((ev["t"] + delay) * self.fs)
            w = self.waves[s["id"]]
            start = max(0, i0 - self.half)
            end = start + self.window
            if end > len(w):
                pad = end - len(w)
                arr = np.pad(w[start:], (0, pad))
            else:
                arr = w[start:end]
            arr = arr.astype("f4")
            if self.ds > 1:
                arr = arr[: len(arr) // self.ds * self.ds].reshape(-1, self.ds).mean(1)
            node_feat.append(arr)
            pos.append([s["lat"], s["lon"]])
        x = torch.tensor(np.stack(node_feat), dtype=torch.float32)
        pos = torch.tensor(pos, dtype=torch.float32)
        y = torch.tensor([[ev["lat"], ev["lon"]]], dtype=torch.float32)
        return Data(x=x, pos=pos, edge_index=self.edge_index, y=y)
