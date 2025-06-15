"""
datamodules_npy.py
──────────────────
Light-weight dataset utilities for the lightning-detection toy project.
"""

from pathlib import Path
import json, numpy as np, torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
class StrikeDataset(Dataset):
    """
    Slice a single .npy waveform into fixed-length windows and give each window
    a binary label (1 = overlaps a synthetic lightning burst, 0 = pure noise).

    Parameters
    ----------
    npy_path : str
        Path to the .npy file for one station **or** an alias *_wave.npy.
    meta_path : str
        Path to the *_meta.json file produced by the simulator.
    chunk_size : int
        Number of samples per window.
    overlap : float in [0,1)
        0   → non-overlapping windows (hop == chunk_size)
        0.5 → 50 % overlap (hop == chunk_size / 2) etc.
    """
    def __init__(self,
                 npy_path: str,
                 meta_path: str,
                 chunk_size: int = 16_384,
                 overlap:    float = 0.0):

        self.npy_path   = Path(npy_path)
        self.meta_path  = Path(meta_path)
        self.chunk_size = int(chunk_size)
        self.overlap    = float(overlap)

        # ---------- load waveform lazily (memory-mapped) ---------------------
        self.wave  = np.load(self.npy_path, mmap_mode="r")
        self.meta  = json.load(open(self.meta_path))
        self.fs    = self.meta["fs"]                           # sample rate
        self.hop   = int(self.chunk_size * (1 - self.overlap))
        if self.hop <= 0:
            raise ValueError("overlap must be < 1.0")

        self.n_win = 1 + (len(self.wave) - self.chunk_size) // self.hop
        self._windows = np.lib.stride_tricks.as_strided(
            self.wave,
            shape   =(self.n_win, self.chunk_size),
            strides =(self.wave.strides[0]*self.hop, self.wave.strides[0]),
        ).astype("f4", copy=False)

        # ---------- build labels (1 if ANY overlap with a burst) -------------
        labels = np.zeros(self.n_win, dtype=np.uint8)
        for ev in self.meta["events"]:
            s0 = int(ev["t"] * self.fs)                    # start sample
            s1 = s0 + int(0.04 * self.fs)                 # +40 ms burst
            first = max(0,       (s0 - self.chunk_size) // self.hop + 1)
            last  = min(self.n_win - 1,  s1 // self.hop)
            labels[first:last+1] = 1
        self.labels = labels

    # ---------- PyTorch Dataset API ------------------------------------------
    def __len__(self) -> int:
        return self.n_win

    def __getitem__(self, idx: int):
        x = self._windows[idx]          # numpy view, shape (chunk_size,)
        # add channel-dim → (1,T)  ;  convert to torch.float32
        return (torch.from_numpy(x).unsqueeze(0),
                torch.tensor(float(self.labels[idx])))


# ── helper view that exposes only noise windows (label == 0) ------------------
class NoiseDataset(StrikeDataset):
    """Use this for noise-only training."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.noise_idx = np.where(self.labels == 0)[0]

    def __len__(self): return len(self.noise_idx)

    def __getitem__(self, i):
        # i is local index into noise-only array
        return super().__getitem__(int(self.noise_idx[i]))[0]   # drop label
