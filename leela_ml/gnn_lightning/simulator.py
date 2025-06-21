import datetime
import json
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import List, Dict

import numpy as np

C = 3.0e5  # km/s ground-wave


def hav_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat, dlon = map(radians, (lat2 - lat1, lon2 - lon1))
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * R * asin(sqrt(a))


def make_noise(rng: np.random.Generator, N: int, t: np.ndarray) -> np.ndarray:
    x = rng.normal(0, 0.003, N)
    x += 0.002 * np.sin(2 * np.pi * 50 * t)
    x += 0.001 * np.sin(2 * np.pi * 62 * t)
    x += 0.001 * np.sin(2 * np.pi * 38 * t)
    x += 0.0015 * np.sin(2 * np.pi * 25 * t)
    return x.astype("f4")


def simulate_dataset(
    minutes: int,
    out_prefix: str,
    stations: List[Dict[str, float]],
    fs: int = 100_000,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    events_mult: float = 1.0,
    clusters_per_minute: int = 3,
) -> None:
    """Generate multi-station lightning data set for GNN training.

    Parameters
    ----------
    minutes : int
        Length of simulation.
    out_prefix : str
        Path prefix for generated files.
    stations : list of dict
        Station locations with ``id``, ``lat`` and ``lon``.
    fs : int, optional
        Sample rate (Hz).  Default ``100_000``.
    seed : int, optional
        RNG seed for repeatability.
    val_frac : float, optional
        Fraction of events reserved for validation.
    test_frac : float, optional
        Fraction of events reserved for test.
    events_mult : float, optional
        Scale factor for the number of flashes in each cluster.  ``1.0`` keeps
        the original configuration.
    clusters_per_minute : int, optional
        Number of event clusters per simulated minute.  More clusters yield a
        larger training set.
    """

    rng = np.random.default_rng(seed)
    N = fs * 60 * minutes
    t = np.arange(N) / fs
    waves = {s["id"]: make_noise(rng, N, t) for s in stations}
    events = []

    if minutes == 1:
        base_times = np.linspace(5, 55, clusters_per_minute)
    else:
        base_times = np.linspace(10, 60 * minutes - 10, clusters_per_minute * minutes)
    cluster_cfg = [
        ("near", (20, 50), (8, 12)),
        ("mid", (100, 200), (5, 9)),
        ("far", (400, 600), (3, 6)),
    ]

    for base_t, (name, d_rng, nf) in zip(base_times, cluster_cfg * minutes):
        nf = int(rng.integers(*nf) * events_mult)
        base_lat = rng.uniform(49, 53)
        base_lon = rng.uniform(-2, 4)
        for _ in range(nf):
            ev_t = base_t + rng.uniform(0, 2)
            d = rng.uniform(*d_rng)
            bearing = rng.uniform(0, 2 * np.pi)
            ev_lat = base_lat + (d / 111) * np.cos(bearing)
            ev_lon = base_lon + (d / 111) * np.sin(bearing) / np.cos(radians(base_lat))
            amp = rng.uniform(0.4, 1.0) / (1 + d / 50)
            freq = rng.uniform(3e3, 9e3)
            events.append(
                dict(
                    t=float(ev_t),
                    lat=float(ev_lat),
                    lon=float(ev_lon),
                    amp=float(amp),
                    freq=float(freq),
                    cluster=name,
                )
            )
            for s in stations:
                dist = hav_km(ev_lat, ev_lon, s["lat"], s["lon"])
                delay = dist / C
                i0 = int((ev_t + delay) * fs)
                dur = int(fs * 0.04)
                if i0 >= N:
                    continue
                subt = np.arange(dur) / fs
                burst = (
                    amp
                    * np.sin(2 * np.pi * freq * subt)
                    * np.exp(-subt / 0.003)
                    / (1 + dist / 50)
                )
                sl = slice(i0, min(i0 + dur, N))
                waves[s["id"]][sl] += burst[: sl.stop - sl.start]

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for s in stations:
        np.save(f"{out_prefix}_{s['id']}.npy", waves[s["id"]])

    meta = dict(
        fs=fs,
        utc_start=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        stations=stations,
        events=events,
    )

    # deterministic split of events for reproducible experiments -----------------
    idx = np.arange(len(events))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    n_test = int(len(idx) * test_frac)
    splits = {
        "val": idx[:n_val].tolist(),
        "test": idx[n_val : n_val + n_test].tolist(),
        "train": idx[n_val + n_test :].tolist(),
    }

    json.dump(meta, open(f"{out_prefix}_meta.json", "w"), indent=2)
    json.dump(splits, open(f"{out_prefix}_splits.json", "w"), indent=2)

    print(
        f"Saved {len(events)} flashes → {out_prefix}_*.npy (train/val/test = "
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})"
    )
