
# Lightning Burst Detection with Deep Learning & Compression

## Introduction & Motivation
**Lightning Burst Detection** is a research‑oriented time‑series project focused on detecting short “burst” events (simulated lightning strikes) within a noisy continuous signal. The repository provides everything you need to:

* **Generate realistic synthetic data** that mimics real lightning bursts.
* **Train and evaluate** three complementary detection pipelines:
  1. **Compression‐based** Normalised Compression Distance (NCD) — zero‑training baseline.
  2. **Autoencoder anomaly detector** — unsupervised 1‑D U‑Net.
  3. **RawResNet1D classifier** — supervised deep learning.

The code is intentionally small and pedagogical, aimed at students or engineers who want to understand time‑series burst detection end‑to‑end.

---

## Project Layout

```text
├── scripts/                 # CLI entry‑points
│   ├── sim_make.py          # generate synthetic recording
│   ├── train_ae.py          # modern AE training (Lightning)
│   ├── train_ae_baseline.py # legacy AE training (raw PyTorch)
│   ├── train_resnet.py      # supervised ResNet training
│   ├── eval_ae.py           # AE burst detection
│   ├── eval_ae_baseline.py  # legacy AE evaluation
│   ├── eval_resnet.py       # ResNet evaluation
│   ├── train_gnn.py         # GNN lightning locator training
│   ├── eval_gnn.py          # evaluate GNN locator
│   └── run_ncd.py           # NCD detector (no training)
├── leela_ml/                # core library code
│   ├── signal_sim/          # synthetic waveform simulator
│   ├── datamodules_npy.py   # StrikeDataset window loader
│   ├── models/              # neural network definitions
│   │   ├── dae_unet.py
│   │   ├── dae_unet_baseline.py
│   │   ├── raw_resnet.py
│   │   └── ncd.py
│   ├── gnn_lightning/       # graph-based lightning locator
├── configs/                 # YAML hyper‑parameter files
├── data/                    # synthetic / real waveforms live here
├── reports/                 # metrics & plots land here
├── notebooks/               # interactive EDA demos
├── requirements.txt
└── README.md                # you are here
```

> **Dependencies:** Python ≥ 3.9, PyTorch 2 .x, PyTorch‑Lightning, NumPy, SciPy, scikit‑learn, matplotlib, seaborn.
> GPU optional – runs on CPU albeit slower.

## Installation

Create a fresh Python environment and install the requirements.  PyTorch and
PyTorch Geometric need to be installed separately (see the official install
guides for CUDA/CPU wheels).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio  # pick the wheel for your system
pip install torch_geometric
```

---

---

## 1 Generating Synthetic Data

The simulator creates a long noisy waveform with embedded burst events.

```bash
python scripts/sim_make.py     --minutes 5 \            # length of recording
    --out     data/storm5 \  # prefix for output files
    --seed    42             # RNG seed for repeatability
```

**Outputs**

| File | Purpose |
|------|---------|
| `storm5_<hash>.npy` | float32 waveform |
| `storm5_<hash>.json` | meta: sample‑rate + burst timestamps |
| `storm5_wave.npy` | *alias* copy of first channel for convenience |

The metadata lists burst start times; default burst length = **40 ms**. Noise floor includes pink‑noise + sensor white‑noise; bursts carry Gaussian envelopes plus harmonics. Drift and ADC clipping simulate real hardware quirks.

---

## 2 Unsupervised Pipelines

### 2·1 Autoencoder (modern)

```bash
python scripts/train_ae.py   --npy    data/storm5_wave.npy   --meta   data/storm5_meta.json   --chunk  4096 --overlap 0.5   --bs     128 --epochs 20   --depth  4 --base 16   --device cuda   --ckpt   lightning_logs/ae_best.ckpt
```

| Flag | Meaning | Typical |
|------|---------|---------|
| `--chunk` | window length (samples) | 2 k – 8 k |
| `--overlap` | data augmentation | 0.5 |
| `--depth` | U‑Net down/up levels | 4 |
| `--base` | filters in first conv | 16 / 32 |
| `--noise_std` | add Gaussian noise | 0.05–0.1 |

**Evaluation**

```bash
python scripts/eval_ae.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --ckpt lightning_logs/ae_best.ckpt   --chunk 512 --overlap 0.9   --mad_k 6 --win_ms 100 --fig_dark
```

Produces window‑ & event‑level metrics plus plots:

* `reports/ae_error_curve.png`
* `reports/ae_events.png`
* `reports/ae_event_timeline.png`

### 2·2 Compression (NCD)

```bash
python scripts/run_ncd.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json \
    --chunk 512 --overlap 0.9   --codec zlib --mad_k 6 --norm
```

Adding ``--norm`` normalises each window before compression which typically
improves contrast between noise and bursts. You can also call
``ncd_adjacent(wins, per_win_norm=True)`` from Python.

No training required. Flags bursts where NCD spikes above rolling median + k × MAD.

---

## 3 Supervised Pipeline (RawResNet1D)

### 3·1 Training

```bash
python scripts/train_resnet.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --chunk 8192 --overlap 0.75   --bs 64 --epochs 40   --accelerator gpu --devices 1   --ckpt lightning_logs/raw_best.ckpt
```

*Event‑aware* split ensures windows from the same burst never leak across train/val/test.  
Class imbalance handled by `WeightedRandomSampler`.

### 3·2 Evaluation

```bash
python scripts/eval_resnet.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --chunk 8192 --ckpt lightning_logs/raw_best.ckpt   --bs 512
```

Outputs VAL & TEST AUROC / F1 and saves `reports/resnet_val_test.png`.

---

## 4 Method Comparison

| Method | Training need | Typical Event‑F1 (synthetic) | Strengths | Weaknesses |
|--------|---------------|------------------------------|-----------|------------|
| **NCD** | none | 0.60–0.75 | zero setup, explainable | slower, many FP |
| **Autoencoder** | unsup. noise only | 0.80–0.90 | adapts, no labels | threshold tuning |
| **ResNet** | labelled bursts | 0.90–0.97 | highest accuracy | needs labels |

---

## 5 Running via Python API

```python
from leela_ml.models.dae_unet import UNet1D
from leela_ml.datamodules_npy import StrikeDataset
from leela_ml.ncd import ncd_adjacent

ds = StrikeDataset("data/storm5_wave.npy", "data/storm5_meta.json",
                   chunk_size=512, overlap=0.9)
x, _ = ds[0]           # torch Tensor (1, 512)
model = UNet1D(depth=4, base=16).eval()
with torch.no_grad():
    recon = model(x.unsqueeze(0))
err = (recon - x).abs().mean()
print("reconstruction error:", err.item())
```

You can likewise call `ncd_adjacent(ds.windows)` to get an NCD score vector.

---

## 6 Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA device not found` | Install CPU‑only wheel: `pip install torch==<ver>+cpu` |
| Large checkpoint rejected by GitHub | `git lfs install && git lfs track "*.pt"` |
| Training slow | Use `--precision 16`, reduce `--depth`, smaller `--chunk` |

---

## 7 Contributing

1. Fork & clone.  
2. Create feature branch.  
3. Run `ruff` + `black .`.  
4. Add unit tests under `tests/`.  
5. PR with clear description.

---

## 8 Graph Neural Network Locator

This repository now includes a proof-of-concept implementation of the graph
neural network workflow described by Tian et al. (2025). The simulator now
splits events into **train/val/test** subsets to avoid data leakage. Use
`simulate_dataset()` under `leela_ml/gnn_lightning/` to create multi-station
recordings. Training logs losses to CSV and writes a `gnn_training.png` curve.
Evaluation reports the mean location error and can plot predicted vs true
locations.

### Example

```bash
# ensure local imports resolve
export PYTHONPATH=$PWD

# generate a realistic multi-station recording (about five minutes)
python scripts/sim_gnn.py --minutes 5 \
    --out data/demo --stations data/synthetic/stations.json \
    --events_mult 5 --clusters_per_minute 6

# train for a good number of epochs and log the loss curve
python scripts/train_gnn.py --prefix data/demo --epochs 50 --bs 32 \
    --ckpt lightning_logs/gnn_best.ckpt

# disable denoising with `--no_denoise` if you want to train on raw snippets

# evaluate on the held‑out test set and plot predictions
python scripts/eval_gnn.py --prefix data/demo --split test \
    --ckpt lightning_logs/gnn_best.ckpt --plot
```

Training writes metrics to `lightning_logs/gnn/metrics.csv` and saves the
learning curve as `lightning_logs/gnn_training.png`.  Evaluation with
`--plot` creates `reports/gnn_pred_vs_true_test.png`.

---

## 9 Future Work

* Multichannel fusion (multiple sensors).  
* Streaming (real‑time) detection.  
* Variational / flow‑based models for richer probabilistic scoring.  
* Hyper‑parameter sweeps via Optuna.

---

*Project created on macOS, validated on Ubuntu 20.04 with CUDA 11.8.  Feel free to raise issues or PRs!*  


