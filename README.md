
# Lightning Burst Detection with Deep Learning & Compression

## Introduction & Motivation

This project provides a complete pipeline for detecting **lightning bursts** (short, high‑intensity events in a time‑series signal) using both deep‑learning models and compression‑based algorithms. The goal is to identify when a lightning strike (or similar burst event) occurs within a noisy continuous signal.

We explore three complementary approaches:

| Approach | Learning Type | Key Script(s) | Training Needed? |
|----------|---------------|---------------|------------------|
| **NCD** (Normalised Compression Distance) | Heuristic / unsupervised | `run_ncd.py` | **No** |
| **Autoencoder (AE)** | Unsupervised deep learning | `train_ae.py`, `eval_ae.py`  | Yes — trains only on noise |
| **RawResNet1D** | Supervised deep learning | `train_resnet.py`, `eval_resnet.py` | Yes — needs labelled bursts |

A synthetic‑data simulator is included so you can benchmark everything end‑to‑end without hunting for real lightning recordings.

---

## Project Layout

```
├── scripts/              # Command‑line entry‑points
│   ├── sim_make.py
│   ├── train_ae.py
│   ├── train_ae_baseline.py
│   ├── eval_ae.py
│   ├── eval_ae_baseline.py
│   ├── run_ncd.py
│   ├── train_resnet.py
│   └── eval_resnet.py
├── leela_ml/             # Core library code
│   ├── signal_sim/       # Synthetic signal generator
│   ├── datamodules_npy.py
│   └── models/
│       ├── dae_unet.py
│       ├── raw_resnet.py
│       └── ncd.py
├── configs/              # YAML configs for Lightning scripts
├── data/                 # Generated data lives here
├── reports/              # Plots & metrics
└── requirements.txt
```

---

## 1  Environment (set‑up once)

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
python -m pip install -U pip
pip install -r requirements.txt    # installs PyTorch + Lightning + utils
```

*Linux + NVIDIA GPU* gives the best speed; Apple M‑series works via `--device mps`.  
CPU‑only machines are perfectly fine for small demos (just slower).

---

## 2  Generate Synthetic Data

```bash
python scripts/sim_make.py   --minutes 5   --out data/synthetic/storm5   --seed 42
```

| Flag        | Meaning | Default |
|-------------|---------|---------|
| `--minutes` | Real‑time length of recording to simulate | `5` |
| `--out`     | Prefix for output files | **required** |
| `--seed`    | RNG seed for full reproducibility | `0` |

Creates:

```
data/synthetic/
 ├─ storm5_0.npy        # raw waveform (float32)
 ├─ storm5_meta.json    # {"fs": 40000, "events":[…]}
 └─ storm5_wave.npy     # alias copy of first channel
```

---

## 3  Unsupervised Pipelines

### 3.1  Train Autoencoder (`train_ae.py`)

```bash
python scripts/train_ae.py   --npy   data/synthetic/storm5_wave.npy   --meta  data/synthetic/storm5_meta.json   --chunk 4096 --overlap 0.5   --bs 128 --epochs 20   --depth 4 --base 16   --ckpt lightning_logs/ae_best.ckpt
```

Important parameters:

| Flag | What it controls | Typical values |
|------|------------------|----------------|
| `--chunk` | Window length fed to AE | 2048 – 8192 |
| `--overlap` | Fractional overlap between windows | 0.3 – 0.8 |
| `--depth` / `--base` | U‑Net capacity | deeper = larger RF |
| `--noise_std` | Extra Gaussian noise during training | 0 – 0.2 |

Output: best checkpoint + `.split.npz` with train/val/test indices.

### 3.2  Detect Bursts with AE (`eval_ae.py`)

```bash
python scripts/eval_ae.py   --npy data/synthetic/storm5_wave.npy   --meta data/synthetic/storm5_meta.json   --ckpt lightning_logs/ae_best.ckpt   --chunk 512 --overlap 0.9   --mad_k 6 --win_ms 100 --fig_dark
```

Plots appear in `reports/` and metrics in console.  
Tune `--mad_k` to trade Precision ↔ Recall.

### 3.3  Compression Detector (`run_ncd.py`)

```bash
python scripts/run_ncd.py   --npy data/synthetic/storm5_wave.npy   --meta data/synthetic/storm5_meta.json   --chunk 512 --overlap 0.9   --codec zlib --mad_k 6
```

No training needed — baseline that works everywhere.

---

## 4  Supervised Pipeline

### 4.1  Train ResNet (`train_resnet.py`)

```bash
python scripts/train_resnet.py   --npy data/synthetic/storm5_wave.npy   --meta data/synthetic/storm5_meta.json   --chunk 8192 --overlap 0.75   --bs 64 --epochs 40   --ckpt lightning_logs/raw_best.ckpt
```

Event‑aware `GroupShuffleSplit` keeps every lightning event in exactly one split → no leakage.

### 4.2  Evaluate ResNet (`eval_resnet.py`)

```bash
python scripts/eval_resnet.py   --npy  data/synthetic/storm5_wave.npy   --meta data/synthetic/storm5_meta.json   --chunk 8192   --ckpt lightning_logs/raw_best.ckpt   --bs 512
```

Outputs AUROC + F1 for **val** & **test**, plus a dual‑panel plot.

---

## 5  Method Comparison (on synthetic, default params)

| Detector | Training need | Event F1 (≈) | Strengths | Weaknesses |
|----------|---------------|--------------|-----------|------------|
| **NCD**  | none          | 0.65 | zero setup | sensitive to any change |
| **AE**   | noise only    | 0.87 | unsupervised, adaptive | needs threshold tuning |
| **ResNet** | labelled bursts | 0.95+ | highest accuracy | needs labels, training |

---

## 6  Troubleshooting

| Symptom | Fix |
|---------|-----|
| *CUDA device not found* | `pip install torch==x.y.z+cpu -f https://download.pytorch.org/whl/torch_stable.html` |
| AE flags too many FP | increase `--mad_k` or retrain with lower `--noise_std` |
| ResNet overfits | larger `--chunk`, add dropout, or more training data |

---

## 7  Contributing

1. Run `ruff` and `black` before committing.  
2. Add/adjust unit tests in `tests/`.  
3. Open an issue for big changes first.

---

### Future Ideas

* Multi‑channel fusion, streaming inference, zstd codec for NCD, hyper‑param sweeps with Optuna, explainable Grad‑CAM for ResNet, etc.
 🚀

