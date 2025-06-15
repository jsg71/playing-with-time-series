# Playing‑with‑Time‑Series  📈

A compact, pedagogy‑friendly codebase for experimenting with time‑series classification & reconstruction.  
It contains **two fully‑working tracks**:

| Track | Purpose | Main entry‑points |
|-------|---------|-------------------|
| **Legacy pipeline** | Reproduce the original Auto‑Encoder baseline exactly, using raw PyTorch. | `scripts/train_ae_baseline.py`, `scripts/eval_ae_baseline.py` |
| **Modern pipeline** | Cleaner PyTorch‑Lightning workflow with modular configs, ResNet & DAE/UNet back‑ends plus an NCD metric. | `scripts/train_resnet.py`, `scripts/train_ae.py`, `scripts/eval_resnet.py`, `scripts/eval_ae.py`, `scripts/run_ncd.py` |

---

## 0 Prerequisites

* Python ≥ 3.9  
* Pip ≥ 23  
* (optional) NVIDIA GPU + CUDA 11.x drivers

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1 Generating synthetic data

```bash
python - <<'PY'
from leela_ml.signal_sim.simulator import make_dataset

make_dataset(
    out_dir="data/synthetic",   # created if it doesn’t exist
    n_series=10_000,            # number of sample windows
    length=2048,                # points per window
    noise_std=0.05,             # Gaussian noise level
    seed=42
)
PY
```

The script produces:

* `signals.npy` – `(n_series, length)` float32 array  
* `labels.npy`  – class index per series  
* `meta.json`   – parameters used for reproducibility

---

## 2 Legacy pipeline

### 2·1 Training (baseline Auto‑Encoder)

```bash
python scripts/train_ae_baseline.py \
  --data_dir data/synthetic \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --latent_dim 64 \
  --out_dir runs/legacy_ae
```

| Flag | Meaning | Sensible range |
|------|---------|---------------|
| `--batch_size` | samples per optimiser step | 64 – 512 |
| `--lr` | Adam learning rate | 1e‑4 – 1e‑2 |
| `--latent_dim` | size of bottleneck vector | 32 – 256 |

Checkpoints & TensorBoard logs land in `runs/legacy_ae/`.

### 2·2 Evaluation

```bash
python scripts/eval_ae_baseline.py \
  --data_dir data/synthetic \
  --ckpt runs/legacy_ae/best.pth \
  --metrics mse psnr
```

Results (`recon_errors.csv`) and plots go to `reports/`.

---

## 3 Modern pipeline (PyTorch‑Lightning)

### 3·1 Config‑driven training

#### a) ResNet classifier

```bash
python scripts/train_resnet.py \
  --config configs/resnet.yaml \
  --data_dir data/synthetic \
  --accelerator gpu --devices 1 \
  --precision 16 \
  --max_epochs 100 \
  --early_stop_patience 10
```

Key config knobs (see `configs/resnet.yaml`):

| Field | Description | Default |
|-------|-------------|---------|
| `model.depth` | number of residual blocks | 34 |
| `optim.lr` | initial LR for AdamW | 1e‑3 |
| `sched` | cosine schedule with warm‑up | enabled |

#### b) Denoising AE / UNet

```bash
python scripts/train_ae.py \
  --config configs/dae.yaml \
  --data_dir data/synthetic \
  --noise_std 0.1 \
  --checkpoint_every_n_epochs 5
```

### 3·2 Evaluation & inference

```bash
python scripts/eval_resnet.py \
  --ckpt lightning_logs/version_7/checkpoints/epoch=89-step=2000.ckpt \
  --data_dir data/holdout

python scripts/eval_ae.py \
  --ckpt lightning_logs/version_3/checkpoints/epoch=44-step=1000.ckpt \
  --data_dir data/holdout
```

### 3·3 NCD metric demo

```bash
python scripts/run_ncd.py \
  --pred_file reports/preds_resnet.csv \
  --gt_file reports/gt.csv
```

Outputs **Normalised Compression Distance** scores for pairwise series.

---

## 4 Notebook workflow

```bash
jupyter lab
# open notebooks/01_eda.ipynb or 02_eda.ipynb
# set DATA_DIR in the first cell if needed
```

---

## 5 Troubleshooting

| Issue | Fix |
|-------|-----|
| *CUDA device not found* | `pip install torch==2.3.0+cpu` *(CPU‑only)* or install matching CUDA wheel |
| *Git rejects >100 MB file* | Keep data outside repo or use `git lfs` |
| *Training slow* | `--precision 16`, lower `batch_size`, fewer epochs |

---

## 6 Contributing

Pull requests welcome!  
* Run `ruff` and `black` before committing.  
* Add/adjust unit tests in `tests/`.  
* Discuss large‑scale changes via an Issue first.

---

## 7 Licence

MIT – see `LICENSE`.

---

## 8 Citation

```text
@misc{goodacre2025playing,
  author       = {Goodacre, J.},
  title        = {{Playing with Time-Series}},
  howpublished = {GitHub},
  year         = {2025},
  url          = {https://github.com/jsg71/playing-with-time-series}
}
```

Happy experimenting!
