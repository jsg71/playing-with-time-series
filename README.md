# Playing-with-Time-Series  📈

A compact playground for experimenting with time-series data using two parallel tracks:

| Track | Purpose | Main entry-points |
|-------|---------|-------------------|
| **Legacy pipeline** | Reproduce the original auto-encoder baseline exactly. | `scripts/train_ae_baseline.py`, `scripts/eval_ae_baseline.py` |
| **Modern pipeline** | Cleaner PyTorch-Lightning workflow with ResNet, Denoising-UNet and an NCD metric. | `scripts/train_resnet.py`, `scripts/train_ae.py`, `scripts/eval_resnet.py`, `scripts/eval_ae.py`, `scripts/run_ncd.py` |

---

## 1 Quick start

```bash
git clone https://github.com/jsg71/playing-with-time-series.git
cd playing-with-time-series

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2 Folder map

```
leela_ml/             ← importable package
├─ models/            model definitions (AE, ResNet, UNet, NCD…)
├─ signal_sim/        synthetic data generator
scripts/              training / evaluation entry points
notebooks/            exploratory Jupyter notebooks
requirements.txt      Python dependencies
```

> **Note:** large datasets / checkpoints are **not** tracked.  Keep them outside Git or use Git LFS.

---

## 3 Legacy pipeline

### 3.1 Train the baseline AE

```bash
python scripts/train_ae_baseline.py \
       --data_dir /path/to/data \
       --epochs 50 --batch_size 128 --lr 1e-3 \
       --out_dir runs/legacy_ae
```

### 3.2 Evaluate the baseline

```bash
python scripts/eval_ae_baseline.py \
       --data_dir /path/to/test_data \
       --ckpt runs/legacy_ae/best.pth
```

---

## 4 Modern pipeline (PyTorch-Lightning)

### 4.1 Training examples

```bash
# ResNet classifier
python scripts/train_resnet.py \
       --config configs/resnet.yaml \
       --data_dir /path/to/data \
       --accelerator gpu --devices 1

# Denoising AE / UNet
python scripts/train_ae.py \
       --config configs/dae.yaml \
       --data_dir /path/to/data
```

### 4.2 Evaluation

```bash
python scripts/eval_resnet.py --ckpt lightning_logs/version_x/ckpt.ckpt --data_dir /path/to/test
python scripts/eval_ae.py     --ckpt lightning_logs/version_y/ckpt.ckpt --data_dir /path/to/test
```

### 4.3 NCD metric demo

```bash
python scripts/run_ncd.py --pred_file reports/preds.csv --gt_file reports/ground_truth.csv
```

---

## 5 Running the notebooks

```bash
jupyter lab
# or: jupyter notebook
```

Set `DATA_DIR` in the first cell to point at your data folder.

---

## 6 Troubleshooting

| Problem                    | Remedy                                                          |
|----------------------------|-----------------------------------------------------------------|
| CUDA not detected          | `pip install torch==<cuda-ver>` or run CPU (`--accelerator cpu`). |
| Git refuses files > 100 MB | Keep them outside Git or track with **Git LFS**.                |
| Training slow on laptop    | Use `--precision 16`, smaller YAML config, or run on CPU.       |

---

## 7 Contributing

1. Open an Issue for major changes.  
2. Follow **PEP 8**; run `ruff` and `black` before committing.  
3. Add tests where sensible.

---

## 8 Licence

Released under the **MIT Licence** – see `LICENSE`.

---

## 9 Citation

```text
@misc{playingwithts2025,
  author       = {Goodacre, J.},
  title        = {Playing with Time-Series},
  howpublished = {GitHub},
  year         = {2025},
  url          = {https://github.com/jsg71/playing-with-time-series}
}
```
