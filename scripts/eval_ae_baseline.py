#!/usr/bin/env python3
"""
eval_ae.py ─ lightning-burst detection with a single (or dual-channel) auto-encoder.

❱  Example
python scripts/eval_ae.py \
    --npy   data/synthetic/storm5_wave.npy \
    --meta  data/synthetic/storm5_meta.json \
    --ckpt  lightning_logs/ae_best.ckpt \
    --chunk 512 --overlap 0.9 \
    --batch 8192 --mad_k 6 --win_ms 40 \
    --device mps  --fig_dark
"""

import argparse, glob, json, numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch
from pathlib import Path
from tqdm              import tqdm
from scipy.ndimage     import binary_dilation, binary_erosion
from sklearn.metrics   import precision_recall_fscore_support, roc_auc_score
from leela_ml.datamodules_npy import StrikeDataset
from leela_ml.models.dae_unet_baseline  import UNet1D

# ────────────── CLI ──────────────
pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
pa.add_argument("--npy",        required=True, help="*_wave.npy or single-station file")
pa.add_argument("--meta",       required=True, help="*_meta.json")
pa.add_argument("--ckpt",       default="lightning_logs/ae_best.ckpt", help="AE checkpoint")
pa.add_argument("--chunk",      type=int,   default=512,  help="window length (samples)")
pa.add_argument("--overlap",    type=float, default=0.9,  help="window overlap fraction")
pa.add_argument("--batch",      type=int,   default=8192, help="windows per forward pass")
pa.add_argument("--device",     choices=["auto","cpu","mps","cuda"], default="auto")
pa.add_argument("--mad_k",      type=float, default=6.0,  help="MAD multiplier for thr")
pa.add_argument("--win_ms",     type=float, default=40.0, help="expected burst length ms")
pa.add_argument("--fig_dark",   action="store_true", help="dark figure style")
pa.add_argument("--dpi",        type=int,   default=180, help="figure dpi")
args = pa.parse_args()
Path("reports").mkdir(exist_ok=True)

# ───────────── device ─────────────
def best_device():
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device in ("cuda","auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if args.device in ("mps","auto") and torch.backends.mps.is_available():
        try:                       # quick test forward
            UNet1D().to("mps")(torch.zeros(1,1,args.chunk,device="mps"))
            return torch.device("mps")
        except Exception: pass
    return torch.device("cpu")
dev = best_device();  print("• device:", dev)

# ───────────── data ──────────────
ds  = StrikeDataset(args.npy, args.meta, args.chunk, args.overlap)
win = ds._windows
lab = ds.labels.astype(bool)
hop = ds.hop
print(f"• windows: {len(win):,}   burst-windows: {lab.sum():,}")

# ───── optional VAL/TEST indices ─────
split_file = Path(args.ckpt).with_suffix("").with_name(Path(args.ckpt).stem+".split.npz")
idx_val = idx_test = None
if split_file.exists():
    sp        = np.load(split_file)
    idx_val   = sp["idx_val"]
    idx_test  = sp["idx_test"]
else:
    print("• split file missing – AUROC skipped")

# ───────────── model ──────────────
state = torch.load(args.ckpt, map_location="cpu")
state = state.get("state_dict", state)
state = {k[4:] if k.startswith("net.") else k: v for k, v in state.items()}
net   = UNet1D().to(dev).eval(); net.load_state_dict(state, strict=False)

# ───────────── forward ────────────
err = np.empty(len(win), dtype=np.float32)
with torch.no_grad():
    for i in tqdm(range(0, len(win), args.batch), desc="infer", unit="batch"):
        j = min(len(win), i+args.batch)
        x = torch.as_tensor(win[i:j]).unsqueeze(1).to(dev, non_blocking=True)
        r = net(x)
        err[i:j] = (r-x).abs().mean((1,2)).cpu().numpy()

# ─── rolling median / MAD threshold ──
from pandas import Series
win_len = int(args.win_ms/1000 * ds.fs / hop) or 1
roll    = Series(err).rolling(win_len, center=True, min_periods=1)
med     = roll.median().values
mad     = roll.apply(lambda v: np.median(np.abs(v-np.median(v))), raw=True).values
thr     = med + args.mad_k * mad
mask    = err > thr

nhit    = max(1, win_len//4)           # enforce min burst duration
mask    = binary_dilation(mask, iterations=nhit)
mask    = binary_erosion (mask, iterations=nhit)

# ───────────── metrics ─────────────
P,R,F,_ = precision_recall_fscore_support(lab, mask, average="binary", zero_division=0)
if idx_val is not None:
    auc_val  = roc_auc_score(lab[idx_val],  err[idx_val]) if lab[idx_val].any()  else np.nan
    auc_test = roc_auc_score(lab[idx_test], err[idx_test]) if lab[idx_test].any() else np.nan
    print(f"Window VAL AUROC {auc_val:.3f}  TEST {auc_test:.3f}", end="   ")
print(f"P={P:.3f} R={R:.3f} F1={F:.3f}")

def runs(b):
    out=[]; active=False
    for i,v in enumerate(b):
        if v and not active: active,start=True,i
        if not v and active: out.append((start,i-1)); active=False
    if active: out.append((start,len(b)-1))
    return out
pred_evt, true_evt = runs(mask), runs(lab)
tp=0; matched=[False]*len(true_evt)
for ps,pe in pred_evt:
    for k,(ts,te) in enumerate(true_evt):
        if matched[k]: continue
        if not (pe<ts or ps>te): tp+=1; matched[k]=True; break
prec_evt = tp/len(pred_evt) if pred_evt else 0
rec_evt  = tp/len(true_evt) if true_evt else 0
f1_evt   = 2*prec_evt*rec_evt/(prec_evt+rec_evt+1e-9)
print(f"Event  P={prec_evt:.3f} R={rec_evt:.3f} F1={f1_evt:.3f}")

# ───────────── plots ──────────────
sns.set_theme(style="darkgrid")
if args.fig_dark:
    plt.rcParams.update(
        {"figure.facecolor": "#111", "axes.facecolor": "#111",
         "axes.edgecolor": "#888", "text.color": "#ddd",
         "xtick.color": "#ddd", "ytick.color": "#ddd"}
    )

dpi = args.dpi
fig_w = 16  # inches

# 1) ERROR CURVE ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_w, 4), dpi=dpi)
ax.semilogy(err, lw=.6, color="#03A9F4", label="L₁ error")
ax.semilogy(thr, lw=1.2, ls="--", color="#FF9800", label="thr")
# put x-markers 30 % above local threshold so they never hide
tops = thr * 1.3
ax.scatter(np.where(lab)[0], tops[lab], marker="x", s=28,
           color="#FFFFFF", label="truth")
ax.set_title("Reconstruction error (log)")
ax.set_ylabel("L₁ error")
ax.legend(framealpha=.25, fontsize=9)
plt.tight_layout()
plt.savefig("reports/ae_err.png", dpi=dpi)

# helper: index->seconds
sec = lambda w: (w*hop)/ds.fs

# 2) RAW WAVEFORM + SPANS -------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_w, 4), dpi=dpi)
dec = max(1, int(ds.fs // 1000))
t   = np.arange(0, len(ds.wave), dec) / ds.fs
ax.plot(t, ds.wave[::dec], lw=.35, color="#E0E0E0")

# draw spans once per event type for speed
def span(ps, pe, col, label=None, alpha=.15):
    ax.axvspan(sec(ps), sec(pe)+args.chunk/ds.fs,
               color=col, alpha=alpha, lw=0, label=label)

for ps, pe in pred_evt:  span(ps, pe, "#F44336")                # FP (red) default
for k, (ts, te) in enumerate(true_evt):
    if matched[k]:
        span(ts, te, "#76FF03", label="TP")                     # lime
    else:
        span(ts, te, "#FF9100", label="FN")                     # orange

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), framealpha=.25, fontsize=9)
ax.set_title("Raw waveform   lime = TP   orange = FN   red = FP")
ax.set_xlabel("time [s]")
ax.set_ylabel("amp")
plt.tight_layout()
plt.savefig("reports/ae_events.png", dpi=dpi)

# 3) TIMELINE -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_w, 2), dpi=dpi)
ax.set_ylim(-1, 1); ax.set_yticks([])
ax.set_xlim(0, len(mask))

# vertical tick lines
for ps, pe in pred_evt:
    ax.vlines(ps, -0.5, 0.5, color="red",   lw=1)               # FP default
for k, (ts, te) in enumerate(true_evt):
    col = "lime" if matched[k] else "orange"
    ax.vlines(ts, -0.8, 0.8, color=col, lw=3)                   # TP/FN

# custom legend
from matplotlib.lines import Line2D
ax.legend(
    handles=[
        Line2D([0], [0], color="lime",   lw=4, label="TP"),
        Line2D([0], [0], color="orange", lw=4, label="FN"),
        Line2D([0], [0], color="red",    lw=2, label="FP"),
    ],
    loc="upper right", framealpha=.25, fontsize=9
)
ax.set_title("TP / FN / FP timeline")
plt.tight_layout()
plt.savefig("reports/ae_pred_timeline.png", dpi=dpi)

print("✓ clearer plots written → ae_err.png / ae_events.png / ae_pred_timeline.png")

