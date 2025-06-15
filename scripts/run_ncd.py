#!/usr/bin/env python3
"""
run_ncd.py – compression-based unsupervised lightning detector
--------------------------------------------------------------

* splits waveform into sliding windows
* NCD against previous window  (zlib | bz2 | lzma)
* adaptive threshold  = median + k·MAD
* merges short gaps, enforces min-duration
* prints window/event metrics & writes three PNGs to ./reports
"""

import argparse, glob, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from pandas     import Series
from tqdm       import tqdm
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from leela_ml.datamodules_npy import StrikeDataset
from leela_ml.models.ncd       import ncd_adjacent

# ── CLI ───────────────────────────────────────────────────────────────────────
pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
pa.add_argument("--npy",   required=True)
pa.add_argument("--meta",  required=True)
pa.add_argument("--chunk", type=int,   default=512)
pa.add_argument("--overlap",type=float,default=0.9)
pa.add_argument("--codec", choices=["zlib","bz2","lzma"], default="zlib")
pa.add_argument("--mad_k", type=float, default=6.0)
pa.add_argument("--min_dur_ms", type=float, default=10.)
pa.add_argument("--gap_ms",     type=float, default=10.)
pa.add_argument("--dpi",   type=int,   default=160)
args = pa.parse_args(); Path("reports").mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
ds  = StrikeDataset(args.npy, args.meta, args.chunk, args.overlap)
win = ds._windows.astype(np.float32, copy=False)
lab = ds.labels.astype(bool)
hop = ds.hop
print(f"• sample_rate: {ds.fs:,.0f} Hz   windows: {len(win):,}   burst-windows (truth): {lab.sum():,}")

# ── NCD ───────────────────────────────────────────────────────────────────────
err = ncd_adjacent(win, codec=args.codec)
print(f"• NCD finished   mean={err.mean():.4f}  med={np.median(err):.4f}")

# ── adaptive threshold --------------------------------------------------------
win_len = max(1, int(args.min_dur_ms/1000 * ds.fs / hop))
roll = Series(err).rolling(win_len, center=True, min_periods=1)
med  = roll.median().values
mad  = roll.apply(lambda v: np.median(np.abs(v-np.median(v))), raw=True).values
thr  = med + args.mad_k * mad
mask = err > thr

# enforce min duration + close small gaps
min_w = max(1, int(args.min_dur_ms/1000 * ds.fs / hop))
gap_w = max(1, int(args.gap_ms    /1000 * ds.fs / hop))
mask  = binary_dilation(mask, iterations=min_w)
mask  = binary_erosion (mask, iterations=min_w)

# ── helpers -------------------------------------------------------------------
def runs(b):
    out=[]; on=False
    for i,v in enumerate(b):
        if v and not on: on,st=True,i
        if not v and on: on=False; out.append((st,i-1))
    if on: out.append((st,len(b)-1))
    return out

pred_evt, true_evt = runs(mask), runs(lab)

# match by overlap
matched_true  = [False]*len(true_evt)
matched_pred  = [False]*len(pred_evt)
tp=0
for i,(ps,pe) in enumerate(pred_evt):
    for j,(ts,te) in enumerate(true_evt):
        if not matched_true[j] and not (pe<ts or ps>te):
            tp+=1; matched_true[j]=matched_pred[i]=True; break

# metrics
P,R,F,_ = precision_recall_fscore_support(lab,mask,average="binary",zero_division=0)
try: auc = roc_auc_score(lab,err)
except ValueError: auc=float("nan")

prec_evt = tp/len(pred_evt) if pred_evt else 0
rec_evt  = tp/len(true_evt) if true_evt else 0
f1_evt   = 2*prec_evt*rec_evt/(prec_evt+rec_evt+1e-9)

print(f"Window  P={P:.3f} R={R:.3f} F1={F:.3f}  AUROC={auc:.3f}")
print(f"Event   P={prec_evt:.3f} R={rec_evt:.3f} F1={f1_evt:.3f}")

# ── plots ---------------------------------------------------------------------
sns.set_style("darkgrid"); dpi=args.dpi; fig_w=16
tsec=lambda w:(w*hop)/ds.fs

# 1 score
plt.figure(figsize=(fig_w,4),dpi=dpi)
plt.plot(err,lw=.4,label="NCD"); plt.plot(thr,lw=.8,ls="--",label="thr")
plt.title("NCD & threshold"); plt.legend(); plt.tight_layout()
plt.savefig("reports/ncd_score.png",dpi=dpi)

# 2 waveform + spans
dec=max(1,int(ds.fs//1000)); t=np.arange(0,len(ds.wave),dec)/ds.fs
plt.figure(figsize=(fig_w,4),dpi=dpi)
plt.plot(t,ds.wave[::dec],lw=.3,color="#999")
for i,(ps,pe) in enumerate(pred_evt):
    col="#76FF03" if matched_pred[i] else "#F44336"
    plt.axvspan(tsec(ps),tsec(pe)+args.chunk/ds.fs,color=col,alpha=.15,lw=0)
for j,(ts,te) in enumerate(true_evt):
    if not matched_true[j]:
        plt.axvspan(tsec(ts),tsec(te)+args.chunk/ds.fs,color="#FF9100",alpha=.15,lw=0)
plt.title("Waveform (green=TP lime, orange=FN, red=FP)"); plt.xlabel("time [s]")
plt.tight_layout(); plt.savefig("reports/ncd_events.png",dpi=dpi)

# 3 timeline
plt.figure(figsize=(fig_w,2),dpi=dpi)
plt.xlim(0,tsec(len(win))); plt.ylim(-0.5,1.5); plt.yticks([0,1],["Pred","True"])
for i,(ps,pe) in enumerate(pred_evt):
    col="#76FF03" if matched_pred[i] else "#F44336"
    plt.broken_barh([(tsec(ps),tsec(pe+1-ps))],(-0.2,0.4),facecolors=col)
for j,(ts,te) in enumerate(true_evt):
    col="#76FF03" if matched_true[j] else "#FF9100"
    plt.broken_barh([(tsec(ts),tsec(te+1-ts))],(0.8,0.4),facecolors=col)
plt.title("Event timeline"); plt.xlabel("time [s]"); plt.tight_layout()
plt.savefig("reports/ncd_pred_timeline.png",dpi=dpi)

print("✓ clearer plots written → ncd_score.png / ncd_events.png / ncd_pred_timeline.png")
