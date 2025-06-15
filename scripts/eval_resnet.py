#!/usr/bin/env python
"""
Strict evaluation: loads the split saved by train_resnet.py, prints
VAL & TEST metrics, and saves reports/resnet_val_test.png.
"""
import argparse, glob, re, numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch
from pathlib import Path
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Subset
from leela_ml.datamodules_npy  import StrikeDataset
from leela_ml.models.raw_resnet import RawResNet1D

# CLI -------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--ckpt", default="lightning_logs/raw.ckpt")
ap.add_argument("--bs",   type=int, default=512)
args = ap.parse_args(); Path("reports").mkdir(exist_ok=True)

if args.npy.endswith("_wave.npy") and not Path(args.npy).is_file():
    args.npy = sorted(glob.glob(args.npy.replace("_wave", "_*.npy")))[0]

spl = np.load(Path(args.ckpt).with_suffix("").with_name("split.npz"))
idx_tr, idx_val, idx_te = spl["idx_tr"], spl["idx_val"], spl["idx_test"]
ds   = StrikeDataset(args.npy, args.meta, chunk=args.chunk)
net  = RawResNet1D(); sd=torch.load(args.ckpt, map_location="cpu")
sd   = {re.sub(r"^net\.", "", k):v for k,v in (sd["state_dict"] if "state_dict" in sd else sd).items()}
net.load_state_dict(sd, strict=False); net.eval()

def infer(indices):
    dl = DataLoader(Subset(ds, indices), args.bs, shuffle=False, num_workers=0)
    p=[]
    with torch.no_grad():
        for x,_ in dl: p.append(torch.sigmoid(net(x)).cpu())
    return torch.cat(p).numpy()

def report(name, prob, y):
    pr, rc, th = precision_recall_curve(y, prob)
    f1 = 2*pr*rc/(pr+rc+1e-9); i=f1.argmax()
    thr=float(th[i]); auc=roc_auc_score(y, prob)
    pred=(prob>=thr).astype(int)
    pr_c, rc_c, f1_c,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    print(f"{name:>4}: AUROC={auc:.3f}  thr={thr:.3f}  "
          f"P={pr_c:.3f} R={rc_c:.3f} F1={f1_c:.3f}")
    return thr

p_val  = infer(idx_val);  thr_val  = report("VAL",  p_val,  ds.labels[idx_val])
p_test = infer(idx_te );  thr_test = report("TEST", p_test, ds.labels[idx_te])
p_all  = infer(range(len(ds)))

# plot ------------------------------------------------------------------------
sns.set_style("darkgrid"); plt.figure(figsize=(12,6))
ax1=plt.subplot(2,1,1)
ax1.plot(p_all); ax1.axvspan(0,idx_tr[-1],color="#ccc",alpha=0.25,label="train")
ax1.axvspan(idx_val[0],idx_val[-1],color="#aec7e8",alpha=0.25,label="val")
ax1.axvspan(idx_te[0],idx_te[-1],color="#98df8a",alpha=0.25,label="test")
ax1.axhline(thr_val ,color="#1f77b4",ls="--",lw=0.8,label=f"thr val  {thr_val:.3f}")
ax1.axhline(thr_test,color="#2ca02c",ls="--",lw=0.8,label=f"thr test {thr_test:.3f}")
ax1.scatter(np.where(ds.labels)[0],[1.05]*int(ds.labels.sum()),
            marker="x",s=10,color="k",label="truth")
ax1.set_ylim(0,1.1); ax1.set_ylabel("P(strike)"); ax1.legend(ncol=4,fontsize=8)

ax2=plt.subplot(2,1,2,sharex=ax1)
ax2.plot(ds.wave,lw=0.4); ax2.set_ylabel("amplitude"); ax2.set_xlabel("chunk idx × hop")
plt.tight_layout(); plt.savefig("reports/resnet_val_test.png")
print("✓ figure → reports/resnet_val_test.png")
