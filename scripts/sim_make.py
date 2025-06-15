#!/usr/bin/env python
import argparse, shutil, glob
from leela_ml.signal_sim.simulator import simulate

p=argparse.ArgumentParser()
p.add_argument("--minutes",type=int,default=5)
p.add_argument("--out",required=True)
p.add_argument("--seed",type=int,default=0)
a=p.parse_args()

simulate(a.minutes,a.out,seed=a.seed)
first=sorted(glob.glob(f"{a.out}_*.npy"))[0]
shutil.copy(first,f"{a.out}_wave.npy")
print("alias →",f"{a.out}_wave.npy")
