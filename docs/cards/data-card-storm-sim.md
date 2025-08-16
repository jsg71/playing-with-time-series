
---
title: "Data Card — Synthetic Lightning‑Storm Waveforms (StormGenerator)"
status: "production-candidate"
owners:
  - team: Lightning Sim / Data & Detection
    email: detection-team@example.internal
dataset-id: "lightning_sim.data.storm_synth"
license: "Internal / Company Confidential"
version: "0.1.0"
tags:
  - synthetic
  - time-series
  - waveforms
  - multi-station
  - anomaly-detection
  - window-level-labels
---

> **TL;DR**  
> Fully synthetic, **reproducible** multi‑station lightning waveform data produced by `StormGenerator`.  
> Outputs per‑station 14‑bit ADC traces (int16), stroke‑level ground truth, and ready‑to‑use window labels aligned to the shared evaluation protocol.

---

## 1) Dataset summary

- **Modality**: 1‑D time‑series per station (int16, 14‑bit).  
- **Stations**: 11 fixed sites across Europe (Iceland → Cyprus).  
- **Sampling rate**: **FS = 109 375 Hz** (Δt ≈ 9.14 µs).  
- **Window geometry** (for downstream models): **WIN = 1024**, **HOP = 512** (≈ 9.36 ms frames, 50 % overlap).  
- **Temporal span**: configurable; default **5 min** active storm region following a short pre‑roll.  
- **Ground truth**: stroke times (per flash), lat/lon of flashes, and **per‑station window truth** derived from propagation delay and burst extent.  
- **Reproducibility**: all randomness flows through `numpy.random.Generator` seeded by `StormConfig.seed` → **bit‑identical** regeneration across machines given the same code/config.

---

## 2) Generation process (StormGenerator)

### 2.1 Core configuration
`StormConfig` fields (defaults shown):
- `seed=424242`, `duration_min=5`  
- `scenario ∈ {near, medium, far}` controls flash rate and spatial spread  
- `difficulty ∈ {1,…,9}` toggles impairments (multipath, RFI, clipping, etc.)  
- `snr_db` controls global SNR (linear factor \(\text{SNR}_\text{lin}=10^{\text{snr\_db}/20}\))  
- `fs=109_375`, `bits=14`, `vref=1.0`

### 2.2 Station network
Hard‑coded lat/lon for 11 sites (`KEF, VAL, LER, HER, GIB, AKR, CAM, WAT, CAB, PAY, TAR`). Example subset:

| Code | Lat    | Lon    |
|:----:|:------:|:------:|
| KEF  | 64.020 | −22.567 |
| VAL  | 51.930 | −10.250 |
| LER  | 60.150 |  −1.130 |
| HER  | 50.867 |   0.336 |
| GIB  | 36.150 |  −5.350 |
| AKR  | 34.588 |  32.986 |

*(Full list is embedded in the generator source.)*

### 2.3 Geometry & propagation
- **Distance** (km) via **haversine**:  
  \[
  d(\varphi_1,\lambda_1;\varphi_2,\lambda_2) = 2R\,\arcsin\!\sqrt{\sin^2\frac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\frac{\Delta\lambda}{2}},\quad R=6371\,\text{km}.
  \]
- **Path loss** (empirical):  
  \[
  L(d) = \Big(\tfrac{100}{d+100}\Big)^{0.85}\,\exp(-10^{-4} d)\,\times\begin{cases}
  \sqrt{2}, & d>600\,\text{km}\\
  1, & \text{otherwise}
  \end{cases}
  \]
- **Propagation delay**: index shift by \(d/300{,}000\) s (speed of light) + Gaussian jitter (≈ 40 µs std).

### 2.4 Flash scheduler & bursts
- **Flash rate** \(\lambda\) depends on `scenario` and `difficulty`. Inter‑flash gaps ~ **Exponential**.  
- Each flash spawns **1–5 strokes** with small (ms) intra‑flash gaps.  
- **Burst synthesis** (~40 ms) per stroke:  
  - Damped sinusoid or divergent template (random \(f_0\), decay \(\tau\)).  
  - Optional **multipath** echoes, **sprite ring**, and **sky‑wave** low‑pass (on long paths) in frequency domain.  
  - Amplitude set by **CG/IC** type, **path loss**, and global **SNR**.

### 2.5 Noise & front‑end impairments
- **Coloured background**: white noise + 50 Hz hum, optional **RFI tones**.  
- **Analogue drift**: slow gain drift; **clock skew**.  
- **Sferic bed**: low‑level impulsive background.  
- **Dropouts**: occasional 0.4 s data loss spans (optional).  
- **Impulsive RFI** / **false transients**: brief synthetic artefacts.

### 2.6 ADC modelling
- 4th‑order **Butterworth** low‑pass at 45 kHz (filtfilt).  
- Optional **clipping** (±0.9 · `vref`).  
- **Quantisation** to 14‑bit:  
  \[
  q = \operatorname{clip}\Big(\mathrm{round}\big(\tfrac{x}{\text{vref}} (2^{13}-1)\big),\; -\!(2^{13}-1),\; 2^{13}-1\Big) \in \mathbb{Z}.
  \]

---

## 3) Contents & schema

A single call to `StormGenerator.generate()` returns a **StormBundle**:

- `quantised: Dict[str, np.ndarray]` — per‑station int16 waveform (same length)  
- `events: List[dict]` — one record per **flash**: `{id, flash_type, lat, lon, stroke_times}`  
- `stroke_records: List[dict]` — one per **station × stroke**: `{event_id, stroke_i, station, flash_type, lat, lon, true_time_s, sample_idx, window_idx}`  
- `df_wave: DataFrame[{time_s, <station columns>}]` — sample timeline + ADC counts per station  
- `df_labels: DataFrame` — tidy table from `stroke_records` (convenient for join/analysis)

**Window labels** (recommended downstream contract):
- **Window size**: `WIN=1024`, **hop** `HOP=512`.  
- For each stroke, mark all windows whose index segment overlaps `[sample_idx, sample_idx + burst_len)`, where `burst_len = int(0.04*FS)` by default.  
- Produces per‑station **Boolean vectors** consistent with the evaluation API.

---

## 4) Splits, sizes & storage

- **Typical size** (5 min storm, 11 stations):  
  - Samples per station: \(N = (5\times60 + \text{pre})\cdot 109{,}375\) ≈ **~33 M**.  
  - Windows per station: \(n_{\text{win}} \approx \lfloor (N-W)/H \rfloor + 1\) (≈ **~65 k**).  
- **Recommended splits** (all **synthetic**; prefer **config‑based** splits instead of random):  
  - **Train**: low/mid `difficulty` with varied `scenario` and `snr_db` → diverse background.  
  - **Val**: hold‑out seeds; tune model thresholds (`pct`, `contamination`, score cut‑offs).  
  - **Test**: unseen seeds and difficult scenes (`difficulty≥7`, `far`) to assess robustness.  
- **Storage**:  
  - Waveforms: `.npy`/`.npz` per station; or Parquet for `df_wave`/`df_labels`.  
  - Metadata: JSON/Parquet for `events` and `stroke_records`.  
  - Bundle serialisation: optional `.npz` with arrays + JSON sidecar for events.

---

## 5) Intended uses

- Benchmarking **unsupervised detectors** (Hilbert thresholding, NCD, Isolation‑Forest, isotree‑IF, CDAE, Graph‑CDAE, OCSVM).  
- End‑to‑end **pipeline tests**: windowing → features/learning → evaluation.  
- **Ablation** studies: flip individual difficulty flags to isolate failure modes.

**Out‑of‑scope**  
- Operational localisation accuracy; real‑world deployment claims; IC/CG classification fidelity.

---

## 6) Statistical properties & metrics

- **Class imbalance**: strokes are rare; window‑level positive rate ~ 0.1–1 % depending on `scenario`, `difficulty`, and λ.  
- **Recommended metrics**:  
  - **Station level**: precision/recall/F1 over windows; PR‑AUC for score‑based models.  
  - **Network level**: stroke precision/recall/F1 with **quorum** ≥ `min_stn` and **tolerance** `tol_win`.  
- **Evaluation details** (network):  
  - TP: at least `min_stn` distinct stations fire in **any** window overlapping a true stroke.  
  - FP: **clusters** of windows with ≥ `min_stn` hot stations that do **not** overlap any ground‑truth stroke.  
  - FN: strokes missed by quorum; TN: windows with neither truth nor prediction.

---

## 7) Reproducibility & versioning

- **Deterministic seed**: `StormConfig.seed` governs all RNG. Log seed per bundle.  
- **Config hash**: include `{scenario, difficulty, snr_db, fs, bits, vref, seed}` + code version in artefact metadata.  
- **Exact regeneration**: storing only `StormConfig` + generator version is sufficient to reproduce identical bundles.  
- **CI**: smoke test generates a short storm and hashes all arrays to assert byte‑identity across environments.

---

## 8) Quality & validation checks

- **ADC sanity**: no counts outside ±(2^{13}-1); clipping rate within expected range when `clipping=True`.  
- **Spectral shape**: 50 Hz line and RFI tones visible at configured amplitudes.  
- **Geodesy**: haversine distances monotone with station ordering (spot checks).  
- **Timing**: stroke insert indices respect propagation delay ± jitter.  
- **Labels**: window masks reflect `[sample_idx, sample_idx + burst_len)` for each station; cross‑validate with Hilbert envelope peaks.

---

## 9) Risks, biases & limitations

- **Synthetic realism**: empirical **path‑loss** and impairment parameters may not match all terrains / ionospheric conditions.  
- **Generative priors**: burst templates and noise models bias detectors that key on those statistics. Validate on **real** data before deployment.  
- **Coverage**: extreme RFI scenarios, sustained dropouts, and hardware faults are simplified.  
- **Evaluation coupling**: window labels assume 40 ms burst length; adjust `burst_len` when exploring longer‑tail sferics.

---

## 10) Privacy, security & governance

- **PII**: none in waveforms; treat auxiliary metadata (paths, usernames) per company policy.  
- **Data locality**: simulation and consumption happen inside the secure environment; no external services called.  
- **Access control**: dataset marked **Company Confidential**; restrict export of large waveform arrays.  
- **Auditability**: seeds, configs, and generator versions are logged per bundle.

---

## 11) How to generate a bundle (reference)

```python
from lightning_sim.sim.generator import StormConfig, StormGenerator

cfg = StormConfig(
    seed=424242,
    duration_min=5,
    scenario="medium",     # "near" | "medium" | "far"
    difficulty=5,          # 1…9
    snr_db=-6.0,           # dB → linear via 10**(snr_db/20)
    fs=109_375, bits=14, vref=1.0
)
gen = StormGenerator(cfg)
bundle = gen.generate()

# bundle.quantised  → {station: int16 waveform}
# bundle.events     → flash metadata with stroke times
# bundle.df_labels  → tidy stroke records (station × stroke)
```

**Window labels** (if you need them stand‑alone):
```python
import numpy as np, pandas as pd

WIN, HOP = 1024, 512
FS       = cfg.fs
BURST    = int(0.04 * FS)

n_win = min((len(x)-WIN)//HOP + 1 for x in bundle.quantised.values())
truth = {s: np.zeros(n_win, bool) for s in bundle.quantised}

for rec in bundle.stroke_records:
    s0, s1 = rec["sample_idx"], rec["sample_idx"] + BURST - 1
    w0 = max(0, int(np.ceil((s0+1 - WIN)/HOP)))
    w1 = min(n_win-1, int(np.floor(s1/HOP)))
    truth[rec["station"]][w0:w1+1] = True
```

---

## 12) File layout suggestions (for packaging)

```
/storm_synth/
  ├── bundles/
  │   ├── v0.1.0/
  │   │   ├── <run-id>/
  │   │   │   ├── quantised_<STN>.npy    # one per station
  │   │   │   ├── events.parquet
  │   │   │   ├── stroke_records.parquet
  │   │   │   ├── df_wave.parquet
  │   │   │   ├── df_labels.parquet
  │   │   │   └── meta.json               # StormConfig + versions + checksums
  │   │   └── ...
  └── README.md
```

---

## 13) Change log

- **v0.1.0** — Initial public (internal) release: 11‑station network; difficulty flags; full ADC pipeline; window labels aligned to evaluator.

---

## 14) References

- Haversine distance — standard great‑circle formula.  
- Butterworth filter — classic IIR low‑pass design (filtfilt for zero‑phase).  
- Lightning signal propagation / sferics (introductory literature for context).
