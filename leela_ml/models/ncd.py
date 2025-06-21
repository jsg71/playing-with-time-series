# ruff: noqa
"""
─────────────────────────────────────────────────────────────────────────────
 ncd.py — Fast Normalised Compression Distance (NCD) for consecutive windows
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

PURPOSE
───────
Compute the **Normalised Compression Distance** between each time‑series
window and its immediate predecessor:

    NCD(wᵢ₋₁, wᵢ) =
        ( C(wᵢ₋₁ + wᵢ) − min[C(wᵢ₋₁), C(wᵢ)] )
        ───────────────────────────────────────
                 max[C(wᵢ₋₁), C(wᵢ)]

where  C(·)  is the byte‑length after lossless compression.
Result is a `float32[ n_windows ]` array in the range **[0, 1 + ε]**
(lower = more similarity).

WHY NCD?
────────
• **Model‑free** — no assumptions about distribution or amplitude scale.
• Captures *any* regularity the chosen compressor exploits
  (repeated motifs, silence, constant offsets, etc.).
• Cheap to code: only compression calls, no heavy maths.

MODULE CONTENTS
───────────────
`_clen(raw: bytes, codec)`     → helper: compressed length in bytes.
`ncd_adjacent(win, codec='zlib')` → main public function.

FUNCTION: ncd_adjacent
──────────────────────
Parameters
    win   : numpy.ndarray, shape (n_win, win_len), dtype float32
            Consecutive windows to compare.
    codec : {'zlib', 'bz2', 'lzma'}, default 'zlib'
            Which Python std‑lib compressor to use.

Returns
    numpy.ndarray, float32, length n_win
        out[0] is copied from out[1] so the vector can be plotted
        without a gap; mathematically it is undefined.

Algorithm
─────────
1. **Pre‑quantise** windows to **int16** (2 bytes per sample) to
   stabilise compression ratio and cut RAM.
2. Pre‑compute `clen[i] = C(wᵢ)` for all windows.
3. Loop i = 1 … n‑1
       joint = C(wᵢ₋₁ + wᵢ)
       NCD   = (joint − min(prev_c, cur_c)) / max(prev_c, cur_c)
4. out[0] ← out[1] for convenience.

Computational complexity
    O(n) compressions  (each `clen` already cached)
    Memory: O(1) extra besides input array.

STRENGTHS
─────────
✓ **Parameter‑light** — only window length & codec matter.
✓ Works on any numeric signal once cast to int16.
✓ Fast for zlib; pure‑Python only relies on compiled std‑lib modules.

WEAKNESSES & CAVEATS
────────────────────
✗ **Codec‑dependent** — zlib may miss patterns bz2/lzma capture and vice‑versa.
✗ Int16 quantisation loses tiny amplitude differences.
✗ Not symmetric (distance to *next* only).
✗ First value is heuristic copy — ignore or drop when analysing.
✗ Compression time grows with `win_len`; may bottleneck for > 10 k samples.

SUGGESTED IMPROVEMENTS
──────────────────────
• Accept `axis` argument to support channels‑first/last arrays.
• Optionally compute *pairwise* NCD matrix (O(n²)) for clustering.
• Expose `quant_bits` to let user pick 8‑, 12‑ or 16‑bit packing.
• Parallelise compression with `concurrent.futures` for multi‑core CPUs.
• Cache `clen` across calls when scanning overlapping windows.

EXAMPLE USAGE
─────────────
```python
import numpy as np, matplotlib.pyplot as plt
from ncd import ncd_adjacent

# toy data : sine + noise
win = np.stack([np.sin(0.01*np.arange(1024)) + 0.05*np.random.randn(1024)
                for _ in range(200)], axis=0).astype(np.float32)

ncd_vec = ncd_adjacent(win, codec='bz2')
plt.plot(ncd_vec); plt.title("NCD between consecutive windows")
plt.show()
```

─────────────────────────────────────────────────────────────────────────────
"""


from __future__ import annotations
import bz2, lzma, zlib, numpy as np
from typing import Literal

Codec = Literal["zlib", "bz2", "lzma"]
_COMP = dict(zlib=zlib.compress, bz2=bz2.compress, lzma=lzma.compress)


def _clen(raw: bytes, codec: Codec) -> int:
    return len(_COMP[codec](raw))


def ncd_adjacent(
    win: np.ndarray,
    codec: Codec = "zlib",
    *,
    per_win_norm: bool = False,
    per_win_nor: bool | None = None,
    **kwargs,
) -> np.ndarray:
    """NCD between each window and its predecessor.

    Parameters
    ----------
    win : np.ndarray
        2-D array of windows ``(n_win, win_len)`` in ``float32``.
    codec : {"zlib", "bz2", "lzma"}
        Which compressor to use.
    per_win_norm : bool, optional
        If ``True`` each window is normalised individually (zero mean and
        scaled to utilise the 16‑bit range).  This often highlights subtle
        shape differences between consecutive windows.
    per_win_nor : bool, optional
        Deprecated alias for :pyparamref:`per_win_norm` kept for older notebooks.
    """
    # legacy notebooks used ``per_win_nor``; allow both names
    if per_win_nor is not None:
        per_win_norm = per_win_nor
    if "per_win_nor" in kwargs:
        per_win_norm = kwargs.pop("per_win_nor")
    if "per_win_norm" in kwargs:
        per_win_norm = kwargs.pop("per_win_norm")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {', '.join(kwargs)}")
    if win.ndim != 2:
        raise ValueError("windows must be 2-D (n_win, win_len)")
    n = len(win)
    if n < 2:
        return np.zeros(n, np.float32)

    if per_win_norm:
        w = win - win.mean(axis=1, keepdims=True)
        scale = 32767 / np.maximum(1.0, np.max(np.abs(w), axis=1, keepdims=True))
        w_i16 = (w * scale).astype(np.int16, copy=False)
    else:
        scale = 32767 / max(1.0, np.max(np.abs(win)))
        w_i16 = (win * scale).astype(np.int16, copy=False)
    clen = [_clen(w.tobytes(), codec) for w in w_i16]

    out = np.empty(n, np.float32)
    out[0] = 0.0  # dummy for first window

    prev_b = w_i16[0].tobytes()
    prev_c = clen[0]

    for i in range(1, n):
        cur_b = w_i16[i].tobytes()
        cur_c = clen[i]
        joint = _clen(prev_b + cur_b, codec)
        out[i] = (joint - min(prev_c, cur_c)) / float(max(prev_c, cur_c))
        prev_b, prev_c = cur_b, cur_c

    out[0] = out[1]
    return out
