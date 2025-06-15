"""
Fast Normalised Compression Distance (NCD) between consecutive windows.

NCD(w_{i-1}, w_i) = (C(w_{i-1}+w_i) – min(C(w_{i-1}),C(w_i))) / max(...)
where C(.) is compressed length.

Supports zlib / bz2 / lzma.  Returns float32[n_windows].
"""

from __future__ import annotations
import bz2, lzma, zlib, numpy as np
from typing import Literal

Codec = Literal["zlib", "bz2", "lzma"]
_COMP = dict(zlib=zlib.compress, bz2=bz2.compress, lzma=lzma.compress)


def _clen(raw: bytes, codec: Codec) -> int:
    return len(_COMP[codec](raw))


def ncd_adjacent(win: np.ndarray, codec: Codec = "zlib") -> np.ndarray:
    """NCD between each window and its predecessor (win must be float32 2-D)."""
    if win.ndim != 2:
        raise ValueError("windows must be 2-D (n_win, win_len)")
    n = len(win)
    if n < 2:
        return np.zeros(n, np.float32)

    w_i16 = win.astype(np.int16, copy=False)
    clen  = [_clen(w.tobytes(), codec) for w in w_i16]

    out   = np.empty(n, np.float32)
    out[0] = 0.0                      # dummy for first window

    prev_b = w_i16[0].tobytes()
    prev_c = clen[0]

    for i in range(1, n):
        cur_b  = w_i16[i].tobytes()
        cur_c  = clen[i]
        joint  = _clen(prev_b + cur_b, codec)
        out[i] = (joint - min(prev_c, cur_c)) / float(max(prev_c, cur_c))
        prev_b, prev_c = cur_b, cur_c

    out[0] = out[1]
    return out
