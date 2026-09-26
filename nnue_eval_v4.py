"""Numba full-recompute evaluator for dual-perspective NNUE v4."""

from pathlib import Path

import numpy as np
from numba import njit

from bitboard import king_square

FEATURES = 6144
_FT = np.zeros((FEATURES, 1), np.float32)
_FTB = np.zeros(1, np.float32)
_HW = np.zeros((1, 2), np.float32)
_HB = np.zeros(1, np.float32)
_OW = np.zeros((1, 1), np.float32)
_OB = np.zeros(1, np.float32)
_MULTIPLIER = np.float32(400.0)
_LOADED = [False]


@njit(cache=False)
def _ctz(x):
    n = 0
    while (x & np.uint64(1)) == 0:
        x >>= np.uint64(1)
        n += 1
    return n


@njit(cache=False, fastmath=True)
def _forward(bb, ft, ftb, hw, hb, ow, ob, multiplier):
    wk = king_square(bb, 0)
    bk = king_square(bb, 1) ^ 56
    wb = (wk // 16) * 2 + ((wk & 7) >= 4)
    bbucket = (bk // 16) * 2 + ((bk & 7) >= 4)
    wa = ftb.copy()
    ba = ftb.copy()

    for bi in range(12):
        bits = bb[bi]
        color_black = bi >= 6
        pt = bi % 6
        wp = (6 if color_black else 0) + pt
        bp = (0 if color_black else 6) + pt
        while bits:
            sq = _ctz(bits)
            bits &= bits - np.uint64(1)
            wi = wb * 768 + wp * 64 + sq
            bidx = bbucket * 768 + bp * 64 + (sq ^ 56)
            for j in range(ftb.shape[0]):
                wa[j] += ft[wi, j]
                ba[j] += ft[bidx, j]

    for j in range(ftb.shape[0]):
        wa[j] = min(np.float32(2.0), max(np.float32(0.0), wa[j]))
        ba[j] = min(np.float32(2.0), max(np.float32(0.0), ba[j]))

    stm = bb[15]
    h = hb.copy()
    width = ftb.shape[0]
    for k in range(hb.shape[0]):
        s = np.float32(0.0)
        for j in range(width):
            first = wa[j] if stm == 0 else ba[j]
            second = ba[j] if stm == 0 else wa[j]
            s += hw[k, j] * first + hw[k, width + j] * second
        h[k] = min(np.float32(2.0), max(np.float32(0.0), h[k] + s))

    out = ob[0]
    for k in range(hb.shape[0]):
        out += ow[0, k] * h[k]
    return int(round(out * multiplier))


def load_model(path: str | Path) -> dict:
    global _FT, _FTB, _HW, _HB, _OW, _OB, _MULTIPLIER
    d = np.load(path, allow_pickle=False)
    if "ft" not in d or d["ft"].shape[0] != FEATURES:
        raise ValueError("nnue_eval_v4 requires a float NNUE v4 export")
    _FT = np.ascontiguousarray(d["ft"].astype(np.float32))
    _FTB = np.ascontiguousarray(d["ft_bias"].astype(np.float32))
    _HW = np.ascontiguousarray(d["hidden"].astype(np.float32))
    _HB = np.ascontiguousarray(d["hidden_bias"].astype(np.float32))
    _OW = np.ascontiguousarray(d["output"].astype(np.float32))
    _OB = np.ascontiguousarray(d["output_bias"].astype(np.float32))
    _MULTIPLIER = np.float32(d["output_multiplier"])
    _LOADED[0] = True
    return {"features": FEATURES, "accum": len(_FTB), "hidden": len(_HB)}


def nnue_eval_state(bb: np.ndarray) -> int:
    if not _LOADED[0]:
        raise RuntimeError("nnue_eval_v4.load_model() not called")
    return int(_forward(bb, _FT, _FTB, _HW, _HB, _OW, _OB, _MULTIPLIER))


@njit(cache=False, fastmath=True)
def eval_state(bb, ev):
    return _forward(bb, _FT, _FTB, _HW, _HB, _OW, _OB, _MULTIPLIER)
