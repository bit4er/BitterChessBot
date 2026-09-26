"""njit NNUE evaluation from the engine's bitboard state array.

Full recompute per call. For a 768->256->32->1 net the input is sparse (~32
pieces) so the first layer is ~32 row-adds of length h1, not a dense 768xh1
matvec -- cheap enough that the incremental accumulator may be unnecessary for a
net this size. This module is what an SPRT tests; if the nps hit is too big we
fall back to the incremental accumulator wired into make/unmake.

Weights come from ``nnue/model/nnue-<tag>.npz`` (``nnue.model.export_npz``).
``load_model`` mutates module globals in place (same pattern as
``evaluate_bb.repack``) so the njit closures keep valid references.

Feature index mirrors ``nnue.encode.feature_index`` exactly:
    stm=white: rel_sq = sq,       same_side = (piece is white)
    stm=black: rel_sq = sq ^ 56,  same_side = (piece is black)
    plane = (0 if same_side else 6) + piece_type_index      # P N B R Q K = 0..5
    feature = plane*64 + rel_sq
Output is centipawns from the side-to-move's point of view (same as
``evaluate_bb.eval_state``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

# --- globals filled by load_model() -----------------------------------------
_L1W = np.zeros((768, 1), np.float32)   # (768, h1)
_L1B = np.zeros(1, np.float32)          # (h1,)
_L2W = np.zeros((1, 1), np.float32)     # (h2, h1)
_L2B = np.zeros(1, np.float32)          # (h2,)
_OW = np.zeros((1, 1), np.float32)      # (1, h2)
_OB = np.zeros(1, np.float32)           # (1,)
_CR = np.float32(127.0 / 64.0)
_LOADED = [False]


@njit(cache=False, fastmath=True)
def _forward(bb, l1w, l1b, l2w, l2b, ow, ob, cr):
    stm = bb[15]
    h1 = l1b.shape[0]
    acc = l1b.copy()
    for bi in range(12):
        bits = bb[bi]
        if bits == 0:
            continue
        color_black = 1 if bi >= 6 else 0
        pt = bi % 6
        if stm == 0:
            same = 1 if color_black == 0 else 0
        else:
            same = 1 if color_black == 1 else 0
        plane = (0 if same == 1 else 6) + pt
        base = plane * 64
        while bits != 0:
            sq = _ctz(bits)
            bits &= bits - np.uint64(1)
            rsq = sq if stm == 0 else (sq ^ 56)
            row = base + rsq
            for j in range(h1):
                acc[j] += l1w[row, j]
    # clipped relu
    for j in range(h1):
        v = acc[j]
        if v < 0.0:
            acc[j] = 0.0
        elif v > cr:
            acc[j] = cr
    h2 = l2b.shape[0]
    hid = l2b.copy()
    for k in range(h2):
        s = 0.0
        for j in range(h1):
            s += l2w[k, j] * acc[j]
        v = hid[k] + s
        if v < 0.0:
            v = 0.0
        elif v > cr:
            v = cr
        hid[k] = v
    out = ob[0]
    for k in range(h2):
        out += ow[0, k] * hid[k]
    return int(round(out))


@njit(cache=False)
def _ctz(x):
    # count trailing zeros of a uint64 (x != 0)
    n = 0
    while (x & np.uint64(1)) == 0:
        x >>= np.uint64(1)
        n += 1
    return n


def load_model(path: str | Path) -> dict:
    global _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR
    d = np.load(path, allow_pickle=False)
    _L1W = np.ascontiguousarray(d["l1_w"].T.astype(np.float32))     # (768, h1)
    _L1B = np.ascontiguousarray(d["l1_b"].astype(np.float32))
    _L2W = np.ascontiguousarray(d["l2_w"].astype(np.float32))       # (h2, h1)
    _L2B = np.ascontiguousarray(d["l2_b"].astype(np.float32))
    _OW = np.ascontiguousarray(d["out_w"].astype(np.float32))       # (1, h2)
    _OB = np.ascontiguousarray(d["out_b"].astype(np.float32))
    _CR = np.float32(float(d["cr_max"]))
    _LOADED[0] = True
    return {"h1": int(d["h1"]), "h2": int(d["h2"]), "eval_scale": float(d["eval_scale"])}


def nnue_eval_state(bb: np.ndarray) -> int:
    if not _LOADED[0]:
        raise RuntimeError("nnue_eval.load_model() not called")
    return int(_forward(bb, _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR))


@njit(cache=False, fastmath=True)
def eval_state(bb, ev):
    """Drop-in for ``evaluate_bb.eval_state`` -- the ``ev`` bundle is ignored.
    Reads the weight globals, which ``load_model`` must have set before the
    first call (lazy njit compile then freezes the loaded arrays)."""
    return _forward(bb, _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR)
