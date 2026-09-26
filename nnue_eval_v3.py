"""njit NNUE evaluation from the engine's bitboard state array -- v3,
king-bucketed HalfKA-lite scheme.

Same full-recompute-per-call approach as ``nnue_eval.py`` (proven cheap enough
for v1/v2's flat 768-input net: ~32 active pieces, so the first layer is a
handful of row-adds, not a dense matvec). The only difference from
``nnue_eval.py`` is the row index: each piece's row is offset into one of
``nnue.kingbucket.N_BUCKETS`` 768-wide blocks, selected by which zone the
side-to-move's *own* king currently sits in (``nnue.kingbucket.
KING_BUCKET_TABLE``, computed on the perspective-relative square, same
mirroring convention as the flat scheme).

Feature index mirrors ``nnue.kingbucket.feature_index_kb`` exactly:
    stm=white: rel_sq = sq,       rel_king = king_sq_own,       same_side = (piece is white)
    stm=black: rel_sq = sq ^ 56,  rel_king = king_sq_own ^ 56,  same_side = (piece is black)
    plane = (0 if same_side else 6) + piece_type_index      # P N B R Q K = 0..5
    bucket = KING_BUCKET_TABLE[rel_king]
    feature = bucket*768 + plane*64 + rel_sq
Output is centipawns from the side-to-move's point of view (same as
``evaluate_bb.eval_state`` / ``nnue_eval.eval_state``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from bitboard import king_square
from nnue.kingbucket import KING_BUCKET_TABLE, N_FEATURES

_FLAT = 768  # one king-bucket block

# --- globals filled by load_model() -----------------------------------------
_L1W = np.zeros((N_FEATURES, 1), np.float32)   # (N_FEATURES, h1)
_L1B = np.zeros(1, np.float32)                 # (h1,)
_L2W = np.zeros((1, 1), np.float32)            # (h2, h1)
_L2B = np.zeros(1, np.float32)                 # (h2,)
_OW = np.zeros((1, 1), np.float32)             # (1, h2)
_OB = np.zeros(1, np.float32)                  # (1,)
_CR = np.float32(127.0 / 64.0)
_CLIPPED_RELU = np.bool_(True)
_GAIN = np.float32(1.0)
_KBT = KING_BUCKET_TABLE  # (64,) int64, own-king relative-square -> bucket
_LOADED = [False]


@njit(cache=False, fastmath=True)
def _forward(bb, l1w, l1b, l2w, l2b, ow, ob, cr, kbt, clipped_relu, gain):
    stm = bb[15]
    king_sq_own = king_square(bb, stm)
    rel_king = king_sq_own if stm == 0 else (king_sq_own ^ 56)
    bucket = kbt[rel_king]
    bucket_base = bucket * 768

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
        base = bucket_base + plane * 64
        while bits != 0:
            sq = _ctz(bits)
            bits &= bits - np.uint64(1)
            rsq = sq if stm == 0 else (sq ^ 56)
            row = base + rsq
            for j in range(h1):
                acc[j] += l1w[row, j]
    for j in range(h1):
        v = acc[j]
        if v < 0.0:
            v = 0.0
        elif clipped_relu and v > cr:
            v = cr
        acc[j] = v
    h2 = l2b.shape[0]
    hid = l2b.copy()
    for k in range(h2):
        s = 0.0
        for j in range(h1):
            s += l2w[k, j] * acc[j]
        v = hid[k] + s
        if v < 0.0:
            v = 0.0
        elif clipped_relu and v > cr:
            v = cr
        hid[k] = v
    out = ob[0]
    for k in range(h2):
        out += ow[0, k] * hid[k]
    return int(round(out * gain))


@njit(cache=False)
def _ctz(x):
    # count trailing zeros of a uint64 (x != 0)
    n = 0
    while (x & np.uint64(1)) == 0:
        x >>= np.uint64(1)
        n += 1
    return n


def load_model(path: str | Path) -> dict:
    global _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR, _CLIPPED_RELU, _GAIN
    d = np.load(path, allow_pickle=False)
    n_features = int(d["l1_w"].shape[1])
    if n_features != N_FEATURES:
        raise ValueError(
            f"nnue_eval_v3 expects {N_FEATURES} (king-bucketed) input features, "
            f"got {n_features} from {path} -- is this a flat v1/v2 export?"
        )
    _L1W = np.ascontiguousarray(d["l1_w"].T.astype(np.float32))     # (N_FEATURES, h1)
    _L1B = np.ascontiguousarray(d["l1_b"].astype(np.float32))
    _L2W = np.ascontiguousarray(d["l2_w"].astype(np.float32))       # (h2, h1)
    _L2B = np.ascontiguousarray(d["l2_b"].astype(np.float32))
    _OW = np.ascontiguousarray(d["out_w"].astype(np.float32))       # (1, h2)
    _OB = np.ascontiguousarray(d["out_b"].astype(np.float32))
    _CR = np.float32(float(d["cr_max"]))
    # absent from pre-Stage-3 exports -- default to the old bounded-activation,
    # no-gain shape so v1/v2/Stage-1/Stage-2 checkpoints keep working unchanged.
    _CLIPPED_RELU = np.bool_(bool(d["clipped_relu"])) if "clipped_relu" in d else np.bool_(True)
    _GAIN = np.float32(float(d["gain"])) if "gain" in d else np.float32(1.0)
    _LOADED[0] = True
    return {"h1": int(d["h1"]), "h2": int(d["h2"]), "eval_scale": float(d["eval_scale"])}


def nnue_eval_state(bb: np.ndarray) -> int:
    if not _LOADED[0]:
        raise RuntimeError("nnue_eval_v3.load_model() not called")
    return int(_forward(bb, _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR, _KBT, _CLIPPED_RELU, _GAIN))


@njit(cache=False, fastmath=True)
def eval_state(bb, ev):
    """Drop-in for ``evaluate_bb.eval_state`` -- the ``ev`` bundle is ignored.
    Reads the weight globals, which ``load_model`` must have set before the
    first call (lazy njit compile then freezes the loaded arrays)."""
    return _forward(bb, _L1W, _L1B, _L2W, _L2B, _OW, _OB, _CR, _KBT, _CLIPPED_RELU, _GAIN)
