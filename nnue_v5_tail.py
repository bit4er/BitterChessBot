"""NNUE-v5 speed-prototype tail -- original design, not derived from Numbfish
or black_numba (neither reference repo's fetched source used this pattern:
Numbfish's own tail is float32 throughout with a fresh per-node allocation;
black_numba's fetched files did not contain an NNUE tail at all -- see
docs/NNUE_V5_SPEED_PROTOTYPE_2026-09-08.md §6 for what was actually found
when both were checked directly). This module and its accumulator wiring in
jsearch.py ARE this thought-experiment branch's own prototype code, kept
separate from any eventual competition implementation per the standing
instruction not to ship a from-reference translation.

Feature indexing, the king-bucket function, and all accumulator maintenance
(``update_accum``/``undo_accum``/``accum_full``/``apply_delta``) are
UNCHANGED, reused as-is from ``nnue_accum.py`` -- both are already
hidden-width-agnostic (every H1/H2 is read from array shapes at runtime, not
hardcoded), so the proven, 158,960-step-verified mutate/undo accumulator
needs no changes at all to serve H=32/64/96. Only this tail is new.

Unlike v4.1's float32 tail (``nnue_accum.eval_from_accum``, which matches a
pre-existing float ``_forward`` reference bit-for-bit by design), this tail
never leaves integer arithmetic: the wide int32 accumulator is brought down
to a small clipped activation with one arithmetic right-shift (no float
divide, no dtype promotion to float32/float64 anywhere), int16 tail weights
multiply against that int32 activation, and every partial sum stays int32.
Overflow is proven safe by construction, not assumed -- see
``_v5_int32_headroom`` below and the report's dtype-analysis section.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def _v5_int32_headroom(h1: int, act_max: int, w_max: int) -> int:
    """Prove int32 is sufficient for the tail's inner dot product: worst-case
    |term| = w_max * act_max, summed over 2*h1 terms (both perspectives
    concatenated). Returns the worst-case magnitude so callers can assert it
    against int32's range rather than assume it."""
    return w_max * act_max * 2 * h1


def build_dummy_weights(h1: int, h2: int, seed: int) -> dict:
    """Random weights at the given widths, correctly typed/shaped for the
    v5 tail -- values are meaningless (this is the speed-only prototype
    phase; DO NOT TRAIN yet, see the report), only dtype/shape/magnitude
    need to be realistic so the benchmark pays the same arithmetic cost a
    trained network would.

    Feature weights (``l1w_q``) reuse ``nnue_accum``'s own int16/QSCALE=4096
    convention unchanged (6144 = 8 king buckets x 12 planes x 64 squares,
    same as v4.1 -- bucket count was not varied here, see report §5: it
    does not affect per-move update cost, only feature-table size, which is
    already tiny at every H considered).
    """
    rng = np.random.default_rng(seed)
    l1w_q = rng.integers(-800, 800, size=(6144, h1), dtype=np.int32).astype(np.int16)
    l1b_q = rng.integers(-2000, 2000, size=h1, dtype=np.int32)
    # Tail weights: int16, bounded so build_dummy_weights' own w_max matches
    # what _v5_int32_headroom is asked to prove safe below.
    hw = rng.integers(-127, 128, size=(h2, 2 * h1), dtype=np.int32).astype(np.int16)
    hb = rng.integers(-2000, 2000, size=h2, dtype=np.int32)
    ow = rng.integers(-127, 128, size=(1, h2), dtype=np.int32).astype(np.int16)
    ob = rng.integers(-2000, 2000, size=1, dtype=np.int32)
    return {"l1w_q": l1w_q, "l1b_q": l1b_q, "hw": hw, "hb": hb, "ow": ow, "ob": ob}


ACT_MAX = 127          # clipped activation range, int8-sized
V5_SHIFT = 6            # acc[j] >> V5_SHIFT -- placeholder for the prototype
                         # phase; a real shift is a training-time calibration,
                         # not a speed question, so it was not tuned here.


@njit(cache=True, inline="always")
def v5_tail(white_acc: np.ndarray, black_acc: np.ndarray, stm: int,
           act1: np.ndarray, act2: np.ndarray,
           hw: np.ndarray, hb: np.ndarray, ow: np.ndarray, ob: np.ndarray,
           shift: int, act_max: int) -> int:
    """2*H1 -> H2 -> 1, pure int32, zero allocations (``act1``/``act2`` are
    preallocated scratch buffers owned by the caller -- ``Searcher``, same
    convention as ``stack``/``mscore``/``scratch`` -- and reused across every
    call, never allocated here). No float anywhere in this function.

    ``act1``/``act2`` are materialised once (not recomputed per output unit)
    because each of their H1 elements is read H2 times below -- the same
    reuse-factor argument the accumulator-opt report (§8h) already measured
    for the v4.1 tail: recomputing per-read would cost H2x the dequant work
    to avoid an H1-sized array that, at these widths (32-96 elements), is
    already tiny.
    """
    h1 = white_acc.shape[0]
    if stm == 0:
        src_first, src_second = white_acc, black_acc
    else:
        src_first, src_second = black_acc, white_acc
    for j in range(h1):
        fv = src_first[j] >> shift
        if fv < 0:
            fv = 0
        elif fv > act_max:
            fv = act_max
        act1[j] = fv
        sv = src_second[j] >> shift
        if sv < 0:
            sv = 0
        elif sv > act_max:
            sv = act_max
        act2[j] = sv
    h2 = hb.shape[0]
    out = ob[0]
    for k in range(h2):
        s = np.int32(0)
        for j in range(h1):
            s += hw[k, j] * act1[j] + hw[k, h1 + j] * act2[j]
        hidk = hb[k] + s
        if hidk < 0:
            hidk = 0
        elif hidk > act_max:
            hidk = act_max
        out += ow[0, k] * hidk
    return int(out)
