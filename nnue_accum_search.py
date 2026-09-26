"""Compile-time bridge for v4.1 accumulator and selective-NNUE experiments.

Exposes a single compile-time constant ``ACCUM_ON`` (frozen at import) that
``jsearch``'s njit functions branch on -- numba constant-folds the untaken
branch, so ``ACCUM_ON = False`` is byte-identical to the pre-accumulator
search. When on, also re-exports the njit ops from ``nnue_accum`` and the
weight arrays from ``nnue_eval_v3`` (populated by its ``load_model``).

``AICHESS_NNUE_ACCUM=1`` turns it on; it additionally requires the v3
king-bucketed NNUE backend to be active (``engine_eval`` must have loaded
``nnue_eval_v3`` -- i.e. ``AICHESS_NNUE`` points at a 12288-feature net).
Default off until gates 5/6/7 pass; the ship build flips the default.

See ``docs/NNUE_ACCUMULATOR_DESIGN_2026-09-07.md`` §10.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit

import engine_eval

_V41 = (
    engine_eval.BACKEND.startswith("nnue:")
    and os.environ.get("AICHESS_NNUE_RESIDUAL", "0") == "1"
    and hasattr(getattr(engine_eval, "nnue_eval", None), "nnue_eval_v4")
)

ACCUM_ON: bool = _V41 and os.environ.get("AICHESS_NNUE_ACCUM", "0") == "1"
REFRESH_ON: bool = ACCUM_ON and os.environ.get("AICHESS_NNUE_ACCUM_REFRESH", "0") == "1"
SPARSE_PCT: int = int(os.environ.get("AICHESS_NNUE_SPARSE_PCT", "0")) if _V41 else 0
SPARSE_ON: bool = 0 < SPARSE_PCT <= 100
ROOT_ON: bool = _V41 and os.environ.get("AICHESS_NNUE_ROOT", "0") == "1"
ORDER_TOP: int = int(os.environ.get("AICHESS_NNUE_ORDER_TOP", "0")) if _V41 else 0
ORDER_ON: bool = ORDER_TOP > 0

if ACCUM_ON or SPARSE_ON or ROOT_ON or ORDER_ON:
    import nnue_eval_v4 as _nv3
    from nnue_accum import accum_full, eval_from_accum, update_accum, undo_accum
    from nnue_accum import quantize_l1

    _L1W_Q, _L1B_Q = quantize_l1(_nv3._FT, _nv3._FTB)
    H1: int = int(_L1B_Q.shape[0])
    _KBT = np.zeros(64, np.int64)
    _L2W = _nv3._HW
    _L2B = _nv3._HB
    _OW = _nv3._OW
    _OB = _nv3._OB
    _CR = np.float32(2.0)
    _CRELU = np.bool_(True)
    _GAIN = _nv3._MULTIPLIER
    _SCALE = np.float32(os.environ.get("AICHESS_NNUE_RESIDUAL_SCALE", "1.0"))
    _CAP = np.int32(os.environ.get("AICHESS_NNUE_RESIDUAL_CAP", "0"))
    _FT = _nv3._FT
    _FTB = _nv3._FTB

    @njit(cache=False, fastmath=True)
    def full_eval(bb):
        return _nv3._forward(bb, _nv3._FT, _nv3._FTB, _nv3._HW, _nv3._HB,
                             _nv3._OW, _nv3._OB, _nv3._MULTIPLIER)
else:
    # dummies so `jsearch` can import unconditionally; never referenced when
    # ACCUM_ON is False (numba prunes those branches).
    H1 = 1
    _L1W_Q = np.zeros((1, 1), np.int32)
    _L1B_Q = np.zeros(1, np.int32)
    _KBT = np.zeros(64, np.int64)
    _L2W = np.zeros((1, 1), np.float32)
    _L2B = np.zeros(1, np.float32)
    _OW = np.zeros((1, 1), np.float32)
    _OB = np.zeros(1, np.float32)
    _CR = np.float32(0.0)
    _CRELU = np.bool_(False)
    _GAIN = np.float32(1.0)
    _SCALE = np.float32(1.0)
    _CAP = np.int32(0)
    _FT = np.zeros((1, 1), np.float32)
    _FTB = np.zeros(1, np.float32)

    @njit(cache=True)
    def accum_full(bb, l1w_q, l1b_q, kbt, persp_white):  # noqa: D103
        return l1b_q.copy()

    @njit(cache=True)
    def eval_from_accum(w, b, stm, l2w, l2b, ow, ob, gain):  # noqa: D103
        return 0

    @njit(cache=True)
    def update_accum(bb, l1w_q, l1b_q, kbt, w, b, wk, bk, frm, to, mc, cc, promo, flag):  # noqa: D103
        return wk, bk

    @njit(cache=True)
    def undo_accum(bb, l1w_q, l1b_q, kbt, w, b, wk, bk, frm, to, mc, cc, promo, flag):  # noqa: D103
        return wk, bk

    @njit(cache=True)
    def full_eval(bb):
        return 0
