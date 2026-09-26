"""Compile-time bridge for the NNUE-v5 speed prototype (2026-09-08), extended
2026-09-09 with an opt-in H256/4-king-bucket/horizontally-mirrored
architecture (isolated worktree claude/nnue-h256-4bucket-mirror -- see
docs/NNUE_H256_4BUCKET_MIRROR_1P7M.md). Does NOT touch the confirmed
H16/H32/H64/H96 safety checkpoint's behavior: MIRROR4=False (the default)
reproduces that build's exact code path, byte-for-byte, unchanged below.

Same pattern as ``nnue_accum_search.py`` (frozen-at-import constants that
``jsearch``'s njit functions branch on, so numba prunes the untaken side and
V5_ON=False is byte-identical to before this file existed). Kept fully
separate from ``nnue_accum_search.py``'s ``ACCUM_ON``/``MAINT_ONLY``/
``TAIL_IGNORED`` machinery -- deliberately not unified with it, so this
prototype cannot perturb the already-verified v4.1 accumulator-opt paths.

``AICHESS_NNUE_V5=1`` turns it on. ``AICHESS_NNUE_V5_H1`` selects the
accumulator width (32/64/96, default 64); H2 is fixed at H1//4 rounded to
the nearest of {8, 16} per the task's own "2H -> 16 -> 1 or 2H -> 8 -> 1"
guidance -- H1=32 -> H2=8, H1=64/96 -> H2=16. Weights are DUMMY (random,
``nnue_v5_tail.build_dummy_weights``) -- this is the pre-training speed-only
phase; see the report for why training was not started.

``AICHESS_NNUE_V5_MIRROR4=1`` (requires ``AICHESS_NNUE_V5=1``) switches
EVERY symbol this module exports (``L1W_Q``/``L1B_Q``/``KBT``/
``update_accum``/``undo_accum``/``update_accum_quiet``/
``undo_accum_quiet``/``accum_full``/``v5_tail``) to the 4-bucket-mirrored
H256 architecture (``nnue_h256_mirror_accum.py``, ``nnue_h256_tail.py``)
instead -- deliberately keeping every exported NAME and CALL SIGNATURE
identical to the H1->H2->1 8-bucket path, so jsearch.py's move-loop
wiring (every ``_v5_update_accum``/``_v5_tail``/etc. call site) needs
ZERO changes to support the new architecture. ``H1`` defaults to 256 in
this mode (still overridable via ``AICHESS_NNUE_V5_H1`` for width-racing
within the new architecture); ``H2``/``HW``/``HB`` become unused 1x1
dummies (the new tail has no intermediate dense layer, per the
architecture's own design -- see nnue_h256_tail.py). ``AICHESS_NNUE_V5_
H256_SCRELU=1`` selects SCReLU over the default CReLU activation.

The tail's result is always discarded by ``_leaf_eval`` UNLESS
``AICHESS_NNUE_V5_PURE=1`` is also set (jsearch.py) -- see that flag's own
docstring in jsearch.py for why. This module alone does not make either
architecture a playable evaluator.
"""

from __future__ import annotations

import os

import numpy as np

# COMPETITION DEFAULT. The packaged submission must run the trained H256 net
# with no environment variable set at all (the platform imports agent.py
# directly), so the presence of the bundled weights file is what turns the
# NNUE path on. Set AICHESS_NNUE_V5=0 to force the classical evaluator back
# on -- that is how speed work measures the classical denominator through
# this same search.
# Lives in weights/ because harness.package ships that directory by default,
# so the archive carries the net without any extra --include.
H256_WEIGHTS_DEFAULT: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "h256_competition.npz",
)
H256_BUNDLED: bool = os.path.isfile(H256_WEIGHTS_DEFAULT)
_D: str = "1" if H256_BUNDLED else "0"

V5_ON: bool = os.environ.get("AICHESS_NNUE_V5", _D) == "1"
MIRROR4: bool = V5_ON and os.environ.get("AICHESS_NNUE_V5_MIRROR4", _D) == "1"
_H256_SCRELU: bool = os.environ.get("AICHESS_NNUE_V5_H256_SCRELU", "0") == "1"
H1: int = int(os.environ.get("AICHESS_NNUE_V5_H1", "256" if MIRROR4 else "64")) if V5_ON else 1
H2: int = (8 if H1 <= 32 else 16) if V5_ON else 1
SHIFT: int = int(os.environ.get("AICHESS_NNUE_V5_SHIFT", "6"))
ACT_MAX: int = 127
OUT_SCALE: int = int(os.environ.get("AICHESS_NNUE_V5_OUT_SCALE", "16"))
_SEED: int = int(os.environ.get("AICHESS_NNUE_V5_SEED", "20260908"))
_WEIGHTS_PATH: str | None = (os.environ.get("AICHESS_NNUE_V5_WEIGHTS")
                             or (H256_WEIGHTS_DEFAULT if H256_BUNDLED else None))

if MIRROR4:
    from nnue_h256_tail import (
        build_dummy_weights as _build_h256_weights,
        h256_tail_crelu, h256_tail_crelu_branchless, h256_tail_screlu,
        _h256_int32_headroom,
    )
    from nnue_h256_mirror_accum import (
        accum_full4 as accum_full, update_accum4 as update_accum,
        undo_accum4 as undo_accum, update_accum_quiet4 as update_accum_quiet,
        undo_accum_quiet4 as undo_accum_quiet,
    )
    from nnue_accum import quantize_l1 as _quantize_l1  # unused here, kept for parity

    HW = np.zeros((1, 1), np.int16)  # unused -- no dense hidden layer in this tail
    HB = np.zeros(1, np.int32)       # unused
    KBT = np.zeros(64, np.int64)

    if _WEIGHTS_PATH:
        with np.load(_WEIGHTS_PATH, allow_pickle=False) as _trained:
            _trained_h1 = int(_trained["h1"])
            _trained_activation = str(_trained["activation"])
            _trained_shift = int(_trained["shift"])
            _trained_act_max = int(_trained["act_max"])
            _trained_out_scale = int(_trained["out_scale"])
            if _trained_h1 != H1:
                raise ValueError(f"trained H1={_trained_h1}, configured H1={H1}")
            if _trained_activation != ("screlu" if _H256_SCRELU else "crelu"):
                raise ValueError(
                    f"trained activation={_trained_activation}, configured "
                    f"activation={'screlu' if _H256_SCRELU else 'crelu'}"
                )
            if _trained_shift != SHIFT or _trained_act_max != ACT_MAX:
                raise ValueError(
                    f"trained quantization shift/act_max={_trained_shift}/{_trained_act_max}, "
                    f"engine={SHIFT}/{ACT_MAX}"
                )
            if _trained_out_scale != OUT_SCALE:
                raise ValueError(
                    f"trained out_scale={_trained_out_scale}, configured out_scale={OUT_SCALE}"
                )
            L1W_Q = _trained["l1w_q"].astype(np.int16, copy=True)
            L1B_Q = _trained["l1b_q"].astype(np.int32, copy=True)
            OW = _trained["ow"].astype(np.int16, copy=True)
            OB = _trained["ob"].astype(np.int32, copy=True)
        if L1W_Q.shape != (3072, H1) or L1B_Q.shape != (H1,):
            raise ValueError(f"invalid trained feature shapes: {L1W_Q.shape}, {L1B_Q.shape}")
        if OW.shape != (2 * H1,) or OB.shape != (1,):
            raise ValueError(f"invalid trained tail shapes: {OW.shape}, {OB.shape}")
    else:
        _rng = np.random.default_rng(_SEED)
        # Feature table: 4 buckets x 12 planes x 64 squares = 3072 rows per
        # perspective, shared by both perspectives.
        L1W_Q = _rng.integers(-800, 800, size=(3072, H1), dtype=np.int32).astype(np.int16)
        L1B_Q = _rng.integers(-2000, 2000, size=H1, dtype=np.int32)
        _WT = _build_h256_weights(H1, _SEED + 1)
        OW = _WT["ow"]
        OB = _WT["ob"]

    _worst = _h256_int32_headroom(H1, ACT_MAX, 127, _H256_SCRELU)
    assert _worst < np.iinfo(np.int32).max, (
        f"h256 tail dot product worst case {_worst:,} would overflow int32 "
        f"({np.iinfo(np.int32).max:,}) -- shrink ACT_MAX, the weight bound, "
        f"or H1"
    )

    from numba import njit

    if _H256_SCRELU:
        @njit(cache=True, inline="always")
        def v5_tail(white_acc, black_acc, stm, act1, act2, hw, hb, ow, ob, shift, act_max):  # noqa: D103
            # act1/act2/hw/hb are unused (identical call signature to the
            # H1->H2->1 tail so jsearch.py's call site needs no change --
            # see this module's own docstring).
            return h256_tail_screlu(white_acc, black_acc, stm, ow, ob, shift, act_max)
    elif os.environ.get("AICHESS_H256_TAIL_IMPL", "branch") == "branchless":
        # A/B control only -- the REJECTED branchless-clip kernel (-5.34%,
        # 0/4 pairwise). Bit-identical output; kept so the null result stays
        # reproducible. See nnue_h256_tail.h256_tail_crelu_branchless.
        @njit(cache=True, inline="always")
        def v5_tail(white_acc, black_acc, stm, act1, act2, hw, hb, ow, ob, shift, act_max):  # noqa: D103
            return h256_tail_crelu_branchless(white_acc, black_acc, stm, ow, ob, shift, act_max)
    else:
        @njit(cache=True, inline="always")
        def v5_tail(white_acc, black_acc, stm, act1, act2, hw, hb, ow, ob, shift, act_max):  # noqa: D103
            return h256_tail_crelu(white_acc, black_acc, stm, ow, ob, shift, act_max)

elif V5_ON:
    from nnue_v5_tail import build_dummy_weights, v5_tail, _v5_int32_headroom
    from nnue_accum import accum_full, update_accum, undo_accum
    from nnue_v5_accum import update_accum_quiet, undo_accum_quiet

    _W = build_dummy_weights(H1, H2, _SEED)
    L1W_Q = _W["l1w_q"]
    L1B_Q = _W["l1b_q"]
    HW = _W["hw"]
    HB = _W["hb"]
    OW = _W["ow"]
    OB = _W["ob"]
    KBT = np.zeros(64, np.int64)

    # int32-sufficiency proof (not assumed): worst-case tail dot-product
    # magnitude vs int32's range, for the widest config this module can be
    # asked to build (H1=96, the largest value AICHESS_NNUE_V5_H1 accepts
    # in this benchmark).
    _worst = _v5_int32_headroom(96, ACT_MAX, 127)
    assert _worst < np.iinfo(np.int32).max, (
        f"v5 tail dot product worst case {_worst:,} would overflow int32 "
        f"({np.iinfo(np.int32).max:,}) -- shrink ACT_MAX or the weight bound"
    )
else:
    from nnue_accum import accum_full, update_accum, undo_accum

    L1W_Q = np.zeros((1, 1), np.int16)
    L1B_Q = np.zeros(1, np.int32)
    HW = np.zeros((1, 1), np.int16)
    HB = np.zeros(1, np.int32)
    OW = np.zeros((1, 1), np.int16)
    OB = np.zeros(1, np.int32)
    KBT = np.zeros(64, np.int64)

    from numba import njit

    @njit(cache=True)
    def v5_tail(white_acc, black_acc, stm, act1, act2, hw, hb, ow, ob, shift, act_max):  # noqa: D103
        return 0

    @njit(cache=True)
    def update_accum_quiet(l1w_q, white_acc, black_acc, white_king, black_king, frm, to, mc, kbt):  # noqa: D103
        pass

    @njit(cache=True)
    def undo_accum_quiet(l1w_q, white_acc, black_acc, white_king, black_king, frm, to, mc, kbt):  # noqa: D103
        pass
