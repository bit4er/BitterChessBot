"""Lazy/deferred accumulator materialization for the H256 architecture
(2026-09-09, Priority 1 of the H256 experiment -- "probably the single
most important H256 optimization" per the task brief).

## Why the existing single-shared-mutable-accumulator design (nnue_accum.py,
nnue_h256_mirror_accum.py) can't just be "made lazy" by skipping some calls

The naive idea -- "track one global valid_ply scalar, skip apply_delta4 on
the way down, walk the ply-diff at eval time" -- is UNSAFE: `dirty[ply]`
records get overwritten every time the search backtracks to a sibling
move at the same ply, but a single global `valid_ply` can be pointing at
a DEEPER ply from an already-abandoned branch when that overwrite
happens, silently corrupting any later "walk backward using dirty[]"
replay (it would use the WRONG branch's recorded deltas). This was
caught during design, not discovered as a bug later -- see
docs/H256_SOURCE_PROVENANCE.md / docs/NNUE_H256_4BUCKET_MIRROR_1P7M.md.

## The actual (correct) design

Per-ply ACCUMULATOR SNAPSHOTS (`acc_stack[ply, :]`, one row per ply, not
a single shared array) plus a per-ply `materialized[ply]` boolean flag:

- `make_move` (the caller, jsearch.py) does NOT call the O(H1) delta
  application at all for a non-crossing move -- it only records the tiny
  dirty tuple (frm, to, moving_code, captured_code, promo, flag) at
  `dirty[new_ply]` (O(1)) and sets `materialized[new_ply] = False`. For a
  KING-BUCKET-CROSSING move (the one case that genuinely cannot be
  deferred -- a full refresh needs the ACTUAL board, which is only
  available synchronously at make-time, before the search recurses past
  it), the caller still does the eager `accum_full4` immediately (same
  as the non-lazy path), writes the result directly into
  `acc_stack[new_ply, :]`, and sets `materialized[new_ply] = True` right
  away.
- Eval time (`_leaf_eval`): call `materialize(ply)` for whichever
  perspective(s) are needed. It walks BACKWARD from `ply` through
  `materialized[]` to find the nearest ancestor ply `q` that is already
  valid (bounded by the root, ply 0, which is always materialized), then
  replays FORWARD from `q+1` to `ply`, applying each intervening ply's
  recorded delta onto a copy of the previous ply's snapshot -- marking
  every intermediate ply materialized too (a legitimate cache: a sibling
  eval near this depth reuses this work for free).
- `unmake_move`: needs NO special handling at all. `materialized[ply]`
  is only ever reset by `make_move`, so an old, now-stale
  `acc_stack[ply]`/`materialized[ply]=True` from an abandoned branch is
  simply never read again -- the next `make_move` that reuses that ply
  slot resets `materialized[ply] = False` unconditionally before
  anything downstream can rely on it.

This is the correctness argument, not merely the design intent -- see
`tools/claude_verify_h256_lazy.py` for the adversarial test that
specifically exercises "materialize deep in branch A, backtrack, take a
different branch B through the same ply slots, materialize again" to
prove no aliasing survives.

## What's NOT deferred

Bucket-crossing king moves still pay the full `accum_full4` cost
synchronously (this is a structural requirement of the single-mutable-
board architecture, not a missed optimization -- see above). A future
Finny-table layer (Priority 2) is the correct next lever for THAT
specific cost, not this module.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from nnue_h256_mirror_accum import _row4, _bucket_and_mirror, accum_full4


@njit(cache=True, inline="always")
def record_dirty(dirty_frm: np.ndarray, dirty_to: np.ndarray, dirty_mc: np.ndarray,
                 dirty_cc: np.ndarray, dirty_promo: np.ndarray, dirty_flag: np.ndarray,
                 ply: int, frm: int, to: int, mc: int, cc: int, promo: int, flag: int) -> None:
    """O(1) -- called unconditionally by make_move for every move, king or
    not, crossing or not (the crossing/refresh case ALSO records this, so
    a re-materialization of a LATER ply that walks through this one via
    the delta path -- impossible for a true crossing ply since it's
    always pre-materialized, but harmless/unused in that case -- has
    consistent data either way)."""
    dirty_frm[ply] = frm
    dirty_to[ply] = to
    dirty_mc[ply] = mc
    dirty_cc[ply] = cc
    dirty_promo[ply] = promo
    dirty_flag[ply] = flag


@njit(cache=True, inline="always")
def _apply_ply_delta(acc: np.ndarray, l1w_q: np.ndarray, persp_white: int,
                     king_sq: int, frm: int, to: int, mc: int, cc: int,
                     promo: int, flag: int, kbt: np.ndarray) -> None:
    """The non-king incremental delta -- same row math as
    nnue_h256_mirror_accum.update_accum4's non-crossing branch, applied
    to an arbitrary target `acc` array (a materialization working copy,
    not necessarily "the" shared accumulator)."""
    FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 2, 3, 4
    us_white = 1 if mc <= 6 else 0
    moving_pt = (mc - 1) % 6
    h1 = acc.shape[0]

    row = _row4(persp_white, frm, us_white, moving_pt, king_sq, kbt)
    for j in range(h1):
        acc[j] -= l1w_q[row, j]

    if flag == FLAG_EP:
        cap_sq = to - 8 if us_white else to + 8
        cap_white = 0 if us_white else 1
        row = _row4(persp_white, cap_sq, cap_white, 0, king_sq, kbt)
        for j in range(h1):
            acc[j] -= l1w_q[row, j]
    elif cc != 0:
        cap_white = 1 if cc <= 6 else 0
        cap_pt = (cc - 1) % 6
        row = _row4(persp_white, to, cap_white, cap_pt, king_sq, kbt)
        for j in range(h1):
            acc[j] -= l1w_q[row, j]

    arrive_pt = (promo - 1) if promo != 0 else moving_pt
    row = _row4(persp_white, to, us_white, arrive_pt, king_sq, kbt)
    for j in range(h1):
        acc[j] += l1w_q[row, j]

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        rank = 0 if us_white else 7
        if flag == FLAG_CASTLE_K:
            r_from, r_to = rank * 8 + 7, rank * 8 + 5
        else:
            r_from, r_to = rank * 8 + 0, rank * 8 + 3
        row = _row4(persp_white, r_from, us_white, 3, king_sq, kbt)
        for j in range(h1):
            acc[j] -= l1w_q[row, j]
        row = _row4(persp_white, r_to, us_white, 3, king_sq, kbt)
        for j in range(h1):
            acc[j] += l1w_q[row, j]


@njit(cache=True, inline="always")
def lazy_record_null(
    materialized_w: np.ndarray, materialized_b: np.ndarray,
    own_king_w: np.ndarray, own_king_b: np.ndarray,
    dirty_mc: np.ndarray, ply: int,
) -> None:
    """Record a NULL MOVE at `ply` -- touches no piece, so the accumulator
    is unchanged, but `ply` still advances (the search recurses one ply
    deeper to probe). `dirty_mc[ply] = 0` is the sentinel `materialize`
    recognizes as "no-op, just carry the parent snapshot forward" (real
    piece codes are always 1..12, so 0 is unambiguous). Own-king-square
    tracking also just carries forward (a null move never moves the
    king). MUST be called at every null-move site, or `materialize()`
    would read whatever stale dirty[]/materialized[] data was last left
    at this ply slot by an unrelated branch -- this was a real bug found
    via the eager-vs-lazy depth-5 node-equivalence check, not merely
    anticipated; see docs/NNUE_H256_4BUCKET_MIRROR_1P7M.md."""
    dirty_mc[ply] = 0
    own_king_w[ply] = own_king_w[ply - 1]
    own_king_b[ply] = own_king_b[ply - 1]
    materialized_w[ply] = False
    materialized_b[ply] = False


@njit(cache=True, inline="always")
def lazy_record(
    bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
    acc_stack_w: np.ndarray, acc_stack_b: np.ndarray,
    materialized_w: np.ndarray, materialized_b: np.ndarray,
    own_king_w: np.ndarray, own_king_b: np.ndarray,
    dirty_frm: np.ndarray, dirty_to: np.ndarray, dirty_mc: np.ndarray,
    dirty_cc: np.ndarray, dirty_promo: np.ndarray, dirty_flag: np.ndarray,
    ply: int, frm: int, to: int, mc: int, cc: int, promo: int, flag: int,
) -> None:
    """The ONE call jsearch.py's move loop needs at make-time (replacing
    the eager update_accum/update_accum_quiet pair entirely -- the quiet/
    general distinction doesn't matter here, it's only relevant at REPLAY
    time inside `materialize`). Call AFTER make_move has mutated bb.

    Records the shared dirty tuple once (O(1)), then for a king move
    checks THIS PERSPECTIVE'S OWN crossing condition independently for
    white and black -- a crossing perspective gets its accumulator
    refreshed eagerly right now (using the just-mutated `bb`, the only
    point this board state is available) and marked materialized; a
    non-crossing perspective (including both perspectives on any
    non-king move) is simply marked NOT materialized, deferring all
    O(H1) work to whenever `materialize()` is actually called for it."""
    record_dirty(dirty_frm, dirty_to, dirty_mc, dirty_cc, dirty_promo, dirty_flag,
                ply, frm, to, mc, cc, promo, flag)
    own_king_w[ply] = own_king_w[ply - 1]
    own_king_b[ply] = own_king_b[ply - 1]
    materialized_w[ply] = False
    materialized_b[ply] = False

    moving_pt = (mc - 1) % 6
    if moving_pt != 5:
        return
    us_white = 1 if mc <= 6 else 0
    if us_white:
        old_rel = own_king_w[ply - 1]
        new_rel = to
        old_b, old_m = _bucket_and_mirror(old_rel)
        new_b, new_m = _bucket_and_mirror(new_rel)
        if old_b != new_b or old_m != new_m:
            acc_stack_w[ply, :] = accum_full4(bb, l1w_q, l1b_q, kbt, 1)
            materialized_w[ply] = True
        own_king_w[ply] = to
    else:
        old_rel = own_king_b[ply - 1] ^ 56
        new_rel = to ^ 56
        old_b, old_m = _bucket_and_mirror(old_rel)
        new_b, new_m = _bucket_and_mirror(new_rel)
        if old_b != new_b or old_m != new_m:
            acc_stack_b[ply, :] = accum_full4(bb, l1w_q, l1b_q, kbt, 0)
            materialized_b[ply] = True
        own_king_b[ply] = to


@njit(cache=True)
def materialize(acc_stack: np.ndarray, materialized: np.ndarray,
                dirty_frm: np.ndarray, dirty_to: np.ndarray, dirty_mc: np.ndarray,
                dirty_cc: np.ndarray, dirty_promo: np.ndarray, dirty_flag: np.ndarray,
                own_king_sq_at: np.ndarray, l1w_q: np.ndarray, persp_white: int,
                target_ply: int, kbt: np.ndarray) -> None:
    """Ensure acc_stack[target_ply, :] is valid, walking back to the
    nearest materialized ancestor and replaying forward. `own_king_sq_at`
    is a per-ply array of THIS perspective's own king square at each ply
    (needed for _row4's bucket/mirror computation on the delta rows --
    non-king moves never change it, so it's just carried forward, not
    recomputed here)."""
    q = target_ply
    while not materialized[q]:
        q -= 1
    for k in range(q + 1, target_ply + 1):
        acc_stack[k, :] = acc_stack[k - 1, :]
        if dirty_mc[k] != 0:
            # dirty_mc[k] == 0 is the null-move sentinel (see
            # lazy_record_null) -- no piece moved, so the copy above is
            # already the correct result; skip the delta entirely.
            king_sq = own_king_sq_at[k]
            _apply_ply_delta(
                acc_stack[k, :], l1w_q, persp_white, king_sq,
                dirty_frm[k], dirty_to[k], dirty_mc[k], dirty_cc[k],
                dirty_promo[k], dirty_flag[k], kbt,
            )
        materialized[k] = True
