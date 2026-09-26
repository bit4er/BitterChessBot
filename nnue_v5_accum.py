"""NNUE-v5 quiet-non-king fast path (2026-09-08 optimization round).

The accumulator-opt report's follow-up decomposition (docs/
NNUE_ACCUMULATOR_CLAUDE_OPT_2026-09-08.md §24) found accumulator-
MAINTENANCE cost dominates the tail at H1=256, because it runs on 100% of
nodes while the tail only runs on ~55%. The v5 speed-prototype report
(docs/NNUE_V5_SPEED_PROTOTYPE_2026-09-08.md §10) then found `update_accum`'s
own per-call cost is essentially FLAT across H1=32/64/96 (0.144-0.177us) --
at these small widths its FIXED overhead, not its O(H1) `apply_delta` loop
work, dominates. This module tests that hypothesis directly by removing
every part of that fixed overhead that a quiet, non-king move never needs.

``nnue_accum.update_accum``/``undo_accum`` (UNCHANGED, still imported and
used for every non-quiet or king move) unconditionally pay, even for an
ordinary pawn push:

  - a king-bucket-crossing test (two divisions-by-16-as-shift, two masks,
    two multiplies) that can only ever matter for a king move
  - an EP-flag branch and a capture branch (both false for a quiet move)
  - a promotion ternary (always false for a quiet move)
  - a castle-flag branch (always false for a quiet move -- castling is
    always encoded as a king move, so it can never reach this code path
    once the caller has already branched on ``is_king``)
  - a "did the king move" branch on the return value

A quiet non-king move needs exactly four ``apply_delta`` calls (remove the
piece from ``frm``, add it at ``to``, both perspectives) and nothing else --
king squares provably cannot change, so this fast path does not even
return them; callers skip the king-square bookkeeping entirely when they
take this path (see jsearch.py's ``_v5_is_king`` gate).

Correctness: this is not a new numerical rule, only a removal of
branches/computations that were always no-ops on this input shape -- the
row math (``_row``/``apply_delta``, imported unchanged from
``nnue_accum.py``) is byte-identical to what ``update_accum`` would have
computed for the same inputs. Verified, not assumed -- see
``tools/claude_verify_v5_fastpath.py``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from nnue_accum import apply_delta


@njit(cache=True, inline="always")
def update_accum_quiet(l1w_q: np.ndarray, white_acc: np.ndarray, black_acc: np.ndarray,
                       white_king: int, black_king: int,
                       frm: int, to: int, moving_code: int, kbt: np.ndarray) -> None:
    """Ordinary non-king, non-capture, non-promotion, non-EP, non-castle
    move only -- caller (jsearch.py) is responsible for the gate. No
    king-bucket test, no EP/capture/promo/castle branches, no return value
    (king squares are unchanged by construction for every move this
    function is ever called on)."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    apply_delta(white_acc, l1w_q, 1, frm, us_white, moving_pt, white_king, -1, kbt)
    apply_delta(black_acc, l1w_q, 0, frm, us_white, moving_pt, black_king, -1, kbt)
    apply_delta(white_acc, l1w_q, 1, to, us_white, moving_pt, white_king, +1, kbt)
    apply_delta(black_acc, l1w_q, 0, to, us_white, moving_pt, black_king, +1, kbt)


@njit(cache=True, inline="always")
def undo_accum_quiet(l1w_q: np.ndarray, white_acc: np.ndarray, black_acc: np.ndarray,
                     white_king: int, black_king: int,
                     frm: int, to: int, moving_code: int, kbt: np.ndarray) -> None:
    """Exact inverse of ``update_accum_quiet`` -- every sign flipped, same
    mechanical mirroring discipline as ``nnue_accum.undo_accum``."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    apply_delta(white_acc, l1w_q, 1, to, us_white, moving_pt, white_king, -1, kbt)
    apply_delta(black_acc, l1w_q, 0, to, us_white, moving_pt, black_king, -1, kbt)
    apply_delta(white_acc, l1w_q, 1, frm, us_white, moving_pt, white_king, +1, kbt)
    apply_delta(black_acc, l1w_q, 0, frm, us_white, moving_pt, black_king, +1, kbt)
