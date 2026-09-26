"""H256 / 4-king-bucket / horizontally-mirrored / two-perspective accumulator
(2026-09-09, isolated NNUE architecture experiment, worktree
claude/nnue-h256-4bucket-mirror -- see docs/NNUE_H256_4BUCKET_MIRROR_1P7M.md).

Adapted from ``nnue_accum.py``'s proven in-place mutate/undo accumulator
design (same function shapes: ``_row``/``accum_full``/``apply_delta``/
``update_accum``/``undo_accum``) -- see that module for the full design
rationale (int32 exact-associativity argument, per-ply single-shared-
accumulator scheme). What changed here, and ONLY here:

1. **4 king buckets instead of 8**, purely by (post-mirror) king file --
   see ``_bucket_and_mirror`` below.
2. **Horizontal mirroring**: every square used in a perspective's feature
   row (that perspective's own king included) is reflected file-wise
   (``sq ^ 7`` -- flips the low 3 bits, i.e. the file, leaves rank alone
   since sq = rank*8+file) whenever that perspective's own king's file is
   >= 4. After mirroring, the king's file is always in {0,1,2,3}, which
   *is* the bucket index directly -- no separate rank-group multiply like
   the 8-bucket scheme used.
3. Feature table is therefore 4 * 768 = 3072 rows per perspective (half
   the row count of the existing 8-bucket/6144-row table), independent of
   hidden width H1.

Mirror formula (frozen, exact, matches ``V5_READY_...``-style
documentation elsewhere in this project):

    own_king_file = king_sq & 7          (0..7, A..H)
    mirror = own_king_file >= 4
    rel_sq   = (color-relative sq) ^ 7  if mirror else (color-relative sq)
    rel_king = (color-relative king) ^ 7 if mirror else (color-relative king)
    bucket = rel_king & 7                (guaranteed 0..3 after the XOR)

"color-relative" means: for White's perspective, squares are used as-is;
for Black's perspective, every square (including the king) is first
flipped vertically (``sq ^ 56``, rank-flip) -- exactly the same
"perspective flip" ``nnue_accum.py::_row`` already uses, unchanged here.
Horizontal mirroring is a SEPARATE, later step applied on top of that,
independently per perspective (each perspective's own king decides its
own mirror state; White's and Black's mirror states can differ).

**The one subtle correctness trap this design had to guard against**
(found during design, not a code review after the fact -- see
docs/H256_SOURCE_PROVENANCE.md and the ledger for the writeup): a king
step between file 3 (d) and file 4 (e) changes the MIRROR flag but keeps
the BUCKET NUMBER the same (bucket 3 = {file 3 unmirrored, file 4
mirrored} -- the mapping folds both onto the same index). Triggering a
refresh only on "did the bucket number change" would silently miss this
exact transition and corrupt the accumulator (every OTHER piece's active
row depends on the mirror flag, not just the king's own row). The
bucket-crossing test below checks the MIRROR flag explicitly, not just
the bucket number, for exactly this reason -- see
``tools/claude_verify_h256_mirror_seam.py`` for the regression test that
specifically exercises this transition.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit

from bitboard import lsb

# Speed-campaign A/B switch, frozen at import so numba constant-folds it and
# prunes the untaken branch (same pattern as nnue_v5_search.py's V5_ON).
# "scalar" selects the pre-optimization kernel; both are bit-identical, see
# tools/h256_accum_parity.py.
_ACCUM_VECTOR: bool = os.environ.get("AICHESS_H256_ACCUM_IMPL", "vector") != "scalar"

# COMPILE-TIME BUDGET SWITCH. `update_accum4`/`undo_accum4` are ~85 lines each
# and expand `apply_delta4`'s 256-element loop several times over. Force-
# inlining them into the move loops of `_negamax_iter`/`_quiesce_iter` inlines
# that numba IR at every call site before type inference and cost ~70 s of
# extra compile time -- cold `import agent` was 104.8 s against the platform's
# 60 s init budget, so the H256 build could not ship at all. Compiled as
# ordinary calls the same import is 30.3 s. The 256-element body dwarfs the
# call overhead, and the search tree is bit-identical either way (same node
# counts). Set AICHESS_H256_ACCUM_INLINE=1 to restore the old force-inlined
# form for A/B measurement -- it is not shippable.
_ACCUM_INLINE = "always" if os.environ.get("AICHESS_H256_ACCUM_INLINE", "0") == "1" else "never"

_FLAT4 = 768  # one king-bucket block (12 planes x 64 squares), same as the 8-bucket table
_NUM_BUCKETS4 = 4


@njit(cache=True, inline="always")
def _bucket_and_mirror(rel_king: int) -> tuple:
    """rel_king is already color-relative (rank-flipped for Black's own
    perspective) but NOT yet mirror-flipped. Returns (bucket, mirror)."""
    king_file = rel_king & 7
    mirror = king_file >= 4
    bucket = (7 - king_file) if mirror else king_file
    return bucket, mirror


@njit(cache=True, inline="always")
def _row4(persp_white: int, sq: int, piece_white: int, pt: int, king_sq: int,
          kbt: np.ndarray) -> int:
    """Feature-vector row for one piece, for one perspective -- 4-bucket,
    horizontally-mirrored variant of ``nnue_accum._row``. ``kbt`` accepted
    for call-signature parity with the 8-bucket module (unused here, same
    convention as the original -- see that module, ``kbt`` is a reserved
    hook, not currently read by either version)."""
    if persp_white:
        rel_sq = sq
        rel_king = king_sq
        same = 1 if piece_white else 0
    else:
        rel_sq = sq ^ 56
        rel_king = king_sq ^ 56
        same = 0 if piece_white else 1
    bucket, mirror = _bucket_and_mirror(rel_king)
    if mirror:
        rel_sq ^= 7
    plane = (0 if same == 1 else 6) + pt
    return bucket * _FLAT4 + plane * 64 + rel_sq


@njit(cache=True)
def accum_full4(bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
                persp_white: int) -> np.ndarray:
    """Full recompute of one perspective's int32 accumulator -- byte-for-byte
    the same structure as ``nnue_accum.accum_full``, over the 4-bucket
    mirrored table."""
    king = lsb(bb[5]) if persp_white else lsb(bb[11])
    h1 = l1b_q.shape[0]
    acc = l1b_q.copy()
    for bi in range(12):
        bits = bb[bi]
        if bits == 0:
            continue
        piece_white = 1 if bi < 6 else 0
        pt = bi % 6
        while bits != 0:
            sq = lsb(bits)
            bits &= bits - np.uint64(1)
            row = _row4(persp_white, sq, piece_white, pt, king, kbt)
            for j in range(h1):
                acc[j] += l1w_q[row, j]
    return acc


@njit(cache=True, inline="always")
def apply_delta4(acc: np.ndarray, l1w_q: np.ndarray, persp_white: int, sq: int,
                 piece_white: int, pt: int, king_sq: int, sign: int,
                 kbt: np.ndarray) -> None:
    """Add (sign=+1) or subtract (sign=-1) one feature row from `acc`.

    Speed campaign notes -- both changes are arithmetically identical to the
    original `acc[j] += sign * l1w_q[row, j]`:

    1. `w = l1w_q[row]` hoists a contiguous 1-D view out of the loop. The
       original indexed the 2-D array per element, which keeps the address
       computation inside the loop body and blocks LLVM's vectorizer.
    2. The sign is branched once, outside the loop, into a pure add loop and
       a pure subtract loop, instead of multiplying every element by +-1.
       Call sites pass literal +-1 and this is inline="always", so LLVM can
       often fold the multiply anyway -- but only when inlining actually
       fires, and it costs nothing to make it unconditional.
    """
    row = _row4(persp_white, sq, piece_white, pt, king_sq, kbt)
    h1 = acc.shape[0]
    if _ACCUM_VECTOR:
        w = l1w_q[row]
        if sign > 0:
            for j in range(h1):
                acc[j] += w[j]
        else:
            for j in range(h1):
                acc[j] -= w[j]
    else:
        for j in range(h1):
            acc[j] += sign * l1w_q[row, j]


@njit(cache=True, inline="always")
def _crosses(us_white: int, old_king_relfile_sq: int, new_king_relfile_sq: int) -> bool:
    """True iff this perspective's own accumulator needs a full refresh --
    bucket number changed OR mirror flag changed (the seam guard, see
    module docstring)."""
    old_bucket, old_mirror = _bucket_and_mirror(old_king_relfile_sq)
    new_bucket, new_mirror = _bucket_and_mirror(new_king_relfile_sq)
    return old_bucket != new_bucket or old_mirror != new_mirror


@njit(cache=True, inline=_ACCUM_INLINE)
def update_accum4(
    bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
    white_acc: np.ndarray, black_acc: np.ndarray, white_king: int, black_king: int,
    frm: int, to: int, moving_code: int, captured_code: int, promo: int, flag: int,
) -> tuple:
    """Same signature/contract as ``nnue_accum.update_accum`` -- drop-in
    compatible with the existing move-loop wiring in jsearch.py via
    nnue_v5_search.py's re-export. Call AFTER make_move has mutated bb."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 2, 3, 4

    old_rel = white_king if us_white else (black_king ^ 56)
    new_rel = to if us_white else (to ^ 56)
    if moving_pt == 5 and _crosses(us_white, old_rel, new_rel):
        if us_white:
            new_acc = accum_full4(bb, l1w_q, l1b_q, kbt, 1)
            for j in range(white_acc.shape[0]):
                white_acc[j] = new_acc[j]
            new_white_king, new_black_king = to, black_king
        else:
            new_acc = accum_full4(bb, l1w_q, l1b_q, kbt, 0)
            for j in range(black_acc.shape[0]):
                black_acc[j] = new_acc[j]
            new_white_king, new_black_king = white_king, to

        other_white = 0 if us_white else 1
        other_king = new_black_king if us_white else new_white_king
        other_acc = black_acc if us_white else white_acc

        if captured_code != 0:
            cap_white = 1 if captured_code <= 6 else 0
            cap_pt = (captured_code - 1) % 6
            apply_delta4(other_acc, l1w_q, other_white, to, cap_white, cap_pt,
                        other_king, -1, kbt)
        apply_delta4(other_acc, l1w_q, other_white, frm, us_white, 5, other_king, -1, kbt)
        apply_delta4(other_acc, l1w_q, other_white, to, us_white, 5, other_king, +1, kbt)

        if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
            rank = 0 if us_white else 7
            if flag == FLAG_CASTLE_K:
                r_from, r_to = rank * 8 + 7, rank * 8 + 5
            else:
                r_from, r_to = rank * 8 + 0, rank * 8 + 3
            apply_delta4(other_acc, l1w_q, other_white, r_from, us_white, 3, other_king, -1, kbt)
            apply_delta4(other_acc, l1w_q, other_white, r_to, us_white, 3, other_king, +1, kbt)

        return new_white_king, new_black_king

    wk, bk = white_king, black_king

    apply_delta4(white_acc, l1w_q, 1, frm, us_white, moving_pt, wk, -1, kbt)
    apply_delta4(black_acc, l1w_q, 0, frm, us_white, moving_pt, bk, -1, kbt)

    if flag == FLAG_EP:
        cap_sq = to - 8 if us_white else to + 8
        cap_white = 0 if us_white else 1
        apply_delta4(white_acc, l1w_q, 1, cap_sq, cap_white, 0, wk, -1, kbt)
        apply_delta4(black_acc, l1w_q, 0, cap_sq, cap_white, 0, bk, -1, kbt)
    elif captured_code != 0:
        cap_white = 1 if captured_code <= 6 else 0
        cap_pt = (captured_code - 1) % 6
        apply_delta4(white_acc, l1w_q, 1, to, cap_white, cap_pt, wk, -1, kbt)
        apply_delta4(black_acc, l1w_q, 0, to, cap_white, cap_pt, bk, -1, kbt)

    arrive_pt = (promo - 1) if promo != 0 else moving_pt
    apply_delta4(white_acc, l1w_q, 1, to, us_white, arrive_pt, wk, +1, kbt)
    apply_delta4(black_acc, l1w_q, 0, to, us_white, arrive_pt, bk, +1, kbt)

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        rank = 0 if us_white else 7
        if flag == FLAG_CASTLE_K:
            r_from, r_to = rank * 8 + 7, rank * 8 + 5
        else:
            r_from, r_to = rank * 8 + 0, rank * 8 + 3
        apply_delta4(white_acc, l1w_q, 1, r_from, us_white, 3, wk, -1, kbt)
        apply_delta4(black_acc, l1w_q, 0, r_from, us_white, 3, bk, -1, kbt)
        apply_delta4(white_acc, l1w_q, 1, r_to, us_white, 3, wk, +1, kbt)
        apply_delta4(black_acc, l1w_q, 0, r_to, us_white, 3, bk, +1, kbt)

    if moving_pt == 5:
        return (to, black_king) if us_white else (white_king, to)
    return white_king, black_king


@njit(cache=True, inline=_ACCUM_INLINE)
def undo_accum4(
    bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
    white_acc: np.ndarray, black_acc: np.ndarray, white_king: int, black_king: int,
    frm: int, to: int, moving_code: int, captured_code: int, promo: int, flag: int,
) -> tuple:
    """Exact inverse of ``update_accum4`` -- same mechanical mirroring
    discipline as ``nnue_accum.undo_accum`` (every +1/-1 swapped, frm/to
    never swapped -- see that module's docstring for why)."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 2, 3, 4

    old_rel = frm if us_white else (frm ^ 56)
    new_rel = to if us_white else (to ^ 56)
    if moving_pt == 5 and _crosses(us_white, old_rel, new_rel):
        if us_white:
            new_acc = accum_full4(bb, l1w_q, l1b_q, kbt, 1)
            for j in range(white_acc.shape[0]):
                white_acc[j] = new_acc[j]
            new_white_king, new_black_king = frm, black_king
        else:
            new_acc = accum_full4(bb, l1w_q, l1b_q, kbt, 0)
            for j in range(black_acc.shape[0]):
                black_acc[j] = new_acc[j]
            new_white_king, new_black_king = white_king, frm

        other_white = 0 if us_white else 1
        other_king = new_black_king if us_white else new_white_king
        other_acc = black_acc if us_white else white_acc

        if captured_code != 0:
            cap_white = 1 if captured_code <= 6 else 0
            cap_pt = (captured_code - 1) % 6
            apply_delta4(other_acc, l1w_q, other_white, to, cap_white, cap_pt,
                        other_king, +1, kbt)
        apply_delta4(other_acc, l1w_q, other_white, frm, us_white, 5, other_king, +1, kbt)
        apply_delta4(other_acc, l1w_q, other_white, to, us_white, 5, other_king, -1, kbt)

        if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
            rank = 0 if us_white else 7
            if flag == FLAG_CASTLE_K:
                r_from, r_to = rank * 8 + 7, rank * 8 + 5
            else:
                r_from, r_to = rank * 8 + 0, rank * 8 + 3
            apply_delta4(other_acc, l1w_q, other_white, r_from, us_white, 3, other_king, +1, kbt)
            apply_delta4(other_acc, l1w_q, other_white, r_to, us_white, 3, other_king, -1, kbt)

        return new_white_king, new_black_king

    wk, bk = white_king, black_king

    if flag == FLAG_EP:
        cap_sq = to - 8 if us_white else to + 8
        cap_white = 0 if us_white else 1
        apply_delta4(white_acc, l1w_q, 1, cap_sq, cap_white, 0, wk, +1, kbt)
        apply_delta4(black_acc, l1w_q, 0, cap_sq, cap_white, 0, bk, +1, kbt)
    elif captured_code != 0:
        cap_white = 1 if captured_code <= 6 else 0
        cap_pt = (captured_code - 1) % 6
        apply_delta4(white_acc, l1w_q, 1, to, cap_white, cap_pt, wk, +1, kbt)
        apply_delta4(black_acc, l1w_q, 0, to, cap_white, cap_pt, bk, +1, kbt)

    arrive_pt = (promo - 1) if promo != 0 else moving_pt
    apply_delta4(white_acc, l1w_q, 1, to, us_white, arrive_pt, wk, -1, kbt)
    apply_delta4(black_acc, l1w_q, 0, to, us_white, arrive_pt, bk, -1, kbt)

    apply_delta4(white_acc, l1w_q, 1, frm, us_white, moving_pt, wk, +1, kbt)
    apply_delta4(black_acc, l1w_q, 0, frm, us_white, moving_pt, bk, +1, kbt)

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        rank = 0 if us_white else 7
        if flag == FLAG_CASTLE_K:
            r_from, r_to = rank * 8 + 7, rank * 8 + 5
        else:
            r_from, r_to = rank * 8 + 0, rank * 8 + 3
        apply_delta4(white_acc, l1w_q, 1, r_to, us_white, 3, wk, -1, kbt)
        apply_delta4(black_acc, l1w_q, 0, r_to, us_white, 3, bk, -1, kbt)
        apply_delta4(white_acc, l1w_q, 1, r_from, us_white, 3, wk, +1, kbt)
        apply_delta4(black_acc, l1w_q, 0, r_from, us_white, 3, bk, +1, kbt)

    if moving_pt == 5:
        return (frm, black_king) if us_white else (white_king, frm)
    return white_king, black_king


@njit(cache=True, inline=_ACCUM_INLINE)
def update_accum_quiet4(l1w_q: np.ndarray, white_acc: np.ndarray, black_acc: np.ndarray,
                        white_king: int, black_king: int,
                        frm: int, to: int, moving_code: int, kbt: np.ndarray) -> None:
    """Quiet-move fast path, 4-bucket-mirrored variant -- same contract as
    ``nnue_v5_accum.update_accum_quiet`` (caller gates: non-king, non-
    capture, non-promo, non-EP, non-castle)."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    apply_delta4(white_acc, l1w_q, 1, frm, us_white, moving_pt, white_king, -1, kbt)
    apply_delta4(black_acc, l1w_q, 0, frm, us_white, moving_pt, black_king, -1, kbt)
    apply_delta4(white_acc, l1w_q, 1, to, us_white, moving_pt, white_king, +1, kbt)
    apply_delta4(black_acc, l1w_q, 0, to, us_white, moving_pt, black_king, +1, kbt)


@njit(cache=True, inline=_ACCUM_INLINE)
def undo_accum_quiet4(l1w_q: np.ndarray, white_acc: np.ndarray, black_acc: np.ndarray,
                      white_king: int, black_king: int,
                      frm: int, to: int, moving_code: int, kbt: np.ndarray) -> None:
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    apply_delta4(white_acc, l1w_q, 1, to, us_white, moving_pt, white_king, -1, kbt)
    apply_delta4(black_acc, l1w_q, 0, to, us_white, moving_pt, black_king, -1, kbt)
    apply_delta4(white_acc, l1w_q, 1, frm, us_white, moving_pt, white_king, +1, kbt)
    apply_delta4(black_acc, l1w_q, 0, frm, us_white, moving_pt, black_king, +1, kbt)
