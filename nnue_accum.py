"""Clean-room incremental accumulator for the dual-perspective v4.1 network.

Feature rows are int16/Q4096 and accumulator vectors are int32. The two fixed
perspectives are maintained independently on a preallocated per-ply search
stack; only a king-bucket crossing refreshes the moving king's perspective.

Two accumulators (white-perspective, black-perspective), each
``int32[h1] = l1_bias_q + sum over active features of l1_w_q[feature_row]``.
Non-king moves update by ``acc += l1_w_q[row]`` / ``acc -= l1_w_q[row]``; a move
of the perspective's OWN king (or castling) changes every one of that
perspective's feature indices at once -> full refresh for that side only
(``accum_full``), the other side stays incremental.

Why int32/QSCALE and not float32 (matching ``_forward`` bit-for-bit): float
accumulation is not associative -- the incremental path sums deltas in
root-to-leaf move order, ``accum_full`` sums pieces in bitboard-index order,
and those two orderings can differ by ~1e-4 x magnitude in float32. After
x gain (~5) and round(), that occasionally flips +/-1cp and can change move
selection -- it fails this project's bit-exact discipline (perft, Zobrist,
movegen, the original nnue_eval_v3 forward-pass gate are all held to exact
match, not epsilon-close). Int32 addition is exactly associative: no
precision loss, no ordering sensitivity, byte-match-forever is achievable
and is exactly what ``nnue/accumulator_v3.py``'s existing (chess.Board-based)
gate already demonstrates. Cost: the served eval shifts by a small,
bounded, one-time quantisation delta from the current float ``_forward``
path -- Stage 4 (probe + self-play) gets a fresh spot-check after wiring
this in, which is a known, bounded cost, not an open-ended drift risk.

``eval_from_accum`` is the tail of ``_forward`` (line 78 on) with the L1
accumulator dequantised (``/ QSCALE``) at the L1->L2 boundary, then float
from there on -- same split as ``accumulator_v3.QuantModelKB.eval_from_accum``.

Weights are passed in (not module globals) so every function here stays
``cache=True``-eligible -- callers pass the arrays from ``quantize_l1()``.

See ``docs/NNUE_ACCUMULATOR_DESIGN_2026-09-07.md``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from bitboard import lsb

_FLAT = 768  # one king-bucket block
QSCALE = 4096


def quantize_l1(l1w: np.ndarray, l1b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float feature rows to int16 and bias to int32 once at load."""
    l1w_q = np.round(l1w.astype(np.float64) * QSCALE).astype(np.int16)
    l1b_q = np.round(l1b.astype(np.float64) * QSCALE).astype(np.int32)
    return l1w_q, l1b_q


@njit(cache=True, inline="always")
def _row(persp_white: int, sq: int, piece_white: int, pt: int, king_sq: int,
         kbt: np.ndarray) -> int:
    """Feature-vector row for one piece, for one perspective. Mirrors the inline
    index in ``nnue_eval_v3._forward`` (rel_sq / rel_king / same / plane /
    bucket). Unchanged from the float version -- pure index math, gate 1
    already covers this exactly (98,304 combos, PASS)."""
    if persp_white:
        rel_sq = sq
        rel_king = king_sq
        same = 1 if piece_white else 0
    else:
        rel_sq = sq ^ 56
        rel_king = king_sq ^ 56
        same = 0 if piece_white else 1
    plane = (0 if same == 1 else 6) + pt
    bucket = (rel_king // 16) * 2 + (1 if (rel_king & 7) >= 4 else 0)
    return bucket * _FLAT + plane * 64 + rel_sq


@njit(cache=True)
def accum_full(bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
               persp_white: int) -> np.ndarray:
    """Full recompute of one perspective's int32 accumulator from the
    bitboards. bb[0..5] = white P N B R Q K, bb[6..11] = black P N B R Q K."""
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
            row = _row(persp_white, sq, piece_white, pt, king, kbt)
            for j in range(h1):
                acc[j] += l1w_q[row, j]
    return acc


@njit(cache=True, inline="always")
def apply_delta(acc: np.ndarray, l1w_q: np.ndarray, persp_white: int, sq: int,
                piece_white: int, pt: int, king_sq: int, sign: int,
                kbt: np.ndarray) -> None:
    """acc += sign * l1_w_q[row] for one piece appearing (+1) / leaving (-1).
    Int32 addition -- exact, order-independent, no drift over a long game."""
    row = _row(persp_white, sq, piece_white, pt, king_sq, kbt)
    h1 = acc.shape[0]
    for j in range(h1):
        acc[j] += sign * l1w_q[row, j]


@njit(cache=True, fastmath=True)
def eval_from_accum(white_q: np.ndarray, black_q: np.ndarray, stm: int,
                    hw: np.ndarray, hb: np.ndarray, ow: np.ndarray,
                    ob: np.ndarray, multiplier: np.float32) -> int:
    """Tail of ``_forward`` -- dequantises the int32 L1 accumulator
    (``/ QSCALE``) then proceeds in float32, matching ``_forward``'s own
    numerical path exactly (its weight arrays are all float32) -- not
    float64, which was matching the stale ``accumulator_v3.QuantModelKB.
    eval_from_accum`` prototype instead and cost ~2x on top of being wrong
    (half the SIMD width, plus a per-call float64 cast of ``l2b``/every
    element). Does not mutate ``acc_q``.

    ``fastmath=True`` matches ``_forward``'s decorator: without it LLVM cannot
    vectorise the L2 reduction (strict IEEE FP add is not associative) or the
    QSCALE-division dequant loop, and this tail runs ~9x slower (5.5 vs
    0.58 us/call in-JIT) -- enough to sink the incremental scheme below
    full-recompute. Since ``_forward`` is already fastmath, sharing the flag
    makes the two eval paths *more* numerically consistent, not less."""
    h1 = white_q.shape[0]
    # Claude-opt: `stm` picks which perspective feeds the "own"/"other" half
    # of the concatenated 512-wide hidden-layer input; Codex's version
    # branched on it 256*32 = 8192 times, once per (j, k) pair, inside the
    # hot double loop, even though the answer cannot change within one call.
    # Deciding it once here, while building the dequantised/clipped rows,
    # removes that branch from the O(h1*h2) inner loop entirely -- same
    # values, same order of the two dot-product terms, so the float32
    # fastmath rounding is identical to the original, not just "close".
    first = np.empty(h1, np.float32)
    second = np.empty(h1, np.float32)
    if stm == 0:
        src_first, src_second = white_q, black_q
    else:
        src_first, src_second = black_q, white_q
    for j in range(h1):
        fv = np.float32(src_first[j]) / np.float32(QSCALE)
        sv = np.float32(src_second[j]) / np.float32(QSCALE)
        first[j] = min(np.float32(2.0), max(np.float32(0.0), fv))
        second[j] = min(np.float32(2.0), max(np.float32(0.0), sv))
    h2 = hb.shape[0]
    # Claude-opt: `hid[k]` was previously materialised into its own array
    # (one np.empty-equivalent alloc via `hb.copy()`) and consumed exactly
    # once, immediately after, by the output dot product below -- a single
    # per-unit scalar folded straight into the output accumulator removes
    # that array (and its allocation) with no reuse lost, unlike the
    # 256-wide `first`/`second` rows above (each read 32 times, so THOSE
    # stay materialised -- see docs/NNUE_ACCUMULATOR_CLAUDE_OPT_2026-09-08.md
    # for the measurement that ruled out fusing that wider loop too).
    out = ob[0]
    for k in range(h2):
        # accumulate the products first, add the bias last -- same order as
        # the original `s = 0.0; ...; hid[k] = clip(hb[k] + s)`, not
        # `hb[k] + products` reassociated, so float32 rounding is identical
        # term-for-term (fastmath still permits reordering, but there is no
        # reason to invite a difference when preserving order is free).
        s = np.float32(0.0)
        for j in range(h1):
            s += hw[k, j] * first[j] + hw[k, h1 + j] * second[j]
        hidk = min(np.float32(2.0), max(np.float32(0.0), hb[k] + s))
        out += ow[0, k] * hidk
    return int(round(out * multiplier))


@njit(cache=True, inline="always")
def update_accum(
    bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
    white_acc: np.ndarray, black_acc: np.ndarray, white_king: int, black_king: int,
    frm: int, to: int, moving_code: int, captured_code: int, promo: int, flag: int,
) -> tuple[int, int]:
    """Update both accumulators for one move, in place. Call *after*
    ``make_move`` has already mutated ``bb``/``mb`` -- needs the post-move
    ``bb`` for the king-refresh full-recompute case.

    ``moving_code``/``captured_code`` are 1..12 piece codes (0 = no capture),
    same convention as ``bitboard.mb``: capture ``moving_code = mb[frm]``
    *before* calling ``make_move``, and pass ``captured_code = undo & 0xF``
    from its return value -- ``make_move`` itself is untouched. ``frm``/``to``/
    ``promo``/``flag`` decode straight from the move int, same as make_move.

    Returns the (possibly updated) ``(white_king, black_king)`` squares --
    unchanged for a non-king move, one of them replaced for a king move.
    Mirrors ``nnue/accumulator_v3.py``'s ``AccumulatorsKB.push`` exactly
    (that version is the correctness oracle -- see verify_accum_incr.py)."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6  # 0..5, P..K -- 0-based, matches _row's pt
    FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 2, 3, 4

    old_rel = white_king if us_white else (black_king ^ 56)
    new_rel = to if us_white else (to ^ 56)
    old_bucket = (old_rel // 16) * 2 + (1 if (old_rel & 7) >= 4 else 0)
    new_bucket = (new_rel // 16) * 2 + (1 if (new_rel & 7) >= 4 else 0)
    if moving_pt == 5 and old_bucket != new_bucket:
        if us_white:
            new_acc = accum_full(bb, l1w_q, l1b_q, kbt, 1)
            for j in range(white_acc.shape[0]):
                white_acc[j] = new_acc[j]
            new_white_king, new_black_king = to, black_king
        else:
            new_acc = accum_full(bb, l1w_q, l1b_q, kbt, 0)
            for j in range(black_acc.shape[0]):
                black_acc[j] = new_acc[j]
            new_white_king, new_black_king = white_king, to

        # other perspective: still incremental -- the moved king is just a
        # piece-plane feature for them, their own king square is unchanged.
        other_white = 0 if us_white else 1
        other_king = new_black_king if us_white else new_white_king
        other_acc = black_acc if us_white else white_acc

        if captured_code != 0:
            cap_white = 1 if captured_code <= 6 else 0
            cap_pt = (captured_code - 1) % 6
            apply_delta(other_acc, l1w_q, other_white, to, cap_white, cap_pt,
                        other_king, -1, kbt)
        apply_delta(other_acc, l1w_q, other_white, frm, us_white, 5, other_king, -1, kbt)
        apply_delta(other_acc, l1w_q, other_white, to, us_white, 5, other_king, +1, kbt)

        if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
            rank = 0 if us_white else 7
            if flag == FLAG_CASTLE_K:
                r_from, r_to = rank * 8 + 7, rank * 8 + 5
            else:
                r_from, r_to = rank * 8 + 0, rank * 8 + 3
            apply_delta(other_acc, l1w_q, other_white, r_from, us_white, 3, other_king, -1, kbt)
            apply_delta(other_acc, l1w_q, other_white, r_to, us_white, 3, other_king, +1, kbt)

        return new_white_king, new_black_king

    # non-king move: both perspectives stay incremental, king squares unchanged
    wk, bk = white_king, black_king

    apply_delta(white_acc, l1w_q, 1, frm, us_white, moving_pt, wk, -1, kbt)
    apply_delta(black_acc, l1w_q, 0, frm, us_white, moving_pt, bk, -1, kbt)

    if flag == FLAG_EP:
        cap_sq = to - 8 if us_white else to + 8
        cap_white = 0 if us_white else 1
        apply_delta(white_acc, l1w_q, 1, cap_sq, cap_white, 0, wk, -1, kbt)
        apply_delta(black_acc, l1w_q, 0, cap_sq, cap_white, 0, bk, -1, kbt)
    elif captured_code != 0:
        cap_white = 1 if captured_code <= 6 else 0
        cap_pt = (captured_code - 1) % 6
        apply_delta(white_acc, l1w_q, 1, to, cap_white, cap_pt, wk, -1, kbt)
        apply_delta(black_acc, l1w_q, 0, to, cap_white, cap_pt, bk, -1, kbt)

    arrive_pt = (promo - 1) if promo != 0 else moving_pt  # promo: 2..5 = N,B,R,Q (bitboard.py encoding)
    apply_delta(white_acc, l1w_q, 1, to, us_white, arrive_pt, wk, +1, kbt)
    apply_delta(black_acc, l1w_q, 0, to, us_white, arrive_pt, bk, +1, kbt)

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        rank = 0 if us_white else 7
        if flag == FLAG_CASTLE_K:
            r_from, r_to = rank * 8 + 7, rank * 8 + 5
        else:
            r_from, r_to = rank * 8 + 0, rank * 8 + 3
        apply_delta(white_acc, l1w_q, 1, r_from, us_white, 3, wk, -1, kbt)
        apply_delta(black_acc, l1w_q, 0, r_from, us_white, 3, bk, -1, kbt)
        apply_delta(white_acc, l1w_q, 1, r_to, us_white, 3, wk, +1, kbt)
        apply_delta(black_acc, l1w_q, 0, r_to, us_white, 3, bk, +1, kbt)

    if moving_pt == 5:
        return (to, black_king) if us_white else (white_king, to)
    return white_king, black_king


@njit(cache=True, inline="always")
def undo_accum(
    bb: np.ndarray, l1w_q: np.ndarray, l1b_q: np.ndarray, kbt: np.ndarray,
    white_acc: np.ndarray, black_acc: np.ndarray, white_king: int, black_king: int,
    frm: int, to: int, moving_code: int, captured_code: int, promo: int, flag: int,
) -> tuple[int, int]:
    """Exact inverse of ``update_accum`` for one move, in place. Call *after*
    ``unmake_move`` has already restored ``bb`` -- needs the pre-move ``bb``
    for the king-refresh full-recompute case, same convention as
    ``update_accum``.

    This is Claude-opt's clean-room single-shared-accumulator scheme: instead
    of copying a fresh child row per node (Codex's per-ply ``[ply][2][H]``
    stack, see ``docs/NNUE_ACCUMULATOR_CLAUDE_OPT_2026-09-08.md``), the search
    keeps ONE ``white_acc``/``black_acc`` pair and mutates it in place across
    the whole DFS traversal -- ``update_accum`` on the way down, this function
    on the way back up. Every row toggle ``update_accum`` applies is an
    independent ``acc[j] += sign * row[j]``; since int32 addition is exact and
    commutative, undoing the move is just re-applying the identical set of
    rows with every sign flipped, in any order -- so this function is a
    line-for-line mirror of ``update_accum`` with every literal ``+1``/``-1``
    swapped. It does not "swap frm/to and replay": a naive replay-in-reverse
    breaks promotions (the piece arriving at ``frm`` after undo is the
    ``moving_pt`` pre-promotion piece, never the promoted ``arrive_pt`` piece,
    so a swapped-args call would restore the wrong piece type). ``frm``/``to``
    are always the ORIGINAL move's squares here, never swapped.

    The king-bucket-crossing test itself is direction-symmetric (whether a
    move spans two buckets does not depend on which way it is being played),
    so it is computed from ``frm``/``to`` directly rather than from the
    passed-in king squares -- this also holds for ``update_accum``'s forward
    case (a genuine king move's pre-move square is always exactly ``frm``),
    so the two functions agree on which branch a given move takes.

    Returns the pre-move ``(white_king, black_king)`` -- unchanged for a
    non-king move, one of them reverted to ``frm`` for a king move."""
    us_white = 1 if moving_code <= 6 else 0
    moving_pt = (moving_code - 1) % 6
    FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 2, 3, 4

    old_rel = frm if us_white else (frm ^ 56)
    new_rel = to if us_white else (to ^ 56)
    old_bucket = (old_rel // 16) * 2 + (1 if (old_rel & 7) >= 4 else 0)
    new_bucket = (new_rel // 16) * 2 + (1 if (new_rel & 7) >= 4 else 0)
    if moving_pt == 5 and old_bucket != new_bucket:
        if us_white:
            new_acc = accum_full(bb, l1w_q, l1b_q, kbt, 1)
            for j in range(white_acc.shape[0]):
                white_acc[j] = new_acc[j]
            new_white_king, new_black_king = frm, black_king
        else:
            new_acc = accum_full(bb, l1w_q, l1b_q, kbt, 0)
            for j in range(black_acc.shape[0]):
                black_acc[j] = new_acc[j]
            new_white_king, new_black_king = white_king, frm

        other_white = 0 if us_white else 1
        other_king = new_black_king if us_white else new_white_king
        other_acc = black_acc if us_white else white_acc

        if captured_code != 0:
            cap_white = 1 if captured_code <= 6 else 0
            cap_pt = (captured_code - 1) % 6
            apply_delta(other_acc, l1w_q, other_white, to, cap_white, cap_pt,
                        other_king, +1, kbt)
        apply_delta(other_acc, l1w_q, other_white, frm, us_white, 5, other_king, +1, kbt)
        apply_delta(other_acc, l1w_q, other_white, to, us_white, 5, other_king, -1, kbt)

        if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
            rank = 0 if us_white else 7
            if flag == FLAG_CASTLE_K:
                r_from, r_to = rank * 8 + 7, rank * 8 + 5
            else:
                r_from, r_to = rank * 8 + 0, rank * 8 + 3
            apply_delta(other_acc, l1w_q, other_white, r_from, us_white, 3, other_king, +1, kbt)
            apply_delta(other_acc, l1w_q, other_white, r_to, us_white, 3, other_king, -1, kbt)

        return new_white_king, new_black_king

    # non-king move (or same-bucket king move): both perspectives incremental
    wk, bk = white_king, black_king

    if flag == FLAG_EP:
        cap_sq = to - 8 if us_white else to + 8
        cap_white = 0 if us_white else 1
        apply_delta(white_acc, l1w_q, 1, cap_sq, cap_white, 0, wk, +1, kbt)
        apply_delta(black_acc, l1w_q, 0, cap_sq, cap_white, 0, bk, +1, kbt)
    elif captured_code != 0:
        cap_white = 1 if captured_code <= 6 else 0
        cap_pt = (captured_code - 1) % 6
        apply_delta(white_acc, l1w_q, 1, to, cap_white, cap_pt, wk, +1, kbt)
        apply_delta(black_acc, l1w_q, 0, to, cap_white, cap_pt, bk, +1, kbt)

    arrive_pt = (promo - 1) if promo != 0 else moving_pt
    apply_delta(white_acc, l1w_q, 1, to, us_white, arrive_pt, wk, -1, kbt)
    apply_delta(black_acc, l1w_q, 0, to, us_white, arrive_pt, bk, -1, kbt)

    apply_delta(white_acc, l1w_q, 1, frm, us_white, moving_pt, wk, +1, kbt)
    apply_delta(black_acc, l1w_q, 0, frm, us_white, moving_pt, bk, +1, kbt)

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        rank = 0 if us_white else 7
        if flag == FLAG_CASTLE_K:
            r_from, r_to = rank * 8 + 7, rank * 8 + 5
        else:
            r_from, r_to = rank * 8 + 0, rank * 8 + 3
        apply_delta(white_acc, l1w_q, 1, r_to, us_white, 3, wk, -1, kbt)
        apply_delta(black_acc, l1w_q, 0, r_to, us_white, 3, bk, -1, kbt)
        apply_delta(white_acc, l1w_q, 1, r_from, us_white, 3, wk, +1, kbt)
        apply_delta(black_acc, l1w_q, 0, r_from, us_white, 3, bk, +1, kbt)

    if moving_pt == 5:
        return (frm, black_king) if us_white else (white_king, frm)
    return white_king, black_king
