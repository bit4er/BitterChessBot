"""``evaluate.py`` recomputed from a ``bitboard.py`` state array, jitted.

Same terms, same weights, same operator order as ``evaluate.evaluate`` -- only
the data source changes (packed uint64 bitboards + mailbox instead of a
python-chess ``Board``) and the whole scan runs in one numba call instead of
~30 piece-loop iterations of CPython bytecode plus python-chess method calls.

The weight scalars are read live from ``evaluate`` on every call (cheap global
loads) and handed to the jitted core; the fused material+PST tables are packed
once at import from ``evaluate``'s tables. If ``evaluate.set_weights`` is ever
used together with this path, call ``repack()`` afterwards -- the trainer, the
only ``set_weights`` caller, uses the python-chess path and never touches this.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

import evaluate as _ev
from bbwrap import BBoard
from bitboard import (
    KING_ATK,
    KNIGHT_ATK,
    U1,
    bishop_attacks,
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
)

_U0 = np.uint64(0)

# file masks a..h, as uint64
_FILES = np.array(
    [np.uint64(0x0101010101010101) << np.uint64(f) for f in range(8)], dtype=np.uint64
)
# pawn-structure masks (static; packed once from evaluate)
_ADJ_FILES = np.array(_ev._ADJ_FILE_MASK, dtype=np.uint64)
_WSPAN = np.array(_ev._WHITE_PASSED_SPAN, dtype=np.uint64)
_BSPAN = np.array(_ev._BLACK_PASSED_SPAN, dtype=np.uint64)
_WDEF = np.array(_ev._WHITE_PAWN_DEFENDERS, dtype=np.uint64)
_BDEF = np.array(_ev._BLACK_PAWN_DEFENDERS, dtype=np.uint64)
_WBACK = np.array(_ev._WHITE_BACKWARD_SPAN, dtype=np.uint64)
_BBACK = np.array(_ev._BLACK_BACKWARD_SPAN, dtype=np.uint64)

# fused material+PST, rows indexed by piece type 1..5 (pawn..queen)
_WPST = np.zeros((6, 64), np.int64)
_BPST = np.zeros((6, 64), np.int64)
_KMG = np.zeros(64, np.int64)
_KEG = np.zeros(64, np.int64)
# [passed_r0..r7, protected, doubled, isolated, rook_open, rook_semi]
_PS = np.zeros(13, np.int64)
# king-attack weights: [knight, bishop, rook, queen, file_open, file_semi, divisor]
_KA = np.zeros(7, np.int64)
_KING_ATK = np.array(KING_ATK, dtype=np.uint64)
# scalar weights: [mobility_weight, bishop_pair, tempo, shield_missing, open_file]
_SCALARS = np.zeros(5, np.int64)
# endgame terms: [connected, outside, king_march, rook_behind, backward,
# islands, ON]  -- last element is the AICHESS_EG_TERMS flag (0/1)
_EG = np.zeros(7, np.int64)


def repack() -> None:
    """Refresh the packed tables from ``evaluate``'s current tables. Call after
    ``evaluate.set_weights`` if this backend is in use (e.g. the Texel tuner)."""
    for pt in (1, 2, 3, 4, 5):
        _WPST[pt] = np.array(_ev._WHITE_PST[pt], np.int64)
        _BPST[pt] = np.array(_ev._BLACK_PST[pt], np.int64)
    _KMG[:] = np.array(_ev._KING_MG_SQ, np.int64)
    _KEG[:] = np.array(_ev._KING_EG_SQ, np.int64)
    _PS[:] = [
        *_ev.PASSED_PAWN,
        _ev.PASSED_PROTECTED,
        _ev.DOUBLED_PAWN,
        _ev.ISOLATED_PAWN,
        _ev.ROOK_OPEN_FILE,
        _ev.ROOK_SEMI_OPEN_FILE,
    ]
    _KA[:] = [*_ev.KING_ATTACK, _ev.KING_ATTACK_DIVISOR]
    _SCALARS[:] = [
        _ev.MOBILITY_WEIGHT,
        _ev.BISHOP_PAIR,
        _ev.TEMPO,
        _ev.SHIELD_MISSING,
        _ev.OPEN_FILE_NEAR_KING,
    ]
    _EG[:] = [
        _ev.PASSED_CONNECTED,
        _ev.PASSED_OUTSIDE,
        _ev.PASSED_KING_MARCH,
        _ev.ROOK_BEHIND_PASSER,
        _ev.BACKWARD_PAWN,
        _ev.PAWN_ISLANDS,
        1 if _ev._EG_TERMS_ON else 0,
    ]


repack()

# all the eval tables in one tuple, so jsearch can pass the whole evaluation to
# its jitted negamax as a single argument. repack() writes every member in
# place, so this stays current.
EVAL_BUNDLE = (
    _WPST, _BPST, _KMG, _KEG, _FILES, _ADJ_FILES,
    _WSPAN, _BSPAN, _WDEF, _BDEF, _PS, _KING_ATK, _KA, _SCALARS,
    _EG, _WBACK, _BBACK,
)


@njit(cache=True, inline="always")
def eval_state(bb: np.ndarray, ev: tuple) -> int:
    """Same as evaluate_bb, but takes the EVAL_BUNDLE tuple instead of a BBoard.
    For the jitted search."""
    return _score(
        bb, ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8], ev[9],
        ev[10], ev[11], ev[12], ev[13][0], ev[13][1], ev[13][2], ev[13][3], ev[13][4],
        ev[14], ev[15], ev[16],
    )


@njit(cache=True)
def _round_half_even(x: float) -> np.int64:
    """CPython ``round(float)`` semantics: nearest, ties to even, on the double."""
    fl = math.floor(x)
    frac = x - fl
    if frac < 0.5:
        return np.int64(fl)
    if frac > 0.5:
        return np.int64(fl) + np.int64(1)
    lo = np.int64(fl)
    return lo if (lo & np.int64(1)) == np.int64(0) else lo + np.int64(1)


@njit(cache=True)
def _king_safety(
    king_sq: int,
    is_white: bool,
    own_pawns: np.uint64,
    enemy_pawns: np.uint64,
    e_knight: np.uint64,
    e_bishop: np.uint64,
    e_rook: np.uint64,
    e_queen: np.uint64,
    occ: np.uint64,
    files: np.ndarray,
    king_atk: np.ndarray,
    shield_missing: int,
    open_file: int,
    ka: np.ndarray,
) -> int:
    king_file = king_sq & 7
    lo = king_file - 1 if king_file - 1 > 0 else 0
    hi = king_file + 1 if king_file + 1 < 7 else 7
    penalty = 0
    for fi in range(lo, hi + 1):
        fm = files[fi]
        if not (own_pawns & fm):
            penalty -= shield_missing
            if not (enemy_pawns & fm):
                penalty -= open_file

    if ka[0] == 0 and ka[1] == 0 and ka[2] == 0 and ka[3] == 0 and ka[4] == 0 and ka[5] == 0:
        return penalty  # term disabled (AICHESS_KING_ATTACK=0)

    zone = king_atk[king_sq] | (U1 << np.uint64(king_sq))
    zone = (
        (zone | (zone << np.uint64(8))) if is_white else (zone | (zone >> np.uint64(8)))
    )

    danger = 0
    b = e_knight
    while b:
        s = lsb(b)
        b &= b - U1
        hit = popcount(KNIGHT_ATK[s] & zone)
        if hit:
            danger += ka[0] * (hit if hit < 3 else 3)
    b = e_bishop
    while b:
        s = lsb(b)
        b &= b - U1
        hit = popcount(bishop_attacks(s, occ) & zone)
        if hit:
            danger += ka[1] * (hit if hit < 3 else 3)
    b = e_rook
    while b:
        s = lsb(b)
        b &= b - U1
        hit = popcount(rook_attacks(s, occ) & zone)
        if hit:
            danger += ka[2] * (hit if hit < 3 else 3)
    b = e_queen
    while b:
        s = lsb(b)
        b &= b - U1
        hit = popcount(queen_attacks(s, occ) & zone)
        if hit:
            danger += ka[3] * (hit if hit < 3 else 3)

    heavy = e_rook | e_queen
    for fi in range(lo, hi + 1):
        fm = files[fi]
        if (heavy & fm) and not (enemy_pawns & fm):
            danger += ka[4] if not (own_pawns & fm) else ka[5]

    penalty -= danger * danger // ka[6]
    return penalty


@njit(cache=True)
def _structure(
    white_pawns: np.uint64,
    black_pawns: np.uint64,
    white_rooks: np.uint64,
    black_rooks: np.uint64,
    files: np.ndarray,
    adj_files: np.ndarray,
    wspan: np.ndarray,
    bspan: np.ndarray,
    wdef: np.ndarray,
    bdef: np.ndarray,
    ps: np.ndarray,
) -> int:
    all_pawns = white_pawns | black_pawns
    score = 0
    protected = ps[8]
    doubled = ps[9]
    isolated = ps[10]
    rook_open = ps[11]
    rook_semi = ps[12]

    p = white_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        if not (adj_files[s & 7] & white_pawns):
            score -= isolated
        if not (wspan[s] & black_pawns):
            score += ps[s >> 3]
            if wdef[s] & white_pawns:
                score += protected
    p = black_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        if not (adj_files[s & 7] & black_pawns):
            score += isolated
        if not (bspan[s] & white_pawns):
            score -= ps[7 - (s >> 3)]
            if bdef[s] & black_pawns:
                score -= protected

    for fi in range(8):
        wn = popcount(white_pawns & files[fi])
        if wn > 1:
            score -= doubled * (wn - 1)
        bn = popcount(black_pawns & files[fi])
        if bn > 1:
            score += doubled * (bn - 1)

    r = white_rooks
    while r:
        fm = files[lsb(r) & 7]
        r &= r - U1
        if not (fm & all_pawns):
            score += rook_open
        elif not (fm & white_pawns):
            score += rook_semi
    r = black_rooks
    while r:
        fm = files[lsb(r) & 7]
        r &= r - U1
        if not (fm & all_pawns):
            score -= rook_open
        elif not (fm & black_pawns):
            score -= rook_semi
    return score


@njit(cache=True)
def _cheb(a: int, b: int) -> int:
    fa = a & 7
    ra = a >> 3
    fb = b & 7
    rb = b >> 3
    df = fa - fb if fa >= fb else fb - fa
    dr = ra - rb if ra >= rb else rb - ra
    return df if df >= dr else dr


@njit(cache=True)
def _pawn_islands_bb(pawns: np.uint64, files: np.ndarray) -> int:
    islands = 0
    prev = False
    for fi in range(8):
        cur = (pawns & files[fi]) != np.uint64(0)
        if cur and not prev:
            islands += 1
        prev = cur
    return islands


@njit(cache=True)
def _endgame(
    white_pawns: np.uint64,
    black_pawns: np.uint64,
    wk: int,
    bk: int,
    white_rooks: np.uint64,
    black_rooks: np.uint64,
    files: np.ndarray,
    adj_files: np.ndarray,
    wspan: np.ndarray,
    bspan: np.ndarray,
    wdef: np.ndarray,
    bdef: np.ndarray,
    wback: np.ndarray,
    bback: np.ndarray,
    eg: np.ndarray,
) -> int:
    """Mirror of ``evaluate._endgame``: White-minus-Black endgame positional
    score in centipawns, before the max(0, 1 - phase/16) scale in ``_score``."""
    connected = eg[0]
    outside = eg[1]
    king_march = eg[2]
    rook_behind = eg[3]
    backward = eg[4]
    islands_w = eg[5]

    score = 0

    white_passed = np.uint64(0)
    p = white_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        if not (wspan[s] & black_pawns):
            white_passed |= U1 << np.uint64(s)
    black_passed = np.uint64(0)
    p = black_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        if not (bspan[s] & white_pawns):
            black_passed |= U1 << np.uint64(s)

    wk_file = wk & 7
    bk_file = bk & 7

    p = white_passed
    while p:
        s = lsb(p)
        p &= p - U1
        fi = s & 7
        promo = 56 + fi
        if white_passed & adj_files[fi]:
            score += connected
        gap = bk_file - fi if bk_file >= fi else fi - bk_file
        if (fi <= 1 or fi >= 6) and gap >= 4:
            score += outside
        score += king_march * (_cheb(bk, promo) - _cheb(wk, promo))
        behind = files[fi] & ((U1 << np.uint64(s)) - U1)
        if white_rooks & behind:
            score += rook_behind
        if black_rooks & behind:
            score -= rook_behind

    p = black_passed
    while p:
        s = lsb(p)
        p &= p - U1
        fi = s & 7
        promo = fi
        if black_passed & adj_files[fi]:
            score -= connected
        gap = wk_file - fi if wk_file >= fi else fi - wk_file
        if (fi <= 1 or fi >= 6) and gap >= 4:
            score -= outside
        score -= king_march * (_cheb(wk, promo) - _cheb(bk, promo))
        behind = files[fi] & ~((U1 << np.uint64(s + 1)) - U1)
        if black_rooks & behind:
            score -= rook_behind
        if white_rooks & behind:
            score += rook_behind

    p = white_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        stop = s + 8
        if (
            stop < 64
            and (black_pawns & bdef[stop])
            and not (white_pawns & wback[s])
            and not (files[s & 7] & black_pawns)
        ):
            score -= backward
    p = black_pawns
    while p:
        s = lsb(p)
        p &= p - U1
        stop = s - 8
        if (
            stop >= 0
            and (white_pawns & wdef[stop])
            and not (black_pawns & bback[s])
            and not (files[s & 7] & white_pawns)
        ):
            score += backward

    score -= islands_w * (_pawn_islands_bb(white_pawns, files) - 1)
    score += islands_w * (_pawn_islands_bb(black_pawns, files) - 1)

    return score


@njit(cache=True)
def _score(
    bb: np.ndarray,
    wpst: np.ndarray,
    bpst: np.ndarray,
    kmg: np.ndarray,
    keg: np.ndarray,
    files: np.ndarray,
    adj_files: np.ndarray,
    wspan: np.ndarray,
    bspan: np.ndarray,
    wdef: np.ndarray,
    bdef: np.ndarray,
    ps: np.ndarray,
    king_atk: np.ndarray,
    ka: np.ndarray,
    mobility_weight: int,
    bishop_pair: int,
    tempo: int,
    shield_missing: int,
    open_file: int,
    eg: np.ndarray,
    wback: np.ndarray,
    bback: np.ndarray,
) -> int:
    white_occ = bb[12]
    black_occ = bb[13]
    occ = bb[14]
    not_white = ~white_occ
    not_black = ~black_occ

    score = 0
    mobility = 0

    # pawns: material + PST only
    p = bb[0]
    while p:
        s = lsb(p)
        p &= p - U1
        score += wpst[1, s]
    p = bb[6]
    while p:
        s = lsb(p)
        p &= p - U1
        score += bpst[1, s]

    # knights, bishops, rooks, queens: material + PST + attack-span mobility
    for pt in range(2, 6):
        w = bb[pt - 1]
        while w:
            s = lsb(w)
            w &= w - U1
            score += wpst[pt, s]
            if pt == 2:
                atk = KNIGHT_ATK[s]
            elif pt == 3:
                atk = bishop_attacks(s, occ)
            elif pt == 4:
                atk = rook_attacks(s, occ)
            else:
                atk = queen_attacks(s, occ)
            mobility += popcount(atk & not_white)
        b = bb[pt + 5]
        while b:
            s = lsb(b)
            b &= b - U1
            score += bpst[pt, s]
            if pt == 2:
                atk = KNIGHT_ATK[s]
            elif pt == 3:
                atk = bishop_attacks(s, occ)
            elif pt == 4:
                atk = rook_attacks(s, occ)
            else:
                atk = queen_attacks(s, occ)
            mobility -= popcount(atk & not_black)

    phase = (
        popcount(bb[1] | bb[7])
        + popcount(bb[2] | bb[8])
        + 2 * popcount(bb[3] | bb[9])
        + 4 * popcount(bb[4] | bb[10])
    )
    if phase > 24:
        phase = 24
    frac = phase / 24

    wk = bb[5]
    if wk:
        ks = lsb(wk)
        score += _round_half_even(kmg[ks] * frac + keg[ks] * (1 - frac))
    bk = bb[11]
    if bk:
        m = lsb(bk) ^ 56
        score -= _round_half_even(kmg[m] * frac + keg[m] * (1 - frac))

    white_pawns = bb[0]
    black_pawns = bb[6]
    ks_white = 0
    ks_black = 0
    if wk:
        ks_white = _king_safety(
            lsb(wk), True, white_pawns, black_pawns,
            bb[7], bb[8], bb[9], bb[10], occ,
            files, king_atk, shield_missing, open_file, ka,
        )
    if bk:
        ks_black = _king_safety(
            lsb(bk), False, black_pawns, white_pawns,
            bb[1], bb[2], bb[3], bb[4], occ,
            files, king_atk, shield_missing, open_file, ka,
        )
    king_safety = ks_white - ks_black
    score += _round_half_even(king_safety * frac)

    score += mobility_weight * mobility

    score += _structure(
        bb[0], bb[6], bb[3], bb[9], files, adj_files, wspan, bspan, wdef, bdef, ps
    )

    if eg[6] and wk and bk:
        eg_scale = 1.0 - phase / 16.0
        if eg_scale > 0.0:
            score += _round_half_even(
                _endgame(
                    bb[0], bb[6], lsb(wk), lsb(bk), bb[3], bb[9],
                    files, adj_files, wspan, bspan, wdef, bdef, wback, bback, eg,
                )
                * eg_scale
            )

    if popcount(bb[2]) >= 2:
        score += bishop_pair
    if popcount(bb[8]) >= 2:
        score -= bishop_pair

    if bb[15] == 0:  # white to move
        return score + tempo
    return -score + tempo


def evaluate_bb(board: BBoard) -> int:
    """Static score in centipawns from the side-to-move's point of view."""
    return int(
        _score(
            board.bb,
            _WPST,
            _BPST,
            _KMG,
            _KEG,
            _FILES,
            _ADJ_FILES,
            _WSPAN,
            _BSPAN,
            _WDEF,
            _BDEF,
            _PS,
            _KING_ATK,
            _KA,
            _ev.MOBILITY_WEIGHT,
            _ev.BISHOP_PAIR,
            _ev.TEMPO,
            _ev.SHIELD_MISSING,
            _ev.OPEN_FILE_NEAR_KING,
            _EG,
            _WBACK,
            _BBACK,
        )
    )
