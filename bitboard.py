"""A numba-JIT bitboard move generator, built to replace python-chess in the
search hot path.

Correctness is checked against python-chess at every layer (see tools/bb_perft.py
and tools/bb_check.py). Nothing here is trusted until perft matches the published
counts exactly -- the same gate the python-chess perft already passed.

State layout
------------
The whole position is one ``np.uint64`` array of length 20 (``bb``) plus an
``np.int8`` mailbox of length 64 (``mb``). Every jitted function takes both.

    bb[ 0.. 5]  white P, N, B, R, Q, K   (piece bitboards)
    bb[ 6..11]  black P, N, B, R, Q, K
    bb[12]      occupancy white
    bb[13]      occupancy black
    bb[14]      occupancy all
    bb[15]      side to move          0 = white, 1 = black
    bb[16]      castling rights       bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
    bb[17]      en-passant target sq  0..63, or 64 for none
    bb[18]      halfmove clock
    bb[19]      fullmove number
    bb[20]      incremental Zobrist   piece-placement XOR only (see zobrist_fast)

    mb[sq]      0 empty, 1..6 white P..K, 7..12 black P..K

Move encoding (one int64)
-------------------------
    bits  0.. 5   from square
    bits  6..11   to square
    bits 12..14   promotion piece type (0 none, else 2..5 = N,B,R,Q)
    bits 15..17   flag: 0 normal, 1 double push, 2 en passant,
                        3 castle king-side, 4 castle queen-side
"""

from __future__ import annotations

import chess
import numpy as np
from numba import njit

# --- indices into the bb array -------------------------------------------------
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
OCC_W, OCC_B, OCC = 12, 13, 14
STM, CASTLE, EP, HALFMOVE, FULLMOVE = 15, 16, 17, 18, 19
HASH = 20
BB_LEN = 21
NO_EP = 64

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

FLAG_NORMAL, FLAG_DOUBLE, FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q = 0, 1, 2, 3, 4

MAX_MOVES = 256

# --- uint64 helpers -----------------------------------------------------------
U1 = np.uint64(1)
U0 = np.uint64(0)
_FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
_DEBRUIJN = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_INDEX = np.array(
    [0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4,
     62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,
     63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11,
     46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6],
    dtype=np.int64,
)


@njit(cache=True, inline="always")
def lsb(bb: np.uint64) -> int:
    """Index of the least significant set bit. bb must be non-zero."""
    isolated = bb & (~bb + U1)
    return _DEBRUIJN_INDEX[np.uint64(isolated * _DEBRUIJN) >> np.uint64(58)]


@njit(cache=True)
def msb(bb: np.uint64) -> int:
    """Index of the most significant set bit. bb must be non-zero."""
    bb |= bb >> U1
    bb |= bb >> np.uint64(2)
    bb |= bb >> np.uint64(4)
    bb |= bb >> np.uint64(8)
    bb |= bb >> np.uint64(16)
    bb |= bb >> np.uint64(32)
    return _DEBRUIJN_INDEX[np.uint64((bb & ~(bb >> U1)) * _DEBRUIJN) >> np.uint64(58)]


@njit(cache=True, inline="always")
def popcount(bb: np.uint64) -> int:
    c = 0
    while bb:
        bb &= bb - U1
        c += 1
    return c


# --- precomputed attack tables (built in Python at import) -------------------
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
# ray direction order: 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW
_RAY_DR = (1, 1, 0, -1, -1, -1, 0, 1)
_RAY_DF = (0, 1, 1, 1, 0, -1, -1, -1)
_ROOK_DIRS = (0, 2, 4, 6)
_BISHOP_DIRS = (1, 3, 5, 7)
_POSITIVE = (True, True, True, False, False, False, False, True)


def _build_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    knight = np.zeros(64, np.uint64)
    king = np.zeros(64, np.uint64)
    pawn = np.zeros((2, 64), np.uint64)
    ray = np.zeros((8, 64), np.uint64)

    for sq in range(64):
        r, f = sq >> 3, sq & 7
        for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)):
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                knight[sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for dr, df in _DIRS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                king[sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for df in (-1, 1):  # white pawn attacks
            nr, nf = r + 1, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                pawn[0, sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for df in (-1, 1):  # black pawn attacks
            nr, nf = r - 1, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                pawn[1, sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
        for d in range(8):
            nr, nf = r + _RAY_DR[d], f + _RAY_DF[d]
            while 0 <= nr < 8 and 0 <= nf < 8:
                ray[d, sq] |= np.uint64(1) << np.uint64(nr * 8 + nf)
                nr += _RAY_DR[d]
                nf += _RAY_DF[d]
    return knight, king, pawn, ray


KNIGHT_ATK, KING_ATK, PAWN_ATK, RAY = _build_tables()
_RANK_1 = np.uint64(0x00000000000000FF)
_RANK_8 = np.uint64(0xFF00000000000000)


def _build_between() -> np.ndarray:
    """BETWEEN[a, b] = the squares strictly between a and b along the ray a->b
    (exclusive of both endpoints); 0 if a and b are not on a common
    rank / file / diagonal. Used for pin resolution and check-block masks in the
    fast legal generator."""
    bt = np.zeros((64, 64), np.uint64)
    for a in range(64):
        ar, af = a >> 3, a & 7
        for d in range(8):
            path = 0
            r, f = ar + _RAY_DR[d], af + _RAY_DF[d]
            while 0 <= r < 8 and 0 <= f < 8:
                b = r * 8 + f
                bt[a, b] = np.uint64(path)
                path |= 1 << b
                r += _RAY_DR[d]
                f += _RAY_DF[d]
    return bt


BETWEEN = _build_between()


# --- magic bitboard tables (built in Python at import) ----------------------
# Magic numbers, masks and shifts are precomputed offline (tools/magicgen.py)
# and baked into magic_numbers.py so nothing is searched for at import time --
# only the attack tables themselves are rebuilt here, from a plain ray-scan
# over every occupancy subset of each square's relevant-occupancy mask
# (standard magic-bitboard technique; identical algorithm to the classical
# ray-scan below, just precomputed into a table indexed by (occ & mask) *
# magic >> shift instead of walked at lookup time).
from magic_numbers import (  # noqa: E402
    MAGIC_BISHOP, MAGIC_ROOK, MASK_BISHOP, MASK_ROOK, SHIFT_BISHOP,
    SHIFT_ROOK,
)

_M64 = 0xFFFFFFFFFFFFFFFF
_ROOK_TABLE_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_TABLE_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _ray_scan_py(square: int, occ: int, dirs) -> int:
    """Plain (unjitted) ray-scan, used only to build the magic tables once at
    import time -- mirrors tools/magicgen.py exactly."""
    r0, f0 = square >> 3, square & 7
    result = 0
    for dr, df in dirs:
        r, f = r0 + dr, f0 + df
        while 0 <= r < 8 and 0 <= f < 8:
            bit = 1 << (r * 8 + f)
            result |= bit
            if occ & bit:
                break
            r += dr
            f += df
    return result


def _subsets(mask: int):
    sub = 0
    while True:
        yield sub
        if sub == mask:
            break
        sub = (sub - mask) & mask


def _build_magic_tables() -> tuple[np.ndarray, np.ndarray]:
    rook_table = np.zeros((64, 4096), np.uint64)
    bishop_table = np.zeros((64, 512), np.uint64)
    for sq in range(64):
        for mask, magic, shift, dirs, table in (
            (MASK_ROOK[sq], MAGIC_ROOK[sq], SHIFT_ROOK[sq],
             _ROOK_TABLE_DIRS, rook_table),
            (MASK_BISHOP[sq], MAGIC_BISHOP[sq], SHIFT_BISHOP[sq],
             _BISHOP_TABLE_DIRS, bishop_table),
        ):
            for occ_subset in _subsets(mask):
                idx = ((occ_subset * magic) & _M64) >> shift
                table[sq, idx] = np.uint64(_ray_scan_py(sq, occ_subset, dirs))
    return rook_table, bishop_table


ROOK_MAGIC_TABLE, BISHOP_MAGIC_TABLE = _build_magic_tables()
MAGIC_ROOK_ARR = np.array(MAGIC_ROOK, dtype=np.uint64)
MAGIC_BISHOP_ARR = np.array(MAGIC_BISHOP, dtype=np.uint64)
MASK_ROOK_ARR = np.array(MASK_ROOK, dtype=np.uint64)
MASK_BISHOP_ARR = np.array(MASK_BISHOP, dtype=np.uint64)
SHIFT_ROOK_ARR = np.array(SHIFT_ROOK, dtype=np.int64)
SHIFT_BISHOP_ARR = np.array(SHIFT_BISHOP, dtype=np.int64)


# --- FEN <-> state (Python, only used at setup / for cross-checks) ----------
_PIECE_TO_MB = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
_MB_TO_PIECE = {v: k for k, v in _PIECE_TO_MB.items()}


def board_from_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    bb = np.zeros(BB_LEN, np.uint64)
    mb = np.zeros(64, np.int8)
    parts = fen.split()
    placement, turn = parts[0], parts[1]
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    halfmove = parts[4] if len(parts) > 4 else "0"
    fullmove = parts[5] if len(parts) > 5 else "1"

    rank = 7
    file = 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            code = _PIECE_TO_MB[ch]
            mb[sq] = code
            bb[code - 1] |= np.uint64(1) << np.uint64(sq)
            file += 1

    bb[STM] = np.uint64(0 if turn == "w" else 1)
    rights = 0
    if "K" in castling:
        rights |= CASTLE_WK
    if "Q" in castling:
        rights |= CASTLE_WQ
    if "k" in castling:
        rights |= CASTLE_BK
    if "q" in castling:
        rights |= CASTLE_BQ
    bb[CASTLE] = np.uint64(rights)
    bb[EP] = np.uint64(NO_EP if ep == "-" else chess.parse_square(ep))
    bb[HALFMOVE] = np.uint64(int(halfmove))
    bb[FULLMOVE] = np.uint64(int(fullmove))
    _recompute_occ(bb)
    bb[HASH] = _pieces_hash(bb)
    return bb, mb


def _recompute_occ(bb: np.ndarray) -> None:
    w = np.uint64(0)
    b = np.uint64(0)
    for i in range(6):
        w |= bb[i]
    for i in range(6, 12):
        b |= bb[i]
    bb[OCC_W] = w
    bb[OCC_B] = b
    bb[OCC] = w | b


def state_to_fen(bb: np.ndarray, mb: np.ndarray) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            code = int(mb[rank * 8 + file])
            if code == 0:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += _MB_TO_PIECE[code]
        if empty:
            row += str(empty)
        rows.append(row)
    placement = "/".join(rows)
    turn = "w" if bb[STM] == 0 else "b"
    rights = int(bb[CASTLE])
    cast = ""
    if rights & CASTLE_WK:
        cast += "K"
    if rights & CASTLE_WQ:
        cast += "Q"
    if rights & CASTLE_BK:
        cast += "k"
    if rights & CASTLE_BQ:
        cast += "q"
    cast = cast or "-"
    ep = "-" if int(bb[EP]) == NO_EP else chess.square_name(int(bb[EP]))
    return f"{placement} {turn} {cast} {ep} {int(bb[HALFMOVE])} {int(bb[FULLMOVE])}"


def move_to_uci(move: int) -> str:
    frm = move & 0x3F
    to = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    uci = chess.square_name(frm) + chess.square_name(to)
    if promo:
        uci += "nbrq"[promo - 2]
    return uci


def encode_move(frm: int, to: int, promo: int = 0, flag: int = 0) -> int:
    return frm | (to << 6) | (promo << 12) | (flag << 15)


# --- attack generation (jitted) --------------------------------------------
#
# Classical ray-scan versions -- kept as the correctness reference for the
# magic-bitboard tables below (tools/bb_magic_check.py cross-checks every
# magic lookup against these across all 64 squares and a large occupancy
# sample). Not used by any call site; `rook_attacks` / `bishop_attacks` /
# `queen_attacks` below are what movegen, eval and `is_attacked` actually
# call.
@njit(cache=True)
def rook_attacks_classical(sq: int, occ: np.uint64) -> np.uint64:
    result = U0
    r = RAY[0, sq]  # north (+), first blocker is the lowest set bit
    blk = r & occ
    if blk:
        r &= ~RAY[0, lsb(blk)]
    result |= r
    r = RAY[2, sq]  # east (+)
    blk = r & occ
    if blk:
        r &= ~RAY[2, lsb(blk)]
    result |= r
    r = RAY[4, sq]  # south (-)
    blk = r & occ
    if blk:
        r &= ~RAY[4, msb(blk)]
    result |= r
    r = RAY[6, sq]  # west (-)
    blk = r & occ
    if blk:
        r &= ~RAY[6, msb(blk)]
    result |= r
    return result


@njit(cache=True)
def bishop_attacks_classical(sq: int, occ: np.uint64) -> np.uint64:
    result = U0
    r = RAY[1, sq]  # north-east (+)
    blk = r & occ
    if blk:
        r &= ~RAY[1, lsb(blk)]
    result |= r
    r = RAY[7, sq]  # north-west (+)
    blk = r & occ
    if blk:
        r &= ~RAY[7, lsb(blk)]
    result |= r
    r = RAY[3, sq]  # south-east (-)
    blk = r & occ
    if blk:
        r &= ~RAY[3, msb(blk)]
    result |= r
    r = RAY[5, sq]  # south-west (-)
    blk = r & occ
    if blk:
        r &= ~RAY[5, msb(blk)]
    result |= r
    return result


@njit(cache=True)
def queen_attacks_classical(sq: int, occ: np.uint64) -> np.uint64:
    return rook_attacks_classical(sq, occ) | bishop_attacks_classical(sq, occ)


# Magic-bitboard versions -- the active implementation. One table lookup
# instead of 4 ray-walks + blocker searches.
@njit(cache=True, inline="always")
def rook_attacks(sq: int, occ: np.uint64) -> np.uint64:
    masked = occ & MASK_ROOK_ARR[sq]
    idx = (masked * MAGIC_ROOK_ARR[sq]) >> np.uint64(SHIFT_ROOK_ARR[sq])
    return ROOK_MAGIC_TABLE[sq, np.int64(idx)]


@njit(cache=True, inline="always")
def bishop_attacks(sq: int, occ: np.uint64) -> np.uint64:
    masked = occ & MASK_BISHOP_ARR[sq]
    idx = (masked * MAGIC_BISHOP_ARR[sq]) >> np.uint64(SHIFT_BISHOP_ARR[sq])
    return BISHOP_MAGIC_TABLE[sq, np.int64(idx)]


@njit(cache=True, inline="always")
def queen_attacks(sq: int, occ: np.uint64) -> np.uint64:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


@njit(cache=True, inline="always")
def is_attacked(bb: np.ndarray, sq: int, by_color: int) -> bool:
    """True if `sq` is attacked by any piece of side `by_color` (0 white, 1 black)."""
    base = 0 if by_color == 0 else 6
    occ = bb[OCC]
    sqb = np.uint64(sq)

    if PAWN_ATK[1 - by_color, sq] & bb[base + 0]:
        return True
    if KNIGHT_ATK[sq] & bb[base + 1]:
        return True
    if KING_ATK[sq] & bb[base + 5]:
        return True
    bishops_queens = bb[base + 2] | bb[base + 4]
    if bishop_attacks(sq, occ) & bishops_queens:
        return True
    rooks_queens = bb[base + 3] | bb[base + 4]
    if rook_attacks(sq, occ) & rooks_queens:
        return True
    _ = sqb
    return False


@njit(cache=True, inline="always")
def king_square(bb: np.ndarray, color: int) -> int:
    return lsb(bb[5] if color == 0 else bb[11])


@njit(cache=True, inline="always")
def in_check(bb: np.ndarray, color: int) -> bool:
    return is_attacked(bb, king_square(bb, color), 1 - color)


# --- Zobrist hashing -------------------------------------------------------
# A position key for the search transposition table. It need not match any
# external convention (python-chess uses its own): it only has to be a pure,
# collision-resistant function of the parts of the state that make two
# positions "the same" for search -- piece placement, side to move, castling
# rights and a usable en-passant file. Built once here from a fixed seed so
# the key is stable across processes (numba `cache=True`).
_zrng = np.random.default_rng(0x9E3779B97F4A7C15)
_Z_PIECE = _zrng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64, endpoint=False)
_Z_CASTLE = _zrng.integers(0, 1 << 64, size=16, dtype=np.uint64, endpoint=False)
_Z_EP = _zrng.integers(0, 1 << 64, size=8, dtype=np.uint64, endpoint=False)
_Z_STM = np.uint64(_zrng.integers(0, 1 << 64, dtype=np.uint64, endpoint=False))


@njit(cache=True)
def zobrist(bb: np.ndarray) -> np.uint64:
    h = U0
    for pc in range(12):
        b = bb[pc]
        while b:
            s = lsb(b)
            b &= b - U1
            h ^= _Z_PIECE[pc, s]
    h ^= _Z_CASTLE[bb[CASTLE] & np.uint64(0xF)]
    ep = bb[EP]
    if ep != np.uint64(NO_EP):
        # Match python-chess's `_transposition_key`: only fold in the en-passant
        # file when the side to move actually has a pawn placed to take it.
        # (python-chess also checks the capture is not self-check; that residual
        # case -- an e.p. that would expose the king -- is rare enough to leave.)
        stm = np.int64(bb[STM])
        our_pawns = bb[0] if stm == 0 else bb[6]
        if PAWN_ATK[1 - stm, int(ep)] & our_pawns:
            h ^= _Z_EP[ep & np.uint64(7)]
    if bb[STM] != U0:
        h ^= _Z_STM
    return h


@njit(cache=True)
def _pieces_hash(bb: np.ndarray) -> np.uint64:
    """The piece-placement half of the Zobrist key -- the part `make_move` /
    `unmake_move` maintain incrementally in bb[HASH]. Identical to the piece
    loop in `zobrist`."""
    h = U0
    for pc in range(12):
        b = bb[pc]
        while b:
            s = lsb(b)
            b &= b - U1
            h ^= _Z_PIECE[pc, s]
    return h


@njit(cache=True, inline="always")
def zobrist_fast(bb: np.ndarray) -> np.uint64:
    """Same value as `zobrist(bb)`, read from the incremental bb[HASH] instead of
    rescanning every piece. The castling / en-passant / side-to-move terms are
    O(1) and recomputed here verbatim from `zobrist`; only the piece scan (the
    expensive part) is replaced by the maintained accumulator.

    Requires bb[HASH] == _pieces_hash(bb), which board_from_fen establishes and
    make_move / unmake_move preserve. `tools/bb_hashcheck` asserts the two agree
    bit-for-bit after every make and unmake.
    """
    h = bb[HASH]
    h ^= _Z_CASTLE[bb[CASTLE] & np.uint64(0xF)]
    ep = bb[EP]
    if ep != np.uint64(NO_EP):
        stm = np.int64(bb[STM])
        our_pawns = bb[0] if stm == 0 else bb[6]
        if PAWN_ATK[1 - stm, int(ep)] & our_pawns:
            h ^= _Z_EP[ep & np.uint64(7)]
    if bb[STM] != U0:
        h ^= _Z_STM
    return h


# --- make / unmake --------------------------------------------------------
@njit(cache=True, inline="always")
def _sync_occ(bb: np.ndarray) -> None:
    w = bb[0] | bb[1] | bb[2] | bb[3] | bb[4] | bb[5]
    b = bb[6] | bb[7] | bb[8] | bb[9] | bb[10] | bb[11]
    bb[OCC_W] = w
    bb[OCC_B] = b
    bb[OCC] = w | b


@njit(cache=True)
def make_move(bb: np.ndarray, mb: np.ndarray, move: int) -> np.int64:
    frm = move & 0x3F
    to = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7

    us = np.int64(bb[STM])
    base = 0 if us == 0 else 6

    old_castle = np.int64(bb[CASTLE])
    old_ep = np.int64(bb[EP])
    old_half = np.int64(bb[HALFMOVE])

    moving_code = np.int64(mb[frm])
    moving_pt = ((moving_code - 1) % 6) + 1
    captured_code = np.int64(0)

    frm_bit = U1 << np.uint64(frm)
    to_bit = U1 << np.uint64(to)

    bb[EP] = np.uint64(NO_EP)
    bb[moving_code - 1] ^= frm_bit
    mb[frm] = 0
    # incremental Zobrist: XOR each piece as it leaves / lands (bb[HASH] holds the
    # piece-placement half only; zobrist_fast recomputes castling/ep/stm)
    hd = _Z_PIECE[moving_code - 1, frm]

    if flag == FLAG_EP:
        cap_sq = to - 8 if us == 0 else to + 8
        captured_code = np.int64(mb[cap_sq])
        bb[captured_code - 1] ^= U1 << np.uint64(cap_sq)
        mb[cap_sq] = 0
        bb[moving_code - 1] ^= to_bit
        mb[to] = moving_code
        hd ^= _Z_PIECE[captured_code - 1, cap_sq] ^ _Z_PIECE[moving_code - 1, to]
    elif flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:  # noqa: SIM109
        bb[moving_code - 1] ^= to_bit
        mb[to] = moving_code
        if flag == FLAG_CASTLE_K:
            rook_from = to + 1
            rook_to = to - 1
        else:
            rook_from = to - 2
            rook_to = to + 1
        rook_code = np.int64(mb[rook_from])
        bb[rook_code - 1] ^= (U1 << np.uint64(rook_from)) | (U1 << np.uint64(rook_to))
        mb[rook_from] = 0
        mb[rook_to] = rook_code
        hd ^= (
            _Z_PIECE[moving_code - 1, to]
            ^ _Z_PIECE[rook_code - 1, rook_from]
            ^ _Z_PIECE[rook_code - 1, rook_to]
        )
    else:
        to_code = np.int64(mb[to])
        if to_code != 0:
            captured_code = to_code
            bb[to_code - 1] ^= to_bit
            hd ^= _Z_PIECE[to_code - 1, to]
        placed_code = (base + promo) if promo != 0 else moving_code
        bb[placed_code - 1] ^= to_bit
        mb[to] = placed_code
        hd ^= _Z_PIECE[placed_code - 1, to]
        if flag == FLAG_DOUBLE:
            bb[EP] = np.uint64(frm + 8 if us == 0 else frm - 8)

    if moving_pt == 6:
        if us == 0:
            bb[CASTLE] &= np.uint64(~(CASTLE_WK | CASTLE_WQ) & 0xF)
        else:
            bb[CASTLE] &= np.uint64(~(CASTLE_BK | CASTLE_BQ) & 0xF)
    if frm == 0 or to == 0:
        bb[CASTLE] &= np.uint64(~CASTLE_WQ & 0xF)
    if frm == 7 or to == 7:
        bb[CASTLE] &= np.uint64(~CASTLE_WK & 0xF)
    if frm == 56 or to == 56:
        bb[CASTLE] &= np.uint64(~CASTLE_BQ & 0xF)
    if frm == 63 or to == 63:
        bb[CASTLE] &= np.uint64(~CASTLE_BK & 0xF)

    if moving_pt == 1 or captured_code != 0:
        bb[HALFMOVE] = U0
    else:
        bb[HALFMOVE] = np.uint64(old_half + 1)

    bb[STM] = np.uint64(1 - us)
    if us == 1:
        bb[FULLMOVE] += U1

    bb[HASH] ^= hd
    _sync_occ(bb)
    return captured_code | (old_castle << 4) | (old_ep << 8) | (old_half << 15)


@njit(cache=True)
def unmake_move(bb: np.ndarray, mb: np.ndarray, move: int, undo: np.int64) -> None:
    frm = move & 0x3F
    to = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7

    captured_code = undo & 0xF
    old_castle = (undo >> 4) & 0xF
    old_ep = (undo >> 8) & 0x7F
    old_half = (undo >> 15) & 0x7F

    us = np.int64(1 - bb[STM])
    base = 0 if us == 0 else 6

    bb[STM] = np.uint64(us)
    bb[CASTLE] = np.uint64(old_castle)
    bb[EP] = np.uint64(old_ep)
    bb[HALFMOVE] = np.uint64(old_half)
    if us == 1:
        bb[FULLMOVE] -= U1

    frm_bit = U1 << np.uint64(frm)
    to_bit = U1 << np.uint64(to)
    placed_code = np.int64(mb[to])
    moving_code = np.int64(base + 1) if promo != 0 else placed_code

    # incremental Zobrist: rebuild the exact delta make_move XORed in, then XOR it
    # back out (XOR is its own inverse). Mirrors make_move branch for branch.
    hd = _Z_PIECE[moving_code - 1, frm]

    if flag == FLAG_EP:
        bb[moving_code - 1] ^= frm_bit | to_bit
        mb[frm] = moving_code
        mb[to] = 0
        cap_sq = to - 8 if us == 0 else to + 8
        bb[captured_code - 1] ^= U1 << np.uint64(cap_sq)
        mb[cap_sq] = captured_code
        hd ^= _Z_PIECE[captured_code - 1, cap_sq] ^ _Z_PIECE[moving_code - 1, to]
    elif flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:  # noqa: SIM109
        bb[moving_code - 1] ^= frm_bit | to_bit
        mb[frm] = moving_code
        mb[to] = 0
        if flag == FLAG_CASTLE_K:
            rook_from = to + 1
            rook_to = to - 1
        else:
            rook_from = to - 2
            rook_to = to + 1
        rook_code = np.int64(mb[rook_to])
        bb[rook_code - 1] ^= (U1 << np.uint64(rook_from)) | (U1 << np.uint64(rook_to))
        mb[rook_from] = rook_code
        mb[rook_to] = 0
        hd ^= (
            _Z_PIECE[moving_code - 1, to]
            ^ _Z_PIECE[rook_code - 1, rook_from]
            ^ _Z_PIECE[rook_code - 1, rook_to]
        )
    else:
        bb[placed_code - 1] ^= to_bit
        mb[to] = 0
        if captured_code != 0:
            bb[captured_code - 1] ^= to_bit
            mb[to] = captured_code
            hd ^= _Z_PIECE[captured_code - 1, to]
        bb[moving_code - 1] ^= frm_bit
        mb[frm] = moving_code
        hd ^= _Z_PIECE[placed_code - 1, to]

    bb[HASH] ^= hd
    _sync_occ(bb)


# --- move generation -----------------------------------------------------
@njit(cache=True)
def _castle_path_clear(
    bb: np.ndarray, occ: np.uint64, k_from: int, k_to: int, between: np.uint64, enemy: int
) -> bool:
    if occ & between:
        return False
    mid = (k_from + k_to) // 2
    return not (
        is_attacked(bb, k_from, enemy)
        or is_attacked(bb, mid, enemy)
        or is_attacked(bb, k_to, enemy)
    )


@njit(cache=True)
def _add_pawn_moves(
    moves: np.ndarray, n: int, frm: int, to: int, flag: int, promo_rank: bool
) -> int:
    if promo_rank:
        moves[n] = frm | (to << 6) | (5 << 12) | (flag << 15)
        moves[n + 1] = frm | (to << 6) | (4 << 12) | (flag << 15)
        moves[n + 2] = frm | (to << 6) | (3 << 12) | (flag << 15)
        moves[n + 3] = frm | (to << 6) | (2 << 12) | (flag << 15)
        return n + 4
    moves[n] = frm | (to << 6) | (flag << 15)
    return n + 1


@njit(cache=True)
def gen_pseudo(bb: np.ndarray, mb: np.ndarray, moves: np.ndarray) -> int:
    us = np.int64(bb[STM])
    them = 1 - us
    base = 0 if us == 0 else 6
    occ = bb[OCC]
    own = bb[OCC_W] if us == 0 else bb[OCC_B]
    enemy = bb[OCC_B] if us == 0 else bb[OCC_W]
    empty = ~occ
    n = 0

    # --- pawns ---
    pawns = bb[base + 0]
    forward = 8 if us == 0 else -8
    start_lo = 8 if us == 0 else 48
    start_hi = 15 if us == 0 else 55
    promo_lo = 56 if us == 0 else 0
    promo_hi = 63 if us == 0 else 7
    ep_sq = np.int64(bb[EP])
    p = pawns
    while p:
        frm = lsb(p)
        p &= p - U1
        one = frm + forward
        if (empty >> np.uint64(one)) & U1:
            promo = promo_lo <= one <= promo_hi
            n = _add_pawn_moves(moves, n, frm, one, FLAG_NORMAL, promo)
            if start_lo <= frm <= start_hi:
                two = one + forward
                if (empty >> np.uint64(two)) & U1:
                    moves[n] = frm | (two << 6) | (FLAG_DOUBLE << 15)
                    n += 1
        caps = PAWN_ATK[us, frm] & enemy
        while caps:
            to = lsb(caps)
            caps &= caps - U1
            promo = promo_lo <= to <= promo_hi
            n = _add_pawn_moves(moves, n, frm, to, FLAG_NORMAL, promo)
        if ep_sq != NO_EP and (PAWN_ATK[us, frm] >> np.uint64(ep_sq)) & U1:
            moves[n] = frm | (int(ep_sq) << 6) | (FLAG_EP << 15)
            n += 1

    # --- knights ---
    kn = bb[base + 1]
    while kn:
        frm = lsb(kn)
        kn &= kn - U1
        tgt = KNIGHT_ATK[frm] & ~own
        while tgt:
            to = lsb(tgt)
            tgt &= tgt - U1
            moves[n] = frm | (to << 6)
            n += 1

    # --- bishops / rooks / queens ---
    for pt_off in (2, 3, 4):
        pieces = bb[base + pt_off]
        while pieces:
            frm = lsb(pieces)
            pieces &= pieces - U1
            if pt_off == 2:
                tgt = bishop_attacks(frm, occ) & ~own
            elif pt_off == 3:
                tgt = rook_attacks(frm, occ) & ~own
            else:
                tgt = queen_attacks(frm, occ) & ~own
            while tgt:
                to = lsb(tgt)
                tgt &= tgt - U1
                moves[n] = frm | (to << 6)
                n += 1

    # --- king ---
    ksq = lsb(bb[base + 5])
    tgt = KING_ATK[ksq] & ~own
    while tgt:
        to = lsb(tgt)
        tgt &= tgt - U1
        moves[n] = ksq | (to << 6)
        n += 1

    # --- castling: rights set, squares between empty, king not passing check ---
    rights = np.int64(bb[CASTLE])
    if us == 0:
        clear = _castle_path_clear(bb, occ, 4, 6, np.uint64(0x60), 1)
        if (rights & CASTLE_WK) and clear:
            moves[n] = 4 | (6 << 6) | (FLAG_CASTLE_K << 15)
            n += 1
        clear = _castle_path_clear(bb, occ, 4, 2, np.uint64(0x0E), 1)
        if (rights & CASTLE_WQ) and clear:
            moves[n] = 4 | (2 << 6) | (FLAG_CASTLE_Q << 15)
            n += 1
    else:
        clear = _castle_path_clear(bb, occ, 60, 62, np.uint64(0x6000000000000000), 0)
        if (rights & CASTLE_BK) and clear:
            moves[n] = 60 | (62 << 6) | (FLAG_CASTLE_K << 15)
            n += 1
        clear = _castle_path_clear(bb, occ, 60, 58, np.uint64(0x0E00000000000000), 0)
        if (rights & CASTLE_BQ) and clear:
            moves[n] = 60 | (58 << 6) | (FLAG_CASTLE_Q << 15)
            n += 1

    _ = them
    _ = base
    return n


# Objective B (docs/V5_OPTIMIZATION_LEDGER.md): qsearch (when not in check)
# only ever wants captures + promotions -- it previously called gen_pseudo
# (full pseudo-legal generation, including every quiet non-promo move) and
# then filtered. This generates only the moves qsearch's own filter
# (`is_cap or promo != 0`) would have kept: pawn captures, en-passant,
# promoting pushes (capture or not), and piece moves whose target is an
# enemy-occupied square. Quiet non-promo pawn pushes, quiet piece moves,
# and castling (never a capture or promotion) are never emitted -- not
# generated-then-discarded, simply never computed. Byte-identical move
# encoding to gen_pseudo's own captures/promotions subset for the same
# position -- verified by direct set-equality, not assumed from mirroring
# the code shape. Order is not required to match (qsearch does its own
# MVV-LVA selection-sort over whatever set it receives).
@njit(cache=True)
def gen_pseudo_captures(bb: np.ndarray, mb: np.ndarray, moves: np.ndarray) -> int:
    us = np.int64(bb[STM])
    base = 0 if us == 0 else 6
    occ = bb[OCC]
    own = bb[OCC_W] if us == 0 else bb[OCC_B]
    enemy = bb[OCC_B] if us == 0 else bb[OCC_W]
    empty = ~occ
    n = 0

    # --- pawns: captures, en passant, and promoting pushes only ---
    pawns = bb[base + 0]
    forward = 8 if us == 0 else -8
    promo_lo = 56 if us == 0 else 0
    promo_hi = 63 if us == 0 else 7
    ep_sq = np.int64(bb[EP])
    p = pawns
    while p:
        frm = lsb(p)
        p &= p - U1
        one = frm + forward
        if (empty >> np.uint64(one)) & U1:
            if promo_lo <= one <= promo_hi:
                n = _add_pawn_moves(moves, n, frm, one, FLAG_NORMAL, True)
        caps = PAWN_ATK[us, frm] & enemy
        while caps:
            to = lsb(caps)
            caps &= caps - U1
            promo = promo_lo <= to <= promo_hi
            n = _add_pawn_moves(moves, n, frm, to, FLAG_NORMAL, promo)
        if ep_sq != NO_EP and (PAWN_ATK[us, frm] >> np.uint64(ep_sq)) & U1:
            moves[n] = frm | (int(ep_sq) << 6) | (FLAG_EP << 15)
            n += 1

    # --- knights ---
    kn = bb[base + 1]
    while kn:
        frm = lsb(kn)
        kn &= kn - U1
        tgt = KNIGHT_ATK[frm] & enemy
        while tgt:
            to = lsb(tgt)
            tgt &= tgt - U1
            moves[n] = frm | (to << 6)
            n += 1

    # --- bishops / rooks / queens ---
    for pt_off in (2, 3, 4):
        pieces = bb[base + pt_off]
        while pieces:
            frm = lsb(pieces)
            pieces &= pieces - U1
            if pt_off == 2:
                tgt = bishop_attacks(frm, occ) & enemy
            elif pt_off == 3:
                tgt = rook_attacks(frm, occ) & enemy
            else:
                tgt = queen_attacks(frm, occ) & enemy
            while tgt:
                to = lsb(tgt)
                tgt &= tgt - U1
                moves[n] = frm | (to << 6)
                n += 1

    # --- king ---
    ksq = lsb(bb[base + 5])
    tgt = KING_ATK[ksq] & enemy
    while tgt:
        to = lsb(tgt)
        tgt &= tgt - U1
        moves[n] = ksq | (to << 6)
        n += 1

    _ = own
    return n


@njit(cache=True)
def gen_legal(bb: np.ndarray, mb: np.ndarray, moves: np.ndarray, scratch: np.ndarray) -> int:
    us = np.int64(bb[STM])
    m = gen_pseudo(bb, mb, scratch)
    n = 0
    for i in range(m):
        mv = scratch[i]
        undo = make_move(bb, mb, mv)
        if not in_check(bb, us):
            moves[n] = mv
            n += 1
        unmake_move(bb, mb, mv, undo)
    return n


@njit(cache=True, inline="always")
def _pinned_mask(
    bb: np.ndarray, us: int, king_sq: int, occ: np.uint64, own: np.uint64
) -> np.uint64:
    """Own pieces pinned to `king_sq` by an enemy slider.

    xray trick: drop our own first-blockers on each ray out of the king, then see
    which enemy rook / bishop / queen becomes visible behind exactly one of them.
    That one blocker is pinned; BETWEEN isolates it.
    """
    ebase = 6 if us == 0 else 0
    e_rq = bb[ebase + 3] | bb[ebase + 4]
    e_bq = bb[ebase + 2] | bb[ebase + 4]
    pinned = U0

    rray = rook_attacks(king_sq, occ)
    xr = rook_attacks(king_sq, occ ^ (rray & own)) & ~rray & e_rq
    while xr:
        s = lsb(xr)
        xr &= xr - U1
        pinned |= BETWEEN[king_sq, s] & own

    bray = bishop_attacks(king_sq, occ)
    xb = bishop_attacks(king_sq, occ ^ (bray & own)) & ~bray & e_bq
    while xb:
        s = lsb(xb)
        xb &= xb - U1
        pinned |= BETWEEN[king_sq, s] & own

    return pinned


@njit(cache=True)
def gen_legal_fast(bb: np.ndarray, mb: np.ndarray, moves: np.ndarray, scratch: np.ndarray) -> int:
    """Same legal move set, in the same order, as `gen_legal` -- but most moves
    are proven legal from a pin mask + a check-block mask instead of a full
    make / in_check / unmake round trip.

    The make/unmake path is kept only for the genuinely ambiguous cases: king
    moves (king-danger squares), en passant (the horizontal discovered-check
    corner case), and moving a pinned piece. Everything else:
      - not in check      -> legal (non-pinned piece can't expose the king)
      - single check      -> legal iff it lands on a capture-or-block square
      - double check       -> illegal (only king moves parry, handled above)
    """
    us = np.int64(bb[STM])
    m = gen_pseudo(bb, mb, scratch)
    king_sq = lsb(bb[5] if us == 0 else bb[11])
    occ = bb[OCC]
    own = bb[OCC_W] if us == 0 else bb[OCC_B]
    ebase = 6 if us == 0 else 0

    checkers = (
        (PAWN_ATK[us, king_sq] & bb[ebase + 0])
        | (KNIGHT_ATK[king_sq] & bb[ebase + 1])
        | (bishop_attacks(king_sq, occ) & (bb[ebase + 2] | bb[ebase + 4]))
        | (rook_attacks(king_sq, occ) & (bb[ebase + 3] | bb[ebase + 4]))
    )
    nchk = popcount(checkers)
    pinned = _pinned_mask(bb, us, king_sq, occ, own)
    # single check: the squares that capture or block the checker
    target = (BETWEEN[king_sq, lsb(checkers)] | checkers) if nchk == 1 else U0

    n = 0
    for i in range(m):
        mv = scratch[i]
        frm = mv & 0x3F
        flag = (mv >> 15) & 7
        if (
            frm == king_sq
            or flag == FLAG_EP
            or ((pinned >> np.uint64(frm)) & U1) != U0
        ):
            undo = make_move(bb, mb, mv)
            ok = not in_check(bb, us)
            unmake_move(bb, mb, mv, undo)
        elif nchk == 0:
            ok = True
        elif nchk == 1:
            to = (mv >> 6) & 0x3F
            ok = ((target >> np.uint64(to)) & U1) != U0
        else:
            ok = False
        if ok:
            moves[n] = mv
            n += 1
    return n


# --- perft (jitted) ------------------------------------------------------
@njit(cache=True)
def _perft(bb: np.ndarray, mb: np.ndarray, depth: int, stack: np.ndarray, ply: int) -> np.int64:
    us = np.int64(bb[STM])
    n = gen_pseudo(bb, mb, stack[ply])
    nodes = np.int64(0)
    if depth == 1:
        for i in range(n):
            mv = stack[ply][i]
            undo = make_move(bb, mb, mv)
            if not in_check(bb, us):
                nodes += 1
            unmake_move(bb, mb, mv, undo)
        return nodes
    for i in range(n):
        mv = stack[ply][i]
        undo = make_move(bb, mb, mv)
        if not in_check(bb, us):
            nodes += _perft(bb, mb, depth - 1, stack, ply + 1)
        unmake_move(bb, mb, mv, undo)
    return nodes


@njit(cache=True)
def perft(bb: np.ndarray, mb: np.ndarray, depth: int) -> np.int64:
    if depth == 0:
        return np.int64(1)
    stack = np.zeros((depth + 1, MAX_MOVES), np.int64)
    return _perft(bb, mb, depth, stack, 0)


@njit(cache=True)
def _perft_legal(
    bb: np.ndarray, mb: np.ndarray, depth: int, stack: np.ndarray, scratch: np.ndarray, ply: int
) -> np.int64:
    n = gen_legal_fast(bb, mb, stack[ply], scratch)
    if depth == 1:
        return np.int64(n)
    nodes = np.int64(0)
    for i in range(n):
        mv = stack[ply][i]
        undo = make_move(bb, mb, mv)
        nodes += _perft_legal(bb, mb, depth - 1, stack, scratch, ply + 1)
        unmake_move(bb, mb, mv, undo)
    return nodes


@njit(cache=True)
def perft_legal(bb: np.ndarray, mb: np.ndarray, depth: int) -> np.int64:
    """perft driven through `gen_legal_fast` -- validates the fast legal
    generator's move set at every node against the published counts."""
    if depth == 0:
        return np.int64(1)
    stack = np.zeros((depth + 1, MAX_MOVES), np.int64)
    scratch = np.zeros(MAX_MOVES, np.int64)
    return _perft_legal(bb, mb, depth, stack, scratch, 0)


def divide(fen: str, depth: int) -> dict[str, int]:
    """perft(depth) split by root move, as {uci: count}, sorted."""
    bb, mb = board_from_fen(fen)
    moves = np.zeros(MAX_MOVES, np.int64)
    scratch = np.zeros(MAX_MOVES, np.int64)
    n = gen_legal(bb, mb, moves, scratch)
    out: dict[str, int] = {}
    for i in range(n):
        mv = int(moves[i])
        undo = make_move(bb, mb, mv)
        out[move_to_uci(mv)] = 1 if depth == 1 else int(perft(bb, mb, depth - 1))
        unmake_move(bb, mb, mv, undo)
    return dict(sorted(out.items()))
