"""Handcrafted static evaluation.

`evaluate(board)` returns a score in centipawns from the point of view of the side to
move, which is the convention negamax wants. It is a pure function of the position:
no search, no terminal detection (the search owns checkmate and stalemate).

The terms are material, piece-square tables, a phase-blended king table, king safety,
mobility, the bishop pair and a tempo bonus.

The weights below (`WEIGHTS`) can be overridden from ``weights/eval.json`` so they
can be tuned offline without touching this file. Missing keys fall back to the
defaults here, so a partial or absent file is fine.
"""

from __future__ import annotations

import json
import os
import pathlib

import chess

# Optional king attack-zone term (enemy piece pressure on the squares around the
# king). Off by default: in testing it did not gain measurable strength and the
# current squared danger score can over-reward speculative attacks, so it is kept
# behind a flag pending a redesign. AICHESS_KING_ATTACK=1 turns it on.
_KING_ATTACK_ON = os.environ.get("AICHESS_KING_ATTACK", "0") != "0"

# Game phase: 24 = full middlegame, 0 = bare king-and-pawn endgame.
_PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
TOTAL_PHASE = 24

# Tunable weights. These may be overridden from weights/eval.json; anything
# missing there falls back to the default here, so a partial or absent file is
# fine. `PARAM_NAMES` is the subset exposed to offline tuning.
_DEFAULT_WEIGHTS: dict[str, int] = {
    "value_knight": 320,
    "value_bishop": 330,
    "value_rook": 500,
    "value_queen": 900,
    "bishop_pair": 30,
    "tempo": 10,
    "mobility": 1,
    "shield_missing": 12,
    "open_file_near_king": 14,
    # king attack zone: danger units per enemy piece bearing on the squares
    # around the king, plus units for an enemy rook/queen on a near file
    "king_attack_knight": 2,
    "king_attack_bishop": 2,
    "king_attack_rook": 3,
    "king_attack_queen": 5,
    "king_attack_file_open": 3,
    "king_attack_file_semi": 1,
    # pawn structure
    "passed_pawn_r2": 5,
    "passed_pawn_r3": 12,
    "passed_pawn_r4": 22,
    "passed_pawn_r5": 38,
    "passed_pawn_r6": 65,
    "passed_pawn_r7": 100,
    "passed_pawn_protected": 12,
    "doubled_pawn": 12,
    "isolated_pawn": 14,
    "rook_open_file": 20,
    "rook_semi_open_file": 10,
    # endgame-only positional terms (AICHESS_EG_TERMS=1); the whole group is
    # scaled by (1 - phase_frac) so it is inert in the middlegame
    "passed_connected": 15,
    "passed_outside": 12,
    "passed_king_march": 6,
    "rook_behind_passer": 16,
    "backward_pawn": 8,
    "pawn_islands": 6,
}
PARAM_NAMES: tuple[str, ...] = (
    "value_knight",
    "value_bishop",
    "value_rook",
    "value_queen",
    "mobility",
    "bishop_pair",
    "passed_pawn_r2",
    "passed_pawn_r3",
    "passed_pawn_r4",
    "passed_pawn_r5",
    "passed_pawn_r6",
    "passed_pawn_r7",
    "passed_pawn_protected",
    "doubled_pawn",
    "isolated_pawn",
    "rook_open_file",
    "rook_semi_open_file",
    "passed_connected",
    "passed_outside",
    "passed_king_march",
    "rook_behind_passer",
    "backward_pawn",
    "pawn_islands",
)
_WEIGHTS_PATH = pathlib.Path(__file__).with_name("weights") / "eval.json"


def _load_weights() -> dict[str, int]:
    weights = dict(_DEFAULT_WEIGHTS)
    try:
        loaded = json.loads(_WEIGHTS_PATH.read_text())
    except (OSError, ValueError):
        return weights
    for key in weights:
        value = loaded.get(key)
        if isinstance(value, (int, float)):
            weights[key] = round(value)
    return weights


WEIGHTS = _load_weights()
PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: WEIGHTS["value_knight"],
    chess.BISHOP: WEIGHTS["value_bishop"],
    chess.ROOK: WEIGHTS["value_rook"],
    chess.QUEEN: WEIGHTS["value_queen"],
    chess.KING: 0,
}
BISHOP_PAIR = WEIGHTS["bishop_pair"]
TEMPO = WEIGHTS["tempo"]
MOBILITY_WEIGHT = WEIGHTS["mobility"]
SHIELD_MISSING = WEIGHTS["shield_missing"]
OPEN_FILE_NEAR_KING = WEIGHTS["open_file_near_king"]

# King-attack term. `KING_ATTACK` is [knight, bishop, rook, queen] danger units
# per attacked square in the king zone (capped at 3 squares/piece), then
# [open-file, semi-open-file] units for an enemy rook/queen on the king's file or
# a neighbour. The unit total is squared and divided by `KING_ATTACK_DIVISOR`, so
# one loose attacker barely registers and a genuine assault compounds. The result
# folds into `_king_safety`, so it is phase-scaled with the rest of king safety.
def _king_attack_weights() -> list[int]:
    if not _KING_ATTACK_ON:
        return [0, 0, 0, 0, 0, 0]
    return [
        WEIGHTS["king_attack_knight"],
        WEIGHTS["king_attack_bishop"],
        WEIGHTS["king_attack_rook"],
        WEIGHTS["king_attack_queen"],
        WEIGHTS["king_attack_file_open"],
        WEIGHTS["king_attack_file_semi"],
    ]


KING_ATTACK = _king_attack_weights()
KING_ATTACK_DIVISOR = 8
_FULL_BB = (1 << 64) - 1

# Pawn structure. PASSED_PAWN is indexed by the pawn's own rank (0..7); only
# ranks 1..6 hold a pawn, the ends stay 0. Black reads it mirrored (7 - rank).
PASSED_PAWN = [
    0,
    WEIGHTS["passed_pawn_r2"],
    WEIGHTS["passed_pawn_r3"],
    WEIGHTS["passed_pawn_r4"],
    WEIGHTS["passed_pawn_r5"],
    WEIGHTS["passed_pawn_r6"],
    WEIGHTS["passed_pawn_r7"],
    0,
]
PASSED_PROTECTED = WEIGHTS["passed_pawn_protected"]
DOUBLED_PAWN = WEIGHTS["doubled_pawn"]
ISOLATED_PAWN = WEIGHTS["isolated_pawn"]
ROOK_OPEN_FILE = WEIGHTS["rook_open_file"]
ROOK_SEMI_OPEN_FILE = WEIGHTS["rook_semi_open_file"]

# Endgame-only positional terms. Off by default: passed-pawn dynamics
# (connected / outside passers, the king race to the queening square, a rook
# behind the passer) plus backward pawns and pawn-island count. The whole
# group is multiplied by max(0, 1 - phase/16) in `evaluate`, so it is exactly
# zero from a full board down through a normal middlegame (phase >= 16) and
# ramps in only as pieces come off. AICHESS_EG_TERMS=1 turns it on.
_EG_TERMS_ON = os.environ.get("AICHESS_EG_TERMS", "0") != "0"
PASSED_CONNECTED = WEIGHTS["passed_connected"]
PASSED_OUTSIDE = WEIGHTS["passed_outside"]
PASSED_KING_MARCH = WEIGHTS["passed_king_march"]
ROOK_BEHIND_PASSER = WEIGHTS["rook_behind_passer"]
BACKWARD_PAWN = WEIGHTS["backward_pawn"]
PAWN_ISLANDS = WEIGHTS["pawn_islands"]

_FILE_MASK = list(chess.BB_FILES)
_ADJ_FILE_MASK = [
    (_FILE_MASK[f - 1] if f > 0 else 0) | (_FILE_MASK[f + 1] if f < 7 else 0) for f in range(8)
]
_RANK_MASK = list(chess.BB_RANKS)


def _passed_spans() -> tuple[list[int], list[int]]:
    """For each square, the mask of squares 'in front' on its own + adjacent
    files -- the zone that must be clear of enemy pawns for a passer."""
    white = [0] * 64
    black = [0] * 64
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        span_files = _FILE_MASK[file_index] | _ADJ_FILE_MASK[file_index]
        for rank in range(rank_index + 1, 8):
            white[square] |= span_files & chess.BB_RANKS[rank]
        for rank in range(0, rank_index):
            black[square] |= span_files & chess.BB_RANKS[rank]
    return white, black


_WHITE_PASSED_SPAN, _BLACK_PASSED_SPAN = _passed_spans()
# Squares a friendly pawn would sit on to defend `square` (== where an enemy
# pawn on `square` would attack).
_WHITE_PAWN_DEFENDERS = list(chess.BB_PAWN_ATTACKS[chess.BLACK])
_BLACK_PAWN_DEFENDERS = list(chess.BB_PAWN_ATTACKS[chess.WHITE])


def _backward_spans() -> tuple[list[int], list[int]]:
    """For each square, the mask of the adjacent files at or behind its own
    rank -- where a friendly pawn would have to be to not leave this one
    backward. White reads `_WHITE_BACKWARD_SPAN`, Black the mirror."""
    white = [0] * 64
    black = [0] * 64
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        adj = _ADJ_FILE_MASK[file_index]
        for rank in range(0, rank_index + 1):
            white[square] |= adj & _RANK_MASK[rank]
        for rank in range(rank_index, 8):
            black[square] |= adj & _RANK_MASK[rank]
    return white, black


_WHITE_BACKWARD_SPAN, _BLACK_BACKWARD_SPAN = _backward_spans()

# Tables are written in reading order (a8 first, h1 last) and converted to be indexed
# by python-chess square numbers (a1 = 0). White reads the table directly; Black reads
# it vertically mirrored (square ^ 56).
_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]  # fmt: skip
_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]  # fmt: skip
_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]  # fmt: skip
_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]  # fmt: skip
_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]  # fmt: skip
_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]  # fmt: skip
_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]  # fmt: skip


def _by_square(reading_order: list[int]) -> list[int]:
    table = [0] * 64
    for index, value in enumerate(reading_order):
        table[index ^ 56] = value
    return table


# Fused material + PST, indexed by square, one table per piece type. White reads
# `_WHITE_PST[pt][square]`, Black reads `_BLACK_PST[pt][square]` (already negated and
# vertically mirrored), so the hot loop is a single add per piece with no branch.
_PST_READ: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK: _ROOK,
    chess.QUEEN: _QUEEN,
}
_PST_BY_SQUARE: dict[chess.PieceType, list[int]] = {
    pt: _by_square(reading) for pt, reading in _PST_READ.items()
}
_WHITE_PST: dict[chess.PieceType, list[int]] = {}
_BLACK_PST: dict[chess.PieceType, list[int]] = {}


def _rebuild_pst() -> None:
    """Refresh the fused material+PST tables from the current PIECE_VALUE."""
    for pt, sq_table in _PST_BY_SQUARE.items():
        _WHITE_PST[pt] = [PIECE_VALUE[pt] + sq_table[sq] for sq in range(64)]
        _BLACK_PST[pt] = [-(PIECE_VALUE[pt] + sq_table[sq ^ 56]) for sq in range(64)]


_rebuild_pst()

_KING_MG_SQ = _by_square(_KING_MG)
_KING_EG_SQ = _by_square(_KING_EG)

_BB_FILES = chess.BB_FILES
_MOBILE_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def _king_safety(board: chess.Board, color: chess.Color, own_pawns: int, enemy_pawns: int) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0
    king_file = chess.square_file(king_square)
    lo = max(0, king_file - 1)
    hi = min(7, king_file + 1)

    penalty = 0
    for file_index in range(lo, hi + 1):
        file_mask = _BB_FILES[file_index]
        if not (own_pawns & file_mask):
            penalty -= SHIELD_MISSING
            if not (enemy_pawns & file_mask):
                penalty -= OPEN_FILE_NEAR_KING

    kn, bi, ro, qu, file_open, file_semi = KING_ATTACK
    if not (kn or bi or ro or qu or file_open or file_semi):
        return penalty  # term disabled (AICHESS_KING_ATTACK=0)

    # Attack-zone pressure: the ring around the king, extended one rank toward the
    # enemy (where the attack comes from), and how hard the enemy pieces hit it.
    enemy = not color
    zone = chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square]
    zone = ((zone | (zone << 8)) & _FULL_BB) if color == chess.WHITE else (zone | (zone >> 8))

    danger = 0
    for piece_type, weight in ((chess.KNIGHT, kn), (chess.BISHOP, bi),
                               (chess.ROOK, ro), (chess.QUEEN, qu)):
        bb = board.pieces_mask(piece_type, enemy)
        while bb:
            low = bb & -bb
            bb ^= low
            hit = (board.attacks_mask(low.bit_length() - 1) & zone).bit_count()
            if hit:
                danger += weight * (hit if hit < 3 else 3)

    heavy = (board.rooks | board.queens) & board.occupied_co[enemy]
    for file_index in range(lo, hi + 1):
        file_mask = _BB_FILES[file_index]
        if (heavy & file_mask) and not (enemy_pawns & file_mask):
            danger += file_open if not (own_pawns & file_mask) else file_semi

    penalty -= danger * danger // KING_ATTACK_DIVISOR
    return penalty


def _structure(
    white_pawns: int, black_pawns: int, white_rooks: int, black_rooks: int
) -> int:
    """Pawn structure + rook files, White minus Black, in centipawns.

    Passed pawn (no enemy pawn on its file or the adjacent files ahead of it),
    scored by rank, with a bonus if a friendly pawn defends it; doubled pawn
    penalty per extra pawn on a file; isolated pawn penalty (no friendly pawn on
    an adjacent file); rook on a fully open / own-half-open file bonus.
    """
    all_pawns = white_pawns | black_pawns
    score = 0

    bb = white_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        if not (_ADJ_FILE_MASK[square & 7] & white_pawns):
            score -= ISOLATED_PAWN
        if not (_WHITE_PASSED_SPAN[square] & black_pawns):
            score += PASSED_PAWN[square >> 3]
            if _WHITE_PAWN_DEFENDERS[square] & white_pawns:
                score += PASSED_PROTECTED

    bb = black_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        if not (_ADJ_FILE_MASK[square & 7] & black_pawns):
            score += ISOLATED_PAWN
        if not (_BLACK_PASSED_SPAN[square] & white_pawns):
            score -= PASSED_PAWN[7 - (square >> 3)]
            if _BLACK_PAWN_DEFENDERS[square] & black_pawns:
                score -= PASSED_PROTECTED

    for file_index in range(8):
        white_on_file = (white_pawns & _FILE_MASK[file_index]).bit_count()
        if white_on_file > 1:
            score -= DOUBLED_PAWN * (white_on_file - 1)
        black_on_file = (black_pawns & _FILE_MASK[file_index]).bit_count()
        if black_on_file > 1:
            score += DOUBLED_PAWN * (black_on_file - 1)

    bb = white_rooks
    while bb:
        low = bb & -bb
        file_mask = _FILE_MASK[(low.bit_length() - 1) & 7]
        bb ^= low
        if not (file_mask & all_pawns):
            score += ROOK_OPEN_FILE
        elif not (file_mask & white_pawns):
            score += ROOK_SEMI_OPEN_FILE

    bb = black_rooks
    while bb:
        low = bb & -bb
        file_mask = _FILE_MASK[(low.bit_length() - 1) & 7]
        bb ^= low
        if not (file_mask & all_pawns):
            score -= ROOK_OPEN_FILE
        elif not (file_mask & black_pawns):
            score -= ROOK_SEMI_OPEN_FILE

    return score


def _pawn_islands(pawns: int) -> int:
    """Number of contiguous groups of occupied files."""
    islands = 0
    prev = False
    for file_index in range(8):
        cur = bool(pawns & _FILE_MASK[file_index])
        if cur and not prev:
            islands += 1
        prev = cur
    return islands


def _endgame(
    white_pawns: int,
    black_pawns: int,
    white_king: int,
    black_king: int,
    white_rooks: int,
    black_rooks: int,
) -> int:
    """Endgame positional score, White minus Black, in centipawns, BEFORE the
    max(0, 1 - phase/16) scale that `evaluate` applies. Passed-pawn relations
    (connected, outside, king race, rook behind) plus backward pawns and pawn
    islands -- the dynamic-endgame signal the flat structure term misses."""
    score = 0

    # pass 1: passed-pawn bitboards (own file + adjacent files ahead clear of
    # enemy pawns -- same test as `_structure`).
    white_passed = 0
    bb = white_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        if not (_WHITE_PASSED_SPAN[square] & black_pawns):
            white_passed |= low
    black_passed = 0
    bb = black_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        if not (_BLACK_PASSED_SPAN[square] & white_pawns):
            black_passed |= low

    wk_file = white_king & 7
    bk_file = black_king & 7

    bb = white_passed
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        file_index = square & 7
        promo = 56 + file_index
        if white_passed & _ADJ_FILE_MASK[file_index]:
            score += PASSED_CONNECTED
        if (file_index <= 1 or file_index >= 6) and abs(bk_file - file_index) >= 4:
            score += PASSED_OUTSIDE
        score += PASSED_KING_MARCH * (
            chess.square_distance(black_king, promo) - chess.square_distance(white_king, promo)
        )
        behind = _FILE_MASK[file_index] & (low - 1)
        if white_rooks & behind:
            score += ROOK_BEHIND_PASSER
        if black_rooks & behind:
            score -= ROOK_BEHIND_PASSER

    bb = black_passed
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        file_index = square & 7
        promo = file_index
        if black_passed & _ADJ_FILE_MASK[file_index]:
            score -= PASSED_CONNECTED
        if (file_index <= 1 or file_index >= 6) and abs(wk_file - file_index) >= 4:
            score -= PASSED_OUTSIDE
        score -= PASSED_KING_MARCH * (
            chess.square_distance(white_king, promo) - chess.square_distance(black_king, promo)
        )
        behind = _FILE_MASK[file_index] & ~((low << 1) - 1) & _FULL_BB
        if black_rooks & behind:
            score -= ROOK_BEHIND_PASSER
        if white_rooks & behind:
            score += ROOK_BEHIND_PASSER

    # backward pawns: on a half-open file, stop square controlled by an enemy
    # pawn, no friendly pawn level-or-behind on an adjacent file.
    bb = white_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        stop = square + 8
        if (
            stop < 64
            and (black_pawns & _BLACK_PAWN_DEFENDERS[stop])
            and not (white_pawns & _WHITE_BACKWARD_SPAN[square])
            and not (_FILE_MASK[square & 7] & black_pawns)
        ):
            score -= BACKWARD_PAWN
    bb = black_pawns
    while bb:
        low = bb & -bb
        square = low.bit_length() - 1
        bb ^= low
        stop = square - 8
        if (
            stop >= 0
            and (white_pawns & _WHITE_PAWN_DEFENDERS[stop])
            and not (black_pawns & _BLACK_BACKWARD_SPAN[square])
            and not (_FILE_MASK[square & 7] & white_pawns)
        ):
            score += BACKWARD_PAWN

    score -= PAWN_ISLANDS * (_pawn_islands(white_pawns) - 1)
    score += PAWN_ISLANDS * (_pawn_islands(black_pawns) - 1)

    return score


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns, from the point of view of the side to move."""
    white_occ = board.occupied_co[chess.WHITE]
    black_occ = board.occupied_co[chess.BLACK]
    not_white = ~white_occ
    not_black = ~black_occ

    score = 0
    mobility = 0
    pieces_mask = board.pieces_mask
    attacks_mask = board.attacks_mask

    # Pawns: material + PST only, no mobility term.
    bb = pieces_mask(chess.PAWN, chess.WHITE)
    table = _WHITE_PST[chess.PAWN]
    while bb:
        low = bb & -bb
        score += table[low.bit_length() - 1]
        bb ^= low
    bb = pieces_mask(chess.PAWN, chess.BLACK)
    table = _BLACK_PST[chess.PAWN]
    while bb:
        low = bb & -bb
        score += table[low.bit_length() - 1]
        bb ^= low

    # Knights, bishops, rooks, queens: material + PST + attack-span mobility.
    for piece_type in _MOBILE_TYPES:
        bb = pieces_mask(piece_type, chess.WHITE)
        table = _WHITE_PST[piece_type]
        while bb:
            low = bb & -bb
            square = low.bit_length() - 1
            score += table[square]
            mobility += (attacks_mask(square) & not_white).bit_count()
            bb ^= low
        bb = pieces_mask(piece_type, chess.BLACK)
        table = _BLACK_PST[piece_type]
        while bb:
            low = bb & -bb
            square = low.bit_length() - 1
            score += table[square]
            mobility -= (attacks_mask(square) & not_black).bit_count()
            bb ^= low

    phase = (
        board.knights.bit_count()
        + board.bishops.bit_count()
        + 2 * board.rooks.bit_count()
        + 4 * board.queens.bit_count()
    )
    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    frac = phase / TOTAL_PHASE

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is not None:
        score += round(_KING_MG_SQ[white_king] * frac + _KING_EG_SQ[white_king] * (1 - frac))
    if black_king is not None:
        mirrored = black_king ^ 56
        score -= round(_KING_MG_SQ[mirrored] * frac + _KING_EG_SQ[mirrored] * (1 - frac))

    white_pawns = board.pawns & white_occ
    black_pawns = board.pawns & black_occ
    king_safety = _king_safety(board, chess.WHITE, white_pawns, black_pawns) - _king_safety(
        board, chess.BLACK, black_pawns, white_pawns
    )
    score += round(king_safety * frac)

    score += MOBILITY_WEIGHT * mobility

    score += _structure(
        white_pawns, black_pawns, board.rooks & white_occ, board.rooks & black_occ
    )

    if _EG_TERMS_ON and white_king is not None and black_king is not None:
        eg_scale = 1.0 - phase / 16.0
        if eg_scale > 0.0:
            score += round(
                _endgame(
                    white_pawns,
                    black_pawns,
                    white_king,
                    black_king,
                    board.rooks & white_occ,
                    board.rooks & black_occ,
                )
                * eg_scale
            )

    if (board.bishops & white_occ).bit_count() >= 2:
        score += BISHOP_PAIR
    if (board.bishops & black_occ).bit_count() >= 2:
        score -= BISHOP_PAIR

    if board.turn == chess.WHITE:
        return score + TEMPO
    return -score + TEMPO


def set_weights(overrides: dict[str, float]) -> dict[str, int]:
    """Update the live evaluation weights in place and return the full weight set.

    Used only by the trainer -- it lets a running `Searcher` (which calls
    `evaluate` through module globals) pick up new weights without re-importing.
    `PIECE_VALUE` is mutated in place so `search.py`, which imported the dict
    object, sees the change too.
    """
    global BISHOP_PAIR, TEMPO, MOBILITY_WEIGHT, SHIELD_MISSING, OPEN_FILE_NEAR_KING
    global PASSED_PROTECTED, DOUBLED_PAWN, ISOLATED_PAWN, ROOK_OPEN_FILE, ROOK_SEMI_OPEN_FILE
    global KING_ATTACK
    global PASSED_CONNECTED, PASSED_OUTSIDE, PASSED_KING_MARCH, ROOK_BEHIND_PASSER
    global BACKWARD_PAWN, PAWN_ISLANDS
    for key, value in overrides.items():
        if key in WEIGHTS:
            WEIGHTS[key] = round(value)
    PIECE_VALUE[chess.KNIGHT] = WEIGHTS["value_knight"]
    PIECE_VALUE[chess.BISHOP] = WEIGHTS["value_bishop"]
    PIECE_VALUE[chess.ROOK] = WEIGHTS["value_rook"]
    PIECE_VALUE[chess.QUEEN] = WEIGHTS["value_queen"]
    BISHOP_PAIR = WEIGHTS["bishop_pair"]
    TEMPO = WEIGHTS["tempo"]
    MOBILITY_WEIGHT = WEIGHTS["mobility"]
    SHIELD_MISSING = WEIGHTS["shield_missing"]
    OPEN_FILE_NEAR_KING = WEIGHTS["open_file_near_king"]
    KING_ATTACK = _king_attack_weights()
    PASSED_PAWN[1:7] = [WEIGHTS[f"passed_pawn_r{r}"] for r in range(2, 8)]
    PASSED_PROTECTED = WEIGHTS["passed_pawn_protected"]
    DOUBLED_PAWN = WEIGHTS["doubled_pawn"]
    ISOLATED_PAWN = WEIGHTS["isolated_pawn"]
    ROOK_OPEN_FILE = WEIGHTS["rook_open_file"]
    ROOK_SEMI_OPEN_FILE = WEIGHTS["rook_semi_open_file"]
    PASSED_CONNECTED = WEIGHTS["passed_connected"]
    PASSED_OUTSIDE = WEIGHTS["passed_outside"]
    PASSED_KING_MARCH = WEIGHTS["passed_king_march"]
    ROOK_BEHIND_PASSER = WEIGHTS["rook_behind_passer"]
    BACKWARD_PAWN = WEIGHTS["backward_pawn"]
    PAWN_ISLANDS = WEIGHTS["pawn_islands"]
    _rebuild_pst()
    return dict(WEIGHTS)
