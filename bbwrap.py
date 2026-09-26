"""Adapter: the numba bitboard state (``bitboard.py``) behind the slice of the
python-chess ``Board`` / ``Move`` API that ``search.py`` and ``evaluate.py``
actually use.

The point is to swap the board representation and move generation under the
search **without** restructuring the search. ``search.py`` keeps calling
``board.push`` / ``board.legal_moves`` / ``board.is_check`` etc.; here those
route to the jitted bitboard routines. Moves are lightweight ``BBMove`` objects
wrapping the packed-int encoding, with equality on (from, to, promo) so a move
reconstructed from UCI (a TT move) still compares equal to the generated one.

Correctness of the underlying movegen is already gated by perft
(``tools/bb_perft.py``) and the layered cross-check (``tools/bb_check.py``);
this file adds only the thin Python veneer, which ``tools/bb_check.py`` also
exercises through the integrated search.
"""

from __future__ import annotations

import chess
import numpy as np

import bitboard as _bb
from bitboard import (
    EP,
    FLAG_EP,
    HALFMOVE,
    KING_ATK,
    KNIGHT_ATK,
    MAX_MOVES,
    NO_EP,
    OCC,
    OCC_B,
    OCC_W,
    STM,
    bishop_attacks,
    board_from_fen,
    gen_legal,
    in_check,
    lsb,
    make_move,
    move_to_uci,
    queen_attacks,
    rook_attacks,
    unmake_move,
    zobrist,
)

_CH_PROMO = {"n": 2, "b": 3, "r": 4, "q": 5}

BB_DARK_SQUARES = 0xAA55AA55AA55AA55
BB_LIGHT_SQUARES = (~BB_DARK_SQUARES) & 0xFFFFFFFFFFFFFFFF
_ALL = 0xFFFFFFFFFFFFFFFF


class BBMove:
    """A move as the packed int from ``bitboard.py``, wearing the tiny bit of
    ``chess.Move`` that the search touches."""

    __slots__ = ("v",)

    def __init__(self, v: int) -> None:
        self.v = int(v)

    @property
    def from_square(self) -> int:
        return self.v & 0x3F

    @property
    def to_square(self) -> int:
        return (self.v >> 6) & 0x3F

    @property
    def promotion(self) -> int | None:
        p = (self.v >> 12) & 0x7
        return p if 2 <= p <= 5 else None

    @property
    def flag(self) -> int:
        return (self.v >> 15) & 0x7

    def uci(self) -> str:
        return move_to_uci(self.v)

    # Equality ignores the flag bits: a castle / e.p. / double-push move must
    # compare equal to the same (from, to, promo) reconstructed from a UCI
    # string, which is how TT moves come back.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BBMove) and (self.v & 0x7FFF) == (other.v & 0x7FFF)

    def __hash__(self) -> int:
        return self.v & 0x7FFF

    def __repr__(self) -> str:
        return f"BBMove({self.uci()})"


# A sentinel for the null move. from/to = 63, promo bits = 7 -- never a real move.
NULL_MOVE = BBMove(0x7FFF)


def move_from_uci(uci: str) -> BBMove:
    """Reconstruct a (comparison-only) BBMove from a UCI string. The flag bits
    are left zero; that is fine because such a move is only ever compared, never
    made."""
    frm = chess.parse_square(uci[0:2])
    to = chess.parse_square(uci[2:4])
    promo = _CH_PROMO[uci[4]] if len(uci) > 4 else 0
    return BBMove(frm | (to << 6) | (promo << 12))


class BBoard:
    """python-chess ``Board`` stand-in over a ``bitboard.py`` state array."""

    __slots__ = ("_moves", "_scratch", "_undo", "bb", "mb")

    def __init__(self, bb: np.ndarray, mb: np.ndarray) -> None:
        self.bb = bb
        self.mb = mb
        self._moves = np.zeros(MAX_MOVES, np.int64)
        self._scratch = np.zeros(MAX_MOVES, np.int64)
        self._undo: list[tuple[int, int, int]] = []

    @classmethod
    def from_fen(cls, fen: str) -> BBoard:
        bb, mb = board_from_fen(fen)
        return cls(bb, mb)

    @classmethod
    def from_board(cls, board: chess.Board) -> BBoard:
        return cls.from_fen(board.fen())

    def fen(self) -> str:
        return _bb.state_to_fen(self.bb, self.mb)

    # -- scalars ------------------------------------------------------------
    @property
    def turn(self) -> bool:
        return bool(self.bb[STM] == 0)  # True == chess.WHITE

    @property
    def halfmove_clock(self) -> int:
        return int(self.bb[HALFMOVE])

    @property
    def fullmove_number(self) -> int:
        return int(self.bb[_bb.FULLMOVE])

    # -- piece masks ------------------------------------------------------------
    @property
    def occupied_co(self) -> tuple[int, int]:
        # indexed by colour bool: chess.WHITE is True == 1, chess.BLACK is False == 0
        return (int(self.bb[OCC_B]), int(self.bb[OCC_W]))

    @property
    def pawns(self) -> int:
        return int(self.bb[0] | self.bb[6])

    @property
    def knights(self) -> int:
        return int(self.bb[1] | self.bb[7])

    @property
    def bishops(self) -> int:
        return int(self.bb[2] | self.bb[8])

    @property
    def rooks(self) -> int:
        return int(self.bb[3] | self.bb[9])

    @property
    def queens(self) -> int:
        return int(self.bb[4] | self.bb[10])

    @property
    def kings(self) -> int:
        return int(self.bb[5] | self.bb[11])

    def pieces_mask(self, piece_type: int, color: bool) -> int:
        base = 0 if color else 6
        return int(self.bb[base + piece_type - 1])

    def king(self, color: bool) -> int | None:
        k = self.bb[5] if color else self.bb[11]
        if k == 0:
            return None
        return int(lsb(k))

    def piece_type_at(self, sq: int) -> int | None:
        code = int(self.mb[sq])
        return None if code == 0 else ((code - 1) % 6) + 1

    def color_at(self, sq: int) -> bool | None:
        code = int(self.mb[sq])
        return None if code == 0 else code <= 6

    def attacks_mask(self, sq: int) -> int:
        code = int(self.mb[sq])
        if code == 0:
            return 0
        pt = ((code - 1) % 6) + 1
        occ = self.bb[OCC]
        if pt == 1:
            return int(_bb.PAWN_ATK[0 if code <= 6 else 1, sq])
        if pt == 2:
            return int(KNIGHT_ATK[sq])
        if pt == 3:
            return int(bishop_attacks(sq, occ))
        if pt == 4:
            return int(rook_attacks(sq, occ))
        if pt == 5:
            return int(queen_attacks(sq, occ))
        return int(KING_ATK[sq])

    # -- predicates ------------------------------------------------------------
    def is_check(self) -> bool:
        return bool(in_check(self.bb, int(self.bb[STM])))

    def is_capture(self, move: BBMove) -> bool:
        return self.mb[move.to_square] != 0 or ((move.v >> 15) & 0x7) == FLAG_EP

    def is_insufficient_material(self) -> bool:
        return self._insufficient(chess.WHITE) and self._insufficient(chess.BLACK)

    def _insufficient(self, color: bool) -> bool:
        own = self.occupied_co[color]
        if own & (self.pawns | self.rooks | self.queens):
            return False
        if own & self.knights:
            other = self.occupied_co[not color]
            return bin(own).count("1") <= 2 and not (other & ~self.kings & ~self.queens & _ALL)
        if own & self.bishops:
            bishops = self.bishops
            same = (not (bishops & BB_DARK_SQUARES)) or (not (bishops & BB_LIGHT_SQUARES))
            return same and not self.pawns and not self.knights
        return True

    # -- move generation ------------------------------------------------------
    @property
    def legal_moves(self) -> list[BBMove]:
        n = gen_legal(self.bb, self.mb, self._moves, self._scratch)
        buf = self._moves
        return [BBMove(int(buf[i])) for i in range(n)]

    # -- hashing ------------------------------------------------------------
    def _transposition_key(self) -> int:
        return int(zobrist(self.bb))

    # -- make / unmake ------------------------------------------------------
    def push(self, move: BBMove) -> None:
        if move is NULL_MOVE:
            old_ep = int(self.bb[EP])
            self.bb[EP] = np.uint64(NO_EP)
            self.bb[STM] = np.uint64(1 - int(self.bb[STM]))
            self._undo.append((1, 0, old_ep))
            return
        undo = int(make_move(self.bb, self.mb, move.v))
        self._undo.append((0, move.v, undo))

    def pop(self) -> None:
        kind, v, info = self._undo.pop()
        if kind == 1:
            self.bb[STM] = np.uint64(1 - int(self.bb[STM]))
            self.bb[EP] = np.uint64(info)
            return
        unmake_move(self.bb, self.mb, v, np.int64(info))
