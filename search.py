"""Iterative-deepening negamax with alpha-beta pruning.

`Searcher` keeps the state that is worth carrying between moves in one game: the
transposition table and the set of positions the game has actually visited (so a won
position can be shuffled to a threefold draw is *avoided*, and a lost one is not walked
into). Killers and the history table are per-search.

The public entry point is `Searcher.search(board, time_left_ms)`, which always returns a
legal UCI move: iterative deepening keeps a best move from the last completed depth, and
a hard deadline checked inside the recursion raises `_TimeUp` to unwind cleanly.
"""

import os
import time
from typing import Any

import chess

from evaluate import PIECE_VALUE, evaluate

# --- board backend --------------------------------------------------------
# The search is written against the python-chess `Board` / `Move` API. By
# default the same calls route to the numba bitboard movegen (bitboard.py)
# through a thin adapter (bbwrap.py) and to a jitted evaluation (evaluate_bb.py):
# it is worth +198 Elo at 10+0.1 and +107 at 120+0.5 over the python-chess
# backend (internal SPRT, 2026-09-02). The search *logic* -- negamax,
# alpha-beta, PVS, the TT, quiescence, null-move, LMR -- is byte-for-byte the
# same either way; only the representation underneath changes. Set
# AICHESS_BITBOARD=0 to fall back to the python-chess backend.
USE_BITBOARD = os.environ.get("AICHESS_BITBOARD", "1") != "0"

# The 2026-09-02 search changes, individually switchable:
#   - threefold (ON): a position only draws on a real third occurrence, not the
#     first repeat from the game history. A pure correctness fix; internal SPRT
#     at 10+0.1 put it at parity (-12 +/- 43 Elo) with the old twofold rule.
#   - contempt (OFF): draws score AICHESS_CONTEMPT cp below equal. Its payoff is
#     against weaker opponents; self-play SPRT can only see the cost, and did
#     (contempt 8 + nullverify together: -48 +/- 36 Elo at 10+0.1). Left at 0.
#   - nullverify (OFF): re-check a null-move fail-high with a real reduced-depth
#     search (guards zugzwang). Node cost with no measurable payoff at our
#     depths; kept behind the flag for very long time controls.
# AICHESS_LEGACY_SEARCH=1 forces all three off (i.e. the pre-2026-09-02 search).
_LEGACY_SEARCH = os.environ.get("AICHESS_LEGACY_SEARCH", "0") == "1"
_USE_THREEFOLD = not _LEGACY_SEARCH and os.environ.get("AICHESS_THREEFOLD", "1") != "0"
_USE_NULLVERIFY = not _LEGACY_SEARCH and os.environ.get("AICHESS_NULLVERIFY", "0") != "0"

# Test-only knobs used by tools/bb_integration.py to prove the two board
# backends compute the *same* chess:
#   AICHESS_NOPRUNE  -- disable the order-dependent heuristics (null-move, LMR,
#                       quiescence delta pruning) so the search is exact
#                       alpha-beta; a different move-generation order otherwise
#                       feeds the reductions differently and scores drift a few cp.
#   AICHESS_NOTT     -- disable the transposition table. This engine's TT stores
#                       lower/upper bounds, which makes even "exact" search
#                       scores slightly approximate; the two backends' Zobrist
#                       keys bucket positions differently, so with the TT on
#                       those approximations land on different positions. With
#                       both knobs set the backends must agree bit-for-bit.
_EXACT_SEARCH = os.environ.get("AICHESS_NOPRUNE", "0") == "1"
_USE_TT = os.environ.get("AICHESS_NOTT", "0") != "1"

# The backend seam is deliberately untyped (`Any`): one side speaks python-chess
# `Board`/`Move`, the other speaks bbwrap `BBoard`/`BBMove`, and the search body
# is written to the common surface. mypy checks the search against `chess.Board`
# (the `else` branch); the bitboard branch is covered by tools/bb_integration.py.
_NULL: Any
_move_from_uci: Any
_evaluate: Any
_to_board: Any

if USE_BITBOARD:  # pragma: no cover - exercised via the env var in tests / SPRT
    from bbwrap import NULL_MOVE as _NULL
    from bbwrap import BBoard
    from bbwrap import move_from_uci as _move_from_uci
    from evaluate_bb import evaluate_bb as _evaluate

    def _to_board(board: Any) -> Any:
        return board if isinstance(board, BBoard) else BBoard.from_board(board)

else:
    _NULL = chess.Move.null()
    _move_from_uci = chess.Move.from_uci
    _evaluate = evaluate

    def _to_board(board: Any) -> Any:
        return board

MATE = 1_000_000
MATE_THRESHOLD = MATE - 1000
_INF = 1 << 30

# Contempt: a draw scores this many centipawns below equal for the side to move,
# so a winning engine plays on instead of repeating. Default 0 -- see the note
# above; set AICHESS_CONTEMPT (e.g. 8) for a tournament with weaker opponents.
_CONTEMPT = 0 if _LEGACY_SEARCH else int(os.environ.get("AICHESS_CONTEMPT", "0"))
_DRAW_SCORE = -_CONTEMPT

MAX_DEPTH = 64
MAX_PLY = 128
MAX_QPLY = 8

_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT_LIMIT = 1_500_000

# Move-ordering score bands.
_TT_BONUS = 30_000_000
_CAPTURE_BONUS = 20_000_000
_PROMO_BONUS = 18_000_000
_KILLER_BONUS = 17_000_000

_MVV_LVA_VICTIM = 10
_NON_PAWN_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

# (depth, score, flag, best move uci or None)
_TTEntry = tuple[int, int, int, str | None]


def _key(board: chess.Board) -> int:
    """Position key for the TT and repetition tracking.

    `_transposition_key()` is python-chess's own full board-state tuple (piece
    bitboards + side to move + castling rights + a real en-passant square) -- the
    same value it uses for threefold detection. Hashing it to a 64-bit int keeps
    the tables small; a hash collision at most mis-scores one node, which a
    heuristic search already tolerates from its depth/eval approximations.

    This replaces `polyglot.zobrist_hash`, which at ~19 us/call was the single
    biggest per-node cost in the old code (`_transposition_key` is ~24x cheaper).
    """
    return hash(board._transposition_key())


class _TimeUp(Exception):
    """Raised deep in the recursion when the hard deadline passes."""


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[int, _TTEntry] = {}
        self.seen: dict[int, int] = {}
        self.killers: list[list[chess.Move | None]] = []
        self.history: dict[tuple[bool, int, int], int] = {}
        self.nodes = 0
        self._deadline = 0.0
        self._path: list[int] = []
        self._checkpoint: tuple[int, chess.Move] | None = None
        # Last completed search, for a UCI adapter's info line. Not used by the agent.
        self.last_depth = 0
        self.last_score = 0
        self.last_move = "0000"

    def reset(self) -> None:
        """Forget everything. Called once after the import-time warm-up."""
        self.tt.clear()
        self.seen.clear()
        self.history.clear()

    # -- public -----------------------------------------------------------------

    def search(self, board: chess.Board, time_left_ms: int, max_depth: int = MAX_DEPTH) -> str:
        board = _to_board(board)
        legal = list(board.legal_moves)
        if not legal:
            return "0000"
        best_move = legal[0]

        self._note_position(board)
        budget_ms = self._budget(board, time_left_ms)
        self._deadline = time.monotonic() + budget_ms / 1000.0
        self.nodes = 0
        self._path = []
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {key: value // 2 for key, value in self.history.items()}
        if len(self.tt) > _TT_LIMIT:
            self.tt.clear()

        root_key = _key(board)
        best_score = 0
        best_depth = 0

        for depth in range(1, min(max_depth, MAX_DEPTH) + 1):
            self._checkpoint = None
            try:
                score, move = self._search_root(board, depth, root_key)
            except _TimeUp:
                if self._checkpoint is not None:
                    best_score, best_move = self._checkpoint
                    best_depth = depth
                break
            best_move, best_score, best_depth = move, score, depth
            if time.monotonic() >= self._deadline:
                break
            if abs(score) >= MATE_THRESHOLD:
                break

        self.last_depth = best_depth
        self.last_score = best_score
        self.last_move = best_move.uci()
        return best_move.uci()

    # -- time -----------------------------------------------------------------

    def _budget(self, board: chess.Board, time_left_ms: int) -> float:
        if time_left_ms <= 600:
            return 40.0
        move_number = board.fullmove_number
        if move_number < 16:
            moves_to_go = 26
        elif move_number < 32:
            moves_to_go = 22
        else:
            moves_to_go = 18
        budget = time_left_ms / moves_to_go
        if time_left_ms > 4000:
            budget += 250.0  # part of the 0.5s increment we get back after moving
        budget = min(budget, time_left_ms * 0.4, time_left_ms - 300.0)
        return max(budget, 30.0)

    def _out_of_time(self) -> bool:
        self.nodes += 1
        return self.nodes % 2048 == 0 and time.monotonic() >= self._deadline

    # -- repetition tracking ------------------------------------------------------

    def _note_position(self, board: chess.Board) -> None:
        key = _key(board)
        self.seen[key] = self.seen.get(key, 0) + 1

    def _is_draw(self, board: chess.Board, key: int) -> bool:
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return True
        if not _USE_THREEFOLD:
            return self._path.count(key) + self.seen.get(key, 0) >= 1
        # Within the current search line a single repeat is a draw: the side to
        # move can just keep repeating to force the third occurrence. (The root
        # key is on `_path`, so a line that walks back to the root counts too.)
        on_path = self._path.count(key)
        if on_path >= 1:
            return True
        # From the game history, though, one earlier occurrence is only a
        # twofold -- it takes a genuine third occurrence to be a draw. This is
        # the fix: previously any position played once before scored 0, so a
        # winning engine would happily walk back into it.
        return self.seen.get(key, 0) >= 2

    # -- ordering ---------------------------------------------------------------

    @staticmethod
    def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        victim_value = PIECE_VALUE[victim] if victim is not None else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_type_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
        score = _MVV_LVA_VICTIM * victim_value - attacker_value
        if move.promotion is not None:
            score += PIECE_VALUE[move.promotion]
        return score

    def _order(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        ply: int,
        tt_move: chess.Move | None,
        captures: frozenset[chess.Move],
    ) -> list[chess.Move]:
        killers = self.killers[ply] if ply < MAX_PLY else []
        side = board.turn

        def key(move: chess.Move) -> int:
            if move == tt_move:
                return _TT_BONUS
            if move in captures:
                return _CAPTURE_BONUS + self._mvv_lva(board, move)
            if move.promotion is not None:
                return _PROMO_BONUS + PIECE_VALUE[move.promotion]
            if move in killers:
                return _KILLER_BONUS - killers.index(move)
            return self.history.get((side, move.from_square, move.to_square), 0)

        return sorted(moves, key=key, reverse=True)

    def _add_killer(self, ply: int, move: chess.Move) -> None:
        if ply >= MAX_PLY:
            return
        slot = self.killers[ply]
        if move != slot[0]:
            slot[1] = slot[0]
            slot[0] = move

    def _add_history(self, side: chess.Color, move: chess.Move, depth: int) -> None:
        entry = (side, move.from_square, move.to_square)
        self.history[entry] = self.history.get(entry, 0) + depth * depth

    # -- mate-score packing for the TT -----------------------------------------

    @staticmethod
    def _to_tt(score: int, ply: int) -> int:
        if score > MATE_THRESHOLD:
            return score + ply
        if score < -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _from_tt(score: int, ply: int) -> int:
        if score > MATE_THRESHOLD:
            return score - ply
        if score < -MATE_THRESHOLD:
            return score + ply
        return score

    def _tt_move(self, key: int) -> Any:
        entry = self.tt.get(key) if _USE_TT else None
        if entry is None or entry[3] is None or entry[3] == "0000":
            return None
        try:
            return _move_from_uci(entry[3])
        except (ValueError, KeyError):
            return None

    # -- search ---------------------------------------------------------------

    def _search_root(
        self, board: chess.Board, depth: int, root_key: int
    ) -> tuple[int, chess.Move]:
        alpha, beta = -_INF, _INF
        legal = list(board.legal_moves)
        captures = frozenset(m for m in legal if board.is_capture(m))
        moves = self._order(board, legal, 0, self._tt_move(root_key), captures)
        best_score = -_INF
        best_move = moves[0]
        self._path.append(root_key)
        try:
            for index, move in enumerate(moves):
                board.push(move)
                try:
                    if index == 0:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
                    else:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                        if alpha < score < beta:
                            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, True)
                finally:
                    board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
                    self._checkpoint = (score, move)
                if score > alpha:
                    alpha = score
        finally:
            self._path.pop()

        if _USE_TT:
            self.tt[root_key] = (depth, self._to_tt(best_score, 0), _EXACT, best_move.uci())
        return best_score, best_move

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool,
    ) -> int:
        if self._out_of_time():
            raise _TimeUp

        key = _key(board)
        if self._is_draw(board, key):
            return _DRAW_SCORE

        alpha_orig = alpha
        entry = self.tt.get(key) if _USE_TT else None
        tt_move: chess.Move | None = None
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_uci = entry
            if tt_uci is not None and tt_uci != "0000":
                try:
                    tt_move = _move_from_uci(tt_uci)
                except (ValueError, KeyError):
                    tt_move = None
            if tt_depth >= depth:
                packed = self._from_tt(tt_score, ply)
                if tt_flag == _EXACT:
                    return packed
                if tt_flag == _LOWER and packed > alpha:
                    alpha = packed
                elif tt_flag == _UPPER and packed < beta:
                    beta = packed
                if alpha >= beta:
                    return packed

        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else _DRAW_SCORE

        if (
            not _EXACT_SEARCH
            and can_null
            and not in_check
            and depth >= 3
            and beta < MATE_THRESHOLD
            and self._has_non_pawn(board, board.turn)
        ):
            board.push(_NULL)
            try:
                score = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            finally:
                board.pop()
            if score >= beta:
                # Verification search: zugzwang is the null move's blind spot --
                # there, having to move is a disadvantage, so "pass and still be
                # fine" is a lie. Re-search the real position at reduced depth
                # with null disabled; only trust the cutoff if it also fails high.
                if not _USE_NULLVERIFY or depth < 4:
                    return beta
                verify = self._negamax(board, depth - 3, beta - 1, beta, ply, False)
                if verify >= beta:
                    return beta

        captures = frozenset(m for m in moves if board.is_capture(m))
        ordered = self._order(board, moves, ply, tt_move, captures)
        best_score = -_INF
        best_move: chess.Move | None = None
        self._path.append(key)
        try:
            for index, move in enumerate(ordered):
                is_quiet = move not in captures and move.promotion is None
                board.push(move)
                try:
                    reduction = 0
                    if (
                        not _EXACT_SEARCH
                        and is_quiet
                        and depth >= 3
                        and index >= 4
                        and not board.is_check()
                    ):
                        reduction = 1
                    if index == 0:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, True)
                    else:
                        score = -self._negamax(
                            board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True
                        )
                        if score > alpha and reduction:
                            score = -self._negamax(
                                board, depth - 1, -alpha - 1, -alpha, ply + 1, True
                            )
                        if alpha < score < beta:
                            score = -self._negamax(
                                board, depth - 1, -beta, -alpha, ply + 1, True
                            )
                finally:
                    board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if is_quiet:
                        self._add_killer(ply, move)
                        self._add_history(board.turn, move, depth)
                    break
        finally:
            self._path.pop()

        if best_score <= alpha_orig:
            flag = _UPPER
        elif best_score >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        if _USE_TT:
            stored = best_move.uci() if best_move is not None else None
            self.tt[key] = (depth, self._to_tt(best_score, ply), flag, stored)
        return best_score

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        if self._out_of_time():
            raise _TimeUp
        if board.is_insufficient_material():
            return _DRAW_SCORE

        in_check = board.is_check()
        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -_INF
            stand_pat = -_INF
        else:
            stand_pat = int(_evaluate(board))
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            if ply >= MAX_PLY - 1 or ply >= MAX_DEPTH + MAX_QPLY:
                return stand_pat
            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]
            moves.sort(key=lambda move: self._mvv_lva(board, move), reverse=True)

        for move in moves:
            if not _EXACT_SEARCH and not in_check and move.promotion is None:
                victim = board.piece_type_at(move.to_square)
                gain = PIECE_VALUE[victim] if victim is not None else PIECE_VALUE[chess.PAWN]
                if stand_pat + gain + 200 < alpha:
                    continue
            board.push(move)
            try:
                score = -self._quiesce(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    @staticmethod
    def _has_non_pawn(board: chess.Board, color: chess.Color) -> bool:
        occ = board.occupied_co[color]
        return bool((board.knights | board.bishops | board.rooks | board.queens) & occ)
