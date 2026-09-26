"""The submission entrypoint. The platform imports this file and calls get_move.

Import time runs once per game inside a 60 second budget before the clock starts, so the
search tables and a throwaway warm-up search happen out here. `get_move` itself only
parses the position, asks the searcher for a move, and -- whatever happens -- returns a
legal UCI move, because a crash or an illegal move is an instant loss.
"""

import os
import time

_INIT_T0 = time.monotonic()   # process start, for the init failsafe below

import warnings

import chess

# numba emits a "cannot cache" NumbaWarning to stderr per magic-globals function
# on its first compile (~13 lines, all from the warm-up below), and the
# platform's validation flags any stderr write. Suppress that category before
# the first jitted call -- a genuine compilation failure raises an exception
# (caught below and in get_move), it does not warn.
try:
    from numba.core.errors import NumbaWarning

    warnings.filterwarnings("ignore", category=NumbaWarning)
except Exception:  # numba layout changed / absent -- not worth failing import
    pass

from jsearch import Searcher

_SEARCHER = Searcher()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return "0000"

    try:
        legal = list(board.legal_moves)
    except (ValueError, AttributeError):
        return "0000"
    if not legal:
        return "0000"
    fallback = legal[0].uci()

    try:
        budget_ms = time_left_ms if isinstance(time_left_ms, int) and time_left_ms > 0 else 0
        move = _SEARCHER.search(board, budget_ms)
        candidate = chess.Move.from_uci(move)
        if candidate in board.legal_moves:
            return move
    except Exception as error:  # never let the search lose us the game
        print(f"search failed, playing fallback: {error!r}")
    return fallback


# Warm the code paths (python-chess move generation, the search recursion, the eval) so
# the first real move does not pay for it, then wipe the state the warm-up produced.
#
# INIT FAILSAFE (2026-09-11). This one search compiles every jitted kernel --
# measured 32.6 s of a 33.0 s cold init on a SINGLE core, which is what the
# platform gives us. LLVM compilation inside it cannot be interrupted by any
# Python-level deadline, so whether to START it is the only control we have.
# A submission was disqualified with "no ready line within the 90 s init
# budget"; if imports were ever unexpectedly slow we skip the warm-up and pay
# the compile on move 1 instead. A slow first move costs clock. A missing
# ready line costs the entire game.
_INIT_WARM_START_BY = float(os.environ.get("AICHESS_INIT_START_BY", "20"))
if time.monotonic() - _INIT_T0 < _INIT_WARM_START_BY:
    try:
        _SEARCHER.search(chess.Board(), 300)
    except Exception as _warm_error:
        print(f"warm-up failed: {_warm_error!r}")
_SEARCHER.reset()
