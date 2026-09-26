# BitterBOT

A chess engine for the [AI Chessathon](https://aichessathon.com), evaluating positions with
**H256** — a from-scratch NNUE (Efficiently Updatable Neural Network) — behind a
[numba](https://numba.pydata.org)-JIT-compiled bitboard search core.

By Ayush Mantri ([@bit4er](https://github.com/bit4er)).

This is the exact build submitted to the competition: a single entry point,
`get_move(fen, time_left_ms) -> str`, run by the platform in a fresh process per game
over a JSON line protocol at 120 s + 0.5 s/move, on one CPU core with 2 GB of RAM and
no network. Everything here is built to those limits.

## Quick start

```
pip install -r requirements.txt
python play.py                  # you play White
python play.py --color black    # you play Black
```

Enter moves in SAN (`Nf3`, `exd5`, `O-O`) or UCI (`g1f3`, `e4d5`, `e1g1`). The engine's
first move takes longer than the rest — that's the JIT compiling the search kernels,
a one-time cost per process.

## Architecture

**Evaluation (`nnue_*.py`)** — H256: a 3072→256→1 network, 4 king buckets,
horizontally mirrored, CReLU activations, trained from self-play and Stockfish-labelled
data. The tail's raw output *is* the score — no hand-crafted evaluation blend, no
fallback path.

**Search (`jsearch.py`, `bitboard.py`)** — iterative-deepening negamax with alpha-beta
and principal-variation search, a persistent transposition table, MVV-LVA + killer +
history move ordering, quiescence search at the leaves, reverse futility pruning, null-move
pruning, and check extensions. The bitboard core (`bitboard.py`, `magic_numbers.py`) is
JIT-compiled with numba for speed.

**Entry point (`agent.py`)** — parses the position, asks the searcher for a move, and
warms the JIT-compiled kernels during the init budget so the first real move in a game
is never paying a cold-compile tax.

## Repo layout

```
play.py                   terminal game vs the engine (start here)
agent.py                  competition entry point
jsearch.py                search
bitboard.py, magic_numbers.py, bbwrap.py     bitboard core
evaluate.py, evaluate_bb.py, engine_eval.py   evaluation dispatch
nnue_*.py                 H256 NNUE (accumulator, tail, training-time variants)
search.py                 earlier classical-eval search kept for reference
weights/h256_competition.npz    trained H256 weights
```
