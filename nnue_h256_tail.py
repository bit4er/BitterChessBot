"""H256 tail: concat(activation(white_acc), activation(black_acc)) -> 1
scalar output, no intermediate dense layer -- per the task's own
instruction ("Do NOT add 32->32 or other dense hidden layers initially.
The objective is maximum knowledge in the sparse feature transformer with
an extremely cheap tail."), matching the general shape Pawnstar and Luna
both use ("concat[side-to-move | opponent] -> 1 output").

Two activations, both benchmarked (see docs/NNUE_H256_4BUCKET_MIRROR_1P7M.md
for the actual numbers):

  CReLU:  x = clip(acc, 0, ACT_MAX)
  SCReLU: x = clip(acc, 0, ACT_MAX); x = x*x   (squared clipped ReLU --
          strictly more expressive per weight at training time; Pawnstar
          uses this)

Pure int32 arithmetic throughout, same discipline as nnue_v5_tail.py --
no float anywhere. Overflow safety is proven the same way that module
does it (worst-case magnitude vs int32 range), not assumed.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def _h256_int32_headroom(h1: int, act_max: int, w_max: int, screlu: bool) -> int:
    """Worst-case |term| summed over 2*h1 terms of the single output dot
    product. SCReLU squares the activation before the weight multiply, so
    its per-term bound is act_max^2 * w_max instead of act_max * w_max."""
    per_term = (act_max * act_max * w_max) if screlu else (act_max * w_max)
    return per_term * 2 * h1


def build_dummy_weights(h1: int, seed: int) -> dict:
    """Random weights, correctly typed/shaped -- speed-only phase, DO NOT
    TRAIN yet. ow/ob are the single output layer's weights (shape (2*h1,)
    int16) and bias (shape (1,) int32) -- no hidden dense layer."""
    rng = np.random.default_rng(seed)
    ow = rng.integers(-127, 128, size=(2 * h1,), dtype=np.int32).astype(np.int16)
    ob = rng.integers(-2000, 2000, size=1, dtype=np.int32)
    return {"ow": ow, "ob": ob}


ACT_MAX = 127
SHIFT = 6


@njit(cache=True, inline="always")
def h256_tail_crelu(white_acc: np.ndarray, black_acc: np.ndarray, stm: int,
                    ow: np.ndarray, ob: np.ndarray, shift: int, act_max: int) -> int:
    """CReLU tail: out = ob + sum_j ow[j]*clip(first[j]) + sum_j ow[h1+j]*clip(second[j]).
    Fused -- no temporary activation array, one pass, matching the task's
    'fused tail' requirement (Priority 5)."""
    h1 = white_acc.shape[0]
    if stm == 0:
        src_first, src_second = white_acc, black_acc
    else:
        src_first, src_second = black_acc, white_acc
    # MEASURED-FASTEST FORM. The per-element `if/elif` clip looks like it
    # should lose to branchless min/max, and the speed campaign tried exactly
    # that -- it was 5.34% SLOWER over an interleaved A/B (0/4 pairwise wins,
    # tools/h256_ab_nps.sh). Reason: accumulator values almost always land
    # inside [0, act_max], so these branches are near-perfectly predicted and
    # effectively free, whereas min/max costs two real instructions on every
    # one of the 512 elements. Neither form vectorizes, so the "branchless"
    # rewrite bought nothing and paid for it. Keep the branches.
    out = ob[0]
    for j in range(h1):
        fv = src_first[j] >> shift
        if fv < 0:
            fv = 0
        elif fv > act_max:
            fv = act_max
        out += ow[j] * fv
        sv = src_second[j] >> shift
        if sv < 0:
            sv = 0
        elif sv > act_max:
            sv = act_max
        out += ow[h1 + j] * sv
    return int(out)


@njit(cache=True, inline="always")
def h256_tail_crelu_branchless(white_acc: np.ndarray, black_acc: np.ndarray, stm: int,
                               ow: np.ndarray, ob: np.ndarray, shift: int,
                               act_max: int) -> int:
    """REJECTED speed experiment, kept as the A/B control (select with
    ``AICHESS_H256_TAIL_IMPL=branchless``).

    Branchless min/max clip plus two independent accumulator chains. The
    hypothesis was that the shipped kernel's per-element ``if/elif`` blocked
    LLVM's vectorizer. It does not help: measured **-5.34%** against the
    branch form over an interleaved A/B (0/4 pairwise wins). The clip
    branches are near-perfectly predicted because accumulator values almost
    always sit inside [0, act_max], so they cost ~nothing, while min/max adds
    two real instructions per element across 512 elements. Neither form
    vectorizes under numba here.

    Bit-identical to ``h256_tail_crelu`` (tools/h256_tail_parity.py, 4014
    adversarial cases incl. exact clip boundaries) -- retained so the null
    result stays reproducible rather than becoming folklore.
    """
    h1 = white_acc.shape[0]
    if stm == 0:
        src_first, src_second = white_acc, black_acc
    else:
        src_first, src_second = black_acc, white_acc
    acc1 = 0
    acc2 = 0
    for j in range(h1):
        fv = min(max(src_first[j] >> shift, 0), act_max)
        acc1 += ow[j] * fv
        sv = min(max(src_second[j] >> shift, 0), act_max)
        acc2 += ow[h1 + j] * sv
    return int(ob[0] + acc1 + acc2)


@njit(cache=True, inline="always")
def h256_tail_screlu(white_acc: np.ndarray, black_acc: np.ndarray, stm: int,
                     ow: np.ndarray, ob: np.ndarray, shift: int, act_max: int) -> int:
    """SCReLU tail: same as CReLU but activation is squared before the
    weight multiply (clip first, then square -- overflow-safe since
    act_max=127 -> 127^2=16,129, still tiny relative to int32 range)."""
    h1 = white_acc.shape[0]
    if stm == 0:
        src_first, src_second = white_acc, black_acc
    else:
        src_first, src_second = black_acc, white_acc
    out = ob[0]
    for j in range(h1):
        fv = src_first[j] >> shift
        if fv < 0:
            fv = 0
        elif fv > act_max:
            fv = act_max
        out += ow[j] * (fv * fv)
        sv = src_second[j] >> shift
        if sv < 0:
            sv = 0
        elif sv > act_max:
            sv = act_max
        out += ow[h1 + j] * (sv * sv)
    return int(out)
