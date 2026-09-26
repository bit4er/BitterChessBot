"""Classical evaluation plus a bounded, optionally gated NNUE-v4 residual."""

import os
from pathlib import Path

import numpy as np
from numba import njit

import evaluate_bb
import nnue_eval_v4

_LAMBDA = np.float32(os.environ.get("AICHESS_NNUE_LAMBDA", "0.2"))
_CAP = np.int32(os.environ.get("AICHESS_NNUE_CAP", "50"))
_GATE = np.int32(os.environ.get("AICHESS_NNUE_GATE", "100000"))


def load_model(path: str | Path) -> dict:
    info = nnue_eval_v4.load_model(path)
    return {**info, "lambda": float(_LAMBDA), "cap": int(_CAP), "gate": int(_GATE)}


@njit(cache=False, fastmath=True)
def eval_state(bb, ev):
    classical = evaluate_bb.eval_state(bb, ev)
    if abs(classical) > _GATE:
        return classical
    neural = nnue_eval_v4.eval_state(bb, ev)
    correction = int(round(_LAMBDA * (neural - classical)))
    if correction < -_CAP:
        correction = -_CAP
    elif correction > _CAP:
        correction = _CAP
    return classical + correction
