"""Classical evaluation plus the independently trained v4.1 residual."""

import os
from pathlib import Path

import numpy as np
from numba import njit

import evaluate_bb
import nnue_eval_v4

_SCALE = np.float32(os.environ.get("AICHESS_NNUE_RESIDUAL_SCALE", "1.0"))
# Zero means no cap; positive values clamp the scaled correction in centipawns.
_CAP = np.int32(os.environ.get("AICHESS_NNUE_RESIDUAL_CAP", "0"))


def load_model(path: str | Path) -> dict:
    info = nnue_eval_v4.load_model(path)
    return {**info, "residual_scale": float(_SCALE), "residual_cap": int(_CAP)}


@njit(cache=False, fastmath=True)
def eval_state(bb, ev):
    classical = evaluate_bb.eval_state(bb, ev)
    correction = int(round(_SCALE * nnue_eval_v4.eval_state(bb, ev)))
    if _CAP > 0:
        if correction < -_CAP:
            correction = -_CAP
        elif correction > _CAP:
            correction = _CAP
    return classical + correction
