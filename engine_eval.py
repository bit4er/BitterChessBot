"""Eval backend switch for the search.

Default: the classical hand eval (``evaluate_bb``). If ``AICHESS_NNUE`` points at
an exported net (``nnue/model/nnue-*.npz``), the search uses the njit NNUE
full-recompute eval instead. ``EVAL_BUNDLE`` is still exported (the NNUE
``eval_state`` ignores it) so ``jsearch`` needs only to change its import line.

The export's own input-feature count picks the backend: 768 -> flat v1/v2
scheme (``nnue_eval.py``), ``nnue.kingbucket.N_FEATURES`` (12288) -> the
king-bucketed v3 scheme (``nnue_eval_v3.py``). Same full-recompute-per-call
approach either way -- v3 just offsets each row into the active king-bucket
block instead of the flat 0..767 range.

    AICHESS_NNUE=nnue/model/nnue-v2.npz  python -m tools.uci_adapter
"""

from __future__ import annotations

import os

import numpy as np

from evaluate_bb import EVAL_BUNDLE  # noqa: F401  (re-exported)

_NNUE_PATH = os.environ.get("AICHESS_NNUE", "").strip()
_RESIDUAL_DISABLED = (
    os.environ.get("AICHESS_NNUE_RESIDUAL", "0") == "1"
    and float(os.environ.get("AICHESS_NNUE_RESIDUAL_SCALE", "1.0")) == 0.0
)

if _NNUE_PATH and not _RESIDUAL_DISABLED:
    exported = np.load(_NNUE_PATH, allow_pickle=False)
    if "ft" in exported:
        if os.environ.get("AICHESS_NNUE_RESIDUAL", "0") == "1":
            import nnue_eval_residual_v41 as nnue_eval
        elif os.environ.get("AICHESS_NNUE_HYBRID", "0") == "1":
            import nnue_eval_hybrid as nnue_eval
        else:
            import nnue_eval_v4 as nnue_eval
        n_features = 6144
    else:
        n_features = int(exported["l1_w"].shape[1])
    from nnue.kingbucket import N_FEATURES as _KB_FEATURES

    if "ft" in exported:
        pass
    elif n_features == _KB_FEATURES:
        import nnue_eval_v3 as nnue_eval
    else:
        import nnue_eval

    nnue_eval.load_model(_NNUE_PATH)
    eval_state = nnue_eval.eval_state
    BACKEND = f"nnue:{os.path.basename(_NNUE_PATH)}"
else:
    from evaluate_bb import eval_state  # noqa: F401

    BACKEND = "classical"
