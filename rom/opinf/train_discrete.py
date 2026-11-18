"""
rom.opinf.train_discrete – Training and regularizer tuning for discrete OpInf models.

Fits autonomous discrete models (a_{k+1} = c + Aa_k + H(a_k⊗a_k)) using properly
aligned state/next-state pairs with trajectory boundary isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opinf

from rom.opinf.derivatives import prepare_discrete_pairs
from rom.opinf.model_factory import build_model, compute_operator_norms, extract_operator_entries
from rom.opinf.predict import assess_stability, compute_rollout_error, predict_discrete


@dataclass
class DiscreteTrainingResult:
    """Result of discrete OpInf training and regularization tuning."""

    model: opinf.models.DiscreteModel
    best_regularizer: float
    sweep_history: List[Dict[str, Any]] = field(default_factory=list)
    operator_norms: Dict[str, float] = field(default_factory=dict)
    train_residual: float = 0.0
    val_rollout_error: Optional[float] = None
    runtime_seconds: float = 0.0


def train_discrete_model(
    train_trajectories: Sequence[np.ndarray],
    operators: str = "cAH",
    regularizers: Optional[Sequence[float]] = None,
    val_trajectories: Optional[Sequence[np.ndarray]] = None,
) -> DiscreteTrainingResult:
    """Train an autonomous discrete OpInf model with regularization tuning.

    Parameters
    ----------
    train_trajectories : sequence of np.ndarray
        List of reduced training state matrices, each of shape ``(r, K_i)``.
    operators : str, optional
        Operator formulation (default: 'cAH').
    regularizers : sequence of float, optional
        Candidate L2 regularizers (default: [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).
    val_trajectories : sequence of np.ndarray, optional
        Held-out validation trajectories for selecting regularizers by dynamical prediction accuracy.

    Returns
    -------
    DiscreteTrainingResult
    """
    t_start = time.perf_counter()

    if regularizers is None:
        regularizers = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # 1. Prepare aligned pairs per trajectory
    states_k, states_next = prepare_discrete_pairs(train_trajectories)
    denom = float(np.linalg.norm(states_next, "fro"))

    sweep_history: List[Dict[str, Any]] = []
    best_model: Optional[opinf.models.DiscreteModel] = None
    best_reg = regularizers[0]
    best_score = float("inf")

    has_validation = val_trajectories is not None and len(val_trajectories) > 0

    for reg in regularizers:
        model = build_model(
            model_type="discrete",
            operators=operators,
            regularizer=reg if reg > 0 else None,
        )

        model.fit(states_k, states_next)

        # One-step training prediction
        pred_next = model.rhs(state=states_k)
        res_norm = float(np.linalg.norm(states_next - pred_next, "fro"))
        train_res = res_norm / denom if denom > 0 else 0.0

        op_norms = compute_operator_norms(model)

        val_err = None
        is_stable_val = True

        if has_validation:
            val_errors = []
            for v_traj in val_trajectories:
                num_steps = v_traj.shape[1]
                a0 = v_traj[:, 0]
                a_pred = predict_discrete(model, a0, num_steps=num_steps)
                stab = assess_stability(a_pred, reference_scale=float(np.max(np.abs(v_traj))))
                if not stab["is_stable"]:
                    is_stable_val = False
                    val_errors.append(float("inf"))
                else:
                    err = compute_rollout_error(v_traj, a_pred)
                    val_errors.append(err["relative_frobenius_error"])

            val_err = float(np.mean(val_errors)) if is_stable_val else float("inf")
            selection_score = val_err
        else:
            selection_score = train_res

        sweep_history.append({
            "regularizer": reg,
            "train_residual": train_res,
            "val_rollout_error": val_err,
            "is_stable": is_stable_val,
            "operator_norms": op_norms,
        })

        if selection_score < best_score:
            best_score = selection_score
            best_reg = reg
            best_model = model

    if best_model is None or not np.isfinite(best_score):
        best_entry = min(sweep_history, key=lambda x: x["train_residual"])
        best_reg = best_entry["regularizer"]
        best_model = build_model(
            model_type="discrete",
            operators=operators,
            regularizer=best_reg if best_reg > 0 else None,
        )
        best_model.fit(states_k, states_next)

    runtime = time.perf_counter() - t_start
    chosen_metrics = next(m for m in sweep_history if m["regularizer"] == best_reg)

    return DiscreteTrainingResult(
        model=best_model,
        best_regularizer=best_reg,
        sweep_history=sweep_history,
        operator_norms=compute_operator_norms(best_model),
        train_residual=chosen_metrics["train_residual"],
        val_rollout_error=chosen_metrics["val_rollout_error"],
        runtime_seconds=runtime,
    )
