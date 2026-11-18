"""
rom.opinf.train_continuous – Training and regularizer tuning for continuous OpInf models.

Fits autonomous continuous models (da/dt = c + Aa + H(a⊗a)) with Tikhonov L2
regularization and performs systematic regularizer sweeps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import opinf

from rom.opinf.derivatives import estimate_trajectory_derivatives
from rom.opinf.model_factory import build_model, compute_operator_norms, extract_operator_entries
from rom.opinf.predict import assess_stability, compute_rollout_error, predict_continuous


@dataclass
class ContinuousTrainingResult:
    """Result of continuous OpInf training and regularization tuning."""

    model: opinf.models.ContinuousModel
    best_regularizer: float
    sweep_history: List[Dict[str, Any]] = field(default_factory=list)
    operator_norms: Dict[str, float] = field(default_factory=dict)
    train_residual: float = 0.0
    val_rollout_error: Optional[float] = None
    runtime_seconds: float = 0.0


def train_continuous_model(
    train_trajectories: Sequence[np.ndarray],
    train_times: Sequence[np.ndarray],
    operators: str = "cAH",
    regularizers: Optional[Sequence[float]] = None,
    val_trajectories: Optional[Sequence[np.ndarray]] = None,
    val_times: Optional[Sequence[np.ndarray]] = None,
    order: int = 2,
    integration_method: str = "BDF",
) -> ContinuousTrainingResult:
    """Train an autonomous continuous OpInf model with regularization tuning.

    Parameters
    ----------
    train_trajectories : sequence of np.ndarray
        List of reduced training state matrices, each of shape ``(r, K_i)``.
    train_times : sequence of np.ndarray
        List of time arrays corresponding to the training trajectories.
    operators : str, optional
        Operator formulation (default: 'cAH').
    regularizers : sequence of float, optional
        Candidate L2 regularizers to evaluate (default: [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).
    val_trajectories : sequence of np.ndarray, optional
        Held-out validation trajectories for selecting regularizers by dynamical prediction accuracy.
    val_times : sequence of np.ndarray, optional
        Time arrays for validation trajectories.
    order : int, optional
        Finite-difference accuracy order for estimating derivatives (default: 2).
    integration_method : str, optional
        ODE solver for validation rollout simulation (default: 'BDF').

    Returns
    -------
    ContinuousTrainingResult
    """
    t_start = time.perf_counter()

    if regularizers is None:
        regularizers = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # 1. Estimate derivatives per trajectory (strictly preserving boundary isolation)
    ddt_list, states_concat, ddts_concat = estimate_trajectory_derivatives(
        train_trajectories, train_times, order=order
    )

    ddt_denom = float(np.linalg.norm(ddts_concat, "fro"))

    sweep_history: List[Dict[str, Any]] = []
    best_model: Optional[opinf.models.ContinuousModel] = None
    best_reg = regularizers[0]
    best_score = float("inf")

    has_validation = val_trajectories is not None and val_times is not None and len(val_trajectories) > 0

    for reg in regularizers:
        model = build_model(
            model_type="continuous",
            operators=operators,
            regularizer=reg if reg > 0 else None,
        )

        # Fit model on concatenated data matrices
        model.fit(states_concat, ddts_concat)

        # Evaluate training residual: || lhs - rhs ||_F / || lhs ||_F
        # Re-evaluate right-hand side
        rhs = model.rhs(t=0.0, state=states_concat)
        res_norm = float(np.linalg.norm(ddts_concat - rhs, "fro"))
        train_res = res_norm / ddt_denom if ddt_denom > 0 else 0.0

        op_norms = compute_operator_norms(model)

        val_err = None
        is_stable_val = True

        if has_validation:
            val_errors = []
            for v_traj, v_t in zip(val_trajectories, val_times):
                a0 = v_traj[:, 0]
                a_pred = predict_continuous(model, a0, v_t, method=integration_method)
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

    # If all validation blew up, fall back to lowest training residual
    if best_model is None or not np.isfinite(best_score):
        best_entry = min(sweep_history, key=lambda x: x["train_residual"])
        best_reg = best_entry["regularizer"]
        best_model = build_model(
            model_type="continuous",
            operators=operators,
            regularizer=best_reg if best_reg > 0 else None,
        )
        best_model.fit(states_concat, ddts_concat)

    runtime = time.perf_counter() - t_start

    # Retrieve metrics for the chosen model
    chosen_metrics = next(m for m in sweep_history if m["regularizer"] == best_reg)

    return ContinuousTrainingResult(
        model=best_model,
        best_regularizer=best_reg,
        sweep_history=sweep_history,
        operator_norms=compute_operator_norms(best_model),
        train_residual=chosen_metrics["train_residual"],
        val_rollout_error=chosen_metrics["val_rollout_error"],
        runtime_seconds=runtime,
    )
