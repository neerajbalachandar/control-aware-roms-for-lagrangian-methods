"""
rom.opinf.predict – Prediction and rollout error evaluation for OpInf models.

Provides robust prediction wrappers for continuous and discrete OpInf models,
detects numerical blow-ups or instability, and evaluates dynamical rollout metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import opinf

logger = logging.getLogger(__name__)


def predict_continuous(
    model: opinf.models.ContinuousModel,
    a0: np.ndarray,
    time_domain: np.ndarray,
    method: str = "BDF",
    **integrator_options,
) -> np.ndarray:
    """Simulate an autonomous continuous-time OpInf model from initial condition a0.

    Parameters
    ----------
    model : opinf.models.ContinuousModel
        Fitted continuous OpInf model.
    a0 : np.ndarray
        Initial reduced state vector of shape ``(r,)``.
    time_domain : np.ndarray
        Time points at which to evaluate the solution, shape ``(K,)``.
    method : str, optional
        ODE integration method supported by scipy.integrate.solve_ivp (default: 'BDF').
    integrator_options : dict, optional
        Additional options passed to scipy.integrate.solve_ivp.

    Returns
    -------
    a_pred : np.ndarray
        Predicted trajectory of shape ``(r, K)``.
    """
    a0 = np.asarray(a0, dtype=np.float64).ravel()
    time_domain = np.asarray(time_domain, dtype=np.float64)

    try:
        a_pred = model.predict(a0, time_domain, method=method, **integrator_options)
    except Exception as exc:
        logger.warning("Continuous model integration failed: %s", exc)
        # Return array filled with NaNs on failure
        return np.full((len(a0), len(time_domain)), np.nan, dtype=np.float64)

    return a_pred


def predict_discrete(
    model: opinf.models.DiscreteModel,
    a0: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Simulate an autonomous discrete-time OpInf model for num_steps.

    Parameters
    ----------
    model : opinf.models.DiscreteModel
        Fitted discrete OpInf model.
    a0 : np.ndarray
        Initial reduced state vector of shape ``(r,)``.
    num_steps : int
        Number of steps to iterate forward.

    Returns
    -------
    a_pred : np.ndarray
        Predicted trajectory of shape ``(r, num_steps)``.
    """
    a0 = np.asarray(a0, dtype=np.float64).ravel()

    try:
        a_pred = model.predict(a0, num_steps)
    except Exception as exc:
        logger.warning("Discrete model simulation failed: %s", exc)
        return np.full((len(a0), num_steps), np.nan, dtype=np.float64)

    return a_pred


def compute_rollout_error(
    a_true: np.ndarray,
    a_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute relative and absolute error metrics between true and predicted rollouts.

    Parameters
    ----------
    a_true : np.ndarray
        Shape ``(r, K)``.
    a_pred : np.ndarray
        Shape ``(r, K)``.

    Returns
    -------
    metrics : dict
        - 'relative_frobenius_error': ||a_true - a_pred||_F / ||a_true||_F
        - 'mean_relative_l2_error': time-averaged relative error per snapshot
        - 'max_absolute_error': maximum entry-wise absolute error
    """
    if a_true.shape != a_pred.shape:
        raise ValueError(f"Shape mismatch: a_true {a_true.shape} != a_pred {a_pred.shape}")

    if not np.all(np.isfinite(a_pred)):
        return {
            "relative_frobenius_error": float("inf"),
            "mean_relative_l2_error": float("inf"),
            "max_absolute_error": float("inf"),
        }

    denom_fro = float(np.linalg.norm(a_true, "fro"))
    diff_fro = float(np.linalg.norm(a_true - a_pred, "fro"))
    rel_fro = diff_fro / denom_fro if denom_fro > 0 else 0.0

    # Per-snapshot L2 relative errors
    true_norms = np.linalg.norm(a_true, axis=0)  # (K,)
    diff_norms = np.linalg.norm(a_true - a_pred, axis=0)  # (K,)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_l2_series = np.where(true_norms > 1e-12, diff_norms / true_norms, diff_norms)

    mean_rel_l2 = float(np.mean(rel_l2_series))
    max_abs = float(np.max(np.abs(a_true - a_pred)))

    return {
        "relative_frobenius_error": rel_fro,
        "mean_relative_l2_error": mean_rel_l2,
        "max_absolute_error": max_abs,
    }


def assess_stability(
    a_pred: np.ndarray,
    reference_scale: Optional[float] = None,
    max_growth_factor: float = 100.0,
) -> Dict[str, Any]:
    """Assess whether a simulated rollout remained numerically stable.

    Parameters
    ----------
    a_pred : np.ndarray
        Predicted trajectory of shape ``(r, K)``.
    reference_scale : float, optional
        Expected maximum magnitude scale (e.g. from initial condition or training data).
    max_growth_factor : float, optional
        Maximum allowed ratio of max magnitude to reference scale.

    Returns
    -------
    stability_info : dict
        Contains 'is_stable', 'contains_nan', 'max_magnitude'.
    """
    if not np.all(np.isfinite(a_pred)):
        return {
            "is_stable": False,
            "contains_nan": True,
            "max_magnitude": float("inf"),
        }

    max_mag = float(np.max(np.abs(a_pred)))

    if reference_scale is not None and reference_scale > 0:
        is_stable = bool(max_mag <= reference_scale * max_growth_factor)
    else:
        # If no reference scale, check that values don't exceed an arbitrary safe bound (e.g. 1e6)
        is_stable = bool(max_mag < 1e6)

    return {
        "is_stable": is_stable,
        "contains_nan": False,
        "max_magnitude": max_mag,
    }
