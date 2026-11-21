"""
rom_opinf.derivatives – Temporal derivative estimation and discrete state pairing.

Supports uniform (order 2, 4, 6) and non-uniform finite differences, strictly
enforcing trajectory boundary isolation so that differentiation is never
performed across discontinuous time intervals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import opinf


def estimate_derivatives_uniform(
    states: np.ndarray,
    dt: float,
    order: int = 2,
) -> np.ndarray:
    """Estimate time derivatives for uniform sampling via opinf.ddt.ddt_uniform.

    Endpoint treatment:
    - Forward finite-difference stencil of specified order at the initial step t_0.
    - Central finite-difference stencil of specified order in the interior (t_1 ... t_{K-2}).
    - Backward finite-difference stencil of specified order at the final step t_{K-1}.

    Parameters
    ----------
    states : np.ndarray
        State array of shape ``(r, K)`` where column j is the state at time t_j.
    dt : float
        Uniform time step size.
    order : {2, 4, 6}, optional
        Accuracy order of the finite-difference stencils (default: 2).

    Returns
    -------
    ddts : np.ndarray
        Derivative matrix of shape ``(r, K)``.
    """
    if states.ndim != 2:
        raise ValueError(f"States must be 2-D (r, K), got shape {states.shape}")
    if order not in (2, 4, 6):
        raise ValueError(f"Order must be 2, 4, or 6, got {order}")
    if states.shape[1] < order + 1:
        raise ValueError(
            f"Need at least {order + 1} snapshots for order {order} differentiation, "
            f"got {states.shape[1]}"
        )

    return opinf.ddt.ddt_uniform(states, dt, order=order)


def estimate_derivatives_nonuniform(
    states: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Estimate time derivatives for non-uniform time sampling via opinf.ddt.ddt_nonuniform.

    Uses cubic spline interpolation or generalized asymmetric finite differences.

    Parameters
    ----------
    states : np.ndarray
        Shape ``(r, K)``.
    time : np.ndarray
        Shape ``(K,)`` of monotonically increasing timestamps.

    Returns
    -------
    ddts : np.ndarray
        Derivative matrix of shape ``(r, K)``.
    """
    if states.ndim != 2:
        raise ValueError(f"States must be 2-D (r, K), got shape {states.shape}")
    if len(time) != states.shape[1]:
        raise ValueError(f"Time length {len(time)} != states column count {states.shape[1]}")

    return opinf.ddt.ddt_nonuniform(states, time)


def estimate_trajectory_derivatives(
    reduced_trajectories: Sequence[np.ndarray],
    times: Sequence[np.ndarray],
    order: int = 2,
    force_nonuniform: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Estimate derivatives for multiple independent trajectories independently.

    CRITICAL GUARANTEE:
    Differentiation is evaluated strictly within each individual trajectory.
    Under no circumstances are finite-difference stencils applied across
    discontinuous trajectory boundaries.

    Parameters
    ----------
    reduced_trajectories : sequence of np.ndarray
        List of state matrices, each of shape ``(r, K_i)``.
    times : sequence of np.ndarray
        List of time arrays, each of shape ``(K_i,)``.
    order : {2, 4, 6}, optional
        Finite difference accuracy order for uniform trajectories.
    force_nonuniform : bool, optional
        Force using non-uniform derivative estimator even for uniform grids.

    Returns
    -------
    ddt_list : list of np.ndarray
        List of derivative matrices ``[ddt_1, ddt_2, ...]``, each of shape ``(r, K_i)``.
    states_concat : np.ndarray
        Horizontally concatenated states ``[a^(1), a^(2), ...]`` of shape ``(r, sum K_i)``.
    ddts_concat : np.ndarray
        Horizontally concatenated derivatives ``[da^(1), da^(2), ...]`` of shape ``(r, sum K_i)``.
    """
    if len(reduced_trajectories) != len(times):
        raise ValueError("Mismatch between number of trajectories and time arrays.")

    ddt_list: List[np.ndarray] = []

    for idx, (states, t) in enumerate(zip(reduced_trajectories, times)):
        diffs = np.diff(t)
        mean_dt = float(np.mean(diffs))
        is_uniform = bool(np.allclose(diffs, mean_dt, rtol=1e-4, atol=1e-8))

        if is_uniform and not force_nonuniform:
            ddt_i = estimate_derivatives_uniform(states, mean_dt, order=order)
        else:
            ddt_i = estimate_derivatives_nonuniform(states, t)

        ddt_list.append(ddt_i)

    states_concat = np.hstack(reduced_trajectories)
    ddts_concat = np.hstack(ddt_list)

    return ddt_list, states_concat, ddts_concat


def prepare_discrete_pairs(
    reduced_trajectories: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare properly aligned state/next-state pairs for discrete OpInf models.

    CRITICAL GUARANTEE:
    For each trajectory i with shape ``(r, K_i)``:
      states_k    = trajectory[:, :-1]
      states_next = trajectory[:, 1:]
    The final state of trajectory i is NEVER paired with the initial state
    of trajectory i+1.

    Parameters
    ----------
    reduced_trajectories : sequence of np.ndarray
        List of reduced state matrices, each of shape ``(r, K_i)``.

    Returns
    -------
    states_k : np.ndarray
        Shape ``(r, sum(K_i - 1))``.
    states_next : np.ndarray
        Shape ``(r, sum(K_i - 1))``.
    """
    states_k_list = []
    states_next_list = []

    for idx, states in enumerate(reduced_trajectories):
        if states.shape[1] < 2:
            raise ValueError(f"Trajectory {idx} has fewer than 2 snapshots; cannot form step pairs.")
        states_k_list.append(states[:, :-1])
        states_next_list.append(states[:, 1:])

    states_k = np.hstack(states_k_list)
    states_next = np.hstack(states_next_list)

    return states_k, states_next


def compare_derivative_estimators(
    states: np.ndarray,
    time: np.ndarray,
) -> Dict[str, Any]:
    """Compare available derivative estimators and report derivative sensitivity.

    Parameters
    ----------
    states : np.ndarray
        Shape ``(r, K)``.
    time : np.ndarray
        Shape ``(K,)``.

    Returns
    -------
    report : dict
        Sensitivity metrics and relative discrepancies between schemes.
    """
    diffs = np.diff(time)
    dt = float(np.mean(diffs))
    is_uniform = bool(np.allclose(diffs, dt, rtol=1e-4, atol=1e-8))

    estimates: Dict[str, np.ndarray] = {}

    if is_uniform and states.shape[1] >= 7:
        estimates["ord2"] = estimate_derivatives_uniform(states, dt, order=2)
        estimates["ord4"] = estimate_derivatives_uniform(states, dt, order=4)
        estimates["ord6"] = estimate_derivatives_uniform(states, dt, order=6)
    elif is_uniform and states.shape[1] >= 5:
        estimates["ord2"] = estimate_derivatives_uniform(states, dt, order=2)
        estimates["ord4"] = estimate_derivatives_uniform(states, dt, order=4)
    elif is_uniform and states.shape[1] >= 3:
        estimates["ord2"] = estimate_derivatives_uniform(states, dt, order=2)

    estimates["nonuniform"] = estimate_derivatives_nonuniform(states, time)

    # Discrepancy analysis
    comparisons = {}
    base_key = "ord2" if "ord2" in estimates else "nonuniform"
    base_ddt = estimates[base_key]
    base_norm = float(np.linalg.norm(base_ddt))

    for k, ddt in estimates.items():
        diff_norm = float(np.linalg.norm(ddt - base_ddt))
        rel_diff = diff_norm / base_norm if base_norm > 0 else 0.0
        comparisons[k] = {
            "frobenius_norm": float(np.linalg.norm(ddt)),
            "rel_diff_vs_baseline": rel_diff,
            "max_abs_diff": float(np.max(np.abs(ddt - base_ddt))),
        }

    return {
        "baseline": base_key,
        "is_uniform_time": is_uniform,
        "schemes": comparisons,
    }
