"""
rom.opinf.model_factory – Factory for constructing Operator Inference models.

Supports autonomous continuous (cAH) and autonomous discrete (cAH) models using
the upstream opinf package, with Tikhonov / L2 regularization. Input-driven
variants are defined as explicitly deferred extensions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import opinf


def build_model(
    model_type: str = "continuous",
    operators: str = "cAH",
    regularizer: Optional[float] = 1e-4,
    **solver_options,
) -> Union[opinf.models.ContinuousModel, opinf.models.DiscreteModel]:
    """Construct an un-fitted OpInf continuous or discrete model.

    Parameters
    ----------
    model_type : {'continuous', 'discrete'}
        Model formulation:
        - 'continuous': da/dt = c + A a + H(a ⊗ a)
        - 'discrete': a_{k+1} = c + A a_k + H(a_k ⊗ a_k)
    operators : str, optional
        Operator string (default: "cAH" for constant, linear, quadratic).
    regularizer : float, optional
        Tikhonov L2 regularizer parameter lambda (default: 1e-4).
        If 0.0 or None, uses unregularized least squares.
    solver_options : dict, optional
        Additional options passed to L2Solver or PlainSolver.

    Returns
    -------
    model : opinf.models.ContinuousModel or opinf.models.DiscreteModel
    """
    model_type_clean = model_type.lower().strip()

    if regularizer is not None and regularizer > 0.0:
        solver = opinf.lstsq.L2Solver(regularizer=regularizer, **solver_options)
    else:
        solver = opinf.lstsq.PlainSolver(**solver_options)

    if model_type_clean in ("continuous", "cont", "ode"):
        return opinf.models.ContinuousModel(operators, solver=solver)
    elif model_type_clean in ("discrete", "disc", "map"):
        return opinf.models.DiscreteModel(operators, solver=solver)
    elif model_type_clean in ("input_continuous", "input_cont"):
        raise NotImplementedError(
            "Input-driven continuous OpInf (da/dt = c + Aa + H(a⊗a) + Bu) is deferred. "
            "Autonomous baseline must be validated first."
        )
    elif model_type_clean in ("input_discrete", "input_disc"):
        raise NotImplementedError(
            "Input-driven discrete OpInf (a_{k+1} = f(a_k, u_k)) is deferred. "
            "Autonomous baseline must be validated first."
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected 'continuous' or 'discrete'.")


def extract_operator_entries(
    model: Union[opinf.models.ContinuousModel, opinf.models.DiscreteModel]
) -> Dict[str, np.ndarray]:
    """Extract individual operator matrices/tensors from a fitted OpInf model."""
    entries = {}
    for op in model.operators:
        op_name = op.__class__.__name__
        if hasattr(op, "entries") and op.entries is not None:
            entries[op_name] = np.array(op.entries, copy=True)
    return entries


def compute_operator_norms(
    model: Union[opinf.models.ContinuousModel, opinf.models.DiscreteModel]
) -> Dict[str, float]:
    """Compute Frobenius norms of fitted operators."""
    norms = {}
    for name, arr in extract_operator_entries(model).items():
        norms[f"norm_{name}"] = float(np.linalg.norm(arr))
    return norms
