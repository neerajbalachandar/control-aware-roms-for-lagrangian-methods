"""
rom.opinf – Operator Inference modeling, derivative estimation, training, and persistence.
"""

from .model_factory import build_model, extract_operator_entries, compute_operator_norms
from .derivatives import (
    estimate_derivatives_uniform,
    estimate_derivatives_nonuniform,
    estimate_trajectory_derivatives,
    prepare_discrete_pairs,
    compare_derivative_estimators,
)
from .predict import (
    predict_continuous,
    predict_discrete,
    compute_rollout_error,
    assess_stability,
)
from .train_continuous import (
    ContinuousTrainingResult,
    train_continuous_model,
)
from .train_discrete import (
    DiscreteTrainingResult,
    train_discrete_model,
)
from .persistence import TrainedROM

__all__ = [
    "build_model",
    "extract_operator_entries",
    "compute_operator_norms",
    "estimate_derivatives_uniform",
    "estimate_derivatives_nonuniform",
    "estimate_trajectory_derivatives",
    "prepare_discrete_pairs",
    "compare_derivative_estimators",
    "predict_continuous",
    "predict_discrete",
    "compute_rollout_error",
    "assess_stability",
    "ContinuousTrainingResult",
    "train_continuous_model",
    "DiscreteTrainingResult",
    "train_discrete_model",
    "TrainedROM",
]
