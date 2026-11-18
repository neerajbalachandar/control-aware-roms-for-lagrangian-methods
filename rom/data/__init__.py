"""
rom.data – Dataset ingestion, trajectory abstraction, and data validation for VarFlExI.
"""

from .trajectory import Trajectory
from .validation import ValidationReport, validate_trajectory, validate_trajectories
from .dataset_reader import DatasetReader

__all__ = [
    "Trajectory",
    "ValidationReport",
    "validate_trajectory",
    "validate_trajectories",
    "DatasetReader",
]
