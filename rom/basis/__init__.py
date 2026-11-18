"""
rom.basis – Proper Orthogonal Decomposition (POD) and state space representation.
"""

from .state_encoder import VelocityStateEncoder
from .pod import PODReducer
from .rank_selection import (
    RankMetrics,
    RankSelectionResult,
    evaluate_candidate_ranks,
)

__all__ = [
    "VelocityStateEncoder",
    "PODReducer",
    "RankMetrics",
    "RankSelectionResult",
    "evaluate_candidate_ranks",
]
