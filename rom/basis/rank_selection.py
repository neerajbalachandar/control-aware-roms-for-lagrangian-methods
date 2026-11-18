"""
rom.basis.rank_selection – Evaluation and selection of POD basis truncation ranks.

Evaluates singular value decay, cumulative energy retention, and relative
projection error across training and held-out validation trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.basis.pod import PODReducer


@dataclass
class RankMetrics:
    """Evaluation metrics for a single candidate POD rank."""

    rank: int
    singular_value: float
    cumulative_energy: float
    residual_energy: float
    train_projection_error: float
    val_projection_error: Optional[float] = None


@dataclass
class RankSelectionResult:
    """Summary of rank evaluation across multiple candidates."""

    metrics: List[RankMetrics] = field(default_factory=list)
    recommended_rank: int = 1
    selection_criterion: str = ""
    singular_values: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_table(self) -> str:
        """Format metrics as a human-readable markdown table."""
        header = "| Rank | Singular Val | Cumul. Energy (%) | Resid. Energy | Train Proj Err | Val Proj Err |\n"
        header += "|------|--------------|-------------------|---------------|----------------|--------------|\n"
        rows = []
        for m in self.metrics:
            val_err_str = f"{m.val_projection_error:.4e}" if m.val_projection_error is not None else "N/A"
            rows.append(
                f"| {m.rank:4d} | {m.singular_value:12.4e} | {m.cumulative_energy * 100:17.6f} | "
                f"{m.residual_energy:13.4e} | {m.train_projection_error:14.4e} | {val_err_str:12s} |"
            )
        return header + "\n".join(rows)


def evaluate_candidate_ranks(
    reducer: PODReducer,
    candidate_ranks: Sequence[int],
    Q_train_centered: np.ndarray,
    Q_val_centered: Optional[np.ndarray] = None,
    energy_target: float = 0.999,
) -> RankSelectionResult:
    """Evaluate candidate POD ranks and recommend an optimal rank.

    Parameters
    ----------
    reducer : PODReducer
        A fitted PODReducer instance.
    candidate_ranks : sequence of int
        Candidate ranks r to evaluate.
    Q_train_centered : np.ndarray
        Centered training snapshots of shape (n, K_train).
    Q_val_centered : np.ndarray, optional
        Centered held-out validation snapshots of shape (n, K_val).
    energy_target : float, optional
        Target cumulative energy fraction (default: 0.999).

    Returns
    -------
    RankSelectionResult
    """
    if not reducer.is_fitted:
        raise RuntimeError("PODReducer must be fitted before evaluating candidate ranks.")

    svals = reducer.singular_values
    total_energy = np.sum(svals ** 2)
    max_avail_rank = len(svals)

    valid_ranks = sorted(list(set(r for r in candidate_ranks if 1 <= r <= max_avail_rank)))
    if not valid_ranks:
        raise ValueError(f"No candidate ranks in valid range [1, {max_avail_rank}]")

    metrics_list: List[RankMetrics] = []
    recommended = valid_ranks[0]
    found_target = False

    orig_r = reducer.r

    try:
        for r in valid_ranks:
            reducer.set_rank(r)

            s_val = float(svals[r - 1])
            cum_energy = float(np.sum(svals[:r] ** 2) / total_energy) if total_energy > 0 else 1.0
            res_energy = float(1.0 - cum_energy)

            train_err = reducer.projection_error(Q_train_centered)

            val_err = None
            if Q_val_centered is not None:
                val_err = reducer.projection_error(Q_val_centered)

            m = RankMetrics(
                rank=r,
                singular_value=s_val,
                cumulative_energy=cum_energy,
                residual_energy=res_energy,
                train_projection_error=train_err,
                val_projection_error=val_err,
            )
            metrics_list.append(m)

            if not found_target and cum_energy >= energy_target:
                recommended = r
                found_target = True

    finally:
        # Restore original rank
        if orig_r in valid_ranks:
            reducer.set_rank(orig_r)

    if not found_target:
        recommended = valid_ranks[-1]

    criterion_desc = f"Smallest rank achieving cumulative energy >= {energy_target * 100:.2f}%"

    return RankSelectionResult(
        metrics=metrics_list,
        recommended_rank=recommended,
        selection_criterion=criterion_desc,
        singular_values=svals,
    )
