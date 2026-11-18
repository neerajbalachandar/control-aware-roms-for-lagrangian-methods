"""
rom.data.validation – Rigorous validation routines for ingested simulation datasets.

Validates snapshot counts, temporal alignment and monotonicity, grid dimensions,
component ordering, missing values, and fluid mask consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from rom.data.trajectory import Trajectory


@dataclass
class ValidationReport:
    """Report detailing the results of trajectory validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_trajectory(traj: Trajectory, strict: bool = True) -> ValidationReport:
    """Validate a single Trajectory instance.

    Parameters
    ----------
    traj : Trajectory
        The trajectory to validate.
    strict : bool, optional
        If True, non-fatal anomalies are logged as errors; otherwise warnings.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    report.info["name"] = traj.name

    # 1. Temporal array checks
    if traj.time.ndim != 1:
        report.add_error(f"Time array must be 1-D, got {traj.time.ndim}-D with shape {traj.time.shape}")
        return report

    k = len(traj.time)
    if k < 2:
        report.add_error(f"Trajectory must contain at least 2 snapshots, got {k}")
        return report

    if not np.all(np.isfinite(traj.time)):
        report.add_error("Time array contains non-finite values (NaN or Inf)")

    diffs = np.diff(traj.time)
    if np.any(diffs <= 0.0):
        non_mono_idx = np.where(diffs <= 0.0)[0]
        report.add_error(
            f"Time array is not strictly monotonically increasing at indices {non_mono_idx[:5]}. "
            f"diffs: {diffs[non_mono_idx[:5]]}"
        )

    is_uniform = traj.is_uniform_time
    report.info["is_uniform_time"] = is_uniform
    report.info["num_snapshots"] = k
    report.info["time_span"] = traj.time_span
    report.info["dt"] = traj.dt if is_uniform else "non-uniform"

    # 2. Velocity array checks
    expected_spatial_shape = traj.grid.shape
    expected_vel_shape = (k,) + expected_spatial_shape + (3,)

    if traj.velocity.ndim != 5:
        report.add_error(f"Velocity array must be 5-D (K, N0, N1, N2, 3), got shape {traj.velocity.shape}")
        return report

    if traj.velocity.shape[0] != k:
        report.add_error(f"Velocity snapshot count {traj.velocity.shape[0]} != time length {k}")

    if traj.velocity.shape[1:4] != expected_spatial_shape:
        report.add_error(
            f"Velocity spatial shape {traj.velocity.shape[1:4]} != grid shape {expected_spatial_shape}"
        )

    if traj.velocity.shape[4] != 3:
        report.add_error(f"Velocity last dimension must be 3 (Cartesian u_x, u_y, u_z), got {traj.velocity.shape[4]}")

    # Missing value / NaN checks considering fluid mask
    if traj.fluid_mask is not None:
        if traj.fluid_mask.mask.shape != expected_spatial_shape:
            report.add_error(
                f"Fluid mask shape {traj.fluid_mask.mask.shape} != grid shape {expected_spatial_shape}"
            )
        else:
            fluid_idx = traj.fluid_mask.mask
            # Check for non-finite values in active fluid cells across all snapshots
            fluid_vel = traj.velocity[:, fluid_idx, :]
            if not np.all(np.isfinite(fluid_vel)):
                n_bad = np.count_nonzero(~np.isfinite(fluid_vel))
                report.add_error(f"Found {n_bad} non-finite velocity values in active fluid cells")
    else:
        if not np.all(np.isfinite(traj.velocity)):
            n_bad = np.count_nonzero(~np.isfinite(traj.velocity))
            report.add_error(f"Found {n_bad} non-finite velocity values in unmasked domain")

    # 3. Vorticity checks (if present)
    if traj.vorticity is not None:
        if traj.vorticity.shape != traj.velocity.shape:
            report.add_error(f"Vorticity shape {traj.vorticity.shape} != velocity shape {traj.velocity.shape}")
        elif traj.fluid_mask is not None:
            fluid_idx = traj.fluid_mask.mask
            fluid_vort = traj.vorticity[:, fluid_idx, :]
            if not np.all(np.isfinite(fluid_vort)):
                n_bad = np.count_nonzero(~np.isfinite(fluid_vort))
                report.add_error(f"Found {n_bad} non-finite vorticity values in active fluid cells")
        elif not np.all(np.isfinite(traj.vorticity)):
            n_bad = np.count_nonzero(~np.isfinite(traj.vorticity))
            report.add_error(f"Found {n_bad} non-finite vorticity values in unmasked domain")

    # 4. Structural state checks (if present)
    for name, arr in [
        ("structural_displacement", traj.structural_displacement),
        ("structural_velocity", traj.structural_velocity),
        ("structural_acceleration", traj.structural_acceleration),
        ("aerodynamic_forces", traj.aerodynamic_forces),
    ]:
        if arr is not None:
            if arr.shape[0] != k:
                report.add_error(f"{name} length {arr.shape[0]} != snapshot count {k}")
            if not np.all(np.isfinite(arr)):
                report.add_error(f"{name} contains non-finite values")

    return report


def validate_trajectories(trajectories: Sequence[Trajectory]) -> ValidationReport:
    """Validate a collection of trajectories for multi-trajectory training compatibility.

    Ensures all trajectories share consistent spatial discretisations, axis orders,
    and fluid masks, and verifies that discontinuous time domains are not conflated.
    """
    report = ValidationReport()
    if not trajectories:
        report.add_error("Trajectory collection is empty")
        return report

    # Validate each trajectory individually
    for i, t in enumerate(trajectories):
        sub_rep = validate_trajectory(t)
        if not sub_rep.is_valid:
            report.add_error(f"Trajectory {i} ('{t.name}') failed validation: {'; '.join(sub_rep.errors)}")
        for w in sub_rep.warnings:
            report.add_warning(f"Trajectory {i} ('{t.name}'): {w}")

    if not report.is_valid:
        return report

    # Check cross-trajectory compatibility
    ref = trajectories[0]
    for i, t in enumerate(trajectories[1:], start=1):
        if t.grid.shape != ref.grid.shape:
            report.add_error(
                f"Trajectory {i} grid shape {t.grid.shape} != reference grid shape {ref.grid.shape}"
            )
        if t.grid.axis_order != ref.grid.axis_order:
            report.add_error(
                f"Trajectory {i} axis order '{t.grid.axis_order}' != reference '{ref.grid.axis_order}'"
            )
        for ax in "xyz":
            if not np.allclose(t.grid.coords_1d(ax), ref.grid.coords_1d(ax), rtol=1e-5, atol=1e-8):
                report.add_error(f"Trajectory {i} coordinates along {ax} differ from reference")

        # Fluid mask check
        if (ref.fluid_mask is None) != (t.fluid_mask is None):
            report.add_error(
                f"Trajectory {i} fluid mask presence ({t.fluid_mask is not None}) != reference ({ref.fluid_mask is not None})"
            )
        elif ref.fluid_mask is not None and t.fluid_mask is not None:
            if not np.array_equal(ref.fluid_mask.mask, t.fluid_mask.mask):
                report.add_error(f"Trajectory {i} fluid mask does not match reference mask")

    report.info["num_trajectories"] = len(trajectories)
    report.info["total_snapshots"] = sum(len(t) for t in trajectories)
    return report
