"""
Unit tests for rom.data (Trajectory, Validation, DatasetReader).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rom.data import (
    DatasetReader,
    Trajectory,
    ValidationReport,
    validate_trajectories,
    validate_trajectory,
)
from rom.eulerian import EulerianGrid, FluidMask


def _create_mock_trajectory(
    k: int = 15,
    nx: int = 5,
    ny: int = 4,
    nz: int = 3,
    dt: float = 0.05,
    with_mask: bool = True,
    with_structural: bool = True,
    name: str = "test_run",
) -> Trajectory:
    time = np.linspace(0.0, (k - 1) * dt, k)
    grid = EulerianGrid(
        bounds={"x": (-1, 1), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        resolution={"x": nx, "y": ny, "z": nz},
        axis_order="xyz",
    )
    mask = None
    if with_mask:
        mask = FluidMask.from_sphere(grid, centre=(0.0, 0.0, 0.0), radius=0.4)

    # Synthetic divergence-free-like rotational velocity
    X, Y, Z = grid.meshgrid()
    vel = np.zeros((k, nx, ny, nz, 3), dtype=np.float64)
    for j in range(k):
        t = time[j]
        vel[j, ..., 0] = Y * np.cos(t)
        vel[j, ..., 1] = -X * np.cos(t)
        vel[j, ..., 2] = 0.1 * np.sin(t)
        if mask is not None:
            vel[j] = mask.apply(vel[j])

    vort = np.zeros_like(vel)
    for j in range(k):
        vort[j, ..., 2] = -2.0 * np.cos(time[j])
        if mask is not None:
            vort[j] = mask.apply(vort[j])

    disp = np.random.randn(k, 3) if with_structural else None
    vel_s = np.random.randn(k, 3) if with_structural else None
    acc_s = np.random.randn(k, 3) if with_structural else None
    forces = np.random.randn(k, 3) if with_structural else None

    params = {"Re": 1e5, "Mach": 0.1}
    coupling = {"tip_deflection": np.linspace(0, 0.05, k)}

    return Trajectory(
        time=time,
        velocity=vel,
        grid=grid,
        vorticity=vort,
        fluid_mask=mask,
        structural_displacement=disp,
        structural_velocity=vel_s,
        structural_acceleration=acc_s,
        aerodynamic_forces=forces,
        parameters=params,
        coupling_data=coupling,
        name=name,
    )


class TestTrajectory:
    def test_trajectory_properties(self):
        traj = _create_mock_trajectory(k=10, dt=0.1)
        assert len(traj) == 10
        assert traj.num_snapshots == 10
        assert traj.is_uniform_time is True
        assert pytest.approx(traj.dt, abs=1e-6) == 0.1
        assert traj.time_span == (0.0, pytest.approx(0.9, abs=1e-6))

    def test_nonuniform_dt(self):
        t = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
        grid = EulerianGrid(bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)}, resolution={"x": 2, "y": 2, "z": 2})
        vel = np.zeros((5, 2, 2, 2, 3))
        traj = Trajectory(time=t, velocity=vel, grid=grid)
        assert traj.is_uniform_time is False
        assert isinstance(traj.dt, np.ndarray)
        assert len(traj.dt) == 4

    def test_slice(self):
        traj = _create_mock_trajectory(k=20, dt=0.05)
        sub = traj.slice(0, 10)
        assert len(sub) == 10
        assert sub.velocity.shape[0] == 10
        if sub.structural_displacement is not None:
            assert sub.structural_displacement.shape[0] == 10
        assert sub.grid == traj.grid


class TestValidation:
    def test_valid_trajectory_passes(self):
        traj = _create_mock_trajectory(k=10)
        rep = validate_trajectory(traj)
        assert rep.is_valid is True
        assert len(rep.errors) == 0

    def test_non_monotonic_time_fails(self):
        traj = _create_mock_trajectory(k=10)
        traj.time[3] = traj.time[2]  # duplicate time
        rep = validate_trajectory(traj)
        assert rep.is_valid is False
        assert any("monotonically" in e for e in rep.errors)

    def test_shape_mismatch_fails(self):
        traj = _create_mock_trajectory(k=10)
        traj.velocity = traj.velocity[:8]  # K=8 but len(time)=10
        rep = validate_trajectory(traj)
        assert rep.is_valid is False
        assert any("snapshot count" in e for e in rep.errors)

    def test_nan_in_fluid_cells_fails(self):
        traj = _create_mock_trajectory(k=10, with_mask=True)
        # Put a NaN in a known fluid cell
        fluid_indices = np.where(traj.fluid_mask.mask)
        first_fluid = (0, fluid_indices[0][0], fluid_indices[1][0], fluid_indices[2][0], 0)
        traj.velocity[first_fluid] = np.nan
        rep = validate_trajectory(traj)
        assert rep.is_valid is False
        assert any("active fluid cells" in e for e in rep.errors)

    def test_multi_trajectory_compatibility(self):
        t1 = _create_mock_trajectory(k=10, name="t1")
        t2 = _create_mock_trajectory(k=15, name="t2")
        rep = validate_trajectories([t1, t2])
        assert rep.is_valid is True
        assert rep.info["num_trajectories"] == 2
        assert rep.info["total_snapshots"] == 25


class TestDatasetReader:
    def test_hdf5_save_and_load_roundtrip(self):
        traj = _create_mock_trajectory(k=12, dt=0.02, name="roundtrip_traj")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name

        try:
            DatasetReader.save_trajectory_to_hdf5(traj, path)
            loaded = DatasetReader.load_file(path)
            assert len(loaded) == 1
            t_load = loaded[0]

            assert len(t_load) == len(traj)
            np.testing.assert_allclose(t_load.time, traj.time, rtol=1e-10)
            # Compare velocities where fluid
            if traj.fluid_mask is not None:
                fl = traj.fluid_mask.mask
                np.testing.assert_allclose(t_load.velocity[:, fl, :], traj.velocity[:, fl, :], rtol=1e-10)
            else:
                np.testing.assert_allclose(t_load.velocity, traj.velocity, rtol=1e-10)

            assert t_load.structural_displacement is not None
            np.testing.assert_allclose(t_load.structural_displacement, traj.structural_displacement)
            assert t_load.parameters["Re"] == 1e5
        finally:
            if os.path.exists(path):
                os.unlink(path)
