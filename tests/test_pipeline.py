"""
Integration test for the entire Dataset-to-ROM training, evaluation, and reconstruction pipeline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from rom.data import DatasetReader, Trajectory
from rom.eulerian import EulerianGrid, FluidMask


def _create_sample_dataset(filepath: str, k: int = 50, nx: int = 6, ny: int = 6, nz: int = 4):
    grid = EulerianGrid(
        bounds={"x": (-1, 1), "y": (-1, 1), "z": (-0.5, 0.5)},
        resolution={"x": nx, "y": ny, "z": nz},
        axis_order="xyz",
    )
    mask = FluidMask.from_sphere(grid, centre=(0, 0, 0), radius=0.4)

    t = np.linspace(0.0, 1.0, k)
    X, Y, Z = grid.meshgrid()

    vel = np.zeros((k, nx, ny, nz, 3), dtype=np.float64)
    for j in range(k):
        # Quasi-periodic rotational vortex motion
        vel[j, ..., 0] = Y * np.cos(2 * np.pi * t[j])
        vel[j, ..., 1] = -X * np.cos(2 * np.pi * t[j])
        vel[j, ..., 2] = 0.05 * np.sin(2 * np.pi * t[j])
        vel[j] = mask.apply(vel[j])

    vort = np.zeros_like(vel)
    for j in range(k):
        vort[j, ..., 2] = -2.0 * np.cos(2 * np.pi * t[j])
        vort[j] = mask.apply(vort[j])

    traj = Trajectory(
        time=t,
        velocity=vel,
        grid=grid,
        vorticity=vort,
        fluid_mask=mask,
        name="pipeline_test_run",
    )

    DatasetReader.save_trajectory_to_hdf5(traj, filepath)
    return traj


def test_full_pipeline_cli():
    python_exe = sys.executable

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_h5 = str(tmp_path / "sample_sim.h5")
        rom_dir = str(tmp_path / "rom_model")
        eval_report = str(tmp_path / "eval_report.json")
        rec_h5 = str(tmp_path / "reconstructed.h5")

        # 1. Create sample dataset
        _create_sample_dataset(dataset_h5, k=40)
        assert os.path.isfile(dataset_h5)

        # 2. Run train_opinf.py
        cmd_train = [
            python_exe,
            "scripts/train_opinf.py",
            "--data", dataset_h5,
            "--model-type", "continuous",
            "--operators", "cAH",
            "--ranks", "2", "3", "5",
            "--regularizers", "1e-5", "1e-3",
            "--output-dir", rom_dir,
            "--train-fraction", "0.7",
            "--val-fraction", "0.3",
        ]
        res_train = subprocess.run(cmd_train, capture_output=True, text=True)
        assert res_train.returncode == 0, f"train_opinf failed:\n{res_train.stderr}"

        # Check required files
        for f in ["metadata.json", "mean_field.npz", "pod_basis.npz", "operators.h5", "grid_metadata.npz", "preprocessing.npz", "training_metrics.json"]:
            assert (Path(rom_dir) / f).is_file(), f"Missing {f}"

        # 3. Run evaluate_opinf.py
        cmd_eval = [
            python_exe,
            "scripts/evaluate_opinf.py",
            "--rom-dir", rom_dir,
            "--data", dataset_h5,
            "--output-report", eval_report,
        ]
        res_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
        assert res_eval.returncode == 0, f"evaluate_opinf failed:\n{res_eval.stderr}"
        assert os.path.isfile(eval_report)

        with open(eval_report, "r") as f:
            rep_data = json.load(f)
        assert "evaluation_summary" in rep_data
        assert rep_data["evaluation_summary"]["average_projection_error"] < 0.1

        # 4. Run reconstruct_rom_fields.py
        cmd_rec = [
            python_exe,
            "scripts/reconstruct_rom_fields.py",
            "--rom-dir", rom_dir,
            "--data", dataset_h5,
            "--output-h5", rec_h5,
        ]
        res_rec = subprocess.run(cmd_rec, capture_output=True, text=True)
        assert res_rec.returncode == 0, f"reconstruct_rom_fields failed:\n{res_rec.stderr}"
        assert os.path.isfile(rec_h5)

        # Verify reconstructed HDF5 dataset structure
        with h5py.File(rec_h5, "r") as f:
            assert "metadata" in f
            assert "snapshots" in f
            assert "velocity" in f["snapshots"]
            assert "vorticity" in f["snapshots"]

            vel = f["snapshots/velocity"]
            vort = f["snapshots/vorticity"]
            assert vel.shape == (40, 6, 6, 4, 3)
            assert vort.shape == (40, 6, 6, 4, 3)
