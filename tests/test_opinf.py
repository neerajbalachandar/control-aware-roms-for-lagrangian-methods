"""
Unit tests for rom.opinf (model_factory, derivatives, training, predict, persistence).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rom.basis import PODReducer, VelocityStateEncoder
from rom.eulerian import EulerianGrid, FluidMask
from rom.opinf import (
    TrainedROM,
    build_model,
    compare_derivative_estimators,
    compute_operator_norms,
    compute_rollout_error,
    estimate_derivatives_nonuniform,
    estimate_derivatives_uniform,
    estimate_trajectory_derivatives,
    extract_operator_entries,
    predict_continuous,
    predict_discrete,
    prepare_discrete_pairs,
    train_continuous_model,
    train_discrete_model,
)


def _generate_synthetic_oscillator_data(
    r: int = 3,
    k: int = 60,
    dt: float = 0.05,
    omega: float = 2.0,
):
    """Generate synthetic oscillatory trajectory satisfying a linear/quadratic ODE."""
    t = np.linspace(0.0, (k - 1) * dt, k)
    a = np.zeros((r, k), dtype=np.float64)
    a[0] = np.cos(omega * t)
    a[1] = np.sin(omega * t)
    if r > 2:
        a[2] = 0.2 * np.cos(2 * omega * t)
    return a, t


class TestModelFactory:
    def test_build_continuous_and_discrete(self):
        m_cont = build_model("continuous", operators="cAH", regularizer=1e-3)
        assert hasattr(m_cont, "operators")
        assert len(m_cont.operators) == 3

        m_disc = build_model("discrete", operators="cAH", regularizer=1e-4)
        assert hasattr(m_disc, "operators")
        assert len(m_disc.operators) == 3

    def test_deferred_input_models(self):
        with pytest.raises(NotImplementedError, match="deferred"):
            build_model("input_continuous")

        with pytest.raises(NotImplementedError, match="deferred"):
            build_model("input_discrete")


class TestDerivatives:
    def test_uniform_derivatives_accuracy(self):
        t = np.linspace(0, 2 * np.pi, 100)
        dt = t[1] - t[0]
        # x(t) = sin(t), dx/dt = cos(t)
        states = np.array([np.sin(t)])
        expected = np.array([np.cos(t)])

        d2 = estimate_derivatives_uniform(states, dt, order=2)
        # Check interior accuracy
        np.testing.assert_allclose(d2[:, 2:-2], expected[:, 2:-2], atol=1e-3)

    def test_trajectory_boundary_isolation(self):
        a1, t1 = _generate_synthetic_oscillator_data(r=2, k=30, dt=0.05)
        a2, t2 = _generate_synthetic_oscillator_data(r=2, k=40, dt=0.05)
        # Shift a2 to create an obvious discontinuity
        a2 = a2 + 5.0

        ddt_list, states_concat, ddts_concat = estimate_trajectory_derivatives([a1, a2], [t1, t2], order=2)
        assert len(ddt_list) == 2
        assert ddt_list[0].shape == a1.shape
        assert ddt_list[1].shape == a2.shape
        assert states_concat.shape == (2, 70)
        assert ddts_concat.shape == (2, 70)

        # Ensure derivative at final step of traj 1 was computed from traj 1 alone
        d_single1 = estimate_derivatives_uniform(a1, 0.05, order=2)
        np.testing.assert_allclose(ddt_list[0], d_single1)

    def test_prepare_discrete_pairs(self):
        a1 = np.array([[1.0, 2.0, 3.0]])
        a2 = np.array([[10.0, 20.0, 30.0, 40.0]])

        states_k, states_next = prepare_discrete_pairs([a1, a2])
        # Traj 1 pairs: (1, 2), (2, 3)
        # Traj 2 pairs: (10, 20), (20, 30), (30, 40)
        # NEVER (3, 10)!
        expected_k = np.array([[1.0, 2.0, 10.0, 20.0, 30.0]])
        expected_next = np.array([[2.0, 3.0, 20.0, 30.0, 40.0]])
        np.testing.assert_array_equal(states_k, expected_k)
        np.testing.assert_array_equal(states_next, expected_next)

    def test_compare_derivative_estimators(self):
        a, t = _generate_synthetic_oscillator_data(r=2, k=30)
        report = compare_derivative_estimators(a, t)
        assert "schemes" in report
        assert "ord2" in report["schemes"]
        assert "nonuniform" in report["schemes"]


class TestTrainingAndPredict:
    def test_continuous_training_and_rollout(self):
        a_train, t_train = _generate_synthetic_oscillator_data(r=2, k=80, dt=0.02)
        a_val, t_val = _generate_synthetic_oscillator_data(r=2, k=40, dt=0.02)

        result = train_continuous_model(
            train_trajectories=[a_train],
            train_times=[t_train],
            operators="cAH",
            regularizers=[1e-4, 1e-2],
            val_trajectories=[a_val],
            val_times=[t_val],
            order=2,
        )

        assert result.model is not None
        assert len(result.sweep_history) == 2
        assert result.train_residual < 0.1

        # Predict rollout from a0
        a_pred = predict_continuous(result.model, a_val[:, 0], t_val)
        err = compute_rollout_error(a_val, a_pred)
        assert err["relative_frobenius_error"] < 0.2

    def test_discrete_training_and_rollout(self):
        a_train, _ = _generate_synthetic_oscillator_data(r=2, k=80, dt=0.02)
        a_val, _ = _generate_synthetic_oscillator_data(r=2, k=40, dt=0.02)

        result = train_discrete_model(
            train_trajectories=[a_train],
            operators="cAH",
            regularizers=[1e-5, 1e-3],
            val_trajectories=[a_val],
        )

        assert result.model is not None
        assert len(result.sweep_history) == 2

        # Predict multi-step rollout
        a_pred = predict_discrete(result.model, a_val[:, 0], num_steps=a_val.shape[1])
        err = compute_rollout_error(a_val, a_pred)
        assert err["relative_frobenius_error"] < 0.2


class TestPersistence:
    def test_rom_persistence_and_equivalence(self):
        # 1. Setup spatial grid, mask, encoder, and reducer
        grid = EulerianGrid(
            bounds={"x": (-1, 1), "y": (-1, 1), "z": (-0.5, 0.5)},
            resolution={"x": 7, "y": 7, "z": 5},
        )
        mask = FluidMask.from_sphere(grid, centre=(0, 0, 0), radius=0.4)
        encoder = VelocityStateEncoder(grid, mask)

        # Create synthetic training field
        k_train = 50
        t_train = np.linspace(0, 1.0, k_train)
        vel_train = np.zeros((k_train, 7, 7, 5, 3))
        X, Y, Z = grid.meshgrid()
        for j in range(k_train):
            vel_train[j, ..., 0] = Y * np.cos(t_train[j])
            vel_train[j, ..., 1] = -X * np.cos(t_train[j])
            vel_train[j] = mask.apply(vel_train[j])

        Q_train = encoder.encode_trajectory(vel_train)
        encoder.fit_mean(Q_train)
        Q_train_cent = encoder.center(Q_train)

        reducer = PODReducer(r=3)
        reducer.fit(Q_train_cent)
        a_train = reducer.project(Q_train_cent)

        # 2. Train continuous OpInf model
        train_res = train_continuous_model(
            [a_train], [t_train], operators="cAH", regularizers=[1e-4]
        )

        rom_original = TrainedROM(
            model=train_res.model,
            encoder=encoder,
            reducer=reducer,
            metadata={"test_run": True},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = rom_original.save(tmpdir)

            # Check that all 6 required files exist
            assert (save_path / "metadata.json").is_file()
            assert (save_path / "mean_field.npz").is_file()
            assert (save_path / "pod_basis.npz").is_file()
            assert (save_path / "operators.h5").is_file()
            assert (save_path / "grid_metadata.npz").is_file()
            assert (save_path / "preprocessing.npz").is_file()

            # Load back
            rom_loaded = TrainedROM.load(save_path)

            # Verify equivalence
            assert rom_original.check_equivalence(rom_loaded) is True

            # Verify predictions match
            a0 = a_train[:, 0]
            pred_orig = rom_original.predict(a0, t_train[:10])
            pred_load = rom_loaded.predict(a0, t_train[:10])
            np.testing.assert_allclose(pred_orig, pred_load, rtol=1e-12)

            # Verify velocity decoding matches
            vel_orig = rom_original.decode_velocity(pred_orig)
            vel_load = rom_loaded.decode_velocity(pred_load)
            np.testing.assert_allclose(
                vel_orig[:, mask.mask, :], vel_load[:, mask.mask, :], rtol=1e-12
            )

            # Verify vorticity recovery
            vort_orig = rom_original.recover_vorticity(vel_orig)
            vort_load = rom_loaded.recover_vorticity(vel_load)
            np.testing.assert_allclose(
                vort_orig[:, mask.mask, :], vort_load[:, mask.mask, :], rtol=1e-12
            )
