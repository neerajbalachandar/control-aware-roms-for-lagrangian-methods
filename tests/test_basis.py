"""
Unit tests for rom.basis (VelocityStateEncoder, PODReducer, RankSelection).
"""

from __future__ import annotations

import numpy as np
import pytest

from rom.basis import (
    PODReducer,
    VelocityStateEncoder,
    evaluate_candidate_ranks,
)
from rom.eulerian import EulerianGrid, FluidMask


def _setup_encoder(with_mask: bool = True):
    grid = EulerianGrid(
        bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)},
        resolution={"x": 6, "y": 6, "z": 4},
    )
    mask = None
    if with_mask:
        mask = FluidMask.from_sphere(grid, centre=(0.5, 0.5, 0.5), radius=0.35)
    encoder = VelocityStateEncoder(grid, mask)
    return grid, mask, encoder


class TestVelocityStateEncoder:
    def test_encoder_dimensions(self):
        grid, mask, encoder = _setup_encoder(with_mask=True)
        assert encoder.n_fluid_cells == mask.num_fluid
        assert encoder.state_dimension == 3 * mask.num_fluid
        assert encoder.n_fluid_cells < grid.num_points

    def test_encode_decode_roundtrip(self):
        grid, mask, encoder = _setup_encoder(with_mask=True)
        k = 8
        # Create synthetic valid velocity field
        vel = np.random.randn(k, 6, 6, 4, 3)
        if mask is not None:
            vel[:, ~mask.mask, :] = np.nan

        Q = encoder.encode_trajectory(vel)
        assert Q.shape == (encoder.state_dimension, k)
        assert np.all(np.isfinite(Q))

        # Decode back to grid
        vel_rec = encoder.decode_trajectory(Q)
        assert vel_rec.shape == vel.shape

        # Compare at fluid cells
        fluid = mask.mask
        np.testing.assert_allclose(vel_rec[:, fluid, :], vel[:, fluid, :], rtol=1e-12)
        # Compare at solid cells
        solid = ~mask.mask
        assert np.all(np.isnan(vel_rec[:, solid, :]))

    def test_mean_centering(self):
        grid, mask, encoder = _setup_encoder(with_mask=False)
        k = 10
        vel = np.random.randn(k, 6, 6, 4, 3)
        Q = encoder.encode_trajectory(vel)

        mean = encoder.fit_mean(Q)
        assert mean.shape == (encoder.state_dimension,)
        np.testing.assert_allclose(mean, np.mean(Q, axis=1), rtol=1e-12)

        Q_centered = encoder.center(Q)
        np.testing.assert_allclose(np.mean(Q_centered, axis=1), 0.0, atol=1e-12)

        Q_restored = encoder.uncenter(Q_centered)
        np.testing.assert_allclose(Q_restored, Q, rtol=1e-12)


class TestPODReducer:
    @pytest.mark.parametrize("solver", ["dense", "method-of-snapshots", "randomized"])
    def test_pod_fitting_and_solvers(self, solver):
        n, k, r = 100, 25, 5
        rng = np.random.default_rng(42)
        # Low rank signal + small noise
        t = np.linspace(0, 1, k)
        modes = rng.standard_normal((n, r))
        coeffs = np.sin(np.outer(np.arange(1, r + 1), t))
        Q = modes @ coeffs

        reducer = PODReducer(r=r, svdsolver=solver)
        reducer.fit(Q)

        assert reducer.is_fitted is True
        assert reducer.r == r
        assert reducer.n == n
        assert reducer.basis_matrix.shape == (n, r)
        assert len(reducer.singular_values) > 0

        # Check orthonormality: V^T V ≈ I_r
        V = reducer.basis_matrix
        np.testing.assert_allclose(V.T @ V, np.eye(r), atol=1e-10)

        # Projection and reconstruction
        a = reducer.project(Q)
        assert a.shape == (r, k)

        Q_rec = reducer.reconstruct(a)
        assert Q_rec.shape == Q.shape
        rel_err = reducer.projection_error(Q)
        assert rel_err < 1e-10

    def test_set_rank(self):
        n, k = 50, 20
        rng = np.random.default_rng(123)
        Q = rng.standard_normal((n, k))
        reducer = PODReducer(r=10)
        reducer.fit(Q)
        assert reducer.r == 10

        reducer.set_rank(4)
        assert reducer.r == 4
        assert reducer.basis_matrix.shape == (n, 4)


class TestRankSelection:
    def test_rank_evaluation_and_recommendation(self):
        n, k = 80, 30
        rng = np.random.default_rng(99)
        # Create signal with fast decaying singular values
        U, _ = np.linalg.qr(rng.standard_normal((n, 10)))
        s = 10.0 ** (-np.arange(10))
        V = rng.standard_normal((10, k))
        Q_train = U @ np.diag(s) @ V
        Q_val = U @ np.diag(s) @ rng.standard_normal((10, 15))

        reducer = PODReducer(r=10)
        reducer.fit(Q_train)

        candidates = [1, 2, 3, 5, 8]
        result = evaluate_candidate_ranks(
            reducer, candidates, Q_train, Q_val, energy_target=0.999
        )

        assert len(result.metrics) == len(candidates)
        assert result.recommended_rank in candidates
        assert result.metrics[-1].train_projection_error <= result.metrics[0].train_projection_error
        table = result.as_table()
        assert "| Rank |" in table
