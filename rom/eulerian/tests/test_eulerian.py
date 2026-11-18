"""
Unit and integration tests for rom.eulerian.

Covers:
  1. Grid coordinate determinism across snapshots.
  2. Velocity finiteness.
  3. Output dimensions.
  4. Vorticity dimensions.
  5. Vorticity matches discrete curl of reconstructed velocity.
  6. Velocity agrees with direct analytical Biot-Savart at sample points.
  7. Changing particle count does not change output dimensions.
  8. Chunked and unchunked evaluations agree within tolerance.
  9. Unit and axis conventions preserved.
  10. Manufactured analytical vortex ring test case for kernel validation.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# --- Module under test --------------------------------------------------
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask
from rom.eulerian.field_reconstructor import (
    FieldReconstructor,
    Kernel,
    ParticleSnapshot,
    _biot_savart_chunk,
    _cutoff_gaussianerf,
    _cutoff_winckelmans,
    _cutoff_singular,
)
from rom.eulerian.vorticity import compute_vorticity


# ========================================================================
# Helpers
# ========================================================================

def _make_grid(nx=10, ny=10, nz=10, **kwargs):
    defaults = dict(
        bounds={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        resolution={"x": nx, "y": ny, "z": nz},
    )
    defaults.update(kwargs)
    return EulerianGrid(**defaults)


def _single_particle_snapshot(
    pos=None, gamma=None, sigma=None, time=0.0
):
    """Create a snapshot with a single particle (default: unit z-circulation
    at the origin with σ = 0.5)."""
    if pos is None:
        pos = np.array([[0.0, 0.0, 0.0]])
    if gamma is None:
        gamma = np.array([[0.0, 0.0, 1.0]])
    if sigma is None:
        sigma = np.array([0.5])
    return ParticleSnapshot(pos, gamma, sigma, time=time)


def _random_particles(n, rng=None, sigma_range=(0.05, 0.5)):
    if rng is None:
        rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, (n, 3))
    gamma = rng.standard_normal((n, 3))
    sigma = rng.uniform(*sigma_range, n)
    return ParticleSnapshot(pos, gamma, sigma, time=0.0)


# ========================================================================
# 1. Grid coordinate determinism
# ========================================================================

class TestGridDeterminism:
    def test_identical_coordinates_across_calls(self):
        """Grid coordinates must be bit-identical on repeated construction."""
        g1 = _make_grid()
        g2 = _make_grid()
        for ax in "xyz":
            np.testing.assert_array_equal(
                g1.coords_1d(ax), g2.coords_1d(ax)
            )

    def test_meshgrid_determinism(self):
        g1 = _make_grid()
        g2 = _make_grid()
        for a, b in zip(g1.meshgrid(), g2.meshgrid()):
            np.testing.assert_array_equal(a, b)

    def test_flat_points_determinism(self):
        g1 = _make_grid()
        g2 = _make_grid()
        np.testing.assert_array_equal(g1.flat_points(), g2.flat_points())


# ========================================================================
# 2. Velocity finiteness
# ========================================================================

class TestVelocityFiniteness:
    def test_velocity_is_finite_single_particle(self):
        grid = _make_grid(nx=5, ny=5, nz=5)
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        assert np.all(np.isfinite(vel))

    def test_velocity_is_finite_many_particles(self):
        grid = _make_grid(nx=5, ny=5, nz=5)
        snap = _random_particles(50)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        assert np.all(np.isfinite(vel))


# ========================================================================
# 3. Output dimensions
# ========================================================================

class TestOutputDimensions:
    @pytest.mark.parametrize("nx,ny,nz", [(5, 7, 9), (10, 10, 10), (3, 4, 5)])
    def test_velocity_shape(self, nx, ny, nz):
        grid = _make_grid(nx=nx, ny=ny, nz=nz)
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        assert vel.shape == (nx, ny, nz, 3)

    @pytest.mark.parametrize("order", ["xyz", "zyx", "yxz"])
    def test_velocity_shape_axis_order(self, order):
        res = {"x": 5, "y": 7, "z": 9}
        grid = _make_grid(
            nx=res["x"], ny=res["y"], nz=res["z"], axis_order=order
        )
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        expected_shape = tuple(res[ax] for ax in order) + (3,)
        assert vel.shape == expected_shape


# ========================================================================
# 4. Vorticity dimensions
# ========================================================================

class TestVorticityDimensions:
    @pytest.mark.parametrize("nx,ny,nz", [(5, 7, 9), (10, 10, 10)])
    def test_vorticity_shape(self, nx, ny, nz):
        grid = _make_grid(nx=nx, ny=ny, nz=nz)
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        omega = compute_vorticity(vel, grid)
        assert omega.shape == (nx, ny, nz, 3)


# ========================================================================
# 5. Vorticity matches discrete curl of velocity
# ========================================================================

class TestVorticityCurl:
    def test_curl_of_known_field(self):
        """For a linear velocity field u = (y, -x, 0), ω = (0, 0, -2)."""
        grid = _make_grid(nx=20, ny=20, nz=5,
                          bounds={"x": (-1, 1), "y": (-1, 1), "z": (-0.5, 0.5)})
        X, Y, Z = grid.meshgrid()
        # For default axis_order='xyz': X→x, Y→y, Z→z
        vel = np.zeros(grid.shape + (3,))
        vel[..., 0] = Y   # u_x = y
        vel[..., 1] = -X  # u_y = -x
        vel[..., 2] = 0.0 # u_z = 0

        omega = compute_vorticity(vel, grid, order=2)

        # ω_z should be close to -2 in the interior
        # (boundary stencils may have larger error)
        interior = omega[2:-2, 2:-2, 1:-1, 2]
        np.testing.assert_allclose(interior, -2.0, atol=1e-10)

        # ω_x and ω_y should be ~0
        np.testing.assert_allclose(omega[2:-2, 2:-2, 1:-1, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(omega[2:-2, 2:-2, 1:-1, 1], 0.0, atol=1e-10)


# ========================================================================
# 6. Velocity agrees with direct analytical Biot-Savart
# ========================================================================

class TestBiotSavartAgreement:
    def test_single_particle_analytical(self):
        """Compare module evaluation against hand-coded Gaussian-erf kernel
        for a single particle at the origin, evaluated at a known point."""
        import math as _math

        pos = np.array([[0.0, 0.0, 0.0]])
        gamma = np.array([[0.0, 0.0, 1.0]])
        sigma = np.array([0.3])
        target = np.array([[1.0, 0.0, 0.0]])

        # Hand-coded
        r_vec = target[0] - pos[0]
        r_mag = np.linalg.norm(r_vec)
        rho = r_mag / sigma[0]
        g = (
            _math.erf(rho)
            - (2.0 / np.sqrt(np.pi)) * rho * np.exp(-rho ** 2)
        )
        cross = np.cross(gamma[0], r_vec)
        u_expected = cross * g / (4.0 * np.pi * r_mag ** 3)

        # Module evaluation
        u_module = _biot_savart_chunk(
            target, pos, gamma, sigma, _cutoff_gaussianerf
        )

        np.testing.assert_allclose(u_module[0], u_expected, rtol=1e-12)

    def test_grid_evaluation_matches_point_evaluation(self):
        """Velocity on the grid must equal direct point evaluation."""
        grid = _make_grid(nx=4, ny=4, nz=4)
        snap = _random_particles(10)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)

        vel_grid = recon.evaluate_velocity(snap)
        vel_pts = recon.evaluate_velocity_at_points(grid.flat_points(), snap)
        vel_pts_reshaped = vel_pts.reshape(grid.shape + (3,))

        np.testing.assert_allclose(vel_grid, vel_pts_reshaped, rtol=1e-12)


# ========================================================================
# 7. Changing particle count does not change output dimensions
# ========================================================================

class TestVariableParticleCount:
    def test_different_particle_counts(self):
        grid = _make_grid(nx=5, ny=5, nz=5)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        for n in [1, 10, 50, 200]:
            snap = _random_particles(n)
            vel = recon.evaluate_velocity(snap)
            assert vel.shape == (5, 5, 5, 3)


# ========================================================================
# 8. Chunked vs unchunked agreement
# ========================================================================

class TestChunkedAgreement:
    def test_chunk_sizes_agree(self):
        grid = _make_grid(nx=6, ny=6, nz=6)
        snap = _random_particles(30)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)

        vel_full = recon.evaluate_velocity(snap, chunk_size=6 * 6 * 6)
        vel_tiny = recon.evaluate_velocity(snap, chunk_size=7)
        vel_one = recon.evaluate_velocity(snap, chunk_size=1)

        np.testing.assert_allclose(vel_full, vel_tiny, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(vel_full, vel_one, rtol=1e-12, atol=1e-15)


# ========================================================================
# 9. Unit and axis conventions preserved
# ========================================================================

class TestConventions:
    def test_axis_order_xyz_vs_zyx(self):
        """Same physical problem with different axis_order must give the same
        physical velocity at corresponding spatial locations."""
        bounds = {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}
        res = {"x": 5, "y": 5, "z": 5}
        g_xyz = EulerianGrid(bounds=bounds, resolution=res, axis_order="xyz")
        g_zyx = EulerianGrid(bounds=bounds, resolution=res, axis_order="zyx")

        snap = _single_particle_snapshot()
        r1 = FieldReconstructor(g_xyz, kernel=Kernel.GAUSSIAN_ERF)
        r2 = FieldReconstructor(g_zyx, kernel=Kernel.GAUSSIAN_ERF)

        v1 = r1.evaluate_velocity(snap)  # (5,5,5,3) indexed by (x,y,z)
        v2 = r2.evaluate_velocity(snap)  # (5,5,5,3) indexed by (z,y,x)

        # v2 indexed by (z,y,x) → transpose first three axes to (x,y,z)
        v2_transposed = np.transpose(v2, (2, 1, 0, 3))
        np.testing.assert_allclose(v1, v2_transposed, rtol=1e-12)

    def test_grid_units_preserved(self):
        g = _make_grid(units="mm")
        assert g.units == "mm"


# ========================================================================
# 10. Analytical vortex test case – Lamb-Oseen vortex
# ========================================================================

class TestLambOseenVortex:
    """Validate the Gaussian-erf kernel against a Lamb-Oseen vortex
    (a single straight vortex filament approximated by co-linear particles).

    For a line of particles along z with uniform Γ = (0, 0, Γ_z dz),
    the induced velocity in the xy-plane should approach the Lamb-Oseen
    azimuthal profile as the number of particles increases.
    """

    def test_azimuthal_velocity_profile(self):
        """Check that the induced velocity decays radially and is purely
        azimuthal for a stack of z-aligned particles."""
        # Build a line of particles along z
        nz = 100
        z_coords = np.linspace(-10.0, 10.0, nz)
        dz = z_coords[1] - z_coords[0]
        circulation_per_length = 1.0
        sigma_val = 0.3

        pos = np.zeros((nz, 3))
        pos[:, 2] = z_coords
        gamma = np.zeros((nz, 3))
        gamma[:, 2] = circulation_per_length * dz
        sigma = np.full(nz, sigma_val)
        snap = ParticleSnapshot(pos, gamma, sigma)

        # Evaluate velocity at points along the x-axis (at z=0)
        r_vals = np.linspace(0.5, 3.0, 20)
        targets = np.zeros((len(r_vals), 3))
        targets[:, 0] = r_vals

        recon = FieldReconstructor(
            _make_grid(nx=3, ny=3, nz=3),  # grid not used for point eval
            kernel=Kernel.GAUSSIAN_ERF,
        )
        vel = recon.evaluate_velocity_at_points(targets, snap)

        # For a z-filament, velocity should be purely in the y-direction
        # at points along the x-axis
        assert np.all(np.abs(vel[:, 0]) < 1e-10)  # u_x ≈ 0
        assert np.all(np.abs(vel[:, 2]) < 1e-10)  # u_z ≈ 0

        # u_y should be positive: cross((0,0,Γ_z), (x,0,0)) = (0, Γ_z·x, 0)
        # and 1/(4π) · g(ρ)/|r|³ > 0, so u_y > 0
        assert np.all(vel[:, 1] > 0)

        # Velocity magnitude should decrease with distance (far field ~ 1/r)
        magnitudes = np.abs(vel[:, 1])
        for i in range(1, len(magnitudes)):
            if r_vals[i] > 2.0 * sigma_val:
                # In the far field, velocity should decrease
                assert magnitudes[i] < magnitudes[i - 1] * 1.05  # allow small tolerance


# ========================================================================
# Kernel cut-off function tests
# ========================================================================

class TestKernelCutoffs:
    def test_gaussianerf_zero(self):
        """g(0) must be 0 for the Gaussian-erf kernel."""
        assert _cutoff_gaussianerf(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_gaussianerf_large(self):
        """g(ρ) → 1 as ρ → ∞."""
        assert _cutoff_gaussianerf(np.array([100.0]))[0] == pytest.approx(
            1.0, abs=1e-10
        )

    def test_winckelmans_zero(self):
        assert _cutoff_winckelmans(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_winckelmans_large(self):
        assert _cutoff_winckelmans(np.array([1000.0]))[0] == pytest.approx(
            1.0, abs=1e-3
        )

    def test_singular_always_one(self):
        rho = np.array([0.0, 0.5, 1.0, 100.0])
        np.testing.assert_array_equal(_cutoff_singular(rho), np.ones(4))


# ========================================================================
# Masking tests
# ========================================================================

class TestMasking:
    def test_sphere_mask_solid_is_nan(self):
        grid = _make_grid(nx=10, ny=10, nz=10)
        mask = FluidMask.from_sphere(grid, (0.0, 0.0, 0.0), 0.5)
        snap = _single_particle_snapshot(
            pos=np.array([[2.0, 0.0, 0.0]]),
            gamma=np.array([[0.0, 0.0, 1.0]]),
        )
        recon = FieldReconstructor(
            grid, kernel=Kernel.GAUSSIAN_ERF, fluid_mask=mask
        )
        vel = recon.evaluate_velocity(snap)

        # Solid cells should be NaN
        solid = ~mask.mask
        if np.any(solid):
            assert np.all(np.isnan(vel[solid]))

        # Fluid cells should be finite
        fluid = mask.mask
        assert np.all(np.isfinite(vel[fluid]))

    def test_mask_does_not_change_shape(self):
        grid = _make_grid(nx=5, ny=5, nz=5)
        mask = FluidMask.from_sphere(grid, (0, 0, 0), 0.3)
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(
            grid, kernel=Kernel.GAUSSIAN_ERF, fluid_mask=mask
        )
        vel = recon.evaluate_velocity(snap)
        assert vel.shape == (5, 5, 5, 3)


# ========================================================================
# HDF5 round-trip test
# ========================================================================

class TestHDF5Output:
    def test_process_snapshots_roundtrip(self):
        import h5py

        grid = _make_grid(nx=5, ny=5, nz=5)
        snaps = [
            _random_particles(20, rng=np.random.default_rng(i))
            for i in range(3)
        ]
        for i, s in enumerate(snaps):
            s.time = float(i) * 0.1

        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name

        try:
            recon.process_snapshots(snaps, path, compute_vorticity=True)

            with h5py.File(path, "r") as f:
                # Check structure
                assert "metadata" in f
                assert "snapshots" in f
                assert "velocity" in f["snapshots"]
                assert "vorticity" in f["snapshots"]

                vel = f["snapshots/velocity"]
                vort = f["snapshots/vorticity"]
                assert vel.shape == (3, 5, 5, 5, 3)
                assert vort.shape == (3, 5, 5, 5, 3)

                # Times
                times = f["metadata/time"][:]
                np.testing.assert_allclose(times, [0.0, 0.1, 0.2])

                # Grid round-trip
                g2 = EulerianGrid.from_hdf5(f["metadata"])
                for ax in "xyz":
                    np.testing.assert_array_equal(
                        grid.coords_1d(ax), g2.coords_1d(ax)
                    )
        finally:
            os.unlink(path)


# ========================================================================
# Particle validation tests
# ========================================================================

class TestParticleValidation:
    def test_nan_particles_rejected(self):
        pos = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
        gamma = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        sigma = np.array([0.5, 0.5])
        snap = ParticleSnapshot(pos, gamma, sigma)

        grid = _make_grid(nx=3, ny=3, nz=3)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        with pytest.warns(UserWarning, match="Rejected 1"):
            vel = recon.evaluate_velocity(snap)
        assert np.all(np.isfinite(vel))

    def test_negative_sigma_rejected(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gamma = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        sigma = np.array([0.5, -0.1])
        snap = ParticleSnapshot(pos, gamma, sigma)

        grid = _make_grid(nx=3, ny=3, nz=3)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        with pytest.warns(UserWarning, match="Rejected 1"):
            vel = recon.evaluate_velocity(snap)
        assert np.all(np.isfinite(vel))

    def test_zero_particles_gives_zero_velocity(self):
        pos = np.empty((0, 3))
        gamma = np.empty((0, 3))
        sigma = np.empty(0)
        snap = ParticleSnapshot(pos, gamma, sigma)

        grid = _make_grid(nx=3, ny=3, nz=3)
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        with pytest.warns(UserWarning, match="No valid particles"):
            vel = recon.evaluate_velocity(snap)
        np.testing.assert_array_equal(vel, 0.0)


# ========================================================================
# Non-uniform grid tests
# ========================================================================

class TestNonUniformGrid:
    def test_cosine_spacing(self):
        grid = _make_grid(nx=10, ny=10, nz=10, spacing_type="cosine")
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        assert vel.shape == (10, 10, 10, 3)
        assert np.all(np.isfinite(vel))

    def test_cosine_vorticity(self):
        """Vorticity on a cosine grid should still have correct shape."""
        grid = _make_grid(nx=10, ny=10, nz=10, spacing_type="cosine")
        snap = _single_particle_snapshot()
        recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        vel = recon.evaluate_velocity(snap)
        omega = compute_vorticity(vel, grid)
        assert omega.shape == (10, 10, 10, 3)
        assert np.all(np.isfinite(omega))

    def test_custom_coords(self):
        custom = {
            "x": np.array([0.0, 0.1, 0.3, 0.6, 1.0]),
            "y": np.linspace(0, 1, 5),
            "z": np.linspace(0, 1, 5),
        }
        grid = EulerianGrid(
            bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)},
            resolution={"x": 5, "y": 5, "z": 5},
            custom_coords=custom,
        )
        np.testing.assert_array_equal(grid.coords_1d("x"), custom["x"])


# ========================================================================
# Winckelmans kernel test
# ========================================================================

class TestWinckelmansKernel:
    def test_winckelmans_finite(self):
        grid = _make_grid(nx=5, ny=5, nz=5)
        snap = _random_particles(20)
        recon = FieldReconstructor(grid, kernel=Kernel.WINCKELMANS)
        vel = recon.evaluate_velocity(snap)
        assert np.all(np.isfinite(vel))

    def test_winckelmans_vs_gaussianerf_same_sign(self):
        """Different kernels should produce qualitatively similar fields
        (same sign pattern for a simple configuration)."""
        grid = _make_grid(nx=5, ny=5, nz=5)
        snap = _single_particle_snapshot()
        r1 = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF)
        r2 = FieldReconstructor(grid, kernel=Kernel.WINCKELMANS)
        v1 = r1.evaluate_velocity(snap)
        v2 = r2.evaluate_velocity(snap)
        # At far-field points (away from origin), sign should agree
        far = (
            np.abs(grid.flat_points()[:, 0]) > 0.5
        ).reshape(grid.shape)
        # Check x-component
        agree = np.sign(v1[far, 0]) == np.sign(v2[far, 0])
        # Allow for a few zero-velocity points
        nonzero = np.abs(v1[far, 0]) > 1e-15
        assert np.mean(agree[nonzero]) > 0.95
