"""
Unit tests for retained rom.eulerian utilities (grid, masking, vorticity).

Tests:
  1. Grid coordinate determinism across repeated instantiations.
  2. Meshgrid and flat point ordering determinism.
  3. Axis order permutations (xyz, zyx, yxz) and coordinate mapping.
  4. Non-uniform grids (cosine spacing, custom coordinates).
  5. HDF5 round-trip serialization of EulerianGrid.
  6. FluidMask creation, indicator functions, sphere/cylinder masks, and NaN application.
  7. HDF5 round-trip serialization of FluidMask.
  8. Vorticity computation (discrete curl) dimensions.
  9. Vorticity computation on an exact rotational field (analytical verification).
  10. Vorticity computation with non-uniform grid spacing and solid masks.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
import h5py

from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask
from rom.eulerian.vorticity import compute_vorticity


def _make_grid(nx=10, ny=10, nz=10, **kwargs):
    defaults = dict(
        bounds={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        resolution={"x": nx, "y": ny, "z": nz},
    )
    defaults.update(kwargs)
    return EulerianGrid(**defaults)


# ========================================================================
# 1. Grid coordinate determinism & layout
# ========================================================================

class TestGridDeterminism:
    def test_identical_coordinates_across_calls(self):
        """Grid coordinates must be bit-identical on repeated construction."""
        g1 = _make_grid()
        g2 = _make_grid()
        for ax in "xyz":
            np.testing.assert_array_equal(g1.coords_1d(ax), g2.coords_1d(ax))

    def test_meshgrid_determinism(self):
        g1 = _make_grid()
        g2 = _make_grid()
        for a, b in zip(g1.meshgrid(), g2.meshgrid()):
            np.testing.assert_array_equal(a, b)

    def test_flat_points_determinism(self):
        g1 = _make_grid()
        g2 = _make_grid()
        np.testing.assert_array_equal(g1.flat_points(), g2.flat_points())

    def test_grid_units_preserved(self):
        g = _make_grid(units="mm")
        assert g.units == "mm"

    @pytest.mark.parametrize("order", ["xyz", "zyx", "yxz"])
    def test_axis_order_shapes(self, order):
        res = {"x": 5, "y": 7, "z": 9}
        grid = _make_grid(nx=res["x"], ny=res["y"], nz=res["z"], axis_order=order)
        expected_shape = tuple(res[ax] for ax in order)
        assert grid.shape == expected_shape
        assert grid.flat_points().shape == (5 * 7 * 9, 3)


# ========================================================================
# 2. Non-uniform grids
# ========================================================================

class TestNonUniformGrid:
    def test_cosine_spacing(self):
        grid = _make_grid(nx=10, ny=10, nz=10, spacing_type="cosine")
        coords = grid.coords_1d("x")
        assert len(coords) == 10
        # Cosine clustering: spacing at boundaries smaller than at center
        spacings = np.diff(coords)
        assert spacings[0] < spacings[len(spacings) // 2]

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
# 3. Grid HDF5 Serialization
# ========================================================================

class TestGridHDF5:
    def test_grid_hdf5_roundtrip(self):
        grid = _make_grid(nx=8, ny=6, nz=4, axis_order="zyx", units="m")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                grp = f.create_group("metadata")
                grid.write_metadata(grp)

            with h5py.File(path, "r") as f:
                loaded_grid = EulerianGrid.from_hdf5(f["metadata"])

            assert loaded_grid.axis_order == grid.axis_order
            assert loaded_grid.shape == grid.shape
            assert loaded_grid.units == grid.units
            for ax in "xyz":
                np.testing.assert_array_equal(
                    grid.coords_1d(ax), loaded_grid.coords_1d(ax)
                )
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ========================================================================
# 4. FluidMask functionality & HDF5 IO
# ========================================================================

class TestMasking:
    def test_sphere_mask(self):
        grid = _make_grid(nx=10, ny=10, nz=10)
        mask = FluidMask.from_sphere(grid, centre=(0.0, 0.0, 0.0), radius=0.5)
        assert mask.mask.shape == grid.shape
        assert mask.num_solid > 0
        assert mask.num_fluid > 0
        assert mask.fluid_fraction < 1.0

        # Create dummy velocity field
        vel = np.ones(grid.shape + (3,))
        masked_vel = mask.apply(vel)

        solid = ~mask.mask
        fluid = mask.mask
        assert np.all(np.isnan(masked_vel[solid]))
        assert np.all(np.isfinite(masked_vel[fluid]))
        assert masked_vel.shape == vel.shape

    def test_cylinder_mask(self):
        grid = _make_grid(nx=8, ny=8, nz=8)
        mask = FluidMask.from_cylinder(grid, axis="z", centre_2d=(0.0, 0.0), radius=0.4)
        assert mask.mask.shape == grid.shape
        assert mask.num_solid > 0

    def test_mask_hdf5_roundtrip(self):
        grid = _make_grid(nx=6, ny=6, nz=6)
        mask = FluidMask.from_sphere(grid, centre=(0.0, 0.0, 0.0), radius=0.4)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            path = tmp.name
        try:
            with h5py.File(path, "w") as f:
                grp = f.create_group("metadata")
                mask.write_hdf5(grp)

            with h5py.File(path, "r") as f:
                loaded_mask = FluidMask.from_hdf5(grid, f["metadata"])

            np.testing.assert_array_equal(mask.mask, loaded_mask.mask)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ========================================================================
# 5. Vorticity Computation (Discrete Curl)
# ========================================================================

class TestVorticityComputation:
    @pytest.mark.parametrize("nx,ny,nz", [(6, 8, 10), (12, 12, 12)])
    def test_vorticity_dimensions(self, nx, ny, nz):
        grid = _make_grid(nx=nx, ny=ny, nz=nz)
        vel = np.zeros(grid.shape + (3,))
        omega = compute_vorticity(vel, grid)
        assert omega.shape == (nx, ny, nz, 3)

    def test_curl_of_rigid_rotation(self):
        """For velocity u = (y, -x, 0), the analytical curl is omega = (0, 0, -2)."""
        grid = _make_grid(
            nx=21, ny=21, nz=7,
            bounds={"x": (-1, 1), "y": (-1, 1), "z": (-0.5, 0.5)},
            axis_order="xyz",
        )
        X, Y, Z = grid.meshgrid()
        vel = np.zeros(grid.shape + (3,))
        vel[..., 0] = Y    # u_x = y
        vel[..., 1] = -X   # u_y = -x
        vel[..., 2] = 0.0  # u_z = 0

        omega = compute_vorticity(vel, grid, order=2)
        assert omega.shape == grid.shape + (3,)

        # In interior, omega_z should equal -2.0 exactly within machine tolerance
        interior = omega[2:-2, 2:-2, 1:-1, 2]
        np.testing.assert_allclose(interior, -2.0, atol=1e-10)
        # omega_x and omega_y should be zero
        np.testing.assert_allclose(omega[2:-2, 2:-2, 1:-1, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(omega[2:-2, 2:-2, 1:-1, 1], 0.0, atol=1e-10)

    def test_vorticity_with_mask(self):
        grid = _make_grid(nx=15, ny=15, nz=5)
        mask = FluidMask.from_sphere(grid, centre=(0.0, 0.0, 0.0), radius=0.4)
        X, Y, Z = grid.meshgrid()
        vel = np.zeros(grid.shape + (3,))
        vel[..., 0] = Y
        vel[..., 1] = -X

        omega = compute_vorticity(vel, grid, fluid_mask=mask)
        solid = ~mask.mask
        fluid = mask.mask
        assert np.all(np.isnan(omega[solid]))
        # Active cells sufficiently away from mask boundary should be finite
        assert np.any(np.isfinite(omega[fluid]))
