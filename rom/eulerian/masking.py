"""
rom.eulerian.masking – Fluid-domain masks for solid boundaries.

A fluid mask is a boolean array with the same spatial shape as the Eulerian
grid.  ``True`` marks a fluid cell; ``False`` marks a solid cell that should
be excluded from physical analysis (e.g., inside a rotor blade or nacelle).

Masks can be constructed from:
  * An analytic signed-distance / indicator function.
  * A list of solid-body surface meshes (future, not yet implemented).
  * A pre-computed boolean array.

Solid-region treatment
----------------------
Velocity and vorticity values in masked (solid) cells are set to ``NaN`` so
that downstream consumers never silently treat them as valid fluid data.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .grid import EulerianGrid


class FluidMask:
    """Boolean fluid-domain mask on an :class:`EulerianGrid`.

    Parameters
    ----------
    grid : EulerianGrid
        The grid on which the mask is defined.
    mask : np.ndarray or None
        Boolean array of shape ``grid.shape``.  ``True`` = fluid, ``False`` =
        solid.  If *None*, the entire domain is treated as fluid.
    """

    def __init__(self, grid: EulerianGrid, mask: Optional[np.ndarray] = None):
        self.grid = grid
        if mask is None:
            self._mask = np.ones(grid.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != grid.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match grid shape {grid.shape}"
                )
            self._mask = mask

    # ---- properties ---------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask array: ``True`` = fluid, ``False`` = solid."""
        return self._mask

    @property
    def num_fluid(self) -> int:
        return int(np.count_nonzero(self._mask))

    @property
    def num_solid(self) -> int:
        return int(np.size(self._mask) - self.num_fluid)

    @property
    def fluid_fraction(self) -> float:
        return self.num_fluid / self._mask.size

    # ---- construction helpers ------------------------------------------

    @classmethod
    def from_indicator(
        cls,
        grid: EulerianGrid,
        indicator: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> "FluidMask":
        """Build a mask from an indicator function.

        Parameters
        ----------
        grid : EulerianGrid
        indicator : callable
            ``indicator(X, Y, Z)`` → boolean array.  Returns ``True`` where
            the point is in the fluid domain.  X, Y, Z are 3-D meshgrid
            arrays ordered by *grid.axis_order*.
        """
        X, Y, Z = grid.meshgrid()
        # Map meshgrid arrays (which are in axis_order) to physical x, y, z
        ax_map = {grid.axis_order[i]: i for i in range(3)}
        grids_ordered = [X, Y, Z]  # in axis_order
        phys_X = grids_ordered[ax_map["x"]]
        phys_Y = grids_ordered[ax_map["y"]]
        phys_Z = grids_ordered[ax_map["z"]]
        m = np.asarray(indicator(phys_X, phys_Y, phys_Z), dtype=bool)
        return cls(grid, m)

    @classmethod
    def from_sphere(
        cls,
        grid: EulerianGrid,
        centre: Tuple[float, float, float],
        radius: float,
        inside_is_solid: bool = True,
    ) -> "FluidMask":
        """Convenience: mask a sphere (solid inside by default)."""
        cx, cy, cz = centre

        def indicator(X, Y, Z):
            r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            if inside_is_solid:
                return r2 > radius ** 2
            else:
                return r2 <= radius ** 2

        return cls.from_indicator(grid, indicator)

    @classmethod
    def from_cylinder(
        cls,
        grid: EulerianGrid,
        axis: str,
        centre_2d: Tuple[float, float],
        radius: float,
        inside_is_solid: bool = True,
    ) -> "FluidMask":
        """Mask an infinite cylinder aligned with *axis* (``'x'``, ``'y'``, or ``'z'``)."""
        other = [a for a in "xyz" if a != axis]

        def indicator(X, Y, Z):
            coords = {"x": X, "y": Y, "z": Z}
            c1, c2 = coords[other[0]], coords[other[1]]
            r2 = (c1 - centre_2d[0]) ** 2 + (c2 - centre_2d[1]) ** 2
            if inside_is_solid:
                return r2 > radius ** 2
            else:
                return r2 <= radius ** 2

        return cls.from_indicator(grid, indicator)

    # ---- application --------------------------------------------------

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Set solid-region entries of *field* to ``NaN``.

        Parameters
        ----------
        field : np.ndarray
            Shape ``(Nx, Ny, Nz)`` for scalars or ``(Nx, Ny, Nz, 3)`` for
            vectors.

        Returns
        -------
        np.ndarray
            A **copy** with NaN in solid cells.
        """
        out = field.astype(np.float64, copy=True)
        if out.ndim == 3:
            out[~self._mask] = np.nan
        elif out.ndim == 4 and out.shape[-1] == 3:
            out[~self._mask, :] = np.nan
        else:
            raise ValueError(
                f"Expected shape {self.grid.shape} or {self.grid.shape + (3,)}, "
                f"got {field.shape}"
            )
        return out

    # ---- HDF5 IO -------------------------------------------------------

    def write_hdf5(self, grp) -> None:
        """Write mask into an HDF5 group."""
        grp.create_dataset("fluid_mask", data=self._mask, compression="gzip")

    @classmethod
    def from_hdf5(cls, grid: EulerianGrid, grp) -> "FluidMask":
        """Read mask from an HDF5 group."""
        m = np.array(grp["fluid_mask"], dtype=bool)
        return cls(grid, m)
