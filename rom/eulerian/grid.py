"""
rom.eulerian.grid – Fixed 3-D Cartesian Eulerian grid.

The grid is deterministic: given the same bounds, resolution, and axis order,
it produces identical coordinates on every call.  This guarantees that all
temporal snapshots share the same spatial discretisation.

Supports
--------
* Uniform and non-uniform (stretched) spacing along each axis.
* Arbitrary axis ordering (default: ``'xyz'``).
* Fluid-domain masks (see ``masking.py``).
* Serialisation of grid metadata to / from HDF5.

Coordinate convention
---------------------
Grid arrays are stored with shape ``(Nx, Ny, Nz)`` where the first index
corresponds to the first character of *axis_order*, etc.  A 3-component
vector field on this grid has shape ``(Nx, Ny, Nz, 3)`` with the last
dimension ordered as (u_x, u_y, u_z) in the **physical** (x, y, z) frame,
regardless of axis_order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EulerianGrid:
    """Immutable 3-D Cartesian grid.

    Parameters
    ----------
    bounds : dict
        ``{'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}``.
    resolution : dict
        ``{'x': Nx, 'y': Ny, 'z': Nz}`` – number of grid points per axis.
    axis_order : str, optional
        Permutation of ``'xyz'`` controlling storage layout (default ``'xyz'``).
    spacing_type : str, optional
        ``'uniform'`` (default) or ``'cosine'`` (cosine-clustered).
    custom_coords : dict or None, optional
        If provided, maps ``'x'``, ``'y'``, ``'z'`` to explicit 1-D arrays
        of grid coordinates.  Overrides *bounds*, *resolution*, and
        *spacing_type* for each axis present.
    units : str, optional
        Length-unit label (default ``'m'``).
    """

    bounds: Dict[str, Tuple[float, float]]
    resolution: Dict[str, int]
    axis_order: str = "xyz"
    spacing_type: str = "uniform"
    custom_coords: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    units: str = "m"

    # ---- derived / cached properties ----------------------------------

    def __post_init__(self):
        # Validate axis_order
        if sorted(self.axis_order) != ["x", "y", "z"]:
            raise ValueError(
                f"axis_order must be a permutation of 'xyz', got '{self.axis_order}'"
            )
        for ax in "xyz":
            if ax not in self.bounds:
                raise ValueError(f"bounds must contain key '{ax}'")
            if ax not in self.resolution:
                raise ValueError(f"resolution must contain key '{ax}'")
            lo, hi = self.bounds[ax]
            if lo >= hi:
                raise ValueError(
                    f"bounds['{ax}'] = ({lo}, {hi}): lower must be < upper"
                )
            if self.resolution[ax] < 2:
                raise ValueError(
                    f"resolution['{ax}'] = {self.resolution[ax]}: need >= 2"
                )
        if self.spacing_type not in ("uniform", "cosine"):
            raise ValueError(
                f"spacing_type must be 'uniform' or 'cosine', got '{self.spacing_type}'"
            )

    # ---- 1-D coordinate vectors ----------------------------------------

    def coords_1d(self, axis: str) -> np.ndarray:
        """Return the 1-D coordinate array for *axis* (``'x'``, ``'y'``, or ``'z'``)."""
        if self.custom_coords is not None and axis in self.custom_coords:
            return np.asarray(self.custom_coords[axis], dtype=np.float64)
        lo, hi = self.bounds[axis]
        n = self.resolution[axis]
        if self.spacing_type == "uniform":
            return np.linspace(lo, hi, n, dtype=np.float64)
        elif self.spacing_type == "cosine":
            # Cosine clustering: finer near boundaries
            eta = np.linspace(0.0, np.pi, n)
            return np.float64(0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(eta))
        # Should never reach here due to __post_init__ validation
        raise ValueError(f"Unknown spacing_type: {self.spacing_type}")  # pragma: no cover

    # ---- grid spacing --------------------------------------------------

    def spacing(self, axis: str) -> np.ndarray:
        """Return cell-spacing array of length ``N-1`` for *axis*."""
        c = self.coords_1d(axis)
        return np.diff(c)

    def mean_spacing(self) -> Dict[str, float]:
        """Return mean spacing per axis."""
        return {ax: float(np.mean(self.spacing(ax))) for ax in "xyz"}

    # ---- 3-D meshgrid --------------------------------------------------

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, Y, Z)`` meshgrid arrays ordered by *axis_order*.

        Each array has shape ``(N_{axis_order[0]}, N_{axis_order[1]}, N_{axis_order[2]})``.
        """
        # Build 1-D arrays in axis_order
        c = [self.coords_1d(ax) for ax in self.axis_order]
        return np.meshgrid(*c, indexing="ij")

    # ---- flat target points -------------------------------------------

    def flat_points(self) -> np.ndarray:
        """Return grid locations as ``(M, 3)`` array in *physical* (x, y, z) order.

        ``M = Nx * Ny * Nz``.
        """
        X, Y, Z = self.meshgrid()
        # meshgrid arrays are indexed by axis_order; we need to map back to xyz
        ax_map = {self.axis_order[i]: i for i in range(3)}
        grids = [X, Y, Z]  # indexed by axis_order position

        # Build columns in physical x, y, z order
        cols = []
        for phys_ax in "xyz":
            idx = ax_map[phys_ax]
            cols.append(grids[idx].ravel())
        return np.column_stack(cols)

    # ---- grid shape ---------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape ``(N0, N1, N2)`` following *axis_order*."""
        return tuple(self.resolution[ax] for ax in self.axis_order)

    @property
    def num_points(self) -> int:
        """Total number of grid points."""
        s = self.shape
        return s[0] * s[1] * s[2]

    # ---- HDF5 metadata -------------------------------------------------

    def write_metadata(self, grp) -> None:
        """Write grid metadata into an open HDF5 group *grp*."""
        for ax in "xyz":
            grp.create_dataset(f"coords_{ax}", data=self.coords_1d(ax))
            lo, hi = self.bounds[ax]
            grp.attrs[f"bounds_{ax}"] = [lo, hi]
            grp.attrs[f"resolution_{ax}"] = self.resolution[ax]
        grp.attrs["axis_order"] = self.axis_order
        grp.attrs["spacing_type"] = self.spacing_type
        grp.attrs["units"] = self.units

        # Store spacing arrays
        for ax in "xyz":
            grp.create_dataset(f"spacing_{ax}", data=self.spacing(ax))

        # Store the fluid mask placeholder flag
        grp.attrs["has_fluid_mask"] = False

    @classmethod
    def from_hdf5(cls, grp) -> "EulerianGrid":
        """Reconstruct a grid from metadata previously written by *write_metadata*."""
        bounds = {}
        resolution = {}
        custom = {}
        for ax in "xyz":
            b = grp.attrs[f"bounds_{ax}"]
            bounds[ax] = (float(b[0]), float(b[1]))
            resolution[ax] = int(grp.attrs[f"resolution_{ax}"])
            custom[ax] = np.array(grp[f"coords_{ax}"])
        axis_order = str(grp.attrs["axis_order"])
        spacing_type = str(grp.attrs.get("spacing_type", "uniform"))
        units = str(grp.attrs.get("units", "m"))
        return cls(
            bounds=bounds,
            resolution=resolution,
            axis_order=axis_order,
            spacing_type=spacing_type,
            custom_coords=custom,
            units=units,
        )
