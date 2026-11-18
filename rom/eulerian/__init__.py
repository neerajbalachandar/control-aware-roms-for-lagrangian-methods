"""
rom.eulerian – Eulerian field reconstruction from Lagrangian vortex-particle data.

This package converts FLOWUnsteady / rVPM particle snapshots (positions,
circulation vectors, core sizes) into velocity and vorticity fields on a
fixed Cartesian grid, stored as chunked HDF5 datasets.

Sub-modules
-----------
grid               Fixed 3-D Cartesian grid with optional non-uniform spacing.
field_reconstructor Velocity evaluation via regularised Biot–Savart kernels.
vorticity           Curl computation (ω = ∇ × u) on structured grids.
masking             Fluid-domain masks for solid boundaries.
"""

from .grid import EulerianGrid
from .field_reconstructor import FieldReconstructor
from .vorticity import compute_vorticity
from .masking import FluidMask

__all__ = [
    "EulerianGrid",
    "FieldReconstructor",
    "compute_vorticity",
    "FluidMask",
]
