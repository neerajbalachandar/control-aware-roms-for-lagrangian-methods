"""
rom.eulerian – Eulerian grid, masking, and vorticity utilities.

Note on Architecture:
---------------------
Particle-to-grid evaluation (Biot-Savart summation) is handled directly by
FLOWUnsteady / rVPM via ``uns.computefluiddomain(pfield, grids)`` during or
after the simulation in Julia.  This module retains only the post-processing
utilities needed on the Python/ROM side:
  - EulerianGrid: Definition and HDF5 serialization of Cartesian grids.
  - FluidMask: Fluid/solid domain masks (e.g., solid boundary exclusion).
  - compute_vorticity: Numerical curl evaluation for vorticity recovery
    from ROM-decoded velocity fields.
"""

from .grid import EulerianGrid
from .masking import FluidMask
from .vorticity import compute_vorticity

__all__ = [
    "EulerianGrid",
    "FluidMask",
    "compute_vorticity",
]
