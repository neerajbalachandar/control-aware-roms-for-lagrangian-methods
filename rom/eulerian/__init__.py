"""
rom.eulerian - Lagrangian vortex particles -> Eulerian fields on a fixed grid.

VarFlExI (FLOWUnsteady rVPM + FEniCS) never writes Eulerian fields during a run;
everything needed to build them is here:

* ``grid.EulerianGrid``        fixed Cartesian grid (shared by all runs)
* ``kernels``                  exact FLOWVPM regularised kernels (g, zeta)
* ``biot_savart``              direct particle->grid sums (torch GPU / numpy), far-field lumping
* ``vic``                      vortex-in-cell: P2G splat + free-space FFT Biot-Savart
* ``bound_vortex``             VLM bound vorticity from the station history
* ``reconstruct``              whole-run driver -> ROM trajectory file
* ``julia_bridge``             convert FLOWUnsteady ``computefluiddomain`` (FMM) output
* ``vorticity.compute_vorticity``  finite-difference curl (for ROM-decoded velocity)
* ``masking.FluidMask``        optional solid masks
* ``xdmf``                     ParaView descriptors
"""

from .biot_savart import evaluate_particles, lump_far_particles, segment_velocity
from .grid import EulerianGrid
from .kernels import KERNELS, g_kernel, zeta_kernel
from .masking import FluidMask
from .reconstruct import (EulerianReconstructor, ReconstructionConfig, cell_volumes, grid_from_config,
                          reconstruct_run, structure_on_steps)
from .vic import fft_biot_savart, splat_vorticity, vic_reconstruct
from .vorticity import compute_vorticity
from .xdmf import write_trajectory_xdmf

__all__ = [
    "EulerianGrid", "FluidMask", "compute_vorticity", "KERNELS", "g_kernel", "zeta_kernel",
    "evaluate_particles", "lump_far_particles", "segment_velocity", "splat_vorticity", "fft_biot_savart",
    "vic_reconstruct", "EulerianReconstructor", "ReconstructionConfig", "grid_from_config", "cell_volumes",
    "reconstruct_run", "structure_on_steps", "write_trajectory_xdmf",
]
