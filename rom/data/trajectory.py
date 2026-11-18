"""
rom.data.trajectory – In-memory representation of VarFlExI simulation trajectories.

A Trajectory encapsulates time history, Eulerian flow fields (velocity, vorticity),
grid discretisation, fluid domain masks, and optional structural / aeroelastic states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask


@dataclass
class Trajectory:
    """Represents a single continuous simulation trajectory.

    Parameters
    ----------
    time : np.ndarray
        1-D array of shape ``(K,)`` containing strictly increasing timestamps.
    velocity : np.ndarray
        5-D array of shape ``(K, N0, N1, N2, 3)`` containing Eulerian velocity fields
        (u_x, u_y, u_z) in the physical coordinate frame.
    grid : EulerianGrid
        Discretisation grid matching the spatial shape ``(N0, N1, N2)``.
    vorticity : np.ndarray, optional
        5-D array of shape ``(K, N0, N1, N2, 3)`` containing vorticity fields (omega_x, omega_y, omega_z).
    fluid_mask : FluidMask, optional
        Boolean fluid mask on the grid.
    structural_displacement : np.ndarray, optional
        Structural displacements of shape ``(K, ...)``, if aeroelastic data is present.
    structural_velocity : np.ndarray, optional
        Structural velocities of shape ``(K, ...)``.
    structural_acceleration : np.ndarray, optional
        Structural accelerations of shape ``(K, ...)``.
    aerodynamic_forces : np.ndarray, optional
        Integrated or distributed aerodynamic forces of shape ``(K, ...)``.
    parameters : dict, optional
        Simulation parameters (e.g., Reynolds number, inflow velocity, RPM).
    coupling_data : dict, optional
        Coupling interface states or loads.
    name : str, optional
        Trajectory identifier.
    """

    time: np.ndarray
    velocity: np.ndarray
    grid: EulerianGrid
    vorticity: Optional[np.ndarray] = None
    fluid_mask: Optional[FluidMask] = None
    structural_displacement: Optional[np.ndarray] = None
    structural_velocity: Optional[np.ndarray] = None
    structural_acceleration: Optional[np.ndarray] = None
    aerodynamic_forces: Optional[np.ndarray] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    coupling_data: Dict[str, Any] = field(default_factory=dict)
    name: str = "trajectory"

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.vorticity is not None:
            self.vorticity = np.asarray(self.vorticity, dtype=np.float64)
        if self.structural_displacement is not None:
            self.structural_displacement = np.asarray(self.structural_displacement, dtype=np.float64)
        if self.structural_velocity is not None:
            self.structural_velocity = np.asarray(self.structural_velocity, dtype=np.float64)
        if self.structural_acceleration is not None:
            self.structural_acceleration = np.asarray(self.structural_acceleration, dtype=np.float64)
        if self.aerodynamic_forces is not None:
            self.aerodynamic_forces = np.asarray(self.aerodynamic_forces, dtype=np.float64)

    def __len__(self) -> int:
        """Return the number of temporal snapshots K."""
        return len(self.time)

    @property
    def num_snapshots(self) -> int:
        """Number of temporal snapshots."""
        return len(self.time)

    @property
    def dt(self) -> Union[float, np.ndarray]:
        """Time step size. Returns scalar float if uniform, otherwise 1-D array of diffs."""
        if len(self.time) < 2:
            return 0.0
        diffs = np.diff(self.time)
        if self.is_uniform_time:
            return float(np.mean(diffs))
        return diffs

    @property
    def is_uniform_time(self) -> bool:
        """Check if temporal sampling is uniform within standard tolerance."""
        if len(self.time) < 2:
            return True
        diffs = np.diff(self.time)
        mean_dt = np.mean(diffs)
        if mean_dt <= 0:
            return False
        return bool(np.allclose(diffs, mean_dt, rtol=1e-4, atol=1e-8))

    @property
    def time_span(self) -> Tuple[float, float]:
        """Return (t_start, t_end)."""
        if len(self.time) == 0:
            return (0.0, 0.0)
        return (float(self.time[0]), float(self.time[-1]))

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> "Trajectory":
        """Extract a sub-trajectory over a slice of snapshot indices.

        Useful for chronological train / validation / test partitioning.
        """
        slc = slice(start, stop, step)
        sub_time = self.time[slc].copy()
        sub_vel = self.velocity[slc].copy()
        sub_vort = self.vorticity[slc].copy() if self.vorticity is not None else None
        sub_disp = self.structural_displacement[slc].copy() if self.structural_displacement is not None else None
        sub_vel_s = self.structural_velocity[slc].copy() if self.structural_velocity is not None else None
        sub_acc_s = self.structural_acceleration[slc].copy() if self.structural_acceleration is not None else None
        sub_forces = self.aerodynamic_forces[slc].copy() if self.aerodynamic_forces is not None else None

        return Trajectory(
            time=sub_time,
            velocity=sub_vel,
            grid=self.grid,
            vorticity=sub_vort,
            fluid_mask=self.fluid_mask,
            structural_displacement=sub_disp,
            structural_velocity=sub_vel_s,
            structural_acceleration=sub_acc_s,
            aerodynamic_forces=sub_forces,
            parameters=dict(self.parameters),
            coupling_data=dict(self.coupling_data),
            name=f"{self.name}[{start}:{stop}:{step}]",
        )

    def summary(self) -> Dict[str, Any]:
        """Return summary dictionary describing trajectory content."""
        return {
            "name": self.name,
            "num_snapshots": self.num_snapshots,
            "time_span": self.time_span,
            "is_uniform_time": self.is_uniform_time,
            "dt": self.dt if self.is_uniform_time else "nonuniform",
            "velocity_shape": self.velocity.shape,
            "has_vorticity": self.vorticity is not None,
            "has_fluid_mask": self.fluid_mask is not None,
            "has_structural": self.structural_displacement is not None,
            "has_forces": self.aerodynamic_forces is not None,
            "parameters": self.parameters,
        }
