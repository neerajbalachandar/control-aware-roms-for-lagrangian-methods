"""
rom.eulerian.field_reconstructor – Velocity reconstruction via regularised
Biot-Savart evaluation.

Mathematical background
-----------------------
FLOWUnsteady / rVPM represents vorticity as a collection of Lagrangian
particles.  Each particle *p* carries:

* Position  ``x_p`` ∈ ℝ³
* Circulation (vortex strength) vector  ``Γ_p`` ∈ ℝ³   (units: m²/s or m³/s
  depending on convention; the code is agnostic—it preserves whatever units
  the solver uses).
* Core (smoothing) radius ``σ_p`` > 0

The velocity induced at an arbitrary target point **x** is

    u(x) = (1 / 4π) Σ_p  (Γ_p × r_p) / |r_p|³  · g(ρ_p)

where

    r_p  = x − x_p
    ρ_p  = |r_p| / σ_p
    g(ρ) is the kernel-dependent regularisation (cut-off) function.

Supported kernels
~~~~~~~~~~~~~~~~~
1. **Gaussian-erf** (default, matches ``vpm.gaussianerf`` in FLOWVPM.jl):

       g(ρ) = erf(ρ) − (2/√π) ρ exp(−ρ²)

   This arises from the analytical integration of a Gaussian blob:

       ζ_σ(r) = (1 / (π^{3/2} σ³)) exp(−(r/σ)²)

2. **Winckelmans** algebraic kernel (``vpm.winckelmans``):

       g(ρ) = ρ³ (ρ² + 5/2) / (ρ² + 1)^{5/2}

3. **Singular** (un-regularised, for validation only):

       g(ρ) = 1

Memory management
-----------------
The evaluation is **chunked**: the set of target (grid) points is split into
blocks of configurable size so that the per-chunk intermediate arrays stay
within a controllable memory footprint.  Chunked and unchunked evaluations
produce identical results to floating-point precision.

Particle validation
-------------------
Before each snapshot evaluation the module:
* Rejects particles whose positions, circulations, or core sizes contain
  ``NaN`` or ``±Inf``.
* Rejects particles with ``σ ≤ 0``.
* Logs the number of rejected particles without aborting (the user is warned).
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.special import erf

from .grid import EulerianGrid
from .masking import FluidMask

logger = logging.getLogger(__name__)


# ========================================================================
# Kernel enumeration
# ========================================================================

class Kernel(Enum):
    """Regularisation kernel for the Biot-Savart evaluation."""
    GAUSSIAN_ERF = "gaussianerf"
    WINCKELMANS = "winckelmans"
    SINGULAR = "singular"


# ========================================================================
# Kernel cut-off functions  g(ρ)
# ========================================================================

def _cutoff_gaussianerf(rho: np.ndarray) -> np.ndarray:
    """Gaussian-erf regularisation:  g(ρ) = erf(ρ) − (2/√π) ρ exp(−ρ²)."""
    return erf(rho) - (2.0 / np.sqrt(np.pi)) * rho * np.exp(-(rho ** 2))


def _cutoff_winckelmans(rho: np.ndarray) -> np.ndarray:
    """Winckelmans algebraic regularisation:  g(ρ) = ρ³(ρ²+5/2)/(ρ²+1)^{5/2}."""
    rho2 = rho ** 2
    return (rho ** 3) * (rho2 + 2.5) / ((rho2 + 1.0) ** 2.5)


def _cutoff_singular(rho: np.ndarray) -> np.ndarray:
    """Un-regularised (singular) kernel:  g(ρ) = 1."""
    return np.ones_like(rho)


_CUTOFF_FUNCS = {
    Kernel.GAUSSIAN_ERF: _cutoff_gaussianerf,
    Kernel.WINCKELMANS: _cutoff_winckelmans,
    Kernel.SINGULAR: _cutoff_singular,
}


# ========================================================================
# Particle data container
# ========================================================================

class ParticleSnapshot:
    """Container for one temporal snapshot of Lagrangian particle data.

    Parameters
    ----------
    positions : (Np, 3) array
        Particle positions in physical (x, y, z) coordinates.
    circulations : (Np, 3) array
        Circulation (vortex-strength) vectors Γ_p.
    core_sizes : (Np,) array
        Smoothing radii σ_p  (must be > 0).
    time : float, optional
        Physical time of the snapshot.
    velocities : (Np, 3) array or None, optional
        Particle velocities (if available, for cross-validation).
    """

    def __init__(
        self,
        positions: np.ndarray,
        circulations: np.ndarray,
        core_sizes: np.ndarray,
        time: float = 0.0,
        velocities: Optional[np.ndarray] = None,
    ):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.circulations = np.asarray(circulations, dtype=np.float64)
        self.core_sizes = np.asarray(core_sizes, dtype=np.float64).ravel()
        self.time = float(time)
        self.velocities = (
            np.asarray(velocities, dtype=np.float64)
            if velocities is not None
            else None
        )

        # Basic shape checks
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (Np, 3), got {self.positions.shape}"
            )
        if self.circulations.shape != self.positions.shape:
            raise ValueError(
                f"circulations shape {self.circulations.shape} != "
                f"positions shape {self.positions.shape}"
            )
        if self.core_sizes.shape[0] != self.positions.shape[0]:
            raise ValueError(
                "core_sizes length does not match number of particles"
            )
        if self.velocities is not None and self.velocities.shape != self.positions.shape:
            raise ValueError(
                f"velocities shape {self.velocities.shape} != "
                f"positions shape {self.positions.shape}"
            )

    @property
    def num_particles(self) -> int:
        return self.positions.shape[0]


# ========================================================================
# Particle validation
# ========================================================================

def _validate_particles(snap: ParticleSnapshot) -> ParticleSnapshot:
    """Return a *new* ParticleSnapshot with invalid particles removed.

    Invalid means: NaN / Inf in position or circulation, or σ ≤ 0.
    """
    valid = (
        np.all(np.isfinite(snap.positions), axis=1)
        & np.all(np.isfinite(snap.circulations), axis=1)
        & np.isfinite(snap.core_sizes)
        & (snap.core_sizes > 0.0)
    )
    n_reject = int(np.count_nonzero(~valid))
    if n_reject > 0:
        warnings.warn(
            f"Rejected {n_reject}/{snap.num_particles} invalid particles "
            f"(NaN/Inf/σ≤0) at t = {snap.time:.6g}.",
            stacklevel=3,
        )
        logger.warning(
            "Rejected %d / %d invalid particles at t = %.6g",
            n_reject,
            snap.num_particles,
            snap.time,
        )
    vel = snap.velocities[valid] if snap.velocities is not None else None
    return ParticleSnapshot(
        positions=snap.positions[valid],
        circulations=snap.circulations[valid],
        core_sizes=snap.core_sizes[valid],
        time=snap.time,
        velocities=vel,
    )


# ========================================================================
# Core Biot-Savart evaluation
# ========================================================================

def _biot_savart_chunk(
    targets: np.ndarray,          # (M, 3)
    positions: np.ndarray,        # (Np, 3)
    circulations: np.ndarray,     # (Np, 3)
    core_sizes: np.ndarray,       # (Np,)
    cutoff_func,
) -> np.ndarray:
    """Evaluate regularised Biot-Savart velocity at *targets*.

    Parameters
    ----------
    targets : (M, 3)
    positions : (Np, 3)
    circulations : (Np, 3)
    core_sizes : (Np,)
    cutoff_func : callable
        g(ρ) → array, where ρ = |r| / σ.

    Returns
    -------
    velocity : (M, 3)
    """
    # r_{mp} = targets[m] - positions[p]  → (M, Np, 3)
    r = targets[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (M, Np, 3)
    r_mag = np.linalg.norm(r, axis=2)  # (M, Np)

    # ρ = |r| / σ
    sigma = core_sizes[np.newaxis, :]  # (1, Np)
    rho = r_mag / sigma               # (M, Np)

    # g(ρ)
    g = cutoff_func(rho)  # (M, Np)

    # Avoid division by zero where |r| = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r3 = np.where(r_mag > 0.0, 1.0 / (r_mag ** 3), 0.0)  # (M, Np)

    # Γ_p × r_{mp}  → (M, Np, 3)
    # circulations shape (Np, 3) broadcast with r shape (M, Np, 3)
    gamma = circulations[np.newaxis, :, :]  # (1, Np, 3)
    cross = np.cross(gamma, r)  # (M, Np, 3)

    # Combine:  (1 / 4π) Σ_p  (Γ_p × r_p) * g(ρ) / |r|³
    factor = g * inv_r3  # (M, Np)
    # Multiply each component of cross by factor
    weighted = cross * factor[:, :, np.newaxis]  # (M, Np, 3)

    # Sum over particles
    velocity = weighted.sum(axis=1) / (4.0 * np.pi)  # (M, 3)
    return velocity


# ========================================================================
# Public reconstructor
# ========================================================================

class FieldReconstructor:
    """Reconstruct Eulerian velocity fields from Lagrangian particle data.

    Parameters
    ----------
    grid : EulerianGrid
        Fixed spatial grid for all snapshots.
    kernel : Kernel or str, optional
        Regularisation kernel (default: ``Kernel.GAUSSIAN_ERF``).
    chunk_size : int, optional
        Maximum number of target points evaluated in one batch
        (default: 4096).  Controls peak memory usage.
    fluid_mask : FluidMask or None, optional
        If provided, solid-region velocities are set to NaN.
    validate_particles : bool, optional
        If True (default), reject particles with NaN / Inf / σ ≤ 0.
    """

    def __init__(
        self,
        grid: EulerianGrid,
        kernel: Kernel | str = Kernel.GAUSSIAN_ERF,
        chunk_size: int = 4096,
        fluid_mask: Optional[FluidMask] = None,
        validate_particles: bool = True,
    ):
        self.grid = grid
        self.kernel = Kernel(kernel) if isinstance(kernel, str) else kernel
        self.chunk_size = int(chunk_size)
        self.fluid_mask = fluid_mask
        self.validate_particles = validate_particles

        self._cutoff = _CUTOFF_FUNCS[self.kernel]
        self._targets = grid.flat_points()  # (M, 3)

    # ---- single-snapshot evaluation ------------------------------------

    def evaluate_velocity(
        self,
        snapshot: ParticleSnapshot,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute velocity at every grid point from a particle snapshot.

        Parameters
        ----------
        snapshot : ParticleSnapshot
        chunk_size : int or None
            Override the default chunk size for this call.

        Returns
        -------
        velocity : np.ndarray, shape ``grid.shape + (3,)``
            Velocity components (u_x, u_y, u_z) in the physical frame.
        """
        if self.validate_particles:
            snapshot = _validate_particles(snapshot)

        if snapshot.num_particles == 0:
            warnings.warn("No valid particles; returning zero velocity field.")
            vel = np.zeros(self.grid.shape + (3,), dtype=np.float64)
            if self.fluid_mask is not None:
                vel = self.fluid_mask.apply(vel)
            return vel

        cs = chunk_size if chunk_size is not None else self.chunk_size
        targets = self._targets  # (M, 3)
        M = targets.shape[0]

        velocity_flat = np.empty((M, 3), dtype=np.float64)

        for start in range(0, M, cs):
            end = min(start + cs, M)
            velocity_flat[start:end] = _biot_savart_chunk(
                targets[start:end],
                snapshot.positions,
                snapshot.circulations,
                snapshot.core_sizes,
                self._cutoff,
            )

        # Reshape to grid
        vel = velocity_flat.reshape(self.grid.shape + (3,))

        if self.fluid_mask is not None:
            vel = self.fluid_mask.apply(vel)

        return vel

    # ---- convenience: evaluate at arbitrary points ---------------------

    def evaluate_velocity_at_points(
        self,
        points: np.ndarray,
        snapshot: ParticleSnapshot,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate velocity at arbitrary points (not necessarily on the grid).

        Parameters
        ----------
        points : (N, 3) array
            Target locations in physical (x, y, z) coordinates.
        snapshot : ParticleSnapshot
        chunk_size : int or None

        Returns
        -------
        velocity : (N, 3) array
        """
        if self.validate_particles:
            snapshot = _validate_particles(snapshot)

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        cs = chunk_size if chunk_size is not None else self.chunk_size
        N = points.shape[0]
        vel = np.empty((N, 3), dtype=np.float64)

        for start in range(0, N, cs):
            end = min(start + cs, N)
            vel[start:end] = _biot_savart_chunk(
                points[start:end],
                snapshot.positions,
                snapshot.circulations,
                snapshot.core_sizes,
                self._cutoff,
            )
        return vel

    # ---- batch / snapshot-collection processing with HDF5 output -------

    def process_snapshots(
        self,
        snapshots,
        output_path: str,
        compute_vorticity: bool = True,
        diff_scheme: str = "central",
        diff_order: int = 2,
    ) -> None:
        """Process an iterable of :class:`ParticleSnapshot` and write HDF5.

        Each snapshot is evaluated one at a time and written to disk
        immediately (chunked HDF5 writes), so peak RAM is bounded by one
        grid-sized velocity field.

        Parameters
        ----------
        snapshots : iterable of ParticleSnapshot
        output_path : str
            Path to the output HDF5 file.
        compute_vorticity : bool
            Whether to also compute and store ω = ∇ × u.
        diff_scheme : str
            Finite-difference scheme for vorticity (``'central'`` or
            ``'second_order'``).
        diff_order : int
            Accuracy order (2 or 4).
        """
        import h5py
        from .vorticity import compute_vorticity as _curl

        snapshots = list(snapshots)
        n_snaps = len(snapshots)
        if n_snaps == 0:
            raise ValueError("No snapshots provided.")

        shape3 = self.grid.shape
        field_shape = shape3 + (3,)

        with h5py.File(output_path, "w") as f:
            # --- metadata group ---
            meta = f.create_group("metadata")
            self.grid.write_metadata(meta)
            meta.create_dataset(
                "time",
                data=np.array([s.time for s in snapshots], dtype=np.float64),
            )
            meta.attrs["kernel"] = self.kernel.value
            meta.attrs["chunk_size"] = self.chunk_size

            if self.fluid_mask is not None:
                self.fluid_mask.write_hdf5(meta)
                meta.attrs["has_fluid_mask"] = True

            # --- snapshot datasets (chunked) ---
            snaps_grp = f.create_group("snapshots")

            vel_ds = snaps_grp.create_dataset(
                "velocity",
                shape=(n_snaps,) + field_shape,
                dtype=np.float64,
                chunks=(1,) + field_shape,
                compression="gzip",
                compression_opts=4,
            )
            vel_ds.attrs["description"] = (
                "Velocity field (u_x, u_y, u_z) in physical frame. "
                "Shape: (n_snapshots, Nx, Ny, Nz, 3)."
            )
            vel_ds.attrs["axis_order"] = self.grid.axis_order

            if compute_vorticity:
                vort_ds = snaps_grp.create_dataset(
                    "vorticity",
                    shape=(n_snaps,) + field_shape,
                    dtype=np.float64,
                    chunks=(1,) + field_shape,
                    compression="gzip",
                    compression_opts=4,
                )
                vort_ds.attrs["description"] = (
                    "Vorticity field (ω_x, ω_y, ω_z). "
                    "Shape: (n_snapshots, Nx, Ny, Nz, 3)."
                )
                vort_ds.attrs["diff_scheme"] = diff_scheme
                vort_ds.attrs["diff_order"] = diff_order

            # --- process each snapshot ---
            for idx, snap in enumerate(snapshots):
                logger.info(
                    "Processing snapshot %d / %d  (t = %.6g)",
                    idx + 1,
                    n_snaps,
                    snap.time,
                )
                vel = self.evaluate_velocity(snap)
                vel_ds[idx] = vel

                if compute_vorticity:
                    omega = _curl(
                        vel,
                        self.grid,
                        scheme=diff_scheme,
                        order=diff_order,
                        fluid_mask=self.fluid_mask,
                    )
                    vort_ds[idx] = omega

            logger.info("Wrote %d snapshots to %s", n_snaps, output_path)
