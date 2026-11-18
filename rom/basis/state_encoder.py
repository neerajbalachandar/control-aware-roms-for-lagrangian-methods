"""
rom.basis.state_encoder – Spatial state preparation and active fluid-domain mapping.

Transforms 3-D Cartesian Eulerian velocity snapshots into flattened 1-D state
vectors on active fluid cells, manages mean-field centering, and reconstructs
the original 3-D Cartesian grid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask


class VelocityStateEncoder:
    """Encodes Eulerian velocity fields onto active fluid cells and reconstructs 3D fields.

    Parameters
    ----------
    grid : EulerianGrid
        The fixed spatial grid.
    fluid_mask : FluidMask, optional
        Mask defining active fluid cells (True=fluid, False=solid).
        If None, all cells in the grid are considered active fluid.
    """

    def __init__(self, grid: EulerianGrid, fluid_mask: Optional[FluidMask] = None):
        self.grid = grid
        self.fluid_mask = fluid_mask

        if fluid_mask is not None:
            if fluid_mask.mask.shape != grid.shape:
                raise ValueError(
                    f"Fluid mask shape {fluid_mask.mask.shape} != grid shape {grid.shape}"
                )
            self._active_mask = fluid_mask.mask
        else:
            self._active_mask = np.ones(grid.shape, dtype=bool)

        # Boolean index of active cells
        self._active_indices = np.where(self._active_mask)
        self._n_fluid_cells = int(np.count_nonzero(self._active_mask))

        # Each active cell has 3 Cartesian velocity components (u_x, u_y, u_z)
        self._state_dimension = 3 * self._n_fluid_cells

        # Mean field storage (computed ONLY on training snapshots)
        self._mean_field: Optional[np.ndarray] = None

    @property
    def state_dimension(self) -> int:
        """Full state dimension n = 3 * N_fluid."""
        return self._state_dimension

    @property
    def n_fluid_cells(self) -> int:
        """Number of active fluid cells."""
        return self._n_fluid_cells

    @property
    def active_mask(self) -> np.ndarray:
        """Boolean mask array of active fluid cells."""
        return self._active_mask

    @property
    def mean_field(self) -> Optional[np.ndarray]:
        """Flattened mean velocity field vector of length n, if fitted."""
        return self._mean_field

    @property
    def mean_field_grid(self) -> Optional[np.ndarray]:
        """Decoded 3D mean velocity field on the grid (N0, N1, N2, 3), if fitted."""
        if self._mean_field is None:
            return None
        return self.decode_snapshot(self._mean_field)

    # ------------------------------------------------------------------------
    # Encoding (Grid -> 1D State Vector)
    # ------------------------------------------------------------------------

    def encode_snapshot(self, velocity: np.ndarray) -> np.ndarray:
        """Flatten a single 3-D velocity snapshot (N0, N1, N2, 3) to 1-D state vector (n,).

        Parameters
        ----------
        velocity : np.ndarray
            Shape ``(N0, N1, N2, 3)``.

        Returns
        -------
        q : np.ndarray
            1-D array of shape ``(n,)`` containing (u_x, u_y, u_z) for each active cell.
        """
        if velocity.shape != self.grid.shape + (3,):
            raise ValueError(
                f"Velocity snapshot shape {velocity.shape} != expected {self.grid.shape + (3,)}"
            )

        # Extract active fluid cells: shape (N_fluid, 3)
        fluid_vel = velocity[self._active_mask, :]  # (N_fluid, 3)
        if not np.all(np.isfinite(fluid_vel)):
            n_nan = np.count_nonzero(~np.isfinite(fluid_vel))
            raise ValueError(f"Encountered {n_nan} non-finite velocity values in active fluid cells")

        # Flatten into 1-D vector: (3 * N_fluid,)
        return fluid_vel.reshape(-1)

    def encode_trajectory(self, velocity_snaps: np.ndarray) -> np.ndarray:
        """Flatten a sequence of velocity snapshots (K, N0, N1, N2, 3) to matrix Q of shape (n, K).

        Parameters
        ----------
        velocity_snaps : np.ndarray
            Shape ``(K, N0, N1, N2, 3)``.

        Returns
        -------
        Q : np.ndarray
            Matrix of shape ``(n, K)`` where column j is snapshot j.
        """
        k = velocity_snaps.shape[0]
        # Extract active cells across all snapshots: (K, N_fluid, 3)
        fluid_vel = velocity_snaps[:, self._active_mask, :]
        if not np.all(np.isfinite(fluid_vel)):
            n_nan = np.count_nonzero(~np.isfinite(fluid_vel))
            raise ValueError(f"Encountered {n_nan} non-finite velocity values in active fluid cells")

        # Reshape to (K, 3 * N_fluid) and transpose to column format (n, K)
        return fluid_vel.reshape(k, -1).T

    # ------------------------------------------------------------------------
    # Decoding (1D State Vector -> Grid)
    # ------------------------------------------------------------------------

    def decode_snapshot(self, q: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Decode a 1-D state vector q of length n into a 3-D grid field (N0, N1, N2, 3).

        Parameters
        ----------
        q : np.ndarray
            1-D array of shape ``(n,)``.
        fill_value : float, optional
            Value used to fill inactive (solid) cells (default: np.nan).

        Returns
        -------
        velocity : np.ndarray
            Shape ``(N0, N1, N2, 3)``.
        """
        if q.shape != (self._state_dimension,):
            raise ValueError(f"State vector shape {q.shape} != expected ({self._state_dimension},)")

        out = np.full(self.grid.shape + (3,), fill_value, dtype=np.float64)
        reshaped_q = q.reshape(self._n_fluid_cells, 3)
        out[self._active_mask, :] = reshaped_q
        return out

    def decode_trajectory(self, Q: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Decode a matrix Q of shape (n, K) into a 5-D field (K, N0, N1, N2, 3).

        Parameters
        ----------
        Q : np.ndarray
            Shape ``(n, K)``.
        fill_value : float, optional
            Value for inactive cells (default: np.nan).

        Returns
        -------
        velocity : np.ndarray
            Shape ``(K, N0, N1, N2, 3)``.
        """
        if Q.shape[0] != self._state_dimension:
            raise ValueError(f"Q row dimension {Q.shape[0]} != state dimension {self._state_dimension}")

        k = Q.shape[1]
        out = np.full((k,) + self.grid.shape + (3,), fill_value, dtype=np.float64)
        # Transpose Q to (K, n), then reshape active part to (K, N_fluid, 3)
        reshaped = Q.T.reshape(k, self._n_fluid_cells, 3)
        out[:, self._active_mask, :] = reshaped
        return out

    # ------------------------------------------------------------------------
    # Mean-Field Handling (Crucial: Fit ONLY on training snapshots!)
    # ------------------------------------------------------------------------

    def fit_mean(self, Q_train: np.ndarray) -> np.ndarray:
        """Compute and store the mean field from TRAINING snapshots Q_train of shape (n, K_train).

        Parameters
        ----------
        Q_train : np.ndarray
            Shape ``(n, K_train)``.

        Returns
        -------
        mean : np.ndarray
            Shape ``(n,)``.
        """
        if Q_train.shape[0] != self._state_dimension:
            raise ValueError(
                f"Q_train row count {Q_train.shape[0]} != state dimension {self._state_dimension}"
            )
        self._mean_field = np.mean(Q_train, axis=1)
        return self._mean_field

    def set_mean(self, mean_field: np.ndarray) -> None:
        """Explicitly set a precomputed mean field."""
        if mean_field.shape != (self._state_dimension,):
            raise ValueError(
                f"mean_field shape {mean_field.shape} != state dimension ({self._state_dimension},)"
            )
        self._mean_field = np.asarray(mean_field, dtype=np.float64).copy()

    def center(self, Q: np.ndarray) -> np.ndarray:
        """Subtract the fitted mean field: Q_centered = Q - mean.

        Parameters
        ----------
        Q : np.ndarray
            Shape ``(n, K)`` or ``(n,)``.

        Returns
        -------
        Q_centered : np.ndarray
        """
        if self._mean_field is None:
            raise RuntimeError("Mean field has not been fitted. Call fit_mean(Q_train) first.")
        if Q.ndim == 1:
            return Q - self._mean_field
        return Q - self._mean_field[:, np.newaxis]

    def uncenter(self, Q_centered: np.ndarray) -> np.ndarray:
        """Restore the fitted mean field: Q = Q_centered + mean.

        Parameters
        ----------
        Q_centered : np.ndarray
            Shape ``(n, K)`` or ``(n,)``.

        Returns
        -------
        Q : np.ndarray
        """
        if self._mean_field is None:
            raise RuntimeError("Mean field has not been fitted.")
        if Q_centered.ndim == 1:
            return Q_centered + self._mean_field
        return Q_centered + self._mean_field[:, np.newaxis]
