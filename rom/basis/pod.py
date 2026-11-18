"""
rom.basis.pod – Proper Orthogonal Decomposition (POD) using upstream OpInf.

Wraps opinf.basis.PODBasis to compute spatial basis modes, project centered
snapshots onto reduced coordinates, and decompress back to the full state dimension.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import opinf


class PODReducer:
    """Proper Orthogonal Decomposition basis manager.

    Parameters
    ----------
    r : int, optional
        Target reduced rank (number of basis vectors).
    cumulative_energy : float, optional
        Target cumulative energy criterion (e.g. 0.999), mutually exclusive with r.
    residual_energy : float, optional
        Target residual energy criterion (e.g. 1e-4).
    svdsolver : str, optional
        Algorithm to compute the SVD:
        - ``'dense'``: standard scipy SVD (default).
        - ``'method-of-snapshots'``: correlation-matrix eigenvalue problem,
          highly efficient when state dimension n >> snapshot count K.
        - ``'randomized'``: randomized SVD for very large datasets.
    svdsolver_options : dict, optional
        Options passed to the underlying SVD solver.
    """

    def __init__(
        self,
        r: Optional[int] = None,
        cumulative_energy: Optional[float] = None,
        residual_energy: Optional[float] = None,
        svdsolver: str = "dense",
        **svdsolver_options,
    ):
        self.r_target = r
        self.cumulative_energy_target = cumulative_energy
        self.residual_energy_target = residual_energy
        self.svdsolver = svdsolver
        self.svdsolver_options = svdsolver_options

        # Choose the exact dimension criterion for PODBasis
        if r is not None:
            criterion_kwargs = {"num_vectors": r}
        elif cumulative_energy is not None:
            criterion_kwargs = {"cumulative_energy": cumulative_energy}
        elif residual_energy is not None:
            criterion_kwargs = {"residual_energy": residual_energy}
        else:
            # Default to extracting full rank or initial rank 10
            criterion_kwargs = {"num_vectors": 10}

        self._basis = opinf.basis.PODBasis(
            svdsolver=svdsolver,
            **criterion_kwargs,
            **svdsolver_options,
        )

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def r(self) -> int:
        """Current reduced dimension (number of active basis vectors)."""
        if not self._is_fitted:
            return 0
        return self._basis.reduced_state_dimension

    @property
    def n(self) -> int:
        """Full state dimension."""
        if not self._is_fitted:
            return 0
        return self._basis.full_state_dimension

    @property
    def basis_matrix(self) -> np.ndarray:
        """The active POD basis matrix V_r of shape (n, r)."""
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return self._basis.entries

    @property
    def singular_values(self) -> np.ndarray:
        """All computed singular values."""
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return self._basis.svdvals

    @property
    def cumulative_energy(self) -> float:
        """Cumulative energy retained by the current r basis vectors."""
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return float(self._basis.cumulative_energy)

    @property
    def residual_energy(self) -> float:
        """Residual energy not captured by the current r basis vectors."""
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return float(self._basis.residual_energy)

    # ------------------------------------------------------------------------
    # Fitting & Rank Adjustment
    # ------------------------------------------------------------------------

    def fit(self, Q_train_centered: np.ndarray) -> "PODReducer":
        """Fit the POD basis on centered training snapshots Q of shape (n, K).

        Parameters
        ----------
        Q_train_centered : np.ndarray
            Shape ``(n, K)``. Must be centered beforehand.
        """
        if Q_train_centered.ndim != 2:
            raise ValueError(f"Snapshot matrix must be 2-D (n, K), got shape {Q_train_centered.shape}")

        self._basis.fit(Q_train_centered)
        self._is_fitted = True
        return self

    def set_rank(self, r: int) -> None:
        """Update the active reduced rank r without recomputing the SVD.

        Parameters
        ----------
        r : int
            New rank, must be <= number of available singular vectors.
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit before setting rank.")
        self._basis.set_dimension(r)

    # ------------------------------------------------------------------------
    # Projection & Reconstruction
    # ------------------------------------------------------------------------

    def project(self, Q_centered: np.ndarray) -> np.ndarray:
        """Project centered snapshots Q onto POD modes: a = V_r^T Q.

        Parameters
        ----------
        Q_centered : np.ndarray
            Shape ``(n, K)`` or ``(n,)``.

        Returns
        -------
        a : np.ndarray
            Reduced state coordinates of shape ``(r, K)`` or ``(r,)``.
        """
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return self._basis.compress(Q_centered)

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Decompress reduced coordinates a back to centered state: Q_rec = V_r a.

        Parameters
        ----------
        a : np.ndarray
            Shape ``(r, K)`` or ``(r,)``.

        Returns
        -------
        Q_rec_centered : np.ndarray
            Shape ``(n, K)`` or ``(n,)``.
        """
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        return self._basis.decompress(a)

    def projection_error(self, Q_centered: np.ndarray) -> float:
        """Compute relative Frobenius projection error: ||Q - V_r V_r^T Q||_F / ||Q||_F.

        Parameters
        ----------
        Q_centered : np.ndarray
            Shape ``(n, K)``.
        """
        if not self._is_fitted:
            raise RuntimeError("PODBasis has not been fitted.")
        Q_rec = self.reconstruct(self.project(Q_centered))
        denom = np.linalg.norm(Q_centered, "fro")
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(Q_centered - Q_rec, "fro") / denom)
