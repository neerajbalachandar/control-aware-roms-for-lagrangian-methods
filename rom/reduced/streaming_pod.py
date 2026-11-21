"""
rom.reduced.streaming_pod - Weighted POD of large snapshot sets that do not fit in RAM.

Method of snapshots with a quadrature-weighted inner product <a, b>_W = a^T W b
(W = node volumes for fields, so energies are physical integrals):

1. mean        q_bar = 1/K sum_k q_k                         (one streaming pass)
2. Gram        C_ij  = <q_i - q_bar, q_j - q_bar>_W            (block-pair passes)
3. eig         C = U L U^T,   sigma_i = sqrt(L_i)
4. modes       V = (Q - q_bar) U_r diag(1/sigma_r),  V^T W V = I_r   (one pass)
5. project     a_k = V^T W (q_k - q_bar)  for *every* trajectory    (one pass)

Because the modes are W-orthonormal, the exact full-field error of any reduced
state x (e.g. a ROM prediction) follows from scalars only:

    ||q_k - q_bar - V x||_W^2 = ||q_k - q_bar||_W^2 - ||a_k||^2 + ||a_k - x||^2

so the snapshot norms ``||q_k - q_bar||_W^2`` are stored with the coordinates and
full-field ROM errors are available at every time step without touching the
fields again.

Snapshot sources are callables yielding ``(n_features,)`` vectors in blocks so
this module is agnostic to fields / structure / circulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A snapshot source: returns an iterator of (n_features, b) float arrays for a list of indices
BlockReader = Callable[[Sequence[int]], Iterator[np.ndarray]]


@dataclass
class SnapshotSet:
    """A list of (reader, indices) pairs = all training snapshots of one block."""

    readers: List[BlockReader]
    indices: List[np.ndarray]

    @property
    def n_snapshots(self) -> int:
        return int(sum(len(i) for i in self.indices))

    def iterate(self, block: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (global_offset, X_block (n, b))."""
        off = 0
        for rd, idx in zip(self.readers, self.indices):
            for i in range(0, len(idx), block):
                ib = idx[i:i + block]
                X = np.concatenate(list(rd(ib)), axis=1)
                yield off, X
                off += X.shape[1]


@dataclass
class PODResult:
    mean: np.ndarray            # (n,)
    modes: np.ndarray           # (n, r_max)  W-orthonormal
    singular_values: np.ndarray  # (K,) all
    weights: np.ndarray         # (n,) quadrature weights (ones if unweighted)

    @property
    def r_max(self) -> int:
        return self.modes.shape[1]

    def energy(self) -> np.ndarray:
        s2 = self.singular_values ** 2
        return s2 / s2.sum() if s2.sum() > 0 else s2

    def cumulative_energy(self) -> np.ndarray:
        return np.cumsum(self.energy())

    def rank_for_energy(self, frac: float) -> int:
        return int(np.searchsorted(self.cumulative_energy(), frac) + 1)

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Coordinates (r_max, b) and centred squared W-norms (b,) of snapshot columns X (n, b)."""
        Xc = X - self.mean[:, None]
        WX = Xc * self.weights[:, None]
        return self.modes.T @ WX, np.einsum("ij,ij->j", Xc, WX)

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        r = a.shape[0]
        out = self.modes[:, :r] @ a
        return out + (self.mean[:, None] if out.ndim == 2 else self.mean)


def streaming_pod(snaps: SnapshotSet, weights: Optional[np.ndarray], r_max: int,
                  block: int = 64, center: bool = True, memory_limit_gb: float = 8.0,
                  dtype=np.float64) -> PODResult:
    """Weighted POD (method of snapshots) of ``snaps`` streaming blocks of ``block`` snapshots."""
    K = snaps.n_snapshots
    if K == 0:
        raise ValueError("No training snapshots")
    # --- pass 1: mean ---------------------------------------------------
    first = next(snaps.iterate(1))[1]
    n = first.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != n:
        raise ValueError(f"weights length {w.shape[0]} != n_features {n}")
    mean = np.zeros(n)
    if center:
        for _, X in snaps.iterate(block):
            mean += X.sum(axis=1)
        mean /= K
    in_memory = n * K * 4 / 1e9 <= memory_limit_gb
    logger.info("POD: n=%d features, K=%d snapshots, r_max=%d (%s)", n, K, r_max,
                "in-memory" if in_memory else "out-of-core")
    sw = np.sqrt(w)
    if in_memory:
        Y = np.empty((n, K), dtype=np.float32)
        for off, X in snaps.iterate(block):
            Y[:, off:off + X.shape[1]] = ((X - mean[:, None]) * sw[:, None]).astype(np.float32)
        C = (Y.T.astype(dtype) @ Y.astype(dtype)) if n * K < 2e8 else _gram_blocked(Y, dtype)
    else:
        # Out-of-core: every block pair (i <= j) is formed once; data are re-read
        # from disk O(K / block) times, memory stays at two blocks.
        C = np.zeros((K, K))
        for off_i, Xi in snaps.iterate(block):
            Yi = (Xi - mean[:, None]) * sw[:, None]
            for off_j, Xj in snaps.iterate(block):
                if off_j < off_i:
                    continue
                Yj = (Xj - mean[:, None]) * sw[:, None]
                Cij = Yi.T @ Yj
                C[off_i:off_i + Yi.shape[1], off_j:off_j + Yj.shape[1]] = Cij
                C[off_j:off_j + Yj.shape[1], off_i:off_i + Yi.shape[1]] = Cij.T
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1], U[:, ::-1]
    lam = np.clip(lam, 0.0, None)
    sv = np.sqrt(lam)
    r = int(min(r_max, np.count_nonzero(sv > sv[0] * 1e-10)))
    T = U[:, :r] / sv[None, :r]
    # --- modes ---------------------------------------------------------
    if in_memory:
        V = (Y.astype(dtype) @ T) / sw[:, None]
        del Y
    else:
        V = np.zeros((n, r))
        for off, X in snaps.iterate(block):
            V += (X - mean[:, None]) @ T[off:off + X.shape[1]]
    return PODResult(mean=mean, modes=V, singular_values=sv, weights=w)


def _gram_blocked(Y: np.ndarray, dtype, b: int = 512) -> np.ndarray:
    K = Y.shape[1]
    C = np.zeros((K, K), dtype=dtype)
    for i in range(0, K, b):
        Yi = Y[:, i:i + b].astype(dtype)
        for j in range(i, K, b):
            Cij = Yi.T @ Y[:, j:j + b].astype(dtype)
            C[i:i + b, j:j + b] = Cij
            C[j:j + b, i:i + b] = Cij.T
    return C
