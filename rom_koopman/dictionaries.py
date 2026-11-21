"""
rom_koopman.dictionaries - Observables (lifting functions) psi(x).

* :class:`MonomialDictionary` - all monomials of total degree <= d, ordered
  [1, x_1..x_n, degree-2 terms, ...] exactly as required by the Mauroy-Goncalves
  lifting method (constant first, then the linear monomials).
* :class:`RBFDictionary` - Gaussian or thin-plate radial basis functions with
  k-means centres (the "dual" method and EDMD with non-polynomial observables).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb
from typing import List, Optional, Sequence, Tuple

import numpy as np


def monomial_exponents(n: int, degree: int, min_degree: int = 0) -> np.ndarray:
    """Exponent matrix (n_terms, n): rows ordered by total degree, then lexicographically."""
    rows: List[np.ndarray] = []
    for d in range(min_degree, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), d):
            e = np.zeros(n, dtype=int)
            for i in combo:
                e[i] += 1
            rows.append(e)
    return np.asarray(rows, dtype=int).reshape(-1, n)


def n_monomials(n: int, degree: int) -> int:
    return comb(n + degree, degree)


@dataclass
class MonomialDictionary:
    n: int
    degree: int
    min_degree: int = 0
    exponents: np.ndarray = field(init=False)

    def __post_init__(self):
        self.exponents = monomial_exponents(self.n, self.degree, self.min_degree)

    @property
    def size(self) -> int:
        return self.exponents.shape[0]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X (n, S) -> Psi (size, S). Uses repeated products (stable for |x| <= ~1)."""
        X = np.atleast_2d(X)
        if X.shape[0] != self.n:
            raise ValueError(f"expected {self.n} rows, got {X.shape[0]}")
        out = np.ones((self.size, X.shape[1]))
        for j in range(self.n):
            p = self.exponents[:, j]
            for k in range(1, p.max() + 1):
                out[p >= k] *= X[j]
        return out

    def degrees(self) -> np.ndarray:
        return self.exponents.sum(1)

    def labels(self, names: Optional[Sequence[str]] = None) -> List[str]:
        names = list(names) if names else [f"x{i + 1}" for i in range(self.n)]
        out = []
        for e in self.exponents:
            terms = [f"{names[i]}" + (f"^{p}" if p > 1 else "") for i, p in enumerate(e) if p > 0]
            out.append("*".join(terms) if terms else "1")
        return out


@dataclass
class RBFDictionary:
    centers: np.ndarray            # (n, Nc)
    kind: str = "gaussian"         # gaussian | thinplate
    gamma: float = 1.0             # gaussian: exp(-gamma ||x - c||^2)

    @classmethod
    def from_data(cls, X: np.ndarray, n_centers: int, kind: str = "gaussian", gamma: Optional[float] = None,
                  seed: int = 0) -> "RBFDictionary":
        from scipy.cluster.vq import kmeans2

        rng = np.random.default_rng(seed)
        sub = X[:, rng.choice(X.shape[1], size=min(X.shape[1], 20 * n_centers), replace=False)]
        C, _ = kmeans2(sub.T, n_centers, minit="++", seed=seed)
        if gamma is None:
            d = np.linalg.norm(C[:, None] - C[None], axis=2)
            med = np.median(d[d > 0]) if (d > 0).any() else 1.0
            gamma = 1.0 / med ** 2
        return cls(centers=C.T, kind=kind, gamma=float(gamma))

    @property
    def size(self) -> int:
        return self.centers.shape[1]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        d2 = np.sum((X[:, None, :] - self.centers[:, :, None]) ** 2, axis=0)  # (Nc, S)
        if self.kind == "gaussian":
            return np.exp(-self.gamma * d2)
        if self.kind == "thinplate":
            with np.errstate(divide="ignore", invalid="ignore"):
                v = 0.5 * d2 * np.log(d2)
            return np.nan_to_num(v)
        raise ValueError(self.kind)
