"""
rom_koopman.lifting - Koopman *lifting* identification of the vector field (Mauroy & Goncalves).

Python port + extension of ``Koopman-identification-master/main/matlab/lifting_ident_main.m``
and ``dual/matlab/lifting_ident_dual.m``:

    A. Mauroy and J. Goncalves, "Linear identification of nonlinear systems: a lifting
       technique based on the Koopman operator", IEEE CDC 2016;
    A. Mauroy and J. Goncalves, "Koopman-based lifting techniques for nonlinear systems
       identification", IEEE TAC 65(6), 2020.

Main method
-----------
For snapshot pairs (x_k, y_k = x_{k+dt}) and a monomial dictionary psi:

1. Koopman matrix (finite section)    K = Psi_X^+ Psi_Y           (least squares)
2. Koopman *generator*                L = logm(K) / dt
3. The generator applied to the linear observable psi_j = x_i is the vector field:
   F_i(x) = sum_k L[k, 1+i] psi_k(x) -> polynomial coefficients of F, read off L.

It needs **no time-derivative estimates** (unlike OpInf), which is attractive for
coarsely sampled or noisy VarFlExI outputs. Inputs are treated as extra states
with zero-order hold (y_u = x_u), exactly as in the MATLAB Duffing example.

Extensions here: many trajectories, ridge-regularised K, principal-logarithm
checks, simulation of the identified polynomial vector field under inputs.

Practical limits: the dictionary has C(n+d, d) functions (n = r + m); d = 2 with
n <= ~25 or d = 3 with n <= ~12 is realistic. The sampling time must be small
enough for logm(K) to be the principal logarithm (|Im eig(L)| < pi/dt).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import logm

from rom_koopman.dictionaries import MonomialDictionary, RBFDictionary

logger = logging.getLogger(__name__)


def _lstsq(A: np.ndarray, B: np.ndarray, ridge: float) -> np.ndarray:
    """argmin ||A K - B||^2 + ridge * tr(A^T A)/n ||K||^2."""
    if ridge <= 0:
        return np.linalg.lstsq(A, B, rcond=None)[0]
    G = A.T @ A
    lam = ridge * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), A.T @ B)


def lifting_ident_main(X: np.ndarray, Y: np.ndarray, deg_F: int, time_step: float, n_inputs: int = 0,
                       deg_basis: Optional[int] = None, factor: float = 1.0, ridge: float = 0.0,
                       threshold: float = 1e-6) -> Dict[str, np.ndarray]:
    """Port of ``lifting_ident_main.m`` (rows = samples, inputs = last columns of X and Y).

    Returns ``coeff`` (n_dic_F, n_syst), ``index`` (exponents of the F monomials),
    ``K``, ``L`` and diagnostics.
    """
    X = np.asarray(X, float) / factor
    Y = np.asarray(Y, float) / factor
    n = X.shape[1]
    n_syst = n - n_inputs
    deg_basis = deg_F if deg_basis is None else deg_basis
    D = MonomialDictionary(n, deg_basis)
    if X.shape[0] < D.size:
        raise ValueError(f"Not enough samples: {X.shape[0]} samples and {D.size} basis functions")
    psi_X = D(X.T).T
    psi_Y = D(Y.T).T
    K = _lstsq(psi_X, psi_Y, ridge)
    if threshold:
        K[np.abs(K) < threshold] = 0.0
    ev = np.linalg.eigvals(K)
    neg_real = np.sum((np.abs(ev.imag) < 1e-10) & (ev.real <= 0))
    if neg_real:
        logger.warning("K has %d eigenvalues on the non-positive real axis: logm is not the principal "
                       "generator - reduce dt (sub-sample less) or regularise more.", neg_real)
    L = logm(K) / time_step
    if np.iscomplexobj(L):
        im = np.max(np.abs(L.imag)) / max(np.max(np.abs(L.real)), 1e-300)
        if im > 1e-6:
            logger.warning("logm(K) has a relative imaginary part of %.2e - using the real part", im)
        L = L.real
    if threshold:
        L[np.abs(L) < threshold] = 0.0
    nF = MonomialDictionary(n, deg_F).size
    coeff = L[:nF, 1:1 + n_syst].copy()
    deg = D.degrees()[:nF]
    coeff *= (factor ** (1 - deg))[:, None]
    return {"coeff": coeff, "index": D.exponents[:nF], "K": K, "L": L, "eig_K": ev}


def lifting_ident_dual(X: np.ndarray, Y: np.ndarray, time_step: float, basis: int = 3, factor: float = 1.0,
                       gamma: float = 1.0 / 100) -> np.ndarray:
    """Port of ``lifting_ident_dual.m``: vector-field values at the sample points.

    ``basis = 0`` -> Gaussian RBFs centred at the samples (exp(-gamma||x-x_k||^2));
    ``basis = d > 0`` -> monomials of total degree <= d (needs #samples <= #functions).
    Returns F_samples (samples, n).
    """
    X = np.asarray(X, float) / factor
    Y = np.asarray(Y, float) / factor
    S, n = X.shape
    if basis == 0:
        psi_X = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
        psi_Y = np.exp(-gamma * np.sum((Y[:, None, :] - X[None, :, :]) ** 2, axis=2))
    else:
        D = MonomialDictionary(n, basis)
        if S > D.size:
            raise ValueError(f"Too many samples: {S} samples and {D.size} basis functions")
        psi_X, psi_Y = D(X.T).T, D(Y.T).T
    K = psi_Y @ np.linalg.pinv(psi_X)
    L = np.real(logm(K)) / time_step
    L[np.abs(L) < 1e-6] = 0.0
    return (L @ X) * factor


# ---------------------------------------------------------------------------
# ROM wrapper
# ---------------------------------------------------------------------------

class LiftingKoopmanROM:
    """Polynomial vector field dx/dt = F(x, u) identified by the lifting method."""

    def __init__(self, deg_F: int = 2, deg_basis: Optional[int] = None, ridge: float = 1e-8,
                 name: str = "koopman-lifting", blowup_factor: float = 50.0, integrator: str = "RK45"):
        self.deg_F = deg_F
        self.deg_basis = deg_basis or deg_F
        self.ridge = ridge
        self.name = name
        self.blowup_factor = blowup_factor
        self.integrator = integrator
        self.coeff = None
        self.r = self.m = None
        self.dict_F: Optional[MonomialDictionary] = None
        self.dt = None
        self.state_bound = 1.0
        self.eig_K = None
        self.L = None

    def fit(self, Xs: Sequence[np.ndarray], Us: Sequence[np.ndarray], dt: float) -> "LiftingKoopmanROM":
        """Xs: list of (r, K_i) scaled states, Us: list of (m, K_i) inputs (zero-order hold)."""
        r = Xs[0].shape[0]
        m = Us[0].shape[0] if Us else 0
        A = np.hstack([np.vstack([X[:, :-1], U[:, :-1]]) if m else X[:, :-1] for X, U in zip(Xs, Us)]).T
        B = np.hstack([np.vstack([X[:, 1:], U[:, :-1]]) if m else X[:, 1:] for X, U in zip(Xs, Us)]).T
        res = lifting_ident_main(A, B, self.deg_F, dt, n_inputs=m, deg_basis=self.deg_basis, ridge=self.ridge)
        self.coeff, self.r, self.m, self.dt = res["coeff"], r, m, dt
        self.dict_F = MonomialDictionary(r + m, self.deg_F)
        self.eig_K = res["eig_K"]
        self.L = res["L"]
        self.state_bound = float(max(np.max(np.abs(X)) for X in Xs))
        return self

    def vector_field(self, x: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
        z = np.concatenate([x, u]) if self.m else x
        return self.coeff.T @ self.dict_F(z[:, None])[:, 0]

    def simulate(self, x0: np.ndarray, t: np.ndarray, U: np.ndarray) -> np.ndarray:
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        K = len(t)
        out = np.full((self.r, K), np.nan)
        bound = self.blowup_factor * max(self.state_bound, 1e-12)
        if self.m:
            uf = interp1d(t, U[:self.m], axis=1, kind="previous", bounds_error=False,
                          fill_value=(U[:self.m, 0], U[:self.m, -1]), assume_sorted=True)
        fun = (lambda tt, x: self.vector_field(x, uf(tt))) if self.m else (lambda tt, x: self.vector_field(x, None))

        def blowup(tt, x):
            return bound - np.max(np.abs(x))
        blowup.terminal = True
        try:
            sol = solve_ivp(fun, (t[0], t[-1]), np.asarray(x0, float), method=self.integrator, t_eval=t,
                            events=blowup, rtol=1e-6, atol=1e-8, max_step=float(np.median(np.diff(t))))
            out[:, :sol.y.shape[1]] = sol.y
        except Exception as exc:  # pragma: no cover
            logger.debug("lifting ROM integration failed: %s", exc)
        return out

    def generator_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the identified Koopman generator L (continuous time)."""
        return np.linalg.eigvals(self.L)

    def coefficient_labels(self, state_names=None, input_names=None):
        names = list(state_names or [f"x{i + 1}" for i in range(self.r)]) + list(input_names or [f"u{i + 1}" for i in range(self.m)])
        return self.dict_F.labels(names)

    # persistence ----------------------------------------------------------
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "lifting.npz", coeff=self.coeff, L=self.L, eig_K=self.eig_K)
        (d / "lifting.json").write_text(json.dumps({"deg_F": self.deg_F, "deg_basis": self.deg_basis, "ridge": self.ridge,
                                                     "name": self.name, "r": self.r, "m": self.m, "dt": self.dt,
                                                     "state_bound": self.state_bound,
                                                     "blowup_factor": self.blowup_factor, "integrator": self.integrator}, indent=2))
        return d

    @classmethod
    def load(cls, directory) -> "LiftingKoopmanROM":
        d = Path(directory)
        meta = json.loads((d / "lifting.json").read_text())
        rom = cls(meta["deg_F"], meta["deg_basis"], meta["ridge"], meta["name"], meta["blowup_factor"], meta["integrator"])
        z = np.load(d / "lifting.npz")
        rom.coeff, rom.L, rom.eig_K = z["coeff"], z["L"], z["eig_K"]
        rom.r, rom.m, rom.dt, rom.state_bound = meta["r"], meta["m"], meta["dt"], meta["state_bound"]
        rom.dict_F = MonomialDictionary(rom.r + rom.m, rom.deg_F)
        return rom
