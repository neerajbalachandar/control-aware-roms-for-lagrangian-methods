"""
rom_koopman.edmdc - Koopman linear predictors with inputs (EDMDc, Korda & Mezic 2018).

Lifted state  z = psi(x~) = [1; x~; phi(x)]   with   x~ = [x_k; x_{k-1}; ...; x_{k-q}]  (q delays)

    z_{k+1} = A z_k + B u_k  (+ N (u_k (x) z_k)   if bilinear)
    x_k     = C z_k          (C selects the x_k entries - exact, no regression)

* linear-in-z dynamics -> convex MPC, LQR, linear reachability (zonotopes /
  ellipsoids) directly on (A, B, C); this is the *control-oriented* Koopman model;
* ``bilinear=True`` gives the Koopman bilinear form (Surana 2016; Peitz et al.
  2020), more accurate when inputs multiply the state (gust convected by the flow);
* ``rollout='relift'`` re-lifts x after every step (nonlinear one-step map, often
  more accurate for long horizons; loses global linearity).

The spectrum of A gives Koopman eigenvalues mu_j (discrete) and
lambda_j = log(mu_j)/dt (continuous); Koopman modes on the state are C v_j and
map to *spatial* modes through the POD basis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from rom_koopman.dictionaries import MonomialDictionary, RBFDictionary

logger = logging.getLogger(__name__)


class EDMDcROM:
    def __init__(self, degree: int = 2, n_rbf: int = 0, rbf_kind: str = "gaussian", delays: int = 0,
                 bilinear: bool = False, ridge: float = 1e-6, rollout: str = "linear",
                 name: str = "koopman-edmdc", blowup_factor: float = 50.0):
        self.degree = degree
        self.n_rbf = n_rbf
        self.rbf_kind = rbf_kind
        self.delays = delays
        self.bilinear = bilinear
        self.ridge = ridge
        self.rollout = rollout
        self.name = name
        self.blowup_factor = blowup_factor
        self.A = self.B = self.N = None
        self.r = self.m = None
        self.dt = None
        self.poly: Optional[MonomialDictionary] = None
        self.rbf: Optional[RBFDictionary] = None
        self.state_bound = 1.0

    # ------------------------------------------------------------------ lifting
    def _delay_stack(self, X: np.ndarray) -> np.ndarray:
        """(r, K) -> (r (q+1), K) with x_{k-j} (first column repeated before the start)."""
        if self.delays == 0:
            return X
        cols = [X]
        for j in range(1, self.delays + 1):
            cols.append(np.hstack([np.repeat(X[:, :1], j, axis=1), X[:, :-j]]))
        return np.vstack(cols)

    def lift(self, Xt: np.ndarray) -> np.ndarray:
        """Xt: delay-stacked states (r(q+1), S) -> z (n_z, S)."""
        x = Xt[:self.r]
        parts = [np.ones((1, Xt.shape[1])), Xt]
        if self.poly is not None:
            parts.append(self.poly(x))
        if self.rbf is not None:
            parts.append(self.rbf(x))
        return np.vstack(parts)

    @property
    def n_z(self) -> int:
        return 1 + self.r * (self.delays + 1) + (self.poly.size if self.poly else 0) + (self.rbf.size if self.rbf else 0)

    @property
    def C(self) -> np.ndarray:
        C = np.zeros((self.r, self.n_z))
        C[:, 1:1 + self.r] = np.eye(self.r)
        return C

    # ------------------------------------------------------------------ fit
    def fit(self, Xs: Sequence[np.ndarray], Us: Sequence[np.ndarray], dt: float) -> "EDMDcROM":
        self.r = Xs[0].shape[0]
        self.m = Us[0].shape[0] if Us else 0
        self.dt = dt
        self.poly = MonomialDictionary(self.r, self.degree, min_degree=2) if self.degree >= 2 else None
        if self.n_rbf:
            self.rbf = RBFDictionary.from_data(np.hstack(Xs), self.n_rbf, kind=self.rbf_kind)
        Z, Zn, Uk = [], [], []
        for X, U in zip(Xs, Us):
            Xt = self._delay_stack(X)
            z = self.lift(Xt)
            Z.append(z[:, :-1])
            Zn.append(z[:, 1:])
            if self.m:
                Uk.append(U[:, :-1])
        Z, Zn = np.hstack(Z), np.hstack(Zn)
        rows = [Z]
        if self.m:
            Uk = np.hstack(Uk)
            rows.append(Uk)
            if self.bilinear:
                rows.append((Uk[:, None, :] * Z[None, :, :]).reshape(self.m * Z.shape[0], -1))
        Phi = np.vstack(rows)
        G = Phi @ Phi.T
        lam = self.ridge * np.trace(G) / G.shape[0]
        M = np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi @ Zn.T).T
        nz = Z.shape[0]
        self.A = M[:, :nz]
        self.B = M[:, nz:nz + self.m] if self.m else np.zeros((nz, 0))
        self.N = M[:, nz + self.m:].reshape(nz, self.m, nz) if (self.m and self.bilinear) else None
        # keep the constant observable exactly constant
        self.A[0] = 0.0
        self.A[0, 0] = 1.0
        self.B[0] = 0.0
        if self.N is not None:
            self.N[0] = 0.0
        self.state_bound = float(max(np.max(np.abs(X)) for X in Xs))
        res = np.linalg.norm(Zn - M @ Phi) / max(np.linalg.norm(Zn), 1e-300)
        logger.info("EDMDc: n_z=%d lifted observables, one-step residual %.3e", nz, res)
        return self

    # ------------------------------------------------------------------ simulate
    def step(self, z: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
        zn = self.A @ z
        if self.m:
            zn = zn + self.B @ u
            if self.N is not None:
                zn = zn + np.einsum("imj,m,j->i", self.N, u, z)
        return zn

    def simulate(self, x0: np.ndarray, t: np.ndarray, U: np.ndarray) -> np.ndarray:
        K = len(t)
        out = np.full((self.r, K), np.nan)
        bound = self.blowup_factor * max(self.state_bound, 1e-12)
        hist = np.repeat(np.asarray(x0, float)[:, None], self.delays + 1, axis=1)  # [x_k, x_{k-1}, ...]
        z = self.lift(hist.T.reshape(-1, 1) if self.delays else hist[:, :1])[:, 0]
        out[:, 0] = x0
        for k in range(1, K):
            u = U[:self.m, k - 1] if self.m else None
            z = self.step(z, u)
            x = z[1:1 + self.r]
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > bound:
                break
            out[:, k] = x
            if self.rollout == "relift":
                hist = np.hstack([x[:, None], hist[:, :-1]])
                z = self.lift(hist.T.reshape(-1, 1) if self.delays else x[:, None])[:, 0]
        return out

    # ------------------------------------------------------------------ spectral analysis
    def spectrum(self, x0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Koopman eigenvalues (discrete & continuous), state-space modes C v_j and amplitudes."""
        mu, V = np.linalg.eig(self.A)
        lam = np.log(mu.astype(complex)) / self.dt
        modes = self.C @ V                          # (r, n_z)
        amp = np.ones(len(mu))
        if x0 is not None:
            hist = np.repeat(np.asarray(x0, float)[:, None], self.delays + 1, axis=1)
            z0 = self.lift(hist.T.reshape(-1, 1) if self.delays else hist[:, :1])[:, 0]
            b = np.linalg.lstsq(V, z0.astype(complex), rcond=None)[0]
            amp = np.abs(b) * np.linalg.norm(modes, axis=0)
        return {"mu": mu, "lambda": lam, "modes": modes, "amplitude": amp}

    # ------------------------------------------------------------------ persistence
    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"A": self.A, "B": self.B}
        if self.N is not None:
            arrays["N"] = self.N
        if self.rbf is not None:
            arrays["rbf_centers"] = self.rbf.centers
        np.savez(d / "edmdc.npz", **arrays)
        meta = {k: getattr(self, k) for k in ("degree", "n_rbf", "rbf_kind", "delays", "bilinear", "ridge", "rollout",
                                              "name", "blowup_factor", "r", "m", "dt", "state_bound")}
        if self.rbf is not None:
            meta["rbf_gamma"] = self.rbf.gamma
        (d / "edmdc.json").write_text(json.dumps(meta, indent=2))
        return d

    @classmethod
    def load(cls, directory) -> "EDMDcROM":
        d = Path(directory)
        meta = json.loads((d / "edmdc.json").read_text())
        rom = cls(meta["degree"], meta["n_rbf"], meta["rbf_kind"], meta["delays"], meta["bilinear"], meta["ridge"],
                  meta["rollout"], meta["name"], meta["blowup_factor"])
        z = np.load(d / "edmdc.npz")
        rom.A, rom.B = z["A"], z["B"]
        rom.N = z["N"] if "N" in z else None
        rom.r, rom.m, rom.dt, rom.state_bound = meta["r"], meta["m"], meta["dt"], meta["state_bound"]
        rom.poly = MonomialDictionary(rom.r, rom.degree, min_degree=2) if rom.degree >= 2 else None
        if "rbf_centers" in z:
            rom.rbf = RBFDictionary(z["rbf_centers"], meta["rbf_kind"], meta["rbf_gamma"])
        return rom
