"""
rom_opinf.model - Input-driven Operator Inference ROM (wrapper around ``opinf`` 0.6).

Continuous form (default)::

    dx/dt = c + A x + H (x (x) x) + B u + N (u (x) x)

Discrete form::

    x_{k+1} = c + A x_k + H (x_k (x) x_k) + B u_k + N (u_k (x) x_k)

Operator letters (``opinf`` convention): c constant, A linear, H quadratic,
G cubic, B input, N state-input (bilinear). x is the block-scaled POD state
(fluid + structure + structural rates), u the scaled inputs (gust / actuation).

Why this structure fits VarFlExI: after POD of the Eulerian velocity the
incompressible Navier-Stokes / vortex dynamics are exactly quadratic in the
velocity, the linear structural dynamics enter linearly, the geometric
non-linearity of the plate adds at most cubic terms, and a freestream gust
u enters affinely (B) and via convection of the perturbation (N).

Regularisation is a diagonal Tikhonov matrix with one weight (lambda_1) for
c, A, B and another (lambda_2) for H, G, N (McQuarrie et al., 2021).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import opinf

logger = logging.getLogger(__name__)

LINEAR_OPS = ("ConstantOperator", "LinearOperator", "InputOperator")


def regularizer_diagonal(model, r: int, m: int, lam1: float, lam2: float) -> np.ndarray:
    d = []
    for op in model.operators:
        n = op.operator_dimension(r, m) if m else op.operator_dimension(r)
        d.append(np.full(n, lam1 if type(op).__name__ in LINEAR_OPS else lam2))
    return np.concatenate(d)


def _uses_inputs(ops: str) -> bool:
    return "B" in ops or "N" in ops


class OpInfROM:
    """Fitted OpInf model with the common ``simulate(x0, t, U)`` interface."""

    def __init__(self, operators: str = "cAHB", form: str = "continuous", name: str = "opinf",
                 integrator: str = "RK45", blowup_factor: float = 50.0):
        if form not in ("continuous", "discrete"):
            raise ValueError("form must be 'continuous' or 'discrete'")
        self.operators = operators
        self.form = form
        self.name = name
        self.integrator = integrator
        self.blowup_factor = blowup_factor
        self.model = None
        self.r = None
        self.m = 0
        self.regularizer: Tuple[float, float] = (0.0, 0.0)
        self.state_bound = 1.0

    # ------------------------------------------------------------------
    def _new_model(self):
        if self.form == "continuous":
            return opinf.models.ContinuousModel(self.operators)
        return opinf.models.DiscreteModel(self.operators)

    def fit(self, states: np.ndarray, lhs: np.ndarray, inputs: Optional[np.ndarray],
            lam1: float, lam2: float) -> "OpInfROM":
        """``states`` (r, K), ``lhs`` = ddts (continuous) or next states (discrete), ``inputs`` (m, K)."""
        r = states.shape[0]
        m = inputs.shape[0] if (inputs is not None and _uses_inputs(self.operators)) else 0
        if _uses_inputs(self.operators) and m == 0:
            raise ValueError(f"Operators '{self.operators}' need inputs but none were given")
        model = self._new_model()
        d = regularizer_diagonal(model, r, m, lam1, lam2)
        model.solver = opinf.lstsq.TikhonovSolver(regularizer=d) if (lam1 > 0 or lam2 > 0) else opinf.lstsq.PlainSolver()
        # ContinuousModel.fit(states, ddts, inputs) / DiscreteModel.fit(states, nextstates, inputs)
        model.fit(states, lhs, inputs if m else None)
        self.model, self.r, self.m = model, r, m
        self.regularizer = (float(lam1), float(lam2))
        self.state_bound = float(np.max(np.abs(states)))
        return self

    # ------------------------------------------------------------------
    def rhs(self, x: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
        if self.form == "continuous":
            return self.model.rhs(0.0, x, (lambda _t: u) if self.m else None)
        return self.model.rhs(x, u if self.m else None)

    def simulate(self, x0: np.ndarray, t: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Roll out from ``x0`` over ``t`` with inputs ``U`` (m, K) sampled at ``t``.

        Returns (r, K); entries after a blow-up (|x| > blowup_factor * training bound)
        are NaN.
        """
        from scipy.integrate import solve_ivp

        t = np.asarray(t, float)
        K = len(t)
        out = np.full((self.r, K), np.nan)
        bound = self.blowup_factor * max(self.state_bound, 1e-12)
        U = U[:self.m] if self.m else None
        if self.form == "discrete":
            x = np.asarray(x0, float).copy()
            out[:, 0] = x
            for k in range(1, K):
                x = self.model.rhs(x, U[:, k - 1] if self.m else None)
                if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > bound:
                    break
                out[:, k] = x
            return out
        if self.m:
            from scipy.interpolate import interp1d

            uf = interp1d(t, U, axis=1, kind="linear", bounds_error=False, fill_value=(U[:, 0], U[:, -1]),
                          assume_sorted=True)
            fun = lambda tt, x: self.model.rhs(tt, x, uf)  # noqa: E731
        else:
            fun = lambda tt, x: self.model.rhs(tt, x)  # noqa: E731

        def blowup(tt, x):
            return bound - np.max(np.abs(x))
        blowup.terminal = True

        opts = {"method": self.integrator, "t_eval": t, "events": blowup, "rtol": 1e-6, "atol": 1e-8,
                "max_step": float(np.median(np.diff(t))) if K > 1 else np.inf}
        if self.integrator in ("BDF", "Radau", "LSODA"):
            opts["jac"] = (lambda tt, x: self.model.jacobian(tt, x, uf)) if self.m else (lambda tt, x: self.model.jacobian(tt, x))
        try:
            sol = solve_ivp(fun, (t[0], t[-1]), np.asarray(x0, float), **opts)
            n = sol.y.shape[1]
            out[:, :n] = sol.y
        except Exception as exc:  # pragma: no cover
            logger.debug("integration failed: %s", exc)
        return out

    # ------------------------------------------------------------------
    def operator_entries(self) -> Dict[str, np.ndarray]:
        return {type(op).__name__: np.asarray(op.entries) for op in self.model.operators}

    def linear_operator(self) -> Optional[np.ndarray]:
        e = self.operator_entries()
        return e.get("LinearOperator")

    def linearize(self, x_eq: np.ndarray, u_eq: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Jacobians (A_lin, B_lin) of the ROM at (x_eq, u_eq) - for LQR/MPC design, reachability."""
        u_eq = np.zeros(self.m) if (self.m and u_eq is None) else u_eq
        if self.form == "continuous":
            J = self.model.jacobian(0.0, x_eq, (lambda _t: u_eq) if self.m else None)
        else:
            J = self.model.jacobian(x_eq, u_eq if self.m else None)
        Bl = None
        if self.m:
            eps = 1e-6
            f0 = self.rhs(x_eq, u_eq)
            Bl = np.stack([(self.rhs(x_eq, u_eq + eps * e) - f0) / eps for e in np.eye(self.m)], axis=1)
        return np.asarray(J), Bl

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "operators.npz", **self.operator_entries())
        meta = {"operators": self.operators, "form": self.form, "name": self.name, "integrator": self.integrator,
                "r": self.r, "m": self.m, "regularizer": self.regularizer, "state_bound": self.state_bound,
                "blowup_factor": self.blowup_factor}
        (directory / "opinf_model.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory) -> "OpInfROM":
        directory = Path(directory)
        meta = json.loads((directory / "opinf_model.json").read_text())
        rom = cls(meta["operators"], meta["form"], meta["name"], meta["integrator"], meta["blowup_factor"])
        ent = np.load(directory / "operators.npz")
        model = rom._new_model()
        # opinf 0.6: set operator entries directly (model then counts as trained)
        for op in model.operators:
            op.set_entries(ent[type(op).__name__])
        rom.model, rom.r, rom.m = model, meta["r"], meta["m"]
        rom.regularizer = tuple(meta["regularizer"])
        rom.state_bound = meta["state_bound"]
        return rom
