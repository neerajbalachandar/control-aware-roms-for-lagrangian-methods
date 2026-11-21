"""
rom.reduced.state - Reduced state assembly, scaling, output maps and the common model interface.

The ROM state is the concatenation of truncated, *block-scaled* POD coordinates

    x = [ a_fluid[:r_f] / s_fluid ;  a_struct[:r_s] / s_struct ;  a_struct_rate[:r_s] / s_rate ; ... ]

with one scalar s_b per block (max |a| over training data), which preserves the
POD geometry inside a block (Galerkin-consistent) while balancing blocks of very
different physical units in the least-squares problems of OpInf / Koopman.

Inputs are scaled per channel by their max |u| over training data (no centring:
u = 0 must keep meaning "no gust / no actuation").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from rom.reduced.dataset import ReducedDataset, ReducedTrajectory


@dataclass
class StateAssembler:
    blocks: List[str]
    ranks: Dict[str, int]
    scales: Dict[str, float] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    input_scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, ds: ReducedDataset, ranks: Dict[str, int], split: str = "train",
            input_names: Optional[Sequence[str]] = None) -> "StateAssembler":
        blocks = [b for b in ds.block_names if ranks.get(b, 0) > 0]
        if not blocks:
            raise ValueError(f"No block with rank > 0 in {ranks}; available {ds.block_names}")
        trajs = ds.split(split)
        scales = {}
        for b in blocks:
            r = ranks[b]
            m = max(float(np.max(np.abs(t.coords[b][:r]))) for t in trajs)
            scales[b] = m if m > 0 else 1.0
        names = list(input_names) if input_names is not None else list(ds.input_names)
        cols = [ds.input_names.index(n) for n in names]
        if cols:
            mu = np.max(np.abs(np.concatenate([t.inputs[:, cols] for t in trajs])), axis=0)
            mu[mu == 0] = 1.0
        else:
            mu = np.zeros(0)
        return cls(blocks=blocks, ranks={b: int(ranks[b]) for b in blocks}, scales=scales,
                   input_names=names, input_scale=mu)

    # ------------------------------------------------------------------
    @property
    def r(self) -> int:
        return int(sum(self.ranks[b] for b in self.blocks))

    @property
    def m(self) -> int:
        return len(self.input_names)

    def slices(self) -> Dict[str, slice]:
        out, i = {}, 0
        for b in self.blocks:
            out[b] = slice(i, i + self.ranks[b])
            i += self.ranks[b]
        return out

    def state(self, traj: ReducedTrajectory) -> np.ndarray:
        """Scaled reduced state X (r, K) of a trajectory (projection of the truth)."""
        return np.vstack([traj.coords[b][:self.ranks[b]] / self.scales[b] for b in self.blocks])

    def inputs(self, traj: ReducedTrajectory, ds_input_names: Sequence[str]) -> np.ndarray:
        """Scaled inputs U (m, K)."""
        if not self.input_names:
            return np.zeros((0, len(traj.time)))
        cols = [list(ds_input_names).index(n) for n in self.input_names]
        return (traj.inputs[:, cols] / self.input_scale).T

    def unscale(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Scaled state (r, K) -> unscaled POD coordinates per block."""
        return {b: X[s] * self.scales[b] for b, s in self.slices().items()}

    def to_dict(self) -> dict:
        return {"blocks": self.blocks, "ranks": self.ranks, "scales": self.scales,
                "input_names": self.input_names,
                "input_scale": (self.input_scale.tolist() if self.input_scale is not None else [])}

    @classmethod
    def from_dict(cls, d: dict) -> "StateAssembler":
        return cls(blocks=list(d["blocks"]), ranks={k: int(v) for k, v in d["ranks"].items()},
                   scales={k: float(v) for k, v in d["scales"].items()}, input_names=list(d["input_names"]),
                   input_scale=np.asarray(d["input_scale"], float))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def block_field_error(traj: ReducedTrajectory, block: str, coords_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Exact full-field error of a predicted block (unscaled coords, (r, K)).

    Returns per-time ``error`` = ||q - q_bar - V_r a_pred||_W, ``projection`` = the
    best-possible error with r modes, and ``norm_ref`` = RMS_t ||q - q_bar||_W.
    """
    r = coords_pred.shape[0]
    a = traj.coords[block][:r]
    sq = traj.sqnorm[block]
    proj2 = np.clip(sq - np.sum(a ** 2, axis=0), 0.0, None)
    err2 = proj2 + np.sum((a - coords_pred) ** 2, axis=0)
    ref = float(np.sqrt(np.mean(sq))) or 1.0
    return {"error": np.sqrt(err2), "projection": np.sqrt(proj2), "norm_ref": ref,
            "relative_error": np.sqrt(err2) / ref, "relative_projection": np.sqrt(proj2) / ref}


# ---------------------------------------------------------------------------
# Output map  y = C [1; x; u] (+ optional quadratic terms)
# ---------------------------------------------------------------------------

def _features(X: np.ndarray, U: np.ndarray, quadratic: bool) -> np.ndarray:
    parts = [np.ones((1, X.shape[1])), X]
    if U.shape[0]:
        parts.append(U)
    if quadratic:
        iu = np.triu_indices(X.shape[0])
        parts.append((X[:, None, :] * X[None, :, :])[iu])
    return np.vstack(parts)


@dataclass
class OutputMap:
    """Ridge-regressed linear(-quadratic) output map from reduced state & input to outputs."""

    names: List[str]
    C: Optional[np.ndarray] = None
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    regularizer: float = 1e-6
    quadratic: bool = False

    def fit(self, Xs: Sequence[np.ndarray], Us: Sequence[np.ndarray], Ys: Sequence[np.ndarray]) -> "OutputMap":
        F = np.hstack([_features(X, U, self.quadratic) for X, U in zip(Xs, Us)])
        Y = np.hstack([y.T for y in Ys])          # (p, K)
        self.y_mean = Y.mean(1)
        self.y_std = Y.std(1)
        self.y_std[self.y_std == 0] = 1.0
        Yn = (Y - self.y_mean[:, None]) / self.y_std[:, None]
        G = F @ F.T
        lam = self.regularizer * np.trace(G) / G.shape[0]
        self.C = np.linalg.solve(G + lam * np.eye(G.shape[0]), F @ Yn.T).T
        return self

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """(K, p) outputs."""
        Yn = self.C @ _features(X, U, self.quadratic)
        return (Yn * self.y_std[:, None] + self.y_mean[:, None]).T

    def to_dict(self) -> dict:
        return {"names": self.names, "C": self.C.tolist(), "y_mean": self.y_mean.tolist(),
                "y_std": self.y_std.tolist(), "regularizer": self.regularizer, "quadratic": self.quadratic}

    @classmethod
    def from_dict(cls, d: dict) -> "OutputMap":
        return cls(names=d["names"], C=np.asarray(d["C"]), y_mean=np.asarray(d["y_mean"]),
                   y_std=np.asarray(d["y_std"]), regularizer=d["regularizer"], quadratic=d["quadratic"])


# ---------------------------------------------------------------------------
# Common model interface
# ---------------------------------------------------------------------------

class ReducedModel(Protocol):
    """Anything that can roll out a reduced state under given inputs."""

    name: str

    def simulate(self, x0: np.ndarray, t: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Return X (r, K) for inputs U (m, K) sampled at times t; NaN-filled on blow-up."""
        ...


def save_json(obj, path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
