"""
rom.varflexi.run_reader - Read a *raw* VarFlExI run directory.

A VarFlExI run (``results/<run_name>/``) written by
``VarFLEXI-rVPM-Fenics/scripts/run_case.py`` looks like::

    <run>/
      run_configuration.json            # copies of fluid/solid/coupling YAML
      config/{fluid,solid,coupling}_params.yaml
      coupling/coupling_history.csv     # lift, drag, cl, cd, cm, timings, ...
      fluid/<name>_pfield.<N>.h5        # Lagrangian vortex particles (X, Gamma, sigma, ...)
      fluid/<name>_pfield.<N>.xmf       # ParaView companion; holds the physical time
      fluid/<name>_staticpfield.<N>.h5  # (optional) static/surface particles
      fluid/<name>_aerodynamic_station_history.csv   # Gamma(y,t), sectional loads
      fluid/<name>_wake_probe_history.csv
      solid/elastodynamics-results.h5   # plate displacement (FEniCS XDMF/HDF5)
      solid/solid_state_history.csv     # tip / mid-span DOFs, energies

Older runs (e.g. ``publish_result/results_validation/*``) may lack the
configuration files; pass ``config_override`` or ``config_dir`` then.

Step / time alignment convention
--------------------------------
Every table is keyed by the integer ``step`` it reports, and the canonical
time of step ``k`` is ``t_k = k * dt`` with ``dt = total_time / n_steps``.

* ``pfield.<N>.h5``            -> step N (state *after* N fluid steps; N=0 is empty)
* ``coupling_history.csv``     -> column ``step`` (its ``time`` column = step*dt)
* ``solid_state_history.csv``  -> column ``step``
* ``VisualisationVector/<i>``  -> step i+1 (XDMF time = (i+1)*dt)
* station / wake-probe history -> column ``step`` (their ``time_s`` column is the
  time at the *start* of the fluid step, i.e. (step-1)*dt; we ignore it and
  key by ``step``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_PFIELD_RE = re.compile(r"^(?P<prefix>.+)_pfield\.(?P<step>\d+)\.h5$")
_XMF_TIME_RE = re.compile(r'<Time\s+Value="([^"]+)"')


# ---------------------------------------------------------------------------
# Particle snapshot container
# ---------------------------------------------------------------------------

@dataclass
class ParticleSnapshot:
    """Vortex particles at one saved step (only physically valid particles)."""

    step: int
    time: float
    X: np.ndarray       # (N, 3) positions [m]
    Gamma: np.ndarray   # (N, 3) vector circulation (strength) [m^3/s]
    sigma: np.ndarray   # (N,)  smoothing radius [m]
    n_static: int = 0

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def total_circulation(self) -> np.ndarray:
        """Sum of particle strengths = integral of vorticity over all space."""
        return self.Gamma.sum(axis=0) if self.n else np.zeros(3)

    def linear_impulse(self) -> np.ndarray:
        """Hydrodynamic impulse I = 1/2 sum x_p x Gamma_p (per unit density)."""
        if self.n == 0:
            return np.zeros(3)
        return 0.5 * np.cross(self.X, self.Gamma).sum(axis=0)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def load_run_config(run_dir: Union[str, Path], config_dir: Optional[Union[str, Path]] = None) -> Dict[str, dict]:
    """Return ``{'fluid': {...}, 'solid': {...}, 'coupling': {...}}``.

    Priority: explicit ``config_dir`` > ``run_configuration.json`` > ``<run>/config/*.yaml``.
    Missing sections are returned as empty dicts (callers fall back to defaults).
    """
    run_dir = Path(run_dir)
    out = {"fluid": {}, "solid": {}, "coupling": {}}
    names = {"fluid": "fluid_params.yaml", "solid": "solid_params.yaml", "coupling": "coupling_params.yaml"}

    def _from_dir(d: Path) -> bool:
        found = False
        for key, fname in names.items():
            p = d / fname
            if p.is_file():
                out[key] = _load_yaml(p)
                found = True
        return found

    if config_dir is not None:
        if not _from_dir(Path(config_dir)):
            raise FileNotFoundError(f"No *_params.yaml found in {config_dir}")
        return out

    rc = run_dir / "run_configuration.json"
    if rc.is_file():
        with open(rc) as fh:
            cfgs = json.load(fh).get("configs", {})
        for key, fname in names.items():
            out[key] = dict(cfgs.get(fname, {}))
        return out

    if (run_dir / "config").is_dir() and _from_dir(run_dir / "config"):
        return out

    logger.warning("No configuration found for run %s - physical defaults will be used. "
                   "Pass config_dir=... to be safe.", run_dir)
    return out


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

class VarFlExIRun:
    """Lazy reader for one VarFlExI run directory.

    Parameters
    ----------
    run_dir : path
        The run root (contains ``fluid/``, ``solid/``, ``coupling/``).
    config_dir : path, optional
        Directory with ``*_params.yaml`` overriding the stored configuration.
    dt : float, optional
        Override the time step (otherwise ``total_time / n_steps`` or inferred
        from the particle XMF files).
    """

    def __init__(self, run_dir: Union[str, Path], config_dir: Optional[Union[str, Path]] = None,
                 dt: Optional[float] = None):
        self.root = Path(run_dir).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.fluid_dir = self.root / "fluid"
        self.solid_dir = self.root / "solid"
        self.coupling_dir = self.root / "coupling"
        self.config = load_run_config(self.root, config_dir)

        self._steps: Dict[int, Path] = {}
        prefixes = set()
        if self.fluid_dir.is_dir():
            for p in self.fluid_dir.iterdir():
                # "<name>_staticpfield.N.h5" does not match "_pfield." (no underscore
                # before "pfield"), so static particle files are never picked up here.
                m = _PFIELD_RE.match(p.name)
                if m:
                    self._steps[int(m.group("step"))] = p
                    prefixes.add(m.group("prefix"))
        if len(prefixes) > 1:
            raise ValueError(f"Several particle prefixes in {self.fluid_dir}: {sorted(prefixes)}")
        self.prefix: Optional[str] = prefixes.pop() if prefixes else None
        self.name: str = self.prefix or self.config["fluid"].get("run_name", self.root.name)

        self.dt = float(dt) if dt is not None else self._infer_dt()

    # ------------------------------------------------------------------ basics
    def _infer_dt(self) -> float:
        c = self.config["coupling"]
        if c.get("total_time") and c.get("n_steps"):
            return float(c["total_time"]) / float(c["n_steps"])
        steps = self.particle_steps
        for s in steps:
            if s > 0:
                t = self._xmf_time(s)
                if t is not None and not np.isclose(t, s):
                    return t / s
        ch = self.coupling_history()
        if ch is not None and len(ch) > 1:
            return float(np.median(np.diff(ch["time"].to_numpy())))
        raise ValueError(f"Cannot infer dt for run {self.root}; pass dt=...")

    @property
    def particle_steps(self) -> List[int]:
        return sorted(self._steps)

    def time_of_step(self, step: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        return np.asarray(step, dtype=np.float64) * self.dt

    def _xmf_time(self, step: int) -> Optional[float]:
        p = self._steps.get(step)
        if p is None:
            return None
        xmf = p.with_suffix(".xmf")
        if not xmf.is_file():
            return None
        m = _XMF_TIME_RE.search(xmf.read_text(errors="ignore"))
        return float(m.group(1)) if m else None

    # --------------------------------------------------------------- particles
    def read_particles(self, step: int, include_static: bool = True,
                       drop_invalid: bool = True) -> ParticleSnapshot:
        """Read the vortex particles saved at ``step``.

        ``pfield.0`` contains a single dummy particle (sigma=0) - such entries
        are dropped (``sigma <= 0``, non-finite, or zero strength).
        """
        if step not in self._steps:
            raise KeyError(f"No particle file for step {step} in {self.fluid_dir}")
        X, G, S = [], [], []
        n_static = 0
        files = [self._steps[step]]
        static = self.fluid_dir / f"{self.prefix}_staticpfield.{step}.h5"
        if include_static and static.is_file():
            files.append(static)
        for i, f in enumerate(files):
            with h5py.File(f, "r") as h:
                x = np.asarray(h["X"][()], dtype=np.float64).reshape(-1, 3)
                g = np.asarray(h["Gamma"][()], dtype=np.float64).reshape(-1, 3)
                s = np.asarray(h["sigma"][()], dtype=np.float64).reshape(-1)
            if drop_invalid:
                ok = (s > 0) & np.all(np.isfinite(x), 1) & np.all(np.isfinite(g), 1) & np.isfinite(s)
                ok &= np.any(g != 0.0, axis=1)
                x, g, s = x[ok], g[ok], s[ok]
            if i == 1:
                n_static = len(s)
            X.append(x)
            G.append(g)
            S.append(s)
        return ParticleSnapshot(step=step, time=float(self.time_of_step(step)),
                                X=np.concatenate(X), Gamma=np.concatenate(G),
                                sigma=np.concatenate(S), n_static=n_static)

    # ------------------------------------------------------------- CSV tables
    def _csv(self, path: Path):
        import pandas as pd

        if not path.is_file():
            return None
        return pd.read_csv(path)

    def _fluid_csv(self, suffix: str):
        if self.prefix is None or not self.fluid_dir.is_dir():
            cands = list(self.fluid_dir.glob(f"*{suffix}")) if self.fluid_dir.is_dir() else []
            return self._csv(cands[0]) if cands else None
        p = self.fluid_dir / f"{self.prefix}{suffix}"
        if not p.is_file():
            cands = list(self.fluid_dir.glob(f"*{suffix}"))
            p = cands[0] if cands else p
        return self._csv(p)

    def coupling_history(self):
        """Integrated loads & solver diagnostics (pandas DataFrame) keyed by ``step``."""
        df = self._csv(self.coupling_dir / "coupling_history.csv")
        if df is None:
            df = self._fluid_csv("_coupling_diagnostics.csv")
        return df

    def solid_state_history(self):
        """Tip / mid-span displacement, velocity, acceleration, energies."""
        return self._csv(self.solid_dir / "solid_state_history.csv")

    def wake_probe_history(self):
        return self._fluid_csv("_wake_probe_history.csv")

    def station_history(self) -> Optional[Dict[str, np.ndarray]]:
        """Span-wise station data pivoted to ``(K, n_station)`` arrays.

        Returns a dict with key ``'step'`` (K,) and one array per numeric column
        (e.g. ``bound_circulation_m2_s``, ``x_cp_m``, ``lift_N``, ``cl_section``).
        """
        df = self._fluid_csv("_aerodynamic_station_history.csv")
        if df is None:
            return None
        df = df.sort_values(["step", "station"])
        steps = np.unique(df["step"].to_numpy())
        stations = np.unique(df["station"].to_numpy())
        out: Dict[str, np.ndarray] = {"step": steps, "station": stations}
        for col in df.columns:
            if col in ("step", "station", "time_s"):
                continue
            piv = df.pivot(index="step", columns="station", values=col)
            out[col] = piv.reindex(index=steps, columns=stations).to_numpy(dtype=np.float64)
        return out

    # ------------------------------------------------------------- structure
    def plate_displacement(self) -> Optional[Dict[str, np.ndarray]]:
        """Plate displacement history from ``solid/elastodynamics-results.h5``.

        Returns ``{'step': (K,), 'geometry': (Nn, 2), 'topology': (Ne, 3),
        'displacement': (K, Nn, 3)}`` with step = dataset index + 1.
        """
        h5p = self.solid_dir / "elastodynamics-results.h5"
        if not h5p.is_file():
            return None
        with h5py.File(h5p, "r") as h:
            geo = np.asarray(h["Mesh/0/mesh/geometry"][()])
            topo = np.asarray(h["Mesh/0/mesh/topology"][()])
            keys = sorted((int(k) for k in h["VisualisationVector"].keys()))
            disp = np.stack([np.asarray(h[f"VisualisationVector/{k}"][()]) for k in keys])
        steps = np.asarray(keys, dtype=int) + 1
        xdmf = self.solid_dir / "elastodynamics-results.xdmf"
        if xdmf.is_file():
            times = [float(t) for t in _XMF_TIME_RE.findall(xdmf.read_text(errors="ignore"))]
            if len(times) == len(keys):
                inferred = np.rint(np.asarray(times) / self.dt).astype(int)
                if not np.array_equal(inferred, steps):
                    logger.warning("Solid XDMF times do not follow (index+1)*dt; using XDMF times.")
                    steps = inferred
        return {"step": steps, "geometry": geo, "topology": topo, "displacement": disp}

    # -------------------------------------------------------------- freestream
    def freestream_params(self) -> dict:
        f = self.config["fluid"]
        return {
            "vinf": float(f.get("vinf", 15.0)),
            "aoa_deg": float(f.get("aoa_deg", 0.0)),
            "gradient": np.asarray(f.get("freestream_gradient", [0.0] * 9), dtype=float),
            "oscillation": np.asarray(f.get("freestream_oscillation", [0.0] * 3), dtype=float),
            "wavenumber": np.asarray(f.get("freestream_wavenumber", [0.0] * 3), dtype=float),
            "frequency_hz": float(f.get("freestream_frequency_hz", 0.0)),
            "phase_rad": np.deg2rad(float(f.get("freestream_phase_deg", 0.0))),
            "reference": np.asarray(f.get("freestream_reference", [0.0] * 3), dtype=float),
        }

    def freestream(self, X: np.ndarray, t: float) -> np.ndarray:
        """Ambient velocity V(x,t) - exact port of VarFlExI ``freestream_at``.

        ``V = V0 [cos a, 0, sin a] + G (x - x_ref) + A * sin(2 pi f t + phi + [ky y, kz z, kx x])``
        """
        p = self.freestream_params()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d = X - p["reference"]
        a = np.deg2rad(p["aoa_deg"])
        base = p["vinf"] * np.array([np.cos(a), 0.0, np.sin(a)])
        G = p["gradient"].reshape(3, 3)  # row-major, as permutedims(reshape(.,3,3)) in Julia
        kx, ky, kz = p["wavenumber"]
        phase = 2 * np.pi * p["frequency_hz"] * t + p["phase_rad"]
        gust = p["oscillation"] * np.sin(np.stack([phase + ky * d[:, 1], phase + kz * d[:, 2],
                                                   phase + kx * d[:, 0]], axis=1))
        return base + d @ G.T + gust

    # ------------------------------------------------------------ geometry
    def wing_geometry(self) -> dict:
        s = self.config["solid"]
        f = self.config["fluid"]
        return {
            "span": float(s.get("span", f.get("span", 0.96))),
            "root_chord": float(s.get("root_chord", f.get("root_chord", 0.24))),
            "tip_chord": float(s.get("tip_chord", f.get("tip_chord", 0.24))),
            "eta_cp": float(f.get("eta_cp", 0.75)),
            "eta_bv": float(f.get("eta_bv", 0.25)),
            "sigma_surface": float(f.get("sigma_surface", 0.0025)) * float(s.get("span", 0.96)),
        }

    def reference_point(self) -> np.ndarray:
        """Wing quarter-chord at mid-span: where gust inputs are sampled."""
        g = self.wing_geometry()
        c = 0.5 * (g["root_chord"] + g["tip_chord"])
        return np.array([0.25 * c, 0.5 * g["span"], 0.0])

    def summary(self) -> dict:
        return {
            "root": str(self.root), "name": self.name, "dt": self.dt,
            "n_particle_snapshots": len(self._steps),
            "first_step": self.particle_steps[0] if self._steps else None,
            "last_step": self.particle_steps[-1] if self._steps else None,
            "has_coupling_history": (self.coupling_dir / "coupling_history.csv").is_file(),
            "has_solid_results": (self.solid_dir / "elastodynamics-results.h5").is_file(),
            "freestream": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in self.freestream_params().items()},
        }


def parse_step_spec(spec: Optional[str], available: Sequence[int]) -> List[int]:
    """Parse ``'0:10:300'`` / ``'1,2,5'`` / ``'100:'`` (Julia-like, inclusive) into steps."""
    available = sorted(int(s) for s in available)
    if not spec or spec in ("all", ":"):
        return available
    out: List[int] = []
    for part in str(spec).split(","):
        f = part.strip().split(":")
        if len(f) == 1:
            out.append(int(f[0]))
        elif len(f) == 2:
            lo = int(f[0]) if f[0] else available[0]
            hi = int(f[1]) if f[1] else available[-1]
            out.extend(range(lo, hi + 1))
        elif len(f) == 3:
            lo = int(f[0]) if f[0] else available[0]
            hi = int(f[2]) if f[2] else available[-1]
            out.extend(range(lo, hi + 1, int(f[1])))
        else:
            raise ValueError(f"Bad step selector '{part}'")
    avail = set(available)
    missing = [s for s in out if s not in avail]
    if missing:
        raise ValueError(f"Requested steps without particle files: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return sorted(set(out))
