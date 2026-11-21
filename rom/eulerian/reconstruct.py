"""
rom.eulerian.reconstruct - Lagrangian vortex particles -> fixed Eulerian grid, per run.

VarFlExI does **not** write Eulerian fields during a simulation; this module
turns ``<run>/fluid/*_pfield.N.h5`` into a ROM trajectory file
(:mod:`rom.data.trajectory_file`) on ONE fixed grid shared by every run:

    u(x,t) = V_inf(x,t)                     freestream + gust  (exact VarFlExI model)
           + sum_p K_sigma(x - x_p) x Gamma_p  particles        (direct / VIC)
           + u_bound(x,t)                   lifting-line bound vortex (station history)
    w(x,t) = sum_p zeta_sigma(x - x_p) Gamma_p

together with the structural state (plate displacement & velocity), the input /
output signals and reconstruction diagnostics (conservation of total vorticity).

**The grid must be identical across all runs and all times** (POD needs a fixed
spatial discretisation). Earlier VarFlExI analyses used per-step cumulative
bounds - those fields cannot be used for POD.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from rom.eulerian.biot_savart import evaluate_particles
from rom.eulerian.bound_vortex import bound_vortex_velocity
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.vic import vic_reconstruct

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionConfig:
    backend: str = "torch"            # torch | numpy | vic
    kernel: str = "gaussianerf"       # must match the FLOWVPM kernel of the run
    precision: str = "float32"        # torch pair arithmetic precision
    device: Optional[str] = None      # 'cuda', 'cuda:1', 'cpu' (torch)
    include_freestream: bool = True
    include_bound_vortex: bool = True
    bound_vortex_core: Optional[float] = None   # [m]; None -> median particle sigma
    fields: List[str] = field(default_factory=lambda: ["velocity", "vorticity"])
    store_induced: bool = False
    far_field_distance: Optional[float] = None  # [m] lump particles farther than this ...
    far_field_bin: Optional[float] = None       # ... into bins of this edge [m]
    vic_eps: Optional[float] = None
    target_chunk: Optional[int] = None
    source_chunk: Optional[int] = None
    include_static: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ReconstructionConfig":
        d = dict(d or {})
        ff = d.pop("far_field", None)
        if isinstance(ff, dict):
            d.setdefault("far_field_distance", ff.get("distance"))
            d.setdefault("far_field_bin", ff.get("bin"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown reconstruction options: {sorted(unknown)}")
        return cls(**d)


def grid_from_config(d: dict) -> EulerianGrid:
    """Build the fixed grid from ``{'bounds': {x:[..],..}, 'resolution': {x:..,..}}``.

    ``spacing`` (scalar or per-axis) may replace ``resolution``.
    """
    bounds = {ax: tuple(float(v) for v in d["bounds"][ax]) for ax in "xyz"}
    if "resolution" in d:
        res = {ax: int(d["resolution"][ax]) for ax in "xyz"}
    else:
        sp = d["spacing"]
        sp = {ax: float(sp[ax]) for ax in "xyz"} if isinstance(sp, dict) else {ax: float(sp) for ax in "xyz"}
        res = {ax: int(round((bounds[ax][1] - bounds[ax][0]) / sp[ax])) + 1 for ax in "xyz"}
    return EulerianGrid(bounds=bounds, resolution=res, axis_order=d.get("axis_order", "xyz"),
                        spacing_type=d.get("spacing_type", "uniform"), units=d.get("units", "m"))


def cell_volumes(grid: EulerianGrid) -> np.ndarray:
    """Trapezoidal quadrature weights (node 'volumes'), shape ``grid.shape``."""
    w = []
    for ax in grid.axis_order:
        c = grid.coords_1d(ax)
        d = np.diff(c)
        wa = np.zeros_like(c)
        wa[:-1] += 0.5 * d
        wa[1:] += 0.5 * d
        w.append(wa)
    return w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]


class EulerianReconstructor:
    """Evaluate Eulerian velocity / vorticity of a particle snapshot on a fixed grid."""

    def __init__(self, grid: EulerianGrid, config: Optional[ReconstructionConfig] = None):
        self.grid = grid
        self.cfg = config or ReconstructionConfig()
        self.points = grid.flat_points()
        self.volumes = cell_volumes(grid)
        if self.cfg.backend == "vic" and grid.axis_order != "xyz":
            raise ValueError("VIC backend requires axis_order='xyz'")

    def evaluate(self, particles, run=None, station=None) -> Dict[str, np.ndarray]:
        """Return fields shaped ``grid.shape + (3,)``: velocity, vorticity, velocity_induced."""
        cfg = self.cfg
        shp = self.grid.shape + (3,)
        want_w = "vorticity" in cfg.fields
        if cfg.backend == "vic":
            res = vic_reconstruct(self.grid, particles.X, particles.Gamma, particles.sigma, kernel=cfg.kernel,
                                  eps=cfg.vic_eps, far_field_distance=cfg.far_field_distance or 0.5,
                                  far_field_bin=cfg.far_field_bin or 0.1, device=cfg.device,
                                  outside_backend="torch" if _has_torch() else "numpy")
            U = res["velocity"].reshape(-1, 3)
            W = res["vorticity"].reshape(-1, 3)
        else:
            res = evaluate_particles(self.points, particles.X, particles.Gamma, particles.sigma,
                                     kernel=cfg.kernel, velocity=True, vorticity=want_w,
                                     backend=cfg.backend, device=cfg.device, precision=cfg.precision,
                                     target_chunk=cfg.target_chunk, source_chunk=cfg.source_chunk,
                                     far_field_distance=cfg.far_field_distance, far_field_bin=cfg.far_field_bin)
            U = res["velocity"]
            W = res.get("vorticity")
        if cfg.include_bound_vortex and run is not None and station is not None:
            g = run.wing_geometry()
            core = cfg.bound_vortex_core or (float(np.median(particles.sigma)) if particles.n else 0.25 * g["root_chord"])
            Ub = bound_vortex_velocity(self.points, station, particles.step, g["span"], g["root_chord"],
                                       g["tip_chord"], g["eta_cp"], g["eta_bv"], core=core)
            if Ub is not None:
                U = U + Ub
        out = {"velocity_induced": U.reshape(shp)}
        if cfg.include_freestream and run is not None:
            out["velocity"] = (U + run.freestream(self.points, particles.time)).reshape(shp)
        else:
            out["velocity"] = U.reshape(shp)
        if W is not None:
            out["vorticity"] = W.reshape(shp)
        return out

    def vorticity_integral(self, vorticity: np.ndarray) -> np.ndarray:
        return np.einsum("ijk,ijkc->c", self.volumes, vorticity)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Structural state on the snapshot steps
# ---------------------------------------------------------------------------

def structure_on_steps(run, steps: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Plate displacement & velocity (central differences) at ``steps`` (step 0 -> zero)."""
    pd_ = run.plate_displacement()
    if pd_ is None:
        return None
    s_all = np.concatenate([[0], pd_["step"]])
    d_all = np.concatenate([np.zeros((1,) + pd_["displacement"].shape[1:]), pd_["displacement"]])
    t_all = run.time_of_step(s_all)
    v_all = np.gradient(d_all, t_all, axis=0)
    idx = np.searchsorted(s_all, steps)
    idx = np.clip(idx, 0, len(s_all) - 1)
    bad = s_all[idx] != steps
    if bad.any():
        logger.warning("No structural output for %d requested steps - nearest used", int(bad.sum()))
    return {"displacement": d_all[idx], "velocity": v_all[idx],
            "geometry": pd_["geometry"], "topology": pd_["topology"]}


# ---------------------------------------------------------------------------
# Whole-run driver
# ---------------------------------------------------------------------------

def reconstruct_run(run, grid: EulerianGrid, out_path: Union[str, Path], steps: Sequence[int],
                    config: Optional[ReconstructionConfig] = None, input_spec=None, output_spec=None,
                    resume: bool = True, dtype=np.float32, progress: bool = True) -> Path:
    """Reconstruct every requested step of ``run`` and write a trajectory file.

    Resumable: completed snapshots are flagged in ``/metadata/completed`` and
    skipped when the command is re-run.
    """
    import json

    from rom.data.trajectory_file import TrajectoryWriter
    from rom.varflexi.signals import assemble_signals

    cfg = config or ReconstructionConfig()
    steps = np.asarray(sorted(int(s) for s in steps))
    times = run.time_of_step(steps)
    fields = ["velocity"] + (["vorticity"] if "vorticity" in cfg.fields else []) + (["velocity_induced"] if cfg.store_induced else [])
    rec = EulerianReconstructor(grid, cfg)
    station = run.station_history() if cfg.include_bound_vortex else None
    if cfg.include_bound_vortex and station is None:
        logger.warning("No station history in %s - bound vortex NOT included", run.root)

    attrs = {"source_run": str(run.root), "run_name": run.name, "dt": run.dt,
             "reconstruction": json.dumps(asdict(cfg))}
    with TrajectoryWriter(out_path, grid, steps, times, fields=fields, dtype=dtype, resume=resume, attrs=attrs) as w:
        sig = assemble_signals(run, steps, input_spec, output_spec)
        w.write_signals(sig)
        st = structure_on_steps(run, steps)
        if st is not None:
            w.write_structure(st["displacement"], st["velocity"], st["geometry"], st["topology"])
        w.write_parameters(run.config)
        done = w.completed
        todo = [k for k in range(len(steps)) if not done[k]]
        logger.info("%s: %d/%d snapshots to reconstruct on grid %s (%d nodes) with backend=%s",
                    run.name, len(todo), len(steps), grid.shape, grid.num_points, cfg.backend)
        t_start = _time.perf_counter()
        for n, k in enumerate(todo):
            t0 = _time.perf_counter()
            p = run.read_particles(int(steps[k]), include_static=cfg.include_static)
            f = rec.evaluate(p, run=run, station=station)
            diag = {"particle_gamma_sum": p.total_circulation(), "n_particles": p.n,
                    "eval_seconds": _time.perf_counter() - t0}
            if "vorticity" in f:
                diag["grid_vorticity_integral"] = rec.vorticity_integral(f["vorticity"])
            w.write_snapshot(k, f, diag)
            if progress and (n % 10 == 0 or n == len(todo) - 1):
                el = _time.perf_counter() - t_start
                eta = el / (n + 1) * (len(todo) - n - 1)
                logger.info("  step %5d  (%d/%d)  N_p=%7d  %.2fs/snap  ETA %.1f min",
                            steps[k], n + 1, len(todo), p.n, diag["eval_seconds"], eta / 60)
    return Path(out_path)
