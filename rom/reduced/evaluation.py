"""
rom.reduced.evaluation - Method-agnostic ROM evaluation and the standard figure set.

Given one or more fitted :class:`rom.reduced.state.ReducedModel` objects that share a
:class:`StateAssembler`, :func:`evaluate_models` rolls every model out from the
initial projected state of each trajectory, *driven only by the recorded inputs*,
and reports

* reduced-state error, exact full-field error per block (vs. the POD projection floor),
* output errors (lift, tip deflection, ...) through the learned output map,
* stability, valid-prediction time, ROM wall time and speed-up over VarFlExI,

and writes the figures that are used to communicate results:

    <traj>/rollout_coordinates.png   <traj>/field_error_<block>.png
    <traj>/outputs.png               <traj>/outputs_psd.png
    <traj>/field_<quantity>_<plane>.png (FOM | ROM | |error| slices)
    summary_field_error.png          summary_output_nrmse.png      speedup.png
"""

from __future__ import annotations

import json
import logging
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.reduced.dataset import ReducedDataset, ReducedTrajectory
from rom.reduced.state import OutputMap, StateAssembler, block_field_error, save_json
from rom.viz import fields as vf
from rom.viz import plots as vp

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = [
    {"quantity": "wy", "plane": "xz", "value": "midspan"},
    {"quantity": "wx", "plane": "yz", "value": "mid_x"},
    {"quantity": "uz_fluct", "plane": "xz", "value": "midspan"},
]


def is_stable(X: np.ndarray, X_ref: np.ndarray, factor: float = 50.0) -> bool:
    return bool(np.all(np.isfinite(X)) and np.max(np.abs(X)) <= factor * max(np.max(np.abs(X_ref)), 1e-12))


def rollout(model, asm: StateAssembler, traj: ReducedTrajectory, ds: ReducedDataset) -> Dict[str, np.ndarray]:
    X = asm.state(traj)
    U = asm.inputs(traj, ds.input_names)
    t0 = _time.perf_counter()
    try:
        Xp = model.simulate(X[:, 0], traj.time, U)
    except Exception as exc:  # pragma: no cover - reported as unstable
        logger.warning("%s failed on %s: %s", getattr(model, "name", "model"), traj.name, exc)
        Xp = np.full_like(X, np.nan)
    wall = _time.perf_counter() - t0
    if Xp.shape[1] < X.shape[1]:
        Xp = np.hstack([Xp, np.full((X.shape[0], X.shape[1] - Xp.shape[1]), np.nan)])
    return {"X_true": X, "X_pred": Xp, "U": U, "wall": wall}


def trajectory_metrics(asm: StateAssembler, traj: ReducedTrajectory, X: np.ndarray, Xp: np.ndarray,
                       Y: Optional[np.ndarray], Yp: Optional[np.ndarray], output_names: List[str],
                       valid_threshold: float = 0.3) -> Dict:
    stable = is_stable(Xp, X)
    m: Dict = {"stable": stable}
    num = np.linalg.norm(np.nan_to_num(X - Xp, nan=1e6), axis=0)
    rms = float(np.sqrt(np.mean(np.sum(X ** 2, 0)))) or 1.0
    m["state_rel_error_mean"] = float(np.mean(num) / rms) if stable else float("inf")
    m["state_rel_error_final"] = float(num[-1] / rms) if stable else float("inf")
    coords = asm.unscale(np.nan_to_num(Xp, nan=0.0))
    m["blocks"] = {}
    for b, a in coords.items():
        fe = block_field_error(traj, b, a)
        e = fe["relative_error"] if stable else np.full_like(fe["relative_error"], np.inf)
        bad = np.nonzero(e > valid_threshold)[0]
        m["blocks"][b] = {"rel_error_mean": float(np.mean(e)), "rel_error_final": float(e[-1]),
                          "projection_floor_mean": float(np.mean(fe["relative_projection"])),
                          "valid_time": float(traj.time[bad[0]] - traj.time[0]) if bad.size else float(traj.time[-1] - traj.time[0])}
    if Y is not None and Yp is not None and Y.shape[1]:
        m["outputs"] = {}
        for i, n in enumerate(output_names):
            sd = float(np.std(Y[:, i])) or 1.0
            rmse = float(np.sqrt(np.mean((Y[:, i] - Yp[:, i]) ** 2))) if stable else float("inf")
            m["outputs"][n] = {"rmse": rmse, "nrmse": rmse / sd}
        m["output_nrmse_mean"] = float(np.mean([v["nrmse"] for v in m["outputs"].values()]))
    return m


def horizon_errors(model, asm: StateAssembler, ds: ReducedDataset, trajs: Sequence[ReducedTrajectory],
                   horizons: Sequence[int], n_starts: int = 12) -> np.ndarray:
    """Mean relative state error after h steps, re-initialising from the projected truth at
    ``n_starts`` start points per trajectory (what an MPC with state feedback experiences)."""
    horizons = np.asarray(horizons, int)
    acc = np.zeros(len(horizons))
    cnt = 0
    for t in trajs:
        X = asm.state(t)
        U = asm.inputs(t, ds.input_names)
        K = X.shape[1]
        H = int(horizons.max())
        if K <= H + 1:
            continue
        rms = np.sqrt(np.mean(np.sum(X ** 2, 0))) or 1.0
        for k0 in np.linspace(0, K - H - 1, n_starts).astype(int):
            Xp = model.simulate(X[:, k0], t.time[k0:k0 + H + 1], U[:, k0:k0 + H + 1])
            e = np.linalg.norm(np.nan_to_num(Xp[:, horizons] - X[:, k0 + horizons], nan=1e3), axis=0) / rms
            acc += e
            cnt += 1
    return acc / max(cnt, 1)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

def _wing(traj_file) -> dict:
    p = traj_file.parameters
    return {"span": float(p.get("solid.span", 0.96)), "chord": float(p.get("solid.root_chord", 0.24))}


def field_snapshots(ds: ReducedDataset, block: str, traj: ReducedTrajectory, sel: np.ndarray,
                    coords_unscaled: Dict[str, np.ndarray]):
    """Truth and ROM fields (len(sel), Nx, Ny, Nz, 3) for a field block on the block grid."""
    from rom.data.trajectory_file import TrajectoryFile
    from rom.reduced.dataset import _crop_slices

    info = ds.block_info[block]
    source = str(info.get("source", ""))
    if not source.startswith("field:"):
        return None
    tf = TrajectoryFile(traj.source_file)
    slices, grid = _crop_slices(tf.grid, info.get("crop"))
    truth = tf.read_field(source.split(":")[1], traj.index[sel])[(slice(None),) + slices]
    basis = ds.basis(block)
    preds = {}
    for name, coords in coords_unscaled.items():
        a = coords[block][:, sel]
        q = basis.reconstruct(a)  # (n, len(sel))
        preds[name] = q.T.reshape((len(sel),) + grid.shape + (3,))
    mean = basis.mean.reshape(grid.shape + (3,))
    return grid, truth.astype(np.float64), preds, mean, tf


def plot_field_views(ds, block, traj, coords_unscaled: Dict[str, np.ndarray], out_dir: Path,
                     n_times: int = 3, views: Optional[List[dict]] = None):
    K = len(traj.time)
    sel = np.unique(np.linspace(int(0.3 * (K - 1)), K - 1, n_times).astype(int))
    res = field_snapshots(ds, block, traj, sel, coords_unscaled)
    if res is None:
        return []
    grid, truth, preds, mean, tf = res
    wing = _wing(tf)
    is_vort = ds.block_info[block].get("source") == "field:vorticity"
    paths = []
    for v in views or DEFAULT_VIEWS:
        q, plane = v["quantity"], v["plane"]
        if is_vort and q.startswith("u"):
            continue
        value = v.get("value", 0.0)
        if value == "midspan":
            value = 0.5 * wing["span"]
        elif value == "mid_x":
            value = float(np.mean(grid.bounds["x"]))
        if is_vort:
            sf = lambda f: vf.scalar_field(None, grid, q, vort=f)  # noqa: E731
        else:
            sf = lambda f: vf.scalar_field(f, grid, q, ref=mean)  # noqa: E731
        T = [sf(truth[i]) for i in range(len(sel))]
        P = {n: [sf(p[i]) for i in range(len(sel))] for n, p in preds.items()}
        paths.append(vf.plot_field_comparison(grid, traj.time[sel], T, P, out_dir / f"field_{q}_{plane}",
                                              quantity=q, plane=plane, value=value, wing=wing,
                                              title=f"{traj.name}: " + vf.LABELS.get(q.replace('_fluct', ''), q)
                                              + (" (fluctuation)" if q.endswith("_fluct") else "")
                                              + f", {plane}-plane"))
    return paths


def export_prediction_fields(ds: ReducedDataset, asm: StateAssembler, traj: ReducedTrajectory, Xp: np.ndarray,
                             block: str, out_h5: Path) -> Optional[Path]:
    """Write ROM-predicted fields in trajectory-file format (+ XDMF) for ParaView."""
    from rom.data.trajectory_file import TrajectoryWriter
    from rom.eulerian.xdmf import write_trajectory_xdmf
    from rom.reduced.dataset import _crop_slices
    from rom.data.trajectory_file import TrajectoryFile

    info = ds.block_info[block]
    source = str(info.get("source", ""))
    if not source.startswith("field:"):
        return None
    tf = TrajectoryFile(traj.source_file)
    _, grid = _crop_slices(tf.grid, info.get("crop"))
    basis = ds.basis(block)
    coords = asm.unscale(np.nan_to_num(Xp))[block]
    fname = source.split(":")[1]
    with TrajectoryWriter(out_h5, grid, traj.step, traj.time, fields=(fname,), resume=False,
                          attrs={"run_name": traj.name + "_rom"}) as w:
        for k in range(len(traj.time)):
            w.write_snapshot(k, {fname: basis.reconstruct(coords[:, k]).reshape(grid.shape + (3,))})
    write_trajectory_xdmf(out_h5)
    return out_h5


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def evaluate_models(models: Sequence, ds: ReducedDataset, asm: StateAssembler, output_map: Optional[OutputMap],
                    out_dir, splits: Sequence[str] = ("val", "test"), field_views: Optional[List[dict]] = None,
                    n_field_times: int = 3, make_field_plots: bool = True, export_fields: bool = False,
                    n_coord_plot: int = 6) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = [t for s in splits for t in ds.split(s)]
    if not trajs:
        raise ValueError(f"No trajectories in splits {splits}")
    all_metrics: Dict = {"splits": list(splits), "trajectories": {}}
    field_blocks = [b for b in asm.blocks if str(ds.block_info.get(b, {}).get("source", "")).startswith("field:")]
    for traj in trajs:
        tdir = out_dir / traj.name
        tdir.mkdir(exist_ok=True)
        preds, metrics, Ys = {}, {}, {}
        Y = traj.outputs if traj.outputs.shape[1] else None
        for model in models:
            r = rollout(model, asm, traj, ds)
            Xp = r["X_pred"]
            Yp = output_map.predict(np.nan_to_num(Xp), r["U"]) if (output_map is not None and Y is not None) else None
            m = trajectory_metrics(asm, traj, r["X_true"], Xp, Y, Yp, ds.output_names)
            m["rom_wall_seconds"] = r["wall"]
            m["fom_wall_seconds"] = traj.fom_walltime_total
            m["speedup"] = traj.fom_walltime_total / r["wall"] if r["wall"] > 0 else float("nan")
            preds[model.name] = Xp
            Ys[model.name] = Yp
            metrics[model.name] = m
            X_true, U = r["X_true"], r["U"]
            if export_fields and field_blocks:
                export_prediction_fields(ds, asm, traj, Xp, field_blocks[0], tdir / f"fields_{model.name}.h5")
        all_metrics["trajectories"][traj.name] = {"split": traj.split, "models": metrics}
        np.savez_compressed(tdir / "predictions.npz", t=traj.time, X_true=X_true, U=U,
                            Y_true=(Y if Y is not None else np.zeros((len(traj.time), 0))),
                            **{f"X_pred__{k}": v for k, v in preds.items()},
                            **{f"Y_pred__{k}": v for k, v in Ys.items() if v is not None})

        # ---- figures --------------------------------------------------------
        sl = asm.slices()
        for b in asm.blocks[:2]:
            s = sl[b]
            vp.plot_rollout_coordinates(traj.time, X_true[s], {k: v[s] for k, v in preds.items()},
                                        tdir / f"rollout_coordinates_{b}", n=n_coord_plot,
                                        labels=[f"{b} $x_{{{i + 1}}}$" for i in range(s.stop - s.start)],
                                        title=f"{traj.name} ({traj.split}): reduced coordinates of '{b}'")
        for b in asm.blocks:
            errs, floor = {}, None
            for k, Xp in preds.items():
                fe = block_field_error(traj, b, asm.unscale(np.nan_to_num(Xp, nan=0.0))[b])
                errs[k] = fe["relative_error"] if metrics[k]["stable"] else np.full(len(traj.time), np.nan)
                floor = fe["relative_projection"]
            vp.plot_error_vs_time(traj.time, errs, floor, tdir / f"field_error_{b}",
                                  title=f"{traj.name}: full-field error of '{b}' (r={asm.ranks[b]})",
                                  ylabel=r"$\|q-\hat q\|_W \,/\, \mathrm{rms}_t\|q-\bar q\|_W$")
        if Y is not None and output_map is not None:
            okY = {k: v for k, v in Ys.items() if v is not None and metrics[k]["stable"]}
            vp.plot_outputs(traj.time, Y, okY, ds.output_names, tdir / "outputs",
                            title=f"{traj.name} ({traj.split}): control-relevant outputs")
            vp.plot_output_psd(traj.time, Y, okY, ds.output_names, tdir / "outputs_psd",
                               title=f"{traj.name}: output spectra")
        if make_field_plots and field_blocks:
            try:
                cu = {k: asm.unscale(np.nan_to_num(v, nan=0.0)) for k, v in preds.items() if metrics[k]["stable"]}
                if cu:
                    plot_field_views(ds, field_blocks[0], traj, cu, tdir, n_field_times, field_views)
            except Exception as exc:  # fields optional (e.g. trajectory files moved)
                logger.warning("Field plots skipped for %s: %s", traj.name, exc)

    # ---- summaries -------------------------------------------------------
    names = [m.name for m in models]
    if field_blocks:
        b = field_blocks[0]
        tab = {t: {n: v["models"][n]["blocks"][b]["rel_error_mean"] for n in names}
               for t, v in all_metrics["trajectories"].items()}
        for t, v in all_metrics["trajectories"].items():
            tab[t]["POD projection"] = v["models"][names[0]]["blocks"][b]["projection_floor_mean"]
        vp.plot_error_summary(tab, out_dir / "summary_field_error", f"mean relative error of '{b}'",
                              title="Full-field error per trajectory")
    if any("output_nrmse_mean" in v["models"][names[0]] for v in all_metrics["trajectories"].values()):
        tab = {t: {n: v["models"][n].get("output_nrmse_mean", np.nan) for n in names}
               for t, v in all_metrics["trajectories"].items()}
        vp.plot_error_summary(tab, out_dir / "summary_output_nrmse", "mean output NRMSE (RMSE / std)",
                              title="Output error per trajectory")
    rows = {}
    for n in names:
        rom = [v["models"][n]["rom_wall_seconds"] for v in all_metrics["trajectories"].values()]
        fom = [v["models"][n]["fom_wall_seconds"] for v in all_metrics["trajectories"].values()]
        rows[n] = {"rom_seconds": float(np.mean(rom)), "fom_seconds": float(np.nanmean(fom)) if np.isfinite(fom).any() else np.nan}
    if any(np.isfinite(r["fom_seconds"]) for r in rows.values()):
        vp.plot_speedup(rows, out_dir / "speedup")
    all_metrics["timing"] = rows
    save_json(all_metrics, out_dir / "metrics.json")
    return all_metrics
