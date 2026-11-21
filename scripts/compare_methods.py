#!/usr/bin/env python3
"""
Stage 4 - Side-by-side comparison of trained ROMs (OpInf vs Koopman) on the same runs.

    python scripts/compare_methods.py --opinf artifacts/opinf --koopman artifacts/koopman \
        --splits test --out artifacts/comparison

Every model is rolled out from the same initial projected state with the same
recorded inputs. Figures per trajectory (outputs, output spectra, full-field
error vs time, FOM | OpInf | Koopman | |error| field slices) plus summaries
(field / output error bars, wall-time speed-up, error vs prediction horizon)
and a ``comparison.md`` table you can paste into slides / reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rom.cli_utils import setup_logging  # noqa: E402
from rom.reduced.dataset import ReducedDataset  # noqa: E402
from rom.reduced.evaluation import horizon_errors, plot_field_views, rollout, trajectory_metrics  # noqa: E402
from rom.reduced.state import block_field_error, save_json  # noqa: E402
from rom.viz import plots as vp  # noqa: E402

logger = logging.getLogger("compare_methods")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opinf", help="rom_opinf.train output dir")
    ap.add_argument("--koopman", help="rom_koopman.train output dir")
    ap.add_argument("--dataset")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--out", default="artifacts/comparison")
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100])
    a = ap.parse_args(argv)
    out = Path(a.out)
    setup_logging(False, out / "compare.log")

    entries = []  # (model, assembler, output_map)
    dataset = a.dataset
    if a.opinf:
        from rom_opinf.training import load_opinf

        rom, asm, omap, meta = load_opinf(a.opinf)
        rom.name = "OpInf"
        entries.append((rom, asm, omap))
        dataset = dataset or meta["dataset"]
    if a.koopman:
        from rom_koopman.training import load_koopman

        models, asm, omap, meta = load_koopman(a.koopman)
        for m in models.values():
            entries.append((m, asm, omap))
        dataset = dataset or meta["dataset"]
    if not entries:
        ap.error("give --opinf and/or --koopman")
    ds = ReducedDataset(dataset)
    trajs = [t for s in a.splits for t in ds.split(s)]
    fluid = [b for b in ds.block_names if str(ds.block_info[b].get("source", "")).startswith("field:")]
    fb = fluid[0] if fluid else None

    summary = {"field": {}, "outputs": {}, "rows": []}
    timing = {}
    for traj in trajs:
        tdir = out / traj.name
        Xu, Yp, ferr, floor = {}, {}, {}, None
        for model, asm, omap in entries:
            r = rollout(model, asm, traj, ds)
            Y = traj.outputs if traj.outputs.shape[1] else None
            yp = omap.predict(np.nan_to_num(r["X_pred"]), r["U"]) if (omap is not None and Y is not None) else None
            m = trajectory_metrics(asm, traj, r["X_true"], r["X_pred"], Y, yp, ds.output_names)
            Xu[model.name] = asm.unscale(np.nan_to_num(r["X_pred"]))
            if m["stable"] and yp is not None:
                Yp[model.name] = yp
            if fb and fb in asm.blocks:
                fe = block_field_error(traj, fb, Xu[model.name][fb])
                ferr[model.name] = fe["relative_error"] if m["stable"] else np.full(len(traj.time), np.nan)
                if floor is None or asm.ranks[fb] > floor[0]:
                    floor = (asm.ranks[fb], fe["relative_projection"])
            summary["field"].setdefault(traj.name, {})[model.name] = m["blocks"].get(fb, {}).get("rel_error_mean", np.nan) if fb else np.nan
            summary["outputs"].setdefault(traj.name, {})[model.name] = m.get("output_nrmse_mean", np.nan)
            timing.setdefault(model.name, {"rom_seconds": [], "fom_seconds": []})
            timing[model.name]["rom_seconds"].append(r["wall"])
            timing[model.name]["fom_seconds"].append(traj.fom_walltime_total)
            summary["rows"].append({"trajectory": traj.name, "model": model.name, "stable": m["stable"],
                                    "field_error": summary["field"][traj.name][model.name],
                                    "output_nrmse": m.get("output_nrmse_mean", np.nan),
                                    "valid_time": m["blocks"].get(fb, {}).get("valid_time", np.nan) if fb else np.nan,
                                    "rom_seconds": r["wall"]})
        if Yp:
            vp.plot_outputs(traj.time, traj.outputs, Yp, ds.output_names, tdir / "outputs",
                            title=f"{traj.name} ({traj.split}): outputs - FOM vs ROMs")
            vp.plot_output_psd(traj.time, traj.outputs, Yp, ds.output_names, tdir / "outputs_psd")
        if ferr:
            vp.plot_error_vs_time(traj.time, ferr, floor[1] if floor else None, tdir / f"field_error_{fb}",
                                  title=f"{traj.name}: full-field error of '{fb}'")
            if fb and fb in asm.blocks:
                summary["field"][traj.name][f"POD projection (r={floor[0]})"] = float(np.mean(floor[1]))
        if fb and not a.no_fields:
            try:
                plot_field_views(ds, fb, traj, {k: v for k, v in Xu.items() if np.all(np.isfinite(ferr.get(k, [0])))}, tdir)
            except Exception as exc:
                logger.warning("field views skipped for %s: %s", traj.name, exc)

    vp.plot_error_summary(summary["field"], out / "summary_field_error", f"mean relative error of '{fb}'",
                          title="Full-field error (lower is better)")
    if any(np.isfinite(v).any() for v in [list(d.values()) for d in summary["outputs"].values()]):
        vp.plot_error_summary(summary["outputs"], out / "summary_output_nrmse", "mean output NRMSE",
                              title="Output error (lower is better)")
    rows = {k: {"rom_seconds": float(np.mean(v["rom_seconds"])), "fom_seconds": float(np.nanmean(v["fom_seconds"]))}
            for k, v in timing.items()}
    if any(np.isfinite(r["fom_seconds"]) for r in rows.values()):
        vp.plot_speedup(rows, out / "speedup")
    K = min(len(t.time) for t in trajs)
    hz = [h for h in a.horizons if h < K - 1]
    if hz:
        errs = {m.name: horizon_errors(m, asm_, ds, trajs, hz) for m, asm_, _ in entries}
        vp.plot_horizon_error(hz, errs, out / "horizon_error", dt=trajs[0].dt)
    save_json(summary, out / "comparison.json")

    lines = ["| trajectory | model | stable | field error | output NRMSE | valid time [s] | ROM wall [s] |",
             "|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        lines.append(f"| {r['trajectory']} | {r['model']} | {r['stable']} | {r['field_error']:.3g} | "
                     f"{r['output_nrmse']:.3g} | {r['valid_time']:.3g} | {r['rom_seconds']:.3g} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    logger.info("\n" + "\n".join(lines))
    logger.info("Comparison figures in %s", out)


if __name__ == "__main__":
    main()
