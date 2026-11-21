#!/usr/bin/env python3
"""
Stage 1 - Lagrangian vortex particles (VarFlExI) -> Eulerian trajectory files on ONE fixed grid.

    python scripts/reconstruct_eulerian.py \
        --run-dir "/media/.../VarFlExI_Results/09-11-momentcons-crm" \
        --config configs/reconstruction.yaml --out-dir data/trajectories

    # several runs at once, a coarse cost estimate first, then the real thing:
    python scripts/reconstruct_eulerian.py --run-dir RUN_A RUN_B --dry-run
    python scripts/reconstruct_eulerian.py --run-dir RUN_A RUN_B

Per run it writes ``<out-dir>/<run_name>.h5`` (+ ``.xmf`` for ParaView) containing
velocity (+ vorticity) snapshots, plate displacement/velocity, inputs/outputs and
diagnostics. Interrupted runs resume where they stopped (``--overwrite`` to restart).

Useful flags
  --steps 1:1:600        Julia-like inclusive range / list (default from config, else all)
  --backend torch|numpy|vic  override the config
  --validate             compare the configured backend against torch-float64 on 2 snapshots
  --qa-steps 100,300,599 write particle/field QA figures for these steps (default: 3 steps)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rom.cli_utils import load_yaml, setup_logging  # noqa: E402
from rom.eulerian import (EulerianReconstructor, ReconstructionConfig, grid_from_config,  # noqa: E402
                          reconstruct_run, write_trajectory_xdmf)
from rom.varflexi import VarFlExIRun, parse_step_spec  # noqa: E402

logger = logging.getLogger("reconstruct_eulerian")


def qa_figures(run, grid, cfg, steps, out_dir):
    from rom.viz.fields import plot_particles_and_field

    rec = EulerianReconstructor(grid, cfg)
    st = run.station_history() if cfg.include_bound_vortex else None
    g = run.wing_geometry()
    for s in steps:
        p = run.read_particles(s)
        f = rec.evaluate(p, run=run, station=st)
        plot_particles_and_field(p.X, p.Gamma, grid, f["velocity"], f.get("vorticity"),
                                 out_dir / f"qa_{run.name}_step{s}", y_plane=0.5 * g["span"], t=p.time,
                                 wing={"span": g["span"], "chord": g["root_chord"]})


def validate(run, grid, cfg, steps):
    ref_cfg = ReconstructionConfig(**{**cfg.__dict__, "backend": "torch", "precision": "float64",
                                      "far_field_distance": None, "far_field_bin": None})
    a, b = EulerianReconstructor(grid, cfg), EulerianReconstructor(grid, ref_cfg)
    st = run.station_history() if cfg.include_bound_vortex else None
    for s in steps:
        p = run.read_particles(s)
        fa, fb = a.evaluate(p, run, st), b.evaluate(p, run, st)
        for k in ("velocity_induced", "vorticity"):
            if k in fa and k in fb:
                e = np.linalg.norm(fa[k] - fb[k]) / max(np.linalg.norm(fb[k]), 1e-300)
                logger.info("validate step %d  %-17s rel. L2 difference vs torch-float64 exact: %.3e", s, k, e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--config", default="configs/reconstruction.yaml")
    ap.add_argument("--out-dir", default="data/trajectories")
    ap.add_argument("--steps")
    ap.add_argument("--backend", choices=["torch", "numpy", "vic"])
    ap.add_argument("--device")
    ap.add_argument("--config-dir", help="VarFlExI *_params.yaml to use for runs without stored config")
    ap.add_argument("--name", help="output name (single run only)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print sizes and a cost estimate only")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--qa-steps")
    ap.add_argument("--no-qa", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    setup_logging(a.verbose, out_dir / "reconstruct.log")
    conf = load_yaml(a.config)
    grid = grid_from_config(conf["grid"])
    rcfg = dict(conf.get("reconstruction", {}))
    if a.backend:
        rcfg["backend"] = a.backend
    if a.device:
        rcfg["device"] = a.device
    cfg = ReconstructionConfig.from_dict(rcfg)
    sig = conf.get("signals", {}) or {}
    logger.info("Grid %s = %d nodes, bounds %s, backend %s", grid.shape, grid.num_points, grid.bounds, cfg.backend)

    for rd in a.run_dir:
        run = VarFlExIRun(rd, config_dir=a.config_dir)
        steps = parse_step_spec(a.steps or conf.get("steps"), run.particle_steps)
        name = a.name if (a.name and len(a.run_dir) == 1) else run.name
        out = out_dir / f"{name}.h5"
        logger.info("Run %s: %d snapshots (steps %d..%d), dt=%.4g s", run.name, len(steps), steps[0], steps[-1], run.dt)
        if a.dry_run:
            p = run.read_particles(steps[-1])
            t0 = time.perf_counter()
            EulerianReconstructor(grid, cfg).evaluate(p, run=run, station=run.station_history())
            sec = time.perf_counter() - t0
            logger.info("  last snapshot: %d particles, %.2f s -> <= %.1f min for the run (%.1f GB float32 output)",
                        p.n, sec, sec * len(steps) / 60,
                        len(steps) * grid.num_points * 3 * 4 * (1 + len(cfg.fields) - 1) / 1e9)
            logger.info("  summary: %s", json.dumps(run.summary(), default=str)[:600])
            continue
        if a.validate:
            validate(run, grid, cfg, [steps[len(steps) // 2], steps[-1]])
        if not a.no_qa:
            qs = parse_step_spec(a.qa_steps, run.particle_steps) if a.qa_steps else \
                [steps[len(steps) // 4], steps[len(steps) // 2], steps[-1]]
            qa_figures(run, grid, cfg, qs, out_dir / "qa")
        reconstruct_run(run, grid, out, steps, cfg, input_spec=sig.get("inputs"), output_spec=sig.get("outputs"),
                        resume=not a.overwrite)
        write_trajectory_xdmf(out)
        logger.info("Wrote %s (+ .xmf)", out)


if __name__ == "__main__":
    main()
