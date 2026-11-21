"""
Evaluate trained Koopman ROMs on held-out VarFlExI runs.

    python -m rom_koopman.evaluate --model artifacts/koopman --splits val test [--export-fields]

Same figure set and metrics as ``rom_opinf.evaluate`` (both call
``rom.reduced.evaluation.evaluate_models``), plus the n-step error-vs-horizon
curve (relevant for MPC, where the model is re-initialised from measurements).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rom.cli_utils import load_yaml, setup_logging
from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import evaluate_models, horizon_errors
from rom.reduced.state import save_json
from rom.viz.plots import plot_horizon_error
from rom_koopman.training import load_koopman

logger = logging.getLogger("rom_koopman.evaluate")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out")
    ap.add_argument("--views", help="YAML with a 'field_views' list")
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--export-fields", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    md = Path(a.model)
    out = Path(a.out) if a.out else md / "evaluation"
    setup_logging(a.verbose, out / "evaluate.log")
    models, asm, omap, meta = load_koopman(md)
    ds = ReducedDataset(a.dataset or meta["dataset"])
    views = load_yaml(a.views).get("field_views") if a.views else None
    m = evaluate_models(list(models.values()), ds, asm, omap, out, splits=a.splits, field_views=views,
                        make_field_plots=not a.no_fields, export_fields=a.export_fields)
    trajs = [t for s in a.splits for t in ds.split(s)]
    K = min(len(t.time) for t in trajs)
    hz = [h for h in a.horizons if h < K - 1]
    if hz:
        errs = {mod.name: horizon_errors(mod, asm, ds, trajs, hz) for mod in models.values()}
        plot_horizon_error(hz, errs, out / "horizon_error", dt=trajs[0].dt)
        save_json({"horizons": hz, **{k: v.tolist() for k, v in errs.items()}}, out / "horizon_error.json")
    for name, v in m["trajectories"].items():
        for mn, mm in v["models"].items():
            logger.info("%-28s %-5s %-16s stable=%s state err=%.3g", name, v["split"], mn, mm["stable"],
                        mm["state_rel_error_mean"])
    logger.info("Figures and metrics in %s", out)


if __name__ == "__main__":
    main()
