"""
Evaluate a trained OpInf ROM on held-out VarFlExI runs.

    python -m rom_opinf.evaluate --model artifacts/opinf --splits val test \
        [--dataset artifacts/reduced/reduced_dataset.h5] [--export-fields]

Each trajectory is predicted from its initial projected state, driven *only* by
its recorded inputs (gust / actuation). Figures + ``metrics.json`` +
``predictions.npz`` go to ``<model>/evaluation/``. ``--export-fields`` also writes
the predicted 3-D fields (+ XDMF) for ParaView.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rom.cli_utils import load_yaml, setup_logging
from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import evaluate_models
from rom_opinf.training import load_opinf

logger = logging.getLogger("rom_opinf.evaluate")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="output directory of rom_opinf.train")
    ap.add_argument("--dataset", help="reduced dataset (default: the one used for training)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", help="default: <model>/evaluation")
    ap.add_argument("--views", help="YAML file with a 'field_views' list (quantity/plane/value)")
    ap.add_argument("--no-fields", action="store_true", help="skip 3-D field slice figures")
    ap.add_argument("--export-fields", action="store_true", help="write predicted fields + XDMF for ParaView")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    model_dir = Path(a.model)
    out = Path(a.out) if a.out else model_dir / "evaluation"
    setup_logging(a.verbose, out / "evaluate.log")
    rom, asm, omap, meta = load_opinf(model_dir)
    ds = ReducedDataset(a.dataset or meta["dataset"])
    views = load_yaml(a.views).get("field_views") if a.views else None
    m = evaluate_models([rom], ds, asm, omap, out, splits=a.splits, field_views=views,
                        make_field_plots=not a.no_fields, export_fields=a.export_fields)
    for name, v in m["trajectories"].items():
        mm = v["models"][rom.name]
        logger.info("%-28s %-5s stable=%s  state err=%.3g  %s", name, v["split"], mm["stable"],
                    mm["state_rel_error_mean"],
                    "  ".join(f"{b}: {x['rel_error_mean']:.3g} (floor {x['projection_floor_mean']:.3g})"
                              for b, x in mm["blocks"].items()))
    logger.info("Figures and metrics in %s", out)


if __name__ == "__main__":
    main()
