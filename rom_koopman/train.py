"""
Train Koopman ROMs (EDMDc linear/bilinear predictor and/or Mauroy-Goncalves lifting) on a reduced dataset.

    python -m rom_koopman.train --dataset artifacts/reduced/reduced_dataset.h5 \
        --config rom_koopman/configs/koopman.yaml --out artifacts/koopman

Writes ``<out>/model/{edmdc,lifting}/``, the shared assembler / output map, and
training figures in ``<out>/figures/``: ridge sweep, Koopman spectrum, spatial
Koopman modes, identified polynomial vector field, fit-quality rollouts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rom.cli_utils import load_yaml, parse_ranks, setup_logging
from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import evaluate_models
from rom_koopman.analysis import koopman_figures
from rom_koopman.plots import plot_ridge_sweep
from rom_koopman.training import KoopmanConfig, save_koopman, train_koopman

logger = logging.getLogger("rom_koopman.train")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "koopman.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", help="override ranks, e.g. fluid=10,structure=3,structure_rate=3")
    ap.add_argument("--methods", nargs="+", choices=["edmdc", "lifting"])
    ap.add_argument("--no-train-plots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    setup_logging(a.verbose, out / "train.log")
    cfg_d = load_yaml(a.config).get("koopman", {})
    if a.ranks:
        cfg_d["ranks"] = parse_ranks(a.ranks)
    if a.methods:
        cfg_d["methods"] = a.methods
    cfg = KoopmanConfig.from_dict(cfg_d)
    ds = ReducedDataset(a.dataset)
    logger.info("\n%s", ds.summary())
    models, asm, omap, sweeps = train_koopman(ds, cfg)
    save_koopman(out, models, asm, omap, sweeps, cfg, a.dataset)
    fig = out / "figures"
    plot_ridge_sweep(sweeps, fig / "ridge_sweep")
    koopman_figures(models, asm, ds, fig)
    if not a.no_train_plots:
        evaluate_models(list(models.values()), ds, asm, omap, out / "train_fit", splits=("train",),
                        make_field_plots=False)
    logger.info("Done. Models in %s/model, figures in %s", out, fig)


if __name__ == "__main__":
    main()
