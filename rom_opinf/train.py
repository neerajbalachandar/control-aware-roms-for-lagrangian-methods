"""
Train an input-driven OpInf ROM on a reduced dataset.

    python -m rom_opinf.train --dataset artifacts/reduced/reduced_dataset.h5 \
        --config rom_opinf/configs/opinf.yaml --out artifacts/opinf

Writes ``<out>/model/`` (operators, assembler, output map, config) and training
figures in ``<out>/figures/``: regularisation sweep, operator structure,
spectra, optional rank sweep, and fit-quality rollouts on the training runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rom.cli_utils import load_yaml, parse_ranks, setup_logging
from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import evaluate_models
from rom.reduced.state import save_json
from rom_opinf.plots import plot_operator_spectra, plot_operator_structure, plot_rank_sweep, plot_regularization_sweep
from rom_opinf.training import OpInfConfig, rank_sweep, save_opinf, train_opinf

logger = logging.getLogger("rom_opinf.train")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="reduced_dataset.h5")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "opinf.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", help="override ranks, e.g. fluid=12,structure=4,structure_rate=4")
    ap.add_argument("--operators", help="override operator string, e.g. cAHBN")
    ap.add_argument("--form", choices=["continuous", "discrete"])
    ap.add_argument("--rank-sweep", action="store_true", help="run the rank sweep of the config")
    ap.add_argument("--no-train-plots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    setup_logging(a.verbose, out / "train.log")

    cfg_d = load_yaml(a.config).get("opinf", {})
    if a.ranks:
        cfg_d["ranks"] = parse_ranks(a.ranks)
    if a.operators:
        cfg_d["operators"] = a.operators
    if a.form:
        cfg_d["form"] = a.form
    cfg = OpInfConfig.from_dict(cfg_d)
    ds = ReducedDataset(a.dataset)
    logger.info("\n%s", ds.summary())

    rom, asm, omap, sweep = train_opinf(ds, cfg)
    save_opinf(out, rom, asm, omap, sweep, cfg, a.dataset)
    fig = out / "figures"
    plot_regularization_sweep(sweep, fig / "regularization_sweep")
    plot_operator_structure(rom, fig / "operator_structure")
    x_mean = np.mean(np.hstack([asm.state(t) for t in ds.split("train")]), axis=1)
    plot_operator_spectra(rom, fig / "operator_spectrum", dt=ds.split("train")[0].dt, x_ref=x_mean)
    if a.rank_sweep and cfg.rank_sweep:
        rows = rank_sweep(ds, cfg)
        save_json(rows, out / "rank_sweep.json")
        plot_rank_sweep(rows, fig / "rank_sweep")
    if not a.no_train_plots:
        evaluate_models([rom], ds, asm, omap, out / "train_fit", splits=("train",), make_field_plots=False)
    logger.info("Done. Model in %s/model, figures in %s", out, fig)


if __name__ == "__main__":
    main()
