#!/usr/bin/env python3
"""
Stage 2 - POD bases + reduced coordinates shared by every ROM method.

    python scripts/build_reduced_dataset.py --config configs/reduced_dataset.yaml \
        --out artifacts/reduced/reduced_dataset.h5

Figures (``<out dir>/figures``):
  pod_spectrum                 singular values + cumulative energy per block
  projection_error_<block>     best achievable error vs number of modes (train / val / test)
  fluid_modes_<comp>_<plane>   leading spatial POD modes (velocity component slices)
  structure_modes              plate POD mode shapes
  signals_overview             inputs and outputs of all runs, coloured by split
  reconstruction_conservation  integral of grid vorticity vs. sum of particle strengths
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rom.cli_utils import load_yaml, setup_logging  # noqa: E402
from rom.reduced.dataset import SPLITS, ReducedDataset, build_reduced_dataset  # noqa: E402
from rom.viz import fields as vf  # noqa: E402
from rom.viz import plots as vp  # noqa: E402
from rom.viz.style import INK, MUTED, SERIES, plt, savefig  # noqa: E402

logger = logging.getLogger("build_reduced_dataset")


def projection_curves(ds: ReducedDataset, block: str):
    out = {}
    for s in SPLITS:
        tr = ds.split(s)
        if not tr:
            continue
        rmax = tr[0].coords[block].shape[0]
        errs = []
        for t in tr:
            a2 = np.cumsum(t.coords[block] ** 2, axis=0)            # (rmax, K)
            e = np.sqrt(np.clip(t.sqnorm[block][None] - a2, 0, None)) / np.sqrt(np.mean(t.sqnorm[block]))
            errs.append(e.mean(1))
        out[s] = {"rank": np.arange(1, rmax + 1), "error": np.mean(errs, 0)}
    return out


def conservation_figure(ds: ReducedDataset, path):
    from rom.data.trajectory_file import TrajectoryFile

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharex=True)
    any_ = False
    for j, t in enumerate(ds.trajectories):
        try:
            d = TrajectoryFile(t.source_file).diagnostics()
            tf = TrajectoryFile(t.source_file)
        except Exception:
            continue
        if "grid_vorticity_integral" not in d:
            continue
        any_ = True
        for c, ax in enumerate(axes):
            ax.plot(tf.time, d["particle_gamma_sum"][:, c], color=INK, lw=1.2, label="sum of particle strengths (all space)" if j == 0 else None)
            ax.plot(tf.time, d["grid_vorticity_integral"][:, c], color=SERIES[0], ls="--", lw=1.2,
                    label="integral of grid vorticity (ROM box)" if j == 0 else None)
            ax.set_title(fr"$\int \omega_{'xyz'[c]}\,dV$", fontsize=9)
            ax.set_xlabel("t [s]")
    if not any_:
        plt.close(fig)
        return None
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Reconstruction check: vorticity captured inside the ROM box vs. total particle circulation", y=1.04)
    fig.tight_layout()
    return savefig(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/reduced_dataset.yaml")
    ap.add_argument("--out", default="artifacts/reduced/reduced_dataset.h5")
    ap.add_argument("--plots-only", action="store_true", help="re-draw figures of an existing dataset")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    setup_logging(a.verbose, out.parent / "build_reduced_dataset.log")
    if a.plots_only:
        ds = ReducedDataset(out)
    else:
        ds = build_reduced_dataset(load_yaml(a.config), out)
    logger.info("\n%s", ds.summary())
    fig = out.parent / "figures"

    vp.plot_pod_spectrum({b: ds.singular_values[b] for b in ds.block_names}, fig / "pod_spectrum")
    for b in ds.block_names:
        if not b.endswith("_rate"):
            vp.plot_projection_error_vs_rank(projection_curves(ds, b), fig / f"projection_error_{b}", b)
    vp.plot_signals_overview(ds.trajectories, ds.input_names, ds.output_names, fig / "signals_overview")
    conservation_figure(ds, fig / "reconstruction_conservation")

    for b in ds.block_names:
        src = str(ds.block_info[b].get("source", ""))
        if src.startswith("field:"):
            grid = ds.block_grid(b)
            basis = ds.basis(b)
            span = 0.96
            try:
                from rom.data.trajectory_file import TrajectoryFile

                span = float(TrajectoryFile(ds.trajectories[0].source_file).parameters.get("solid.span", span))
            except Exception:
                pass
            wing = {"span": span, "chord": 0.24}
            labels = [f"POD mode {i + 1} ({100 * e:.1f}% energy)" for i, e in enumerate(basis.energy()[:6])]
            comps = ("uz", "ux") if src.endswith("velocity") else ("wy", "wx")
            vf.plot_vector_modes(grid, basis.modes, fig / f"{b}_modes_{comps[0]}_xz", component=comps[0], plane="xz",
                                 value=0.5 * span, n=6, labels=labels, wing=wing,
                                 title=f"Leading POD modes of '{b}' ({comps[0]}, side view at mid-span)")
            vf.plot_vector_modes(grid, basis.modes, fig / f"{b}_modes_{comps[1]}_yz", component=comps[1], plane="yz",
                                 value=float(np.mean(grid.bounds["x"])), n=6, labels=labels, wing=wing,
                                 title=f"Leading POD modes of '{b}' ({comps[1]}, cross-flow plane)")
        elif src.startswith("structural:") and not b.endswith("_rate"):
            with h5py.File(ds.path, "r") as h:
                if f"blocks/{b}/mesh_geometry" in h:
                    geo, topo = h[f"blocks/{b}/mesh_geometry"][()], h[f"blocks/{b}/mesh_topology"][()]
                    basis = ds.basis(b)
                    labels = [f"plate mode {i + 1} ({100 * e:.1f}%)" for i, e in enumerate(basis.energy()[:6])]
                    vf.plot_plate_modes(geo, topo.astype(int), basis.modes, fig / f"{b}_modes", labels=labels)
    for b in ds.block_names[:2]:
        tr = ds.trajectories
        vp.plot_coordinates([t.time for t in tr[:3]], [t.coords[b] for t in tr[:3]], [f"{t.name} ({t.split})" for t in tr[:3]],
                            fig / f"coordinates_{b}", n=6, title=f"POD coordinates of '{b}'")
    logger.info("Figures in %s", fig)


if __name__ == "__main__":
    main()
