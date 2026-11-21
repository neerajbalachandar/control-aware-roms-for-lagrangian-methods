"""
rom.eulerian.julia_bridge - Use FLOWUnsteady's FMM grid evaluation instead of the Python backends.

VarFlExI ships ``fluid/postprocess_particle_fields.jl`` which calls
``uns.computefluiddomain(pfield, [grid])`` (ExaFMM) and writes GeometricTools
grids ``<run>_fluiddomain.<step>.{h5,xmf}`` with datasets ``nodes, U, W, Uinf,
Uinduced``.  This is the fastest option for very large particle counts.

To make it ROM-compatible **run it with fixed bounds** (never ``--bounds-file``),
e.g. for the default near-wake grid (cells = nodes - 1)::

    julia fluid/postprocess_particle_fields.jl --input <run>/fluid \
        --output <run>/fluid_domain_rom --steps 1:1:600 \
        --nx 96 --ny 48 --nz 28 --bounds -0.12,2.28,-0.12,1.08,-0.30,0.42

then convert with ``scripts/convert_julia_fluiddomain.py`` (which calls
:func:`convert_julia_fluiddomain`). Note that the Julia output contains the
static (VLM) particles only if ``*_staticpfield`` files were saved; otherwise
add the bound vortex in Python with ``--bound-vortex``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from rom.eulerian.grid import EulerianGrid

logger = logging.getLogger(__name__)
_FD_RE = re.compile(r"^(?P<prefix>.+)_fluiddomain\.(?P<step>\d+)\.h5$")


def read_gt_grid(path: Union[str, Path]) -> Tuple[EulerianGrid, np.ndarray, Dict[str, np.ndarray]]:
    """Read a GeometricTools grid file; return (grid, permutation, fields reordered to grid.flat order)."""
    with h5py.File(path, "r") as h:
        nodes = np.asarray(h["nodes"][()], float)
        if nodes.shape[0] == 3 and nodes.shape[1] != 3:
            nodes = nodes.T
        fields = {}
        for k in ("U", "W", "Uinf", "Uinduced"):
            if k in h:
                a = np.asarray(h[k][()], float)
                fields[k] = a.T if (a.shape[0] == 3 and a.shape[1] != 3) else a
    coords = [np.unique(np.round(nodes[:, i], 12)) for i in range(3)]
    if np.prod([len(c) for c in coords]) != nodes.shape[0]:
        raise ValueError(f"{path}: nodes do not form a rectilinear grid")
    grid = EulerianGrid(bounds={ax: (float(c[0]), float(c[-1])) for ax, c in zip("xyz", coords)},
                        resolution={ax: len(c) for ax, c in zip("xyz", coords)},
                        custom_coords={ax: c for ax, c in zip("xyz", coords)})
    ijk = [np.searchsorted(coords[i], np.round(nodes[:, i], 12)) for i in range(3)]
    lin = (ijk[0] * len(coords[1]) + ijk[1]) * len(coords[2]) + ijk[2]
    perm = np.empty_like(lin)
    perm[lin] = np.arange(len(lin))
    return grid, perm, {k: v[perm] for k, v in fields.items()}


def convert_julia_fluiddomain(run, fd_dir: Union[str, Path], out_path: Union[str, Path],
                              steps: Optional[Sequence[int]] = None, input_spec=None, output_spec=None,
                              add_bound_vortex: bool = False, bound_vortex_core: Optional[float] = None) -> Path:
    """Convert a directory of Julia ``*_fluiddomain.N.h5`` files into a ROM trajectory file."""
    from rom.data.trajectory_file import TrajectoryWriter
    from rom.eulerian.bound_vortex import bound_vortex_velocity
    from rom.eulerian.reconstruct import cell_volumes, structure_on_steps
    from rom.varflexi.signals import assemble_signals

    fd_dir = Path(fd_dir)
    files = {}
    for p in fd_dir.iterdir():
        m = _FD_RE.match(p.name)
        if m:
            files[int(m.group("step"))] = p
    if not files:
        raise FileNotFoundError(f"No *_fluiddomain.N.h5 in {fd_dir}")
    steps = sorted(files) if steps is None else sorted(int(s) for s in steps)
    grid, _, _ = read_gt_grid(files[steps[0]])
    vol = cell_volumes(grid)
    station = run.station_history() if add_bound_vortex else None
    pts = grid.flat_points()
    with TrajectoryWriter(out_path, grid, steps, run.time_of_step(np.asarray(steps)),
                          fields=("velocity", "vorticity"), resume=False,
                          attrs={"source_run": str(run.root), "run_name": run.name, "dt": run.dt,
                                 "reconstruction": '{"backend": "julia_fmm"}'}) as w:
        w.write_signals(assemble_signals(run, steps, input_spec, output_spec))
        st = structure_on_steps(run, np.asarray(steps))
        if st is not None:
            w.write_structure(st["displacement"], st["velocity"], st["geometry"], st["topology"])
        w.write_parameters(run.config)
        for k, s in enumerate(steps):
            g, _, f = read_gt_grid(files[s])
            if g.shape != grid.shape or not all(np.allclose(g.coords_1d(a), grid.coords_1d(a)) for a in "xyz"):
                raise ValueError(f"Grid of step {s} differs from step {steps[0]} - re-run the Julia "
                                 "post-processing with fixed --bounds (no --bounds-file).")
            U = f["U"]
            if add_bound_vortex and station is not None:
                gw = run.wing_geometry()
                Ub = bound_vortex_velocity(pts, station, s, gw["span"], gw["root_chord"], gw["tip_chord"],
                                           gw["eta_cp"], gw["eta_bv"], core=bound_vortex_core or 0.02)
                if Ub is not None:
                    U = U + Ub
            W = f.get("W", np.zeros_like(U)).reshape(grid.shape + (3,))
            w.write_snapshot(k, {"velocity": U.reshape(grid.shape + (3,)), "vorticity": W},
                             {"grid_vorticity_integral": np.einsum("ijk,ijkc->c", vol, W)})
    return Path(out_path)
