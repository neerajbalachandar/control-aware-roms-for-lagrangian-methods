"""
rom.reduced.dataset - Build / load the *reduced dataset* shared by every ROM method.

Building (``scripts/build_reduced_dataset.py``) takes the trajectory files written
by the Eulerian reconstruction and

1. splits them into train / val / test (by file, never by time inside one run),
2. computes one POD basis per *block* from the TRAINING snapshots only
   (fluid velocity/vorticity field, plate displacement, bound circulation ...),
3. projects every trajectory on those bases and stores coordinates, squared
   snapshot norms (for exact full-field errors), inputs, outputs and timing.

OpInf and Koopman both read this file, so they are compared on *identical*
bases, scalings and splits.

File layout (``reduced_dataset.h5``)::

    attrs: config (json), blocks (json), input_names, output_names, dt
    /blocks/<b>/  mean, modes, singular_values, weights   attrs: source, feature_shape, crop, ...
    /blocks/<b>/grid/  (field blocks) Eulerian grid of the (cropped) field
    /trajectories/<name>/  time, step, inputs, outputs, coords/<b>, sqnorm/<b>
                           attrs: split, source_file, fom_walltime_total
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from rom.data.trajectory_file import TrajectoryFile, list_trajectory_files
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.reconstruct import cell_volumes
from rom.reduced.streaming_pod import PODResult, SnapshotSet, streaming_pod

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Block definitions
# ---------------------------------------------------------------------------

@dataclass
class BlockSpec:
    name: str
    source: str                      # field:<velocity|vorticity> | structural:displacement | signals:bound_circulation
    r_max: int = 40
    weighting: str = "auto"          # auto | volume | area | none
    crop: Optional[Dict[str, Sequence[float]]] = None   # field sub-box {x:[a,b], ...}
    rate: bool = False               # structural: also project structural:velocity (-> '<name>_rate')
    center: bool = True

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "BlockSpec":
        return cls(name=name, **d)


def _crop_slices(grid: EulerianGrid, crop) -> Tuple[Tuple[slice, slice, slice], EulerianGrid]:
    if not crop:
        return (slice(None),) * 3, grid
    sl, coords = [], {}
    for ax in grid.axis_order:
        c = grid.coords_1d(ax)
        if ax in crop and crop[ax] is not None:
            lo, hi = crop[ax]
            i0 = int(np.searchsorted(c, lo - 1e-12))
            i1 = int(np.searchsorted(c, hi + 1e-12, side="right"))
        else:
            i0, i1 = 0, len(c)
        sl.append(slice(i0, i1))
        coords[ax] = c[i0:i1]
    sub = EulerianGrid(bounds={a: (float(coords[a][0]), float(coords[a][-1])) for a in "xyz"},
                       resolution={a: len(coords[a]) for a in "xyz"}, axis_order=grid.axis_order,
                       custom_coords=coords)
    return tuple(sl), sub


def _plate_nodal_area(geometry: np.ndarray, topology: np.ndarray) -> np.ndarray:
    p = geometry[topology]                       # (Ne, 3, 2)
    a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                     - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.zeros(geometry.shape[0])
    for j in range(3):
        np.add.at(w, topology[:, j], a / 3.0)
    return w


class _BlockIO:
    """Feature extraction (snapshot -> column vector) for one block of one trajectory file."""

    def __init__(self, spec: BlockSpec, tf: TrajectoryFile):
        self.spec = spec
        self.tf = tf
        kind, _, what = spec.source.partition(":")
        self.kind, self.what = kind, what
        if kind == "field":
            self.slices, self.grid = _crop_slices(tf.grid, spec.crop)
        else:
            self.slices, self.grid = None, None

    def feature_shape(self) -> Tuple[int, ...]:
        if self.kind == "field":
            return self.grid.shape + (3,)
        if self.kind == "structural":
            return tuple(self.tf.structure(self.what, slice(0, 1)).shape[1:])
        if self.kind == "signals":
            return tuple(self.tf.signal(self.what).shape[1:])
        raise ValueError(f"Unknown block source '{self.spec.source}'")

    def weights(self) -> np.ndarray:
        mode = self.spec.weighting
        shp = self.feature_shape()
        if mode == "none":
            return np.ones(int(np.prod(shp)))
        if self.kind == "field" and mode in ("auto", "volume"):
            return np.repeat(cell_volumes(self.grid).ravel(), 3)
        if self.kind == "structural" and mode in ("auto", "area"):
            geo, topo = self.tf.structure("mesh_geometry"), self.tf.structure("mesh_topology")
            if geo is not None and topo is not None:
                return np.repeat(_plate_nodal_area(geo, topo.astype(int)), shp[-1])
            return np.ones(int(np.prod(shp)))
        if self.kind == "signals" and mode in ("auto", "area"):
            y = self.tf.signal("station_y")
            if y is not None and len(y) == shp[0]:
                d = np.gradient(np.asarray(y, float))
                return np.abs(d)
        return np.ones(int(np.prod(shp)))

    def reader(self, source_override: Optional[str] = None):
        what = source_override or self.what

        def read(indices):
            indices = np.asarray(indices)
            if self.kind == "field":
                for _, blk in self.tf.iter_field(what, indices, block=len(indices)):
                    blk = blk[(slice(None),) + self.slices]
                    yield blk.reshape(blk.shape[0], -1).T.astype(np.float64)
            elif self.kind == "structural":
                d = self.tf.structure(what)[indices]
                yield d.reshape(d.shape[0], -1).T.astype(np.float64)
            else:
                d = np.asarray(self.tf.signal(what))[indices]
                yield d.reshape(d.shape[0], -1).T.astype(np.float64)
        return read


# ---------------------------------------------------------------------------
# Reduced dataset containers
# ---------------------------------------------------------------------------

@dataclass
class ReducedTrajectory:
    name: str
    split: str
    time: np.ndarray
    step: np.ndarray
    inputs: np.ndarray                   # (K, m)
    outputs: np.ndarray                  # (K, p)
    coords: Dict[str, np.ndarray]        # block -> (r_max, K)
    sqnorm: Dict[str, np.ndarray]        # block -> (K,)  ||q - mean||_W^2
    source_file: str = ""
    fom_walltime_total: float = float("nan")
    index: Optional[np.ndarray] = None   # indices into the trajectory file

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time))) if len(self.time) > 1 else float("nan")


class ReducedDataset:
    """Loaded ``reduced_dataset.h5`` (bases are loaded lazily)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        with h5py.File(self.path, "r") as h:
            self.config = json.loads(h.attrs["config"])
            self.block_names: List[str] = json.loads(h.attrs["blocks"])
            self.input_names: List[str] = json.loads(h.attrs["input_names"])
            self.output_names: List[str] = json.loads(h.attrs["output_names"])
            self.block_info = {b: {k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v)
                                   for k, v in h[f"blocks/{b}"].attrs.items()} for b in self.block_names}
            self.singular_values = {b: h[f"blocks/{b}/singular_values"][()] for b in self.block_names}
            self.trajectories: List[ReducedTrajectory] = []
            for name in h["trajectories"]:
                g = h[f"trajectories/{name}"]
                self.trajectories.append(ReducedTrajectory(
                    name=name, split=str(g.attrs["split"]), time=g["time"][()], step=g["step"][()],
                    inputs=g["inputs"][()], outputs=g["outputs"][()],
                    coords={b: g[f"coords/{b}"][()] for b in g["coords"]},
                    sqnorm={b: g[f"sqnorm/{b}"][()] for b in g["sqnorm"]},
                    source_file=str(g.attrs.get("source_file", "")),
                    fom_walltime_total=float(g.attrs.get("fom_walltime_total", np.nan)),
                    index=g["index"][()] if "index" in g else None))
        self._bases: Dict[str, PODResult] = {}

    # ------------------------------------------------------------------
    def split(self, name: str) -> List[ReducedTrajectory]:
        return [t for t in self.trajectories if t.split == name]

    def get(self, name: str) -> ReducedTrajectory:
        for t in self.trajectories:
            if t.name == name:
                return t
        raise KeyError(name)

    def basis(self, block: str) -> PODResult:
        if block.endswith("_rate") and block not in self.block_names:
            block = block[:-5]
        base = self.block_info.get(block, {}).get("basis_of", block)
        if base not in self._bases:
            with h5py.File(self.path, "r") as h:
                g = h[f"blocks/{base}"]
                self._bases[base] = PODResult(mean=g["mean"][()], modes=g["modes"][()].astype(np.float64),
                                              singular_values=g["singular_values"][()], weights=g["weights"][()])
        return self._bases[base]

    def block_grid(self, block: str) -> Optional[EulerianGrid]:
        base = self.block_info.get(block, {}).get("basis_of", block)
        with h5py.File(self.path, "r") as h:
            if f"blocks/{base}/grid" in h:
                return EulerianGrid.from_hdf5(h[f"blocks/{base}/grid"])
        return None

    def summary(self) -> str:
        lines = [f"Reduced dataset {self.path}",
                 f"  blocks : " + ", ".join(f"{b} (r_max={self.block_info[b].get('r_max')})" for b in self.block_names),
                 f"  inputs : {self.input_names}", f"  outputs: {self.output_names}"]
        for s in SPLITS:
            tr = self.split(s)
            lines.append(f"  {s:5s}: {len(tr)} trajectories, {sum(len(t.time) for t in tr)} snapshots "
                         f"({', '.join(t.name for t in tr)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _select_indices(tf: TrajectoryFile, t_start, t_end, stride) -> np.ndarray:
    ok = tf.completed.copy()
    if t_start is not None:
        ok &= tf.time >= t_start - 1e-12
    if t_end is not None:
        ok &= tf.time <= t_end + 1e-12
    idx = np.nonzero(ok)[0]
    idx = idx[::max(1, int(stride))]
    if len(idx) and not np.all(tf.completed[idx[0]:idx[-1] + 1]):
        logger.warning("%s has incomplete snapshots inside the selected window", tf.path.name)
    return idx


def build_reduced_dataset(config: dict, out_path: Union[str, Path]) -> ReducedDataset:
    """Build ``reduced_dataset.h5`` from a config dict (see ``configs/reduced_dataset.yaml``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tw = config.get("time_window", {}) or {}
    t_start, t_end = tw.get("t_start"), tw.get("t_end")
    stride = int(config.get("stride", 1))

    files: Dict[str, List[Path]] = {s: list_trajectory_files(config["trajectories"].get(s, []) or []) for s in SPLITS}
    if not files["train"]:
        raise ValueError("No training trajectories given")
    tfs = {s: [TrajectoryFile(p) for p in files[s]] for s in SPLITS}
    names = [tf.name for s in SPLITS for tf in tfs[s]]
    if len(set(names)) != len(names):
        # disambiguate by file stem
        for s in SPLITS:
            for tf in tfs[s]:
                tf.name = tf.path.stem
    sel = {id(tf): _select_indices(tf, t_start, t_end, stride) for s in SPLITS for tf in tfs[s]}

    # ---- signals: harmonise channel names over all trajectories -----------
    all_tf = [tf for s in SPLITS for tf in tfs[s]]

    def harmonise(attr: str, wanted) -> List[str]:
        union: List[str] = []
        for tf in all_tf:
            for n in getattr(tf, attr):
                if n not in union:
                    union.append(n)
        if wanted not in (None, "all"):
            union = [n for n in wanted if n in union]
        return union

    in_names = harmonise("input_names", config.get("inputs", "all"))
    out_names = harmonise("output_names", config.get("outputs", "all"))

    def signals(tf: TrajectoryFile, idx):
        def take(arr, have, want):
            out = np.zeros((len(idx), len(want)))
            for j, n in enumerate(want):
                if n in have:
                    out[:, j] = arr[idx, have.index(n)]
            return out
        return take(tf.inputs, tf.input_names, in_names), take(tf.outputs, tf.output_names, out_names)

    if config.get("drop_zero_inputs", True):
        keep = []
        for j, n in enumerate(in_names):
            if any(np.any(signals(tf, sel[id(tf)])[0][:, j] != 0) for tf in tfs["train"]):
                keep.append(n)
            else:
                logger.info("Input '%s' is identically zero over training data - dropped", n)
        in_names = keep

    # ---- POD per block -------------------------------------------------------
    specs = [BlockSpec.from_dict(k, v) for k, v in config["blocks"].items()]
    pods: Dict[str, PODResult] = {}
    ios_ref: Dict[str, _BlockIO] = {}
    for spec in specs:
        ios = [_BlockIO(spec, tf) for tf in tfs["train"]]
        try:
            ios[0].feature_shape()
        except Exception as exc:  # missing data in the files
            logger.warning("Block '%s' (%s) unavailable: %s - skipped", spec.name, spec.source, exc)
            continue
        ss = SnapshotSet(readers=[io.reader() for io in ios], indices=[sel[id(io.tf)] for io in ios])
        logger.info("POD of block '%s' (%s) on %d training snapshots", spec.name, spec.source, ss.n_snapshots)
        pods[spec.name] = streaming_pod(ss, ios[0].weights(), spec.r_max, block=int(config.get("block_size", 32)),
                                        center=spec.center, memory_limit_gb=float(config.get("memory_limit_gb", 8)))
        ios_ref[spec.name] = ios[0]
    specs = [s for s in specs if s.name in pods]
    block_names = [s.name for s in specs] + [f"{s.name}_rate" for s in specs if s.rate and s.source.startswith("structural")]

    # ---- write ----------------------------------------------------------------
    with h5py.File(out_path, "w") as h:
        h.attrs["config"] = json.dumps(config, default=str)
        h.attrs["blocks"] = json.dumps(block_names)
        h.attrs["input_names"] = json.dumps(in_names)
        h.attrs["output_names"] = json.dumps(out_names)
        for spec in specs:
            pod = pods[spec.name]
            g = h.create_group(f"blocks/{spec.name}")
            g.create_dataset("mean", data=pod.mean)
            g.create_dataset("modes", data=pod.modes.astype(np.float32), compression="gzip", compression_opts=3)
            g.create_dataset("singular_values", data=pod.singular_values)
            g.create_dataset("weights", data=pod.weights)
            g.attrs["source"] = spec.source
            g.attrs["r_max"] = pod.r_max
            g.attrs["feature_shape"] = json.dumps(list(ios_ref[spec.name].feature_shape()))
            g.attrs["crop"] = json.dumps(spec.crop)
            io = ios_ref[spec.name]
            if io.grid is not None:
                io.grid.write_metadata(g.create_group("grid"))
            if io.kind == "structural":
                geo = io.tf.structure("mesh_geometry")
                topo = io.tf.structure("mesh_topology")
                if geo is not None:
                    g.create_dataset("mesh_geometry", data=geo)
                    g.create_dataset("mesh_topology", data=topo)
            if spec.rate and spec.source.startswith("structural"):
                gr = h.create_group(f"blocks/{spec.name}_rate")
                gr.attrs["basis_of"] = spec.name
                gr.attrs["source"] = "structural:velocity"
                gr.attrs["r_max"] = pod.r_max
                gr.create_dataset("singular_values", data=pod.singular_values)

        for s in SPLITS:
            for tf in tfs[s]:
                idx = sel[id(tf)]
                if len(idx) < 3:
                    logger.warning("Trajectory %s has <3 usable snapshots - skipped", tf.name)
                    continue
                g = h.create_group(f"trajectories/{tf.name}")
                g.attrs["split"] = s
                g.attrs["source_file"] = str(tf.path.resolve())
                g.create_dataset("time", data=tf.time[idx])
                g.create_dataset("step", data=tf.steps[idx])
                g.create_dataset("index", data=idx)
                U, Y = signals(tf, idx)
                g.create_dataset("inputs", data=U)
                g.create_dataset("outputs", data=Y)
                wt = tf.signal("fom_step_walltime")
                if wt is not None:
                    # wall time to simulate the selected window (all FOM steps between first and last snapshot)
                    st = tf.steps
                    m = (st > tf.steps[idx[0]]) & (st <= tf.steps[idx[-1]])
                    per_step = np.nanmedian(np.asarray(wt)[m]) if m.any() else np.nan
                    g.attrs["fom_walltime_total"] = float(per_step * (tf.steps[idx[-1]] - tf.steps[idx[0]]))
                for spec in specs:
                    io = _BlockIO(spec, tf)
                    pod = pods[spec.name]
                    A, N = [], []
                    for i in range(0, len(idx), 64):
                        for X in io.reader()(idx[i:i + 64]):
                            a, nrm = pod.project(X)
                            A.append(a)
                            N.append(nrm)
                    g.create_dataset(f"coords/{spec.name}", data=np.concatenate(A, 1))
                    g.create_dataset(f"sqnorm/{spec.name}", data=np.concatenate(N))
                    if spec.rate and spec.source.startswith("structural"):
                        A, N = [], []
                        for i in range(0, len(idx), 64):
                            for X in io.reader("velocity")(idx[i:i + 64]):
                                WX = X * pod.weights[:, None]
                                A.append(pod.modes.T @ WX)
                                N.append(np.einsum("ij,ij->j", X, WX))
                        g.create_dataset(f"coords/{spec.name}_rate", data=np.concatenate(A, 1))
                        g.create_dataset(f"sqnorm/{spec.name}_rate", data=np.concatenate(N))
    logger.info("Reduced dataset written to %s", out_path)
    return ReducedDataset(out_path)
