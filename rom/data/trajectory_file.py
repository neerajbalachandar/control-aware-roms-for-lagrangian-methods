"""
rom.data.trajectory_file - Streaming writer / lazy reader for ROM trajectory files.

One HDF5 file per VarFlExI run (= one trajectory)::

    /metadata/   time (K,), step (K,), coords_x/y/z, grid attrs, completed (K,) bool
                 attrs: source_run, run_name, dt, reconstruction (json), created
    /snapshots/  velocity  (K, Nx, Ny, Nz, 3) float32   total velocity incl. freestream
                 vorticity (K, Nx, Ny, Nz, 3) float32   (optional)
                 velocity_induced (K, ...)               (optional, particles + bound vortex only)
    /structural/ displacement (K, Nn, 3), velocity (K, Nn, 3), mesh_geometry (Nn, 2),
                 mesh_topology (Ne, 3)
    /signals/    inputs (K, m)  attrs names ; outputs (K, p) attrs names ;
                 bound_circulation (K, n_station) ; station_y ; particle_count (K,) ;
                 fom_step_walltime (K,)
    /diagnostics/ particle_gamma_sum (K,3), grid_vorticity_integral (K,3), n_particles (K,),
                  eval_seconds (K,)
    /parameters/ attrs: flattened run configuration (fluid.*, solid.*, coupling.*)

Snapshots are chunked one-per-time-step so ROM stages can stream them.
"""

from __future__ import annotations

import json
import time as _time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from rom.eulerian.grid import EulerianGrid

FIELD_NAMES = ("velocity", "vorticity", "velocity_induced")


def _flatten_config(cfg: dict, prefix: str = "") -> Dict[str, object]:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_config(v, key + "."))
        elif isinstance(v, (int, float, str, bool, np.number)):
            out[key] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v) and len(v) > 0:
            out[key] = np.asarray(v, dtype=float)
        elif v is None:
            out[key] = "None"
    return out


class TrajectoryWriter:
    """Create (or resume) a trajectory file and fill snapshots one at a time."""

    def __init__(self, path: Union[str, Path], grid: EulerianGrid, steps: Sequence[int],
                 times: Sequence[float], fields: Sequence[str] = ("velocity", "vorticity"),
                 dtype=np.float32, resume: bool = True, attrs: Optional[dict] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.steps = np.asarray(steps, dtype=np.int64)
        self.times = np.asarray(times, dtype=np.float64)
        self.fields = tuple(fields)
        K = len(self.steps)
        shape = (K,) + grid.shape + (3,)
        exists = self.path.is_file()
        if exists and resume:
            self.h = h5py.File(self.path, "a")
            old = self.h["metadata/step"][()]
            if not np.array_equal(old, self.steps):
                self.h.close()
                raise ValueError(f"{self.path} exists with different steps; use --overwrite")
            for f in self.fields:
                if f"snapshots/{f}" not in self.h:
                    self.h.close()
                    raise ValueError(f"{self.path} exists without field '{f}'; use --overwrite")
        else:
            self.h = h5py.File(self.path, "w")
            meta = self.h.create_group("metadata")
            grid.write_metadata(meta)
            meta.create_dataset("time", data=self.times)
            meta.create_dataset("step", data=self.steps)
            meta.create_dataset("completed", data=np.zeros(K, dtype=bool))
            meta.attrs["created"] = _time.strftime("%Y-%m-%d %H:%M:%S")
            for k, v in (attrs or {}).items():
                meta.attrs[k] = v if not isinstance(v, dict) else json.dumps(v)
            snaps = self.h.create_group("snapshots")
            for f in self.fields:
                snaps.create_dataset(f, shape=shape, dtype=dtype, chunks=(1,) + grid.shape + (3,),
                                     compression="gzip", compression_opts=3, shuffle=True)
            diag = self.h.create_group("diagnostics")
            diag.create_dataset("particle_gamma_sum", data=np.full((K, 3), np.nan))
            diag.create_dataset("grid_vorticity_integral", data=np.full((K, 3), np.nan))
            diag.create_dataset("n_particles", data=np.zeros(K, dtype=np.int64))
            diag.create_dataset("eval_seconds", data=np.full(K, np.nan))
        self.h.flush()

    # ------------------------------------------------------------------
    @property
    def completed(self) -> np.ndarray:
        return self.h["metadata/completed"][()]

    def write_snapshot(self, k: int, fields: Dict[str, np.ndarray], diagnostics: Optional[dict] = None):
        for f in self.fields:
            if f in fields and fields[f] is not None:
                self.h[f"snapshots/{f}"][k] = fields[f].astype(self.h[f"snapshots/{f}"].dtype)
        for key, val in (diagnostics or {}).items():
            if f"diagnostics/{key}" in self.h:
                self.h[f"diagnostics/{key}"][k] = val
        self.h["metadata/completed"][k] = True
        self.h.flush()

    def _replace(self, name: str, data, attrs: Optional[dict] = None):
        if name in self.h:
            del self.h[name]
        ds = self.h.create_dataset(name, data=data)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v
        return ds

    def write_signals(self, signals) -> None:
        """Write a :class:`rom.varflexi.signals.RunSignals`."""
        g = "signals"
        self._replace(f"{g}/inputs", signals.inputs, {"names": json.dumps(signals.input_names)})
        self._replace(f"{g}/outputs", signals.outputs, {"names": json.dumps(signals.output_names)})
        for key in ("bound_circulation", "station_y", "particle_count", "fom_step_walltime"):
            val = getattr(signals, key)
            if val is not None:
                self._replace(f"{g}/{key}", val)

    def write_structure(self, displacement: np.ndarray, velocity: Optional[np.ndarray],
                        geometry: Optional[np.ndarray], topology: Optional[np.ndarray]) -> None:
        self._replace("structural/displacement", displacement)
        if velocity is not None:
            self._replace("structural/velocity", velocity)
        if geometry is not None:
            self._replace("structural/mesh_geometry", geometry)
        if topology is not None:
            self._replace("structural/mesh_topology", topology)

    def write_parameters(self, config: dict) -> None:
        if "parameters" in self.h:
            del self.h["parameters"]
        grp = self.h.create_group("parameters")
        for k, v in _flatten_config(config).items():
            try:
                grp.attrs[k] = v
            except (TypeError, ValueError):
                grp.attrs[k] = str(v)

    def close(self):
        self.h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TrajectoryFile:
    """Lazy, read-only access to a trajectory file (nothing large is loaded eagerly)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        with h5py.File(self.path, "r") as h:
            meta = h["metadata"]
            self.grid = EulerianGrid.from_hdf5(meta)
            self.time = meta["time"][()]
            self.steps = meta["step"][()] if "step" in meta else np.arange(len(self.time))
            self.completed = meta["completed"][()] if "completed" in meta else np.ones(len(self.time), bool)
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in meta.attrs.items()}
            self.fields = [f for f in FIELD_NAMES if f"snapshots/{f}" in h]
            self.input_names: List[str] = json.loads(h["signals/inputs"].attrs["names"]) if "signals/inputs" in h else []
            self.output_names: List[str] = json.loads(h["signals/outputs"].attrs["names"]) if "signals/outputs" in h else []
            self.parameters = dict(h["parameters"].attrs) if "parameters" in h else {}
            self.has_structure = "structural/displacement" in h
        self.name = str(self.attrs.get("run_name", self.path.stem))

    def __len__(self) -> int:
        return len(self.time)

    def _read(self, key: str, idx=slice(None)):
        with h5py.File(self.path, "r") as h:
            if key not in h:
                return None
            return h[key][idx]

    # signals ---------------------------------------------------------------
    @property
    def inputs(self) -> np.ndarray:
        v = self._read("signals/inputs")
        return np.zeros((len(self), 0)) if v is None else np.asarray(v, float)

    @property
    def outputs(self) -> np.ndarray:
        v = self._read("signals/outputs")
        return np.zeros((len(self), 0)) if v is None else np.asarray(v, float)

    def signal(self, name: str) -> Optional[np.ndarray]:
        return self._read(f"signals/{name}")

    def diagnostics(self) -> Dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as h:
            return {k: h[f"diagnostics/{k}"][()] for k in h["diagnostics"]} if "diagnostics" in h else {}

    def structure(self, key: str = "displacement", idx=slice(None)) -> Optional[np.ndarray]:
        return self._read(f"structural/{key}", idx)

    # fields ----------------------------------------------------------------
    def read_field(self, name: str, idx=slice(None)) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(f"Field '{name}' not in {self.path} (has {self.fields})")
        with h5py.File(self.path, "r") as h:
            ds = h[f"snapshots/{name}"]
            if isinstance(idx, (list, np.ndarray)):
                idx = np.asarray(idx)
                order = np.argsort(idx)
                data = ds[np.sort(idx)]
                out = np.empty_like(data)
                out[order] = data
                return out
            return ds[idx]

    def iter_field(self, name: str, indices: Sequence[int], block: int = 16) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(indices_block, data_block)`` with data shape (b, Nx, Ny, Nz, 3)."""
        indices = np.asarray(indices)
        with h5py.File(self.path, "r") as h:
            ds = h[f"snapshots/{name}"]
            for i in range(0, len(indices), block):
                ib = indices[i:i + block]
                if np.all(np.diff(ib) == 1):
                    yield ib, ds[ib[0]:ib[-1] + 1]
                else:
                    yield ib, ds[np.sort(ib)]

    def summary(self) -> dict:
        return {"path": str(self.path), "name": self.name, "K": len(self), "grid_shape": self.grid.shape,
                "fields": self.fields, "inputs": self.input_names, "outputs": self.output_names,
                "completed": int(np.sum(self.completed)), "t0": float(self.time[0]), "t1": float(self.time[-1])}


def list_trajectory_files(paths: Sequence[Union[str, Path]]) -> List[Path]:
    """Expand files / directories / globs into a sorted list of ``.h5`` trajectory files."""
    import glob

    out: List[Path] = []
    for p in paths:
        p = str(p)
        if any(ch in p for ch in "*?["):
            out += [Path(x) for x in glob.glob(p)]
        elif Path(p).is_dir():
            out += sorted(Path(p).glob("*.h5"))
        else:
            out.append(Path(p))
    return sorted(set(out))
