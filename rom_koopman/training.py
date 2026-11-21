"""
rom_koopman.training - Fit Koopman ROMs (EDMDc and/or lifting) on the shared reduced dataset.

Both variants use the *same* block-scaled POD state and inputs as OpInf. For each
variant the ridge parameter is chosen by validation rollout error (driven by the
recorded inputs only), exactly like OpInf's regularisation selection.
"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import is_stable
from rom.reduced.state import OutputMap, StateAssembler, save_json
from rom_koopman.dictionaries import n_monomials
from rom_koopman.edmdc import EDMDcROM
from rom_koopman.lifting import LiftingKoopmanROM

logger = logging.getLogger(__name__)


@dataclass
class KoopmanConfig:
    ranks: Dict[str, int] = field(default_factory=lambda: {"fluid": 10, "structure": 3, "structure_rate": 3})
    inputs: Optional[List[str]] = None
    methods: List[str] = field(default_factory=lambda: ["edmdc", "lifting"])
    edmdc: dict = field(default_factory=lambda: {"degree": 2, "n_rbf": 0, "rbf_kind": "gaussian", "delays": 0,
                                                 "bilinear": True, "rollout": "linear",
                                                 "ridge": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]})
    lifting: dict = field(default_factory=lambda: {"deg_F": 2, "deg_basis": 2, "ridge": [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
                                                   "max_dictionary": 3000, "integrator": "RK45"})
    selection_split: str = "val"
    output_map_regularizer: float = 1e-6
    output_map_quadratic: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "KoopmanConfig":
        bad = set(d) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"Unknown Koopman options {sorted(bad)}")
        base = cls()
        for k in ("edmdc", "lifting"):
            if k in d:
                d[k] = {**getattr(base, k), **d[k]}
        return cls(**{**asdict(base), **d})


def _score(rom, asm, ds, trajs) -> float:
    errs = []
    for t in trajs:
        X = asm.state(t)
        Xp = rom.simulate(X[:, 0], t.time, asm.inputs(t, ds.input_names))
        if not is_stable(Xp, X):
            return float("inf")
        rms = np.sqrt(np.mean(np.sum(X ** 2, 0))) or 1.0
        errs.append(np.mean(np.linalg.norm(X - Xp, axis=0)) / rms)
    return float(np.mean(errs))


def _uniform_dt(trajs) -> float:
    dts = [np.diff(t.time) for t in trajs]
    dt = float(np.median(np.concatenate(dts)))
    if any(np.max(np.abs(d - dt)) > 1e-6 * dt + 1e-12 for d in dts):
        raise ValueError("Koopman models need uniformly sampled trajectories with the same dt "
                         "(use the same 'stride' for every run in the reduced dataset).")
    return dt


def train_koopman(ds: ReducedDataset, cfg: KoopmanConfig):
    """Return ({name: rom}, assembler, output_map, sweeps)."""
    ranks = {b: r for b, r in cfg.ranks.items() if b in ds.block_names}
    asm = StateAssembler.fit(ds, ranks, input_names=cfg.inputs)
    train = ds.split("train")
    sel = ds.split(cfg.selection_split) or train
    dt = _uniform_dt(train + sel)
    Xs = [asm.state(t) for t in train]
    Us = [asm.inputs(t, ds.input_names) for t in train]
    models, sweeps = {}, {}

    if "edmdc" in cfg.methods:
        p = dict(cfg.edmdc)
        ridges = p.pop("ridge")
        ridges = ridges if isinstance(ridges, list) else [ridges]
        rows, best, best_s = [], None, np.inf
        for lam in ridges:
            t0 = _time.perf_counter()
            rom = EDMDcROM(ridge=lam, name="Koopman EDMDc", **p).fit(Xs, Us, dt)
            s = _score(rom, asm, ds, sel)
            rows.append({"ridge": lam, "score": s, "seconds": _time.perf_counter() - t0})
            logger.info("  EDMDc ridge=%8.1e  n_z=%d  -> %s error %.4g", lam, rom.n_z, cfg.selection_split, s)
            if s < best_s:
                best, best_s = rom, s
        if best is None:
            logger.warning("All EDMDc candidates unstable; keeping the most regularised one")
            best = EDMDcROM(ridge=ridges[-1], name="Koopman EDMDc", **p).fit(Xs, Us, dt)
        models["edmdc"] = best
        sweeps["edmdc"] = {"rows": rows, "best_ridge": best.ridge, "best_score": best_s, "n_z": best.n_z}

    if "lifting" in cfg.methods:
        p = dict(cfg.lifting)
        ridges = p.pop("ridge")
        ridges = ridges if isinstance(ridges, list) else [ridges]
        maxd = p.pop("max_dictionary", 3000)
        nvar = asm.r + asm.m
        nd = n_monomials(nvar, p.get("deg_basis") or p["deg_F"])
        if nd > maxd:
            logger.warning("Lifting method skipped: %d variables -> %d monomials > max_dictionary=%d. "
                           "Reduce ranks / degree or raise max_dictionary.", nvar, nd, maxd)
        else:
            rows, best, best_s = [], None, np.inf
            for lam in ridges:
                t0 = _time.perf_counter()
                try:
                    rom = LiftingKoopmanROM(ridge=lam, name="Koopman lifting", **p).fit(Xs, Us, dt)
                    s = _score(rom, asm, ds, sel)
                except Exception as exc:
                    logger.warning("lifting ridge=%g failed: %s", lam, exc)
                    rom, s = None, np.inf
                rows.append({"ridge": lam, "score": s, "seconds": _time.perf_counter() - t0})
                logger.info("  lifting ridge=%8.1e  dictionary=%d  -> %s error %.4g", lam, nd, cfg.selection_split, s)
                if rom is not None and s < best_s:
                    best, best_s = rom, s
            if best is None:
                logger.warning("All lifting candidates unstable or failed; keeping the most regularised one")
                best = LiftingKoopmanROM(ridge=ridges[-1], name="Koopman lifting", **p).fit(Xs, Us, dt)
            models["lifting"] = best
            sweeps["lifting"] = {"rows": rows, "best_ridge": best.ridge, "best_score": best_s, "dictionary": nd}

    omap = None
    if ds.output_names:
        omap = OutputMap(names=list(ds.output_names), regularizer=cfg.output_map_regularizer,
                         quadratic=cfg.output_map_quadratic).fit(Xs, Us, [t.outputs for t in train])
    return models, asm, omap, sweeps


def save_koopman(out_dir, models, asm, omap, sweeps, cfg, dataset_path) -> Path:
    out = Path(out_dir) / "model"
    for k, m in models.items():
        m.save(out / k)
    save_json(asm.to_dict(), out / "assembler.json")
    if omap is not None:
        save_json(omap.to_dict(), out / "output_map.json")
    save_json(sweeps, Path(out_dir) / "ridge_sweep.json")
    save_json({"config": asdict(cfg), "dataset": str(Path(dataset_path).resolve()), "methods": list(models)},
              out / "train_config.json")
    return Path(out_dir)


def load_koopman(out_dir):
    out = Path(out_dir) / "model"
    meta = json.loads((out / "train_config.json").read_text())
    models = {}
    for k in meta["methods"]:
        models[k] = EDMDcROM.load(out / k) if k == "edmdc" else LiftingKoopmanROM.load(out / k)
    asm = StateAssembler.from_dict(json.loads((out / "assembler.json").read_text()))
    p = out / "output_map.json"
    omap = OutputMap.from_dict(json.loads(p.read_text())) if p.is_file() else None
    return models, asm, omap, meta
