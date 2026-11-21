"""
rom_opinf.training - Fit input-driven OpInf ROMs on the shared reduced dataset.

Pipeline
--------
1. ``StateAssembler`` -> scaled reduced state x (fluid | structure | structure_rate ...).
2. Left-hand side:
   * continuous: dx/dt by finite differences *within* each trajectory (order 2/4/6);
     for the structural displacement block the exactly-projected structural
     velocity (``structure_rate`` block) replaces the finite difference when
     available (``use_projected_rates``) - the FEniCS solver gives it for free.
   * discrete: (x_k, x_{k+1}) pairs within each trajectory.
3. Grid search over (lambda_1, lambda_2); each candidate is *rolled out* on the
   validation trajectories driven by their recorded inputs; the candidate with
   the lowest mean validation state error among those that stay bounded is kept.
4. Output map y = C [1; x; u] by ridge regression on training data.
"""

from __future__ import annotations

import itertools
import json
import logging
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.reduced.dataset import ReducedDataset
from rom.reduced.evaluation import is_stable
from rom.reduced.state import OutputMap, StateAssembler, block_field_error, save_json
from rom_opinf.derivatives import estimate_trajectory_derivatives, prepare_discrete_pairs
from rom_opinf.model import OpInfROM

logger = logging.getLogger(__name__)


@dataclass
class OpInfConfig:
    ranks: Dict[str, int] = field(default_factory=lambda: {"fluid": 12, "structure": 4, "structure_rate": 4})
    operators: str = "cAHB"
    form: str = "continuous"
    inputs: Optional[List[str]] = None          # None -> all dataset inputs
    ddt_order: int = 4
    use_projected_rates: bool = True
    lambda1: List[float] = field(default_factory=lambda: [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0])
    lambda2: List[float] = field(default_factory=lambda: [1e-6, 1e-4, 1e-2, 1e0, 1e2])
    integrator: str = "RK45"
    selection_split: str = "val"                 # falls back to 'train' if empty
    output_map_regularizer: float = 1e-6
    output_map_quadratic: bool = False
    rank_sweep: Optional[Dict[str, List[int]]] = None   # e.g. {"fluid": [4, 8, 12, 16, 24]}

    @classmethod
    def from_dict(cls, d: dict) -> "OpInfConfig":
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"Unknown OpInf options {sorted(bad)}")
        return cls(**d)


def _lhs(cfg: OpInfConfig, asm: StateAssembler, trajs, Xs):
    times = [t.time for t in trajs]
    if cfg.form == "discrete":
        return None
    ddt_list, _, _ = estimate_trajectory_derivatives(Xs, times, order=cfg.ddt_order)
    sl = asm.slices()
    if cfg.use_projected_rates:
        for b in asm.blocks:
            rb = f"{b}_rate"
            if rb in asm.blocks and asm.ranks[rb] >= asm.ranks[b]:
                for k, t in enumerate(trajs):
                    ddt_list[k][sl[b]] = t.coords[rb][:asm.ranks[b]] / asm.scales[b]
                logger.info("Using projected structural velocity as d/dt of block '%s'", b)
    return ddt_list


def _fit(cfg, asm, trajs, Xs, Us, ddts, lam1, lam2) -> OpInfROM:
    rom = OpInfROM(cfg.operators, cfg.form, name="opinf", integrator=cfg.integrator)
    if cfg.form == "continuous":
        rom.fit(np.hstack(Xs), np.hstack(ddts), np.hstack(Us) if asm.m else None, lam1, lam2)
    else:
        Xk, Xn = prepare_discrete_pairs(Xs)
        Uk = np.hstack([U[:, :-1] for U in Us]) if asm.m else None
        rom.fit(Xk, Xn, Uk, lam1, lam2)
    return rom


def _score(rom: OpInfROM, asm, ds, trajs) -> float:
    errs = []
    for t in trajs:
        X = asm.state(t)
        Xp = rom.simulate(X[:, 0], t.time, asm.inputs(t, ds.input_names))
        if not is_stable(Xp, X):
            return float("inf")
        rms = np.sqrt(np.mean(np.sum(X ** 2, 0))) or 1.0
        errs.append(np.mean(np.linalg.norm(X - Xp, axis=0)) / rms)
    return float(np.mean(errs))


def train_opinf(ds: ReducedDataset, cfg: OpInfConfig, ranks: Optional[Dict[str, int]] = None):
    """Return (rom, assembler, output_map, sweep_record)."""
    ranks = dict(ranks or cfg.ranks)
    for b in list(ranks):
        if b not in ds.block_names:
            logger.warning("Block '%s' not in dataset (has %s) - ignored", b, ds.block_names)
            ranks.pop(b)
        elif ranks[b] > ds.block_info[b].get("r_max", ranks[b]):
            raise ValueError(f"rank {ranks[b]} > r_max of block '{b}'")
    asm = StateAssembler.fit(ds, ranks, input_names=cfg.inputs)
    train = ds.split("train")
    sel = ds.split(cfg.selection_split) or train
    Xs = [asm.state(t) for t in train]
    Us = [asm.inputs(t, ds.input_names) for t in train]
    if ("B" in cfg.operators or "N" in cfg.operators) and asm.m == 0:
        logger.warning("No inputs in dataset - dropping B/N from operators '%s'", cfg.operators)
        cfg = OpInfConfig(**{**asdict(cfg), "operators": cfg.operators.replace("B", "").replace("N", "")})
    ddts = _lhs(cfg, asm, train, Xs)

    grid = np.full((len(cfg.lambda1), len(cfg.lambda2)), np.inf)
    best, best_score = None, np.inf
    t0 = _time.perf_counter()
    for (i, l1), (j, l2) in itertools.product(enumerate(cfg.lambda1), enumerate(cfg.lambda2)):
        try:
            rom = _fit(cfg, asm, train, Xs, Us, ddts, l1, l2)
            s = _score(rom, asm, ds, sel)
        except Exception as exc:
            logger.debug("lambda=(%g,%g) failed: %s", l1, l2, exc)
            s = np.inf
        grid[i, j] = s
        logger.info("  lambda1=%8.1e lambda2=%8.1e  -> %s error %.4g", l1, l2, cfg.selection_split, s)
        if s < best_score:
            best, best_score = rom, s
    if best is None:
        logger.warning("Every regularisation candidate blew up on %s; keeping the largest regularisation. "
                       "Consider smaller ranks, more data, or the discrete form.", cfg.selection_split)
        best = _fit(cfg, asm, train, Xs, Us, ddts, cfg.lambda1[-1], cfg.lambda2[-1])
    omap = OutputMap(names=list(ds.output_names), regularizer=cfg.output_map_regularizer,
                     quadratic=cfg.output_map_quadratic)
    if ds.output_names:
        omap.fit(Xs, Us, [t.outputs for t in train])
    else:
        omap = None
    sweep = {"lambda1": cfg.lambda1, "lambda2": cfg.lambda2, "score": grid.tolist(),
             "selection_split": cfg.selection_split, "best": list(best.regularizer), "best_score": best_score,
             "ranks": asm.ranks, "seconds": _time.perf_counter() - t0}
    logger.info("Selected lambda=(%g, %g) with %s error %.4g", *best.regularizer, cfg.selection_split, best_score)
    return best, asm, omap, sweep


def rank_sweep(ds: ReducedDataset, cfg: OpInfConfig) -> Dict:
    """Validation error vs rank of one block (others fixed) - shows where adding modes stops helping."""
    out = {}
    for block, values in (cfg.rank_sweep or {}).items():
        rows = []
        for r in values:
            ranks = dict(cfg.ranks)
            ranks[block] = r
            if f"{block}_rate" in ranks:
                ranks[f"{block}_rate"] = r
            rom, asm, _, sweep = train_opinf(ds, cfg, ranks)
            sel = ds.split(cfg.selection_split) or ds.split("train")
            fe, pe = [], []
            for t in sel:
                X = asm.state(t)
                Xp = rom.simulate(X[:, 0], t.time, asm.inputs(t, ds.input_names))
                e = block_field_error(t, block, asm.unscale(np.nan_to_num(Xp))[block])
                fe.append(np.mean(e["relative_error"]) if is_stable(Xp, X) else np.inf)
                pe.append(np.mean(e["relative_projection"]))
            rows.append({"rank": r, "rollout_error": float(np.mean(fe)), "projection_error": float(np.mean(pe)),
                         "lambda": list(rom.regularizer)})
        out[block] = rows
    return out


def save_opinf(out_dir, rom: OpInfROM, asm: StateAssembler, omap: Optional[OutputMap], sweep: dict,
               cfg: OpInfConfig, dataset_path) -> Path:
    out_dir = Path(out_dir)
    rom.save(out_dir / "model")
    save_json(asm.to_dict(), out_dir / "model" / "assembler.json")
    if omap is not None:
        save_json(omap.to_dict(), out_dir / "model" / "output_map.json")
    save_json(sweep, out_dir / "regularization_sweep.json")
    save_json({"config": asdict(cfg), "dataset": str(Path(dataset_path).resolve())}, out_dir / "model" / "train_config.json")
    return out_dir


def load_opinf(out_dir):
    out_dir = Path(out_dir)
    rom = OpInfROM.load(out_dir / "model")
    asm = StateAssembler.from_dict(json.loads((out_dir / "model" / "assembler.json").read_text()))
    p = out_dir / "model" / "output_map.json"
    omap = OutputMap.from_dict(json.loads(p.read_text())) if p.is_file() else None
    meta = json.loads((out_dir / "model" / "train_config.json").read_text())
    return rom, asm, omap, meta
