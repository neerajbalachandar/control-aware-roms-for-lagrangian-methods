"""
rom.varflexi.signals - Inputs u(t) and control-relevant outputs y(t) of a VarFlExI run.

ROMs for control need an explicit input/output structure:

* **inputs u(t)** - everything that is *imposed* on the aeroelastic system:
    - the freestream gust evaluated at the wing reference point
      (``gust_x, gust_y, gust_z`` [m/s]) - a *disturbance* input;
    - any actuation signal you add to VarFlExI (flap, heave/pitch, twist, ...) -
      write it to ``<run>/control_inputs.csv`` with a ``time`` (or ``step``)
      column + one column per channel; it is interpolated to the snapshot steps;
    - optionally constant run parameters (``aoa_deg``, ``vinf``) to make a
      parametric ROM across runs.
* **outputs y(t)** - quantities a controller / reachability analysis cares
  about: lift, drag, pitching moment, tip heave & twist (and rates), wake
  probe velocities.  They are *not* part of the ROM state; a linear output
  map y = C [1; x; u] is learned on top of the reduced state.

Columns that a run does not provide are skipped with a warning, so the same
config works for every run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.varflexi.run_reader import VarFlExIRun

logger = logging.getLogger(__name__)

DEFAULT_INPUTS: List[str] = ["gust_x", "gust_y", "gust_z", "control:*"]
DEFAULT_OUTPUTS: List[str] = [
    "coupling:lift",
    "coupling:drag",
    "coupling:cm",
    "solid:tip_w_displacement",
    "solid:tip_w_velocity",
    "solid:tip_theta_y_displacement",
    "solid:midspan_w_displacement",
    "probe:probe_1_vz_m_s",
    "probe:probe_2_vz_m_s",
    "probe:probe_3_vz_m_s",
]


@dataclass
class RunSignals:
    steps: np.ndarray
    time: np.ndarray
    inputs: np.ndarray                    # (K, m)
    input_names: List[str]
    outputs: np.ndarray                   # (K, p)
    output_names: List[str]
    bound_circulation: Optional[np.ndarray] = None  # (K, n_station)
    station_y: Optional[np.ndarray] = None
    particle_count: Optional[np.ndarray] = None
    fom_step_walltime: Optional[np.ndarray] = None  # fluid+solid wall time per step [s]
    extras: Dict[str, np.ndarray] = field(default_factory=dict)


def _lookup(df, steps: np.ndarray, column: str) -> Optional[np.ndarray]:
    """Values of ``column`` at ``steps`` (NaN where missing; step 0 -> nearest)."""
    if df is None or column not in df.columns or "step" not in df.columns:
        return None
    s = df["step"].to_numpy()
    v = df[column].to_numpy(dtype=np.float64)
    order = np.argsort(s)
    s, v = s[order], v[order]
    out = np.interp(steps, s, v, left=v[0], right=v[-1])
    return out


def _control_csv(run: VarFlExIRun, steps: np.ndarray):
    import pandas as pd

    p = run.root / "control_inputs.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    if "step" in df.columns:
        x_src, x_dst = df["step"].to_numpy(float), steps.astype(float)
    elif "time" in df.columns:
        x_src, x_dst = df["time"].to_numpy(float), run.time_of_step(steps)
    else:
        raise ValueError(f"{p} needs a 'time' or 'step' column")
    out = {}
    for c in df.columns:
        if c in ("time", "step"):
            continue
        out[c] = np.interp(x_dst, x_src, df[c].to_numpy(float))
    return out


def assemble_signals(run: VarFlExIRun, steps: Sequence[int],
                     input_spec: Optional[Sequence[str]] = None,
                     output_spec: Optional[Sequence[str]] = None,
                     gust_point: Optional[np.ndarray] = None) -> RunSignals:
    """Collect inputs / outputs for ``steps``.

    Spec grammar (both lists):

    * ``gust_x|gust_y|gust_z`` - gust component at ``gust_point`` (default: run reference point)
    * ``freestream_x|..._z``   - total ambient velocity at the reference point
    * ``param:<key>``          - constant fluid-config value (e.g. ``param:aoa_deg``)
    * ``control:<col>`` / ``control:*`` - columns of ``<run>/control_inputs.csv``
    * ``coupling:<col>``       - column of ``coupling_history.csv``
    * ``solid:<col>``          - column of ``solid_state_history.csv``
    * ``probe:<col>``          - column of the wake-probe history
    """
    steps = np.asarray(sorted(int(s) for s in steps))
    t = run.time_of_step(steps)
    input_spec = list(input_spec) if input_spec is not None else DEFAULT_INPUTS
    output_spec = list(output_spec) if output_spec is not None else DEFAULT_OUTPUTS
    gp = run.reference_point() if gust_point is None else np.asarray(gust_point, float)

    fs = run.freestream_params()
    a = np.deg2rad(fs["aoa_deg"])
    base = fs["vinf"] * np.array([np.cos(a), 0.0, np.sin(a)])
    amb = np.stack([run.freestream(gp, ti)[0] for ti in t]) if len(t) else np.zeros((0, 3))

    tables = {
        "coupling": run.coupling_history(),
        "solid": run.solid_state_history(),
        "probe": run.wake_probe_history(),
    }
    control = None

    def resolve(token: str) -> Dict[str, np.ndarray]:
        nonlocal control
        if token in ("gust_x", "gust_y", "gust_z"):
            i = "xyz".index(token[-1])
            return {token: amb[:, i] - base[i]}
        if token in ("freestream_x", "freestream_y", "freestream_z"):
            i = "xyz".index(token[-1])
            return {token: amb[:, i]}
        src, _, col = token.partition(":")
        if src == "param":
            val = run.config["fluid"].get(col)
            if val is None:
                return {}
            return {col: np.full(len(steps), float(val))}
        if src == "control":
            if control is None:
                control = _control_csv(run, steps)
            if col == "*":
                return {f"control_{k}": v for k, v in control.items()}
            return {f"control_{col}": control[col]} if col in control else {}
        if src in tables:
            v = _lookup(tables[src], steps, col)
            return {} if v is None else {col: v}
        raise ValueError(f"Unknown signal token '{token}'")

    def build(spec):
        names, cols = [], []
        for tok in spec:
            got = resolve(tok)
            if not got and not tok.endswith("*"):
                logger.warning("Signal '%s' not available in run %s - skipped", tok, run.name)
            for k, v in got.items():
                names.append(k)
                cols.append(v)
        arr = np.stack(cols, axis=1) if cols else np.zeros((len(steps), 0))
        return arr, names

    U, un = build(input_spec)
    Y, yn = build(output_spec)

    # NOTE: all-zero channels (e.g. no gust in a steady run) are kept here so that
    # every trajectory has the same channels; the reduced-dataset builder drops
    # channels that are zero across *all* training trajectories.
    sig = RunSignals(steps=steps, time=t, inputs=U, input_names=un, outputs=Y, output_names=yn)

    st = run.station_history()
    if st is not None and "bound_circulation_m2_s" in st:
        gam = st["bound_circulation_m2_s"]
        idx = np.clip(np.searchsorted(st["step"], steps), 0, len(st["step"]) - 1)
        sig.bound_circulation = gam[idx]
        sig.station_y = np.nanmean(st["y_cp_m"], axis=0) if "y_cp_m" in st else None

    ch = tables["coupling"]
    if ch is not None:
        pc = _lookup(ch, steps, "particle_count")
        sig.particle_count = pc
        wt = [_lookup(ch, steps, c) for c in ("fluid_step_time", "solid_step_time")]
        wt = [w for w in wt if w is not None]
        if wt:
            sig.fom_step_walltime = np.nansum(np.stack(wt), axis=0)
    return sig
