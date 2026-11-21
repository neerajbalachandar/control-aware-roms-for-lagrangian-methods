"""
rom.eulerian.bound_vortex - Lifting-surface (bound) vorticity that is *not* in the pfield files.

In VarFlExI the wing is a VLM lattice; its bound vortices are not vortex
particles, so the ``*_pfield.N.h5`` files miss the bound circulation unless the
static particles (``*_staticpfield.N.h5``) were saved.  Without it the
reconstructed field near the wing lacks the circulation that produces lift
(up-wash ahead, down-wash behind the wing).

We rebuild it from ``*_aerodynamic_station_history.csv``: per station i we know
the deformed control point ``x_cp,i`` (at ``eta_cp`` of the chord), the bound
circulation ``Gamma_i`` and the section pitch.  The bound vortex lies at
``eta_bv`` of the chord, i.e. ``x_bv,i = x_cp,i - (eta_cp - eta_bv) c_i t_i`` with
the chord direction ``t_i = (cos th, 0, -sin th)``.  Station segments join the
mid-points between neighbouring bound points; a segment carries Gamma_i along
+y (positive Gamma -> positive lift for V_inf along +x; checked against the
sign of the shed tip-vortex particles).  The trailing legs are already
represented by the shed particles.

This is a lifting-line approximation (n_chord = 1 in all current VarFlExI
runs, so it is in fact exactly the VLM bound-vortex row up to the station
discretisation).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from rom.eulerian.biot_savart import segment_velocity


def bound_vortex_segments(station: Dict[str, np.ndarray], row: int, chord: np.ndarray,
                          eta_cp: float = 0.75, eta_bv: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segments (A, B, Gamma) of the bound vortex for station-history row ``row``."""
    xcp = np.stack([station["x_cp_m"][row], station["y_cp_m"][row], station["z_cp_m"][row]], axis=1)
    gam = np.asarray(station["bound_circulation_m2_s"][row], float)
    th = np.deg2rad(station["section_chord_pitch_deg"][row]) if "section_chord_pitch_deg" in station else np.zeros(len(gam))
    t = np.stack([np.cos(th), np.zeros_like(th), -np.sin(th)], axis=1)
    pb = xcp - (eta_cp - eta_bv) * chord[:, None] * t
    mid = 0.5 * (pb[1:] + pb[:-1])
    first = pb[0] - (mid[0] - pb[0])
    last = pb[-1] + (pb[-1] - mid[-1])
    edges = np.vstack([first, mid, last])
    return edges[:-1], edges[1:], gam


def bound_vortex_velocity(targets: np.ndarray, station: Dict[str, np.ndarray], step: int,
                          span: float, root_chord: float, tip_chord: float,
                          eta_cp: float = 0.75, eta_bv: float = 0.25, core: float = 0.01,
                          chunk: int = 20000) -> Optional[np.ndarray]:
    """Velocity induced at ``targets`` by the bound vortex at ``step`` (None if unavailable)."""
    if station is None or "bound_circulation_m2_s" not in station:
        return None
    steps = station["step"]
    rows = np.nonzero(steps == step)[0]
    if rows.size == 0:
        return None
    row = int(rows[0])
    y = station["y_cp_m"][row]
    chord = root_chord + (tip_chord - root_chord) * np.clip(y / span, 0, 1)
    A, B, g = bound_vortex_segments(station, row, chord, eta_cp, eta_bv)
    out = np.empty((targets.shape[0], 3))
    for i in range(0, targets.shape[0], chunk):
        out[i:i + chunk] = segment_velocity(targets[i:i + chunk], A, B, g, core)
    return out
