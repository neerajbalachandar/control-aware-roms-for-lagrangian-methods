"""
rom.testing.synthetic - Tiny VarFlExI-*like* run folders for tests and the end-to-end demo.

This is NOT a physical simulation. It produces the same files, column names and
conventions as a real VarFlExI run, from a cheap model:

* quasi-steady lifting line with elliptic circulation, effective angle of attack
  alpha + (w_gust - dh/dt)/V, where w_gust is the configured freestream gust;
* a 1-DOF nonlinear (Duffing) plate-bending oscillator driven by the lift
  (geometric non-linearity), mapped to a quadratic span-wise deflection shape;
* trailing (-dGamma/dy) and shed (-dGamma/dt) vortex particles released at the
  trailing edge and convected with the local freestream (no self-induction).

It is only meant to exercise every stage of the pipeline quickly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def make_synthetic_run(root, name: str = "synthetic", n_steps: int = 80, total_time: float = 1.0,
                       gust_amplitude: float = 1.0, gust_frequency: float = 3.0, gust_phase_deg: float = 0.0,
                       vinf: float = 15.0, aoa_deg: float = 5.0, n_span: int = 10, span: float = 0.96,
                       chord: float = 0.24, sigma: Optional[float] = None, particles_per_step: int = 3,
                       lambda_vpm: float = 2.125, control: Optional[np.ndarray] = None, seed: int = 0) -> Path:
    root = Path(root) / name
    for sub in ("fluid", "solid", "coupling", "config"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    dt = total_time / n_steps
    rho = 1.225
    pps = max(1, int(particles_per_step))
    # VarFlExI: sigma_vpm_overwrite = lambda_vpm * |V_inf| dt / p_per_step (overlapping cores)
    sigma = float(sigma) if sigma is not None else lambda_vpm * vinf * dt / pps
    a0 = np.deg2rad(aoa_deg)
    fluid_cfg = {"run_name": name, "particles_per_step": particles_per_step, "lambda_vpm": lambda_vpm, "vinf": vinf, "aoa_deg": aoa_deg, "rho": rho, "n_span": n_span, "eta_cp": 0.75,
                 "eta_bv": 0.25, "sigma_surface": 0.0025, "freestream_gradient": [0.0] * 9,
                 "freestream_oscillation": [0.0, 0.0, gust_amplitude], "freestream_wavenumber": [0.0, 0.0, 0.0],
                 "freestream_frequency_hz": gust_frequency, "freestream_phase_deg": gust_phase_deg,
                 "freestream_reference": [0.0, 0.0, 0.0]}
    solid_cfg = {"span": span, "root_chord": chord, "tip_chord": chord, "nx": 5, "ny": 9}
    coupling_cfg = {"total_time": total_time, "n_steps": n_steps}
    (root / "run_configuration.json").write_text(json.dumps({"config_source": "synthetic", "configs": {
        "fluid_params.yaml": fluid_cfg, "solid_params.yaml": solid_cfg, "coupling_params.yaml": coupling_cfg}}))

    yb = np.linspace(0, span, n_span + 1)
    yc = 0.5 * (yb[1:] + yb[:-1])
    dy = np.diff(yb)
    shape0 = np.sqrt(np.clip(1 - ((yc - span / 2) / (span / 2)) ** 2, 0, None))
    G0 = 0.5 * np.pi * chord * vinf * a0 * shape0           # thin-airfoil scale circulation
    wn, zeta, mass = 2 * np.pi * 6.0, 0.04, 0.5             # plate bending oscillator (tip heave)
    k3 = 4.0 * wn ** 2 / 0.05 ** 2                          # hardening cubic stiffness
    h, hd = 0.0, 0.0
    ctrl = np.zeros(n_steps + 1) if control is None else np.asarray(control, float)

    X = np.zeros((0, 3))
    G = np.zeros((0, 3))
    S = np.zeros(0)
    G_prev = np.zeros(n_span)
    rows_c, rows_s, rows_st, rows_p = [], [], [], []
    nx, ny = solid_cfg["nx"], solid_cfg["ny"]
    gx, gy = np.meshgrid(np.linspace(0, chord, nx), np.linspace(0, span, ny), indexing="xy")
    geo = np.stack([gx.ravel(), gy.ravel()], 1)
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p = j * nx + i
            tris += [[p, p + 1, p + nx], [p + 1, p + nx + 1, p + nx]]
    topo = np.asarray(tris)
    disp = []

    def freestream(x, t):
        ph = 2 * np.pi * gust_frequency * t + np.deg2rad(gust_phase_deg)
        return vinf * np.array([np.cos(a0), 0.0, np.sin(a0)]) + np.array([0.0, 0.0, gust_amplitude]) * np.sin(ph)

    def write_pfield(step, t):
        fn = root / "fluid" / f"{name}_pfield.{step}.h5"
        Xw, Gw, Sw = (X, G, S) if len(S) else (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1))
        with h5py.File(fn, "w") as f:
            f["X"], f["Gamma"], f["sigma"] = Xw, Gw, Sw
            f["np"], f["nt"], f["t"] = len(Sw), step, step
            for k in ("velocity", "vorticity"):
                f[k] = np.zeros_like(Xw)
            f["circulation"], f["vol"], f["static"] = np.ones(len(Sw)), np.zeros(len(Sw)), np.zeros(len(Sw))
        fn.with_suffix(".xmf").write_text(f'<Xdmf><Domain><Grid><Time Value="{t}" /></Grid></Domain></Xdmf>')

    write_pfield(0, 0.0)
    for k in range(1, n_steps + 1):
        t = k * dt
        w_g = freestream(np.zeros(3), t)[2] - vinf * np.sin(a0)
        a_eff = a0 + (w_g - hd) / vinf + 0.05 * ctrl[k]
        Gam = G0 * a_eff / a0 * (1 + 0.15 * (h / 0.05) * yc / span)
        lift = rho * vinf * float(np.sum(Gam * dy))
        # structure: semi-implicit Euler of m h'' + c h' + k h + k3 h^3 = 0.02 * lift
        acc = (0.02 * lift - 2 * zeta * wn * mass * hd - mass * wn ** 2 * h - mass * k3 * h ** 3) / mass
        hd += dt * acc
        h += dt * hd
        w_shape = h * (gy.ravel() / span) ** 2
        twist = 0.1 * h * (gy.ravel() / span)
        d = np.zeros((len(geo), 3))
        d[:, 2] = w_shape + twist * (gx.ravel() - chord / 4)
        disp.append(d)
        z_te = h * (yb / span) ** 2
        # shed particles at the trailing edge: p_per_step sub-rows per step (as VarFlExI),
        # spread over the distance convected during the step
        dG_dy = np.diff(np.concatenate([[0.0], Gam, [0.0]]))            # trailing strength at panel edges
        dGt = -(Gam - G_prev) * dy                                         # shed (span-wise) strength
        X = X + dt * freestream(None, t)
        newX, newG = [], []
        for q in range(pps):
            off = (q + 0.5) / pps * vinf * dt
            newX.append(np.stack([np.full(n_span + 1, chord + off), yb, z_te], 1))
            newG.append(np.stack([dG_dy * vinf * dt / pps, np.zeros(n_span + 1), np.zeros(n_span + 1)], 1))
            newX.append(np.stack([np.full(n_span, chord + off), yc, 0.5 * (z_te[1:] + z_te[:-1])], 1))
            newG.append(np.stack([np.zeros(n_span), dGt / pps, np.zeros(n_span)], 1))
        X = np.vstack([X] + newX)
        G = np.vstack([G] + newG)
        S = np.concatenate([S, np.full(sum(len(b) for b in newX), sigma)])
        keep = np.any(G != 0, axis=1)
        X, G, S = X[keep], G[keep], S[keep]
        G_prev = Gam
        write_pfield(k, t)
        cl = lift / (0.5 * rho * vinf ** 2 * span * chord)
        rows_c.append([k, t, lift, 0.02 * lift * a_eff, cl, 0.02 * cl, -0.1 * cl, len(S), 0.4, 0.2])
        for i in range(n_span):
            rows_st.append([k, t - dt, i + 1, 0.75 * chord, yc[i], h * (yc[i] / span) ** 2, Gam[i],
                            rho * vinf * Gam[i] * dy[i], np.rad2deg(0.1 * h * yc[i] / span)])
        rows_s.append([k, t, h, hd, acc, 0.1 * h, 0.25 * h, 0.5 * lift * dt])
        rows_p.append([k, t - dt] + list(freestream(None, t) + np.array([0, 0, -0.3 * Gam.mean()])))

    import csv

    def wcsv(path, header, rows):
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(header)
            wr.writerows(rows)

    wcsv(root / "coupling" / "coupling_history.csv",
         ["step", "time", "lift", "drag", "cl", "cd", "cm", "particle_count", "fluid_step_time", "solid_step_time"], rows_c)
    wcsv(root / "fluid" / f"{name}_aerodynamic_station_history.csv",
         ["step", "time_s", "station", "x_cp_m", "y_cp_m", "z_cp_m", "bound_circulation_m2_s", "lift_N",
          "section_chord_pitch_deg"], rows_st)
    wcsv(root / "solid" / "solid_state_history.csv",
         ["step", "time_s", "tip_w_displacement", "tip_w_velocity", "tip_w_acceleration", "tip_theta_y_displacement",
          "midspan_w_displacement", "E_total_J"], rows_s)
    wcsv(root / "fluid" / f"{name}_wake_probe_history.csv",
         ["step", "time_s", "probe_1_vx_m_s", "probe_1_vy_m_s", "probe_1_vz_m_s"], rows_p)
    with h5py.File(root / "solid" / "elastodynamics-results.h5", "w") as f:
        f["Mesh/0/mesh/geometry"] = geo
        f["Mesh/0/mesh/topology"] = topo
        for i, d in enumerate(disp):
            f[f"VisualisationVector/{i}"] = d
    (root / "solid" / "elastodynamics-results.xdmf").write_text(
        "<Xdmf>" + "".join(f'<Time Value="{(i + 1) * dt}" />' for i in range(len(disp))) + "</Xdmf>")
    if control is not None:
        wcsv(root / "control_inputs.csv", ["step", "flap"], [[k, ctrl[k]] for k in range(n_steps + 1)])
    return root
