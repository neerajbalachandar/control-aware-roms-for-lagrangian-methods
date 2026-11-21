"""
rom.viz.fields - Slices of 3-D Eulerian fields, POD / Koopman modes, plate mode shapes,
particle clouds.

Quantities (``quantity`` argument):
    'ux','uy','uz'          velocity components [m/s]
    'ux_fluct', ...         minus a reference (e.g. the POD mean) - pass ``ref=``
    'speed'                 |u|
    'wx','wy','wz','wmag'   vorticity (from the field if given, else curl of velocity)
Planes: 'xz' (side view, at y = value), 'xy' (top view, at z = value), 'yz' (cross-flow, at x = value).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rom.eulerian.grid import EulerianGrid
from rom.eulerian.vorticity import compute_vorticity
from rom.viz.style import DIVERGING, INK, INK_2, SEQUENTIAL, plt, savefig

LABELS = {"ux": r"$u_x$ [m/s]", "uy": r"$u_y$ [m/s]", "uz": r"$u_z$ [m/s]", "speed": r"$|u|$ [m/s]",
          "wx": r"$\omega_x$ [1/s]", "wy": r"$\omega_y$ [1/s]", "wz": r"$\omega_z$ [1/s]",
          "wmag": r"$|\omega|$ [1/s]"}
_AX = {"x": 0, "y": 1, "z": 2}


def scalar_field(vel: Optional[np.ndarray], grid: EulerianGrid, quantity: str,
                 vort: Optional[np.ndarray] = None, ref: Optional[np.ndarray] = None) -> np.ndarray:
    q = quantity.replace("_fluct", "")
    if q in ("ux", "uy", "uz", "speed"):
        v = vel - ref if (ref is not None and quantity.endswith("_fluct")) else vel
        return np.linalg.norm(v, axis=-1) if q == "speed" else v[..., "xyz".index(q[1])]
    if q in ("wx", "wy", "wz", "wmag"):
        w = vort if vort is not None else compute_vorticity(vel, grid)
        return np.linalg.norm(w, axis=-1) if q == "wmag" else w[..., "xyz".index(q[1])]
    raise ValueError(f"Unknown quantity '{quantity}'")


def take_slice(f: np.ndarray, grid: EulerianGrid, plane: str, value: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Return (A, B, F2d, label_A, label_B) for imshow-like plotting (F2d indexed [B, A])."""
    if grid.axis_order != "xyz":
        raise ValueError("slicing assumes axis_order='xyz'")
    normal ={"xz": "y", "xy": "z", "yz": "x"}[plane]
    c = grid.coords_1d(normal)
    i = int(np.argmin(np.abs(c - value)))
    sl = [slice(None)] * 3
    sl[_AX[normal]] = i
    F = f[tuple(sl)]
    a, b = plane[0], plane[1]
    return grid.coords_1d(a), grid.coords_1d(b), F.T, a, b


def _pcolor(ax, A, B, F, cmap, vmin, vmax):
    return ax.pcolormesh(A, B, F, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)


def _decorate(ax, la, lb, wing: Optional[dict], plane: str):
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(f"{la} [m]")
    ax.set_ylabel(f"{lb} [m]")
    if wing:
        c = wing.get("chord", 0.24)
        s = wing.get("span", 0.96)
        if plane == "xz":
            ax.plot([0, c], [0, 0], color=INK, lw=2.5, solid_capstyle="butt")
        elif plane == "xy":
            ax.add_patch(plt.Rectangle((0, 0), c, s, fill=False, ec=INK, lw=1.2))
        elif plane == "yz":
            ax.plot([0, s], [0, 0], color=INK, lw=2.5, solid_capstyle="butt")


def plot_field_comparison(grid: EulerianGrid, times: Sequence[float], truth: Sequence[np.ndarray],
                          preds: Dict[str, Sequence[np.ndarray]], path, quantity: str = "wy",
                          plane: str = "xz", value: float = 0.48, wing: Optional[dict] = None,
                          title: str = ""):
    """Rows = times; columns = FOM | each ROM | |error| of each ROM. ``truth``/``preds`` hold
    2-D scalar fields already computed with :func:`scalar_field` and :func:`take_slice` or 3-D."""
    names = list(preds)
    ncol = 1 + 2 * len(names)
    A0, B0, _, _, _ = take_slice(truth[0], grid, plane, value)
    aspect = (B0[-1] - B0[0]) / max(A0[-1] - A0[0], 1e-12)
    panel_w = 3.1
    fig, axes = plt.subplots(len(times), ncol, figsize=(panel_w * ncol + 1.2, len(times) * (panel_w * aspect + 0.55) + 0.6),
                             squeeze=False, layout="constrained")
    signed = quantity not in ("speed", "wmag")
    for i, t in enumerate(times):
        A, B, Ft, la, lb = take_slice(truth[i], grid, plane, value)
        lim = np.nanpercentile(np.abs(Ft), 99.5) or 1.0
        vmin, vmax = (-lim, lim) if signed else (0, lim)
        cmap = DIVERGING if signed else SEQUENTIAL
        im = _pcolor(axes[i, 0], A, B, Ft, cmap, vmin, vmax)
        axes[i, 0].set_title(f"FOM   t = {t:.3f} s" if i == 0 else f"t = {t:.3f} s", fontsize=9)
        elim = 0
        errs = []
        for j, n in enumerate(names):
            _, _, Fp, _, _ = take_slice(preds[n][i], grid, plane, value)
            _pcolor(axes[i, 1 + j], A, B, Fp, cmap, vmin, vmax)
            axes[i, 1 + j].set_title(n if i == 0 else "", fontsize=9)
            errs.append(np.abs(Fp - Ft))
            elim = max(elim, np.nanpercentile(errs[-1], 99.5))
        for j, n in enumerate(names):
            ime = _pcolor(axes[i, 1 + len(names) + j], A, B, errs[j], SEQUENTIAL, 0, elim or 1.0)
            axes[i, 1 + len(names) + j].set_title(f"|error| {n}" if i == 0 else "", fontsize=9)
        for jj, ax in enumerate(axes[i]):
            _decorate(ax, la, lb, wing, plane)
            if jj > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        fig.colorbar(im, ax=axes[i, len(names)], shrink=0.95, pad=0.02, label=LABELS.get(quantity.replace("_fluct", ""), quantity))
        if names:
            fig.colorbar(ime, ax=axes[i, -1], shrink=0.95, pad=0.02, label="|error|")
    for ax in axes[:-1].ravel():
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    normal = {"xz": "y", "xy": "z", "yz": "x"}[plane]
    fig.suptitle((title or f"{LABELS.get(quantity.replace('_fluct', ''), quantity)} on the {plane}-plane")
                 + f"   ({normal} = {value:.3g} m)")
    return savefig(fig, path)


def plot_vector_modes(grid: EulerianGrid, modes: np.ndarray, path, component: str = "uz", plane: str = "xz",
                      value: float = 0.48, n: int = 6, labels: Optional[List[str]] = None,
                      wing: Optional[dict] = None, title: str = ""):
    """Spatial modes (n_features, r) of a vector field block -> slices of one component."""
    n = min(n, modes.shape[1])
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.0 * nrows + 0.6), squeeze=False, layout="constrained")
    ci = "xyz".index(component[-1])
    for k in range(n):
        ax = axes.ravel()[k]
        f = modes[:, k].reshape(grid.shape + (3,))
        if component.startswith("w"):
            f = compute_vorticity(f, grid)
        A, B, F, la, lb = take_slice(f[..., ci], grid, plane, value)
        lim = np.nanmax(np.abs(F)) or 1.0
        _pcolor(ax, A, B, F, DIVERGING, -lim, lim)
        _decorate(ax, la, lb, wing, plane)
        ax.set_title(labels[k] if labels else f"mode {k + 1}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title or f"Spatial modes ({LABELS.get(component, component)}, {plane}-plane)")
    return savefig(fig, path)


def plot_plate_modes(geometry: np.ndarray, topology: np.ndarray, modes: np.ndarray, path, n: int = 6,
                     component: int = 2, labels: Optional[List[str]] = None, title: str = ""):
    """Plate mode shapes (n_nodes*3, r) -> filled contours of the chosen displacement component."""
    import matplotlib.tri as mtri

    n = min(n, modes.shape[1])
    tri = mtri.Triangulation(geometry[:, 0], geometry[:, 1], topology)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    for k in range(n):
        ax = axes.ravel()[k]
        w = modes[:, k].reshape(-1, 3)[:, component]
        lim = np.max(np.abs(w)) or 1.0
        ax.tricontourf(tri, w, levels=21, cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(labels[k] if labels else f"structural mode {k + 1}", fontsize=9)
        ax.set_xlabel("x (chord) [m]")
        ax.set_ylabel("y (span) [m]")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title or "Plate POD modes (out-of-plane displacement w)", y=1.0)
    fig.tight_layout()
    return savefig(fig, path)


def plot_particles_and_field(X: np.ndarray, Gamma: np.ndarray, grid: EulerianGrid, vel: np.ndarray,
                             vort: Optional[np.ndarray], path, y_plane: float = 0.48, t: float = 0.0,
                             wing: Optional[dict] = None, max_points: int = 60000):
    """Reconstruction QA: particle cloud (side view) | |omega| slice | u_z slice."""
    rng = np.random.default_rng(0)
    sel = rng.choice(len(X), size=min(max_points, len(X)), replace=False) if len(X) else np.arange(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    g = np.linalg.norm(Gamma[sel], axis=1)
    ax = axes[0]
    sc = ax.scatter(X[sel, 0], X[sel, 2], c=g, s=0.6, cmap=SEQUENTIAL,
                    vmax=np.percentile(g, 99) if len(g) else 1, rasterized=True)
    b = grid.bounds
    ax.add_patch(plt.Rectangle((b["x"][0], b["z"][0]), b["x"][1] - b["x"][0], b["z"][1] - b["z"][0],
                               fill=False, ec=INK_2, ls="--", lw=1.0))
    ax.set_title(f"vortex particles (N={len(X):,}), t={t:.3f} s", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.grid(False)
    fig.colorbar(sc, ax=ax, label=r"$|\Gamma_p|$", shrink=0.8)
    wm = scalar_field(vel, grid, "wmag", vort=vort)
    A, B, F, la, lb = take_slice(wm, grid, "xz", y_plane)
    im = _pcolor(axes[1], A, B, F, SEQUENTIAL, 0, np.percentile(F, 99.5) or 1)
    _decorate(axes[1], la, lb, wing, "xz")
    axes[1].set_title(fr"$|\omega|$ on grid, y={y_plane:g} m", fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=0.8, label=LABELS["wmag"])
    A, B, F, la, lb = take_slice(vel[..., 2], grid, "xz", y_plane)
    lim = np.percentile(np.abs(F - np.median(F)), 99.5) or 1
    im = _pcolor(axes[2], A, B, F, DIVERGING, np.median(F) - lim, np.median(F) + lim)
    _decorate(axes[2], la, lb, wing, "xz")
    axes[2].set_title(r"$u_z$ on grid (incl. freestream)", fontsize=9)
    fig.colorbar(im, ax=axes[2], shrink=0.8, label=LABELS["uz"])
    fig.tight_layout()
    return savefig(fig, path)
