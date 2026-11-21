"""
rom.viz.plots - Shared line / spectrum / summary figures (data, bases, ROM predictions).

All functions take plain arrays + a path and return the PNG path.  Every method
(OpInf, Koopman, ...) uses the same functions so results are visually comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.viz.style import (INK, INK_2, MUTED, SERIES, SPLIT_COLORS, method_color, method_style, plt,
                           savefig)


def _grid_axes(n: int, ncols: int = 3, w: float = 3.2, h: float = 2.0, sharex=True):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.7), sharex=sharex, squeeze=False,
                             layout="constrained")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    # the lowest *visible* panel of every column keeps its x tick labels
    for c in range(ncols):
        rows = [r for r in range(nrows) if r * ncols + c < n]
        if rows:
            axes[rows[-1], c].tick_params(labelbottom=True)
    return fig, axes.ravel()[:n]


def _bottom_axes(axes, ncols: int = 3):
    n = len(axes)
    return [axes[i] for i in range(n) if i + ncols >= n]


def log_ticks(ax):
    """Readable log axis when the data span < ~2 decades (labels at 1, 2, 5 x 10^k)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    lo, hi = ax.get_ylim()
    if lo > 0 and hi / lo < 300:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())


def _legend_once(fig, axes, ncol=None):
    handles, labels = [], []
    for ax in axes:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=ncol or len(labels), frameon=False)


# ---------------------------------------------------------------------------
# Data & basis
# ---------------------------------------------------------------------------

def plot_pod_spectrum(singular_values: Dict[str, np.ndarray], path, ranks: Optional[Dict[str, int]] = None,
                      n_show: int = 80):
    """Per block: normalised singular values and *unresolved* energy 1 - E(r), both on log axes."""
    blocks = [b for b in singular_values if not b.endswith("_rate")]
    fig, axes = plt.subplots(2, len(blocks), figsize=(3.6 * len(blocks), 5.0), squeeze=False, layout="constrained")
    for j, b in enumerate(blocks):
        s = np.asarray(singular_values[b])
        s = s[s > 0]
        k = np.arange(1, min(n_show, len(s)) + 1)
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        ax = axes[0, j]
        ax.semilogy(k, s[:len(k)] / s[0], "o-", color=SERIES[0], ms=3, lw=1.2)
        ax.set_title(b)
        ax.set_ylabel(r"$\sigma_i / \sigma_1$" if j == 0 else "")
        ax = axes[1, j]
        resid = np.clip(1 - e[:len(k)], 1e-16, None)
        ax.semilogy(k, resid, "o-", color=SERIES[0], ms=3, lw=1.2)
        txt = []
        for lvl in (0.9, 0.99, 0.999, 0.9999):
            r = int(np.searchsorted(e, lvl) + 1)
            ax.axhline(1 - lvl, color=MUTED, lw=0.7, ls=":")
            txt.append(f"{100 * lvl:g}% energy: r = {r}")
        ax.text(0.97, 0.97, "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e4e3df"))
        if ranks and b in ranks:
            for a_ in axes[:, j]:
                a_.axvline(ranks[b], color=SERIES[1], lw=1.2, ls="--")
        ax.set_xlabel("number of modes r")
        ax.set_ylabel("unresolved energy  1 - E(r)" if j == 0 else "")
    fig.suptitle("POD spectra (training snapshots)")
    return savefig(fig, path)


def plot_projection_error_vs_rank(curves: Dict[str, Dict[str, np.ndarray]], path, block: str):
    """curves[split] = {'rank': r, 'error': mean relative projection error}."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for split, c in curves.items():
        ax.semilogy(c["rank"], c["error"], "-", color=SPLIT_COLORS.get(split, INK), label=split)
    ax.set_xlabel("number of POD modes r")
    ax.set_ylabel("relative projection error")
    ax.set_title(f"Best-possible error with r modes - {block}")
    ax.legend()
    return savefig(fig, path)


def plot_signals_overview(trajs: Sequence, input_names: List[str], output_names: List[str], path,
                          max_outputs: int = 6):
    """Inputs and outputs of every trajectory, coloured by split."""
    names = [("in", n) for n in input_names] + [("out", n) for n in output_names[:max_outputs]]
    if not names:
        return None
    fig, axes = _grid_axes(len(names), ncols=3, w=3.3, h=1.9)
    for ax, (kind, n) in zip(axes, names):
        for t in trajs:
            arr = t.inputs if kind == "in" else t.outputs
            names_ = input_names if kind == "in" else output_names
            ax.plot(t.time, arr[:, names_.index(n)], color=SPLIT_COLORS.get(t.split, INK), lw=1.0,
                    alpha=0.85, label=t.split)
        ax.set_title(("input: " if kind == "in" else "output: ") + n, fontsize=9)
    for ax in _bottom_axes(axes):
        ax.set_xlabel("t [s]")
    _legend_once(fig, axes)
    return savefig(fig, path)


def plot_coordinates(t_list, X_list, labels, path, n: int = 6, title: str = "", colors=None):
    """First n reduced coordinates of several trajectories."""
    n = min(n, X_list[0].shape[0])
    fig, axes = _grid_axes(n, ncols=3, w=3.2, h=1.8)
    for i, ax in enumerate(axes):
        for j, (t, X, lb) in enumerate(zip(t_list, X_list, labels)):
            ax.plot(t, X[i], color=(colors[j] if colors else SERIES[j % 3]), lw=1.0, label=lb)
        ax.set_title(f"$x_{{{i + 1}}}$")
    for ax in _bottom_axes(axes):
        ax.set_xlabel("t [s]")
    _legend_once(fig, axes)
    fig.suptitle(title)
    return savefig(fig, path)


# ---------------------------------------------------------------------------
# ROM predictions
# ---------------------------------------------------------------------------

def plot_rollout_coordinates(t, X_true, preds: Dict[str, np.ndarray], path, n: int = 6,
                             labels: Optional[List[str]] = None, title: str = ""):
    n = min(n, X_true.shape[0])
    fig, axes = _grid_axes(n, ncols=3, w=3.2, h=1.8)
    for i, ax in enumerate(axes):
        ax.plot(t, X_true[i], color=INK, lw=1.6, label="FOM (projected)")
        for j, (name, X) in enumerate(preds.items()):
            ax.plot(t, X[i], color=method_color(name, j), ls=method_style(name), lw=1.4, label=name)
        ax.set_title(labels[i] if labels else f"$x_{{{i + 1}}}$")
    for ax in _bottom_axes(axes):
        ax.set_xlabel("t [s]")
    _legend_once(fig, axes)
    fig.suptitle(title)
    return savefig(fig, path)


def plot_error_vs_time(t, errors: Dict[str, np.ndarray], projection: Optional[np.ndarray], path,
                       title: str = "", ylabel: str = "relative field error", t_train_end: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    if projection is not None:
        ax.semilogy(t, np.maximum(projection, 1e-12), color=MUTED, ls=":", lw=1.4, label="POD projection (floor)")
    for j, (name, e) in enumerate(errors.items()):
        ax.semilogy(t, np.maximum(e, 1e-12), color=method_color(name, j), ls=method_style(name), lw=1.5, label=name)
    if t_train_end is not None:
        ax.axvline(t_train_end, color=INK_2, lw=0.8)
        ax.annotate("end of training window", (t_train_end, ax.get_ylim()[1]), rotation=90, va="top", ha="right",
                    fontsize=7, color=INK_2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    log_ticks(ax)
    return savefig(fig, path)


def plot_outputs(t, Y_true, preds: Dict[str, np.ndarray], names: List[str], path, title: str = "",
                 units: Optional[Dict[str, str]] = None, max_n: int = 9):
    idx = list(range(min(max_n, len(names))))
    fig, axes = _grid_axes(len(idx), ncols=3, w=3.3, h=2.0)
    for ax, i in zip(axes, idx):
        ax.plot(t, Y_true[:, i], color=INK, lw=1.6, label="FOM")
        for j, (name, Y) in enumerate(preds.items()):
            ax.plot(t, Y[:, i], color=method_color(name, j), ls=method_style(name), lw=1.4, label=name)
        u = (units or {}).get(names[i], "")
        ax.set_title(names[i] + (f" [{u}]" if u else ""), fontsize=9)
    for ax in _bottom_axes(axes):
        ax.set_xlabel("t [s]")
    _legend_once(fig, axes)
    fig.suptitle(title)
    return savefig(fig, path)


def plot_output_psd(t, Y_true, preds: Dict[str, np.ndarray], names: List[str], path, title: str = "",
                    max_n: int = 6):
    from scipy.signal import welch

    fs = 1.0 / float(np.median(np.diff(t)))
    idx = list(range(min(max_n, len(names))))
    fig, axes = _grid_axes(len(idx), ncols=3, w=3.3, h=2.1, sharex=False)
    nper = min(256, len(t))
    for ax, i in zip(axes, idx):
        f, p = welch(Y_true[:, i] - Y_true[:, i].mean(), fs=fs, nperseg=nper)
        ax.loglog(f[1:], p[1:], color=INK, lw=1.6, label="FOM")
        for j, (name, Y) in enumerate(preds.items()):
            if np.all(np.isfinite(Y[:, i])):
                f, p = welch(Y[:, i] - Y[:, i].mean(), fs=fs, nperseg=nper)
                ax.loglog(f[1:], p[1:], color=method_color(name, j), ls=method_style(name), lw=1.3, label=name)
        ax.set_title(names[i], fontsize=9)
        ax.set_xlabel("f [Hz]")
    _legend_once(fig, axes)
    fig.suptitle(title or "Power spectral density of outputs")
    return savefig(fig, path)


def plot_error_summary(table: Dict[str, Dict[str, float]], path, metric_label: str, title: str = "",
                       log: bool = True):
    """Grouped horizontal bars: table[trajectory][method] = value."""
    trajs = list(table)
    methods = []
    for tr in trajs:
        for m in table[tr]:
            if m not in methods:
                methods.append(m)
    finite = [v for tr in trajs for v in table[tr].values() if np.isfinite(v) and v > 0]
    log = log and bool(finite)
    fig, ax = plt.subplots(figsize=(5.8, 0.45 * len(trajs) * max(1, len(methods)) + 1.2))
    hgt = 0.8 / len(methods)
    for j, m in enumerate(methods):
        vals = [table[tr].get(m, np.nan) for tr in trajs]
        vals = [v if np.isfinite(v) else np.nan for v in vals]
        y = np.arange(len(trajs)) + (j - (len(methods) - 1) / 2) * hgt
        ax.barh(y, vals, height=hgt * 0.9, color=method_color(m, j), label=m)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2g}", (v, yi), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=7, color=INK_2)
            else:
                ax.annotate("unstable / n.a.", (0, yi), xycoords=("axes fraction", "data"), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(np.arange(len(trajs)))
    ax.set_yticklabels(trajs)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(metric_label)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_horizon_error(horizons, errs: Dict[str, np.ndarray], path, dt: float, title: str = ""):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for j, (name, e) in enumerate(errs.items()):
        ax.semilogy(np.asarray(horizons) * dt, e, marker="o", ms=3, color=method_color(name, j),
                    ls=method_style(name), label=name)
    ax.set_xlabel("prediction horizon [s]")
    ax.set_ylabel("mean relative state error")
    log_ticks(ax)
    ax.set_title(title or "Error growth with prediction horizon")
    ax.legend()
    return savefig(fig, path)


def plot_continuous_spectrum(eigs: Dict[str, np.ndarray], path, title: str = "", weights=None):
    """Continuous-time eigenvalues: growth rate Re(lambda) vs frequency Im(lambda)/2pi."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.axhline(0, color=INK_2, lw=0.8)
    for j, (name, ev) in enumerate(eigs.items()):
        ev = np.asarray(ev)
        s = 18 if weights is None else 10 + 60 * weights[name] / max(weights[name].max(), 1e-12)
        ax.scatter(ev.imag / (2 * np.pi), ev.real, s=s, facecolor="none", edgecolor=method_color(name, j),
                   lw=1.2, label=name)
    ax.set_xlabel(r"frequency  Im$(\lambda)/2\pi$  [Hz]")
    ax.set_ylabel(r"growth rate  Re$(\lambda)$  [1/s]")
    ax.set_title(title or "Continuous-time spectrum")
    ax.legend()
    return savefig(fig, path)


def plot_discrete_spectrum(eigs: Dict[str, np.ndarray], path, title: str = ""):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=0.8)
    for j, (name, ev) in enumerate(eigs.items()):
        ax.scatter(np.real(ev), np.imag(ev), s=16, facecolor="none", edgecolor=method_color(name, j),
                   lw=1.1, label=name)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re$(\mu)$")
    ax.set_ylabel(r"Im$(\mu)$")
    ax.set_title(title or "Discrete-time spectrum (unit circle)")
    ax.legend(loc="lower left")
    return savefig(fig, path)


def plot_matrix(M: np.ndarray, path, title: str, xlabel: str = "", ylabel: str = "", log: bool = True):
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(min(10, 2 + 0.06 * M.shape[1]), min(6, 1.5 + 0.12 * M.shape[0])))
    A = np.abs(M)
    norm = LogNorm(vmin=max(A[A > 0].min() if (A > 0).any() else 1e-12, A.max() * 1e-6), vmax=A.max() or 1) if log else None
    im = ax.imshow(np.where(A > 0, A, np.nan), aspect="auto", cmap="rom_sequential", norm=norm, interpolation="nearest")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="|entry|")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return savefig(fig, path)


def plot_speedup(rows: Dict[str, Dict[str, float]], path):
    """rows[method] = {'rom_seconds':..., 'fom_seconds':...} -> bar chart of wall time (log)."""
    names = ["FOM (VarFlExI)"] + list(rows)
    vals = [np.nanmean([r["fom_seconds"] for r in rows.values()])] + [r["rom_seconds"] for r in rows.values()]
    cols = [INK] + [method_color(n, j) for j, n in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(5.0, 0.6 * len(names) + 1.0))
    ax.barh(names, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        lab = f"{v:.3g} s" + ("" if i == 0 or not np.isfinite(vals[0]) else f"   ({vals[0] / v:.0f}x faster)")
        ax.annotate(lab, (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("wall time for the same simulated interval [s]")
    ax.set_title("Online cost: full-order vs reduced-order")
    return savefig(fig, path)
