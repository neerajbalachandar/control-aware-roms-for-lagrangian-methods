"""rom_koopman.plots - Koopman-specific figures: spectra, spatial Koopman modes, lifting coefficients."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from rom.viz.style import DIVERGING, INK, INK_2, MUTED, SEQUENTIAL, SERIES, plt, savefig


def plot_koopman_spectrum(spec: dict, dt: float, path, title: str = "Koopman EDMDc spectrum", n_label: int = 6):
    """Two panels: discrete eigenvalues vs unit circle | continuous growth rate vs frequency.
    Marker area ~ mode amplitude on the initial condition; the n_label most energetic are annotated."""
    mu, lam, amp = spec["mu"], spec["lambda"], spec["amplitude"]
    keep = np.abs(mu) > 1e-8
    mu, lam, amp = mu[keep], lam[keep], amp[keep]
    s = 10 + 120 * amp / max(amp.max(), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=0.8)
    ax.scatter(mu.real, mu.imag, s=s, facecolor="none", edgecolor=SERIES[1], lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re $\mu$")
    ax.set_ylabel(r"Im $\mu$")
    ax.set_title("discrete-time eigenvalues")
    ax = axes[1]
    ax.axhline(0, color=INK_2, lw=0.8)
    f = lam.imag / (2 * np.pi)
    ax.scatter(f, lam.real, s=s, facecolor="none", edgecolor=SERIES[1], lw=1.1)
    top = np.argsort(-amp)[:n_label]
    for i in top:
        if f[i] > 0.05:
            ax.annotate(f"{f[i]:.1f} Hz", (f[i], lam[i].real), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=INK_2)
    ax.set_xlabel(r"frequency Im$\lambda/2\pi$ [Hz]")
    ax.set_ylabel(r"growth rate Re$\lambda$ [1/s]")
    ax.set_title(r"continuous-time  $\lambda = \log(\mu)/\Delta t$")
    nyq = 0.5 / dt
    ax.set_xlim(-nyq * 1.02, nyq * 1.02)
    fig.suptitle(title + "  (marker size = modal amplitude)", y=1.02)
    fig.tight_layout()
    return savefig(fig, path)


def plot_lifting_coefficients(coeff: np.ndarray, labels: List[str], state_labels: List[str], path,
                              max_terms: int = 60, title: str = "Identified polynomial vector field (lifting method)"):
    """Heat map of |coefficients| of the strongest dictionary terms."""
    from matplotlib.colors import LogNorm

    mag = np.abs(coeff).max(1)
    idx = np.argsort(-mag)[:max_terms]
    idx = np.sort(idx)
    M = np.abs(coeff[idx]).T
    fig, ax = plt.subplots(figsize=(min(16, 1.5 + 0.22 * len(idx)), 1.4 + 0.28 * coeff.shape[1]))
    vmax = M.max() or 1.0
    im = ax.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap=SEQUENTIAL,
                   norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), interpolation="nearest")
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=6)
    ax.set_yticks(range(coeff.shape[1]))
    ax.set_yticklabels([f"d{s}/dt" for s in state_labels], fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="|coefficient|", pad=0.01)
    ax.set_title(title)
    return savefig(fig, path)


def plot_ridge_sweep(sweeps: dict, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for j, (k, sw) in enumerate(sweeps.items()):
        r = np.array([x["ridge"] for x in sw["rows"]], float)
        s = np.array([x["score"] for x in sw["rows"]], float)
        r = np.where(r > 0, r, np.min(r[r > 0]) * 0.1 if (r > 0).any() else 1e-12)
        col = SERIES[1] if k == "edmdc" else SERIES[2]
        ax.loglog(r, np.where(np.isfinite(s), s, np.nan), "o-", color=col, label=k)
        for ri, si in zip(r, s):
            if not np.isfinite(si):
                ax.annotate("unstable", (ri, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1), rotation=90,
                            va="top", fontsize=7, color=INK_2)
    ax.set_xlabel("ridge parameter")
    ax.set_ylabel("validation rollout error")
    ax.set_title("Koopman regularisation selection")
    ax.legend()
    return savefig(fig, path)
