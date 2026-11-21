"""rom_opinf.plots - OpInf-specific figures (regularisation, operators, spectra, rank sweep)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from rom.viz.plots import plot_continuous_spectrum, plot_discrete_spectrum, plot_matrix
from rom.viz.style import INK, INK_2, MUTED, SEQUENTIAL, SERIES, plt, savefig


def plot_regularization_sweep(sweep: dict, path):
    L1, L2 = np.asarray(sweep["lambda1"]), np.asarray(sweep["lambda2"])
    S = np.asarray(sweep["score"], float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(L2), 0.9 + 0.6 * len(L1)))
    finite = np.isfinite(S)
    Z = np.where(finite, np.log10(np.where(finite, S, 1)), np.nan)
    im = ax.imshow(Z, cmap=SEQUENTIAL.reversed(), aspect="auto", origin="lower")
    for i in range(len(L1)):
        for j in range(len(L2)):
            txt = f"{S[i, j]:.2g}" if finite[i, j] else "unstable"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color=INK if (not finite[i, j] or Z[i, j] > np.nanmedian(Z)) else "white")
    bi, bj = np.unravel_index(np.argmin(np.where(finite, S, np.inf)), S.shape) if finite.any() else (None, None)
    if bi is not None:
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=SERIES[1], lw=2.2))
    ax.set_xticks(range(len(L2)))
    ax.set_xticklabels([f"{v:.0e}" for v in L2], rotation=45)
    ax.set_yticks(range(len(L1)))
    ax.set_yticklabels([f"{v:.0e}" for v in L1])
    ax.set_xlabel(r"$\lambda_2$ (quadratic / bilinear operators)")
    ax.set_ylabel(r"$\lambda_1$ (c, A, B)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"log10 {sweep.get('selection_split', 'val')} rollout error")
    ax.set_title("Tikhonov regularisation selection (orange box = chosen)")
    return savefig(fig, path)


def plot_operator_spectra(rom, path, dt: float, x_ref=None):
    """Eigenvalues of A and of the Jacobian at the training-mean state."""
    A = rom.linear_operator()
    eigs = {}
    if A is not None:
        eigs["opinf: A"] = np.linalg.eigvals(A)
    if x_ref is not None:
        J, _ = rom.linearize(x_ref)
        eigs["opinf: Jacobian at mean state"] = np.linalg.eigvals(J)
    if not eigs:
        return None
    if rom.form == "continuous":
        return plot_continuous_spectrum(eigs, path, title="OpInf: continuous-time eigenvalues")
    return plot_discrete_spectrum(eigs, path, title="OpInf: discrete-time eigenvalues")


def plot_operator_structure(rom, path, block_slices: Dict[str, slice] = None):
    ent = rom.operator_entries()
    order = ["ConstantOperator", "LinearOperator", "InputOperator", "QuadraticOperator", "StateInputOperator", "CubicOperator"]
    mats, labels = [], []
    for k in order:
        if k in ent:
            M = np.atleast_2d(ent[k])
            if M.shape[0] != rom.r:
                M = M.reshape(rom.r, -1)
            mats.append(M)
            labels.append(f"{k.replace('Operator', '')} ({M.shape[1]})")
    full = np.hstack(mats)
    p = plot_matrix(full, path, "OpInf operator entries  [ " + " | ".join(labels) + " ]",
                    xlabel="operator column", ylabel="reduced state component")
    return p


def plot_rank_sweep(rows: Dict[str, List[dict]], path):
    blocks = list(rows)
    fig, axes = plt.subplots(1, len(blocks), figsize=(4.6 * len(blocks), 3.3), squeeze=False)
    for ax, b in zip(axes[0], blocks):
        r = [x["rank"] for x in rows[b]]
        e = [x["rollout_error"] if np.isfinite(x["rollout_error"]) else np.nan for x in rows[b]]
        p = [x["projection_error"] for x in rows[b]]
        ax.semilogy(r, p, ":", color=MUTED, lw=1.5, marker="s", ms=4, label="POD projection")
        ax.semilogy(r, e, "--", color=SERIES[0], lw=1.6, marker="o", ms=4, label="OpInf rollout (field error)")
        for ri, ei in zip(r, e):
            if not np.isfinite(ei):
                ax.annotate("unstable", (ri, ax.get_ylim()[1]), rotation=90, va="top", fontsize=7, color=INK_2)
        ax.set_xlabel(f"rank of '{b}'")
        ax.set_ylabel("relative error")
        ax.set_title(f"Rank sweep - {b}")
        ax.legend()
    fig.tight_layout()
    return savefig(fig, path)
