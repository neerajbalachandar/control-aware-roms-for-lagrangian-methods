"""
rom.eulerian.biot_savart - Particle -> point evaluation of velocity and vorticity.

Direct O(N_targets x N_particles) summation of the regularised Biot-Savart law
(see :mod:`rom.eulerian.kernels`), chunked so memory stays bounded.

Backends
--------
``torch``  (recommended)  GPU if available, else multithreaded CPU. float32 pair
           arithmetic with float64 accumulation across source chunks is
           accurate to ~1e-6 relative - far below ROM truncation errors.
``numpy``  reference implementation (slow for >1e9 pairs; used in tests).

A typical VarFlExI snapshot (2.8e5 particles) on a 1.4e5-node grid is 4e10
pairs: ~5-20 s on a laptop GPU (RTX 2000 Ada) in float32.

Far-field binning (optional, both backends)
-------------------------------------------
Particles farther than ``far_field_distance`` from the target box are lumped
into cubic bins of edge ``far_field_bin``: each bin becomes one particle with
the summed strength placed at the |Gamma|-weighted centroid (vortex monopole).
The induced-velocity error scales like (bin/distance)^2 and is usually
negligible for wake vorticity many chords downstream, while cutting the
source count by 10-100x.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from rom.eulerian.kernels import CONST4, ZETA_CUTOFF, g_kernel, zeta_kernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Far-field lumping
# ---------------------------------------------------------------------------

def lump_far_particles(X: np.ndarray, Gamma: np.ndarray, sigma: np.ndarray,
                       box_min: np.ndarray, box_max: np.ndarray,
                       distance: float, bin_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Merge particles farther than ``distance`` from the axis-aligned target box.

    Returns the new (X, Gamma, sigma) and a small info dict.
    """
    d_out = np.maximum(np.maximum(box_min - X, X - box_max), 0.0)
    dist = np.linalg.norm(d_out, axis=1)
    far = dist > distance
    nf = int(far.sum())
    if nf == 0:
        return X, Gamma, sigma, {"n_far": 0, "n_bins": 0}
    Xf, Gf, Sf = X[far], Gamma[far], sigma[far]
    keys = np.floor(Xf / bin_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    nb = int(inv.max()) + 1
    w = np.linalg.norm(Gf, axis=1) + 1e-300
    wsum = np.bincount(inv, weights=w, minlength=nb)
    Xc = np.stack([np.bincount(inv, weights=w * Xf[:, i], minlength=nb) for i in range(3)], 1) / wsum[:, None]
    Gc = np.stack([np.bincount(inv, weights=Gf[:, i], minlength=nb) for i in range(3)], 1)
    Sc = np.maximum(np.bincount(inv, weights=w * Sf, minlength=nb) / wsum, 0.5 * bin_size)
    Xn = np.concatenate([X[~far], Xc])
    Gn = np.concatenate([Gamma[~far], Gc])
    Sn = np.concatenate([sigma[~far], Sc])
    return Xn, Gn, Sn, {"n_far": nf, "n_bins": nb}


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def _direct_numpy(targets, X, G, S, kernel, want_u, want_w, target_chunk, source_chunk):
    M = targets.shape[0]
    U = np.zeros((M, 3)) if want_u else None
    W = np.zeros((M, 3)) if want_w else None
    for i0 in range(0, M, target_chunk):
        T = targets[i0:i0 + target_chunk]
        for j0 in range(0, X.shape[0], source_chunk):
            Xs, Gs, Ss = X[j0:j0 + source_chunk], G[j0:j0 + source_chunk], S[j0:j0 + source_chunk]
            d = T[:, None, :] - Xs[None, :, :]
            r2 = np.einsum("ijk,ijk->ij", d, d)
            r = np.sqrt(r2)
            rho = r / Ss[None, :]
            if want_u:
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(r2 > 0, -CONST4 * g_kernel(np, rho, kernel) / (r2 * r), 0.0)
                cr = np.cross(d, Gs[None, :, :])
                U[i0:i0 + target_chunk] += np.einsum("ij,ijk->ik", f, cr)
            if want_w:
                z = zeta_kernel(np, rho, kernel) / Ss[None, :] ** 3
                W[i0:i0 + target_chunk] += z @ Gs
    return U, W


# ---------------------------------------------------------------------------
# torch backend
# ---------------------------------------------------------------------------

def _direct_torch(targets, X, G, S, kernel, want_u, want_w, target_chunk, source_chunk,
                  device, precision):
    import torch

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = torch.float32 if precision == "float32" else torch.float64
    Xs_all = torch.as_tensor(X, dtype=dt, device=dev)
    Gs_all = torch.as_tensor(G, dtype=dt, device=dev)
    Ss_all = torch.as_tensor(S, dtype=dt, device=dev)
    M = targets.shape[0]
    U = np.zeros((M, 3)) if want_u else None
    W = np.zeros((M, 3)) if want_w else None
    with torch.no_grad():
        for i0 in range(0, M, target_chunk):
            T = torch.as_tensor(targets[i0:i0 + target_chunk], dtype=dt, device=dev)
            Uacc = torch.zeros((T.shape[0], 3), dtype=torch.float64, device=dev)
            Wacc = torch.zeros((T.shape[0], 3), dtype=torch.float64, device=dev)
            for j0 in range(0, Xs_all.shape[0], source_chunk):
                Xs = Xs_all[j0:j0 + source_chunk]
                Gs = Gs_all[j0:j0 + source_chunk]
                Ss = Ss_all[j0:j0 + source_chunk]
                d = T[:, None, :] - Xs[None, :, :]                     # (mt, ns, 3)
                r2 = (d * d).sum(-1)
                r = torch.sqrt(r2)
                rho = r / Ss[None, :]
                if want_u:
                    f = -CONST4 * g_kernel(torch, rho, kernel) / (r2 * r)
                    f = torch.where(r2 > 0, f, torch.zeros_like(f))
                    # (d x G) summed with weights f, without materialising the cross product:
                    # sum_j f_ij (d_ij x G_j) = (sum_j f_ij d_ij x G_j)
                    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
                    gx, gy, gz = Gs[:, 0], Gs[:, 1], Gs[:, 2]
                    ux = (f * (dy * gz - dz * gy)).sum(1)
                    uy = (f * (dz * gx - dx * gz)).sum(1)
                    uz = (f * (dx * gy - dy * gx)).sum(1)
                    Uacc += torch.stack([ux, uy, uz], 1).to(torch.float64)
                if want_w:
                    z = zeta_kernel(torch, rho, kernel) / (Ss[None, :] ** 3)
                    Wacc += (z @ Gs).to(torch.float64)
            if want_u:
                U[i0:i0 + target_chunk] = Uacc.cpu().numpy()
            if want_w:
                W[i0:i0 + target_chunk] = Wacc.cpu().numpy()
    return U, W


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_particles(targets: np.ndarray, X: np.ndarray, Gamma: np.ndarray, sigma: np.ndarray,
                       kernel: str = "gaussianerf", velocity: bool = True, vorticity: bool = True,
                       backend: str = "torch", device: Optional[str] = None,
                       precision: str = "float32", target_chunk: Optional[int] = None,
                       source_chunk: Optional[int] = None,
                       far_field_distance: Optional[float] = None,
                       far_field_bin: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Induced velocity and/or vorticity of vortex particles at ``targets``.

    Parameters
    ----------
    targets : (M, 3) evaluation points.
    X, Gamma, sigma : particle positions (N,3), strengths (N,3), core sizes (N,).
    kernel : one of :data:`rom.eulerian.kernels.KERNELS` (must match the simulation).
    far_field_distance, far_field_bin : enable monopole lumping of far particles
        for the *velocity* (vorticity always uses the exact near particles).

    Returns ``{'velocity': (M,3), 'vorticity': (M,3)}`` (requested keys only;
    the velocity is the particle-*induced* part, without freestream).
    """
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Gamma = np.ascontiguousarray(Gamma, dtype=np.float64)
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    M = targets.shape[0]
    out: Dict[str, np.ndarray] = {}
    if X.shape[0] == 0:
        if velocity:
            out["velocity"] = np.zeros((M, 3))
        if vorticity:
            out["vorticity"] = np.zeros((M, 3))
        return out

    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            logger.warning("torch not available - falling back to numpy backend")
            backend = "numpy"

    if backend == "torch":
        tc = target_chunk or 2048
        sc = source_chunk or 8192
        run = lambda T, Xp, Gp, Sp, wu, ww: _direct_torch(T, Xp, Gp, Sp, kernel, wu, ww, tc, sc, device, precision)  # noqa: E731
    elif backend == "numpy":
        tc = target_chunk or 512
        sc = source_chunk or 4096
        run = lambda T, Xp, Gp, Sp, wu, ww: _direct_numpy(T, Xp, Gp, Sp, kernel, wu, ww, tc, sc)  # noqa: E731
    else:
        raise ValueError(f"Unknown backend '{backend}' (torch|numpy)")

    if velocity:
        Xu, Gu, Su = X, Gamma, sigma
        if far_field_distance is not None and far_field_bin is not None:
            Xu, Gu, Su, info = lump_far_particles(X, Gamma, sigma, targets.min(0), targets.max(0),
                                                  far_field_distance, far_field_bin)
            logger.debug("far-field lumping: %s", info)
        out["velocity"] = run(targets, Xu, Gu, Su, True, False)[0]
    if vorticity:
        # zeta is compactly supported in practice: only particles near the box matter.
        cut = ZETA_CUTOFF.get(kernel, 6.0) * (sigma.max() if sigma.size else 0.0)
        lo, hi = targets.min(0) - cut, targets.max(0) + cut
        near = np.all((X >= lo) & (X <= hi), axis=1)
        out["vorticity"] = run(targets, X[near], Gamma[near], sigma[near], False, True)[1]
    return out


def estimate_pair_cost(n_targets: int, n_particles: int, pairs_per_second: float = 3e9) -> float:
    """Rough wall-time [s] estimate for a direct evaluation."""
    return n_targets * n_particles / pairs_per_second


def segment_velocity(targets: np.ndarray, A: np.ndarray, B: np.ndarray, gamma: np.ndarray,
                     core: float) -> np.ndarray:
    """Velocity of straight vortex segments A->B with circulation ``gamma``.

    Regularised Biot-Savart for finite segments (van Garrel / Vatistas-n=2-like
    cut-off: |r1 x r2|^2 + (core |r0|)^2 in the denominator), vectorised over
    (targets x segments). Used for the lifting-line bound vortex.
    """
    T = np.asarray(targets, float)[:, None, :]
    r0 = (B - A)[None, :, :]
    r1 = T - A[None, :, :]
    r2 = T - B[None, :, :]
    c = np.cross(r1, r2)
    c2 = np.einsum("ijk,ijk->ij", c, c)
    n1 = np.linalg.norm(r1, axis=2)
    n2 = np.linalg.norm(r2, axis=2)
    l0 = np.einsum("ijk,ijk->ij", r0, r0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.einsum("ijk,ijk->ij", r0, r1 / n1[..., None] - r2 / n2[..., None])
        f = gamma[None, :] / (4 * math.pi) * k / (c2 + core ** 2 * l0)
        f = np.where(np.isfinite(f), f, 0.0)
    return np.einsum("ij,ijk->ik", f, c)
