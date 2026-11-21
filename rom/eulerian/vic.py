"""
rom.eulerian.vic - Vortex-in-cell (particle -> mesh -> FFT) reconstruction.

A fast CPU alternative to the direct Biot-Savart sum for *uniform* grids:

1. **Particle-to-grid vorticity** (exact, local): ``omega(x_i) = sum_p zeta(|x_i-x_p|/s_p)/s_p^3 Gamma_p``
   evaluated with a stencil loop over the kernel support (no N x M pairs).
2. **Free-space Biot-Savart by FFT** (Hockney-Eastwood zero padding):
   ``u_i = sum_j  omega_j x K(x_i - x_j) h^3`` with the regularised kernel
   ``K(d) = d g(|d|/eps) / (4 pi |d|^3)`` - exact discrete convolution of the gridded
   vorticity with unbounded-domain boundary conditions (no periodic images).
3. **Outside particles**: vorticity whose support leaves the box is not on the grid,
   so its velocity is added by a direct sum (with far-field lumping) - the wake of a
   VarFlExI run extends far downstream of any practical ROM box.

Accuracy is controlled by h / sigma: velocity error vs. the direct sum is ~3% at
h/sigma = 0.8 and ~1% at h/sigma = 0.4 (default eps = h/4). Validate against ``backend='torch'`` on a few snapshots
(``scripts/reconstruct_eulerian.py --validate``).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from rom.eulerian.biot_savart import evaluate_particles
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.kernels import ZETA_CUTOFF, g_kernel, zeta_kernel

logger = logging.getLogger(__name__)


def _uniform_spacing(grid: EulerianGrid) -> np.ndarray:
    h = []
    for ax in "xyz":
        d = grid.spacing(ax)
        if not np.allclose(d, d[0], rtol=1e-6):
            raise ValueError("VIC backend requires a uniform grid")
        h.append(float(d[0]))
    return np.asarray(h)


def splat_vorticity(grid: EulerianGrid, X: np.ndarray, Gamma: np.ndarray, sigma: np.ndarray,
                    kernel: str = "gaussianerf", batch: int = 200_000) -> np.ndarray:
    """Exact kernel vorticity on a uniform grid, shape ``grid.shape + (3,)`` (xyz axis order)."""
    if grid.axis_order != "xyz":
        raise ValueError("VIC backend expects axis_order='xyz'")
    h = _uniform_spacing(grid)
    o = np.array([grid.coords_1d(a)[0] for a in "xyz"])
    n = np.array(grid.shape)
    W = np.zeros(tuple(n) + (3,))
    if X.shape[0] == 0:
        return W
    cut = ZETA_CUTOFF.get(kernel, 6.0)
    # keep particles whose support intersects the grid
    lo = o - cut * sigma.max()
    hi = o + (n - 1) * h + cut * sigma.max()
    keep = np.all((X >= lo) & (X <= hi), axis=1)
    X, Gamma, sigma = X[keep], Gamma[keep], sigma[keep]
    if X.shape[0] == 0:
        return W
    w = np.ceil(cut * sigma.max() / h).astype(int)
    flat = W.reshape(-1, 3)
    for b0 in range(0, X.shape[0], batch):
        Xb, Gb, Sb = X[b0:b0 + batch], Gamma[b0:b0 + batch], sigma[b0:b0 + batch]
        base = np.rint((Xb - o) / h).astype(np.int64)
        inv_s3 = 1.0 / Sb ** 3
        for di in range(-w[0], w[0] + 1):
            ii = base[:, 0] + di
            for dj in range(-w[1], w[1] + 1):
                jj = base[:, 1] + dj
                for dk in range(-w[2], w[2] + 1):
                    kk = base[:, 2] + dk
                    ok = (ii >= 0) & (ii < n[0]) & (jj >= 0) & (jj < n[1]) & (kk >= 0) & (kk < n[2])
                    if not ok.any():
                        continue
                    node = o + np.stack([ii[ok], jj[ok], kk[ok]], 1) * h
                    rho = np.linalg.norm(node - Xb[ok], axis=1) / Sb[ok]
                    z = zeta_kernel(np, rho, kernel) * inv_s3[ok]
                    lin = (ii[ok] * n[1] + jj[ok]) * n[2] + kk[ok]
                    for c in range(3):
                        np.add.at(flat[:, c], lin, z * Gb[ok, c])
    return W


def fft_biot_savart(omega: np.ndarray, h: np.ndarray, eps: Optional[float] = None,
                    kernel: str = "gaussianerf") -> np.ndarray:
    """Free-space velocity of a gridded vorticity field via zero-padded FFT convolution."""
    n = np.array(omega.shape[:3])
    # eps = smoothing of the discrete Green's function; the gridded vorticity is already
    # smooth (sigma-blobs), so a small eps (h/4) is most accurate (measured: 3.4% at
    # h/sigma=0.8, 0.9% at h/sigma=0.4 vs. the direct sum).
    eps = float(eps if eps is not None else 0.25 * h.max())
    N = 2 * n
    ax = [np.arange(N[i]) for i in range(3)]
    ax = [np.where(a < n[i], a, a - N[i]) * h[i] for i, a in enumerate(ax)]
    DX, DY, DZ = np.meshgrid(*ax, indexing="ij")
    r = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(r > 0, g_kernel(np, r / eps, kernel) / (4 * np.pi * r ** 3), 0.0)
    K = [np.fft.rfftn(f * D) for D in (DX, DY, DZ)]
    del DX, DY, DZ, r, f
    Wh = [np.fft.rfftn(omega[..., c], s=N) for c in range(3)]
    vol = float(np.prod(h))
    # u = omega x K  (component-wise convolution)
    ux = np.fft.irfftn(Wh[1] * K[2] - Wh[2] * K[1], s=N)
    uy = np.fft.irfftn(Wh[2] * K[0] - Wh[0] * K[2], s=N)
    uz = np.fft.irfftn(Wh[0] * K[1] - Wh[1] * K[0], s=N)
    sl = tuple(slice(0, n[i]) for i in range(3))
    return np.stack([ux[sl], uy[sl], uz[sl]], axis=-1) * vol


def vic_reconstruct(grid: EulerianGrid, X: np.ndarray, Gamma: np.ndarray, sigma: np.ndarray,
                    kernel: str = "gaussianerf", eps: Optional[float] = None,
                    outside_backend: str = "numpy", far_field_distance: Optional[float] = 0.5,
                    far_field_bin: Optional[float] = 0.1, device: Optional[str] = None) -> Dict[str, np.ndarray]:
    """VIC velocity + vorticity on ``grid`` (particle-induced part only).

    Particles whose kernel support lies fully inside the grid are handled by the
    FFT; all others contribute through a direct (optionally lumped) sum.
    """
    h = _uniform_spacing(grid)
    o = np.array([grid.coords_1d(a)[0] for a in "xyz"])
    top = np.array([grid.coords_1d(a)[-1] for a in "xyz"])
    cut = ZETA_CUTOFF.get(kernel, 6.0) * sigma
    inside = np.all((X - cut[:, None] >= o) & (X + cut[:, None] <= top), axis=1)
    W_in = splat_vorticity(grid, X[inside], Gamma[inside], sigma[inside], kernel)
    U = fft_biot_savart(W_in, h, eps=eps, kernel=kernel)
    # vorticity of the partially-inside particles (exact) for the output field
    W = W_in + splat_vorticity(grid, X[~inside], Gamma[~inside], sigma[~inside], kernel)
    if (~inside).any():
        pts = grid.flat_points()
        Uo = evaluate_particles(pts, X[~inside], Gamma[~inside], sigma[~inside], kernel=kernel,
                                velocity=True, vorticity=False, backend=outside_backend, device=device,
                                far_field_distance=far_field_distance, far_field_bin=far_field_bin)["velocity"]
        U = U + Uo.reshape(U.shape)
    logger.debug("VIC: %d inside / %d outside particles", int(inside.sum()), int((~inside).sum()))
    return {"velocity": U, "vorticity": W}
