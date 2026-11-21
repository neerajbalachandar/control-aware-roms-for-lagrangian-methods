"""
rom.eulerian.kernels - Regularised vortex-particle kernels (exact FLOWVPM definitions).

For a particle p with strength Gamma_p (vector circulation) and core size sigma_p,
FLOWVPM (``FLOWVPM_kernel.jl`` / ``FLOWVPM_UJ.jl``) evaluates

    u(x)     = sum_p  g(rho) * (-1/4pi) * (d x Gamma_p) / |d|^3 ,   d = x - x_p
    omega(x) = sum_p  zeta(rho) / sigma_p^3 * Gamma_p ,               rho = |d| / sigma_p

with (``const1 = (2 pi)^-1.5``, ``const2 = sqrt(2/pi)``, ``const3 = 3/(4 pi)``,
``const4 = 1/(4 pi)``):

=============  ===========================================  ===================================
kernel         g(rho)                                        zeta(rho)
=============  ===========================================  ===================================
gaussianerf    erf(rho/sqrt2) - const2 rho exp(-rho^2/2)     const1 exp(-rho^2/2)  (FLOWVPM default)
gaussian       1 - exp(-rho^3)                               const3 exp(-rho^3)
winckelmans    rho^3 (rho^2 + 2.5) / (rho^2 + 1)^2.5         const4 * 7.5 / (rho^2 + 1)^3.5
singular       1                                             0  (point vortex)
=============  ===========================================  ===================================

The functions below are written against an ``xp`` array namespace so the same code
runs on NumPy and torch.
"""

from __future__ import annotations

import math

import numpy as np

CONST1 = 1.0 / (2.0 * math.pi) ** 1.5
CONST2 = math.sqrt(2.0 / math.pi)
CONST3 = 3.0 / (4.0 * math.pi)
CONST4 = 1.0 / (4.0 * math.pi)
SQRT2 = math.sqrt(2.0)

KERNELS = ("gaussianerf", "gaussian", "winckelmans", "singular")

# Radius (in units of sigma) beyond which zeta is negligible (< ~1e-7 of its peak).
ZETA_CUTOFF = {"gaussianerf": 5.5, "gaussian": 2.6, "winckelmans": 12.0, "singular": 0.0}


def _erf(xp, x):
    if xp is np:
        from scipy.special import erf

        return erf(x)
    return xp.erf(x)


def g_kernel(xp, rho, kernel: str = "gaussianerf"):
    """Velocity regularisation g(rho) (-> 1 as rho -> inf)."""
    if kernel == "gaussianerf":
        return _erf(xp, rho / SQRT2) - CONST2 * rho * xp.exp(-0.5 * rho * rho)
    if kernel == "gaussian":
        return 1.0 - xp.exp(-rho * rho * rho)
    if kernel == "winckelmans":
        r2 = rho * rho
        return rho * r2 * (r2 + 2.5) / (r2 + 1.0) ** 2.5
    if kernel == "singular":
        return xp.ones_like(rho)
    raise ValueError(f"Unknown kernel '{kernel}', choose from {KERNELS}")


def zeta_kernel(xp, rho, kernel: str = "gaussianerf"):
    """Vorticity distribution zeta(rho) with  int zeta 4 pi rho^2 drho = 1."""
    if kernel == "gaussianerf":
        return CONST1 * xp.exp(-0.5 * rho * rho)
    if kernel == "gaussian":
        return CONST3 * xp.exp(-rho * rho * rho)
    if kernel == "winckelmans":
        return CONST4 * 7.5 / (rho * rho + 1.0) ** 3.5
    if kernel == "singular":
        return xp.zeros_like(rho)
    raise ValueError(f"Unknown kernel '{kernel}', choose from {KERNELS}")
