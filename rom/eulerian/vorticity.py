"""
rom.eulerian.vorticity – Curl computation on structured Cartesian grids.

Computes  ω = ∇ × u  using finite differences that are consistent with
the grid's spacing (uniform or non-uniform).

Supported schemes
-----------------
* ``'central'`` – Standard central differences.
  - order 2: second-order accurate  (3-point stencil).
  - order 4: fourth-order accurate  (5-point stencil, uniform spacing only).
* Boundary treatment: one-sided (forward / backward) differences of
  matching accuracy at domain edges.

Non-uniform spacing
-------------------
When the grid has non-uniform spacing, the second-order central difference
uses the generalised formula for unequal intervals:

    ∂f/∂x |_i  ≈  [ f_{i+1} h_{i-1}² − f_{i-1} h_i² + f_i (h_i² − h_{i-1}²) ]
                   / ( h_i h_{i-1} (h_i + h_{i-1}) )

where h_i = x_{i+1} − x_i.

Fluid-mask treatment
--------------------
If a :class:`FluidMask` is supplied, solid cells in the vorticity output
are set to NaN.  Finite-difference stencils that would reach into solid
cells produce NaN naturally (NaN propagation), which is the physically
correct behaviour – no silent fill with zero or extrapolation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grid import EulerianGrid
from .masking import FluidMask


# =====================================================================
# Partial derivative helpers
# =====================================================================

def _partial_uniform_o2(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Central 2nd-order partial derivative along *axis* (uniform spacing).

    Boundaries use 2nd-order one-sided stencils.
    """
    n = f.shape[axis]
    result = np.empty_like(f)

    # Interior: central difference
    slc_m1 = [slice(None)] * f.ndim
    slc_p1 = [slice(None)] * f.ndim
    slc_c = [slice(None)] * f.ndim

    slc_m1[axis] = slice(0, n - 2)
    slc_p1[axis] = slice(2, n)
    slc_c[axis] = slice(1, n - 1)

    result[tuple(slc_c)] = (f[tuple(slc_p1)] - f[tuple(slc_m1)]) / (2.0 * dx)

    # Forward difference at lower boundary (2nd order, 3-point)
    s0 = [slice(None)] * f.ndim
    s1 = [slice(None)] * f.ndim
    s2 = [slice(None)] * f.ndim
    s0[axis] = 0
    s1[axis] = 1
    s2[axis] = 2
    result[tuple(s0)] = (-3.0 * f[tuple(s0)] + 4.0 * f[tuple(s1)] - f[tuple(s2)]) / (2.0 * dx)

    # Backward difference at upper boundary (2nd order, 3-point)
    s0[axis] = n - 1
    s1[axis] = n - 2
    s2[axis] = n - 3
    result[tuple(s0)] = (3.0 * f[tuple(s0)] - 4.0 * f[tuple(s1)] + f[tuple(s2)]) / (2.0 * dx)

    return result


def _partial_uniform_o4(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Central 4th-order partial derivative along *axis* (uniform spacing).

    Uses a 5-point interior stencil.  Falls back to lower-order stencils
    at boundaries.
    """
    n = f.shape[axis]
    if n < 5:
        return _partial_uniform_o2(f, dx, axis)

    result = np.empty_like(f)

    def _sl(a, s):
        """Build a tuple of slices with *s* at position *a*."""
        out = [slice(None)] * f.ndim
        out[a] = s
        return tuple(out)

    # Interior (indices 2 … n-3)
    result[_sl(axis, slice(2, n - 2))] = (
        -f[_sl(axis, slice(4, n))]
        + 8.0 * f[_sl(axis, slice(3, n - 1))]
        - 8.0 * f[_sl(axis, slice(1, n - 3))]
        + f[_sl(axis, slice(0, n - 4))]
    ) / (12.0 * dx)

    # Boundary layers – use 2nd-order one-sided stencils
    # i = 0
    result[_sl(axis, 0)] = (
        -3.0 * f[_sl(axis, 0)] + 4.0 * f[_sl(axis, 1)] - f[_sl(axis, 2)]
    ) / (2.0 * dx)
    # i = 1
    result[_sl(axis, 1)] = (f[_sl(axis, 2)] - f[_sl(axis, 0)]) / (2.0 * dx)
    # i = n-2
    result[_sl(axis, n - 2)] = (
        f[_sl(axis, n - 1)] - f[_sl(axis, n - 3)]
    ) / (2.0 * dx)
    # i = n-1
    result[_sl(axis, n - 1)] = (
        3.0 * f[_sl(axis, n - 1)]
        - 4.0 * f[_sl(axis, n - 2)]
        + f[_sl(axis, n - 3)]
    ) / (2.0 * dx)

    return result


def _partial_nonuniform_o2(
    f: np.ndarray, coords: np.ndarray, axis: int
) -> np.ndarray:
    """Central 2nd-order partial derivative for non-uniform spacing."""
    n = f.shape[axis]
    h = np.diff(coords)  # (n-1,)
    result = np.empty_like(f)

    def _sl(a, s):
        out = [slice(None)] * f.ndim
        out[a] = s
        return tuple(out)

    # Interior (generalised central difference)
    for i in range(1, n - 1):
        hm = h[i - 1]  # h_{i-1}
        hp = h[i]      # h_i
        coeff_m = -(hp ** 2)
        coeff_p = hm ** 2
        coeff_c = hp ** 2 - hm ** 2
        denom = hp * hm * (hp + hm)
        result[_sl(axis, i)] = (
            coeff_p * f[_sl(axis, i + 1)]
            + coeff_m * f[_sl(axis, i - 1)]
            + coeff_c * f[_sl(axis, i)]
        ) / denom

    # Boundaries – 2nd-order forward / backward using first two / last two intervals
    # Lower boundary (i = 0)
    h0, h1 = h[0], h[1]
    c0 = -(2.0 * h0 + h1) / (h0 * (h0 + h1))
    c1 = (h0 + h1) / (h0 * h1)
    c2 = -h0 / (h1 * (h0 + h1))
    result[_sl(axis, 0)] = (
        c0 * f[_sl(axis, 0)]
        + c1 * f[_sl(axis, 1)]
        + c2 * f[_sl(axis, 2)]
    )

    # Upper boundary (i = n-1)
    hm2, hm1 = h[-2], h[-1]
    c0 = hm1 / (hm2 * (hm2 + hm1))
    c1 = -(hm2 + hm1) / (hm2 * hm1)
    c2 = (2.0 * hm1 + hm2) / (hm1 * (hm2 + hm1))
    result[_sl(axis, n - 1)] = (
        c0 * f[_sl(axis, n - 3)]
        + c1 * f[_sl(axis, n - 2)]
        + c2 * f[_sl(axis, n - 1)]
    )

    return result


# =====================================================================
# General partial-derivative dispatcher
# =====================================================================

def _partial(
    f: np.ndarray,
    grid: EulerianGrid,
    phys_axis: str,
    scheme: str = "central",
    order: int = 2,
) -> np.ndarray:
    """Compute ∂f/∂(phys_axis) on the grid.

    *phys_axis* is ``'x'``, ``'y'``, or ``'z'`` (physical).  The function
    determines which storage axis this corresponds to via *grid.axis_order*.
    """
    # Map physical axis → storage axis index
    storage_idx = grid.axis_order.index(phys_axis)
    coords = grid.coords_1d(phys_axis)
    dx_arr = np.diff(coords)
    is_uniform = np.allclose(dx_arr, dx_arr[0], rtol=1e-12)

    if is_uniform:
        dx = float(dx_arr[0])
        if order >= 4 and scheme == "central":
            return _partial_uniform_o4(f, dx, storage_idx)
        else:
            return _partial_uniform_o2(f, dx, storage_idx)
    else:
        return _partial_nonuniform_o2(f, coords, storage_idx)


# =====================================================================
# Public interface
# =====================================================================

def compute_vorticity(
    velocity: np.ndarray,
    grid: EulerianGrid,
    scheme: str = "central",
    order: int = 2,
    fluid_mask: Optional[FluidMask] = None,
) -> np.ndarray:
    """Compute vorticity  ω = ∇ × u  from a velocity field on a grid.

    Parameters
    ----------
    velocity : np.ndarray
        Shape ``(Nx, Ny, Nz, 3)`` with components ``(u_x, u_y, u_z)`` in
        the physical frame.
    grid : EulerianGrid
    scheme : str
        ``'central'`` (default).
    order : int
        Accuracy order: 2 (default) or 4.
    fluid_mask : FluidMask or None
        If provided, solid-cell vorticity values are set to NaN.

    Returns
    -------
    omega : np.ndarray
        Shape ``(Nx, Ny, Nz, 3)`` – vorticity ``(ω_x, ω_y, ω_z)``.
    """
    if velocity.shape != grid.shape + (3,):
        raise ValueError(
            f"velocity shape {velocity.shape} != expected {grid.shape + (3,)}"
        )

    ux = velocity[..., 0]
    uy = velocity[..., 1]
    uz = velocity[..., 2]

    # ω_x = ∂u_z/∂y − ∂u_y/∂z
    omega_x = _partial(uz, grid, "y", scheme, order) - _partial(uy, grid, "z", scheme, order)
    # ω_y = ∂u_x/∂z − ∂u_z/∂x
    omega_y = _partial(ux, grid, "z", scheme, order) - _partial(uz, grid, "x", scheme, order)
    # ω_z = ∂u_y/∂x − ∂u_x/∂y
    omega_z = _partial(uy, grid, "x", scheme, order) - _partial(ux, grid, "y", scheme, order)

    omega = np.stack([omega_x, omega_y, omega_z], axis=-1)

    if fluid_mask is not None:
        omega = fluid_mask.apply(omega)

    return omega
