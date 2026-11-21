"""Eulerian reconstruction: kernels, direct sums, VIC, far-field lumping, bound vortex, run reader."""

import numpy as np
import pytest

from rom.eulerian import EulerianGrid, cell_volumes, evaluate_particles, vic_reconstruct
from rom.eulerian.biot_savart import segment_velocity
from rom.eulerian.kernels import KERNELS, zeta_kernel

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal([0.5, 0.5, 0.5], 0.08, (n, 3))
    G = rng.normal(0, 1e-3, (n, 3))
    S = np.full(n, 0.05)
    return X, G, S


def _grid(n=17):
    return EulerianGrid(bounds={"x": (0, 1), "y": (0, 1), "z": (0, 1)}, resolution={"x": n, "y": n, "z": n})


@pytest.mark.parametrize("kernel", [k for k in KERNELS if k != "singular"])
def test_zeta_normalised(kernel):
    r = np.linspace(0, 40, 200001)
    integral = np.trapezoid(zeta_kernel(np, r, kernel) * 4 * np.pi * r ** 2, r)
    assert integral == pytest.approx(1.0, rel=2e-3)


def test_far_field_matches_singular_biot_savart():
    X, G, S = _cloud(1)
    T = np.array([[4.0, 0.3, -0.2]])
    u = evaluate_particles(T, X, G, S, backend="numpy", vorticity=False)["velocity"]
    d = T - X
    ref = np.cross(G, d) / (4 * np.pi * np.linalg.norm(d) ** 3)
    np.testing.assert_allclose(u, ref, rtol=1e-12)


@pytest.mark.skipif(not HAS_TORCH, reason="torch missing")
def test_torch_matches_numpy():
    X, G, S = _cloud()
    P = _grid().flat_points()
    a = evaluate_particles(P, X, G, S, backend="numpy")
    b = evaluate_particles(P, X, G, S, backend="torch", precision="float64", device="cpu")
    for k in ("velocity", "vorticity"):
        np.testing.assert_allclose(b[k], a[k], rtol=1e-10, atol=1e-14)


def test_vorticity_integral_equals_total_strength():
    X, G, S = _cloud()
    g = _grid(33)
    w = evaluate_particles(g.flat_points(), X, G, S, backend="numpy", velocity=False)["vorticity"]
    integral = np.einsum("ijk,ijkc->c", cell_volumes(g), w.reshape(g.shape + (3,)))
    np.testing.assert_allclose(integral, G.sum(0), atol=1e-4 * np.linalg.norm(G.sum(0)))


def test_vic_close_to_direct():
    X, G, S = _cloud()
    g = _grid(33)
    ref = evaluate_particles(g.flat_points(), X, G, S, backend="numpy")
    v = vic_reconstruct(g, X, G, S, outside_backend="numpy")
    e = np.linalg.norm(v["velocity"].reshape(-1, 3) - ref["velocity"]) / np.linalg.norm(ref["velocity"])
    assert e < 0.05
    np.testing.assert_allclose(v["vorticity"].reshape(-1, 3), ref["vorticity"], rtol=0, atol=1e-6 * np.abs(ref["vorticity"]).max())


def test_far_field_lumping_small_error():
    X, G, S = _cloud()
    rng = np.random.default_rng(1)
    Xf = np.vstack([X, rng.uniform([3, 0, 0], [8, 1, 1], (2000, 3))])
    Gf = np.vstack([G, rng.normal(0, 1e-3, (2000, 3))])
    Sf = np.full(len(Xf), 0.05)
    P = _grid().flat_points()
    a = evaluate_particles(P, Xf, Gf, Sf, backend="numpy", vorticity=False)["velocity"]
    b = evaluate_particles(P, Xf, Gf, Sf, backend="numpy", vorticity=False, far_field_distance=1.0,
                           far_field_bin=0.1)["velocity"]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-3


def test_long_segment_approaches_infinite_line():
    A, B = np.array([[0.0, -500.0, 0.0]]), np.array([[0.0, 500.0, 0.0]])
    T = np.array([[0.0, 0.0, 0.3]])
    u = segment_velocity(T, A, B, np.array([2.0]), core=1e-6)
    # infinite vortex along +y, point above (+z): u = Gamma/(2 pi d) in +x
    np.testing.assert_allclose(u[0], [2.0 / (2 * np.pi * 0.3), 0, 0], rtol=1e-5, atol=1e-9)
