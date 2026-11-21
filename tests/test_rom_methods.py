"""OpInf and Koopman model classes on systems with known answers."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rom_koopman.dictionaries import MonomialDictionary, n_monomials
from rom_koopman.edmdc import EDMDcROM
from rom_koopman.lifting import LiftingKoopmanROM, lifting_ident_dual
from rom_opinf.derivatives import estimate_trajectory_derivatives
from rom_opinf.model import OpInfROM

A = np.array([[-0.5, 2.0], [-2.0, -0.5]])
B = np.array([[1.0], [0.5]])


def _forced_linear(t):
    u = lambda tt: np.array([np.sin(1.3 * tt)])  # noqa: E731
    s = solve_ivp(lambda tt, x: A @ x + B @ u(tt), (t[0], t[-1]), [1.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    return s.y, np.stack([u(tt) for tt in t], 1)


def test_opinf_recovers_forced_linear_system(tmp_path):
    t = np.linspace(0, 10, 1001)
    X, U = _forced_linear(t)
    _, Xc, D = estimate_trajectory_derivatives([X], [t], order=4)
    rom = OpInfROM("AB").fit(Xc, D, U, 1e-10, 1e-10)
    np.testing.assert_allclose(rom.operator_entries()["LinearOperator"], A, atol=1e-5)
    np.testing.assert_allclose(rom.operator_entries()["InputOperator"], B, atol=1e-5)
    Xp = rom.simulate(X[:, 0], t, U)
    assert np.linalg.norm(Xp - X) / np.linalg.norm(X) < 1e-3
    rom.save(tmp_path)
    np.testing.assert_allclose(OpInfROM.load(tmp_path).simulate(X[:, 0], t, U), Xp)
    J, Bl = rom.linearize(np.zeros(2))
    np.testing.assert_allclose(J, A, atol=1e-5)
    np.testing.assert_allclose(Bl, B, atol=1e-4)


def test_opinf_blowup_returns_nan():
    rom = OpInfROM("A").fit(np.eye(2), 5 * np.eye(2), None, 0, 0)  # dx/dt = 5x
    Xp = rom.simulate(np.array([1.0, 1.0]), np.linspace(0, 10, 101), np.zeros((0, 101)))
    assert np.isnan(Xp[:, -1]).all()


def test_monomial_dictionary_order():
    D = MonomialDictionary(3, 2)
    assert D.size == n_monomials(3, 2) == 10
    assert D.labels(["x", "y", "u"])[:4] == ["1", "x", "y", "u"]


def test_lifting_identifies_forced_duffing(tmp_path):
    f = lambda t, x: [x[1], x[0] - x[0] ** 3 - 0.2 * x[1] + 0.2 * x[0] ** 2 * np.cos(t)]  # noqa: E731
    rng = np.random.default_rng(1)
    dt = 0.02
    t = np.arange(0, 10 + 1e-9, dt)
    Xs = [solve_ivp(f, (0, 10), rng.uniform(-1, 1, 2), t_eval=t, rtol=1e-10, atol=1e-12).y for _ in range(5)]
    Us = [np.cos(t)[None]] * 5
    rom = LiftingKoopmanROM(deg_F=3).fit(Xs, Us, dt)
    lab = rom.coefficient_labels(["x", "y"], ["u"])
    c = dict(zip(lab, rom.coeff[:, 1]))
    assert c["x"] == pytest.approx(1.0, abs=0.01)
    assert c["y"] == pytest.approx(-0.2, abs=0.01)
    assert c["x^3"] == pytest.approx(-1.0, abs=0.01)
    assert c["x^2*u"] == pytest.approx(0.2, abs=0.01)
    assert dict(zip(lab, rom.coeff[:, 0]))["y"] == pytest.approx(1.0, abs=0.01)
    rom.save(tmp_path)
    np.testing.assert_allclose(LiftingKoopmanROM.load(tmp_path).coeff, rom.coeff)


def test_edmdc_exact_for_linear_system(tmp_path):
    dt = 0.01
    t = np.arange(0, 5, dt)
    X, U = _forced_linear(t)
    m = EDMDcROM(degree=1, ridge=1e-12).fit([X], [U], dt)
    Xp = m.simulate(X[:, 0], t, U)
    assert np.linalg.norm(Xp - X) / np.linalg.norm(X) < 2e-2       # ZOH vs continuous input
    lam = m.spectrum()["lambda"]
    assert np.min(np.abs(lam - (-0.5 + 2j))) < 1e-2
    m.save(tmp_path)
    np.testing.assert_allclose(EDMDcROM.load(tmp_path).simulate(X[:, 0], t, U), Xp)


def test_dual_method_linear():
    lin = lambda t, x: [x[1], -x[0] - 0.1 * x[1]]  # noqa: E731
    s = solve_ivp(lin, (0, 2), [1, 0], t_eval=np.arange(0, 2.0001, 0.1), rtol=1e-10)
    F = lifting_ident_dual(s.y[:, :-1].T[:9], s.y[:, 1:].T[:9], 0.1, basis=3)
    ref = np.array([lin(0, x) for x in s.y[:, :-1].T[:9]])
    assert np.max(np.abs(F - ref)) < 1e-3
