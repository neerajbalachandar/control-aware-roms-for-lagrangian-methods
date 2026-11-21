"""
rom_koopman.analysis - Spectral / modal analysis figures of fitted Koopman ROMs.

* EDMDc: discrete + continuous spectrum, and *spatial Koopman modes*: a state-space
  mode C v_j is split into its fluid-block part, unscaled, and pushed through the
  fluid POD basis -> a 3-D velocity field oscillating at Im(lambda_j)/2pi and
  growing / decaying at Re(lambda_j).
* lifting: generator spectrum and the identified polynomial vector-field
  coefficients (which terms the data actually need).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from rom.reduced.dataset import ReducedDataset
from rom.reduced.state import StateAssembler
from rom.viz import fields as vf
from rom.viz import plots as vp
from rom_koopman.plots import plot_koopman_spectrum, plot_lifting_coefficients

logger = logging.getLogger(__name__)


def koopman_figures(models: Dict, asm: StateAssembler, ds: ReducedDataset, fig_dir: Path, n_modes: int = 6):
    fig_dir = Path(fig_dir)
    train = ds.split("train")
    x0 = asm.state(train[0])[:, 0]
    if "edmdc" in models:
        m = models["edmdc"]
        spec = m.spectrum(x0)
        plot_koopman_spectrum(spec, m.dt, fig_dir / "koopman_spectrum_edmdc")
        _spatial_modes(spec, asm, ds, fig_dir, n_modes)
    if "lifting" in models:
        m = models["lifting"]
        ev = m.generator_eigenvalues()
        ev = ev[np.abs(ev.imag) < np.pi / m.dt]
        vp.plot_continuous_spectrum({"Koopman lifting: generator L": ev}, fig_dir / "koopman_generator_spectrum_lifting",
                                    title="Lifting method: eigenvalues of the identified generator")
        sl = asm.slices()
        state_labels = [f"{b}{i + 1}" for b in asm.blocks for i in range(asm.ranks[b])]
        labels = m.coefficient_labels(state_labels, asm.input_names)
        plot_lifting_coefficients(m.coeff, labels, state_labels, fig_dir / "lifting_vector_field_coefficients")


def _spatial_modes(spec, asm, ds, fig_dir, n_modes):
    fluid = [b for b in asm.blocks if str(ds.block_info[b].get("source", "")).startswith("field:")]
    if not fluid:
        return
    b = fluid[0]
    grid = ds.block_grid(b)
    if grid is None:
        return
    sl = asm.slices()[b]
    lam, modes, amp = spec["lambda"], spec["modes"], spec["amplitude"]
    # keep one of each conjugate pair, drop the constant observable (lambda ~ 0, real)
    cand = [j for j in np.argsort(-amp) if lam[j].imag >= 0 and abs(lam[j]) > 1e-8]
    cand = cand[:n_modes]
    if not cand:
        return
    basis = ds.basis(b)
    fields_ = []
    labels = []
    for j in cand:
        coords = np.real(modes[sl, j]) * asm.scales[b]
        f = basis.modes[:, :coords.shape[0]] @ coords
        fields_.append(f / (np.max(np.abs(f)) or 1.0))
        labels.append(f"f = {lam[j].imag / (2 * np.pi):.2f} Hz, growth {lam[j].real:.2f} 1/s")
    M = np.stack(fields_, axis=1)
    span = 0.96
    try:
        from rom.data.trajectory_file import TrajectoryFile

        tf = TrajectoryFile(ds.split("train")[0].source_file)
        span = float(tf.parameters.get("solid.span", span))
    except Exception:
        pass
    for comp, plane, val in (("uz", "xz", 0.5 * span), ("ux", "yz", float(np.mean(grid.bounds["x"])))):
        vf.plot_vector_modes(grid, M, fig_dir / f"koopman_modes_{comp}_{plane}", component=comp, plane=plane,
                             value=val, n=len(cand), labels=labels, wing={"span": span, "chord": 0.24},
                             title=f"Spatial Koopman modes (Re part, {comp}, {plane}-plane) - most energetic first")
