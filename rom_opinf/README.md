# `rom_opinf/`: Operator Inference ROMs for VarFlExI

Non-intrusive, physics-structured reduced-order models of the coupled
**vortex-particle wake + flexible plate** system. Built on the upstream
[`opinf`](../opinf-main) 0.6 library (vendored in `opinf-main/`, installed with pip).

```
rom_opinf/
├── model.py         OpInfROM: input-driven continuous / discrete OpInf, rollout with blow-up guard,
│                    linearisation (A_lin, B_lin) for controller design, save/load
├── training.py      config, derivative / pair assembly, Tikhonov grid search by validation rollout,
│                    output map, rank sweep, save/load
├── derivatives.py   finite differences that never cross trajectory boundaries
├── plots.py         regularisation heat map, operator structure, spectra, rank sweep
├── train.py         CLI:  python -m rom_opinf.train
├── evaluate.py      CLI:  python -m rom_opinf.evaluate
└── configs/opinf.yaml
```

## 1. What is being learned

The reduced state `x` is the concatenation of **block-scaled POD coordinates**
from the shared reduced dataset (see the [main README](../README.md#3-stage-2-reduced-dataset)):

| block | content | typical rank |
|---|---|---|
| `fluid` | POD of the Eulerian velocity (incl. freestream) on the fixed grid | 8–30 |
| `structure` | POD of the plate displacement (FEniCS, 1296 nodes × 3) | 2–6 |
| `structure_rate` | the plate velocity projected on the *same* modes | = `structure` |

and the input `u` contains the gust components at the wing (and any actuation
you add to VarFlExI through `control_inputs.csv`). The default model is

```
dx/dt = c + A x + H (x ⊗ x) + B u            (operators "cAHB")
```

Why each term is there:

* **H (quadratic)**: after a Galerkin-type projection the incompressible
  vorticity/velocity dynamics are exactly quadratic (convection `u·∇u`,
  vortex stretching). OpInf infers the projected operators instead of
  deriving them, which is the only option here because the full-order model is a
  Lagrangian particle method with no Eulerian operator to project.
* **A (linear)**: structural stiffness and damping, linearised aerodynamic
  coupling, and the mean-flow linearisation.
* **B (input)**: the gust enters the freestream affinely.
* **N (`u ⊗ x`, optional, `cAHBN`)**: the gust *convects* the wake, a
  state-input product. Try it when gust amplitudes are large.
* **G (cubic, optional)**: geometric nonlinearity of the plate (Green–Lagrange
  membrane strains) is cubic in the displacement.

`use_projected_rates: true` uses the FEniCS plate velocity projected on the
structural modes as the exact time derivative of the `structure` block, so only
the fluid derivatives come from finite differences.

The discrete form `x_{k+1} = c + A x_k + H(x_k⊗x_k) + B u_k` (`form: discrete`)
avoids derivative estimation altogether. It is often more robust for
transport-dominated wakes, but it is tied to the sampling time.

## 2. Regularisation and model selection

Least squares with a diagonal Tikhonov matrix: weight `λ1` on `c, A, B` and
`λ2` on `H, G, N` (McQuarrie, Huang & Willcox 2021). For each pair on the grid
the model is **rolled out on the validation runs, driven only by their recorded
inputs**. The pair with the lowest mean validation error that stays bounded
(|x| < 50 × training bound) is kept. The heat map
`figures/regularization_sweep.png` shows the whole grid, with unstable cells
marked.

The output map `y = C [1; x; u]` (lift, drag, cm, tip heave and twist, wake
probes) is fitted by ridge regression on training data. It turns any
predicted reduced state into control-relevant outputs.

## 3. Commands

```bash
# train (+ training figures). Ranks / operators can be overridden on the CLI
python -m rom_opinf.train --dataset artifacts/reduced/reduced_dataset.h5 \
    --config rom_opinf/configs/opinf.yaml --out artifacts/opinf [--rank-sweep]

python -m rom_opinf.train ... --ranks fluid=16,structure=4,structure_rate=4 --operators cAHBN
python -m rom_opinf.train ... --form discrete --out artifacts/opinf_discrete

# evaluate on held-out runs (figures + metrics.json + predictions.npz)
python -m rom_opinf.evaluate --model artifacts/opinf --splits val test [--export-fields]
```

## 4. Figures produced

| file | shows |
|---|---|
| `figures/regularization_sweep.png` | validation rollout error over (λ1, λ2); chosen cell boxed |
| `figures/operator_structure.png` | magnitude of every inferred operator entry `[c A B H …]` |
| `figures/operator_spectrum.png` | eigenvalues of `A` and of the Jacobian at the mean state (growth vs frequency) |
| `figures/rank_sweep.png` | rollout field error vs. fluid rank, against the POD projection floor |
| `evaluation/<run>/rollout_coordinates_*.png` | FOM-projected vs predicted reduced coordinates |
| `evaluation/<run>/field_error_<block>.png` | exact full-field error vs time + POD floor |
| `evaluation/<run>/field_<q>_<plane>.png` | FOM / ROM / \|error\| slices (ω_y side view, ω_x cross-flow, u_z) |
| `evaluation/<run>/outputs.png`, `outputs_psd.png` | lift, moment, tip deflection… in time and frequency |
| `evaluation/summary_*.png`, `speedup.png` | per-run error bars, wall-time speed-up vs VarFlExI |

`--export-fields` writes the predicted 3-D velocity (+ `.xmf`) so ParaView can
render ROM and FOM iso-surfaces side by side.

## 5. Known limitations and what to try

1. **Transport-dominated wakes.** The near wake is advected through the box at
   V∞ ≈ 15 m/s, so the field in the box depends on the *input history* over
   the last box-length/V∞ ≈ 0.16 s. That is a delay, and a low-order ODE can only
   approximate it, so POD-OpInf models can drift or go unstable. On
   the synthetic demo the continuous-time model is unstable while the discrete
   Koopman predictor is not. Remedies, roughly in order of effort:
   * `form: discrete`; larger `λ2`; fewer fluid modes; longer training runs;
   * start the time window after the wake has crossed the box (`t_start ≥ 0.2 s`);
   * crop the box to the near wake (`crop:` in the dataset config);
   * add the bound-circulation block (`signals:bound_circulation`). It is a
     low-dimensional, almost instantaneous carrier of the aerodynamic state;
   * structure-preserving OpInf (energy-preserving `H`, stability constraints).
     See the research-gap notes in the [main README](../README.md#7-research-gaps-and-recommendations).
2. **Coarse time sampling.** VarFlExI's dt = 2.5 ms with 6 Hz structural modes
   is fine, but strides > 4 make 4th-order finite differences unreliable.
3. **Rank of the training data.** Every input channel must actually vary in the
   training runs (different gust frequencies and amplitudes, chirps, PRBS
   actuation). Otherwise `B` is not identifiable. Channels that are zero in every
   training run are dropped automatically.
