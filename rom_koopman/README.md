# `rom_koopman/`: Koopman-operator ROMs for VarFlExI

Two complementary Koopman approaches, trained on **the same reduced state and
inputs as OpInf** so the comparison is apples to apples:

| variant | model | strengths | use it for |
|---|---|---|---|
| **Lifting method** (Mauroy & Gonçalves; port of [`Koopman-identification-master`](../Koopman-identification-master)) | identifies the polynomial vector field `dx/dt = F(x, u)` from `L = logm(K)/Δt` | no time-derivative estimates; interpretable coefficients; continuous time | system identification, checking which nonlinear terms the data actually need |
| **EDMDc** (Korda & Mezić 2018), optionally bilinear | `z_{k+1} = A z_k + B u_k (+ N(u_k⊗z_k))`, `x = C z`, `z = ψ(x)` | **linear in the lifted state**, so convex MPC, LQR and linear reachability apply directly | control-oriented prediction, spectral analysis, Koopman modes |

```
rom_koopman/
├── dictionaries.py  monomial (Mauroy-Gonçalves ordering) and RBF observables
├── lifting.py       lifting_ident_main / lifting_ident_dual (ports) + LiftingKoopmanROM
├── edmdc.py         EDMDcROM: linear / bilinear predictor, delays, re-lifting, spectrum, modes
├── training.py      config, ridge selection by validation rollout, output map, save/load
├── analysis.py      spectra, spatial Koopman modes (via the POD basis), coefficient maps
├── plots.py         Koopman-specific figures
├── train.py         CLI:  python -m rom_koopman.train
├── evaluate.py      CLI:  python -m rom_koopman.evaluate
└── configs/koopman.yaml
```

## 1. Lifting method (main), as ported

For pairs `(x_k, x_{k+1})` (inputs appended as extra states with zero-order
hold, exactly like the MATLAB Duffing example) and the monomial dictionary
`ψ = [1, x_1..x_n, degree-2 terms, …]`:

1. `K = Ψ_X⁺ Ψ_Y` (least squares, optional ridge), a finite section of the Koopman operator;
2. `L = logm(K) / Δt`, the Koopman **generator**;
3. the column of `L` belonging to the observable `x_i` holds the coefficients of
   `F_i` in the dictionary. The identified vector field is integrated with
   `solve_ivp` under the recorded inputs.

Checked in `tests/test_rom_methods.py`: on the forced Duffing system the port
recovers `ẏ = x − x³ − 0.2y + 0.2x²u` to 1e-3.

Practical limits:

* The dictionary has `C(r+m+d, d)` functions. Keep `r + m ≲ 25` for `d = 2` and
  `≲ 12` for `d = 3` (`max_dictionary` guards this). The default config uses 16
  states + 3 inputs, giving 210 functions.
* `logm(K)` must be the principal logarithm, so the sampling time must
  resolve the fastest retained dynamics (`|Im λ| < π/Δt`). Don't sub-sample
  aggressively (`stride` in the dataset config). The code warns if `K` has
  eigenvalues on the negative real axis.
* The dual method (`lifting_ident_dual`, RBF or monomial) is provided for
  vector-field estimation at sample points. It is not used for rollouts.

## 2. EDMDc: the control-oriented Koopman model

```
z = [1; x_k; x_{k-1}..x_{k-q}; monomials(x_k) of degree 2..d; RBFs(x_k)]
z_{k+1} = A z_k + B u_k + N (u_k ⊗ z_k)      (N only if bilinear: true)
x_k     = C z_k                              (C selects x: exact, no regression)
```

* `bilinear: true` (default) captures input–state coupling, such as the gust convecting the wake.
* `delays: q` adds a Hankel (time-delay) embedding. Use it if the reduced state misses
  the wake history (transport delay through the box).
* `rollout: relift` re-lifts after every step. That is usually more accurate, but the
  predictor is then no longer globally linear.

The spectrum of `A` gives discrete Koopman eigenvalues `μ_j` and continuous
`λ_j = log μ_j / Δt`. A state-space mode `C v_j` restricted to the fluid
block and pushed through the fluid POD basis is a **spatial Koopman mode**: a
3-D velocity field that oscillates at `Im λ_j / 2π` Hz and grows or decays at
`Re λ_j`.

## 3. Commands

```bash
python -m rom_koopman.train --dataset artifacts/reduced/reduced_dataset.h5 \
    --config rom_koopman/configs/koopman.yaml --out artifacts/koopman
python -m rom_koopman.train ... --methods edmdc --ranks fluid=20,structure=4,structure_rate=4
python -m rom_koopman.evaluate --model artifacts/koopman --splits val test [--export-fields]
```

## 4. Figures produced

| file | shows |
|---|---|
| `figures/ridge_sweep.png` | validation rollout error vs ridge parameter, per variant |
| `figures/koopman_spectrum_edmdc.png` | unit-circle spectrum and growth-rate vs frequency (marker size = modal amplitude) |
| `figures/koopman_modes_uz_xz.png`, `…_ux_yz.png` | spatial Koopman modes (most energetic first), side and cross-flow views |
| `figures/koopman_generator_spectrum_lifting.png` | eigenvalues of the identified generator `L` |
| `figures/lifting_vector_field_coefficients.png` | which monomials the identified vector field uses |
| `evaluation/horizon_error.png` | n-step error vs horizon with re-initialisation, the error an MPC loop sees |
| `evaluation/<run>/…` | same per-run figure set as OpInf (coordinates, field error, slices, outputs, PSD) |

## 5. Notes for VarFlExI data

* Koopman models are discrete in time and need **one uniform Δt** across all runs.
  Use the same `stride` everywhere (training refuses mixed sampling).
* A finite polynomial dictionary is rarely Koopman-invariant for a convective
  wake. Expect EDMDc to be accurate over short and medium horizons and to drift
  over long ones; check `horizon_error.png`. That is exactly the regime MPC
  needs, because the model is re-initialised from measurements every step.
* The lifting method is continuous-time and inherits the transport-delay
  difficulty discussed in the [OpInf README](../rom_opinf/README.md#5-known-limitations-and-what-to-try).
  On the synthetic demo it is unstable, while EDMDc is stable.
* For reachability, EDMDc gives `(A, B, C)`. Propagate zonotopes or ellipsoids
  of initial lifted states and bounded inputs through the linear map, and add a
  bound on the lifting residual. See the
  [research-gap section](../README.md#7-research-gaps-and-recommendations).
