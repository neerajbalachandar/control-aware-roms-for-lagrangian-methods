# Control-aware ROMs for Lagrangian (vortex-particle) FSI: VarFlExI

Reduced-order models of **VarFlExI** simulations (FLOWUnsteady reformulated-VPM
wake + FEniCS nonlinear Reissner–Mindlin plate, coupled with work-conserving
CRM/RBF transfer) using two data-driven families:

| folder | method | one-line idea |
|---|---|---|
| [`rom_opinf/`](rom_opinf/README.md) | **Operator Inference** | infer `dx/dt = c + Ax + H(x⊗x) + Bu (+N(u⊗x))` from projected snapshots |
| [`rom_koopman/`](rom_koopman/README.md) | **Koopman**: Mauroy–Gonçalves lifting + EDMDc | identify the vector field via `logm` of the Koopman matrix / a linear predictor in lifted observables |
| [`rom/`](rom) | shared core | VarFlExI reader, **Lagrangian → Eulerian reconstruction**, streaming POD, reduced dataset, evaluation, figures |
| `opinf-main/`, `Koopman-identification-master/` | upstream references (unchanged) | `opinf` 0.6 source; MATLAB/Julia lifting codes that `rom_koopman/lifting.py` ports |

> **Status (2026-11-21).** The pipeline is complete and tested (36 tests, including an
> end-to-end run on synthetic VarFlExI-like data). The real-data stages have *not*
> been run at scale yet, because the training runs still have to be generated.
> Section 5 lists what to simulate, and section 6 gives the exact commands to run afterwards.

---

## Contents
1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [Stage 1: Lagrangian → Eulerian reconstruction](#2-stage-1-lagrangian--eulerian-reconstruction)
3. [Stage 2: reduced dataset](#3-stage-2-reduced-dataset)
4. [Stages 3–4: ROMs, evaluation, comparison and figures](#4-stages-34-roms-evaluation-comparison-and-figures)
5. [Generating the training data (what to simulate)](#5-generating-the-training-data-what-to-simulate)
6. [Commands to run once the data exist](#6-commands-to-run-once-the-data-exist)
7. [Research gaps and recommendations](#7-research-gaps-and-recommendations)
8. [Repository layout, installation and tests](#8-repository-layout-installation-and-tests)
9. [References](#9-references)

---

## 1. Pipeline at a glance

```
 VarFlExI run folders (raw)                     one fixed Eulerian grid for ALL runs
 ├─ fluid/<run>_pfield.N.h5   X, Γ, σ ──┐
 ├─ fluid/*station_history.csv  Γ(y,t) ─┤  Stage 1  scripts/reconstruct_eulerian.py
 ├─ coupling/coupling_history.csv  L,D,M┤    u = V∞(x,t) + Σ_p K_σ(x−x_p)×Γ_p + u_bound      (torch GPU | numpy | VIC)
 ├─ solid/elastodynamics-results.h5  w  ┤    ω = Σ_p ζ_σ(x−x_p) Γ_p
 └─ run_configuration.json (gust, AoA)  ┘    ─► data/trajectories/<run>.h5 (+ .xmf for ParaView)
                                              velocity, vorticity, plate disp./vel., inputs u(t), outputs y(t)
                                                        │
                                  Stage 2  scripts/build_reduced_dataset.py
                                    weighted streaming POD per block (fluid | structure | Γ(y)),
                                    TRAIN runs only; project every run ─► artifacts/reduced/reduced_dataset.h5
                                                        │
               ┌────────────────────────────────────────┴───────────────────────────────┐
     Stage 3a  python -m rom_opinf.train/evaluate                 Stage 3b  python -m rom_koopman.train/evaluate
     dx/dt = c + Ax + H(x⊗x) + Bu                                 lifting: F(x,u) from logm(K)/Δt ; EDMDc: z⁺ = Az + Bu (+N u⊗z)
               └────────────────────────────────────────┬───────────────────────────────┘
                                  Stage 4  scripts/compare_methods.py
                         identical inputs, initial states, metrics and figures for every model
```

Design decisions:

* **One fixed grid.** POD needs every snapshot on the same discretisation. Earlier
  VarFlExI analyses (`rect_nonlinear/analysis/.../reconstructed_fields`) used
  per-step cumulative bounds, so those fields **cannot** be used for POD.
* **Split by run, never by time.** Val and test runs are unseen forcing histories.
* **Same basis, scaling and inputs for every method**, so the comparison is fair.
* **Exact full-field errors without re-reading fields.** The W-orthonormal POD
  modes give `‖q−q̄−Vx‖²_W = ‖q−q̄‖²_W − ‖a‖² + ‖a−x‖²`, and the snapshot norms
  are stored with the coordinates.
* **Inputs and outputs are first-class.** Every model is driven only by the
  recorded inputs `u(t)` (gust and actuation), and a learned output map gives
  `y = C[1;x;u]` (lift, drag, cm, tip heave and twist, wake probes).

## 2. Stage 1: Lagrangian → Eulerian reconstruction

VarFlExI writes only particles, so `rom/eulerian/` builds the Eulerian fields.

**Physics reproduced exactly (checked against the FLOWVPM sources):**

* FLOWVPM regularised kernels (`gaussianerf` default; `gaussian`, `winckelmans`,
  `singular`): `u = Σ g(ρ)(−1/4π)(d×Γ)/|d|³` and `ω = Σ ζ(ρ)Γ/σ³` (`rom/eulerian/kernels.py`).
* VarFlExI freestream with gust, `V = V∞[cos α,0,sin α] + G(x−x_ref) + A sin(2πft+φ+k·x)`,
  ported from `freestream_at` (`run_reader.VarFlExIRun.freestream`).
* **Bound vorticity of the VLM wing.** It is *not* in the pfield files unless
  `*_staticpfield` files were saved, and none of the inspected runs have them.
  It is rebuilt from the station history as a lifting line at `η_bv = 0.25`
  using the *deformed* control points (`bound_vortex.py`). The sign was checked
  against the shed tip-vortex particles (tip Γₓ > 0, root Γₓ < 0).
* The structural state (plate displacement and velocity), loads and gusts are
  aligned **by step index** (the CSV `time` columns use different conventions;
  see `rom/varflexi/run_reader.py`).

**Backends:**

| backend | how | accuracy | cost (09-11-momentcons-crm, step 599: 2.8e5 particles) |
|---|---|---|---|
| `torch` (default) | direct O(N·M) sum on GPU, float32 pairs / float64 accumulation, **far-field lumping** of particles > 0.5 m from the box into 0.1 m bins | 2e-7 vs float64; lumping 4e-5 | 36k nodes: 20 s → **3 s** with lumping (RTX 2000 Ada) |
| `numpy` | same, CPU reference | exact | tests and small grids only |
| `vic` | particle→grid splat + free-space (Hockney–Eastwood) FFT Biot–Savart + direct sum for outside particles | 3% at h/σ=0.8, 1% at h/σ=0.4 | CPU-only option |
| Julia FMM | VarFlExI's `fluid/postprocess_particle_fields.jl` with **fixed** `--bounds`, then `scripts/convert_julia_fluiddomain.py` | FMM tolerance | fastest for very large grids |

Measured checks on real data (09-11-momentcons-crm, t = 0.75 s): up-wash
+0.65 m/s ahead of the wing and down-wash −2.1 m/s behind it at mid-span,
consistent lift sign, and loads and plate output read correctly. The default
grid (`configs/reconstruction.yaml`: x ∈ [−0.12, 2.28], y ∈ [−0.12, 1.08],
z ∈ [−0.30, 0.42] m, h = 0.03 m, 83k nodes) costs roughly **0.5–1.5 h per 600-step
run** on this GPU. Use `--dry-run` for an estimate per run.

Diagnostics written per snapshot: `∫ω dV` over the box vs. `ΣΓ_p` over all
particles (this tells you how much wake leaves the box), particle count, and
evaluation time. QA figures `qa_<run>_step<N>.png` show the particle cloud, |ω|
and u_z slices.

## 3. Stage 2: reduced dataset

`scripts/build_reduced_dataset.py` + `configs/reduced_dataset.yaml`:

* **blocks**: `fluid` (velocity or vorticity field, volume-weighted inner
  product, optional crop box), `structure` (plate displacement, area-weighted)
  plus `structure_rate` (plate velocity on the same modes), and optionally
  `circulation` (span-wise bound Γ(y,t), 80 stations);
* **streaming method-of-snapshots POD** (`rom/reduced/streaming_pod.py`):
  in-memory below `memory_limit_gb`, otherwise out-of-core block pairs, so
  thousands of 83k×3 snapshots fit on a workstation;
* a `time_window` that drops the impulsive-start transient. It should be at least the
  time the wake needs to cross the box, ≈ 2.4 m / 15 m/s ≈ 0.16 s;
* channels that are zero in all training runs are dropped (they are unidentifiable).

Figures: POD spectra (σᵢ/σ₁ and unresolved energy), projection error vs rank
per split, leading fluid modes (side and cross-flow views), plate mode shapes,
input and output overview coloured by split, and the vorticity-conservation check.

## 4. Stages 3–4: ROMs, evaluation, comparison and figures

The theory, options and limitations of each method are in its own README:
[OpInf](rom_opinf/README.md) and [Koopman](rom_koopman/README.md). Both write the
**same evaluation figure set** per held-out run:

| figure | content |
|---|---|
| `rollout_coordinates_<block>.png` | FOM-projected vs predicted reduced coordinates |
| `field_error_<block>.png` | exact full-field relative error vs time, with the POD projection floor |
| `field_wy_xz.png`, `field_wx_yz.png`, `field_uz_fluct_xz.png` | **FOM / ROM / \|error\|** slices at 3 times (shed vorticity side view, tip vortices cross-flow, vertical velocity) |
| `outputs.png`, `outputs_psd.png` | lift, cm, tip deflection and velocity, wake probes in time and frequency |
| `summary_field_error.png`, `summary_output_nrmse.png`, `speedup.png` | per-run bars, wall-time speed-up vs VarFlExI |
| `horizon_error.png` | n-step error with re-initialisation (what an MPC loop experiences) |

plus method-specific figures: the OpInf regularisation heat map, operator
structure and spectrum, rank sweep; the Koopman unit-circle and continuous
spectra, **spatial Koopman modes**, identified-vector-field coefficients, and
ridge sweep. `--export-fields` writes predicted 3-D fields with `.xmf`, so you can
make ParaView renders of ROM and FOM iso-surfaces side by side.

**Demo** (`python scripts/run_synthetic_demo.py --work artifacts/synthetic_demo`, ~6 min):
on synthetic gust-forced runs (not physical; plumbing only), test run with
frequency and amplitude outside the training set:

| model | full-field error (POD floor 4.3%) | output NRMSE | stable |
|---|---|---|---|
| OpInf `cAHB`, r = 4+2+2 | 4.7% | 0.11 | yes |
| Koopman EDMDc (bilinear, deg 2) | 7.7% | 0.16 | yes |
| Koopman lifting (deg 2, ridge 1e-3) | 7.4% | 0.10 | yes |

Lessons from the demo that will matter for real data. (1) Rank and regularisation
selection decide between "accurate" and "unstable". With fluid rank 6–10 the
continuous models degraded to about 60% error, while rank 4 gave 4.7%. Always run
`--rank-sweep`. (2) The lifting method needs ridge ≥ 1e-3, because otherwise `K` has
eigenvalues on the negative real axis and `logm` has no principal branch. (3) The
discrete-time forms are more forgiving: discrete OpInf gave 3.3% on validation.

## 5. Generating the training data (what to simulate)

The inspected existing runs (`05-11-rbf`, `06-11-crm`, `09-11-momentcons-crm`)
have **no gust and no actuation** (`freestream_oscillation = [0,0,0]`). They
contain only the impulsive-start transient, so they cannot identify any input
operator. They are still useful as an *autonomous* test of transient prediction.

To learn input-output dynamics you need runs whose **inputs vary**. With the
current VarFlExI options (`config/fluid_params.yaml`):

| set | parameters | purpose |
|---|---|---|
| train (≥ 6 runs) | `freestream_oscillation: [0, 0, A]`, `freestream_frequency_hz` f ∈ {1, 2, 3, 5, 7, 9} Hz (spanning the first bending mode), A ∈ {0.75, 1.5} m/s, random `freestream_phase_deg` | identify B, N and the nonlinear response |
| val (2) | f = 4, 6 Hz | regularisation and rank selection |
| test (2–3) | f = 2.5 Hz with A = 2 m/s (amplitude extrapolation), f = 8 Hz, optionally a two-tone gust | generalisation |

Keep **identical** `total_time` (≥ 1.5 s, better 2–3 s so that several periods
follow the transient), `n_steps`, `save_every: 1`, AoA, V∞ and structural
parameters across runs, unless you deliberately build a parametric ROM. Per run
the particles take about 17 GB, and a reconstructed trajectory file about
0.6–1.2 GB (gzip float32, 83k nodes).

For **control-aware** ROMs, add actuation: a trailing-edge flap or twist schedule,
or a root heave/pitch motion (`fluid/fluid_validation_span_heave_viscous.jl`
already has heave). Write the commanded signal to `<run>/control_inputs.csv`
(`time` or `step` column plus one column per channel). It is picked up
automatically as `control_<name>`. Excite it with chirps, multisines or PRBS,
independently of the gust, so disturbance and control channels are separately
identifiable.

## 6. Commands to run once the data exist

```bash
cd control-aware-roms-for-lagrangian-methods
pip install -e .            # or run from the repo root (scripts add it to sys.path)
python -m pytest -q -m "not slow"        # 35 fast tests (~10 s)

R=/media/dysco/New\ Volume/Neeraj/VarFlExI_Results     # or wherever the new runs are

# 1) Eulerian reconstruction (resumable; run several in one call; --dry-run first)
python scripts/reconstruct_eulerian.py --run-dir "$R/gust_f1_A0p75" "$R/gust_f2_A0p75" ... --dry-run
python scripts/reconstruct_eulerian.py --run-dir "$R/gust_f1_A0p75" "$R/gust_f2_A0p75" ... \
       --config configs/reconstruction.yaml --out-dir data/trajectories --validate
#    (coarse/faster grid: --config configs/reconstruction_coarse.yaml ; CPU only: --backend vic)

# 2) reduced dataset (edit train/val/test file lists in the config first)
python scripts/build_reduced_dataset.py --config configs/reduced_dataset.yaml \
       --out artifacts/reduced/reduced_dataset.h5

# 3a) Operator Inference
python -m rom_opinf.train --dataset artifacts/reduced/reduced_dataset.h5 --out artifacts/opinf --rank-sweep
python -m rom_opinf.train --dataset artifacts/reduced/reduced_dataset.h5 --out artifacts/opinf_disc --form discrete
python -m rom_opinf.evaluate --model artifacts/opinf --splits val test --export-fields

# 3b) Koopman
python -m rom_koopman.train --dataset artifacts/reduced/reduced_dataset.h5 --out artifacts/koopman
python -m rom_koopman.evaluate --model artifacts/koopman --splits val test

# 4) comparison (figures + comparison.md table)
python scripts/compare_methods.py --opinf artifacts/opinf --koopman artifacts/koopman \
       --splits test --out artifacts/comparison
```

Ranks, operators and methods can be overridden on the CLI (`--ranks fluid=16,structure=4,structure_rate=4`,
`--operators cAHBN`, `--methods edmdc`). Start from the rank where the
projection-error curve of stage 2 flattens, then let `--rank-sweep` decide.

Already-reconstructed Julia grids (fixed bounds only):
`python scripts/convert_julia_fluiddomain.py --run-dir RUN --fd-dir RUN/fluid_domain_rom --out data/trajectories/RUN.h5 --bound-vortex`.

## 7. Research gaps and recommendations

**Is this the right method?** POD + OpInf/Koopman is a sound, non-intrusive
*baseline*, and the only practical intrusive-free route for a Lagrangian FOM.
For the broader goal, though (a **control-aware ROM that preserves control-relevant
structure and supports reachability for a nonlinear FSI system**), it leaves
these gaps:

### 7.1 State representation for Lagrangian, convective wakes
* *Gap:* Eulerian POD of an advected wake has a slowly decaying Kolmogorov
  n-width. The field in a fixed box is essentially a **delayed** image of the
  forcing history, so low-order ODEs approximate a delay and drift.
  Reconstruction to a grid is also the most expensive stage.
* *Suggestions:*
  1. **Lagrangian-native observables**: span-wise bound circulation Γ(y,t)
     (already exported; add the `circulation` block), wake impulse `½Σx×Γ`,
     and circulation moments. These are low-dimensional and cheap, and they
     carry the load-relevant aerodynamics (the Wagner/Küssner indicial-response view).
  2. **Transport-aware bases**: shifted/registered POD along V∞, or
     **quadratic-manifold OpInf** (Geelhoed et al.; Barnett & Farhat), which
     cover advection with far fewer coordinates.
  3. **Explicit delay structure**: time-delay embeddings (EDMDc `delays`), or a
     state-space realisation of the wake delay. This is what the synthetic demo
     indicates: discrete models were more robust.

### 7.2 Control-awareness (preserving what matters for control)
* *Gap:* POD ranks modes by **energy**, not by controllability or
  observability. Low-energy modes can dominate the input-to-lift map, and
  energetic wake modes can be irrelevant to control.
* *Suggestions:*
  1. **Balanced / input-output-aware bases without adjoints**: empirical
     Gramians from impulse runs per input, balanced POD via output projection
     (Rowley 2005), or ERA/OKID realisations from gust and actuator impulse responses.
  2. **Goal-oriented weighting**: weight the POD inner product or the OpInf
     loss by the outputs (lift, root bending moment, tip twist), for example
     with an output-weighted residual term.
  3. **Frequency-domain fidelity**: the gust-frequency sweep you will run is a
     sampled transfer function. Validate (and fit with the **Loewner
     framework**) the gust→lift and gust→tip-deflection responses. Report
     Bode/H∞ errors, not only time-domain errors. Linearised ROMs
     (`OpInfROM.linearize`) give poles, zeros and Hankel singular values to compare.
  4. Separate disturbance `d` from control `u` channels, and identify with
     independent excitations (section 5).

### 7.3 Structure preservation and stability guarantees
* *Gap:* plain least squares does not guarantee stability, energy
  conservation of the quadratic term, or the **power balance at the FSI
  interface** that VarFlExI enforces (work-conservative CRM/RBF transfer).
* *Suggestions:*
  1. **Energy-preserving / stable OpInf**: skew-symmetric `H`
     (physics-informed regularisation, Sawant–Kramer–Peherstorfer), or
     guaranteed-stable quadratic parametrisations (Goyal–Pontes–Benner).
  2. **Port-Hamiltonian OpInf** for the coupled system, with fluid and
     structure as two ports exchanging the interface power, mirroring
     VarFlExI's conservative transfer. It gives passivity, which composes
     with controllers.
  3. **Stable Koopman** (spectrally constrained `A`) for long-horizon EDMDc.

### 7.4 Reachability for this nonlinear system
* *Gap:* reachable sets of a ROM say nothing about the FOM unless the ROM error
  is bounded. Neither method provides such a bound by itself.
* *Suggestions:*
  1. **Koopman route (linear)**: EDMDc `(A,B,C)` with zonotope or ellipsoid
     reachability (CORA, JuliaReach), inflated by a **data-driven residual
     bound** on the lifting error (Bak et al. 2022; conformance checking).
  2. **OpInf route (polynomial)**: quadratic ROMs suit Taylor-model / polynomial
     zonotope reachability or SOS barrier certificates. Bound the projection plus
     inference error a posteriori (Lipschitz/logarithmic-norm estimates).
  3. **Calibrated uncertainty**: conformal prediction on validation runs gives
     distribution-free error tubes around ROM trajectories. These are cheap,
     honest and directly usable in **tube MPC** and probabilistic reachability.
     The `horizon_error` analysis already computes the needed n-step residuals.
  4. Reach targets should be the **outputs** (tip deflection, strain, lift
     limits), so reachability works in the output map's space.

### 7.5 Parametric and operational envelope
Vary V∞, AoA and stiffness (the existing `results_KG*` and `tipdisp_*` families
suggest such sweeps exist) and use parametric OpInf (`opinf` supports
interpolated/affine operators) or parameter-dependent Koopman, so that one ROM
covers the flight envelope used for control design.

**Suggested next step after the first real results:** add the `circulation`
block and actuated runs, compare discrete OpInf vs EDMDc (with delays) on
gust→lift Bode plots, then implement an energy-preserving or stable OpInf
variant and a residual-inflated EDMDc reachability study on tip deflection.

## 8. Repository layout, installation and tests

```
rom/                         shared core (package)
├── varflexi/                run_reader.py (raw folders, freestream, alignment), signals.py (inputs/outputs)
├── eulerian/                kernels, biot_savart (torch/numpy, lumping), vic, bound_vortex, reconstruct,
│                            julia_bridge, xdmf, grid, masking, vorticity (FD curl)
├── data/                    trajectory_file.py (streaming writer / lazy reader)
├── reduced/                 streaming_pod, dataset, state (assembler, output map), evaluation
├── viz/                     style (validated colour-blind-safe palette), plots, fields
└── testing/synthetic.py     VarFlExI-like fake runs for tests/demo
rom_opinf/                   Operator Inference method      (README inside)
rom_koopman/                 Koopman lifting + EDMDc method (README inside)
scripts/                     reconstruct_eulerian, convert_julia_fluiddomain, build_reduced_dataset,
                             compare_methods, run_synthetic_demo
configs/                     reconstruction(.yaml|_coarse.yaml), reduced_dataset.yaml
tests/                       reconstruction physics, reader/POD/exact errors, OpInf/Koopman, end-to-end
```

Requirements: Python ≥ 3.10, numpy, scipy, h5py, matplotlib, pandas, pyyaml,
`opinf==0.6.0`, and optionally `torch` (GPU reconstruction).
`python -m pytest -q` runs everything; `-m "not slow"` skips the ~7-min end-to-end test.

## 9. References

* Peherstorfer & Willcox, *Data-driven operator inference for nonintrusive projection-based model reduction*, CMAME 2016; Kramer, Peherstorfer & Willcox, *Learning nonlinear reduced models from data with operator inference*, Annu. Rev. Fluid Mech. 2024.
* McQuarrie, Huang & Willcox, *Data-driven reduced-order models via regularised operator inference for a single-injector combustion process*, J. R. Soc. N.Z. 2021.
* Mauroy & Gonçalves, *Linear identification of nonlinear systems: a lifting technique based on the Koopman operator*, CDC 2016; *Koopman-based lifting techniques for nonlinear systems identification*, IEEE TAC 2020.
* Korda & Mezić, *Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control*, Automatica 2018.
* Rowley, *Model reduction for fluids using balanced proper orthogonal decomposition*, IJBC 2005.
* Bak, Bogomolov, Hencey, Kochdumper, Lew, Potomkin, *Reachability of Koopman linearized systems using random Fourier feature observables and polynomial zonotope refinement*, CAV 2022.
* Alvarez & Ning, reformulated vortex particle method (rVPM) and FLOWUnsteady/FLOWVPM — see the FLOWUnsteady documentation (flow.byu.edu/FLOWUnsteady) for the papers.
