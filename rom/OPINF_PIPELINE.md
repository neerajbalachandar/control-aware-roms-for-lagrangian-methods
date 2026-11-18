# VarFlExI Dataset-to-ROM Pipeline Documentation

This module implements an end-to-end Reduced-Order Modeling (ROM) pipeline based on Proper Orthogonal Decomposition (POD) and Operator Inference (OpInf) using the upstream `opinf` library (v0.6.0).

---

## Architectural Decision

- **Simulation & Eulerian Field Evaluation:** Happens externally in VarFlExI / FLOWUnsteady (in Julia) using native APIs such as `uns.computefluiddomain(pfield, grids)`.
- **ROM Pipeline:** This repository strictly acts as a downstream consumer of exported HDF5 datasets. It performs:
  1. Dataset ingestion & validation.
  2. State preparation & POD projection.
  3. OpInf model training (continuous & discrete) with Tikhonov regularization sweeps.
  4. Model evaluation & dynamical rollout assessment.
  5. Standalone model persistence & field reconstruction.
- **Zero Simulation Dependencies at Inference Time:** The trained ROM requires **no** Julia, VarFlExI, or FLOWUnsteady runtime dependencies.

---

## Pipeline Architecture

```
rom/
├── data/
│   ├── dataset_reader.py      # Flexible HDF5 reader (single & multi-trajectory)
│   ├── trajectory.py          # In-memory trajectory abstraction (flow + aeroelastic)
│   └── validation.py          # Data validation (time monotonicity, grid match, NaNs)
├── basis/
│   ├── state_encoder.py       # Active fluid-domain mapping & mean centering
│   ├── pod.py                 # PODBasis wrapper (dense, method-of-snapshots, randomized)
│   └── rank_selection.py      # Singular value decay, energy retention, rank recommendation
├── opinf/
│   ├── model_factory.py       # Continuous & discrete OpInf model builder
│   ├── derivatives.py         # Finite differences with trajectory boundary isolation
│   ├── train_continuous.py    # Autonomous continuous OpInf training (da/dt = c + Aa + H(a⊗a))
│   ├── train_discrete.py      # Autonomous discrete OpInf training (a_{k+1} = c + Aa_k + H(a_k⊗a_k))
│   ├── predict.py             # Rollout simulation & numerical stability checks
│   └── persistence.py         # Standalone ROM persistence & reconstruction interface
└── eulerian/
    ├── grid.py                # Fixed 3D Cartesian EulerianGrid with metadata IO
    ├── masking.py             # FluidMask for immersed solid boundaries
    └── vorticity.py           # Numerical curl (omega = curl(u)) for vorticity recovery

scripts/
├── train_opinf.py             # Training pipeline with rank & regularizer sweeps
├── evaluate_opinf.py          # Projection vs dynamical rollout evaluation
└── reconstruct_rom_fields.py  # 3D velocity & vorticity field reconstruction to HDF5
```

---

## Model Persistence Bundle (`rom_model/`)

A trained ROM is saved as a completely self-contained folder containing:
1. `metadata.json`: Model type, operator string (`cAH`), rank $r$, full state dimension $n$, regularizer, and training summary.
2. `mean_field.npz`: Mean velocity vector $\bar{\mathbf{q}} \in \mathbb{R}^n$.
3. `pod_basis.npz`: Spatial POD basis matrix $\mathbf{V}_r \in \mathbb{R}^{n \times r}$, singular values, and cumulative energy.
4. `operators.h5`: Saved OpInf model operators ($c, A, H$) in native HDF5 format.
5. `grid_metadata.npz`: Eulerian grid bounds, resolutions, coordinate arrays, and fluid mask.
6. `preprocessing.npz`: Active fluid indices and mask mapping.
7. `training_metrics.json`: Full rank evaluation table and regularizer sweep results.

---

## Exact Commands to Train and Evaluate the ROM

### 1. Run Test Suite
```bash
python3 -m pytest rom/eulerian/tests tests/ -v
```

### 2. Train an Autonomous Continuous OpInf Model
```bash
python3 scripts/train_opinf.py \
    --data path/to/varflexi_simulation.h5 \
    --model-type continuous \
    --operators cAH \
    --ranks 2 4 6 8 10 15 20 \
    --energy-target 0.999 \
    --regularizers 0.0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1.0 \
    --train-fraction 0.7 \
    --val-fraction 0.15 \
    --output-dir rom_model
```

### 3. Train an Autonomous Discrete OpInf Model
```bash
python3 scripts/train_opinf.py \
    --data path/to/varflexi_simulation.h5 \
    --model-type discrete \
    --operators cAH \
    --ranks 2 4 6 8 10 \
    --regularizers 1e-6 1e-5 1e-4 1e-3 \
    --output-dir rom_model_discrete
```

### 4. Evaluate the Trained Model
```bash
python3 scripts/evaluate_opinf.py \
    --rom-dir rom_model \
    --data path/to/test_simulation.h5 \
    --output-report evaluation_report.json
```

### 5. Reconstruct 3D Eulerian Fields
```bash
python3 scripts/reconstruct_rom_fields.py \
    --rom-dir rom_model \
    --data path/to/test_simulation.h5 \
    --output-h5 reconstructed_eulerian_fields.h5
```
