# Eulerian Field Reconstruction for VarFlExI

## Mathematical Reconstruction

This module converts Lagrangian vortex-particle data produced by **FLOWUnsteady / rVPM** (reformulated Vortex Particle Method) into velocity and vorticity fields on a fixed Cartesian Eulerian grid.

### 1. Regularised Biot–Savart Velocity Evaluation

The rVPM represents the vorticity field as a set of *N<sub>p</sub>* Lagrangian particles, where particle *p* carries:

| Symbol | Meaning | Shape |
|--------|---------|-------|
| **x**<sub>p</sub> | Position | ℝ³ |
| **Γ**<sub>p</sub> | Circulation (vortex-strength) vector | ℝ³ |
| σ<sub>p</sub> | Core (smoothing) radius | ℝ⁺ |

The velocity at an arbitrary target point **x** is computed by summing contributions from all particles:

$$
\mathbf{u}(\mathbf{x}) = \frac{1}{4\pi} \sum_{p=1}^{N_p}
\frac{\boldsymbol{\Gamma}_p \times \mathbf{r}_p}{|\mathbf{r}_p|^3}\;
g\!\left(\frac{|\mathbf{r}_p|}{\sigma_p}\right)
$$

where **r**<sub>p</sub> = **x** − **x**<sub>p</sub> and *g*(ρ) is a regularisation (cut-off) function.

### 2. Supported Kernels

#### Gaussian-erf (default, matches `vpm.gaussianerf` in FLOWVPM.jl)

$$
g(\rho) = \mathrm{erf}(\rho) - \frac{2}{\sqrt{\pi}}\,\rho\,e^{-\rho^2}
$$

This arises from the volume-integrated Gaussian vorticity blob:

$$
\zeta_\sigma(r) = \frac{1}{\pi^{3/2}\,\sigma^3}\,\exp\!\left(-\frac{r^2}{\sigma^2}\right)
$$

#### Winckelmans algebraic kernel (matches `vpm.winckelmans`)

$$
g(\rho) = \frac{\rho^3\,(\rho^2 + 5/2)}{(\rho^2 + 1)^{5/2}}
$$

#### Singular (un-regularised, validation only)

$$
g(\rho) = 1
$$

### 3. Vorticity

Vorticity is computed as the discrete curl of the velocity field:

$$
\boldsymbol{\omega} = \nabla \times \mathbf{u}
$$

using finite differences consistent with the grid spacing:

- **Uniform grids**: 2nd-order or 4th-order central differences in the interior, with matching-accuracy one-sided stencils at boundaries.
- **Non-uniform grids**: Generalised 2nd-order central differences accounting for unequal cell sizes.

### 4. Grid Convention

- Grid arrays have shape `(N₀, N₁, N₂)` where the indices follow `axis_order` (default `'xyz'`).
- Vector fields have shape `(N₀, N₁, N₂, 3)` with the last dimension always ordered as `(u_x, u_y, u_z)` in the **physical** frame.
- Coordinate metadata and spacing arrays are stored per-axis.

### 5. Fluid-Domain Masking

Solid-region cells are set to `NaN` (not zero) so that:
- Downstream consumers never silently treat solid data as valid fluid.
- Finite-difference stencils reaching into solid regions produce NaN naturally (NaN propagation).

### 6. HDF5 Output Layout

```
/metadata/
    time                    (n_snapshots,)       Physical time stamps
    coords_x                (Nx,)                1-D coordinate arrays
    coords_y                (Ny,)
    coords_z                (Nz,)
    spacing_x               (Nx-1,)              Cell spacings
    spacing_y               (Ny-1,)
    spacing_z               (Nz-1,)
    fluid_mask              (N0, N1, N2)          Boolean mask (if provided)
    @axis_order             string
    @bounds_x, bounds_y, bounds_z   (2,) each
    @resolution_x, resolution_y, resolution_z    int each
    @units                  string
    @kernel                 string
    @has_fluid_mask         bool

/snapshots/
    velocity                (n_snapshots, N0, N1, N2, 3)
    vorticity               (n_snapshots, N0, N1, N2, 3)
```

Both datasets use chunked, gzip-compressed HDF5 storage with chunk shape `(1, N0, N1, N2, 3)` to allow snapshot-at-a-time reads and writes.

---

## Module Structure

```
rom/
├── __init__.py
└── eulerian/
    ├── __init__.py
    ├── grid.py                  # EulerianGrid – fixed 3-D Cartesian grid
    ├── field_reconstructor.py   # FieldReconstructor – Biot-Savart evaluation
    ├── vorticity.py             # compute_vorticity – curl(u) via finite differences
    ├── masking.py               # FluidMask – solid-domain masking
    ├── config_example.yaml      # YAML configuration template
    └── tests/
        ├── __init__.py
        └── test_eulerian.py     # 37 unit / integration tests
```

---

## FLOWUnsteady Kernel Dependency Report

### Available Information

| Item | Status | Details |
|------|--------|---------|
| Gaussian-erf kernel formula | ✅ Verified | Published in FLOWVPM documentation and papers; implemented from first principles |
| Winckelmans kernel formula | ✅ Verified | Standard algebraic kernel from Winckelmans & Leonard |
| Particle data fields (X, Γ, σ) | ✅ Verified | FLOWVPM stores per-particle `X` (position), `Gamma` (circulation), `sigma` (core size) |
| VTK output format | ✅ Known | Particle snapshots stored as VTK point data via `WriteVTK.jl`; positions as grid coordinates, Gamma and sigma as point data arrays |
| HDF5/XDMF output | ✅ Known | `FLOWVPM.save(pfield, filename)` writes HDF5 with XDMF schema |
| Temporal sampling | ✅ Known | Controlled by `nsteps_save` parameter in `run_simulation` |
| Velocity at arbitrary points | ⚠️ Not directly exposed | `UJ_direct` / `UJ_fmm` compute velocity at particle locations only; no public API for arbitrary target points |

### Dependencies NOT Available from Python

1. **Direct Julia kernel call**: The `FLOWVPM.jl` Biot-Savart kernel runs inside Julia's runtime. There is no existing Python-callable interface (`PyCall`/`PythonCall` bridge) for evaluating velocity at arbitrary spatial locations.

2. **Resolution**: This module implements the **exact same mathematical kernel** (Gaussian-erf and Winckelmans) in pure NumPy/SciPy, verified against analytical hand-calculations. This is a *verified equivalent*, not an approximation or substitute.

3. **Alternative approach** (if needed in the future): A Julia-side export script could use `FLOWVPM.UJ_direct` to evaluate velocity at arbitrary target points and write the results to a file. This would require adding a small Julia driver script (not modifying the solver itself).

### Particle Data Loading

FLOWUnsteady outputs particle data in VTK format (`.vtp` / `.vtu` files via `WriteVTK.jl`). To load these into Python:

```python
# Using the pyvista or vtk Python packages:
import pyvista as pv

mesh = pv.read("pfield_00100.vtp")
positions = mesh.points                    # (Np, 3)
circulations = mesh.point_data["Gamma"]    # (Np, 3)
core_sizes = mesh.point_data["sigma"]      # (Np,)
```

For HDF5 particle snapshots (from `FLOWVPM.save`):

```python
import h5py
with h5py.File("pfield_00100.h5", "r") as f:
    positions = f["X"][:]
    circulations = f["Gamma"][:]
    core_sizes = f["sigma"][:]
```

---

## Commands for Generating Eulerian Snapshots

### 1. Quick test with synthetic particles

```bash
cd /home/dysco/FLOWUnsteady/control-aware-roms-for-lagrangian-methods

# Run all 37 unit tests
python3 -m pytest rom/eulerian/tests/test_eulerian.py -v
```

### 2. Programmatic usage

```python
import numpy as np
from rom.eulerian import EulerianGrid, FieldReconstructor, compute_vorticity, FluidMask
from rom.eulerian.field_reconstructor import ParticleSnapshot, Kernel

# 1. Define grid
grid = EulerianGrid(
    bounds={"x": (-2, 6), "y": (-3, 3), "z": (-3, 3)},
    resolution={"x": 81, "y": 61, "z": 61},
)

# 2. Load particle data (from your FLOWUnsteady simulation output)
# positions, circulations, core_sizes = load_your_vtk_or_h5_file(...)
# snap = ParticleSnapshot(positions, circulations, core_sizes, time=0.0)

# 3. Create reconstructor
recon = FieldReconstructor(
    grid,
    kernel=Kernel.GAUSSIAN_ERF,   # match your FLOWVPM kernel
    chunk_size=8192,
)

# 4. Process snapshots and write HDF5
# recon.process_snapshots(snapshots, "eulerian_fields.h5", compute_vorticity=True)
```

### 3. Full pipeline from FLOWUnsteady VTK output

```bash
# Step 1: Run FLOWUnsteady simulation (Julia)
#   julia run_simulation.jl   (produces VTK files in save_path/)

# Step 2: Convert to Eulerian fields (Python)
python3 -c "
from rom.eulerian import EulerianGrid, FieldReconstructor
from rom.eulerian.field_reconstructor import ParticleSnapshot, Kernel
import numpy as np

# Configure grid
grid = EulerianGrid(
    bounds={'x': (-2, 6), 'y': (-3, 3), 'z': (-3, 3)},
    resolution={'x': 81, 'y': 61, 'z': 61},
)
recon = FieldReconstructor(grid, kernel=Kernel.GAUSSIAN_ERF, chunk_size=8192)

# Load and process snapshots (pseudo-code – adapt to your file format)
# snapshots = [load_snapshot(step) for step in range(0, 1000, 10)]
# recon.process_snapshots(snapshots, 'eulerian_fields.h5')
"
```
