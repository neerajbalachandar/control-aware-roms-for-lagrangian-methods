"""
rom.opinf.persistence – Standalone ROM serialization, loading, and inference interface.

Persists trained OpInf models and spatial encoders into a self-contained bundle:
  rom_model/
      metadata.json
      mean_field.npz
      pod_basis.npz
      operators.h5
      grid_metadata.npz
      preprocessing.npz

The saved model can be loaded and executed for reduced rollout, velocity field decoding,
and vorticity recovery completely independently of VarFlExI, FLOWUnsteady, or Julia.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import opinf

from rom.basis.pod import PODReducer
from rom.basis.state_encoder import VelocityStateEncoder
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask
from rom.eulerian.vorticity import compute_vorticity
from rom.opinf.model_factory import extract_operator_entries
from rom.opinf.predict import predict_continuous, predict_discrete


class TrainedROM:
    """Encapsulates a fully trained OpInf model, spatial basis, and grid reconstruction.

    Parameters
    ----------
    model : opinf.models.ContinuousModel or opinf.models.DiscreteModel
    encoder : VelocityStateEncoder
    reducer : PODReducer
    metadata : dict, optional
    """

    def __init__(
        self,
        model: Union[opinf.models.ContinuousModel, opinf.models.DiscreteModel],
        encoder: VelocityStateEncoder,
        reducer: PODReducer,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.encoder = encoder
        self.reducer = reducer
        self.metadata = dict(metadata) if metadata else {}

        # Determine model type
        if isinstance(model, opinf.models.ContinuousModel):
            self.model_type = "continuous"
        elif isinstance(model, opinf.models.DiscreteModel):
            self.model_type = "discrete"
        else:
            self.model_type = getattr(model, "model_type", "unknown")

    @property
    def r(self) -> int:
        return self.reducer.r

    @property
    def n(self) -> int:
        return self.encoder.state_dimension

    @property
    def grid(self) -> EulerianGrid:
        return self.encoder.grid

    # ------------------------------------------------------------------------
    # Prediction & Field Decoding
    # ------------------------------------------------------------------------

    def predict(
        self,
        a0: np.ndarray,
        time_domain_or_steps: Union[np.ndarray, int],
        **options,
    ) -> np.ndarray:
        """Simulate autonomous reduced state trajectory.

        Parameters
        ----------
        a0 : np.ndarray
            Initial reduced state vector of shape ``(r,)``.
        time_domain_or_steps : np.ndarray or int
            If continuous: 1-D array of timestamps ``(K,)``.
            If discrete: integer number of iteration steps K.

        Returns
        -------
        a_pred : np.ndarray
            Shape ``(r, K)``.
        """
        if self.model_type == "continuous":
            if not isinstance(time_domain_or_steps, np.ndarray):
                raise TypeError("Continuous model requires a 1-D time_domain array.")
            return predict_continuous(self.model, a0, time_domain_or_steps, **options)
        elif self.model_type == "discrete":
            if not isinstance(time_domain_or_steps, (int, np.integer)):
                raise TypeError("Discrete model requires integer number of steps.")
            return predict_discrete(self.model, a0, int(time_domain_or_steps))
        else:
            raise RuntimeError(f"Unsupported model type '{self.model_type}'")

    def decode_velocity(self, a: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
        """Decode reduced coordinates a into 3-D Eulerian velocity fields.

        Parameters
        ----------
        a : np.ndarray
            Reduced state vector of shape ``(r,)`` or matrix ``(r, K)``.
        fill_value : float, optional
            Fill value for solid (inactive) cells (default: np.nan).

        Returns
        -------
        velocity : np.ndarray
            Shape ``(N0, N1, N2, 3)`` if a is 1-D, or ``(K, N0, N1, N2, 3)`` if a is 2-D.
        """
        is_single = a.ndim == 1
        a_2d = a[:, np.newaxis] if is_single else a

        # Reconstruct centered full state: Q_rec = V_r a
        Q_centered = self.reducer.reconstruct(a_2d)

        # Uncenter with mean field: Q = Q_centered + mean
        Q = self.encoder.uncenter(Q_centered)

        # Decode to 3D grid
        decoded = self.encoder.decode_trajectory(Q, fill_value=fill_value)

        return decoded[0] if is_single else decoded

    def recover_vorticity(
        self,
        velocity: np.ndarray,
        scheme: str = "central",
        order: int = 2,
    ) -> np.ndarray:
        """Compute vorticity fields omega = curl(u) from decoded velocity fields.

        Parameters
        ----------
        velocity : np.ndarray
            Shape ``(N0, N1, N2, 3)`` or ``(K, N0, N1, N2, 3)``.
        scheme : str, optional
            Finite difference scheme (default: 'central').
        order : int, optional
            Accuracy order (default: 2).

        Returns
        -------
        vorticity : np.ndarray
            Vorticity with the same shape as velocity.
        """
        grid = self.encoder.grid
        mask = self.encoder.fluid_mask

        if velocity.ndim == 4:
            return compute_vorticity(velocity, grid, scheme=scheme, order=order, fluid_mask=mask)
        elif velocity.ndim == 5:
            k = velocity.shape[0]
            vort = np.empty_like(velocity)
            for j in range(k):
                vort[j] = compute_vorticity(velocity[j], grid, scheme=scheme, order=order, fluid_mask=mask)
            return vort
        else:
            raise ValueError(f"Expected 4-D or 5-D velocity array, got shape {velocity.shape}")

    # ------------------------------------------------------------------------
    # Persistence (Save & Load)
    # ------------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> Path:
        """Save the ROM bundle to the specified directory.

        Creates:
          - metadata.json
          - mean_field.npz
          - pod_basis.npz
          - operators.h5
          - grid_metadata.npz
          - preprocessing.npz
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 1. metadata.json
        inv_abbrev = {
            opinf.operators.ConstantOperator: "c",
            opinf.operators.LinearOperator: "A",
            opinf.operators.QuadraticOperator: "H",
            opinf.operators.CubicOperator: "G",
            opinf.operators.InputOperator: "B",
            opinf.operators.StateInputOperator: "N",
        }
        op_str = (
            "".join(inv_abbrev.get(type(op), op.__class__.__name__) for op in self.model.operators)
            if hasattr(self.model, "operators")
            else "cAH"
        )
        meta = {
            "model_type": self.model_type,
            "operators": op_str,
            "r": self.r,
            "n": self.n,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "custom_metadata": self.metadata,
        }
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 2. mean_field.npz
        if self.encoder.mean_field is not None:
            np.savez_compressed(
                dir_path / "mean_field.npz",
                mean_field=self.encoder.mean_field,
            )
        else:
            np.savez_compressed(
                dir_path / "mean_field.npz",
                mean_field=np.zeros(self.n, dtype=np.float64),
            )

        # 3. pod_basis.npz
        np.savez_compressed(
            dir_path / "pod_basis.npz",
            basis_vectors=self.reducer.basis_matrix,
            singular_values=self.reducer.singular_values,
            cumulative_energy=self.reducer.cumulative_energy,
            residual_energy=self.reducer.residual_energy,
        )

        # 4. operators.h5 (Native OpInf format)
        op_path = dir_path / "operators.h5"
        self.model.save(str(op_path), overwrite=True)

        # 5. grid_metadata.npz
        g = self.encoder.grid
        grid_dict = {
            "axis_order": g.axis_order,
            "spacing_type": g.spacing_type,
            "units": g.units,
            "coords_x": g.coords_1d("x"),
            "coords_y": g.coords_1d("y"),
            "coords_z": g.coords_1d("z"),
            "has_fluid_mask": self.encoder.fluid_mask is not None,
        }
        for ax in "xyz":
            grid_dict[f"bounds_{ax}"] = np.array(g.bounds[ax])
            grid_dict[f"resolution_{ax}"] = np.array(g.resolution[ax])

        if self.encoder.fluid_mask is not None:
            grid_dict["fluid_mask"] = self.encoder.fluid_mask.mask

        np.savez_compressed(dir_path / "grid_metadata.npz", **grid_dict)

        # 6. preprocessing.npz
        np.savez_compressed(
            dir_path / "preprocessing.npz",
            state_dimension=self.n,
            n_fluid_cells=self.encoder.n_fluid_cells,
            active_mask=self.encoder.active_mask,
        )

        return dir_path

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "TrainedROM":
        """Load a persisted TrainedROM bundle from disk."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        # 1. metadata.json
        with open(dir_path / "metadata.json", "r") as f:
            meta = json.load(f)

        model_type = meta.get("model_type", "continuous")

        # 2. grid_metadata.npz
        g_data = np.load(dir_path / "grid_metadata.npz")
        bounds = {ax: tuple(g_data[f"bounds_{ax}"]) for ax in "xyz"}
        resolution = {ax: int(g_data[f"resolution_{ax}"]) for ax in "xyz"}
        custom_coords = {ax: g_data[f"coords_{ax}"] for ax in "xyz"}
        axis_order = str(g_data["axis_order"])
        spacing_type = str(g_data["spacing_type"])
        units = str(g_data["units"])

        grid = EulerianGrid(
            bounds=bounds,
            resolution=resolution,
            axis_order=axis_order,
            spacing_type=spacing_type,
            custom_coords=custom_coords,
            units=units,
        )

        fluid_mask = None
        if bool(g_data.get("has_fluid_mask", False)) and "fluid_mask" in g_data:
            fluid_mask = FluidMask(grid, g_data["fluid_mask"])

        # 3. Spatial Encoder
        encoder = VelocityStateEncoder(grid, fluid_mask)

        # 4. mean_field.npz
        mean_data = np.load(dir_path / "mean_field.npz")
        encoder.set_mean(mean_data["mean_field"])

        # 5. pod_basis.npz
        basis_data = np.load(dir_path / "pod_basis.npz")
        basis_vectors = basis_data["basis_vectors"]
        r = basis_vectors.shape[1]

        # Reconstruct PODReducer
        reducer = PODReducer(r=r)
        # Directly initialize basis entries and svdvals
        reducer._basis._LinearBasis__entries = basis_vectors
        reducer._basis._LinearBasis__weights = None
        reducer._basis._BasisTemplate__n = basis_vectors.shape[0]
        reducer._basis._BasisTemplate__r = r
        reducer._basis._PODBasis__svdvals = basis_data["singular_values"]
        reducer._basis._PODBasis__cumulative_energy = float(basis_data["cumulative_energy"])
        reducer._basis._PODBasis__residual_energy = float(basis_data["residual_energy"])
        reducer._is_fitted = True

        # 6. operators.h5
        op_path = dir_path / "operators.h5"
        if model_type == "continuous":
            model = opinf.models.ContinuousModel.load(str(op_path))
        elif model_type == "discrete":
            model = opinf.models.DiscreteModel.load(str(op_path))
        else:
            raise ValueError(f"Unknown model_type '{model_type}' in metadata")

        return cls(
            model=model,
            encoder=encoder,
            reducer=reducer,
            metadata=meta.get("custom_metadata", {}),
        )

    def check_equivalence(
        self,
        other: "TrainedROM",
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> bool:
        """Verify numerical equivalence between this ROM and another instance."""
        if self.model_type != other.model_type:
            return False
        if self.r != other.r or self.n != other.n:
            return False

        # Compare operators
        self_ops = extract_operator_entries(self.model)
        other_ops = extract_operator_entries(other.model)

        if set(self_ops.keys()) != set(other_ops.keys()):
            return False

        for k in self_ops:
            if not np.allclose(self_ops[k], other_ops[k], rtol=rtol, atol=atol):
                return False

        # Compare basis matrices
        if not np.allclose(self.reducer.basis_matrix, other.reducer.basis_matrix, rtol=rtol, atol=atol):
            return False

        # Compare mean fields
        if not np.allclose(self.encoder.mean_field, other.encoder.mean_field, rtol=rtol, atol=atol):
            return False

        # Compare grid coords
        for ax in "xyz":
            if not np.allclose(self.grid.coords_1d(ax), other.grid.coords_1d(ax), rtol=rtol, atol=atol):
                return False

        return True
