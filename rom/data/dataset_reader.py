"""
rom.data.dataset_reader – Flexible HDF5 reader for independently generated VarFlExI datasets.

Inspects HDF5 group structures and metadata dynamically without assuming fixed
dimensions. Supports both single-trajectory and multi-trajectory containers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from rom.data.trajectory import Trajectory
from rom.data.validation import validate_trajectory
from rom.eulerian.grid import EulerianGrid
from rom.eulerian.masking import FluidMask


class DatasetReader:
    """Reader for VarFlExI HDF5 datasets."""

    @staticmethod
    def _read_grid_from_group(meta_grp: h5py.Group) -> EulerianGrid:
        """Parse an EulerianGrid from an HDF5 metadata group without assuming hardcoded shapes."""
        # Check if bounds are stored in attrs or datasets
        bounds = {}
        resolution = {}
        custom_coords = {}

        for ax in "xyz":
            coords_key = f"coords_{ax}"
            if coords_key in meta_grp:
                coords = np.array(meta_grp[coords_key], dtype=np.float64)
                custom_coords[ax] = coords
                bounds[ax] = (float(coords[0]), float(coords[-1]))
                resolution[ax] = len(coords)
            elif f"bounds_{ax}" in meta_grp.attrs and f"resolution_{ax}" in meta_grp.attrs:
                b = meta_grp.attrs[f"bounds_{ax}"]
                bounds[ax] = (float(b[0]), float(b[1]))
                resolution[ax] = int(meta_grp.attrs[f"resolution_{ax}"])
            else:
                raise KeyError(f"Unable to resolve grid coordinate information for axis '{ax}'")

        axis_order = str(meta_grp.attrs.get("axis_order", "xyz"))
        spacing_type = str(meta_grp.attrs.get("spacing_type", "uniform"))
        units = str(meta_grp.attrs.get("units", "m"))

        return EulerianGrid(
            bounds=bounds,
            resolution=resolution,
            axis_order=axis_order,
            spacing_type=spacing_type,
            custom_coords=custom_coords if custom_coords else None,
            units=units,
        )

    @classmethod
    def load_trajectory_from_group(
        cls,
        grp: Union[h5py.File, h5py.Group],
        name: str = "trajectory",
        validate: bool = True,
    ) -> Trajectory:
        """Load a single Trajectory from an open HDF5 group or file."""
        # 1. Locate metadata and grid
        if "metadata" in grp:
            meta_grp = grp["metadata"]
        else:
            meta_grp = grp

        grid = cls._read_grid_from_group(meta_grp)

        # 2. Extract time array
        time: Optional[np.ndarray] = None
        for key in ["time", "t", "times"]:
            if key in meta_grp:
                time = np.array(meta_grp[key], dtype=np.float64)
                break
            elif key in grp:
                time = np.array(grp[key], dtype=np.float64)
                break

        if time is None:
            raise KeyError(f"No temporal array ('time' or 't') found in group {grp.name}")

        # 3. Extract velocity snapshots
        snaps_grp = grp["snapshots"] if "snapshots" in grp else grp
        vel_key = None
        for key in ["velocity", "u", "U", "vel"]:
            if key in snaps_grp:
                vel_key = key
                break

        if vel_key is None:
            raise KeyError(f"No velocity dataset found in group {grp.name}")

        velocity = np.array(snaps_grp[vel_key], dtype=np.float64)

        # 4. Extract vorticity (optional)
        vorticity = None
        for key in ["vorticity", "omega", "w", "vort"]:
            if key in snaps_grp:
                vorticity = np.array(snaps_grp[key], dtype=np.float64)
                break

        # 5. Extract fluid mask (optional)
        fluid_mask = None
        for key in ["fluid_mask", "mask"]:
            if key in meta_grp:
                mask_arr = np.array(meta_grp[key], dtype=bool)
                fluid_mask = FluidMask(grid, mask_arr)
                break
            elif key in grp:
                mask_arr = np.array(grp[key], dtype=bool)
                fluid_mask = FluidMask(grid, mask_arr)
                break

        # 6. Extract structural data (optional)
        struct_grp = grp.get("structural", grp)
        disp = None
        s_vel = None
        s_acc = None
        aero_forces = None

        for k in ["displacement", "disp", "q_struct"]:
            if k in struct_grp:
                disp = np.array(struct_grp[k], dtype=np.float64)
                break

        for k in ["velocity", "struct_vel", "qdot_struct"]:
            if k in struct_grp and struct_grp != snaps_grp:
                s_vel = np.array(struct_grp[k], dtype=np.float64)
                break

        for k in ["acceleration", "acc", "qddot_struct"]:
            if k in struct_grp:
                s_acc = np.array(struct_grp[k], dtype=np.float64)
                break

        for k in ["aerodynamic_forces", "forces", "loads"]:
            if k in struct_grp:
                aero_forces = np.array(struct_grp[k], dtype=np.float64)
                break

        # 7. Extract parameters and attributes
        params = {}
        for attr_key, attr_val in grp.attrs.items():
            params[attr_key] = attr_val
        if "parameters" in grp and isinstance(grp["parameters"], h5py.Group):
            for attr_key, attr_val in grp["parameters"].attrs.items():
                params[attr_key] = attr_val

        # 8. Extract coupling data (optional)
        coupling = {}
        if "coupling" in grp and isinstance(grp["coupling"], h5py.Group):
            for k in grp["coupling"].keys():
                coupling[k] = np.array(grp["coupling"][k])
            for attr_key, attr_val in grp["coupling"].attrs.items():
                coupling[attr_key] = attr_val

        traj = Trajectory(
            time=time,
            velocity=velocity,
            grid=grid,
            vorticity=vorticity,
            fluid_mask=fluid_mask,
            structural_displacement=disp,
            structural_velocity=s_vel,
            structural_acceleration=s_acc,
            aerodynamic_forces=aero_forces,
            parameters=params,
            coupling_data=coupling,
            name=name,
        )

        if validate:
            rep = validate_trajectory(traj)
            if not rep.is_valid:
                raise ValueError(
                    f"Dataset validation failed for trajectory '{name}':\n"
                    + "\n".join(f"  - {e}" for e in rep.errors)
                )

        return traj

    @classmethod
    def load_file(
        cls,
        filepath: Union[str, Path],
        validate: bool = True,
    ) -> List[Trajectory]:
        """Load one or more trajectories from an HDF5 file.

        Automatically detects whether the file contains a single trajectory
        or multiple trajectories under grouped namespaces.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        trajectories: List[Trajectory] = []

        with h5py.File(filepath, "r") as f:
            # Check if root contains sub-trajectories
            traj_subgroups = [
                k for k in f.keys()
                if isinstance(f[k], h5py.Group) and (k.startswith("trajectory") or k.startswith("run") or k.startswith("sim"))
            ]

            if traj_subgroups:
                for grp_name in sorted(traj_subgroups):
                    traj = cls.load_trajectory_from_group(f[grp_name], name=grp_name, validate=validate)
                    trajectories.append(traj)
            else:
                # Single trajectory at root level
                name = filepath.stem
                traj = cls.load_trajectory_from_group(f, name=name, validate=validate)
                trajectories.append(traj)

        return trajectories

    @classmethod
    def load_directory(
        cls,
        dirpath: Union[str, Path],
        pattern: str = "*.h5",
        validate: bool = True,
    ) -> List[Trajectory]:
        """Load all matching HDF5 dataset files from a directory."""
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Directory not found: {dirpath}")

        files = sorted(dirpath.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {dirpath}")

        all_trajectories: List[Trajectory] = []
        for fp in files:
            trajs = cls.load_file(fp, validate=validate)
            all_trajectories.extend(trajs)

        return all_trajectories

    @classmethod
    def save_trajectory_to_hdf5(
        cls,
        trajectory: Trajectory,
        filepath: Union[str, Path],
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Export a Trajectory to the standard VarFlExI HDF5 format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        k = len(trajectory.time)
        spatial_shape = trajectory.grid.shape
        field_shape = spatial_shape + (3,)

        with h5py.File(filepath, "w") as f:
            # 1. Metadata
            meta = f.create_group("metadata")
            trajectory.grid.write_metadata(meta)
            meta.create_dataset("time", data=trajectory.time)

            if trajectory.fluid_mask is not None:
                trajectory.fluid_mask.write_hdf5(meta)
                meta.attrs["has_fluid_mask"] = True

            # 2. Snapshots
            snaps = f.create_group("snapshots")
            snaps.create_dataset(
                "velocity",
                data=trajectory.velocity,
                chunks=(1,) + field_shape,
                compression=compression,
                compression_opts=compression_opts,
            )

            if trajectory.vorticity is not None:
                snaps.create_dataset(
                    "vorticity",
                    data=trajectory.vorticity,
                    chunks=(1,) + field_shape,
                    compression=compression,
                    compression_opts=compression_opts,
                )

            # 3. Structural data (if present)
            if any(
                arr is not None for arr in [
                    trajectory.structural_displacement,
                    trajectory.structural_velocity,
                    trajectory.structural_acceleration,
                    trajectory.aerodynamic_forces,
                ]
            ):
                struct = f.create_group("structural")
                if trajectory.structural_displacement is not None:
                    struct.create_dataset("displacement", data=trajectory.structural_displacement)
                if trajectory.structural_velocity is not None:
                    struct.create_dataset("velocity", data=trajectory.structural_velocity)
                if trajectory.structural_acceleration is not None:
                    struct.create_dataset("acceleration", data=trajectory.structural_acceleration)
                if trajectory.aerodynamic_forces is not None:
                    struct.create_dataset("aerodynamic_forces", data=trajectory.aerodynamic_forces)

            # 4. Parameters
            if trajectory.parameters:
                params_grp = f.create_group("parameters")
                for pk, pv in trajectory.parameters.items():
                    try:
                        params_grp.attrs[pk] = pv
                    except (TypeError, ValueError):
                        pass

            # 5. Coupling data
            if trajectory.coupling_data:
                coup_grp = f.create_group("coupling")
                for ck, cv in trajectory.coupling_data.items():
                    if isinstance(cv, np.ndarray):
                        coup_grp.create_dataset(ck, data=cv)
                    else:
                        try:
                            coup_grp.attrs[ck] = cv
                        except (TypeError, ValueError):
                            pass
