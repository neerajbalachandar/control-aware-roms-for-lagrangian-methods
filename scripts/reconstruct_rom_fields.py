#!/usr/bin/env python3
"""
scripts/reconstruct_rom_fields.py – Reconstruct 3D Eulerian velocity and vorticity fields.

Workflow:
  1. Loads a standalone TrainedROM bundle.
  2. Extracts an initial condition from a simulation trajectory (or zero/mean state).
  3. Simulates the reduced-order dynamics forward in time.
  4. Decodes the full 3D Cartesian velocity fields on the Eulerian grid.
  5. Computes vorticity fields via numerical curl.
  6. Exports a standard chunked HDF5 dataset.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rom.data import DatasetReader
from rom.opinf import TrainedROM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D Eulerian velocity and vorticity fields from a trained ROM."
    )
    parser.add_argument(
        "--rom-dir",
        type=str,
        default="rom_model",
        help="Path to trained ROM bundle directory (default: 'rom_model').",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to trajectory HDF5 file supplying initial condition and time domain.",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="rom_reconstructed_fields.h5",
        help="Output HDF5 path (default: 'rom_reconstructed_fields.h5').",
    )
    parser.add_argument(
        "--no-vorticity",
        action="store_true",
        help="Skip vorticity curl evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print("VARFLEXI OPERATOR INFERENCE PIPELINE – FIELD RECONSTRUCTION")
    print("=" * 78)

    t0 = time.perf_counter()

    # 1. Load Standalone ROM
    rom_dir = Path(args.rom_dir)
    print(f"Loading ROM bundle from '{rom_dir}'...")
    rom = TrainedROM.load(rom_dir)
    print(f"  Model Type:        {rom.model_type}")
    print(f"  Reduced Rank (r):  {rom.r}")
    print(f"  Grid Shape:        {rom.grid.shape}")

    # 2. Load Initial Condition from Reference Trajectory
    data_path = Path(args.data)
    trajs = DatasetReader.load_file(data_path, validate=True)
    ref_traj = trajs[0]
    print(f"Using initial condition from trajectory '{ref_traj.name}' ({len(ref_traj)} snapshots)")

    # Project initial snapshot to reduced coordinates: a0 = V_r^T (q_0 - mean)
    q0 = rom.encoder.encode_snapshot(ref_traj.velocity[0])
    q0_cent = rom.encoder.center(q0)
    a0 = rom.reducer.project(q0_cent)

    # 3. Simulate Reduced Dynamics
    k = len(ref_traj)
    print(f"Simulating forward for {k} time steps...")
    if rom.model_type == "continuous":
        a_pred = rom.predict(a0, ref_traj.time)
    else:
        a_pred = rom.predict(a0, k)

    # 4. Decode 3D Velocity Fields
    print("Decoding 3D Cartesian velocity fields onto Eulerian grid...")
    vel_decoded = rom.decode_velocity(a_pred)  # (K, N0, N1, N2, 3)

    # 5. Recover Vorticity Fields via Discrete Curl
    vort_decoded = None
    if not args.no_vorticity:
        print("Recovering 3D vorticity fields omega = curl(u)...")
        vort_decoded = rom.recover_vorticity(vel_decoded, scheme="central", order=2)

    # 6. Save Chunked HDF5 Output
    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing reconstructed fields to '{out_path}'...")

    field_shape = rom.grid.shape + (3,)
    with h5py.File(out_path, "w") as f:
        meta = f.create_group("metadata")
        rom.grid.write_metadata(meta)
        meta.create_dataset("time", data=ref_traj.time)
        meta.attrs["source_rom"] = str(rom_dir)
        meta.attrs["model_type"] = rom.model_type
        meta.attrs["reduced_rank"] = rom.r

        if rom.encoder.fluid_mask is not None:
            rom.encoder.fluid_mask.write_hdf5(meta)
            meta.attrs["has_fluid_mask"] = True

        snaps = f.create_group("snapshots")
        snaps.create_dataset(
            "velocity",
            data=vel_decoded,
            chunks=(1,) + field_shape,
            compression="gzip",
            compression_opts=4,
        )

        if vort_decoded is not None:
            snaps.create_dataset(
                "vorticity",
                data=vort_decoded,
                chunks=(1,) + field_shape,
                compression="gzip",
                compression_opts=4,
            )

    print(f"SUCCESS: Saved {k} reconstructed snapshots to {out_path}.")
    print(f"Total Execution Time: {time.perf_counter() - t0:.2f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
