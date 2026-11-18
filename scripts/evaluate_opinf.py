#!/usr/bin/env python3
"""
scripts/evaluate_opinf.py – Comprehensive evaluation of trained OpInf ROMs.

Evaluates and contrasts:
  1. POD spatial projection accuracy (truncation error limit).
  2. Autonomous dynamical rollout accuracy (temporal trajectory prediction).
  3. Generalization across held-out time segments or simulation trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rom.data import DatasetReader, Trajectory
from rom.opinf import TrainedROM, assess_stability, compute_rollout_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OpInf ROM on test/validation trajectories."
    )
    parser.add_argument(
        "--rom-dir",
        type=str,
        default="rom_model",
        help="Directory containing the saved ROM bundle (default: 'rom_model').",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HDF5 dataset file or directory containing evaluation trajectories.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="evaluation_report.json",
        help="Path to save the JSON evaluation report (default: 'evaluation_report.json').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print("VARFLEXI OPERATOR INFERENCE PIPELINE – EVALUATION")
    print("=" * 78)

    t0_eval = time.perf_counter()

    # 1. Load Trained ROM (completely standalone, no Julia/FLOWUnsteady!)
    rom_dir = Path(args.rom_dir)
    print(f"Loading trained ROM bundle from '{rom_dir}'...")
    rom = TrainedROM.load(rom_dir)
    print(f"  Model Type:        {rom.model_type}")
    print(f"  Reduced Rank (r):  {rom.r}")
    print(f"  Full State Dim (n):{rom.n}")
    print(f"  Spatial Grid:      {rom.grid.shape}")

    # 2. Ingest Evaluation Data
    data_path = Path(args.data)
    if data_path.is_dir():
        trajectories = DatasetReader.load_directory(data_path, validate=True)
    else:
        trajectories = DatasetReader.load_file(data_path, validate=True)

    print(f"\nEvaluating on {len(trajectories)} trajectory(s)...")

    results: List[Dict[str, Any]] = []

    header = (
        f"{'Trajectory':<20} | {'Snapshots':<9} | {'Proj Err (%)':<13} | "
        f"{'Rollout Err (%)':<16} | {'Status':<10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for idx, traj in enumerate(trajectories):
        t_start_traj = time.perf_counter()
        k = len(traj)

        # Encode full velocity snapshots to (n, K)
        Q = rom.encoder.encode_trajectory(traj.velocity)
        Q_denom = float(np.linalg.norm(Q, "fro"))

        # Center with ROM's mean field
        Q_centered = rom.encoder.center(Q)

        # A. POD Spatial Projection Error
        a_proj = rom.reducer.project(Q_centered)  # (r, K)
        Q_rec_cent = rom.reducer.reconstruct(a_proj)  # (n, K)
        Q_rec_full = rom.encoder.uncenter(Q_rec_cent)

        proj_fro = float(np.linalg.norm(Q - Q_rec_full, "fro"))
        rel_proj_err = proj_fro / Q_denom if Q_denom > 0 else 0.0

        # B. Autonomous Dynamical Rollout
        a0 = a_proj[:, 0]
        if rom.model_type == "continuous":
            a_pred = rom.predict(a0, traj.time)
        else:
            a_pred = rom.predict(a0, k)

        stab = assess_stability(a_pred, reference_scale=float(np.max(np.abs(a_proj))))

        if stab["is_stable"]:
            # Evaluate error in reduced coordinates
            red_err = compute_rollout_error(a_proj, a_pred)

            # Reconstruct full velocity fields from dynamical rollout
            Q_dyn_full = rom.encoder.uncenter(rom.reducer.reconstruct(a_pred))
            dyn_fro = float(np.linalg.norm(Q - Q_dyn_full, "fro"))
            rel_dyn_err = dyn_fro / Q_denom if Q_denom > 0 else 0.0
            status = "STABLE"
        else:
            rel_dyn_err = float("inf")
            red_err = {
                "relative_frobenius_error": float("inf"),
                "mean_relative_l2_error": float("inf"),
                "max_absolute_error": float("inf"),
            }
            status = "UNSTABLE"

        traj_eval_time = time.perf_counter() - t_start_traj

        print(
            f"{traj.name:<20} | {k:<9d} | {rel_proj_err * 100:12.4f}% | "
            f"{rel_dyn_err * 100:15.4f}% | {status:<10}"
        )

        results.append({
            "name": traj.name,
            "num_snapshots": k,
            "time_span": traj.time_span,
            "relative_projection_error": rel_proj_err,
            "relative_rollout_error_physical": rel_dyn_err,
            "reduced_rollout_metrics": red_err,
            "stability": stab,
            "eval_time_seconds": traj_eval_time,
        })

    # Summary report
    avg_proj_err = float(np.mean([r["relative_projection_error"] for r in results]))
    finite_dyn_errs = [r["relative_rollout_error_physical"] for r in results if np.isfinite(r["relative_rollout_error_physical"])]
    avg_dyn_err = float(np.mean(finite_dyn_errs)) if finite_dyn_errs else float("inf")

    print("\nSummary Metrics:")
    print(f"  Average Spatial Projection Error:  {avg_proj_err * 100:.4f}%")
    print(f"  Average Dynamical Rollout Error:   {avg_dyn_err * 100:.4f}%")
    print(f"  Total Evaluation Time:             {time.perf_counter() - t0_eval:.2f} s")

    report_payload = {
        "rom_metadata": rom.metadata,
        "evaluation_summary": {
            "average_projection_error": avg_proj_err,
            "average_rollout_error": avg_dyn_err,
            "total_eval_time_seconds": time.perf_counter() - t0_eval,
        },
        "per_trajectory": results,
    }

    out_file = Path(args.output_report)
    with open(out_file, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved evaluation report to '{out_file}'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
