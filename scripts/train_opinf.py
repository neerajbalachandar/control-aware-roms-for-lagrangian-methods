#!/usr/bin/env python3
"""
scripts/train_opinf.py – Train an Operator Inference ROM from VarFlExI simulation datasets.

Workflow:
  1. Ingests HDF5 dataset(s) with rigorous validation.
  2. Partitions into chronological train / validation splits (preserving boundary isolation).
  3. Prepares flattened Eulerian velocity states on active fluid cells.
  4. Fits mean field and POD basis strictly on training snapshots.
  5. Sweeps POD ranks and regularization parameters.
  6. Fits continuous or discrete OpInf model with L2/Tikhonov regularization.
  7. Persists the complete self-contained rom_model/ bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rom.basis import PODReducer, VelocityStateEncoder, evaluate_candidate_ranks
from rom.data import DatasetReader, Trajectory, validate_trajectories
from rom.opinf import (
    TrainedROM,
    compute_operator_norms,
    train_continuous_model,
    train_discrete_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an Operator Inference (OpInf) ROM from VarFlExI Eulerian datasets."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HDF5 dataset file or directory containing HDF5 trajectory files.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["continuous", "discrete"],
        default="continuous",
        help="OpInf formulation: 'continuous' (da/dt = c+Aa+H(a⊗a)) or 'discrete' (a_{k+1}=c+Aa+H(a⊗a)).",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="cAH",
        help="Operator string (default: 'cAH').",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10, 15],
        help="Candidate POD ranks to evaluate.",
    )
    parser.add_argument(
        "--energy-target",
        type=float,
        default=0.999,
        help="Target cumulative energy fraction for POD truncation (default: 0.999).",
    )
    parser.add_argument(
        "--svdsolver",
        type=str,
        choices=["dense", "method-of-snapshots", "randomized"],
        default="dense",
        help="SVD solver algorithm for POD.",
    )
    parser.add_argument(
        "--regularizers",
        type=float,
        nargs="+",
        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        help="Candidate Tikhonov regularization strengths lambda.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of each trajectory used for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of each trajectory used for validation (default: 0.15).",
    )
    parser.add_argument(
        "--diff-order",
        type=int,
        choices=[2, 4, 6],
        default=2,
        help="Finite difference order for estimating continuous-time derivatives (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rom_model",
        help="Output directory for the trained ROM bundle (default: 'rom_model').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print("VARFLEXI OPERATOR INFERENCE PIPELINE – TRAINING")
    print("=" * 78)

    t0_total = time.perf_counter()

    # 1. Dataset Ingestion
    data_path = Path(args.data)
    if data_path.is_dir():
        print(f"Loading trajectory files from directory: {data_path}")
        trajectories = DatasetReader.load_directory(data_path, validate=True)
    else:
        print(f"Loading dataset file: {data_path}")
        trajectories = DatasetReader.load_file(data_path, validate=True)

    print(f"Successfully loaded {len(trajectories)} trajectory(s).")
    for i, t in enumerate(trajectories):
        print(f"  [{i+1}] '{t.name}': {len(t)} snapshots, time span {t.time_span}, dt={t.dt}")

    # Validate multi-trajectory compatibility
    compat_report = validate_trajectories(trajectories)
    if not compat_report.is_valid:
        print("\nERROR: Dataset validation failed:")
        for err in compat_report.errors:
            print(f"  - {err}")
        sys.exit(1)

    ref_grid = trajectories[0].grid
    ref_mask = trajectories[0].fluid_mask
    print(f"\nSpatial Grid: shape={ref_grid.shape}, resolution={ref_grid.resolution}, units={ref_grid.units}")
    if ref_mask is not None:
        print(f"Fluid Mask: active fluid cells = {ref_mask.num_fluid} / {ref_grid.num_points} "
              f"({ref_mask.fluid_fraction*100:.1f}%)")

    # 2. Chronological Splitting (preserving boundary isolation)
    train_trajs: List[Trajectory] = []
    val_trajs: List[Trajectory] = []
    test_trajs: List[Trajectory] = []

    for t in trajectories:
        n_snap = len(t)
        n_train = max(2, int(n_snap * args.train_fraction))
        n_val = max(1, int(n_snap * args.val_fraction))
        n_test = n_snap - n_train - n_val

        train_trajs.append(t.slice(0, n_train))
        val_trajs.append(t.slice(n_train, n_train + n_val))
        if n_test > 0:
            test_trajs.append(t.slice(n_train + n_val, n_snap))

    print(f"\nChronological Split:")
    print(f"  Train:      {sum(len(t) for t in train_trajs)} snapshots across {len(train_trajs)} run(s)")
    print(f"  Validation: {sum(len(t) for t in val_trajs)} snapshots across {len(val_trajs)} run(s)")
    if test_trajs:
        print(f"  Test:       {sum(len(t) for t in test_trajs)} snapshots across {len(test_trajs)} run(s)")

    # 3. State Preparation & Active Domain Mapping
    encoder = VelocityStateEncoder(ref_grid, ref_mask)
    print(f"State Dimension: n = {encoder.state_dimension} (3 velocity components * {encoder.n_fluid_cells} fluid cells)")

    # Encode training and validation trajectories
    Q_train_list = [encoder.encode_trajectory(t.velocity) for t in train_trajs]
    Q_val_list = [encoder.encode_trajectory(t.velocity) for t in val_trajs]

    Q_train_concat = np.hstack(Q_train_list)
    Q_val_concat = np.hstack(Q_val_list)

    # Fit mean field strictly on training snapshots!
    mean_vec = encoder.fit_mean(Q_train_concat)
    print(f"Computed mean field from {Q_train_concat.shape[1]} training snapshots (norm = {np.linalg.norm(mean_vec):.4e})")

    # Center snapshots
    Q_train_cent_list = [encoder.center(q) for q in Q_train_list]
    Q_val_cent_list = [encoder.center(q) for q in Q_val_list]
    Q_train_cent_concat = np.hstack(Q_train_cent_list)
    Q_val_cent_concat = np.hstack(Q_val_cent_list)

    # 4. Proper Orthogonal Decomposition (POD)
    print(f"\nComputing POD basis using svdsolver='{args.svdsolver}'...")
    reducer = PODReducer(r=max(args.ranks), svdsolver=args.svdsolver)
    reducer.fit(Q_train_cent_concat)

    # Evaluate candidate ranks
    rank_eval = evaluate_candidate_ranks(
        reducer,
        candidate_ranks=args.ranks,
        Q_train_centered=Q_train_cent_concat,
        Q_val_centered=Q_val_cent_concat,
        energy_target=args.energy_target,
    )
    print("\nPOD Rank Evaluation:")
    print(rank_eval.as_table())
    chosen_rank = rank_eval.recommended_rank
    print(f"\nSelected POD rank r = {chosen_rank} ({rank_eval.selection_criterion})")
    reducer.set_rank(chosen_rank)

    # Project snapshots to reduced coordinates
    a_train_list = [reducer.project(q) for q in Q_train_cent_list]
    a_val_list = [reducer.project(q) for q in Q_val_cent_list]

    # 5. OpInf Model Training
    print(f"\nTraining {args.model_type.upper()} OpInf model with operators='{args.operators}'...")
    train_times = [t.time for t in train_trajs]
    val_times = [t.time for t in val_trajs]

    if args.model_type == "continuous":
        train_result = train_continuous_model(
            train_trajectories=a_train_list,
            train_times=train_times,
            operators=args.operators,
            regularizers=args.regularizers,
            val_trajectories=a_val_list,
            val_times=val_times,
            order=args.diff_order,
        )
    else:
        train_result = train_discrete_model(
            train_trajectories=a_train_list,
            operators=args.operators,
            regularizers=args.regularizers,
            val_trajectories=a_val_list,
        )

    model = train_result.model
    print(f"Optimal Regularizer: lambda = {train_result.best_regularizer:.2e}")
    print(f"Training Residual:   {train_result.train_residual:.4e}")
    if train_result.val_rollout_error is not None:
        print(f"Validation Rollout:  {train_result.val_rollout_error:.4e}")
    print("Operator Norms:")
    for op_k, op_v in train_result.operator_norms.items():
        print(f"  {op_k}: {op_v:.4e}")

    # 6. Save ROM Bundle
    output_dir = Path(args.output_dir)
    print(f"\nPersisting ROM bundle to '{output_dir}'...")

    summary_metadata = {
        "dataset": str(data_path),
        "model_type": args.model_type,
        "operators": args.operators,
        "r": chosen_rank,
        "n": encoder.state_dimension,
        "best_regularizer": train_result.best_regularizer,
        "train_residual": train_result.train_residual,
        "val_rollout_error": train_result.val_rollout_error,
        "operator_norms": train_result.operator_norms,
        "cumulative_energy": reducer.cumulative_energy,
        "runtime_seconds": time.perf_counter() - t0_total,
    }

    rom = TrainedROM(
        model=model,
        encoder=encoder,
        reducer=reducer,
        metadata=summary_metadata,
    )
    rom.save(output_dir)

    # Save training sweep metrics json
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(
            {
                "summary": summary_metadata,
                "rank_evaluation": [
                    {
                        "rank": m.rank,
                        "singular_value": m.singular_value,
                        "cumulative_energy": m.cumulative_energy,
                        "train_projection_error": m.train_projection_error,
                        "val_projection_error": m.val_projection_error,
                    }
                    for m in rank_eval.metrics
                ],
                "regularizer_sweep": train_result.sweep_history,
            },
            f,
            indent=2,
        )

    print(f"SUCCESS: Saved all 6 files to {output_dir}/:")
    for fname in ["metadata.json", "mean_field.npz", "pod_basis.npz", "operators.h5", "grid_metadata.npz", "preprocessing.npz", "training_metrics.json"]:
        print(f"  - {fname}")

    print(f"\nTotal Pipeline Execution Time: {time.perf_counter() - t0_total:.2f} seconds.")
    print("=" * 78)


if __name__ == "__main__":
    main()
