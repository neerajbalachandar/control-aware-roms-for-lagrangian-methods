#!/usr/bin/env python3
"""
End-to-end smoke test / demo of the whole pipeline on synthetic VarFlExI-like runs (~1-3 min).

    python scripts/run_synthetic_demo.py --work /tmp/rom_demo [--backend numpy]

Stages executed (the same CLIs you use on real data):
  0. rom.testing.synthetic   -> 5 fake run folders (gusts of different frequency / amplitude)
  1. reconstruct_eulerian    -> trajectory files on a small fixed grid
  2. build_reduced_dataset   -> POD bases + coordinates (+ figures)
  3. rom_opinf.train/evaluate and rom_koopman.train/evaluate
  4. compare_methods
The synthetic data are NOT physical; numbers are meaningless, only the plumbing is tested.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rom.testing.synthetic import make_synthetic_run  # noqa: E402

RUNS = {  # name: (gust frequency [Hz], amplitude [m/s], split)
    "gust_f1p5_A1": (1.5, 1.0, "train"),
    "gust_f3p0_A1": (3.0, 1.0, "train"),
    "gust_f6p0_A1": (6.0, 1.0, "train"),
    "gust_f4p5_A1": (4.5, 1.0, "val"),
    "gust_f2p0_A1p5": (2.0, 1.5, "test"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="artifacts/synthetic_demo")
    ap.add_argument("--backend", default="torch", choices=["torch", "numpy", "vic"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="smaller regularisation grids (used by the tests)")
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    runs = work / "runs"
    for name, (f, A, _) in RUNS.items():
        make_synthetic_run(runs, name, n_steps=a.steps, total_time=1.0, gust_amplitude=A, gust_frequency=f)

    cfg = {"grid": {"bounds": {"x": [-0.1, 1.5], "y": [-0.1, 1.06], "z": [-0.18, 0.30]}, "spacing": 0.06},
           "reconstruction": {"backend": a.backend, "kernel": "gaussianerf", "include_freestream": True,
                              "include_bound_vortex": True, "fields": ["velocity", "vorticity"],
                              "far_field": {"distance": 0.5, "bin": 0.1}},
           "steps": None,
           "signals": {"inputs": ["gust_x", "gust_y", "gust_z", "control:*"],
                       "outputs": ["coupling:lift", "coupling:cm", "solid:tip_w_displacement",
                                   "solid:tip_w_velocity", "probe:probe_1_vz_m_s"]}}
    (work / "configs").mkdir(parents=True, exist_ok=True)
    (work / "configs" / "reconstruction.yaml").write_text(yaml.safe_dump(cfg))
    import reconstruct_eulerian

    reconstruct_eulerian.main(["--run-dir", *[str(runs / n) for n in RUNS], "--config",
                               str(work / "configs" / "reconstruction.yaml"), "--out-dir", str(work / "trajectories"),
                               "--qa-steps", str(a.steps)])

    tr = {s: [str(work / "trajectories" / f"{n}.h5") for n, v in RUNS.items() if v[2] == s] for s in ("train", "val", "test")}
    rd = {"trajectories": tr, "time_window": {"t_start": 0.15}, "stride": 1, "inputs": "all", "outputs": "all",
          "blocks": {"fluid": {"source": "field:velocity", "r_max": 20, "weighting": "volume"},
                     "structure": {"source": "structural:displacement", "r_max": 4, "weighting": "area", "rate": True}}}
    (work / "configs" / "reduced_dataset.yaml").write_text(yaml.safe_dump(rd))
    import build_reduced_dataset

    ds_path = work / "reduced" / "reduced_dataset.h5"
    build_reduced_dataset.main(["--config", str(work / "configs" / "reduced_dataset.yaml"), "--out", str(ds_path)])

    op = {"opinf": {"ranks": {"fluid": 4, "structure": 2, "structure_rate": 2}, "operators": "cAHB",
                    "lambda1": [1e-5, 1e-2, 1.0] if a.quick else [1e-8, 1e-5, 1e-2, 1e-1, 1.0],
                    "lambda2": [1e-1, 1e2] if a.quick else [1e-3, 1e-1, 1e0, 1e2],
                    "rank_sweep": {"fluid": [2, 4, 6, 8]}}}
    (work / "configs" / "opinf.yaml").write_text(yaml.safe_dump(op))
    from rom_opinf import evaluate as oe
    from rom_opinf import train as ot

    ot.main(["--dataset", str(ds_path), "--config", str(work / "configs" / "opinf.yaml"), "--out", str(work / "opinf"),
             *([] if a.quick else ["--rank-sweep"])])
    oe.main(["--model", str(work / "opinf"), "--splits", "val", "test"])

    kp = {"koopman": {"ranks": {"fluid": 4, "structure": 2, "structure_rate": 2},
                      "edmdc": {"degree": 2, "bilinear": True, "ridge": [1e-8, 1e-4] if a.quick else [1e-10, 1e-8, 1e-6, 1e-4]},
                      "lifting": {"deg_F": 2, "ridge": [1e-3] if a.quick else [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]}}}
    (work / "configs" / "koopman.yaml").write_text(yaml.safe_dump(kp))
    from rom_koopman import evaluate as ke
    from rom_koopman import train as kt

    kt.main(["--dataset", str(ds_path), "--config", str(work / "configs" / "koopman.yaml"), "--out", str(work / "koopman")])
    ke.main(["--model", str(work / "koopman"), "--splits", "val", "test"])

    import compare_methods

    compare_methods.main(["--opinf", str(work / "opinf"), "--koopman", str(work / "koopman"), "--splits", "test",
                          "--out", str(work / "comparison")])
    print(f"\nDemo finished - browse the figures under {work}")


if __name__ == "__main__":
    main()
