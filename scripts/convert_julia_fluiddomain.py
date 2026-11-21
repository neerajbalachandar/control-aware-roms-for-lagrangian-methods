#!/usr/bin/env python3
"""
Alternative stage 1 - convert FLOWUnsteady ``computefluiddomain`` (FMM) output into a trajectory file.

First run VarFlExI's own post-processor with FIXED bounds (cells = nodes - 1):

    cd VarFLEXI-rVPM-Fenics
    julia fluid/postprocess_particle_fields.jl --input <run>/fluid --output <run>/fluid_domain_rom \
        --steps 1:1:600 --nx 96 --ny 48 --nz 28 --bounds -0.12,2.28,-0.12,1.08,-0.30,0.42

then

    python scripts/convert_julia_fluiddomain.py --run-dir <run> --fd-dir <run>/fluid_domain_rom \
        --out data/trajectories/<run_name>.h5 [--bound-vortex]

``--bound-vortex`` adds the lifting-line bound vortex when no static particles
were saved (the Julia evaluation only sees particles in the pfield files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rom.cli_utils import load_yaml, setup_logging  # noqa: E402
from rom.eulerian.julia_bridge import convert_julia_fluiddomain  # noqa: E402
from rom.eulerian.xdmf import write_trajectory_xdmf  # noqa: E402
from rom.varflexi import VarFlExIRun  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fd-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/reconstruction.yaml", help="for the 'signals' section")
    ap.add_argument("--bound-vortex", action="store_true")
    ap.add_argument("--bound-vortex-core", type=float)
    a = ap.parse_args(argv)
    setup_logging()
    sig = (load_yaml(a.config).get("signals", {}) if Path(a.config).is_file() else {}) or {}
    run = VarFlExIRun(a.run_dir)
    out = convert_julia_fluiddomain(run, a.fd_dir, a.out, input_spec=sig.get("inputs"), output_spec=sig.get("outputs"),
                                    add_bound_vortex=a.bound_vortex, bound_vortex_core=a.bound_vortex_core)
    write_trajectory_xdmf(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
