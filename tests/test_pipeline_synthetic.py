"""End-to-end: synthetic runs -> reconstruction -> reduced dataset -> OpInf + Koopman -> comparison."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    import run_synthetic_demo

    run_synthetic_demo.main(["--work", str(tmp_path), "--steps", "60", "--quick", "--backend", "numpy"])
    assert (tmp_path / "reduced" / "reduced_dataset.h5").is_file()
    assert (tmp_path / "reduced" / "figures" / "pod_spectrum.png").is_file()
    for m in ("opinf", "koopman"):
        metrics = json.loads((tmp_path / m / "evaluation" / "metrics.json").read_text())
        assert set(metrics["trajectories"]) == {"gust_f4p5_A1", "gust_f2p0_A1p5"}
    assert (tmp_path / "comparison" / "comparison.md").is_file()
