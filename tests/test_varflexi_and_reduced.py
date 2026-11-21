"""Run reader on a synthetic VarFlExI folder, trajectory files, streaming POD, exact field errors."""

import numpy as np
import pytest

from rom.data.trajectory_file import TrajectoryFile
from rom.eulerian import ReconstructionConfig, grid_from_config, reconstruct_run
from rom.reduced.dataset import ReducedDataset, build_reduced_dataset
from rom.reduced.state import StateAssembler, block_field_error
from rom.reduced.streaming_pod import SnapshotSet, streaming_pod
from rom.testing.synthetic import make_synthetic_run
from rom.varflexi import VarFlExIRun, assemble_signals, parse_step_spec


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    return [make_synthetic_run(root, f"r{i}", n_steps=40, total_time=0.4, gust_frequency=f, gust_amplitude=1.0)
            for i, f in enumerate((2.0, 4.0, 3.0))]


def test_reader(runs):
    run = VarFlExIRun(runs[0])
    assert run.dt == pytest.approx(0.01)
    assert run.particle_steps[0] == 0 and run.particle_steps[-1] == 40
    assert run.read_particles(0).n == 0                        # dummy particle dropped
    p = run.read_particles(10)
    assert p.n > 0 and p.time == pytest.approx(0.1)
    v = run.freestream(np.zeros((1, 3)), 0.125)[0]
    assert v[2] == pytest.approx(15 * np.sin(np.deg2rad(5)) + np.sin(2 * np.pi * 2.0 * 0.125))
    st = run.station_history()
    assert st["bound_circulation_m2_s"].shape == (40, 10)
    pd_ = run.plate_displacement()
    assert pd_["step"][0] == 1 and pd_["displacement"].shape[0] == 40
    sig = assemble_signals(run, [1, 5, 10])
    assert "gust_z" in sig.input_names and "lift" in sig.output_names
    assert parse_step_spec("2:2:10", run.particle_steps) == [2, 4, 6, 8, 10]


def test_streaming_pod_matches_svd_both_paths():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(300, 12)) @ rng.normal(size=(12, 50)) + 0.01 * rng.normal(size=(300, 50))
    w = rng.uniform(0.5, 2.0, 300)
    rd = lambda idx: iter([Q[:, idx]])  # noqa: E731
    ss = SnapshotSet([rd], [np.arange(50)])
    a = streaming_pod(ss, w, 8, block=7)
    b = streaming_pod(ss, w, 8, block=7, memory_limit_gb=0.0)        # out-of-core path
    Qc = (Q - Q.mean(1, keepdims=True)) * np.sqrt(w)[:, None]
    s = np.linalg.svd(Qc, compute_uv=False)
    np.testing.assert_allclose(a.singular_values[:8], s[:8], rtol=1e-6)
    np.testing.assert_allclose(b.singular_values[:8], s[:8], rtol=1e-6)
    np.testing.assert_allclose(a.modes.T @ (w[:, None] * a.modes), np.eye(8), atol=1e-6)
    np.testing.assert_allclose(np.abs(a.modes), np.abs(b.modes), atol=1e-6)


def test_reconstruct_build_and_exact_errors(runs, tmp_path):
    grid = grid_from_config({"bounds": {"x": [-0.1, 0.9], "y": [-0.1, 1.06], "z": [-0.2, 0.2]}, "spacing": 0.1})
    cfg = ReconstructionConfig(backend="numpy", far_field_distance=0.5, far_field_bin=0.1)
    files = []
    for r in runs:
        run = VarFlExIRun(r)
        files.append(reconstruct_run(run, grid, tmp_path / f"{run.name}.h5", run.particle_steps[1:], cfg))
    tf = TrajectoryFile(files[0])
    assert tf.read_field("velocity", 3).shape == grid.shape + (3,)
    assert np.all(tf.completed)
    conf = {"trajectories": {"train": [str(files[0]), str(files[1])], "val": [str(files[2])], "test": []},
            "time_window": {"t_start": 0.05}, "blocks": {
                "fluid": {"source": "field:velocity", "r_max": 10},
                "structure": {"source": "structural:displacement", "r_max": 3, "rate": True}}}
    ds = build_reduced_dataset(conf, tmp_path / "rd.h5")
    ds = ReducedDataset(tmp_path / "rd.h5")
    assert set(ds.block_names) == {"fluid", "structure", "structure_rate"}
    # exact field-error identity vs explicit reconstruction
    t = ds.split("val")[0]
    basis = ds.basis("fluid")
    rng = np.random.default_rng(0)
    a_pred = t.coords["fluid"][:5] + 0.01 * rng.normal(size=(5, len(t.time)))
    fe = block_field_error(t, "fluid", a_pred)
    q = TrajectoryFile(t.source_file).read_field("velocity", t.index).reshape(len(t.time), -1).T
    qp = basis.reconstruct(a_pred)
    direct = np.sqrt(np.einsum("ik,ik->k", q - qp, (q - qp) * basis.weights[:, None]))
    np.testing.assert_allclose(fe["error"], direct, rtol=1e-4)
    asm = StateAssembler.fit(ds, {"fluid": 4, "structure": 2})
    X = asm.state(t)
    assert X.shape == (6, len(t.time)) and np.max(np.abs(asm.state(ds.split("train")[0]))) <= 1 + 1e-12
