"""CUDA backend for gdmaps.run_grid: CPU-side contracts and GPU agreement checks.

Only artificial fixtures with seeds 99990-99999 are used; no study system,
seed or saved outcome is loaded. GPU tests skip cleanly when nvcc, the host
compiler or a CUDA device is unavailable. They build the library into a
temporary directory unless NEURAL_SHRIMP_GDMAP_CUDA_LIBRARY names a build.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from neural_shrimp import gdmaps_cuda as gc
from neural_shrimp.gdmaps import (
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_DRIFT, STATUS_FIXED, STATUS_PERIODIC,
    STATUS_UNRESOLVED, GridMap, TwoLayerNet, iterate_and_screen, run_grid, screen_tails,
    sohl_dickstein_measure,
)
from neural_shrimp.models import TanhRegressor, make_teacher_regression

FAST = dict(tail_steps=384, ftle_steps=1000, ftle_align=200, sd_horizons=(100, 500))


def s1_like(seed=99990):
    """Sohl-Dickstein-style net: width 8, d 8, no bias, n equal to the parameter count."""
    rng = np.random.default_rng(seed)
    net = TwoLayerNet(d=8, width=8, in_scale=np.sqrt(2.0 / 8), out_scale=1.0 / 8, bias=False,
                      loss_scale=1.0)
    p = net.n_parameters
    return net, rng.normal(size=(p, 8)), rng.normal(size=p), rng.normal(size=p)


def s2_like(seed=99991):
    """TanhRegressor-style net: d 2, width 3, biases, half MSE, ridge 1e-3, n 24."""
    net = TwoLayerNet(d=2, width=3, bias=True, loss_scale=0.5, ridge=1e-3)
    data = make_teacher_regression(seed=seed, d=2, teacher_width=3, n_train=24)
    theta0 = TanhRegressor(2, 3).initialize(seed=seed + 1, scale=0.2)
    return net, data.train_x, data.train_y, theta0


def rate_grid(net, x, y, theta0, hidden_range, output_range, side):
    """Rates as decades around the Gauss-Newton edge rates 2 / lambda at initialization."""
    hidden_max, output_max = net.gauss_newton_layer_max(theta0, x, y)
    lh, lo = np.meshgrid(2.0 / hidden_max * 10.0 ** np.linspace(*hidden_range, side),
                         2.0 / output_max * 10.0 ** np.linspace(*output_range, side), indexing="ij")
    return GridMap(net, x, y, lh.ravel(), lo.ravel())


# --------------------------------------------------------------------------
# CPU-side contracts (no GPU needed)


def test_openblas_k_blocks_follow_level3_driver_rule():
    assert gc.openblas_k_blocks(24) == [24]
    assert gc.openblas_k_blocks(256) == [256]
    assert gc.openblas_k_blocks(257) == [128, 257]
    assert gc.openblas_k_blocks(272) == [136, 272]
    assert gc.openblas_k_blocks(300) == [152, 300]
    assert gc.openblas_k_blocks(520) == [256, 388, 520]


def test_option_defaults_and_validation_match_cpu_engine():
    assert gc.OPTION_DEFAULTS["sd_horizons"] == (100, 500, 1000)
    assert gc.OPTION_DEFAULTS["tail_steps"] == 512 and gc.OPTION_DEFAULTS["tangent_seed"] == 0
    steps, opts = gc._resolve_options(20000, {})
    assert steps == 20000 and opts["max_period"] == 64 and opts["atol"] == 1e-8
    seed = np.random.SeedSequence(99998)
    assert gc._resolve_options(20000, {"tangent_seed": seed})[1]["tangent_seed"] is seed
    with pytest.raises(TypeError):
        gc._resolve_options(5000, {"no_such_option": 1})
    for bad, message in (({"tail_steps": 0}, "tail and FTLE"), ({"ftle_align": -1}, "tail and FTLE"),
                         ({"tail_steps": 6000}, "steps must cover"),
                         ({"min_repeats": 1}, "invalid recurrence"),
                         ({"sd_horizons": (10,)}, "steps >= 20")):
        with pytest.raises(ValueError, match=message):
            gc._resolve_options(5000, bad)
    # The CPU engine raises the same messages for the window checks.
    net, x, y, theta0 = s2_like()
    grid = GridMap(net, x, y, [0.1], [0.1])
    with pytest.raises(ValueError, match="steps must cover"):
        run_grid(grid, theta0, 100, tail_steps=200)


def test_chunked_tangent_draws_equal_one_shot_draws():
    full = np.random.default_rng(0).normal(size=(1000, 13))
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    parts = []
    for size in (1, 333, 2, 664):
        part = rng.normal(size=(size, 13))
        part /= np.linalg.norm(part, axis=1, keepdims=True)
        parts.append(part)
    np.testing.assert_array_equal(np.concatenate(parts), full)


def test_chunked_sohl_dickstein_measures_equal_whole_grid_call():
    rng = np.random.default_rng(99997)
    losses = rng.uniform(0.05, 3.0, size=(120, 9))
    losses[50:, 4] = np.inf
    losses[:, 6] = np.nan
    whole = {h: sohl_dickstein_measure(losses[:h]) for h in (20, 100, 120)}
    for bounds in ([0, 9], [0, 1, 9], [0, 4, 5, 9], [0, 2, 3, 8, 9]):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            chunk = gc.chunk_sd_measures(losses[:, lo:hi], 9, (20, 100, 120))
            for h, (convergent, score) in chunk.items():
                np.testing.assert_array_equal(convergent, whole[h][0][lo:hi])
                np.testing.assert_array_equal(score, whole[h][1][lo:hi])
    single = gc.chunk_sd_measures(losses[:, :1], 1, (100,))[100]
    np.testing.assert_array_equal(single[1], sohl_dickstein_measure(losses[:100, :1])[1])


def test_build_refuses_overwrite_and_records_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(gc.shutil, "which", lambda value: "/usr/bin/" + value)
    monkeypatch.setattr(gc.subprocess, "check_output", lambda *a, **k: "release 12.4, V12.4\n")

    class Compiler:
        returncode = 0

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            Path(command[-1]).write_bytes(b"synthetic library, not executable")

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(gc.subprocess, "Popen", Compiler)
    receipt = gc.build(tmp_path)
    assert "--fmad=false" in receipt["command"] and "-arch=sm_89" in receipt["command"]
    assert "--use_fast_math" not in receipt["command"] and receipt["fmad"] is False
    assert receipt["gpu_initialized"] is False
    assert json.loads(Path(receipt["receipt"]).read_text())["library_sha256"] == receipt["library_sha256"]
    with pytest.raises(FileExistsError):
        gc.build(tmp_path)
    other = gc.build(tmp_path / "contract", variant="contract")
    assert "--fmad=true" in other["command"] and other["free_contraction"] is True


def test_load_requires_an_explicit_library(monkeypatch):
    monkeypatch.delenv(gc.LIBRARY_ENV, raising=False)
    with pytest.raises(FileNotFoundError):
        gc.load()


# --------------------------------------------------------------------------
# GPU checks


@pytest.fixture(scope="session")
def cuda(tmp_path_factory):
    library = os.environ.get(gc.LIBRARY_ENV)
    if not library:
        if not shutil.which("nvcc") or not shutil.which("g++-12"):
            pytest.skip("nvcc or g++-12 unavailable")
        if not shutil.which("nvidia-smi"):
            pytest.skip("no NVIDIA driver tools")
        try:
            subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pytest.skip("no visible CUDA device")
        library = gc.build(tmp_path_factory.mktemp("gdmap_cuda"))["library"]
    backend = gc.load(library)
    try:
        info = backend.device_info()
    except gc.GdmapCudaError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    if (info["major"], info["minor"]) != (8, 9):
        pytest.skip("backend qualified for sm_89 only")
    return backend


@pytest.fixture(scope="session")
def s1_runs(cuda):
    net, x, y, theta0 = s1_like()
    grid = rate_grid(net, x, y, theta0, (-1.5, 1.0), (-1.5, 0.5), 8)
    cpu = run_grid(grid, theta0, 3000, **FAST)
    gpu = cuda.run_grid(grid, theta0, 3000, **FAST)
    return grid, theta0, cpu, gpu


@pytest.fixture(scope="session")
def s2_runs(cuda):
    net, x, y, theta0 = s2_like()
    grid = rate_grid(net, x, y, theta0, (-4.5, -0.5), (-1.6, 0.4), 12)
    cpu = run_grid(grid, theta0, 8000, **FAST)
    gpu = cuda.run_grid(grid, theta0, 8000, **FAST)
    return grid, theta0, cpu, gpu


def _relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("fixture,seed", [("s1", 99992), ("s2", 99993), ("w16", 99994)])
def test_single_step_gradient_and_hvp_agree_with_cpu(cuda, fixture, seed):
    rng = np.random.default_rng(seed)
    if fixture == "s1":
        net, x, y, _ = s1_like()
    elif fixture == "s2":
        net, x, y, _ = s2_like()
    else:
        net = TwoLayerNet(d=16, width=16, in_scale=np.sqrt(2.0 / 16), out_scale=1.0 / 16,
                          bias=False, loss_scale=1.0)
        x = rng.normal(size=(net.n_parameters, 16))
        y = rng.normal(size=net.n_parameters)
    theta = rng.normal(size=(9, net.n_parameters))
    vectors = rng.normal(size=theta.shape)
    grad, loss = net.gradient(theta, x, y, return_loss=True)
    hvp = net.hvp(theta, x, y, vectors)
    out = cuda.evaluate(net, x, y, theta, vectors)
    assert _relative(out["grad"], grad) < 1e-13
    assert _relative(out["hvp"], hvp) < 1e-13
    np.testing.assert_allclose(out["loss"], loss, rtol=1e-13, atol=0)


@pytest.mark.parametrize("d,width,n,bias", [(2, 3, 24, True), (8, 8, 72, False), (16, 16, 272, False)])
def test_identity_network_is_bitwise_identical_to_cpu(cuda, d, width, n, bias):
    # Without tanh, every operation is mirrored in order, including both OpenBLAS
    # K blocks at n = 272. This holds for even d on the reference host's BLAS.
    rng = np.random.default_rng(99995)
    net = TwoLayerNet(d=d, width=width, activation="identity", bias=bias, in_scale=0.7,
                      out_scale=0.3, loss_scale=0.5, ridge=1e-3)
    x, y = rng.normal(size=(n, d)), rng.normal(size=n)
    theta = rng.normal(size=(7, net.n_parameters))
    vectors = rng.normal(size=theta.shape)
    grad, loss = net.gradient(theta, x, y, return_loss=True)
    out = cuda.evaluate(net, x, y, theta, vectors)
    np.testing.assert_array_equal(out["grad"], grad)
    np.testing.assert_array_equal(out["hvp"], net.hvp(theta, x, y, vectors))
    np.testing.assert_array_equal(out["loss"], loss)


def test_one_step_map_and_tangent_match_cpu(cuda):
    net, x, y, theta0 = s2_like()
    grid = GridMap(net, x, y, [0.3, 3.0, 40.0], [0.2, 1.0, 2.5])
    opts = dict(tail_steps=1, ftle_steps=1, ftle_align=0, sd_horizons=())
    cpu = run_grid(grid, theta0, 1, **opts)
    gpu = cuda.run_grid(grid, theta0, 1, **opts)
    assert _relative(gpu["final_state"], cpu["final_state"]) < 1e-14
    np.testing.assert_allclose(gpu["ftle"], cpu["ftle"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(gpu["last_evaluated_loss"], cpu["last_evaluated_loss"], rtol=1e-14, atol=0)


def test_device_iteration_equals_evaluate_plus_host_update(cuda):
    # The advance kernel's update is exactly theta - rates * grad, as NumPy does it.
    net, x, y, theta0 = s2_like()
    grid = rate_grid(net, x, y, theta0, (-4.5, -0.5), (-1.6, 0.4), 5)
    device = cuda.run_grid(grid, theta0, 30, tail_steps=1, ftle_steps=1, ftle_align=0, sd_horizons=())
    hosted = np.repeat(theta0[None], grid.cells, axis=0)
    with np.errstate(all="ignore"):
        for _ in range(30):
            hosted = hosted - grid.rates * cuda.evaluate(net, x, y, hosted)["grad"]
    alive = device["status"] != STATUS_DIVERGENT
    assert alive.sum() >= 10
    np.testing.assert_array_equal(device["final_state"][alive], hosted[alive])


def test_result_keys_dtypes_and_shapes_match_cpu(s2_runs):
    _, _, cpu, gpu = s2_runs
    assert list(gpu) == list(cpu)
    for key in cpu:
        assert gpu[key].dtype == cpu[key].dtype, key
        assert gpu[key].shape == cpu[key].shape, key


def _aligned_distance(grid, index, cpu_state, gpu_state, period):
    map_fn, _ = grid.cell_map(index)
    state, best = np.array(cpu_state), np.inf
    for _ in range(max(int(period), 1)):
        best = min(best, float(np.max(np.abs(gpu_state - state))))
        state = map_fn(state)
    return best


def test_fixed_and_periodic_cells_agree_up_to_phase(s2_runs):
    grid, _, cpu, gpu = s2_runs
    orbit = np.flatnonzero(np.isin(cpu["status"], (STATUS_FIXED, STATUS_PERIODIC)))
    assert len(orbit) >= 10 and np.any(cpu["status"] == STATUS_PERIODIC)
    np.testing.assert_array_equal(gpu["status"][orbit], cpu["status"][orbit])
    np.testing.assert_array_equal(gpu["period"][orbit], cpu["period"][orbit])
    for i in orbit:
        distance = _aligned_distance(grid, i, cpu["final_state"][i], gpu["final_state"][i],
                                     cpu["period"][i])
        scale = 1e-8 + 1e-6 * np.max(np.abs(cpu["final_state"][i]))
        assert distance <= scale, (i, distance)
        if cpu["recurrence_residual"][i] <= 1e-3:  # converged far inside the tolerance
            assert distance <= 1e-9, (i, distance)


def test_divergent_cells_agree(s1_runs, s2_runs, capsys):
    for name, (_, _, cpu, gpu) in (("S1-like", s1_runs), ("S2-like", s2_runs)):
        divergent = cpu["status"] == STATUS_DIVERGENT
        assert divergent.any()
        np.testing.assert_array_equal(gpu["status"] == STATUS_DIVERGENT, divergent)
        assert np.all(np.isnan(gpu["final_state"][divergent]))
        np.testing.assert_array_equal(gpu["diverged_at"][~divergent], -1)
        cpu_step, gpu_step = cpu["diverged_at"][divergent], gpu["diverged_at"][divergent]
        # An escape after a chaotic transient can shift by a few steps; an early
        # escape, before rounding differences can grow, must match exactly.
        early = np.minimum(cpu_step, gpu_step) < 16
        np.testing.assert_array_equal(gpu_step[early], cpu_step[early])
        with capsys.disabled():
            print(f"\n{name}: {divergent.sum()} divergent cells, divergence step equal for "
                  f"{np.mean(cpu_step == gpu_step):.3f}, max shift {np.max(np.abs(cpu_step - gpu_step))}")


def test_chaotic_cells_report_ftle_and_status_agreement(s1_runs, s2_runs, capsys):
    for name, (_, _, cpu, gpu) in (("S1-like", s1_runs), ("S2-like", s2_runs)):
        chaotic = cpu["status"] == STATUS_CHAOTIC
        assert chaotic.any()
        agreement = np.mean(gpu["status"] == cpu["status"])
        both = chaotic & (gpu["status"] == STATUS_CHAOTIC)
        difference = np.abs(gpu["ftle"][both] - cpu["ftle"][both])
        with capsys.disabled():
            print(f"\n{name}: status agreement {agreement:.3f} over {len(cpu['status'])} cells; "
                  f"{chaotic.sum()} CPU-chaotic cells, {both.sum()} chaotic on both; |dFTLE| "
                  f"median {np.median(difference):.2e} max {np.max(difference):.2e}")
        # Chaotic trajectories decorrelate, so only the classification is compared.
        assert agreement >= 0.9


def test_sohl_dickstein_measures_agree(s1_runs, s2_runs, capsys):
    for name, (_, _, cpu, gpu) in (("S1-like", s1_runs), ("S2-like", s2_runs)):
        for horizon in FAST["sd_horizons"]:
            np.testing.assert_array_equal(gpu[f"sd_convergent_{horizon}"],
                                          cpu[f"sd_convergent_{horizon}"])
            relative = np.abs(gpu[f"sd_score_{horizon}"] - cpu[f"sd_score_{horizon}"]) / np.abs(
                cpu[f"sd_score_{horizon}"])
            with capsys.disabled():
                print(f"\n{name}: sd_score_{horizon} relative difference median "
                      f"{np.median(relative):.1e}, max {np.max(relative):.1e}")
        # Cells without chaotic transients have matching loss histories.
        calm = (np.isin(cpu["status"], (STATUS_UNRESOLVED, STATUS_DRIFT))
                | ((cpu["status"] == STATUS_DIVERGENT) & (cpu["diverged_at"] < 16)))
        assert calm.any()
        np.testing.assert_allclose(gpu["sd_score_100"][calm], cpu["sd_score_100"][calm], rtol=1e-9,
                                   atol=0)


def test_drift_labels_and_window_norm_growth_agree(s1_runs, s2_runs):
    total = 0
    for _, _, cpu, gpu in (s1_runs, s2_runs):
        drift = cpu["status"] == STATUS_DRIFT
        total += drift.sum()
        np.testing.assert_array_equal(gpu["status"] == STATUS_DRIFT, drift)
        np.testing.assert_array_equal(np.isnan(gpu["window_norm_growth"]),
                                      np.isnan(cpu["window_norm_growth"]))
        difference = np.abs(gpu["window_norm_growth"] - cpu["window_norm_growth"])
        assert np.all(difference[drift] <= 1e-3)
        fixed = cpu["status"] == STATUS_FIXED
        assert np.all(difference[fixed] <= 1e-12)
    assert total >= 3


def test_device_screen_matches_cpu_screen_on_synthetic_tails(cuda):
    steps = 384
    t = np.arange(steps)
    fixed = np.ones((steps, 3))
    period3 = np.array([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])[t % 3]
    period64 = np.random.default_rng(99990).normal(size=(64, 3))[t % 64]
    drift = np.column_stack([1e-3 * t, np.zeros(steps), np.zeros(steps)])
    nan = fixed.copy()
    nan[-1, 0] = np.nan
    last_pair = drift.copy()
    last_pair[-2:] = 5.0
    early_break = period3.copy()
    early_break[10, 0] += 0.5  # violation far from the end of the tail
    noisy = period3 + 1e-9 * np.random.default_rng(99991).normal(size=period3.shape)
    tails = np.stack([fixed, period3, period64, drift, nan, last_pair, early_break, noisy])
    for options in ({}, {"max_period": 8}, {"atol": 1e-10, "rtol": 1e-8}):
        cpu_period, cpu_residual = screen_tails(tails, **options)
        gpu_period, gpu_residual = cuda.screen_tails(tails, **options)
        np.testing.assert_array_equal(gpu_period, cpu_period)
        np.testing.assert_array_equal(gpu_residual, cpu_residual)
    np.testing.assert_array_equal(cuda.screen_tails(tails)[0], [1, 3, 64, 0, -1, 0, 0, 3])


def test_repeated_and_rechunked_gpu_runs_are_bit_identical(cuda, s2_runs):
    grid, theta0, _, first = s2_runs
    again = cuda.run_grid(grid, theta0, 8000, **FAST)
    rechunked = cuda.run_grid(grid, theta0, 8000, chunk_cells=37, tail_cells=5, launch_steps=777,
                              screen="host", **FAST)
    for key in first:
        np.testing.assert_array_equal(again[key], first[key], err_msg=key)
        np.testing.assert_array_equal(rechunked[key], first[key], err_msg=key)


def test_device_screen_is_bitwise_identical_to_cpu_screen(cuda, s1_runs):
    grid, theta0, _, device = s1_runs
    host = cuda.run_grid(grid, theta0, 3000, screen="host", **FAST)
    for key in device:
        np.testing.assert_array_equal(host[key], device[key], err_msg=key)


def test_nonfinite_initial_parameters_mirror_cpu(cuda):
    net, x, y, theta0 = s2_like()
    theta0 = theta0.copy()
    theta0[3] = np.nan
    grid = GridMap(net, x, y, [0.1, 1.0, 5.0], [0.1, 0.5, 1.0])
    opts = dict(tail_steps=50, ftle_steps=50, ftle_align=10, sd_horizons=(20,))
    cpu = run_grid(grid, theta0, 100, **opts)
    gpu = cuda.run_grid(grid, theta0, 100, **opts)
    for key in cpu:
        np.testing.assert_array_equal(gpu[key], cpu[key], err_msg=key)


def test_metadata_records_versions_and_resources(cuda):
    net, x, y, theta0 = s2_like()
    grid = GridMap(net, x, y, [0.5, 1.0], [0.5, 1.0])
    _, meta = cuda.run_grid(grid, theta0, 200, return_metadata=True, tail_steps=20, ftle_steps=20,
                            ftle_align=0, sd_horizons=())
    assert meta["device"]["major"] == 8 and meta["device"]["driver_version"] > 0
    assert meta["device"]["runtime_version"] >= 12000
    assert meta["library_sha256"] and meta["planned_device_bytes"] <= meta["device_byte_budget"]
    if meta["build_command"] is not None:
        assert "--fmad=false" in meta["build_command"] and meta["nvcc_version"]


def test_rejects_unsupported_networks(cuda):
    rng = np.random.default_rng(99996)
    net = TwoLayerNet(d=17, width=2)
    grid = GridMap(net, rng.normal(size=(10, 17)), rng.normal(size=10), [0.1], [0.1])
    with pytest.raises(ValueError, match="unsupported size"):
        cuda.run_grid(grid, np.zeros(net.n_parameters), 600)
    with pytest.raises(TypeError):
        cuda.run_grid(object(), np.zeros(3), 600)


def test_iterate_and_screen_signature_is_the_option_source():
    # A changed CPU default must flow to the GPU path, not be duplicated.
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(iterate_and_screen).parameters.items()
                if v.default is not inspect.Parameter.empty}
    assert gc.OPTION_DEFAULTS == defaults
