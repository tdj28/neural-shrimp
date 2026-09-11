"""EXP-006 orchestration audits with stubs only; no new target trajectories."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hopfield_drift_probe as probe
import prepare_hopfield_drift as preparation
import run_hopfield_drift as driver


@pytest.fixture
def plan():
    return json.loads(driver.PLAN.read_text())


def fake_cell_result(dt, period=9):
    events = np.full((128, 4), np.nan)
    events[:36] = np.column_stack((np.arange(36.), np.tile([1., 2., dt], (36, 1))))
    stats = np.zeros(20)
    stats[0:4] = [36., 0., 1900., 2000.]
    stats[16] = 1.
    stats[18:] = [round(10000 / dt), round(2000 / dt)]
    return np.ones(3), events, stats, 36, period, 0., float(period)


def test_frozen_census_has_every_matched_arm_phase_dose_and_step(plan):
    specs = list(driver.trial_specs(plan))
    assert driver.validate(plan) == len(specs) == 4320
    assert len({driver.trial_id(s) for s in specs}) == len(specs)
    for anchor in plan["anchors"]:
        for phase in plan["phases"]:
            for dt in plan["dt"]:
                paired = [s for s in specs if s["anchor"] == anchor["id"] and s["phase"] == phase and s["dt"] == dt]
                assert len(paired) == 45
                assert sum(s["radius"] == 0. and s["arm"] == "zero" for s in paired) == 1
                for radius in plan["radii"]:
                    assert {s["arm"] for s in paired if s["radius"] == radius} == set(plan["arms"])


def test_all_policy_directions_have_matched_raw_weight_dose(plan):
    local = np.array([.6, .8])
    for anchor in plan["anchors"]:
        for radius in plan["radii"]:
            choices = driver.directions(anchor, local, radius)
            assert set(choices) == set(driver.ARMS)
            for direction in choices.values():
                assert np.linalg.norm(radius * direction) == pytest.approx(radius, rel=1e-12)
            np.testing.assert_array_equal(choices["local_minus"], -choices["local_plus"])
            np.testing.assert_array_equal(choices["random_minus"], -choices["random_plus"])


def test_trial_short_and_long_calls_share_initial_parameters_and_phase_epoch(monkeypatch, tmp_path, plan):
    (tmp_path / "trials").mkdir()
    calls = []
    initial = np.array([.13, -.09, .2])
    times = np.linspace(0., 300., 3001)
    states = np.tile(initial, (len(times), 1))
    period, epoch = 9., 9. / 4
    def fake_trace(fn, alpha, gamma, state, dt, **kwargs):
        calls.append(("short", alpha, gamma, state.copy(), dt))
        assert kwargs["duration"] == 300 and kwargs["sample_dt"] == .1
        return times, states
    def fake_long(fn, alpha, gamma, state, p, dt, **kwargs):
        calls.append(("long", alpha, gamma, state.copy(), dt))
        return fake_cell_result(dt)
    def fake_wave(reference, candidate_times, activations, supplied_period, **kwargs):
        np.testing.assert_array_equal(candidate_times, times + epoch)
        np.testing.assert_array_equal(activations, np.tanh(states))
        assert supplied_period == period and kwargs["phase_grid"] == plan["phase_grid"]
        return {"aligned_rms": {"aggregate": .1}}
    monkeypatch.setattr(probe, "integrate_trace", fake_trace)
    monkeypatch.setattr(driver.base, "cell", fake_long)
    monkeypatch.setattr(driver, "waveform_errors", fake_wave)
    anchor = plan["anchors"][0]
    for dt in plan["dt"]:
        for arm in plan["arms"]:
            spec = dict(anchor=anchor["id"], phase=1, dt=dt, radius=.001, arm=arm)
            record = driver.execute_trial(spec, anchor, initial, np.array([1., 0.]),
                                          np.zeros((8, 3)), period, epoch, None, None, plan, tmp_path)
            assert record["status"] == "complete"
            short, long = calls[-2:]
            assert short[1:3] == long[1:3] == (record["alpha"], record["gamma"])
            assert short[4] == long[4] == dt
            np.testing.assert_array_equal(short[3], initial)
            np.testing.assert_array_equal(long[3], initial)
            with np.load(tmp_path / record["raw"], allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw["initial"], initial)
                assert np.linalg.norm(raw["delta"]) == pytest.approx(.001, rel=1e-12)


def test_long_failure_preserves_already_written_short_trace(monkeypatch, tmp_path, plan):
    (tmp_path / "trials").mkdir()
    initial = np.array([.13, -.09, .2])
    times, states = np.array([0., 300.]), np.array([initial, initial * 2])
    monkeypatch.setattr(probe, "integrate_trace", lambda *args, **kwargs: (times, states))
    def fail_long(*args, **kwargs):
        np.savez_compressed(kwargs["failure_path"], native_status=-2, failed=True)
        raise RuntimeError("Synthetic failure before any target integration")
    monkeypatch.setattr(driver.base, "cell", fail_long)
    anchor = plan["anchors"][0]
    spec = dict(anchor=anchor["id"], phase=0, dt=.005, radius=0., arm="zero")
    row = driver.execute_trial(spec, anchor, initial, np.array([1., 0.]),
                               np.zeros((8, 3)), 9., 0., None, None, plan, tmp_path)
    assert row["status"] == "failed"
    with np.load(tmp_path / row["raw"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["states"], states)
        assert "final" not in saved
    assert len(list((tmp_path / "trials").glob("*-native-failure.npz"))) == 1


def test_atomic_payload_write_preserves_previous_file_and_failed_partial(monkeypatch, tmp_path):
    path = tmp_path / "trial.npz"
    driver.atomic_npz(path, short_trace=np.arange(6.))
    original_bytes = path.read_bytes()
    def failed_write(stream, **kwargs):
        stream.write(b"synthetic interrupted replacement")
        raise OSError("Injected write failure")
    monkeypatch.setattr(driver.np, "savez_compressed", failed_write)
    with pytest.raises(OSError, match="Injected"):
        driver.atomic_npz(path, short_trace=np.arange(6.), long_trace=np.arange(10.))
    assert path.read_bytes() == original_bytes
    assert path.with_name("trial.partial.npz").read_bytes() == b"synthetic interrupted replacement"
    with pytest.raises(FileExistsError):
        driver.atomic_npz(path, retry="must not erase partial evidence")


@pytest.fixture
def completed_stub_trial(monkeypatch, tmp_path, plan):
    (tmp_path / "trials").mkdir()
    anchor, local = plan["anchors"][0], np.array([1., 0.])
    phases = np.linspace(0., 1., 9)
    orbit = np.column_stack((.2 * np.sin(2*np.pi*phases),
                             .3 * np.cos(2*np.pi*phases),
                             .4 * np.sin(4*np.pi*phases + .2)))
    cert = dict(period=8., orbit_states=orbit.tolist(), orbit_activations=np.tanh(orbit).tolist())
    initial = orbit[2].copy()
    times = np.linspace(0., 300., 3001)
    states = np.tile(initial, (len(times), 1))
    monkeypatch.setattr(probe, "integrate_trace", lambda *args, **kwargs: (times, states))
    monkeypatch.setattr(driver.base, "cell", lambda *args, dt, **kwargs: fake_cell_result(dt, period=1))
    spec = dict(anchor=anchor["id"], phase=1, dt=.005, radius=.001, arm="outward")
    row = driver.execute_trial(spec, anchor, initial, local, np.tanh(orbit[:-1]),
                               8., 2., None, None, plan, tmp_path)
    assert row["status"] == "complete"
    with np.load(tmp_path / row["raw"], allow_pickle=False) as saved:
        raw = {key: saved[key].copy() for key in saved.files}
    return row, raw, anchor, cert, local, plan


def test_raw_verifier_rederives_a_completed_synthetic_trial(completed_stub_trial):
    driver.verify_complete_trial(*completed_stub_trial)


@pytest.mark.parametrize("change", ["swapped_arm", "wrong_phase", "altered_metric", "altered_initial"])
def test_raw_verifier_rejects_action_phase_and_metric_mismatches(completed_stub_trial, change):
    row, raw, anchor, cert, local, plan = completed_stub_trial
    if change == "swapped_arm":
        row["arm"] = "inward"
        row["id"] = driver.trial_id(row)
        row["raw"] = f"trials/{row['id']}.npz"
    elif change == "wrong_phase":
        row["phase"] = 2
        row["baseline_phase_epoch"] = 4.
        row["id"] = driver.trial_id(row)
        row["raw"] = f"trials/{row['id']}.npz"
    elif change == "altered_metric":
        row["wave_metrics"]["aligned_rms"]["aggregate"] += .01
    else:
        raw["initial"][0] += .01
    with pytest.raises(AssertionError):
        driver.verify_complete_trial(row, raw, anchor, cert, local, plan)


def test_metric_verifier_accepts_equivalent_phase_and_optimizer_counts():
    reference = np.zeros((32, 3))
    times = np.linspace(0., 2., 21)
    activations = np.full((21, 3), .2)
    saved = driver.waveform_errors(reference, times, activations, 1.)
    # A constant waveform has no identifiable phase; every phase has equal RMS.
    saved["phase_offset_cycles"] = .31
    saved["phase_search"]["refinement_evaluations"] += 1
    driver.verify_wave_metrics(saved, reference, times, activations, 1., 128)


def test_baseline_restarts_steps_and_certifies_only_shared_nine(monkeypatch, tmp_path, plan):
    calls, seeds = [], []
    def fake_cell(fn, alpha, gamma, initial, p, dt, **kwargs):
        calls.append((dt, copy.deepcopy(initial)))
        return fake_cell_result(dt)
    def fake_certify(alpha, gamma, state, period, expected):
        seeds.append((state.copy(), period, expected))
        return dict(verified=True, attracting=True)
    monkeypatch.setattr(driver.base, "cell", fake_cell)
    monkeypatch.setattr(driver.base, "certify", fake_certify)
    result = driver.baseline(None, plan["anchors"][0], plan, tmp_path)
    assert result["eligible"]
    assert calls == [(.01, [.1, .1, .1]), (.005, [.1, .1, .1])]
    np.testing.assert_array_equal(seeds[0][0], [1., 2., .005])
    assert seeds[0][1:] == (9., 9)
    monkeypatch.setattr(driver.base, "cell", lambda *args, dt, **kwargs: fake_cell_result(dt, 18 if dt == .01 else 9))
    result = driver.baseline(None, plan["anchors"][1], plan, tmp_path)
    assert not result["eligible"] and len(seeds) == 1


@pytest.mark.parametrize("dirty,pushed", [(True, True), (False, False)])
def test_source_gate_rejects_before_creating_target_attempt(monkeypatch, tmp_path, dirty, pushed):
    def fake_git(*args):
        if args[0] == "status": return "modified-source" if dirty else ""
        if args[0] == "rev-parse": return "synthetic-source"
        if args[0] == "branch": return "origin/frozen" if pushed else ""
        raise AssertionError(args)
    monkeypatch.setattr(driver.base, "git", fake_git)
    monkeypatch.setattr(driver.base, "build_engine", lambda *args: pytest.fail("Target build reached before source gate"))
    with pytest.raises(AssertionError):
        driver.run(tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_stubbed_full_runner_retains_4320_records_and_phase_initial_states(monkeypatch, tmp_path, plan):
    p = copy.deepcopy(plan)
    p["source_artifacts"] = []
    plan_path = tmp_path / "stub-plan.json"
    plan_path.write_text(json.dumps(p))
    monkeypatch.setattr(driver, "PLAN", plan_path)
    monkeypatch.setattr(driver.base, "git", lambda *args: "" if args[0] == "status" else "synthetic-frozen-source")
    monkeypatch.setattr(driver.base, "build_engine", lambda *args: (None, {"synthetic": True}))
    monkeypatch.setattr(probe, "build_trace", lambda *args: (None, {"synthetic": True}))
    monkeypatch.setattr(driver, "run_continuations", lambda *args: None)
    monkeypatch.setattr(driver, "verify", lambda *args: None)
    orbit = np.column_stack((np.arange(9.), np.arange(9.) + 20., np.arange(9.) + 40.))
    def fake_baseline(fn, anchor, p, output):
        return dict(anchor=anchor["id"], eligible=anchor["id"] == p["anchors"][0]["id"],
                    certificate=dict(period=8., orbit_states=orbit.tolist(),
                                     orbit_activations=np.tanh(orbit).tolist()))
    monkeypatch.setattr(driver, "baseline", fake_baseline)
    sensitivity_initials, trial_calls = [], []
    def fake_sensitivity(alpha, gamma, initial, times):
        sensitivity_initials.append(initial.copy())
        assert times[0] == 0 and times[-1] == 300 and len(times) == 3001
        return dict(minimum_eigenvector=np.array([1., 0.]), synthetic=True)
    monkeypatch.setattr(probe, "local_parameter_sensitivity", fake_sensitivity)
    def fake_trial(spec, anchor, initial, local, reference, period, epoch, *args):
        np.testing.assert_array_equal(initial, orbit[2 * spec["phase"]])
        np.testing.assert_array_equal(reference, np.tanh(orbit[:-1]))
        assert epoch == spec["phase"] * period / 4
        trial_calls.append(spec.copy())
        return {**spec, "id": driver.trial_id(spec), "status": "complete",
                "wave_metrics": {"aligned_rms": {"aggregate": .1}}}
    monkeypatch.setattr(driver, "execute_trial", fake_trial)
    output = tmp_path / "stub-attempt"
    driver.run(output)
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert len(records) == len({r["id"] for r in records}) == 4320
    assert len(trial_calls) == 360
    assert sum(r["status"] == "baseline_not_evaluable" for r in records) == 3960
    np.testing.assert_array_equal(sensitivity_initials, orbit[[0, 2, 4, 6]])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["anchor_attempts"] == 12 and summary["eligible_anchor_count"] == 1
    assert summary["anchor_coverage"] == 1 / 12


def test_preparation_reads_prior_evidence_only_and_never_integrates(monkeypatch, plan):
    packet = copy.deepcopy(plan["atlas_selection"])
    packet["anchors"] = copy.deepcopy(plan["anchors"])
    monkeypatch.setattr(preparation, "load_anchor_packet", lambda *args, **kwargs: packet)
    monkeypatch.setattr(preparation.base, "load_grid", lambda *args: {"stub": True})
    calls = []
    def fake_margin(grid, anchor, radius, angle_count):
        assert grid == {"stub": True} and angle_count == 512
        calls.append((anchor["id"], radius))
        return {"direction": [1., 0.]}
    monkeypatch.setattr(preparation, "atlas_margin_direction", fake_margin)
    monkeypatch.setattr(preparation.base, "cell", lambda *args, **kwargs: pytest.fail("Preparation integrated target"))
    monkeypatch.setattr(preparation.base, "certify", lambda *args, **kwargs: pytest.fail("Preparation certified target"))
    compiled = preparation.compile_plan()
    assert len(calls) == 48 and driver.validate(compiled) == 4320
    assert all(a["atlas_margin"][str(r)]["direction"] == [1., 0.]
               for a in compiled["anchors"] for r in compiled["radii"])
