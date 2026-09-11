"""Synthetic orchestration checks: no EXP-005 target trajectories are run."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("run_hopfield_controls", TOOLS / "run_hopfield_controls.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


@pytest.fixture
def plan():
    return json.loads(controls.PLAN.read_text())


def synthetic_cell(plan, point, dt, period=1, ftle=.01):
    events = np.full((128, 4), np.nan)
    events[:4] = np.column_stack((np.arange(1., 5.), np.ones((4, 3))))
    stats = np.zeros(20)
    stats[0:4] = [4., ftle, 1900., 2000.]
    stats[16] = 1.
    stats[18:20] = [round(10000/dt), round(2000/dt)]
    return dict(alpha=point["alpha"], gamma=point["gamma"], dt=dt,
                initial=np.array(plan["initial_state"]), final=np.full(3, 7.),
                events=events, stats=stats, n_events=4, period=period,
                residual=0., period_time=1.)


def test_plan_is_fixed_and_has_eight_independent_cells(plan):
    assert controls.validate(plan) == 8
    broken = copy.deepcopy(plan)
    broken["dt"] = [.02, .01]
    with pytest.raises(AssertionError):
        controls.validate(broken)


def test_period_and_irregular_gates_require_both_steps(plan):
    a = synthetic_cell(plan, plan["points"][0], .01)
    b = synthetic_cell(plan, plan["points"][0], .005)
    assert controls.common_period([a, b]) == 1
    b["period"] = 2
    assert controls.common_period([a, b]) == 0
    a["period"] = b["period"] = 0
    assert controls.finite_horizon_irregular([a, b], .005)
    b["stats"][1] = .005
    assert not controls.finite_horizon_irregular([a, b], .005)
    b["stats"][1] = np.nan
    assert not controls.finite_horizon_irregular([a, b], .005)


def test_each_cell_restarts_and_saved_output_matches(monkeypatch, tmp_path, plan):
    calls = []
    def fake_cell(fn, alpha, gamma, initial, p, dt, failure_path):
        calls.append((alpha, gamma, dt, initial.copy()))
        c = synthetic_cell(p, dict(alpha=alpha, gamma=gamma), dt)
        return tuple(c[key] for key in ["final", "events", "stats", "n_events", "period", "residual", "period_time"])
    monkeypatch.setattr(controls.base, "cell", fake_cell)
    controls.execute_cells(None, plan, tmp_path)
    assert len(calls) == 8
    for (alpha, gamma, dt, initial), (point, step) in zip(calls, [(p, d) for p in plan["points"] for d in plan["dt"]]):
        assert (alpha, gamma, dt) == (point["alpha"], point["gamma"], step)
        np.testing.assert_array_equal(initial, [.1, .1, .1])
        saved = controls.read_cell(controls.cell_path(tmp_path, point["name"], step))
        controls.verify_cell(saved, point, step, plan)


def test_second_cell_failure_retains_first(monkeypatch, tmp_path, plan):
    calls = 0
    def fake_cell(fn, alpha, gamma, initial, p, dt, failure_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            np.savez_compressed(failure_path, native_status=-2)
            raise RuntimeError("Injected native failure, no integration")
        c = synthetic_cell(p, dict(alpha=alpha, gamma=gamma), dt)
        return tuple(c[key] for key in ["final", "events", "stats", "n_events", "period", "residual", "period_time"])
    monkeypatch.setattr(controls.base, "cell", fake_cell)
    with pytest.raises(RuntimeError, match="Injected"):
        controls.execute_cells(None, plan, tmp_path)
    assert controls.cell_path(tmp_path, "core", .01).exists()
    assert (tmp_path / "cells/core-dt-0.005-failure.npz").exists()
    assert json.loads((tmp_path / "progress.json").read_text())["dt"] == .005


def test_certificate_uses_finest_step_and_does_not_require_old_period(monkeypatch, plan):
    point = plan["points"][0]
    coarse = synthetic_cell(plan, point, .01, period=1)
    fine = synthetic_cell(plan, point, .005, period=1)
    fine["events"][3, 1:] = [9., 8., 7.]
    seeds = []
    def fake_certify(alpha, gamma, state, duration, q):
        seeds.append((state.copy(), duration, q))
        return {"verified": False, "attracting": False, "reason": "Synthetic audit stub"}
    monkeypatch.setattr(controls.base, "certify", fake_certify)
    result = controls.diagnose_point(point, [fine, coarse], plan)
    assert result["shared_peak_period"] == 1
    np.testing.assert_array_equal(seeds[0][0], [9., 8., 7.])
    coarse["period"] = 2
    result = controls.diagnose_point(point, [coarse, fine], plan)
    assert result["shared_peak_period"] == 0 and len(seeds) == 1


def test_raw_cell_verification_rejects_wrong_timing_or_event_period(plan):
    point = plan["points"][0]
    cell = synthetic_cell(plan, point, .01)
    controls.verify_cell(cell, point, .01, plan)
    cell["stats"][3] = 1999.
    with pytest.raises(AssertionError):
        controls.verify_cell(cell, point, .01, plan)
    cell["stats"][3] = 2000.
    cell["period_time"] = 2.
    with pytest.raises(AssertionError):
        controls.verify_cell(cell, point, .01, plan)
