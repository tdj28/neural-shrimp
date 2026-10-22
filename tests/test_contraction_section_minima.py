import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from contraction_section_minima import search_section_minima


class Objective:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __call__(self, s):
        self.calls.append(s)
        return {"radius": self.function(s), "index": len(self.calls)-1}


def search(objective, mesh=(-1, -.5, 0, .5, 1), **changes):
    settings = dict(coordinate_tolerance=1e-7, max_evaluations=200,
                    radius_floor=1e-12, signal_multiple=10,
                    validation_offsets=(.02, .01))
    settings.update(changes)
    result = search_section_minima(objective, mesh, **settings)
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("function,location", [
    (lambda s: 1 + (s-.137)**2, .137),
    (lambda s: 1 + abs(s-.137), .137),
    (lambda s: 1 + abs(s+.119) + .3*(s+.119), -.119),
])
def test_smooth_and_nonsmooth_minima_are_evaluated_not_fitted(function, location):
    objective = Objective(function)
    result = search(objective)
    assert result["complete"] and result["status"] == "converged"
    assert len(result["minima"]) == 1
    minimum = result["minima"][0]
    assert abs(minimum["best_sample"]["s"]-location) < 1e-7
    assert minimum["refinement"]["width"] <= 1e-7
    assert len(minimum["validation"]) == 2
    assert all(len(row["samples"]) == 2 and row["resolved"] for row in minimum["validation"])
    for row in result["evaluations"]:
        assert row["radius"] == function(row["s"])
        assert objective.calls[row["index"]] == row["s"]
    assert len(objective.calls) == len(set(objective.calls)) == result["evaluation_count"]


def test_two_minima_are_both_retained_and_refined():
    objective = Objective(lambda s: 1 + (s*s-1)**2)
    result = search(objective, mesh=(-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2))
    assert result["status"] == "converged"
    assert len(result["mesh_brackets"]) == len(result["minima"]) == 2
    assert [m["best_sample"]["s"] for m in result["minima"]] == pytest.approx([-1, 1], abs=1e-7)
    assert all(m["refinement"]["width"] <= 1e-7 for m in result["minima"])


@pytest.mark.parametrize("function", [lambda s: 1, lambda s: 2+s, lambda s: 2-s,
                                      lambda s: 1 + max(0, abs(s)-.5)])
def test_flat_monotonic_and_plateau_do_not_default_to_endpoint_or_argmin(function):
    objective = Objective(function)
    result = search(objective)
    assert result["complete"] and result["status"] == "no_bracket"
    assert result["mesh_brackets"] == result["candidates"] == result["minima"] == []
    assert len(objective.calls) == 5


def test_noisy_weak_brackets_are_preserved_without_spending_refinement_budget():
    objective = Objective(lambda s: 1 + 1e-10*math.cos(2*math.pi*s))
    mesh = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2)
    result = search(objective, mesh, radius_floor=1e-9)
    assert result["status"] == "low_signal" and result["complete"]
    assert len(result["mesh_brackets"]) == len(result["candidates"]) == 4
    assert not result["minima"]
    assert all(c["status"] == "low_signal" and not c["validation"] for c in result["candidates"])
    assert len(objective.calls) == len(mesh)


def test_small_offset_signal_gate_is_not_replaced_by_large_scale_signal():
    objective = Objective(lambda s: 1+s*s)
    result = search(objective, radius_floor=1e-6, signal_multiple=10,
                    validation_offsets=(.1, .001))
    candidate = result["candidates"][0]
    assert candidate["refinement"]["status"] == "converged"
    assert [r["resolved"] for r in candidate["validation"]] == [True, False]
    assert candidate["status"] == result["status"] == "low_signal"
    assert not result["minima"]


def test_resolved_lower_neighbor_is_not_mislabeled_low_signal():
    # An unresolved narrow extra well can be absent from the mesh/refinement
    # samples but detected by the actual finite-offset check. Keep that failure.
    objective = Objective(lambda s: 1+abs(s)-.6*math.exp(-((s-.02)/1e-6)**2))
    result = search(objective)
    candidate = result["candidates"][0]
    assert candidate["refinement"]["status"] == "converged"
    assert any(row["lower_neighbor_resolved"] for row in candidate["validation"])
    assert result["status"] == candidate["status"] == "nonconverged"
    assert "lower neighboring" in candidate["reason"] and not result["minima"]


def test_budget_preserves_all_mesh_brackets_and_partial_refinement():
    objective = Objective(lambda s: 1+(s*s-1)**2)
    mesh = (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2)
    result = search(objective, mesh, max_evaluations=len(mesh)+2)
    assert result["status"] == "budget_exhausted" and not result["complete"]
    assert len(result["mesh_brackets"]) == len(result["candidates"]) == 2
    assert len(objective.calls) == len(mesh)+2
    assert result["candidates"][0]["refinement"]["steps"] == 2
    assert all(c["status"] == "budget_exhausted" for c in result["candidates"])


def test_partial_mesh_never_claims_no_bracket():
    result = search(Objective(lambda s: 1+s*s), max_evaluations=3)
    assert not result["mesh_complete"] and not result["complete"]
    assert result["status"] == "budget_exhausted"
    assert len(result["mesh_samples"]) == result["evaluation_count"] == 3


def test_step_budget_nonconvergence_is_distinct_from_callback_budget():
    result = search(Objective(lambda s: 1+(s-.137)**2), max_refinement_steps=1)
    assert result["status"] == "nonconverged" and not result["complete"]
    assert result["evaluation_count"] == 6
    assert result["candidates"][0]["refinement"]["width"] > 1e-7
    assert not result["minima"] and not result["candidates"][0]["validation"]


def test_validation_may_leave_refinement_bracket_within_section_support():
    objective = Objective(lambda s: 1+abs(s-.2))
    result = search(objective, validation_offsets=(.4, .1))
    candidate = result["candidates"][0]
    assert result["status"] == "converged"
    assert candidate["bracket"]["right"]["s"] == .5
    assert candidate["validation"][0]["samples"][1]["s"] > .5
    assert all(-1 <= s <= 1 for s in objective.calls)


def test_validation_cannot_leave_the_overall_declared_section_support():
    objective = Objective(lambda s: 1+abs(s-.2))
    result = search(objective, validation_offsets=(.9, .1))
    assert result["status"] == "nonconverged"
    assert "leave section support" in result["candidates"][0]["reason"]
    assert not result["candidates"][0]["validation"]
    assert all(-1 <= s <= 1 for s in objective.calls)


@pytest.mark.parametrize("tolerance,expected", [(1e-4, "converged"), (1e-12, "budget_exhausted")])
def test_declared_forty_attempt_budget_is_configuration_not_a_new_default(tolerance, expected):
    objective = Objective(lambda s: 1+abs(s-.137))
    result = search(objective, max_evaluations=40, coordinate_tolerance=tolerance)
    assert result["settings"]["max_evaluations"] == 40
    assert result["status"] == expected
    assert result["evaluation_count"] == len(objective.calls) <= 40
    if expected == "budget_exhausted":
        assert len(objective.calls) == 40 and not result["complete"]
    else:
        assert result["complete"] and len(result["minima"]) == 1


def test_actual_mesh_endpoint_evaluations_are_reused_for_validation():
    objective = Objective(lambda s: 1+abs(s))
    result = search(objective, mesh=(-2, -1, 0, 1, 2), validation_offsets=(1, .25))
    assert result["status"] == "converged"
    assert result["cache_hits"] >= 2
    assert objective.calls.count(-1) == objective.calls.count(1) == 1
    assert len(objective.calls) == len(set(objective.calls))


@pytest.mark.parametrize("bad", [
    {}, {"radius": float("nan"), "index": 0}, {"radius": -1, "index": 0},
    {"radius": 1, "index": None}, {"radius": 1, "index": 0, "qualified": False},
])
def test_bad_or_unqualified_objective_is_a_retained_failure(bad):
    result = search(lambda s: bad)
    assert result["status"] == "evaluation_failed" and not result["complete"]
    assert result["evaluation_count"] == 1 and result["evaluations"][0]["status"] == "failed"
    assert not result["minima"]


def test_callback_failure_and_events_preserve_attempted_coordinate():
    events = []
    def fail(s):
        raise RuntimeError("orbit qualification failed")
    result = search(fail, record=events.append)
    assert result["evaluations"][0]["s"] == -1
    assert result["evaluations"][0]["error_type"] == "RuntimeError"
    assert events[0]["event"] == "evaluation_started"
    assert events[1]["event"] == "evaluation_failed"
    assert events[-1]["status"] == "evaluation_failed"
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"coordinate_tolerance": 0}, {"radius_floor": -1}, {"signal_multiple": .5},
    {"validation_offsets": (.1,.1)}, {"validation_offsets": (.1,)},
    {"validation_offsets": (.1,1e-8)}, {"max_evaluations": 0},
    {"max_refinement_steps": True},
])
def test_invalid_settings_fail_before_any_evaluation(changes):
    objective = Objective(lambda s: 1+s*s)
    with pytest.raises(ValueError):
        search(objective, **changes)
    assert not objective.calls


@pytest.mark.parametrize("mesh", [(-1,0), (-1,0,0,1), (1,0,-1), (-1,float("nan"),1),
                                   (-1e308,0,1e308)])
def test_invalid_mesh_fails_before_any_evaluation(mesh):
    objective = Objective(lambda s: 1+s*s)
    with pytest.raises(ValueError):
        search(objective, mesh)
    assert not objective.calls
