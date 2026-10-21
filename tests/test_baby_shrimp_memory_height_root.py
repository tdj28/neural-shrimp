"""Analytic moving saddles only: root persistence and saved-linkage checks."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_finish as audit
import baby_shrimp_memory_height_root as roots
import baby_shrimp_saddle_memory as driver
from trace_saddle_continuation import ContinuationError, TraceSaddleContinuation


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def execute(tmp_path, monkeypatch, *, positive=False, fail=False, target_failure=False, iterations=10):
    run = tmp_path / "run"
    raw = {}
    class Evaluator:
        def __init__(self, args, plan, out, emit):
            self.out, self.attempts = out, 0
        def __call__(self, center, beta, cyclic=False):
            if target_failure and beta == 1.:
                raise TimeoutError("synthetic target budget expired")
            if fail and beta not in (0., 1.):
                raise TimeoutError("synthetic global budget expired")
            center = np.asarray(center)
            x, y = center - [.3*beta, -.2*beta]
            height = (.1+beta*beta) if positive else (beta+.2*beta*beta-.6)
            tau = float(height-2*x*x+3*y*y)
            index = self.attempts
            self.attempts += 1
            row = dict(index=index, coordinate=center.tolist(), beta=beta, qualified=True,
                       tau=tau, delta=-1e-8, radius=.5,
                       phases=[dict(phase=s, tau=tau) for s in ((0, 6, 12) if cyclic else (0,))])
            directory = self.out / f"point-{index:04d}"
            directory.mkdir()
            write(directory / "point.json", row)
            raw[index] = row
            return row
        def baseline(self):
            return dict(passed=True, point=self([0., 0.], 0., cyclic=True))
    monkeypatch.setattr(driver, "Evaluator", Evaluator)
    monkeypatch.setattr(driver, "print", lambda *args, **kwargs: None, raising=False)
    plan = tmp_path / "plan.json"
    write(plan, {"maximum_evaluations": 1000})
    monkeypatch.setattr(sys, "argv", ["driver", "--plan", str(plan), "--out", str(run), "--mode", "continue",
        "--source-commit", "a"*40, "--betas", "0", "1", "--stencils", ".1", ".03",
        "--correction-tolerance", "1e-9", "--height-root", "--height-tolerance", "1e-10",
        "--height-parameter-tolerance", "1e-12", "--height-root-iterations", str(iterations),
        "--height-root-interior-fraction", ".001", "--height-root-progress-safeguard"])
    error = None
    try:
        driver.main()
    except (ValueError, RuntimeError, TimeoutError) as caught:
        error = caught
    protocol = audit.read(run / "protocol.json")
    result = audit.read(run / ("result.json" if (run / "result.json").exists() else "continuation.json"))
    centers, _ = audit.accepted_centers(run, protocol, raw, result)
    return run, protocol, result, centers, raw, error


def test_driver_never_overwrites_root_callbacks_and_keeps_fresh_targets_separate(tmp_path, monkeypatch):
    run, protocol, result, centers, raw, error = execute(tmp_path, monkeypatch)
    assert error is None
    checked = roots.verify_height_root(run, protocol, result, centers, raw)
    assert checked["complete"] and len(checked["iterations"]) > 1
    assert [p["parameter"] for p in result["points"]] == [0., 1.]
    assert len(list(run.glob("saddle-*.json"))) == 2+len(checked["iterations"])
    assert checked["saddle_artifacts"] == [f"saddle-{i:03d}.json" for i in range(2, 2+len(checked["iterations"]))]
    assert abs(checked["result"]["saddle"]["tau"]) <= 1e-10
    assert "tools/baby_shrimp_memory_height_root.py" in protocol["source_sha256"]
    assert "tools/baby_shrimp_memory_finish.py" in protocol["source_sha256"]


def test_failed_target_before_root_stage_remains_auditable(tmp_path, monkeypatch):
    run, protocol, result, centers, raw, error = execute(tmp_path, monkeypatch, target_failure=True)
    assert "target budget" in str(error) and not (run / "height-root.json").exists()
    checked = roots.verify_height_root(run, protocol, result, centers, raw)
    assert not checked["complete"] and "not started" in checked["status"]
    assert checked["recorded_target_parameters"] == [0.] and len(centers) == 1
    altered = {**result, "complete": True}
    with pytest.raises(ValueError, match="lacks its saved root receipt"):
        roots.verify_height_root(run, protocol, altered, centers, raw)


def test_no_sign_bracket_fails_after_preserving_all_fresh_targets(tmp_path, monkeypatch):
    run, protocol, result, centers, raw, error = execute(tmp_path, monkeypatch, positive=True)
    assert "no strict sign-changing" in str(error)
    assert [p["parameter"] for p in result["points"]] == [0., 1.]
    assert not (run / "result.json").exists()
    checked = roots.verify_height_root(run, protocol, result, centers, raw)
    assert not checked["complete"] and checked["failure"] and checked["initial_bracket"] is None


@pytest.mark.parametrize("failure", ["budget", "iterations"])
def test_partial_root_failure_retains_bracket_and_completed_callbacks(tmp_path, monkeypatch, failure):
    run, protocol, result, centers, raw, error = execute(tmp_path, monkeypatch,
        fail=failure == "budget", iterations=1 if failure == "iterations" else 10)
    assert error is not None and (run / "failure.json").exists()
    checked = roots.verify_height_root(run, protocol, result, centers, raw)
    assert not checked["complete"] and checked["initial_bracket"] and checked["failure"]
    assert checked["result"] is None and len(result["points"]) == 2
    assert len(checked["iterations"]) == (1 if failure == "iterations" else 0)


@pytest.mark.parametrize("change,match", [("artifact", "ordering"), ("bracket", "initial bracket"),
                                         ("root", "saddle linkage"), ("iteration", "candidate differs"),
                                         ("progress", "candidate differs"), ("missing-root-field", "saddle linkage")])
def test_tampered_root_linkage_is_rejected(tmp_path, monkeypatch, change, match):
    run, protocol, result, centers, raw, error = execute(tmp_path, monkeypatch)
    assert error is None
    record = audit.read(run / "height-root.json")
    if change == "artifact":
        record["iterations"][0]["artifact"] = "saddle-000.json"
    elif change == "bracket":
        record["initial_bracket"][0]["tau"] += .1
    elif change == "root":
        record["result"]["saddle"]["center"][0] += .1
    elif change == "missing-root-field":
        del record["result"]["saddle"]["history"]
    elif change == "progress":
        record["iterations"][0]["progress_secant_eligible"] = True
    else:
        record["iterations"][0]["parameter"] += .01
    write(run / "height-root.json", record)
    result["height_root"] = record
    with pytest.raises(ValueError, match=match):
        roots.verify_height_root(run, protocol, result, centers, raw)


def test_parent_candidates_use_fresh_replay_height_and_preserve_receipt_binding(tmp_path):
    sources = [dict(parameter=b, center=[0., 0.], tau=99., cyclic_point_index=i) for i, b in enumerate((0., 1.))]
    provenance = dict(endpoints=sources, receipt_sha256="a"*64, raw_manifest_sha256="b"*64)
    replay = dict(passed=True, endpoints=[dict(source=p, point_index=i) for i, p in enumerate(sources)])
    raw = {i: dict(index=i, coordinate=p["center"], beta=p["parameter"], qualified=True,
                   tau=(-1. if i == 0 else 1.), phases=[dict(phase=s) for s in (0, 6, 12)])
           for i, p in enumerate(sources)}
    rows = roots.candidates(tmp_path, [], provenance=provenance, replay=replay, raw=raw)
    assert [r["tau"] for r in rows] == [-1., 1.]
    assert roots.select_bracket(rows) == rows
    assert all(r["source"]["receipt_sha256"] == "a"*64 for r in rows)
    altered = copy.deepcopy(replay)
    altered["endpoints"][0]["source"]["center"][0] = 1.
    with pytest.raises(ValueError, match="replay source differs"):
        roots.candidates(tmp_path, [], provenance=provenance, replay=altered, raw=raw)


def test_residual_progress_control_and_fraction_only_adverse_control():
    def evaluate(center, beta):
        x, y = center - [.3*beta, -.2*beta]
        return beta+.2*beta*beta-.6-2*x*x+3*y*y
    results = {}
    for fraction, progress in ((.1, False), (.001, False), (.001, True)):
        engine = TraceSaddleContinuation(evaluate, correction_tolerance=1e-9)
        endpoints = [engine.correct(beta, [.3*beta, -.2*beta]) for beta in (0., 1.)]
        before = len(engine.evaluations)
        result = engine.solve_height_zero(*endpoints, height_tolerance=1e-10,
            parameter_tolerance=1e-12, max_iterations=30, interior_fraction=fraction, progress_safeguard=progress)
        results[(fraction, progress)] = (result, len(engine.evaluations)-before)
    exact = (np.sqrt(1.48)-1)/.4
    for result, _ in results.values():
        assert abs(result["saddle"]["tau"]) <= 1e-10
        assert result["saddle"]["parameter"] == pytest.approx(exact, abs=1e-10)
    # Retain the adverse control: the smaller fraction alone costs MORE here.
    assert results[(.001, False)][1] > results[(.1, False)][1]
    assert results[(.001, True)][1] <= results[(.1, False)][1]/2
    assert len(results[(.001, True)][0]["history"]) <= 10
    print({str(setting): {"iterations": len(row[0]["history"]), "evaluations": row[1],
                         "height_residual": row[0]["saddle"]["tau"]} for setting, row in results.items()})


def test_stagnating_false_position_cannot_claim_progress_or_escape_iteration_budget():
    def evaluate(center, beta):
        return beta**10-1e-6-center[0]**2+center[1]**2
    engine = TraceSaddleContinuation(evaluate)
    endpoints = [engine.correct(beta, [0., 0.]) for beta in (0., 1.)]
    before = len(engine.evaluations)
    with pytest.raises(ContinuationError, match="did not converge"):
        engine.solve_height_zero(*endpoints, height_tolerance=1e-10, parameter_tolerance=1e-12,
                                 max_iterations=10, interior_fraction=.001, progress_safeguard=True)
    history = [e for e in engine.events if e["stage"] == "height_root_iteration"]
    assert len(history) == 10 and len(engine.evaluations)-before == 170
    assert not any(row["progress_secant_eligible"] for row in history)
    assert all(row["bracket"][0] < row["parameter"] < row["bracket"][1]
               and row["bracket_heights"][0]*row["bracket_heights"][1] < 0 for row in history)
    assert abs(history[-1]["tau"]) > 1e-10


@pytest.mark.parametrize("fraction", [0., -.1, .5, 1., float("nan")])
def test_invalid_interior_fraction_refused(fraction):
    engine = TraceSaddleContinuation(lambda center, beta: beta-center[0]**2+center[1]**2)
    with pytest.raises(ValueError, match="positive root tolerances"):
        engine.solve_height_zero({}, {}, interior_fraction=fraction)
