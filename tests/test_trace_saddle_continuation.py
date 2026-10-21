import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trace_saddle_continuation import ContinuationError, TraceSaddleContinuation


def translated_saddle(center, parameter):
    x, y = center - np.array([0.4 * parameter, -0.2 * parameter])
    return {"tau": parameter - 0.3 - 2 * x * x + 3 * y * y, "qualified": True}


def test_continues_saddle_and_corrects_height_root_instead_of_fixed_slice():
    events = []
    worker = TraceSaddleContinuation(translated_saddle, correction_tolerance=1e-10, record=events.append)
    result = worker.continue_values([0, 0.2, 0.5], [0, 0])
    assert result["complete"]
    for point in result["points"]:
        gamma = point["parameter"]
        np.testing.assert_allclose(point["center"], [0.4 * gamma, -0.2 * gamma], atol=1e-12)
        assert point["tau"] == pytest.approx(gamma - 0.3, abs=1e-14)
        np.testing.assert_allclose(point["hessian_eigenvalues"], [-4, 6], atol=1e-10)
        assert point["correction_norm"] <= 1e-10
    root = worker.solve_height_zero(*result["points"][1:], height_tolerance=1e-12)
    assert root["complete"]
    assert root["saddle"]["parameter"] == pytest.approx(0.3, abs=1e-12)
    np.testing.assert_allclose(root["saddle"]["center"], [0.12, -0.06], atol=1e-12)
    assert abs(translated_saddle(np.zeros(2), 0.3)["tau"]) > 1e-3
    assert any(e["stage"] == "saddle_predictor" and len(e["based_on"]) == 2 for e in events)
    json.dumps(events, allow_nan=False)


def test_rejects_missing_height_bracket():
    worker = TraceSaddleContinuation(translated_saddle)
    points = worker.continue_values([0, 0.1], [0, 0])["points"]
    with pytest.raises(ContinuationError, match="no strict sign-changing"):
        worker.solve_height_zero(*points)
    assert worker.events[-1]["stage"] == "failed"


@pytest.mark.parametrize("callback,match", [
    (lambda x, p: x[0] ** 2 + x[1] ** 2, "not a nondegenerate saddle"),
    (lambda x, p: -x[0] ** 2 + 1e-12 * x[1] ** 2, "ill-conditioned"),
])
def test_rejects_no_saddle_and_ill_conditioned_hessian(callback, match):
    worker = TraceSaddleContinuation(callback)
    with pytest.raises(ContinuationError, match=match):
        worker.correct(0, [0, 0])
    assert len(worker.evaluations) == 9
    assert worker.events[-1]["stage"] == "failed"


def test_failed_callback_and_budget_retain_attempts():
    def fail(x, p):
        raise ValueError("orbit precision gate failed")
    worker = TraceSaddleContinuation(fail)
    with pytest.raises(ContinuationError, match="orbit precision gate failed"):
        worker.correct(0, [0, 0])
    assert worker.evaluations[0]["status"] == "failed"
    assert worker.events[-1]["stage"] == "evaluation_failed"
    worker = TraceSaddleContinuation(translated_saddle, max_evaluations=2)
    with pytest.raises(ContinuationError, match="maximum callback"):
        worker.correct(0, [0, 0])
    assert len(worker.evaluations) == 2


def test_nonconvergence_and_bounds_never_return_a_default():
    worker = TraceSaddleContinuation(translated_saddle, max_iterations=1, correction_tolerance=1e-12)
    with pytest.raises(ContinuationError, match="did not converge"):
        worker.correct(0.4, [0, 0])
    assert not worker.results
    worker = TraceSaddleContinuation(translated_saddle, coordinate_bounds=[[-0.05, 0.05], [-1, 1]])
    with pytest.raises(ContinuationError, match="coordinate bounds"):
        worker.correct(0, [0, 0])


def test_callback_cannot_hide_failed_qualification():
    worker = TraceSaddleContinuation(lambda x, p: {"tau": 0, "qualified": False})
    with pytest.raises(ContinuationError, match="qualification failed"):
        worker.correct(0, [0, 0])
