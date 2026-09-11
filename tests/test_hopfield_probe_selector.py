"""Injected and analytic-waveform tests; no neural integration is performed."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("hopfield_probe_selector", TOOLS / "hopfield_probe_selector.py")
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def pool():
    return selector.experiment_candidates([0., 1.], [0., 1.], [1., 0.])


def synthetic_evaluator(direction, *, radius, duration, index):
    return {"metrics": {"aligned_rms": {"aggregate": (index - 7) ** 2 / 121},
                         "synchronous_rms": {"aggregate": 0.}},
            "cost": {"native_steps": 15000, "synthetic": True},
            "cache": {"times": np.array([0., duration]),
                       "direction": direction.copy(), "radius": radius}}


def test_fixed_21_candidate_order_and_duplicates_are_preserved():
    candidates = pool()
    assert len(candidates) == 21
    assert [c["index"] for c in candidates] == list(range(21))
    assert [c["label"] for c in candidates[16:]] == [
        "local_plus", "local_minus", "tangent_plus", "tangent_minus", "atlas_margin"]
    np.testing.assert_allclose(np.linalg.norm([c["direction"] for c in candidates], axis=1), 1, atol=1e-15)
    assert candidates[4]["direction"] == candidates[16]["direction"] == candidates[18]["direction"]
    coordinates = selector.direction_candidates("coordinate_signed")
    for coordinate, index in zip(coordinates, selector.COORDINATE_INDICES):
        assert coordinate["direction"] == candidates[index]["direction"]


def test_every_candidate_evaluated_and_metrics_cost_cache_retained():
    called, emitted = [], []
    def evaluate(direction, *, radius, duration, index):
        called.append(index)
        assert radius == .016 and duration == 75
        return synthetic_evaluator(direction, radius=radius, duration=duration, index=index)
    result = selector.select_with_probe(evaluate, pool(), .016,
                                        on_candidate=lambda record, cache: emitted.append((record, cache)))
    assert result["valid"] and result["selected_index"] == 7
    assert called == list(range(21))
    assert len(emitted) == result["candidate_count"] == result["new_probe_calls"] == 21
    assert result["required_probe_calls"] == 21 and result["failed_candidate_count"] == 0
    for row in result["candidate_records"]:
        assert row["cost"] == {"native_steps": 15000, "synthetic": True}
        assert "synchronous_rms" in row["metrics"]
        assert row["evaluation_wall_seconds"] >= 0
        np.testing.assert_allclose(np.linalg.norm(row["weight_delta"]), .016, atol=1e-15)
        np.testing.assert_array_equal(result["cached_probes"][row["index"]]["times"], [0, 75])
    assert result["evaluation_wall_seconds"] == sum(r["evaluation_wall_seconds"] for r in result["candidate_records"])


def test_exact_ties_choose_lowest_index_but_near_ties_are_not_collapsed():
    def tied(direction, *, radius, duration, index):
        return {"metrics": {"aligned_rms": {"aggregate": .1}}}
    result = selector.select_with_probe(tied, pool(), .001)
    assert result["selected_index"] == 0
    assert selector.select_from_records(result["candidate_records"], [12, 8, 4, 0])["selected_index"] == 0
    result["candidate_records"][4]["metrics"]["aligned_rms"]["aggregate"] = .1 - 1e-14
    result["candidate_records"][4]["score"] = .1 - 1e-14
    assert selector.select_from_records(result["candidate_records"], [12, 8, 4, 0])["selected_index"] == 4


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -0.001, 2.001, None, True])
def test_failed_or_nonfinite_required_candidate_invalidates_without_substitution(bad_score):
    called = []
    def evaluate(direction, *, radius, duration, index):
        called.append(index)
        result = synthetic_evaluator(direction, radius=radius, duration=duration, index=index)
        if index == 8: result["metrics"]["aligned_rms"]["aggregate"] = bad_score
        return result
    result = selector.select_with_probe(evaluate, pool(), .004)
    assert called == list(range(21))
    assert not result["valid"] and result["selected_index"] is None
    assert result["invalid_indices"] == [8] and result["failed_candidate_count"] == 1
    assert result["candidate_records"][8]["status"] == "failed"
    assert result["cached_probes"][8] is not None
    local = selector.select_from_records(result["candidate_records"], selector.POLICY_INDICES["local_signed"])
    coordinates = selector.select_from_records(result["candidate_records"], selector.POLICY_INDICES["coordinate_signed"])
    assert local["valid"] and local["required_probe_calls"] == 2 and local["new_probe_calls"] == 0
    assert not coordinates["valid"] and coordinates["required_probe_calls"] == 4


def test_exception_payload_and_later_candidates_are_retained():
    class InjectedFailure(RuntimeError):
        def __init__(self):
            super().__init__("synthetic partial trajectory failure")
            self.payload = {"partial_states": np.array([[1., 2., 3.]])}
    def evaluate(direction, *, radius, duration, index):
        if index == 2: raise InjectedFailure()
        return synthetic_evaluator(direction, radius=radius, duration=duration, index=index)
    result = selector.select_with_probe(evaluate, pool(), .001)
    assert not result["valid"] and len(result["candidate_records"]) == 21
    assert "synthetic partial trajectory failure" in result["candidate_records"][2]["error"]
    np.testing.assert_array_equal(result["cached_probes"][2]["partial_states"], [[1, 2, 3]])
    assert result["candidate_records"][-1]["status"] == "complete"


def test_prefixed_subsets_select_fair_signs_from_same_frozen_pool():
    result = selector.select_with_probe(synthetic_evaluator, pool(), .016)
    records = result["candidate_records"]
    assert selector.select_from_records(records, [16, 17])["selected_index"] == 16
    assert selector.select_from_records(records, [18, 19])["selected_index"] == 18
    assert selector.select_from_records(records, [0, 4, 8, 12])["selected_index"] == 8
    assert selector.select_from_records(records, range(21))["selected_index"] == 7
    missing = selector.select_from_records(records[:-1], range(21))
    assert not missing["valid"] and missing["missing_indices"] == [20]
    with pytest.raises(ValueError, match="unique"):
        selector.select_from_records(records, [0, 0])


def test_probe_score_matches_existing_metric_with_original_phase_epoch():
    phases = np.arange(1024) / 1024
    reference = np.column_stack([.6 * np.sin(2 * np.pi * phases),
                                 .4 * np.cos(2 * np.pi * phases),
                                 .3 * np.sin(4 * np.pi * phases)])
    times = np.linspace(0, 75, 751)
    epoch, period = 2.5, 11.
    from hopfield_drift_metrics import periodic_interpolate, waveform_errors
    candidate = periodic_interpolate(reference, (times + epoch) / period + .125)
    result = selector.score_probe(reference, times, candidate, period, phase_epoch=epoch)
    expected = waveform_errors(reference, times + epoch, candidate, period, phase_grid=128)
    assert result == expected
    assert result["aligned_rms"]["aggregate"] < 1e-12
    assert result["synchronous_rms"]["aggregate"] > .1
    with pytest.raises(ValueError, match="full elapsed"):
        selector.score_probe(reference, times[1:], candidate[1:], period)
    with pytest.raises(ValueError, match="75"):
        selector.score_probe(reference, times, candidate, period, duration=300)
    with pytest.raises(ValueError, match="full elapsed"):
        selector.score_probe(reference, times * 4, candidate, period)


def test_invalid_action_inputs_fail_before_any_evaluator_call():
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluator must not be called")
    for candidates in [[], [[2., 0.]], [[np.nan, 0.]], [{"index": 1, "direction": [1., 0.]}]]:
        with pytest.raises(ValueError): selector.select_with_probe(forbidden, candidates, .001)
    with pytest.raises(ValueError): selector.select_with_probe(forbidden, pool(), 0)
    with pytest.raises(ValueError): selector.select_with_probe(forbidden, pool(), .001, duration=300)
    with pytest.raises(ValueError): selector.direction_candidates("adapt_from_outcome")


def test_evaluator_cannot_mutate_declared_directions():
    candidates = pool()
    original = np.array([c["direction"] for c in candidates])
    def mutate(direction, *, radius, duration, index):
        direction[:] = 0
        return {"metrics": {"aligned_rms": {"aggregate": .1}}}
    result = selector.select_with_probe(mutate, candidates, .001)
    np.testing.assert_array_equal([c["direction"] for c in candidates], original)
    np.testing.assert_array_equal([c["direction"] for c in result["candidate_records"]], original)


def test_reused_callback_buffers_do_not_overwrite_earlier_probe_evidence():
    reused = {"metrics": {"aligned_rms": {"aggregate": 0.}},
              "cost": {"index": 0}, "cache": {"states": np.zeros((2, 3))}}
    def evaluate(direction, *, radius, duration, index):
        reused["metrics"]["aligned_rms"]["aggregate"] = index / 20
        reused["cost"]["index"] = index
        reused["cache"]["states"][:] = index
        return reused
    result = selector.select_with_probe(evaluate, pool(), .001)
    assert result["valid"] and result["selected_index"] == 0
    for row in result["candidate_records"]:
        assert row["metrics"]["aligned_rms"]["aggregate"] == row["index"] / 20
        assert row["cost"]["index"] == row["index"]
        np.testing.assert_array_equal(result["cached_probes"][row["index"]]["states"],
                                       np.full((2, 3), row["index"]))
