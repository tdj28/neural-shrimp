"""Ordering, shared coordinates and closure honesty for offline orbit figures."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import plot_baby_shrimp_memory_orbits as plot


def cycle():
    angle = np.arange(18) * 2 * np.pi / 18
    points = np.zeros((18, 896), dtype=np.float64)
    points[:, 0], points[:, 1] = np.cos(angle), np.sin(angle)
    return points


def point_record(points):
    returns = np.linalg.norm(points[1:] - points[0], axis=1) / np.linalg.norm(points[0])
    return {"primitive": {"closed": True, "no_lower_period": True, "all_returns": [*returns, 0.]},
            "tokens": [1, 2, 3, 4, 5, 6] * 3}


def samples():
    points = cycle()
    return [{"orbit": points + np.eye(1, 896, 2) * i * .01,
             "tokens": [1, 2, 3, 4, 5, 6] * 3,
             "summary": {"parameter": beta}} for i, beta in enumerate(plot.BETAS)]


def test_six_step_pairs_include_all_three_temporal_returns_and_wrap():
    for phase in range(6):
        source, target, pairs = plot.six_step_pairs(np.arange(18.), phase)
        assert source.tolist() == [phase, phase + 6, phase + 12]
        assert target.tolist() == [phase + 6, phase + 12, phase]
        np.testing.assert_array_equal(pairs, np.column_stack([source, target]))


@pytest.mark.parametrize("values,phase", [(np.arange(17.), 0), (np.r_[np.nan, np.arange(17.)], 0),
                                          (np.arange(18.), -1), (np.arange(18.), 6)])
def test_invalid_period_or_phase_cannot_silently_change_return_order(values, phase):
    with pytest.raises(ValueError):
        plot.six_step_pairs(values, phase)


def test_joint_coordinate_keeps_between_case_displacement():
    a = np.array([[0., 0.], [1., 0.], [2., 0.]])
    b = a + [10., 0.]
    mean, basis, shares, projected = plot.joint_pca([a, b])
    assert projected[1][:, 0].mean() - projected[0][:, 0].mean() == pytest.approx(10.)
    np.testing.assert_allclose(mean, [6., 0.])
    np.testing.assert_allclose(basis @ basis.T, np.eye(2))
    assert shares[0] == pytest.approx(1.)


def test_cyclic_alignment_changes_indices_and_tokens_together():
    cases = samples()
    cases[1]["orbit"] = np.roll(cases[1]["orbit"], 5, axis=0)
    cases[1]["tokens"] = np.roll(cases[1]["tokens"], 5).tolist()
    aligned, records = plot.align_samples(cases)
    assert records[1]["best_right_roll"] == 13
    np.testing.assert_allclose(aligned[1][:, :2], cases[0]["orbit"][:, :2])
    arrays, derived = plot.derive(cases)
    assert arrays["portrait_coordinates"].shape == (3, 18, 2)
    assert len(derived["phases"]) == 6
    for phase in range(6):
        assert arrays[f"phase_{phase}_pairs"].shape == (3, 3, 2)
        for pairs in arrays[f"phase_{phase}_pairs"]:
            np.testing.assert_allclose(pairs[:, 1], np.roll(pairs[:, 0], -1))


def test_alignment_rejects_mismatched_phase_tokens():
    cases = samples()
    cases[2]["tokens"] = [7, 8, 9, 10, 11, 12] * 3
    with pytest.raises(ValueError, match="token phases differ"):
        plot.align_samples(cases)


def test_recorded_closure_is_never_promoted_to_independent_endpoint_check():
    points = cycle()
    recorded = plot.closure_record(points, point_record(points))
    assert recorded["endpoint_relative_closure"] is None
    assert not recorded["endpoint_available"]
    assert "not independently recomputed" in recorded["closure_scope"]
    checked = plot.closure_record(points, point_record(points), points[0])
    assert checked["endpoint_available"]
    assert checked["endpoint_relative_closure"] == 0


def test_rejects_stale_or_nonclosing_saved_endpoint():
    points = cycle()
    with pytest.raises(ValueError, match="endpoint closure differs"):
        plot.closure_record(points, point_record(points), points[0] + .01)


def test_rejects_reordered_saved_cycle_even_when_it_contains_same_points():
    points = cycle()
    stale = point_record(points)
    reordered = points.copy()
    reordered[[1, 6]] = reordered[[6, 1]]
    with pytest.raises(ValueError, match="primitive-order"):
        plot.closure_record(reordered, stale)


def test_rejects_readout_period_three_when_figure_claims_six():
    points = cycle()
    row = point_record(points)
    row["tokens"] = [1, 2, 3] * 6
    with pytest.raises(ValueError, match="primitive period six"):
        plot.closure_record(points, row)
