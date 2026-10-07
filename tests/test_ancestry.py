import numpy as np

from neural_shrimp.ancestry import (classify_ancestry, closest_to_one, continue_segment, cycle_amplitude,
                                    existence_ray, same_cycle)


def _quadratic_family(params):
    # x -> x + a - x^2 (+ contracting y): fixed points exist for a > 0, fold at a = 0
    a = params[0]

    def map_fn(state):
        x, y = state
        return np.array([x + a - x * x, 0.3 * y])

    def jacobian(state):
        x, _ = state
        return np.array([[1.0 - 2.0 * x, 0.0], [0.0, 0.3]])

    return map_fn, jacobian


def test_existence_ray_types_a_fold_and_locates_it():
    a0 = 0.04
    x0 = np.array([np.sqrt(a0), 0.0])  # stable branch x = +sqrt(a)
    result = existence_ray(_quadratic_family, x0, 1, [a0, 0.0], [-1.0, 0.0], 0.1, 0.005,
                           bisection_tolerance=1e-12)
    assert result["type"] == "fold"
    assert abs(result["s_boundary"] - a0) < 1e-6


def test_existence_ray_reports_none_within_range():
    x0 = np.array([np.sqrt(0.04), 0.0])
    result = existence_ray(_quadratic_family, x0, 1, [0.04, 0.0], [1.0, 0.0], 0.05, 0.005)
    assert result["type"] == "none_within_range"


def test_continue_segment_reaches_end_and_refuses_impossible_target():
    x0 = np.array([np.sqrt(0.04), 0.0])
    ok = continue_segment(_quadratic_family, x0, 1, [0.04, 0.0], [0.09, 0.0])
    assert ok["ok"] and abs(ok["orbit"][0][0] - 0.3) < 1e-8
    bad = continue_segment(_quadratic_family, x0, 1, [0.04, 0.0], [-0.02, 0.0])
    assert not bad["ok"]


def test_cycle_helpers():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert same_cycle(a, np.roll(a, 1, axis=0), 1e-12)
    assert not same_cycle(a, a + 0.1, 1e-3)
    assert abs(cycle_amplitude(a, np.array([[0.0, 0.0]])) - 2.0) < 1e-12
    mu, dist = closest_to_one({"multipliers": [{"real": 0.5, "imag": 0.0}, {"real": 0.98, "imag": 0.1}]})
    assert abs(mu - complex(0.98, 0.1)) < 1e-12 and abs(dist - abs(complex(-0.02, 0.1))) < 1e-12


def test_classify_ancestry_outcomes():
    def level(delta, center, width, amp):
        return {"delta": delta, "left": {"type": "fold", "u": center - width / 2},
                "right": {"type": "fold", "u": center + width / 2}, "amplitude": amp,
                "primitive": True, "continuity": True}

    deltas = [1e-2 / 4 ** k for k in range(6)]
    passing = [level(d, 0.3 * d, d ** 1.5, np.sqrt(d)) for d in deltas]
    assert classify_ancestry(passing, 1e-4)[0] == "pass"
    elsewhere = [level(d, 0.02 + 0.3 * d, d ** 1.5, np.sqrt(d)) for d in deltas]
    assert classify_ancestry(elsewhere, 1e-4)[0] == "alternative_ancestry"
    missing = passing[:3] + [{**passing[3], "right": {"type": "lost", "u": 0.0}}] + passing[4:]
    assert classify_ancestry(missing, 1e-4)[0] == "exhausted_budget"
    finite_amplitude = [{**lv, "right": {"type": "none_within_range", "u": None}, "amplitude": 0.5}
                        for lv in passing]
    assert classify_ancestry(finite_amplitude, 1e-4)[0] == "alternative_ancestry"
    no_parent = passing[:2] + [{**passing[2], "amplitude": None}] + passing[3:]
    assert classify_ancestry(no_parent, 1e-4)[0] == "exhausted_budget"
    assert classify_ancestry(passing, 1e-4, controls_passed=False)[0] == "invalid"
