import numpy as np
import pytest

from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.warmup import branch_access, orbit_distance, reduced_reference, rollout, summarize


def reference(orbit, h=-1.0, o=-1.0):
    return {"h": h, "o": o, "orbit": np.asarray(orbit), "alpha": 0.01,
            "loss_mean": 1.0, "loss_max": 1.0, "branch_reference_applies": True}


def test_full_state_distance_is_phase_invariant_but_keeps_hidden_coordinates():
    orbit = np.array([[0., 1., 2.], [2., 3., 4.]])
    assert orbit_distance(orbit[::-1], orbit) == pytest.approx([0, 0])
    assert orbit_distance(np.array([[0.3, 1, 2]]), orbit)[0] == pytest.approx(0.3)


def test_global_reduction_scales_both_rates_and_invalidates_local_reference():
    r = reference([[0., 0.], [1., 1.]], h=1.0, o=-0.5)
    low = reduced_reference(r, r, 0.5)
    assert 10 ** low['h'] == pytest.approx(0.5 * 10 ** r['h'])
    assert 10 ** low['o'] == pytest.approx(0.5 * 10 ** r['o'])
    assert not low['branch_reference_applies']
    with pytest.raises(ValueError):
        reduced_reference(r, r, 1.5)


def test_variable_rate_pairs_match_manual_analytic_gradient_updates():
    net = TwoLayerNet(1, 1, activation="identity", bias=False)
    x, y = np.array([[1.], [2.]]), np.array([1., 2.])
    theta = np.array([[0.2, 0.3]])
    refs = [reference([[0., 0.], [1., 1.]], h=-1, o=-2),
            reference([[0., 0.], [1., 1.]], h=-2, o=-1)]
    got = rollout(net, x, y, theta, [(2, [refs[0]], 1), (1, [refs[1]], 2)])
    manual, expected_losses = theta.copy(), []
    for ref in [refs[0], refs[0], refs[1]]:
        ls = []
        for _ in range(2):
            a, b = manual[0]
            ls.append(1.25 * (a * b - 1) ** 2)
            grad = 2.5 * (a * b - 1) * np.array([b, a])
            manual -= np.array([[10 ** ref['h'], 10 ** ref['o']]]) * grad
        expected_losses.append(ls)
    assert got['state'][-1] == pytest.approx(manual)
    assert got['loss'][:, 0] == pytest.approx(np.array(expected_losses))


def test_loss_spike_and_local_departure_are_separate_and_invalid_reference_is_excluded():
    stable = reference([[0., 0.], [0.1, 0.1]])
    trace = {"phase": np.zeros(3), "loss": np.ones((3, 1, 2)),
             "state": np.ones((3, 1, 2)), "distance": np.ones((3, 1)),
             "reference_applies": np.ones((3, 1), dtype=bool),
             "reference_loss_max": np.ones((3, 1))}
    result = summarize(trace, 0, stable)
    assert result['local_departure_above_0_5']
    assert not result['loss_spike_above_25_percent']
    trace['loss'] *= 2
    trace['distance'] *= 0
    assert summarize(trace, 0, stable)['loss_spike_above_25_percent']
    assert not summarize(trace, 0, stable)['local_departure_above_0_5']
    trace['reference_applies'][:] = False
    assert summarize(trace, 0, stable)['local_departure_above_0_5'] is None


def test_accessibility_exposes_preexisting_distance_instead_of_calling_it_a_crossing():
    trace = {"phase": np.array([0, 0, 1, 1]), "alpha": np.array([[-.1], [-.1], [0.], [.1]]),
             "distance": np.array([[1.], [.6], [.7], [.8]])}
    got = branch_access(trace, 0)
    assert got['distance_after_prehold'] == .6
    assert not got['near_reference_at_crossing_within_0_01']
    assert got['first_departure_after_crossing_pairs'] == 0
    trace['distance'] = np.array([[.1], [.001], [.002], [.7]])
    got = branch_access(trace, 0)
    assert got['near_reference_at_crossing_within_0_01']
    assert got['first_departure_after_crossing_pairs'] == 1
