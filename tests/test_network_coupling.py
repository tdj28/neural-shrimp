"""Analytic graph/map checks; no model loads or N100 outcomes."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import network_coupling as nc  # noqa: E402


def test_two_copy_prediction_recovers_known_threshold():
    a = np.array([[0., 1.], [1., 0.]])
    r = nc.stability_prediction(a, np.log(2), [0, .25, .5, .75, 1])
    np.testing.assert_allclose(r["laplacian_eigenvalues"], [0, 2])
    np.testing.assert_allclose(r["unrestricted_open_interval"], [.25, .75])
    assert r["stable"].tolist() == [False, False, True, False, False]
    assert np.isneginf(r["transverse_exponents"][2, 0])


def test_irregular_degrees_use_similar_symmetric_matrix():
    a = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    w = nc.normalize_affinity(a)
    assert not np.allclose(w, w.T)
    mu, components = nc.laplacian_spectrum(a)
    np.testing.assert_allclose(mu, np.sort(np.linalg.eigvals(np.eye(3) - w)), atol=1e-14)
    assert components == 1
    x = np.arange(9).reshape(3, 3)
    mixed = nc.mix_outputs(x, w, .4)
    np.testing.assert_allclose(a.sum(axis=1) @ mixed, a.sum(axis=1) @ x)


def test_complete_graph_synchronizes_exactly_at_averaging_strength():
    a = np.ones((6, 6)) - np.eye(6)
    w = nc.normalize_affinity(a)
    x = np.arange(18).reshape(6, 3)
    expected = np.broadcast_to(x.mean(axis=0), x.shape)
    np.testing.assert_allclose(nc.mix_outputs(x, w, 5 / 6), expected)
    np.testing.assert_allclose(nc.laplacian_spectrum(a)[0][1:], 6 / 5)
    # Invariance of the synchronized manifold does not depend on regular degree.
    a[0, 1] = a[1, 0] = 3
    np.testing.assert_allclose(nc.mix_outputs(expected, nc.normalize_affinity(a), .8), expected)


def test_disconnected_graph_keeps_unstable_transverse_zero_mode():
    a = np.kron(np.eye(2), np.array([[0., 1.], [1., 0.]]))
    r = nc.stability_prediction(a, .4, [.5, .8])
    assert r["components"] == 2 and r["eigenratio"] is None
    assert r["unrestricted_open_interval"] is None
    assert not r["has_stable_strength_in_0_1"] and not r["stable"].any()
    np.testing.assert_allclose(r["transverse_exponents"][:, 0], .4)


def test_nearest_neighbors_are_union_and_ties_are_deterministic():
    p = [[0, 0], [1, 0], [2, 0]]
    a, w = nc.distance_graph(p, 1, neighbors=1)
    np.testing.assert_allclose(a, [[0, np.exp(-1), 0], [np.exp(-1), 0, np.exp(-1)], [0, np.exp(-1), 0]])
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert np.array_equal(a, a.T)


def test_direct_distances_resolve_tiny_differences():
    x = np.array([[1., 1.], [1. + 1e-10, 1.]])
    d = nc.pairwise_relative_distances(x)
    assert d[0, 1] == pytest.approx((x[1, 0] - 1) / np.sqrt(2), rel=1e-12)
    assert d[0, 1] > 0 and np.array_equal(d, d.T)


def test_close_pair_chain_is_not_called_synchronized_cluster():
    d = np.array([[0., .75e-8, 1.5e-8], [.75e-8, 0., .75e-8], [1.5e-8, .75e-8, 0.]])
    r = nc.persistent_pair_summary(d)
    assert not r["global_sync"] and r["sustained_pair_count"] == 2
    assert len(r["components"]) == 1
    assert not r["components"][0]["all_pairs_sustained"]
    # Strict threshold: equality is not below it.
    assert not nc.persistent_pair_summary(np.array([[0., 1e-8], [1e-8, 0.]]))["global_sync"]


@pytest.mark.parametrize("p,length,k", [([[0, 0]], 1, None), ([[0, 0], [1, 0]], 0, None),
                                      ([[0, 0], [1, 0]], 1, 2), ([[0, 0], [1, 0]], 1, 1.5),
                                      ([[0, 0], [1, 0]], 1e-6, None)])
def test_bad_graphs_fail_loudly(p, length, k):
    with pytest.raises(ValueError):
        nc.distance_graph(p, length, k)


@pytest.mark.parametrize("a", [np.zeros((2, 2)), [[0, 1], [2, 0]], [[0, -1], [-1, 0]],
                               [[1, 1], [1, 0]], [[0, float("nan")], [float("nan"), 0]]])
def test_invalid_affinity_refused(a):
    with pytest.raises(ValueError):
        nc.normalize_affinity(a)


def test_unresolved_gap_fails_instead_of_reporting_a_precise_threshold():
    a = np.array([[0., 1., 1e-30, 0], [1, 0, 0, 0], [1e-30, 0, 0, 1], [0, 0, 1, 0]])
    with pytest.raises(ValueError, match="unresolved"):
        nc.laplacian_spectrum(a)


def test_numerical_jacobian_has_predicted_transverse_factors():
    # Analytic F(x)=4*x*(1-x), checked against finite differences of the full
    # coupled map, rather than duplicating the helper's formula in a test.
    a, w = nc.distance_graph([[0, 0], [.2, .1], [.8, .9]], .5)
    strength, x = .6, np.full((3, 1), .2)
    h, jac = 1e-6, np.empty((3, 3))
    for k in range(3):
        e = np.eye(3)[:, k:k + 1] * h
        plus = nc.mix_outputs(4 * (x + e) * (1 - x - e), w, strength)
        minus = nc.mix_outputs(4 * (x - e) * (1 - x + e), w, strength)
        jac[:, k] = ((plus - minus) / (2 * h))[:, 0]
    expected = 2.4 * (1 - strength * nc.laplacian_spectrum(a)[0])
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(jac)), np.sort(expected), atol=2e-10)
