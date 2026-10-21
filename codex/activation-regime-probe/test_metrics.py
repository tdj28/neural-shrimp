import numpy as np
import pytest

from metrics import dimensions, layout, predict, recurrence, reference, roughness, threshold


def test_primitive_period_and_drift_refusal():
    cycle = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    x = np.tile(cycle, (20, 1))
    assert recurrence(x)["period"] == 4
    assert recurrence(x + np.arange(len(x))[:, None] * 1e-3)["period"] == 0
    assert recurrence(np.ones((20, 3)))["period"] == 1


def test_dimension_rank_and_invariance():
    rng = np.random.default_rng(42)
    x = np.tile(rng.normal(size=(8, 20)), (8, 1))
    d = dimensions(x)
    assert 1 < d["raw_pr"] <= 7 + 1e-12
    assert dimensions(x[:, ::-1])["raw_pr"] == pytest.approx(d["raw_pr"])
    assert dimensions(x * np.arange(1, 21))["standardized_pr"] == pytest.approx(d["standardized_pr"])
    assert dimensions(np.ones((64, 20)))["raw_pr"] == 0


def test_reference_gates_and_unresolved():
    rng = np.random.default_rng(8)
    periodic = np.tile(rng.normal(size=(8, 4)), (65, 1))
    assert reference(periodic, np.full((512, 2), -.5))["label"] == "periodic"
    irregular = rng.normal(size=(520, 4))
    assert reference(irregular, np.full((512, 2), .1))["label"] == "finite-time chaotic"
    mixed = np.full((512, 2), .1); mixed[:128] = -.01
    assert reference(irregular, mixed)["label"] == "unresolved"
    assert reference(irregular, np.tile([.1, .5], (512, 1)))["label"] == "unresolved"
    with pytest.raises(ValueError):
        reference(irregular, np.full((512, 2), np.nan))


def test_layout_and_roughness():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 24))
    order = layout(x, 4, 6)
    assert sorted(order.ravel()) == list(range(24))
    s = roughness(x, order)["median"]
    assert roughness(x * 7 + 3, order)["median"] == pytest.approx(s)
    with pytest.raises(ValueError):
        roughness(x, np.zeros((4, 6), dtype=int))


def test_missing_calibration_fails():
    with pytest.raises(ValueError):
        threshold(1, 1)
    assert predict(.5, threshold(0, 1)) == "periodic"
    assert predict(.8, threshold(0, 1)) == "finite-time chaotic"
    assert predict(.2, threshold(1, 0)) == "finite-time chaotic"


def test_short_window_recurrence_comparators():
    cycle = np.random.default_rng(7).normal(size=(8, 12))
    x = np.tile(cycle, (2, 1))
    assert recurrence(x)["period"] == 0
    assert recurrence(x, min_cycles=2)["period"] == 8
    drift = x + np.arange(16)[:, None] * 1e-3
    assert recurrence(drift, min_cycles=2)["period"] == 0
