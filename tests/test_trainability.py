import numpy as np
import pytest

from neural_shrimp.gdmaps import (
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_FIXED, STATUS_PERIODIC, STATUS_UNRESOLVED,
)
from neural_shrimp.models import TanhRegressor, make_teacher_regression
from neural_shrimp.trainability import (
    SYSTEMS, array_sha256, chaos_sample, edge_rates, log_axis, make_s1, make_s1r, make_s2,
    ring_statistics, select_components, zoom_axes, zoom_window,
)


def test_make_s1_follows_published_mean_field_defaults_and_is_deterministic():
    net, x, y, theta0 = make_s1(123)
    assert (net.d, net.width, net.bias, net.loss_scale, net.ridge) == (8, 8, False, 1.0, 0.0)
    assert net.in_scale == pytest.approx(np.sqrt(2.0 / 8))
    assert net.out_scale == pytest.approx(1.0 / 8)
    assert net.n_parameters == 72 and x.shape == (72, 8) and y.shape == (72,)
    assert theta0.shape == (72,)
    _, x2, y2, theta2 = make_s1(123)
    assert array_sha256(x) == array_sha256(x2) and array_sha256(theta0) == array_sha256(theta2)
    _, x3, _, _ = make_s1(124)
    assert array_sha256(x) != array_sha256(x3)


def test_make_s2_reuses_the_earlier_regressor_conventions():
    net, x, y, theta0 = make_s2(77)
    data = make_teacher_regression(seed=77, d=2, teacher_width=3, n_train=24,
                                   n_validation=48, n_test=96)
    np.testing.assert_array_equal(x, data.train_x)
    np.testing.assert_array_equal(y, data.train_y)
    np.testing.assert_array_equal(theta0, TanhRegressor(2, 3).initialize(seed=100077, scale=0.2))
    assert (net.bias, net.loss_scale, net.ridge, net.n_parameters) == (True, 0.5, 1e-3, 13)


def test_edge_rates_are_two_over_layer_curvature():
    net, x, y, theta0 = make_s2(5)
    hidden, output = net.gauss_newton_layer_max(theta0, x, y)
    rates = edge_rates(net, theta0, x, y)
    assert rates == pytest.approx((2.0 / hidden, 2.0 / output))


def test_log_axis_is_symmetric_with_requested_resolution():
    axis = log_axis(1.25, 1.5, 256)
    assert axis[0] == pytest.approx(-0.25) and axis[-1] == pytest.approx(2.75)
    assert len(axis) == 256 and np.all(np.diff(axis) > 0)


def test_s1r_shares_s1_inputs_and_adds_ridge():
    net, x, y, theta0 = make_s1r(31)
    base, x0, y0, t0 = make_s1(31)
    assert net.ridge == 1e-3 and base.ridge == 0.0
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(theta0, t0)
    assert set(SYSTEMS) == {"S1", "S1R", "S2"}


def test_zoom_window_prefers_period_three_or_more_bordering_chaos():
    status = np.full((64, 64), STATUS_FIXED)
    period = np.ones((64, 64), dtype=int)
    assert zoom_window(status, period) is None
    # A plateau of period-2 cells next to chaos must not attract the zoom.
    status[0:32, 0:32] = STATUS_PERIODIC
    period[0:32, 0:32] = 2
    status[0:32, 32:34] = STATUS_CHAOTIC
    period[0:32, 32:34] = 0
    no_primary = zoom_window(status, period)
    assert no_primary is not None and no_primary[2] == 0 and no_primary[3] > 0
    # A small period-5 island inside chaos wins.
    status[44:52, 44:52] = STATUS_CHAOTIC
    period[44:52, 44:52] = 0
    status[47:49, 47:49] = STATUS_PERIODIC
    period[47:49, 47:49] = 5
    r0, c0, primary, secondary = zoom_window(status, period)
    assert primary == 4 and r0 <= 47 and r0 + 32 >= 49 and c0 <= 47 and c0 + 32 >= 49


def test_ring_statistics_flags_proper_divisor_neighbors():
    status = np.full((7, 7), STATUS_CHAOTIC)
    period = np.zeros((7, 7), dtype=int)
    status[2:5, 2:5] = STATUS_PERIODIC
    period[2:5, 2:5] = 6
    status[1, 3] = STATUS_PERIODIC
    period[1, 3] = 3     # proper divisor of 6
    status[5, 3] = STATUS_PERIODIC
    period[5, 3] = 12    # multiple: allowed
    status[3, 1] = STATUS_FIXED
    members = np.flatnonzero(((status == STATUS_PERIODIC) & (period == 6)).ravel())
    ring, divisor, chaotic = ring_statistics(status, period, members, 6)
    assert len(ring) == 16
    assert sorted(divisor) == sorted([1 * 7 + 3, 3 * 7 + 1])
    assert len(chaotic) == 13


def test_zoom_axes_span_window_centers():
    log_h = np.linspace(0, 3, 256)
    log_o = np.linspace(-1, 2, 256)
    zh, zo = zoom_axes(log_h, log_o, 16, 48)
    assert zh[0] == log_h[16] and zh[-1] == pytest.approx(log_h[47])
    assert zo[0] == log_o[48] and zo[-1] == pytest.approx(log_o[79])
    assert len(zh) == len(zo) == 256


def test_select_components_separates_embedded_cascade_and_fixed():
    status = np.full((14, 14), STATUS_CHAOTIC)
    period = np.zeros((14, 14), dtype=int)
    # Cascade-adjacent: period 4 touching a period-2 domain.
    status[0:3, 0:4] = STATUS_PERIODIC
    period[0:3, 0:4] = 2
    status[0:3, 4:6] = STATUS_PERIODIC
    period[0:3, 4:6] = 4
    # Embedded: period 5 surrounded by chaos.
    status[8:10, 8:10] = STATUS_PERIODIC
    period[8:10, 8:10] = 5
    # Fixed component with unresolved surroundings.
    status[11:14, 0:3] = STATUS_FIXED
    period[11:14, 0:3] = 1
    status[10, 0:4] = STATUS_UNRESOLVED
    lists = select_components(status, period, min_size=4)
    assert [(r["period"], r["size"]) for r in lists["embedded"]] == [(5, 4)]
    assert lists["embedded"][0]["ring_chaotic_fraction"] == 1.0
    others = sorted((r["period"], r["size"]) for r in lists["other"])
    assert others == [(2, 12), (4, 6)]
    cascade = [r for r in lists["other"] if r["period"] == 4][0]
    assert cascade["ring_divisor_cells"]
    assert [(r["period"], r["size"]) for r in lists["fixed"]] == [(1, 9)]
    assert lists["embedded"][0]["representative"] in {8 * 14 + 8, 8 * 14 + 9, 9 * 14 + 8, 9 * 14 + 9}


def test_chaos_sample_is_seeded_and_without_replacement():
    status = np.full((10, 10), STATUS_FIXED)
    status.ravel()[[3, 17, 42, 55, 99]] = STATUS_CHAOTIC
    first = chaos_sample(status, 3, 2810)
    assert first == chaos_sample(status, 3, 2810)
    assert len(set(first)) == 3 and set(first) <= {3, 17, 42, 55, 99}
    assert sorted(chaos_sample(status, 16, 1)) == [3, 17, 42, 55, 99]
    assert chaos_sample(np.full((4, 4), STATUS_FIXED), 16, 1) == []
