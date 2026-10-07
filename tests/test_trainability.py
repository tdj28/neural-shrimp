import numpy as np
import pytest

from neural_shrimp.gdmaps import STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_FIXED, STATUS_PERIODIC
from neural_shrimp.models import TanhRegressor, make_teacher_regression
from neural_shrimp.trainability import (
    array_sha256, chaos_sample, edge_rates, log_axis, make_s1, make_s2,
    periodic_components, zoom_axes, zoom_window,
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


def test_zoom_window_counts_periodic_and_chaotic_cells_with_tie_breaking():
    status = np.full((64, 64), STATUS_FIXED)
    period = np.ones((64, 64), dtype=int)
    assert zoom_window(status, period) is None
    status[40:44, 40:44] = STATUS_CHAOTIC
    period[40:44, 40:44] = 0
    status[5, 5] = STATUS_PERIODIC
    period[5, 5] = 3
    r0, c0, count = zoom_window(status, period)
    assert count == 16 and r0 <= 40 and r0 + 32 >= 44 and c0 <= 40 and c0 + 32 >= 44
    # A window containing both regions wins over either alone; ties go to smallest row/col.
    tie_status = np.full((64, 64), STATUS_DIVERGENT)
    tie_period = np.full((64, 64), -1)
    tie_status[0, 40] = tie_status[40, 0] = STATUS_CHAOTIC
    assert zoom_window(tie_status, tie_period)[:2] == (0, 16)
    # Period-1 periodic labels are not counted.
    lone = np.full((64, 64), STATUS_PERIODIC)
    assert zoom_window(lone, np.ones((64, 64), dtype=int)) is None


def test_zoom_axes_span_window_centers():
    log_h = np.linspace(0, 3, 256)
    log_o = np.linspace(-1, 2, 256)
    zh, zo = zoom_axes(log_h, log_o, 16, 48)
    assert zh[0] == log_h[16] and zh[-1] == pytest.approx(log_h[47])
    assert zo[0] == log_o[48] and zo[-1] == pytest.approx(log_o[79])
    assert len(zh) == len(zo) == 256


def test_periodic_components_selection_order_representative_and_chaos_border():
    status = np.full((12, 12), STATUS_FIXED)
    period = np.ones((12, 12), dtype=int)
    status[0:3, 0:3] = STATUS_PERIODIC   # period-4 block, 9 cells, touches chaos at (3, 1)
    period[0:3, 0:3] = 4
    status[3, 1] = STATUS_CHAOTIC
    period[3, 1] = 0
    status[8:10, 8:10] = STATUS_PERIODIC  # period-3 block, 4 cells, isolated
    period[8:10, 8:10] = 3
    status[6, 6] = STATUS_PERIODIC        # too small
    period[6, 6] = 5
    status[0:3, 3] = STATUS_PERIODIC      # adjacent but different period: separate component
    period[0:3, 3] = 8
    records = periodic_components(status, period, min_size=3)
    assert [(r["period"], r["size"]) for r in records] == [(4, 9), (3, 4), (8, 3)]
    assert records[0]["representative"] == 1 * 12 + 1
    assert records[0]["borders_chaos"] and not records[1]["borders_chaos"]
    assert records[2]["borders_chaos"] is False
    assert records[1]["representative"] == 8 * 12 + 8
    fixed = periodic_components(status, period, min_size=4, max_components=1, fixed=True)
    assert len(fixed) == 1 and fixed[0]["period"] == 1


def test_chaos_sample_is_seeded_and_without_replacement():
    status = np.full((10, 10), STATUS_FIXED)
    status.ravel()[[3, 17, 42, 55, 99]] = STATUS_CHAOTIC
    first = chaos_sample(status, 3, 2810)
    assert first == chaos_sample(status, 3, 2810)
    assert len(set(first)) == 3 and set(first) <= {3, 17, 42, 55, 99}
    assert sorted(chaos_sample(status, 16, 1)) == [3, 17, 42, 55, 99]
    assert chaos_sample(np.full((4, 4), STATUS_FIXED), 16, 1) == []
