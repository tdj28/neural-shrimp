import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fold_panel_data import coordinates, channel_mask, intervals


def test_signed_zero_crossing_is_not_a_jump():
    states = np.array([[.002, 0], [.001, 0], [0, 0], [-.001, 0]])
    v, t = coordinates(states, np.zeros(2), np.array([1., 0]), np.array([1., 0]), -1)
    np.testing.assert_allclose(v, [-.002, -.001, 0, .001])
    assert np.all(np.diff(v) > 0)
    assert (t == 0).all()


def test_projection_alone_cannot_enter_channel():
    v, t = coordinates(np.array([[0., 10.], [.001, .002]]), np.zeros(2),
                       np.array([1., 0]), np.array([1., 0]), 1)
    assert channel_mask(v, t, .004).tolist() == [False, True]


def test_phase_union_preserves_separate_readings_and_endpoints():
    v = np.array([[0., 1., 1., 0.], [1., 0., 1., 1.]])
    m = channel_mask(v, np.zeros_like(v), .004)
    assert intervals(m.any(axis=0)) == [(0, 1), (3, 3)]
    assert intervals(m[0]) == [(0, 0), (3, 3)]
    with pytest.raises(ValueError):
        channel_mask(v * np.nan, np.zeros_like(v), .004)
