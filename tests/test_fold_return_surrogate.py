import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
from fold_return_surrogate import DepartureDetector, make_model, predict, rollout, refinement_points


def test_exit_requires_long_visit_and_same_phase_and_bounded_gap():
    d = DepartureDetector(1.)
    found = []
    for t in range(40):
        v = .1 if t % 2 == 0 and t < 20 else (3.1 if t % 2 == 0 else 8.)
        event = d.update(t, v, 0.)
        if event: found.append(event)
    assert len(found) == 1 and found[0]['exit'] == 20 and found[0]['entry'] == 0
    d = DepartureDetector(1.)
    for t in range(70):
        v = .1 if t % 2 == 0 and t < 20 else (1.1 if t < 60 else 3.1)
        assert d.update(t, v, 0.) is None


def test_initial_exit_is_not_a_return_and_short_visit_is_ignored():
    d = DepartureDetector(1.)
    for t in range(32):
        v = .1 if t % 2 == 1 and 3 <= t <= 11 else 4.
        assert d.update(t, v, 0.) is None


def table():
    return make_model([{'q': 3., 'kind': 'return', 'updates': 100, 'next_q': 3.2},
                       {'q': 3.5, 'kind': 'return', 'updates': 100, 'next_q': 3.4},
                       {'q': 4., 'kind': 'reference_p3', 'updates': 256}])


def test_interpolation_does_not_bridge_capture_and_abstains_outside():
    m = table()
    p = predict(m, 3.25)
    assert p['next_q'] == pytest.approx(3.3) and p['branch_resolved']
    assert predict(m, 3.9)['kind'] == 'reference_p3'
    assert not predict(m, 3.9)['branch_resolved']
    assert predict(m, 4.01)['kind'] == 'outside_domain'


def test_rollout_uses_own_predictions_and_stops_on_capture():
    m = make_model([{'q': 3., 'kind': 'return', 'updates': 5, 'next_q': 3.9},
                    {'q': 3.5, 'kind': 'return', 'updates': 5, 'next_q': 3.9},
                    {'q': 4., 'kind': 'reference_p3', 'updates': 20}])
    r = rollout(m, 3.1)
    assert len(r) == 2 and r[-1]['kind'] == 'reference_p3' and r[-1]['elapsed'] == 25


def test_invalid_training_and_refinement():
    with pytest.raises(ValueError): make_model([{'q': 3.}, {'q': 3.}])
    q = refinement_points(table()['rows'], 1)
    assert q == [3.75]
