import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fold_return_escape as fe
import fold_return_unwrap as uw
from fold_return_zoom import zoom_items, zoom_batches


def test_ulam_recovers_escape_rate_of_tripling_map_with_middle_hole():
    # x -> 3x mod 1 with the hole [1/3, 2/3): exactly one third of the uniform mass leaves per step.
    rng = np.random.default_rng(0)
    x = rng.random(60000)
    pts = [(3+1.12*a, None if 1/3 <= a < 2/3 else 3+1.12*((3*a) % 1)) for a in x]
    est = fe.ulam(pts, bins=27)
    assert est['kappa'] == pytest.approx(1/3, abs=0.01)


def test_ulam_refuses_empty_bins():
    with pytest.raises(ValueError):
        fe.ulam([(3.01, 3.02)], bins=4)


def test_garwood_interval_matches_tabulated_values():
    assert fe.garwood(0) == pytest.approx((0.0, 3.6889), abs=1e-3)
    assert fe.garwood(10) == pytest.approx((4.7954, 18.3904), abs=1e-3)


def test_unwrapped_time_is_continuous_across_a_wrap():
    step = 1.1
    # Side A: departure at t with the previous reading just below the line.
    tA, _ = uw.unwrap(110, 3-1e-9+step, 3-1e-9)
    # Side B: the same reading now just past the line, one channel step (two updates) earlier.
    tB, _ = uw.unwrap(108, 3+1e-9, 3+1e-9-step)
    assert tA == pytest.approx(tB, abs=1e-8)
    with pytest.raises(ValueError):
        uw.unwrap(100, 2.99, 2.5)


def test_exit_step_inverse_round_trips():
    rows = []
    for k, qp in enumerate(np.linspace(3.0, 4.09, 60)):
        qm = qp-1.1-0.01*(qp-3)**2
        t = 100+2*(k % 3)+(k % 2)
        tau, f = uw.unwrap(t, qp, qm)
        rows.append({'kind': 'return', 'next_q': qp, 'q_minus': qm, 'updates': t, 'tau': tau})
    s = uw.ExitStep(rows)
    for r in rows:
        t, q = s.wrap(r['tau'], r['updates'] % 2)
        assert t == r['updates'] and q == pytest.approx(r['next_q'], abs=1e-8)


def test_zoom_design_is_deterministic_and_covers_every_item_once():
    spec = {'domain': [3.0, 4.12], 'bases': 64, 'seed': 5, 'unresolved_interval': [3.05, 3.56],
            'deltas': [1e-3, 1e-6], 'replicates': 16}
    a, inside = zoom_items(spec); b, _ = zoom_items(spec)
    assert a == b and len(a) == 64+2*len(inside)+min(16, len(inside))
    batches = zoom_batches(a, 16)
    flat = sorted(i for bt in batches for i in bt)
    assert flat == list(range(len(a)))
    assert all(a[i]['role'] == 'replicate' for i in batches[-1])
