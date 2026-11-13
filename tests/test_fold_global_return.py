import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("fold_global_return", Path(__file__).parents[1] / "tools/fold_global_return.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
audit_spec = importlib.util.spec_from_file_location("audit_fold_global_return", Path(__file__).parents[1] / "tools/audit_fold_global_return.py")
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def test_return_requires_persistent_same_phase_and_rejects_graze():
    det = m.FirstReturn(2)
    for step in range(1, 6):
        # Row 0 returns on odd readings; row 1 grazes only on the first.
        found, entry = det.update(step, [step % 2 == 1, step == 1])
        assert found.tolist() == ([0] if step == 5 else [])
    assert entry == 1
    assert det.done.tolist() == [True, False]


def test_either_phase_events_and_both_censored_ends():
    v = np.array([[0., 0., 9., 9., 9., 9., 9.], [9., 9., 9., 0., 0., 9., 0.]])
    events = m.pair_events(v, np.zeros_like(v), 1.)
    assert [(r['entry_pair'], r['last_inside_pair']) for r in events] == [(0, 1), (3, 4), (6, 6)]
    assert events[0]['left_censored'] and events[-1]['right_censored']
    assert events[0]['next_entry_pair'] == 3 and events[0]['outside_pairs'] == 1
    assert events[1]['entry_phase'] == 1


def test_full_state_recurrence_rejects_a_misleading_projection():
    x = np.tile([[1., 2.], [-1., 3.]], (100, 1))
    assert m.recurrence(x)['lag'] == 2
    x[:, 1] += np.arange(200) * .01
    assert m.recurrence(x) is None
    with pytest.raises(ValueError):
        m.recurrence(x[:10])


def test_backup_detects_corruption_and_preserves_original(tmp_path):
    source, dest = tmp_path / 'source', tmp_path / 'dest'
    source.mkdir(); dest.mkdir()
    f = source / 'data.bin'
    f.write_bytes(b'science')
    receipt = m.backup_file(f, source, dest)
    assert receipt['sha256'] == m.sha(dest / 'data.bin')
    (dest / 'data.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        m.backup_file(f, source, dest)
    assert f.read_bytes() == b'science'


def test_independent_return_rejects_transverse_false_positive():
    # Projection alone says all five samples are inside, but the first one is
    # far across the channel. No three-sample same-phase visit is present.
    x = np.array([[.2, .2], [4., 0.], [.2, .001], [4., 0.], [.2, .001]])
    assert audit.independent_return(x, np.zeros(2), np.array([1., 0.]), np.array([1., 0.]), 1.) is None
    x = np.vstack([x, [4., 0.], [.3, .001]])
    ret = audit.independent_return(x, np.zeros(2), np.array([1., 0.]), np.array([1., 0.]), 1.)
    assert ret['entry_update'] == 2 and ret['confirmed_update'] == 6


def test_prediction_check_can_detect_missing_transverse_information():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(128, 5))
    rows = [{'split': 'test' if i // 16 in (3, 7) else 'development'} for i in range(128)]
    outcomes = [{'kind': 'return', 'entry_v': float(a[1]), 'entry_update': float(10 + 2*a[1])} for a in x]
    left = right = np.array([1., 0., 0., 0., 0.])
    # Only three transverse dimensions; keep the fourth null to make the
    # synthetic return coordinate completely observable by the proposed model.
    x[:, 4] = 0.
    result = audit.prediction_checks(x, rows, outcomes, np.zeros(5), left, right, 1.)
    scores = result['models']
    assert result['test_rows'] == 32
    assert scores['critical_plus_three_transverse_PCs_quadratic']['entry_v_over_U_RMSE'] < .02
    assert scores['critical_quadratic']['entry_v_over_U_RMSE'] > .2
