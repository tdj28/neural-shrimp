import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("fold_global_return", Path(__file__).parents[1] / "tools/fold_global_return.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


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
