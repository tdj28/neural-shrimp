import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
from fold_return_followup import Engine
from audit_fold_return_followup import departures
from fold_return_surrogate import DepartureDetector
from fold_return_growth import summarize_window
from fold_return_study_controller import owned_process_active


def test_independent_departure_extraction_matches_stream_with_phase_switches():
    x = np.zeros((100, 896)); x[:, 0] = 8.
    x[2:26:2, 0] = .1; x[26:32:2, 0] = 1.2; x[32, 0] = 3.2
    x[49:75:2, 0] = -.1; x[75:79:2, 0] = 1.1; x[79, 0] = 3.3
    left = np.zeros(896); left[0] = 1.
    independent, _ = departures(x, np.zeros(896), left, left, 1.)
    d = DepartureDetector(1.)
    streamed = [e for i in range(len(x)) if (e := d.update(i, x[i,0], 0.))]
    assert [e['exit'] for e in independent] == [32, 79] == [e['exit'] for e in streamed]


def fake_engine(tmp_path, fail_at=None, backup_fail=False):
    e = Engine.__new__(Engine); e.B = 16
    e.orbit = np.zeros((3, 896)); e.orbit[:, 0] = [4, 5, 6]
    left = np.zeros(896); left[0] = 1
    e.data = {'centre': np.zeros(896), 'left': left, 'right': left}
    e.parameters = {'width': 1., 'transverse_limit': .05}
    e.plan = {'local_wall_seconds_per_stage': 30}
    class Storage:
        def __init__(self): self.files = []; self.out = tmp_path
        def save(self, name, value):
            self.files.append((name, value))
            if backup_fail and 'updates-' in name: raise IOError('Archive unavailable')
    e.evidence = Storage(); e.calls = 0
    def step(x):
        e.calls += 1
        if e.calls == fail_at: raise RuntimeError('Injected model failure')
        return x+.00001, np.zeros((16,4), int), np.ones((16,4))*.1
    e.step = step
    return e


def test_failed_update_preserves_unflushed_prefix(tmp_path):
    e = fake_engine(tmp_path, fail_at=3)
    with pytest.raises(RuntimeError): e.run_batch(e.orbit[:2], 'sample', 10, False)
    prefix = [v for n,v in e.evidence.files if 'interrupted-prefix' in n]
    assert len(prefix) == 1 and prefix[0]['states'].shape == (2,16,896)
    assert prefix[0]['valid_through'].tolist() == [2,2]+[0]*14


def test_backup_failure_prevents_any_further_model_updates(tmp_path):
    e = fake_engine(tmp_path, backup_fail=True)
    with pytest.raises(IOError): e.run_batch(e.orbit[:2], 'sample', 130, False)
    assert e.calls == 128


def test_growth_excludes_frozen_padding_and_declares_short_windows():
    logs = np.ones((12,3))*.2
    logs[8:,0] = 1000.  # Frozen padding must never contribute.
    result = summarize_window(logs, np.array([8,12,3]), 4, 16)
    assert result[0]['mean'] == pytest.approx(.2)
    assert result[0]['measured_updates'] == 4
    assert result[1]['measured_updates'] == 8 and not result[1]['full_planned_window']
    assert result[2]['mean'] is None and result[2]['measured_updates'] == 0


def test_monitor_rejects_pid_reuse_without_signalling_any_process():
    import os
    pid = os.getpid()
    command = [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
    assert owned_process_active(pid,command)
    with pytest.raises(ValueError): owned_process_active(pid,['unrelated-command'])
    assert not owned_process_active(2147483647,command)
