"""Independent EXP007 orchestration checks; all numerical solvers are injected."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_hopfield_holdout as h
from hopfield_drift_probe import ProbeFailure
from test_hopfield_holdout_runner import plan


def test_preview_and_both_target_steps_restart_at_same_state(monkeypatch, tmp_path):
    p = plan()
    initial = np.array([.2, -.3, .4])
    original = initial.copy()
    direction = np.array([.6, .8])
    anchor = dict(alpha=1., gamma=10.)
    calls = []
    scored = []

    def integrate(fn, alpha, gamma, state, dt, *, duration, **kw):
        calls.append((alpha, gamma, state.copy(), dt, duration))
        times = np.arange(round(duration / .1) + 1) * .1
        return times, np.tile(state, (len(times), 1))

    def preview_score(reference, times, activations, period, **kw):
        scored.append((times.copy(), period, kw['phase_epoch']))
        return {'aligned_rms': {'aggregate': .1}}

    monkeypatch.setattr(h, 'integrate_trace', integrate)
    monkeypatch.setattr(h, 'score_probe', preview_score)
    monkeypatch.setattr(h, 'waveform_errors', lambda *a, **kw: {'aligned_rms': {'aggregate': .1}})
    monkeypatch.setattr(h, 'atomic_npz', lambda *a, **kw: None)
    epoch = 337 * 10. / 2700
    ref = np.zeros((2700, 3))
    preview = h.run_preview(16, direction, anchor, initial, ref, 10., epoch,
                            .001, None, p, tmp_path, 'synthetic')
    assert preview['status'] == 'complete'
    for dt in p['dt']:
        spec = dict(anchor='synthetic', phase=337, radius=.001, arm='local', dt=dt)
        row = h.run_target(spec, anchor, initial, ref, 10., epoch,
                           direction, None, p, tmp_path)
        assert row['status'] == 'complete' and row['phase_epoch'] == epoch
    assert [(c[3], c[4]) for c in calls] == [(.005, 75), (.01, 300), (.005, 300)]
    for alpha, gamma, state, _, _ in calls:
        np.testing.assert_array_equal(state, original)
        np.testing.assert_array_equal([alpha, gamma], np.array([1., 10.]) + .001 * direction)
    np.testing.assert_array_equal(initial, original)
    assert len(scored[0][0]) == 751 and scored[0][0][0] == 0 and scored[0][0][-1] == 75
    assert scored[0][1:] == (10., epoch)


def test_fatal_sensitivity_failure_preserves_payload_and_exclusive_attempt(monkeypatch, tmp_path):
    p = plan()
    p['source_artifacts'] = []
    for anchor in p['anchors']:
        anchor.update(alpha=1., gamma=10.)
    source = tmp_path / 'synthetic-plan.json'
    source.write_text(json.dumps(p))
    monkeypatch.setattr(h, 'PLAN', source)
    monkeypatch.setattr(h.base, 'git', lambda *args: '' if args[0] == 'status' else 'synthetic-source')
    monkeypatch.setattr(h.base, 'build_engine', lambda path: (None, {'synthetic': True}))
    monkeypatch.setattr(h, 'build_trace', lambda path: (None, {'synthetic': True}))
    monkeypatch.setattr(h, 'baseline', lambda fn, a, p, d: dict(
        anchor=a['id'], eligible=True, certificate=dict(period=10.,
            orbit_states=np.zeros((2701, 3)).tolist(), orbit_activations=np.zeros((2701, 3)).tolist())))
    payload = dict(alpha=1., gamma=10., initial=np.zeros(3),
                   requested_times=np.array([0., 300.]), returned_times=np.array([0.]),
                   augmented_states=np.zeros((9, 1)), nfev=1, wall_seconds=0.)

    def fail(*args, **kw):
        raise ProbeFailure('injected sensitivity failure', payload)

    monkeypatch.setattr(h, 'local_parameter_sensitivity', fail)
    monkeypatch.setattr(h, 'run_target', lambda *a, **kw: pytest.fail('No target may run after this failure'))
    output = tmp_path / 'attempt'
    with pytest.raises(ProbeFailure, match='injected sensitivity failure'):
        h.run(output)
    assert len(json.loads((output / 'baselines.json').read_text())) == 12
    assert 'injected sensitivity failure' in json.loads((output / 'failure.json').read_text())['error']
    with np.load(output / 'unhandled-numerical-failure.npz', allow_pickle=False) as raw:
        for name, value in payload.items():
            np.testing.assert_array_equal(raw[name], value)
    assert not (output / 'receipt.json').exists()
    assert (output / 'records.jsonl').read_text() == ''
    with pytest.raises(FileExistsError):
        h.run(output)
