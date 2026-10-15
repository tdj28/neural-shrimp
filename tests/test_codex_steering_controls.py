"""Only generated toy arrays; no real scan or producer module is loaded."""
import importlib.util
from pathlib import Path
import json
import zipfile
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE if (HERE / 'tabulate.py').exists() else HERE.parent / 'codex' / 'steering-controls-crosscheck'
spec = importlib.util.spec_from_file_location('task16_tabulate', SOURCE / 'tabulate.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def data(tmp_path, layout='full', done=6, ls=None, qs=None, name='x.npz', offset=0):
    l = np.array([[0., .001, .002], [.01, -.1, .1]]) if ls is None else np.asarray(ls)
    q = np.array([[0, 0, 0], [2, 1, 0]]) if qs is None else np.asarray(qs)
    payload = {'log10_temperature': np.array([-.1, .2]), 'alpha': np.array([-.01, 0., .01]),
               'lyapunov': l if layout == 'full' else l.ravel(),
               'period': q if layout == 'full' else q.ravel()}
    if layout == 'partial':
        payload['done'] = np.array(done)
    p = tmp_path / name
    np.savez(p, **payload)
    return {'path': name, 'bytes': p.stat().st_size, 'sha256': T.digest(p),
            'temperature_hex': [float(x).hex() for x in payload['log10_temperature']],
            'alpha_hex': [float(x).hex() for x in payload['alpha']],
            'shape': [2, 3], 'layout': layout, 'row_offset': offset, 'expected_done': done,
            'original_attempt_outcome': 'timeout' if layout == 'partial' else 'completed',
            'outcome_authority': 'synthetic fixture only'}


def dataset(s):
    return {'id': 'toy', 'shape': [2, 3], 'max_period': 4, 'axis_segment': 0,
            'segments': s, 'metadata': {'synthetic': True}}


def test_strict_threshold_and_period_discordance(tmp_path):
    s = data(tmp_path)
    report, rows, _ = T.tabulate_dataset(tmp_path, dataset([s]))
    assert [r['disposition'] for r in rows] == ['no_period_not_above_threshold',
        'no_period_not_above_threshold', 'chaotic', 'detected_period', 'detected_period', 'chaotic']
    assert report['summary']['chaotic'] == 2
    assert report['summary']['detected_period_above_exponent_threshold'] == 1
    assert report['summary']['median_detected_period'] == 1.5


def test_partial_row_and_zero_suffix_not_completed(tmp_path):
    s = data(tmp_path, 'partial', done=4)
    report, rows, _ = T.tabulate_dataset(tmp_path, dataset([s]))
    assert report['summary']['completed'] == 4
    assert [r['disposition'] for r in rows[4:]] == ['unfinished', 'unfinished']
    assert rows[3]['completed'] and rows[4]['lyapunov'] is None
    assert report['segments'][0]['original_attempt_outcome'] == 'timeout'


def test_nonfinite_and_invalid_periods_remain_visible(tmp_path):
    s = data(tmp_path, ls=[[np.nan, np.inf, -np.inf], [.1, .1, .1]], qs=[[2, 0, 0], [-1, .5, 5]])
    report, rows, _ = T.tabulate_dataset(tmp_path, dataset([s]))
    assert report['summary']['valid'] == 0
    assert report['summary']['dispositions'] == {'invalid': 6}
    assert report['summary']['chaotic_share_valid'] is None
    assert report['summary']['period_histogram_all_completed_valid_period_fields'] == {'0': 2, '2': 1}
    assert [r['lyapunov'] for r in rows[:3]] == ['NaN', '+Inf', '-Inf']
    json.dumps(report, allow_nan=False)


def test_missing_file_does_not_drop_grid(tmp_path):
    s = data(tmp_path)
    (tmp_path / s['path']).unlink()
    report, rows, _ = T.tabulate_dataset(tmp_path, dataset([s]))
    assert not report['admitted'] and len(rows) == 6
    assert report['summary']['dispositions'] == {'admission_failed': 6}


def test_overlap_refuses_instead_of_overwrite(tmp_path):
    a = data(tmp_path, name='a.npz')
    b = data(tmp_path, name='b.npz')
    report, rows, _ = T.tabulate_dataset(tmp_path, dataset([a, b]))
    assert not report['admitted'] and 'overlapping' in report['error']


def test_literal_axis_mismatch_not_tolerance_merged(tmp_path):
    s = data(tmp_path)
    a = T.load_segment(tmp_path, s)
    b = list(a)
    b[0] = b[0].copy()
    b[0][0] = np.nextafter(b[0][0], np.inf)
    with pytest.raises(ValueError, match='frozen temperature axis mismatch'):
        T.assemble(dataset([s, s]), [a, tuple(b)])


def test_duplicate_axis_refused():
    with pytest.raises(ValueError, match='unique'):
        T.axis(np.array([1., 1.]), 2, 'toy')
    with pytest.raises(ValueError, match='unique'):
        T.axis(np.array([2, 1], dtype=np.uint64), 2, 'toy')


def test_bins_exact_zero_signed_nearest_empty_and_last_endpoint(tmp_path):
    s = data(tmp_path)
    _, rows, _ = T.tabulate_dataset(tmp_path, dataset([s]))
    for r, a in zip(rows, [-.12, -.06, -.001, .001, .03, .12]):
        r['alpha'] = a
    sub = T.subsets(rows, [-.12, -.06, -.001, .001, .03, .12], [-.1, .2])
    assert [b['all']['expected'] for b in sub['absolute_alpha_bins']] == [2, 0, 1, 1, 2]
    assert sub['zero_and_signed_nearest']['exact_zero']['summary']['expected'] == 0
    assert sub['absolute_alpha_bins'][1]['all']['chaotic_share_valid'] is None
    assert sub['zero_and_signed_nearest']['nearest_negative']['alpha'] == [-.001]
    assert sub['zero_and_signed_nearest']['nearest_positive']['alpha'] == [.001]


def test_layout_done_hash_and_object_refusal(tmp_path):
    s = data(tmp_path, 'partial', done=4)
    wrong = dict(s, expected_done=3)
    with pytest.raises(ValueError, match='frozen metadata'):
        T.load_segment(tmp_path, wrong)
    with pytest.raises(ValueError, match='input hash'):
        T.load_segment(tmp_path, dict(s, sha256='0' * 64))
    with pytest.raises(ValueError, match='layout'):
        T.load_segment(tmp_path, dict(s, layout='full'))
    bad = tmp_path / 'object.npz'
    np.savez(bad, log10_temperature=[-.1, .2], alpha=[-.01, 0, .01],
             lyapunov=np.array([[{}, {}, {}], [{}, {}, {}]], dtype=object), period=np.zeros((2, 3)))
    obj = dict(s, path=bad.name, sha256=T.digest(bad), bytes=bad.stat().st_size, layout='full', expected_done=6)
    with pytest.raises(ValueError, match='Object arrays'):
        T.load_segment(tmp_path, obj)


def test_duplicate_zip_member_refused(tmp_path):
    s = data(tmp_path)
    p = tmp_path / s['path']
    with zipfile.ZipFile(p, 'a') as z:
        z.writestr('alpha.npy', z.read('alpha.npy'))
    s.update(bytes=p.stat().st_size, sha256=T.digest(p))
    with pytest.raises(ValueError, match='duplicate archive'):
        T.load_segment(tmp_path, s)


def toy_ordinal():
    t = np.array([0., 1., 2., 3.]); a = np.array([-.01, .01])
    top = np.array([np.nextafter(2., np.inf), 3.])
    low_l = np.full((4, 2), .01); low_q = np.zeros((4, 2))
    high_l = np.full((2, 2), -.2); high_q = np.zeros((2, 2))
    def segment(ts, offset, size, selection):
        return {'path': 'synthetic', 'sha256': '0' * 64, 'row_offset': offset,
                'shape': [size, 2], 'temperature_hex': [float(x).hex() for x in ts],
                'alpha_hex': [float(x).hex() for x in a],
                'selected_local_flat_range': selection, 'original_attempt_outcome': 'timeout',
                'outcome_authority': 'synthetic'}
    spec = {'id': 'toy', 'shape': [4, 2], 'axis_segment': 0, 'max_period': 4,
            'ordinal_amendment': 'a' * 64,
            'segments': [segment(t, 0, 4, [0, 4]), segment(top, 2, 2, [0, 4])]}
    return spec, [(t, a, low_l, low_q, 6), (top, a, high_l, high_q, 4)]


def test_fixed_ordinal_keeps_unused_overlap_and_actual_coordinates():
    spec, inputs = toy_ordinal()
    data = T.assemble(spec, inputs)
    rows = T.cell_records(spec, data)
    assert len(rows) == 8 and len(data['source_rows']) == 12
    assert T.summarize(rows)['chaotic'] == 4
    unused = T.overlap_rows(rows, data['source_rows'])
    assert len(unused) == 2 and unused[0]['exponent_selected_minus_nonselected'] == -.2 - .01
    assert unused[0]['temperature_difference'] != 0
    assert rows[4]['log10_temperature'] == inputs[1][0][0]
    assert rows[4]['canonical_temperature'] == 2.
    assert data['segments'][0]['nonselected_completed'] == 2
    assert len([r for r in data['source_rows'] if not r['completed']]) == 2
    # Literal common-grid comparison loses the one rounded temperature row.
    reference = [dict(r, log10_temperature=r['canonical_temperature']) for r in rows]
    compared = T.matched_groups({'x': rows, 'y': reference},
        [{'id': 'toy', 'datasets': ['x', 'y'], 'metadata_basis': 'synthetic'}])
    assert compared[0]['common_valid_exact_coordinates'] == 6


def test_ordinal_missing_duplicate_or_changed_frozen_axis_refused():
    spec, inputs = toy_ordinal()
    with pytest.raises(ValueError, match='missing/extra'):
        T.assemble(spec, inputs[:1])
    spec['segments'][0]['selected_local_flat_range'] = [0, 6]
    with pytest.raises(ValueError, match='overlapping selected'):
        T.assemble(spec, inputs)
    spec, inputs = toy_ordinal()
    inputs[1][0][0] = np.nextafter(inputs[1][0][0], np.inf)
    with pytest.raises(ValueError, match='frozen temperature'):
        T.assemble(spec, inputs)


def test_uncovered_cells_are_not_completed_or_filled_from_unused():
    spec, inputs = toy_ordinal()
    spec['segments'][0]['selected_local_flat_range'] = [0, 2]
    data = T.assemble(spec, inputs)
    rows = T.cell_records(spec, data)
    assert [r['disposition'] for r in rows[2:4]] == ['uncovered', 'uncovered']
    assert len(T.overlap_rows(rows, data['source_rows'])) == 4


def test_actual_metadata_only_specification_no_array_decode(monkeypatch):
    # This uses only source/schema/authority JSON from the expressly admitted metadata inventory.
    root = Path('/tmp/task16-original-authority-001')
    if not root.exists():
        pytest.skip('metadata exports are not present in this checkout')
    r = importlib.util.spec_from_file_location('task16_run', SOURCE / 'run.py')
    R = importlib.util.module_from_spec(r); r.loader.exec_module(R)
    def no_arrays(*args, **kwargs):
        raise AssertionError('forbidden NPZ decode in metadata control')
    monkeypatch.setattr(np, 'load', no_arrays)
    refs = {'inventory': {'sha256': 'fdd511be93d478850fda51706fdbaa019e9b6292eb9b9accf1bb2eb619d34f19'},
            'commands': {'sha256': '44c5191c45d43537e7a3f3a946512e855ee309be460b11125fde524e30d9f761'},
            'join_policy': {'sha256': 'fc031137e7430a2d227aadc1fd1df8765cb6a3e3ef82435f1492ec10a7cc2ccb'}}
    meta = json.loads(Path('/tmp/task16-input-metadata-001.json').read_text())
    auth = json.loads((root / 'authority.json').read_text())
    commands = json.loads((root / 'random-partition-command-authority.json').read_text())
    specs = R.specifications(meta, auth, commands, refs)
    assert len(specs) == 16 and sum(np.prod(s['shape']) for s in specs) == 112640
    assert sum(np.prod(x['shape']) for s in specs for x in s['segments']) == 124928
    for s in specs:
        if s['id'].startswith('q05-random'):
            assert s['segments'][0]['selected_local_flat_range'] == [0, 12288]
            assert s['segments'][1]['row_offset'] == 96
    original = meta['segments'][0]['sha256']
    meta['segments'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='original member binding'):
        R.specifications(meta, auth, commands, refs)
    meta['segments'][0]['sha256'] = original
    meta['segments'][0]['dataset_id'] = 'unrequested-dataset'
    with pytest.raises(ValueError, match='dataset IDs'):
        R.specifications(meta, auth, commands, refs)
