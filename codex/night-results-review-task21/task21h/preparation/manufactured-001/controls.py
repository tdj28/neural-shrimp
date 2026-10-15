"""Manufactured-only focused controls and one full-shape output observation."""
import argparse
import hashlib
import importlib.util
import json
import resource
import tempfile
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('own_adapter', HERE / 'run.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
r = a.r


def refusal(fn):
    try:
        fn()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('expected refusal')


def check_controls():
    passed = []
    g = r.grid(np.arange(8.), [0.], np.array([[.001, .002, np.inf, np.nan, 1, 1, 1, 1]]),
               np.array([[0., 0, 0, 0, 2, -1, 49, 1.5]]), kind='LS', max_period=48)
    c = r.counts(g)
    assert (c['literal_chaotic'], c['finite_valid_chaotic'], c['invalid_period'], c['nonfinite_LE'],
            c['periodic_valid'], c['finite_valid_cells']) == (2, 1, 3, 2, 1, 3)
    assert c['literal_share_all_allocated'] == .25 and c['finite_valid_chaotic_share_valid'] == 1/3
    passed.append('strict_threshold_nonfinite_invalid_period_and_denominators')
    bad = r.grid([0.], [0.], np.array([[.2]]), np.array([[0]]), kind='TB', max_period=64,
                 saved_chaotic=np.array([[False]]))
    assert r.counts(bad)['saved_vs_reconstructed_chaos_mismatches'] == 1
    assert list(r.cells(bad, 'manufactured'))[0]['literal_chaotic'] is False
    passed.append('saved_mask_mismatch_retained_without_substitution')
    n = r.grid([-.02, .02], [0.], np.ones((1, 2)), np.zeros((1, 2), dtype=int), kind='LS', max_period=48)
    assert r.summary(n)['abs_alpha_below_0.02']['allocated_cells'] == 0
    assert r.summary(n)['abs_alpha_below_0.02']['literal_share_all_allocated'] is None
    assert r.summary(n)['exact_x_zero_indices'] == []
    passed.append('strict_nearzero_and_empty_denominator')
    p = r.grid([-.1, .1], [-.1, .1], np.array([[.2, .2], [.2, -.2]]), np.zeros((2, 2), dtype=int),
               kind='TB', max_period=64, saved_chaotic=np.array([[True, True], [True, False]]))
    x = r.grid([-.1, .1], [0.], np.array([[-.2, .2]]), np.array([[2, 0]]),
               kind='TB', max_period=64, saved_chaotic=np.array([[False, True]]))
    y = r.grid([0.], [-.1, .1], np.array([[-.2], [-.2]]), np.array([[0], [3]]),
               kind='TB', max_period=64, saved_chaotic=np.array([[False], [False]]))
    d, ledger = r.compare_controls(p, x, y)
    assert d['exact_controls_nonchaotic']['count'] == 2 and d['exact_controls_periodic']['count'] == 1
    assert d['near_axis_nonchaotic_all_plane']['count'] == 0 and d['nearest_row'] == d['nearest_column'] == 0
    assert len(ledger) == 4 and ledger[0]['a2_period'] == 0
    passed.append('exact_zero_vs_nearest_axis_nonchaotic_vs_periodic')
    # A row/column exclusion is explicit; it is not silently imposed on the exact-control count.
    px = r.grid([-.1, .1], [-.1, .1], np.array([[-.1, -.1], [-.1, .2]]), np.zeros((2, 2), dtype=int),
                kind='TB', max_period=64, saved_chaotic=np.array([[False, False], [False, True]]))
    dd, ll = r.compare_controls(px, x, y)
    assert dd['near_axis_nonchaotic_all_plane']['eligible_cells'] == 4
    assert dd['near_axis_nonchaotic_excluding_nearest_row_column']['eligible_cells'] == 1
    assert ll[3]['near_axis_nonchaotic_excluding_nearest_row_column'] is True
    passed.append('all_plane_vs_explicit_nearest_row_column_exclusion')
    ybad = dict(y); ybad['y'] = np.array([-.1, np.nextafter(.1, 1.)])
    refusal(lambda: r.compare_controls(p, x, ybad))
    refusal(lambda: r.grid([1., 1.], [0.], np.zeros((1, 2)), np.zeros((1, 2)), kind='LS', max_period=48))
    refusal(lambda: r.grid([0.], [0.], np.zeros((2, 2)), np.zeros((2, 2)), kind='LS', max_period=48))
    passed.append('exact_axes_duplicates_and_shape_refusals')
    lo = r.grid([-.1, .1], [-.3, -.2], np.ones((2, 2)), np.zeros((2, 2), dtype=int), kind='LS', max_period=48)
    actual = np.nextafter(-.1, 0.)
    hi = r.grid([-.1, .1], [actual, 0.], np.ones((2, 2)), np.zeros((2, 2), dtype=int), kind='LS', max_period=48)
    assert r.join_ls(lo, hi)['y'].tolist() == [-.3, -.2, actual, 0.]
    passed.append('fixed_ordinal_join_preserves_actual_nonexact_coordinate')
    leaves = r.comparison_leaves('x', {'a': [1., np.nextafter(1., 2.)], 'b': 1}, {'a': [1., 1.]})
    assert len(leaves) == 3 and leaves[1]['difference_own_minus_saved'] > 0 and not leaves[2]['equal']
    passed.append('all_comparison_leaves_and_last_bit_difference')
    with tempfile.TemporaryDirectory(prefix='task21h-fixture-') as tmp:
        tmp = Path(tmp); f = tmp / 'grid.npz'
        np.savez(f, alpha=np.array([-.1, .1]), log10_temperature=np.array([0.]),
                 lyapunov=np.array([[.1, -.1]]), period=np.array([[0, 1]], dtype=np.int64))
        with np.load(f, allow_pickle=False) as z:
            headers = [{'member': k+'.npy', 'shape': list(z[k].shape), 'dtype': z[k].dtype.str} for k in z.files]
        fake = {'files': {'grid.npz': a.pin(f)}, 'metadata': {'grid.npz': {'npz_members': headers,
                'coordinate_and_done_metadata': {'alpha': [-.1, .1], 'log10_temperature': [0.]}}}}
        assert a.admit_arrays(fake, 'grid.npz')['period'].tolist() == [[0, 1]]
        fake['metadata']['grid.npz']['coordinate_and_done_metadata']['alpha'][0] = -.2
        refusal(lambda: a.admit_arrays(fake, 'grid.npz'))
        card, authority, out = tmp/'card.json', tmp/'authority.json', tmp/'existing'
        out.mkdir(); (out/'sentinel').write_text('keep'); authority.write_text('{}')
        card.write_text(json.dumps({'admission': 'reviewed', 'output': str(out), 'source_sha256': {}, 'inputs': {}}))
        old = a.load, a.input_pins, a.verify
        try:
            a.load = lambda _: {'runs': [None]*15, 'target_file_bytes': 2543411, 'source_map': {}, 'authority': {'path': str(authority)}}
            a.input_pins = lambda *_: []
            a.verify = lambda _: None
            refusal(lambda: a.execute(card, out))
        finally:
            a.load, a.input_pins, a.verify = old
        assert sorted(x.name for x in out.iterdir()) == ['sentinel'] and (out/'sentinel').read_text() == 'keep'
    passed += ['manufactured_NPZ_header_coordinate_association', 'existing_output_refused_without_mutation']
    return passed


def whole_shape(output):
    tic = time.monotonic(); grids = {}
    x = np.linspace(-.12, .12, 128); full_y = np.linspace(-1, .3, 128)
    for b in (4, 8, 12, 16, 20):
        for half in (['full'] if b == 12 else ['lo', 'hi']):
            y = full_y if half == 'full' else full_y[:64] if half == 'lo' else full_y[64:]
            i, j = np.indices((len(y), 128)); P = np.where((i+j)%3 == 0, 7, 0).astype(np.int64)
            L = np.where((2*i+j)%5 < 2, .2, -.2)
            g = r.grid(x, y, L, P, kind='LS', max_period=48)
            key = 'map128-philosophy' if b == 12 else f'ls-l{b}-{half}'
            grids[key] = {'grid': g, 'args': {}, 'source_commit': 'manufactured'}
    x = np.linspace(-.12, .12, 96)
    for order in ('grat4-host16', 'host4-grat16'):
        for lt in (-.1, .1):
            for suffix in ('plane', 'a1', 'a2'):
                xx, yy = (x, x) if suffix == 'plane' else (x, np.array([0.])) if suffix == 'a1' else (np.array([0.]), x)
                i, j = np.indices((len(yy), len(xx))); P = np.where((i+j)%3 == 0, 7, 0).astype(np.int64)
                L = np.where((2*i+j)%5 < 2, .2, -.2); C = (L > .001) & (P == 0)
                key = (order if suffix == 'plane' else 'c-'+order+'-'+suffix) + f'/tb-t{lt:+.3f}'
                args = {k: 'manufactured' for k in ('model', 'block1', 'vectors1', 'concept1', 'block2', 'vectors2', 'concept2', 'transient', 'record', 'eps', 'tol')}
                args['chunk'] = 2304 if suffix == 'plane' else 2048
                grids[key] = {'grid': r.grid(xx, yy, L, P, kind='TB', max_period=64, saved_chaotic=C),
                              'args': args, 'source_commit': 'manufactured'}
    layers, joins, ledger = a.all_results(grids)
    reports = {name: r.summary(item['grid']) for name, item in grids.items()}
    count = 0
    with open(output/'cells.jsonl', 'x') as f:
        for name, item in grids.items():
            for row in r.cells(item['grid'], name):
                f.write(json.dumps(row, allow_nan=False)+'\n'); count += 1
    with open(output/'control-joins.jsonl', 'x') as f:
        for row in ledger: f.write(json.dumps(row, allow_nan=False)+'\n')
    a.write_json(output/'scientific-shape.json', {'grids': reports, 'layers': layers, 'joins': joins})
    assert count == a.CELL_COUNT and len(ledger) == a.JOIN_COUNT and len(grids) == a.NPZ_COUNT
    return {'seconds_through_scientific_serialization': time.monotonic()-tic,
            'observed_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'source_grid_count': len(grids), 'source_cell_count': count, 'join_cell_count': len(ledger),
            'output_files': [a.pin(output / n) for n in ('cells.jsonl', 'control-joins.jsonl', 'scientific-shape.json')],
            'scope': 'One manufactured full-shape arithmetic/serialization observation; no target data, input hashes/I/O excluded, not a worst-case timing/size guarantee.'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); args = ap.parse_args()
    args.out.mkdir(parents=False, exist_ok=False)
    try:
        passed = check_controls()
        measured = whole_shape(args.out)
        a.write_json(args.out/'report.json', {'passed': passed, 'failed': 0, 'measurement': measured,
                    'sources': [a.pin(HERE/n) for n in ('reduce.py', 'run.py', 'controls.py')]})
        print(json.dumps({'controls_passed': len(passed), 'failed': 0, 'seconds': measured['seconds_through_scientific_serialization'],
                          'bytes': sum(x['bytes'] for x in measured['output_files']), 'peak_kib': measured['observed_peak_kib']}))
    except BaseException as e:
        a.write_json(args.out/'failure.json', {'type': type(e).__name__, 'message': str(e)})
        raise


if __name__ == '__main__':
    main()
