"""Task21h bounded saved-data adapter. Prepared source; no producer execution."""
import argparse
import hashlib
import importlib.util
import json
import resource
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('task21h_reduce', HERE / 'reduce.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
EXPECTED = ([f'ls-l{b}-{h}' for b in (4, 8, 16, 20) for h in ('lo', 'hi')] +
            ['grat4-host16', 'host4-grat16'] +
            [f'c-{o}-{a}' for o in ('grat4-host16', 'host4-grat16') for a in ('a1', 'a2')] + ['map128-philosophy'])
NPZ_COUNT = 21
CELL_COUNT = 119552
JOIN_COUNT = 36864


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def verify(p):
    r.require(pin(p['path']) == p, f"file pin mismatch: {p['path']}")


def load(p):
    verify(p)
    return json.loads(Path(p['path']).read_text())


def write_json(path, value):
    with open(path, 'x') as f:
        json.dump(r.clean(value), f, indent=1, allow_nan=False)
        f.write('\n')


def input_pins(inputs, source_map):
    rows = [inputs[k] for k in ('authority', 'source_map', 'coordinate_association')]
    rows += [p for run in inputs['runs'] for p in run['files'].values()]
    rows += [{k: p[k] for k in ('path', 'bytes', 'sha256')} for group in ('source_files', 'claims') for p in source_map[group]]
    unique = {}
    for p in rows:
        if p['path'] in unique:
            r.require(unique[p['path']] == p, 'conflicting duplicate pin')
        unique[p['path']] = p
    return list(unique.values())


def admit_arrays(run, name):
    filepin = run['files'][name]
    metadata = run['metadata'][name]
    headers = metadata['npz_members']
    with np.load(filepin['path'], allow_pickle=False) as z:
        r.require(sorted(z.files) == sorted(h['member'][:-4] for h in headers), 'NPZ member inventory mismatch')
        data = {}
        for h in headers:
            key = h['member'][:-4]
            a = z[key]
            r.require(list(a.shape) == h['shape'] and a.dtype.str == h['dtype'], 'NPZ header mismatch')
            r.require(a.dtype.kind in 'bif', 'unexpected nonnumeric member')
            data[key] = a.copy()
    for name, expected in metadata['coordinate_and_done_metadata'].items():
        r.require(name in data and np.array_equal(data[name], np.asarray(expected)), 'coordinate/done metadata mismatch')
    return data


def make_grids(inputs):
    out, comparisons = {}, []
    r.require([x['id'] for x in inputs['runs']] == EXPECTED, 'fixed original run order/coverage differs')
    count = 0
    for run in inputs['runs']:
        for name in sorted(k for k in run['files'] if k.endswith('.npz')):
            count += 1
            a = admit_arrays(run, name)
            json_name = name[:-4] + '.json'
            saved = json.loads(Path(run['files'][json_name]['path']).read_text())
            args = saved['args']
            r.require(args == run['metadata'][json_name]['args'], 'original args mismatch')
            if run['kind'] in ('LS', 'LS-reference'):
                r.require(name == 'scan.npz' and args['model'] == 'qwen0.5b' and args['concept'] == 'philosophy', 'LS identity mismatch')
                r.require(args['prompt'] == 'open' and args['opener'] == 'Hello!' and args['window'] == 1, 'LS prefix mismatch')
                r.require(args['dtype'] == 'float64' and args['transient'] == 500 and args['samples'] == 300 and args['record'] == 80,
                          'LS fixed horizon mismatch')
                r.require(args['max_period'] == 48 and args['tol'] == 1e-8 and args['eps'] == 1e-6 and args['chaos_level'] == 1e-3,
                          'LS fixed classifier settings mismatch')
                r.require(args.get('init_states') is None and not args.get('save_states', False), 'unexpected warm start/state saving')
                r.require('done' not in a, 'expected completed shaped LS scan, not partial checkpoint')
                g = r.grid(a['alpha'], a['log10_temperature'], a['lyapunov'], a['period'], kind='LS', max_period=48)
                r.require(list(g['L'].shape) == args['resolution'], 'LS declared resolution mismatch')
                identifier = run['id']; base = r.counts(g)
                for field, own in [('cells', g['L'].size), ('lyapunov_positive', base['LE_positive_literal']),
                                   ('lyapunov_range', [float(a['lyapunov'].min()), float(a['lyapunov'].max())]),
                                   ('period_counts', base['period_histogram'])]:
                    comparisons += r.comparison_leaves(identifier + '/' + field, own, saved[field])
            else:
                r.require(args['model'] == 'qwen0.5b' and args['block1'] == 4 and args['block2'] == 16, 'TB block/model mismatch')
                r.require(args['transient'] == 500 and args['record'] == 300 and args['eps'] == 1e-6 and args['tol'] == 1e-8,
                          'TB fixed finite-run settings mismatch')
                r.require(float(a['log10_temperature']) == saved['log10_temperature'] in (-.1, .1), 'TB temperature mismatch')
                g = r.grid(a['alpha1'], a['alpha2'], a['lyapunov'], a['period'], kind='TB', max_period=64, saved_chaotic=a['chaotic'])
                r.require(g['L'].shape == (args.get('resolution2') or args['resolution'], args.get('resolution1') or args['resolution']),
                          'TB declared rectangular shape mismatch')
                identifier = run['id'] + '/' + name[:-4]
                comparisons.append(r.comparison(identifier + '/chaotic_share', float(g['literal'].mean()), saved['chaotic_share']))
            r.require(identifier not in out, 'duplicate grid')
            out[identifier] = {'grid': g, 'args': args, 'run': run['id'], 'original_attempt': run['attempt'],
                               'source_commit': run['source_commit'], 'original_status_lines': run['original_tool_status_lines']}
    r.require(count == NPZ_COUNT and sum(x['grid']['L'].size for x in out.values()) == CELL_COUNT, 'full fixed cell coverage differs')
    return out, comparisons


def all_results(grids):
    layers, joins, control_ledger = {}, {}, []
    ref = grids['map128-philosophy']['grid']
    for block in (4, 8, 12, 16, 20):
        g = ref if block == 12 else r.join_ls(grids[f'ls-l{block}-lo']['grid'], grids[f'ls-l{block}-hi']['grid'])
        r.require(g['L'].shape == (128, 128) and np.array_equal(g['x'], ref['x']), 'joined LS shape/alpha association differs')
        layers[str(block)] = r.summary(g)
        layers[str(block)]['temperature_ordinal_comparison_to_block12'] = [
            {'row': i, 'actual': float(x), 'block12': float(y), 'difference': float(x-y), 'equal': bool(x == y)}
            for i, (x, y) in enumerate(zip(g['y'], ref['y']))]
        layers[str(block)]['coordinate_scope'] = 'Original low-then-high row order; retain actual coordinates. No interpolation or claim of exact shared temperature coordinates.'
    shared = ('model', 'block1', 'vectors1', 'concept1', 'block2', 'vectors2', 'concept2', 'transient', 'record', 'eps', 'tol')
    for order in ('grat4-host16', 'host4-grat16'):
        for lt in (-.1, .1):
            stem = f'tb-t{lt:+.3f}'
            plane, a1, a2 = [grids[k + '/' + stem] for k in (order, 'c-' + order + '-a1', 'c-' + order + '-a2')]
            for key in shared:
                r.require(plane['args'][key] == a1['args'][key] == a2['args'][key], f'TB/TBC association mismatch: {key}')
            key = order + '/' + stem
            result, ledger = r.compare_controls(plane['grid'], a1['grid'], a2['grid'])
            result['batch_sizes'] = {name: item['args']['chunk'] for name, item in [('plane', plane), ('a1', a1), ('a2', a2)]}
            result['source_commits'] = {name: item['source_commit'] for name, item in [('plane', plane), ('a1', a1), ('a2', a2)]}
            result['scope'] = 'Exact control coordinates, finite one-twin labels. Different batch layouts/twin positions; two hooks remain registered with one zero. No (0,0) control and no causal/global-chaos inference.'
            joins[key] = result
            control_ledger.extend({'comparison': key, **row} for row in ledger)
    r.require(len(control_ledger) == JOIN_COUNT, 'control-join full cell coverage differs')
    return layers, joins, control_ledger


def execute(card_path, output):
    started = time.monotonic()
    card = json.loads(card_path.read_text())
    r.require(card['admission'] == 'reviewed', 'prepared card is unlaunchable')
    r.require(output.resolve() == Path(card['output']).resolve(), 'fixed output path differs')
    for name, expected in card['source_sha256'].items():
        r.require(sha(HERE / name) == expected, 'captured scientific source mismatch')
    inputs = load(card['inputs'])
    source_map = load(inputs['source_map'])
    pins = [card['inputs']] + input_pins(inputs, source_map)
    r.require(len(inputs['runs']) == 15 and inputs['target_file_bytes'] == 2543411, 'fixed metadata allocation differs')
    for p in pins:
        verify(p)
    # Original status/closure is inherited from the exact reviewed authority, not claimed as our observation.
    authority = json.loads(Path(inputs['authority']['path']).read_text())
    output.mkdir(parents=False, exist_ok=False)
    try:
        grids, comparisons = make_grids(inputs)
        layers, joins, join_ledger = all_results(grids)
        with open(output / 'cells.jsonl', 'x') as f:
            for name, item in grids.items():
                for row in r.cells(item['grid'], name):
                    f.write(json.dumps(row, allow_nan=False) + '\n')
        with open(output / 'control-joins.jsonl', 'x') as f:
            for row in join_ledger:
                f.write(json.dumps(row, allow_nan=False) + '\n')
        report = {'schema': 'task21h-independent-saved-grid-v1', 'computational_status': 'complete',
                  'source_sha256': card['source_sha256'], 'input_pins': pins,
                  'original_authority': inputs['authority'], 'source_map': inputs['source_map'],
                  'coordinate_association': inputs['coordinate_association'],
                  'grids': {name: {k: v for k, v in item.items() if k != 'grid'} | {'summary': r.summary(item['grid'])}
                            for name, item in grids.items()},
                  'layer_sweep': layers, 'two_block_controls': joins, 'producer_comparisons': comparisons,
                  'comparison_count': len(comparisons), 'comparison_nonzero_or_unequal': sum(not x['equal'] for x in comparisons),
                  'coverage': {'original_runs': 15, 'source_grids': NPZ_COUNT, 'source_cells': CELL_COUNT, 'control_join_cells': JOIN_COUNT},
                  'scope': 'Saved LE/period/mask arithmetic only. Complete-shaped saved artifacts do not rewrite original LS tool failures. No model, derivatives, spectra, generated replies, native attractor or vector-norm reconstruction. Literal summaries and finite qualifications both retained.',
                  'runtime': {'seconds_before_posthash_report_write': time.monotonic()-started,
                              'observed_peak_kib_before_posthash_report_write': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              'scope': 'Includes cells/join serialization; excludes final pin rehash and report/manifest serialization. External supervisor observes full worker wall/closure, not a full-lifetime peak RSS.'}}
        for p in pins:
            verify(p)
        write_json(output / 'report.json', report)
        manifest = {'files': [pin(output / x) for x in ('cells.jsonl', 'control-joins.jsonl', 'report.json')],
                    'computational_status': 'complete'}
        write_json(output / 'manifest.json', manifest)
    except BaseException as e:
        write_json(output / 'failure.json', {'type': type(e).__name__, 'message': str(e)})
        raise


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--card', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); execute(args.card, args.out)


if __name__ == '__main__':
    main()
