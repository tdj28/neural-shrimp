"""Task25 own six-component saved-data adapter. Prepared, never self-admits.

Only the separately reviewed card selects immutable original evidence. No
producer imports, native model operations, interpolation, or retry paths.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
COMPONENT_HASHES = {
    'reducers': 'a0ecd8120b7032d6bab1403bc42c3cbcce43d285be663c1a5b97c2f869246f33',
    'prefix': 'f82cdc47c028874008bddbe1beda37ab2c1bd2dbaeb86596e0e924019f59f97d',
    'start': '5e2b3b43acfec280b3045cbe9af6dfc7500c846d37d803484a5d6565a8f86bc9',
    'plateau': '818a0ae44f2bb58afe2868070bb9e22e5059a14ea5ce539b8a8b67360d5e07d3',
    'depth': '454dcda9ff4cac3fb7ac1f8e2affc784740117da58fca538e1efba7e0b749fea',
}
COMPONENTS = ('scale', 'gain', 'ps', 'cp', 'plateau', 'depth')
P3_ROLES = (
    ('qwen_reference', 'qwen7b-wide', 'Qwen2.5', '7B'),
    ('qwen14', 'qwen14b-wide', 'Qwen2.5', '14B'),
    ('qwen32', 'qwen32b-wide', 'Qwen2.5', '32B'),
    ('olmo_reference', 'olmo2-7b-wide', 'OLMo-2', '7B'),
    ('olmo13', 'olmo2-13b-wide', 'OLMo-2', '13B'),
    ('olmo32', 'olmo2-32b-wide', 'OLMo-2', '32B'),
)
P3_VIEW_NAMES = ('manuscript_wide7B', 'historical_narrow_qwen7B')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            size += len(part)
            h.update(part)
    return {'path': str(p), 'bytes': size, 'sha256': h.hexdigest()}


def check_pin(expected):
    require(set(expected) == {'path', 'bytes', 'sha256'}, 'exact path/bytes/SHA pin required')
    require(pin(expected['path']) == expected, 'input/source bytes differ: '+expected['path'])


def read_json(path):
    return json.loads(Path(path).read_text())


def at(value, pointer):
    require(pointer == '' or pointer.startswith('/'), 'JSON pointer required')
    for token in pointer.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def exact(actual, expected):
    # JSON metadata must not silently equate a Boolean with a number.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(exact(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def modules():
    result = {}
    for name, expected in COMPONENT_HASHES.items():
        path = HERE/(name+'.py')
        require(pin(path)['sha256'] == expected, 'accepted component source changed: '+name)
        spec = importlib.util.spec_from_file_location('task25_'+name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def unique(rows, field, expected=None):
    keys = [r[field] for r in rows]
    require(len(keys) == len(set(keys)), 'duplicate role/identity: '+field)
    if expected is not None:
        require(set(keys) == set(expected), 'fixed role coverage differs: '+field)
    return {r[field]: r for r in rows}


def availability(desc):
    require(type(desc['available']) is bool, 'explicit availability required')
    require('provenance' in desc and 'reason' in desc, 'original disposition and missing reason required')
    if not desc['available']:
        require(bool(desc['reason']) and desc.get('file_id') is None, 'unavailable role cannot substitute a file')
    return desc['available']


def document(desc, pins):
    if not availability(desc):
        return None
    value = read_json(pins[desc['file_id']]['path'])
    require(exact(value['args'], desc['expected_args']), 'original JSON argument association differs')
    for assertion in desc.get('metadata_assertions', []):
        require(exact(at(value, assertion['pointer']), assertion['expected']), 'metadata assertion differs: '+assertion['pointer'])
    return value


def saved_rows(desc, pins):
    value = document(desc, pins)
    if value is None:
        return None
    rows = value['cells']
    require(isinstance(rows, list) and len(rows) == desc['expected_rows'], 'saved cell coverage differs')
    for field, expected in desc['expected_coordinates'].items():
        require(exact([r[field] for r in rows], expected), 'saved coordinate order differs: '+field)
    return rows


def arrays(desc, pins):
    import numpy as np
    if not availability(desc):
        return None
    with np.load(pins[desc['file_id']]['path'], allow_pickle=False) as z:
        require(set(z.files) == set(desc['members']), 'NPZ member inventory differs')
        result = {}
        for key, header in desc['members'].items():
            value = z[key]
            require(list(value.shape) == header['shape'] and value.dtype.str == header['dtype'], 'NPZ shape/dtype differs: '+key)
            require(not value.dtype.hasobject, 'object arrays forbidden')
            result[key] = value.copy()
    for key, expected in desc['expected_axes'].items():
        require(exact(result[key].tolist(), expected), 'exact original axis differs: '+key)
    return result


def completed(desc, n, data=None):
    rule = desc['completion']
    require(rule['kind'] in ('all_saved_rows', 'recorded_prefix'), 'explicit completion basis required')
    count = n if rule['kind'] == 'all_saved_rows' else rule['count']
    require(type(count) is int and 0 <= count <= n, 'completion prefix outside array')
    if data is not None and 'done' in data:
        require(rule['kind'] == 'recorded_prefix' and int(data['done']) == count, 'saved done prefix differs')
    return [i < count for i in range(n)]


def scale_inputs(roles, pins, component):
    wanted = {s for m in component.MODEL_SPECS for group in ('primary_sources', 'supplemental_sources') for s in m[group]}
    unique(roles, 'id', wanted)
    result = {}
    for d in roles:
        a = arrays(d, pins)
        rows = []
        if a is not None:
            t = a['log10_temperature'].reshape(-1)
            require(a['alpha'].size == 1 and float(a['alpha'].reshape(-1)[0]) == 0., 'scale alpha must be exact zero')
            require(len(t) == d['expected_rows'], 'scale temperature coverage differs')
            values = {k: a[k].reshape(-1) for k in ('lyapunov', 'period', 'entropy')}
            require(all(len(v) == len(t) for v in values.values()), 'scale cell-axis shape mismatch')
            flags = completed(d, len(t), a)
            rows = [{'log10_temperature': float(t[i]), **{k: float(v[i]) for k, v in values.items()}, 'completed': flags[i]} for i in range(len(t))]
        result[d['id']] = {'id': d['id'], 'available': d['available'], 'run_disposition': d['provenance'],
                           'expected_rows': d['expected_rows'], 'rows': rows, 'alpha': d['alpha'], 'max_period': d['max_period']}
    return result


def gain_inputs(roles, specs, pins):
    unique(roles, 'id', {s['source_id'] for s in specs})
    result = {}
    fields = ('log10_temperature', 'entropy_mean', 'log_g_net', 'log_g_read', 'lyapunov', 'period', 'boundary_term')
    for d in roles:
        rows = saved_rows(d, pins)
        flags = completed(d, len(rows)) if rows is not None else []
        normalized = [{**{k: r[k] for k in fields}, 'completed': flags[i]} for i, r in enumerate(rows or [])]
        result[d['id']] = {'id': d['id'], 'available': d['available'], 'expected_rows': d['expected_rows'],
                           'run_disposition': d['provenance'], 'rows': normalized}
    return result


def selected_gain_views(views, specs, sources, pins, reducer):
    """Disclosed source-identity views; the original all-source result is untouched."""
    unique(views, 'name', P3_VIEW_NAMES)
    require([v['name'] for v in views] == list(P3_VIEW_NAMES), 'fixed P3 view order differs')
    by_source = unique(specs, 'source_id')
    result = []
    for view in views:
        basis = view['selector_basis']
        require(isinstance(basis, dict) and basis['file_id'] in pins and basis['manifest_file_id'] in pins,
                'P3 source-selection basis must reference authenticated files')
        require(isinstance(basis['source_commit'], str) and len(basis['source_commit']) == 40
                and all(c in '0123456789abcdef' for c in basis['source_commit']), 'P3 basis source commit required')
        require(all(isinstance(basis[k], list) and basis[k] and all(type(n) is int and n > 0 for n in basis[k])
                    for k in ('source_lines', 'excerpt_lines')) and bool(basis['scope']), 'P3 source line/scope basis required')
        require(isinstance(view['exposure'], str) and bool(view['exposure']), 'P3 prior-exposure disclosure required')
        expected = [{'role': role, 'source_id': 'qwen7b' if view['name'] == P3_VIEW_NAMES[1] and role == 'qwen_reference' else source,
                     'family': family, 'model': model} for role, source, family, model in P3_ROLES]
        require(exact(view['entries'], expected), 'exact fixed P3 source/role/family/model selection differs')
        selected = []
        for row in view['entries']:
            source_id = row['source_id']
            require(source_id in by_source and source_id in sources, 'selected P3 source missing')
            spec = by_source[source_id]
            require(spec['family'] == row['family'] and spec['model'] == row['model'], 'selected P3 source identity differs')
            require(sources[source_id]['available'] is True, 'selected P3 source unavailable; no fallback')
            selected.append(spec)  # Preserve the original label and identity without renaming.
        result.append({'name': view['name'], 'selector_basis': basis, 'exposure': view['exposure'],
                       'selected_sources': view['entries'], 'selected_specs': selected,
                       'result': reducer.gain_summary(selected, sources),
                       'scope': 'Same unchanged all-cell medians, bins, thresholds and P3 operators on an explicitly named source subset. Missing bins remain undefined. All15 source rows/bins and default ambiguous-reference P3 remain in the parent gain output.'})
    return result


def sensitivity(scale):
    result = {}
    for row in scale['models']:
        ident = row['identity']
        key = (ident['family'], ident['model'])
        current = row['literal_summary']
        if current is None:
            result[key] = {'missing': False, 'source_refusal': row['source_refusals']}
        else:
            current = dict(current)
            value = current.get('chaos_decades')
            if isinstance(value, dict) and set(value) == {'nonfinite'}:
                current['chaos_decades'] = float(value['nonfinite'])
            result[key] = current
    return result


def reduce_all(manifest, pins, m):
    roles = manifest['roles']
    require(set(roles) == set(COMPONENTS), 'all six components must be explicitly inventoried')
    result = {}
    result['scale'] = m['reducers'].scale_summary(scale_inputs(roles['scale'], pins, m['reducers']))
    specs = manifest['gain_specs']
    require(len(specs) == 15 and manifest['gain_order_scope'], 'fifteen explicit gain display/source associations required')
    gain_sources = gain_inputs(roles['gain'], specs, pins)
    result['gain'] = m['reducers'].gain_summary(specs, gain_sources)
    result['gain']['source_selected_P3_views'] = selected_gain_views(manifest['p3_source_selections'], specs, gain_sources, pins, m['reducers'])
    stages = {}
    wanted = {f'{model}/{i}/{stage}' for model in m['prefix'].MODELS for i in range(10) for stage in ('coarse', 'fine')}
    unique(roles['ps'], 'id', wanted)
    for d in roles['ps']:
        model, index, stage = d['id'].split('/')
        require(d['expected_rows'] == (48 if stage == 'coarse' else 24), 'fixed PS stage coverage differs')
        entry = stages.setdefault(model, {}).setdefault(int(index), {'provenance': {}})
        entry[stage] = saved_rows(d, pins)
        entry['provenance'][stage] = d['provenance']
    result['ps'] = m['prefix'].reduce_ps(stages)
    unique(roles['cp'], 'id', {s[0] for s in m['start'].FROZEN})
    docs = {}
    for d in roles['cp']:
        doc = document(d, pins)
        if doc is not None:
            require(exact(doc['log10T'], d['expected_axes']['log10T']) and exact(doc['gain_log10T'], d['expected_axes']['gain_log10T']), 'CP axis differs')
            require([r['prefix_index'] for r in doc['prefixes']] == list(range(10)), 'CP fixed prefix order differs')
        docs[d['id']] = doc
    result['cp'] = m['start'].reduce_cp(docs, sensitivity(result['scale']))
    unique(roles['plateau'], 'id', m['plateau'].EXPECTED)
    plateau_docs, reductions = {}, []
    for d in roles['plateau']:
        rows = saved_rows(d, pins)
        require(d['expected_rows'] == m['plateau'].EXPECTED[d['id']][0], 'fixed plateau inventory differs')
        selection = None
        if d['id'] in ('PTa', 'PTb'):
            selection = read_json(pins[d['selection_file_id']]['path'])['meta']
            require(len(selection) == d['expected_rows'], 'original plateau selection coverage differs')
        doc = None if rows is None else {'cells': rows, 'args': d['expected_args']}
        plateau_docs[d['id']] = rows
        reductions.append(m['plateau'].reduce_plateau(d['id'], doc, selection, d['provenance']))
    left, right = plateau_docs['PTc'], plateau_docs['B7b_history']
    result['plateau'] = {'runs': reductions, 'PTc_vs_B7b_same_offset':
        m['plateau'].compare_exact_coordinates(left, right, 1e-5) if left is not None and right is not None else {'status': 'unavailable'}}
    unique(roles['depth'], 'layer', (20, 35, 50, 65))
    depths = []
    for d in roles['depth']:
        a = arrays(d, pins)
        summary = read_json(pins[d['summary_file_id']]['path']) if a is not None else None
        if summary is not None:
            require(exact(summary['args'], d['expected_args']), 'depth source argument association differs')
        depths.append(m['depth'].reduce_depth(d['layer'], a, d['expected_args'], d['provenance'], summary))
    result['depth'] = {'layers': depths, **m['depth'].compare_layers(depths)}
    return result


def compare_saved(result, manifest, pins, reducer):
    ledger = []
    for d in manifest['published_comparisons']:
        require(d['component'] in COMPONENTS and d['source_association'] and d['rounding_scope'], 'published comparison requires explicit source/precision association')
        if not d['available']:
            ledger.append({**d, 'status': 'unavailable'})
            continue
        original = at(read_json(pins[d['file_id']]['path']), d['producer_pointer'])
        require(d['own_pointer'] == '' or d['own_pointer'].startswith('/'), 'own JSON pointer required')
        try:
            own = at(reducer.json_ready(result[d['component']]), d['own_pointer'])
        except (KeyError, IndexError, TypeError) as e:
            # An admitted input can legitimately produce a source refusal or
            # undefined score. Keep every other component and all saved leaves.
            ledger.append({**d, 'status': 'own_scientific_node_unavailable',
                           'reason': type(e).__name__+': '+str(e),
                           'all_leaf_comparisons': reducer.compare_leaves(reducer._MISSING, original),
                           'scope': 'Full component disposition is retained in its output. Prefer whole stable projection/score nodes in the frozen contract.'})
            continue
        ledger.append({**d, 'status': 'compared', 'all_leaf_comparisons': reducer.compare_leaves(own, original)})
    return {'rows': ledger, 'coverage_scope': manifest['published_comparison_scope'],
            'scope': 'Exact saved values/schema comparisons, every difference retained. No new agreement tolerance or inferred unpublished values.'}


def write_json(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, separators=(',', ':'), allow_nan=False)
        out.write('\n')


def execute(card, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # No failure write can touch an existing directory.
    started = time.monotonic()
    try:
        require(card['status'] == 'reviewed_for_execution', 'prepared cards cannot run')
        require(card['schema'] == 'task25-six-component-card-v1', 'wrong card schema')
        require(set(card['source_sha256']) == {'run', *COMPONENT_HASHES}, 'exact scientific source inventory required')
        for name, expected in card['source_sha256'].items():
            require(pin(HERE/(name+'.py'))['sha256'] == expected, 'runtime source differs: '+name)
        check_pin(card['inputs'])
        manifest = read_json(card['inputs']['path'])
        require(manifest['schema'] == 'task25-six-component-inputs-v1', 'wrong input schema')
        pin_rows = manifest['pins']
        pins = {k: {field: v[field] for field in ('path', 'bytes', 'sha256')} for k, v in unique(pin_rows, 'id').items()}
        require(len({p['path'] for p in pins.values()}) == len(pins), 'duplicate pinned path; use one ID')
        for p in pins.values():
            check_pin(p)
        for assertion in manifest['authority_assertions']:
            require(exact(at(read_json(pins[assertion['file_id']]['path']), assertion['pointer']), assertion['expected']), 'literal original authority assertion differs')
        require(bool(manifest['authority_basis']) and bool(manifest['limitations']), 'finite authority basis/limitations required')
        m = modules()
        result = reduce_all(manifest, pins, m)
        comparisons = compare_saved(result, manifest, pins, m['reducers'])
        for name in COMPONENTS:
            write_json(out/(name+'.json'), m['reducers'].json_ready(result[name]))
        write_json(out/'comparisons.json', m['reducers'].json_ready(comparisons))
        write_json(out/'input-manifest.json', manifest)
        observed = {'seconds_through_component_serialization_before_final_rehash': time.monotonic()-started,
                    'ru_maxrss_KiB_before_final_rehash_report': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'scope': 'Sampled worker timing/RSS, not full lifetime or independently measured full peak. Supervisor supplies whole-worker wall.'}
        for p in pins.values():
            check_pin(p)
        check_pin(card['inputs'])
        outputs = [pin(out/(name+'.json')) for name in (*COMPONENTS, 'comparisons', 'input-manifest')]
        report = {'schema': 'task25-independent-six-component-v1', 'computational_status': 'complete',
                  'source_sha256': card['source_sha256'], 'inputs': card['inputs'], 'input_pin_count': len(pins),
                  'components': list(COMPONENTS), 'output_files': outputs, 'runtime': observed,
                  'authority_basis': manifest['authority_basis'], 'limitations': manifest['limitations'],
                  'scientific_scope': 'Computational completion is independent of scientific gates, missing inputs, undefined results, source refusals and discrepancies; all remain in component outputs.'}
        write_json(out/'report.json', report)
        return report
    except BaseException as e:
        write_json(out/'failure.json', {'error_type': type(e).__name__, 'message': str(e), 'elapsed': time.monotonic()-started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    execute(read_json(args.card), args.out)


if __name__ == '__main__':
    main()
