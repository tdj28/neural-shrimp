"""One combined manufactured-only control/serialization proposal; not admitted.

No original data paths or producer imports. All synthetic inputs and failure
artifacts remain inside the fresh output directory. No implicit execution.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent


def sibling(name):
    spec = importlib.util.spec_from_file_location('task25_manufactured_'+name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refused(call):
    try:
        call()
    except (ValueError, KeyError, FileExistsError):
        return
    raise AssertionError('expected refusal')


def build_fixture(root, a, m, ps_control, pd_control):
    import numpy as np
    root.mkdir(exist_ok=False)
    manifest = {'schema': 'task25-six-component-inputs-v1', 'pins': [],
                'roles': {k: [] for k in a.COMPONENTS}, 'gain_specs': [],
                'gain_order_scope': 'Manufactured fixed folder order; no historical CLI attestation.',
                'authority_basis': 'Manufactured only; no original authority claim.',
                'limitations': ['All values and coordinates are manufactured.', 'Layer50 present here is synthetic, not original evidence.'],
                'authority_assertions': [], 'published_comparisons': [], 'p3_source_selections': [],
                'published_comparison_scope': 'Manufactured source-refusal/null/nonzero comparison coverage; actual summary pins remain unbound.'}

    def put(name, value, npz=False):
        path = root/(name+('.npz' if npz else '.json'))
        if npz:
            np.savez(path, **value)
        else:
            path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=True)+'\n')
        manifest['pins'].append({'id': name, **a.pin(path)})
        return name

    def desc(identity, file_id, args=None):
        return {'id': identity, 'file_id': file_id, 'available': True, 'reason': None,
                'provenance': {'manufactured': True, 'original_process_claim': False},
                'expected_args': args, 'metadata_assertions': []}

    def array_desc(identity, value, args=None):
        d = desc(identity, put(identity, value, True), args)
        d['members'] = {k: {'shape': list(v.shape), 'dtype': v.dtype.str} for k, v in value.items()}
        d['expected_axes'] = {k: value[k].tolist() for k in ('log10_temperature', 'alpha')}
        return d

    def cell_desc(identity, rows, args):
        d = desc(identity, put(identity, {'args': args, 'cells': rows}), args)
        d.update(expected_rows=len(rows), expected_coordinates={
            k: [r[k] for r in rows] for k in ('log10_temperature', 'alpha') if all(k in r for r in rows)})
        return d

    shape = a.read_json(HERE/'manufactured-shapes.json')
    for ordinal, spec in enumerate(shape['scale']):
        n = spec['rows']
        t = np.linspace(-1, .9, n)
        data = {'log10_temperature': t, 'alpha': np.array([0.]),
                'lyapunov': np.full((n, 1), .02), 'period': np.zeros((n, 1), dtype=np.int64),
                'entropy': np.linspace(0, 12, n).reshape(n, 1)}
        if ordinal == 0:
            data['lyapunov'][0, 0] = float('nan')  # Available source; scientific refusal must survive full adapter.
        d = array_desc('scale-'+str(ordinal), data)
        d.update(id=spec['id'], expected_rows=n, alpha=0., max_period=32, completion={'kind': 'all_saved_rows'})
        manifest['roles']['scale'].append(d)
    for ordinal, spec in enumerate(shape['gain']):
        n = spec['rows']
        entropy_high = 5.9 if spec['id'] == 'qwen7b' else 12.  # Historical narrow view must keep its empty6–10 bin.
        rows = [{'log10_temperature': float(t), 'entropy_mean': float(h), 'log_g_net': 1.+ordinal*.01,
                 'log_g_read': -.9, 'lyapunov': .1+ordinal*.01, 'period': 0, 'boundary_term': 0.}
                for t, h in zip(np.linspace(-.3, .9, n), np.linspace(0, entropy_high, n))]
        d = cell_desc('gain-'+str(ordinal), rows, {'synthetic': True, 'model': spec['model']})
        d['id'] = spec['id']
        d['completion'] = {'kind': 'all_saved_rows'}
        manifest['roles']['gain'].append(d)
        # This is an explicit synthetic display mapping, not historical CLI evidence.
        family = 'Qwen2.5' if spec['model'].startswith('qwen') else 'OLMo-2' if spec['model'].startswith('olmo') else 'Llama-3.1' if '3.1' in spec['model'] else 'Llama-3.3'
        model = spec['model'].split('-')[-1].replace('qwen', '').replace('b', 'B')
        manifest['gain_specs'].append({'source_id': d['id'], 'label': spec['id'], 'family': family, 'model': model})
    for mi, model in enumerate(m['prefix'].MODELS):
        for index in range(10):
            coarse = [ps_control.cell(float(t), float(4*t+2), le=(.01, .012)) for t in np.linspace(-.4, .9, 48)]
            bounds = m['prefix'].transition(coarse)['log10T_bounds']
            fine = [ps_control.cell(float(t), float(4*t+2), period=4 if j%4 == 0 else 0) for j, t in enumerate(np.linspace(*bounds, 24))]
            for stage, rows in (('coarse', coarse), ('fine', fine)):
                d = cell_desc(f'ps-{model}-{index}-{stage}', rows, {'synthetic': True, 'model': model, 'prefix_index': index})
                d['id'] = f'{model}/{index}/{stage}'
                manifest['roles']['ps'].append(d)
    for ordinal, spec in enumerate(m['start'].FROZEN):
        doc = ps_control.cp_document(ordinal*.02)
        d = desc(spec[0], put('cp-'+str(ordinal), doc), doc['args'])
        d['expected_axes'] = {k: doc[k] for k in ('log10T', 'gain_log10T')}
        manifest['roles']['cp'].append(d)
    for role, (n, eps) in m['plateau'].EXPECTED.items():
        rows = [pd_control.cell(role, i) for i in range(n)]
        d = cell_desc('plateau-'+role, rows, {'synthetic': True, 'eps': eps})
        d['id'] = role
        if role in ('PTa', 'PTb'):
            selected = [{'log10T': r['log10_temperature'], 'alpha': r['alpha'], 'map_label': 'synthetic selected',
                         'map_exponent': .02, 'map_period': 0} for r in rows]
            d['selection_file_id'] = put('selection-'+role, {'meta': selected})
        manifest['roles']['plateau'].append(d)
    for layer in (20, 35, 50, 65):
        data, args = pd_control.depth_fixture(layer)
        d = array_desc('depth-'+str(layer), data, args)
        d['layer'] = layer
        # Available native-shaped data may yield a scientific mismatch; retain it.
        summary = {'args': args, 'cells': data['lyapunov'].size, 'lyapunov_positive': -1,
                   'period_counts': {}, 'lyapunov_range': [-999, 999]}
        d['summary_file_id'] = put('depth-summary-'+str(layer), summary)
        manifest['roles']['depth'].append(d)
    authority = put('manufactured-authority', {'closed': True, 'scope': 'manufactured only'})
    manifest['authority_assertions'] = [{'file_id': authority, 'pointer': '/closed', 'expected': True}]
    saved = put('manufactured-comparison', {'literal_score': {'PS1': 'different on purpose'}, 'value': .123})
    for name in a.P3_VIEW_NAMES:
        entries = [{'role': role, 'source_id': 'qwen7b' if name == 'historical_narrow_qwen7B' and role == 'qwen_reference' else source,
                    'family': family, 'model': model} for role, source, family, model in a.P3_ROLES]
        manifest['p3_source_selections'].append({'name': name, 'entries': entries,
            'selector_basis': {'file_id': saved, 'manifest_file_id': authority, 'source_commit': '0'*40,
                               'source_lines': [1], 'excerpt_lines': [1], 'scope': 'Manufactured selector basis only; no original source claim.'},
            'exposure': 'Manufactured values. Actual source selections are disclosed after prior manuscript/log/producer-summary exposure, not fresh prespecification.'})
    manifest['published_comparisons'] = [
        {'component': 'ps', 'available': True, 'file_id': saved, 'producer_pointer': '/literal_score',
         'own_pointer': '/literal_score', 'source_association': 'manufactured', 'rounding_scope': 'exact saved JSON'},
        {'component': 'cp', 'available': True, 'file_id': saved, 'producer_pointer': '/value',
         'own_pointer': '/score/source_refusal_dependent_field', 'source_association': 'manufactured unavailable scientific node', 'rounding_scope': 'exact saved JSON'},
        {'component': 'scale', 'available': True, 'file_id': saved, 'producer_pointer': '/value',
         'own_pointer': '/producer_projection/rows/0/chaos_decades', 'source_association': 'manufactured available source with retained nonfinite exponent', 'rounding_scope': 'exact saved JSON'}]
    path = root/'inputs.json'
    a.write_json(path, manifest)
    card = {'schema': 'task25-six-component-card-v1', 'status': 'reviewed_for_execution',
            'inputs': a.pin(path), 'source_sha256': {k: a.pin(HERE/(k+'.py'))['sha256'] for k in ('run', *a.COMPONENT_HASHES)}}
    return manifest, card


def adapter_controls(root, a, m, manifest, card):
    pins = {r['id']: {k: r[k] for k in ('path', 'bytes', 'sha256')} for r in manifest['pins']}
    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append({'group': name, 'passed': True})
        except Exception as e:
            checks.append({'group': name, 'passed': False, 'type': type(e).__name__, 'message': str(e)})

    def pin_and_identity_refusals():
        wrong = dict(next(iter(pins.values())), sha256='0'*64)
        refused(lambda: a.check_pin(wrong))
        refused(lambda: a.unique([{'id': 'x'}, {'id': 'x'}], 'id'))
        refused(lambda: a.unique([{'id': 'x'}], 'id', ['x', 'y']))
        assert not a.exact(True, 1)
    check('adapter_pin_identity_boolean_refusals', pin_and_identity_refusals)

    def cell_argument_and_coordinate_refusals():
        d = copy.deepcopy(manifest['roles']['ps'][0])
        d['expected_args']['prefix_index'] = 99
        refused(lambda: a.saved_rows(d, pins))
        d = copy.deepcopy(manifest['roles']['ps'][0])
        d['expected_coordinates']['log10_temperature'].reverse()
        refused(lambda: a.saved_rows(d, pins))
    check('adapter_exact_json_associations', cell_argument_and_coordinate_refusals)

    def array_and_completion_refusals():
        import numpy as np
        d = copy.deepcopy(manifest['roles']['scale'][0])
        d['members']['lyapunov']['shape'] = [1]
        refused(lambda: a.arrays(d, pins))
        d = copy.deepcopy(manifest['roles']['scale'][0])
        d['expected_axes']['alpha'] = [1.]
        refused(lambda: a.arrays(d, pins))
        assert a.completed({'completion': {'kind': 'recorded_prefix', 'count': 2}}, 3, {'done': np.array(2)}) == [True, True, False]
        refused(lambda: a.completed({'completion': {'kind': 'all_saved_rows'}}, 3, {'done': np.array(2)}))
    check('adapter_npz_axes_shape_and_completion', array_and_completion_refusals)

    def sensitivity_and_missing_baseline():
        rows = [{'identity': {'family': 'F', 'model': 'M'}, 'literal_summary': None, 'source_refusals': ['retained nonfinite']}]
        current = a.sensitivity({'models': rows})
        assert current[('F', 'M')]['source_refusal'] == ['retained nonfinite']
        assert all(r['status'] == 'unavailable' for r in m['depth'].compare_layers([])['comparisons'])
    check('adapter_source_refusal_and_unbound_baseline', sensitivity_and_missing_baseline)

    def scientific_comparison_dispositions():
        # Available input with a genuine component source refusal, not a missing role.
        malformed = {'prefixes': [{'prefix_index': 0}]}
        cp = m['start'].reduce_cp({'qwen7b': malformed})
        assert cp['source_refusals'] == ['qwen7b'] and cp['score'] is None
        d = copy.deepcopy(manifest['published_comparisons'][1])
        d['own_pointer'] = '/score/CP1/rho'
        ledger = a.compare_saved({'cp': cp}, {'published_comparisons': [d], 'published_comparison_scope': 'manufactured'}, pins, m['reducers'])
        assert ledger['rows'][0]['status'] == 'own_scientific_node_unavailable'
        assert ledger['rows'][0]['all_leaf_comparisons'][0]['disposition'] == 'missing_own_leaf'
        diff = m['reducers'].compare_leaves(.5, .4)[0]
        assert diff['disposition'] == 'different' and diff['signed_difference'] != 0
    check('adapter_available_scientific_refusal_comparison', scientific_comparison_dispositions)

    def source_selected_p3_views():
        specs = manifest['gain_specs']
        sources = a.gain_inputs(manifest['roles']['gain'], specs, pins)
        all_sources = m['reducers'].gain_summary(specs, sources)
        qwen_default = [r for r in all_sources['P3'] if r['family'] == 'Qwen2.5']
        assert len(all_sources['models']) == 15 and all(r['P3_verdict'] is None for r in qwen_default)
        views = a.selected_gain_views(manifest['p3_source_selections'], specs, sources, pins, m['reducers'])
        wide, narrow = [v['result'] for v in views]
        assert all(r['finite_defined'] for r in wide['P3'])
        gaps = [r for r in narrow['P3'] if r['family'] == 'Qwen2.5' and r['bin'] == '6-10']
        assert len(gaps) == 4 and all(r['P3_verdict'] is None and r['sides'][1]['cells'] == 0 for r in gaps)
        for view in views:
            for selected in view['result']['models']:
                original = next(r for r in all_sources['models'] if r['spec']['source_id'] == selected['spec']['source_id'])
                assert selected['spec'] == original['spec']
                # Member IDs include the view-local source ordinal; scalar bins must be unchanged.
                assert [{k: v for k, v in b.items() if k != 'member_ids'} for b in selected['bins']] == [
                    {k: v for k, v in b.items() if k != 'member_ids'} for b in original['bins']]
        for mutation in ('missing', 'duplicate', 'wrong_source', 'wrong_family'):
            bad = copy.deepcopy(manifest['p3_source_selections'])
            entries = bad[0]['entries']
            if mutation == 'missing': entries.pop()
            if mutation == 'duplicate': entries[-1] = dict(entries[0])
            if mutation == 'wrong_source': entries[0]['source_id'] = 'qwen7b'
            if mutation == 'wrong_family': entries[0]['family'] = 'OLMo-2'
            refused(lambda: a.selected_gain_views(bad, specs, sources, pins, m['reducers']))
        unavailable = copy.deepcopy(sources)
        unavailable['qwen7b-wide']['available'] = False
        refused(lambda: a.selected_gain_views(manifest['p3_source_selections'], specs, unavailable, pins, m['reducers']))
        duplicate_specs = copy.deepcopy(specs)
        duplicate_specs[-1]['source_id'] = duplicate_specs[0]['source_id']
        refused(lambda: a.selected_gain_views(manifest['p3_source_selections'], duplicate_specs, sources, pins, m['reducers']))
        wrong_specs = copy.deepcopy(specs)
        next(s for s in wrong_specs if s['source_id'] == 'qwen7b-wide')['family'] = 'OLMo-2'
        refused(lambda: a.selected_gain_views(manifest['p3_source_selections'], wrong_specs, sources, pins, m['reducers']))
    check('adapter_explicit_P3_source_views_and_refusals', source_selected_p3_views)

    def exclusive_ownership_and_prepared_refusal():
        existing = root/'existing-output'
        existing.mkdir()
        (existing/'sentinel').write_text('unchanged')
        refused(lambda: a.execute(card, existing))
        assert sorted(x.name for x in existing.iterdir()) == ['sentinel']
        prepared = dict(card, status='prepared_not_admitted')
        fresh = root/'prepared-refusal'
        refused(lambda: a.execute(prepared, fresh))
        assert sorted(x.name for x in fresh.iterdir()) == ['failure.json']
    check('adapter_exclusive_output_and_prepared_refusal', exclusive_ownership_and_prepared_refusal)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        a = sibling('run')
        m = a.modules()
        # The unchanged scale/gain controls import this exact own module name.
        sys.modules['reducers'] = m['reducers']
        sg, ps, pd = (sibling(name) for name in ('scale_gain_controls', 'prefix_start_controls', 'plateau_depth_controls'))
        outcomes = sg.run_controls()['groups']
        for suite, names in ((ps, ('check_labels', 'check_selection', 'check_ps_coverage', 'check_ps_frozen_rules', 'check_ranks',
                                  'check_cp_profiles', 'check_cp_saturation_and_nonfinite', 'check_cp_membership', 'check_frozen_vs_sensitivity')),
                             (pd, ('check_pt_confirmation', 'check_boundary_and_negative', 'check_qf_probe', 'check_qf_missing_ladder',
                                   'check_coverage_and_pairing', 'check_depth_semantics', 'check_depth_partial_shape', 'check_depth_comparison'))):
            for name in names:
                try:
                    getattr(suite, name)()
                    outcomes.append({'group': name, 'passed': True})
                except Exception as e:
                    outcomes.append({'group': name, 'passed': False, 'type': type(e).__name__, 'message': str(e)})
        manifest, card = build_fixture(args.out/'manufactured-inputs', a, m, ps, pd)
        outcomes += adapter_controls(args.out, a, m, manifest, card)
        a.write_json(args.out/'controls.json', outcomes)
        a.require(all(r['passed'] for r in outcomes), 'one or more manufactured controls failed; preserved individually')
        measured = time.monotonic()
        report = a.execute(card, args.out/'manufactured-run')
        elapsed = time.monotonic()-measured
        assert report['components'] == list(a.COMPONENTS)
        assert a.read_json(args.out/'manufactured-run/scale.json')['models'][0]['source_refusals']
        assert a.read_json(args.out/'manufactured-run/comparisons.json')['rows'][2]['status'] == 'own_scientific_node_unavailable'
        assert len(a.read_json(args.out/'manufactured-run/gain.json')['source_selected_P3_views']) == 2
        outputs = [a.pin(p) for p in sorted((args.out/'manufactured-run').iterdir())]
        result = {'schema': 'task25-six-component-manufactured-v1', 'passed': len(outcomes), 'failed': 0,
                  'controls': outcomes, 'shape': a.read_json(HERE/'manufactured-shapes.json'),
                  'input_bytes': sum(p['bytes'] for p in manifest['pins']),
                  'full_adapter_seconds_through_final_report': elapsed, 'total_seconds_before_manufactured_report': time.monotonic()-started,
                  'ru_maxrss_KiB_before_manufactured_report': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'output_files': outputs, 'output_bytes': sum(p['bytes'] for p in outputs),
                  'scope': 'Manufactured counts, values and local serialization/hash/decode only. Actual original compressed bytes/string lengths, scientific dispositions, receipt authority and full-worker peak remain unmeasured. No actual baseline50 authority is supplied by this fixture.'}
        a.write_json(args.out/'report.json', result)
    except BaseException as e:
        (args.out/'failure.json').write_text(json.dumps({'type': type(e).__name__, 'message': str(e), 'elapsed': time.monotonic()-started})+'\n')
        raise


if __name__ == '__main__':
    main()
