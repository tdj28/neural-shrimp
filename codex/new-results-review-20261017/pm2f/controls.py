"""Manufactured-only PM2F controls; prepared source, not yet executed.

Run once within the eventually reviewed audit reservation. No target files.
"""
import importlib.util
import json
import resource
import time
from pathlib import Path


def load_core():
    p = Path(__file__).with_name('core.py')
    s = importlib.util.spec_from_file_location('pm2f_independent_core', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def args(model='manufactured'):
    return {'model': model, 'twins': 2, 'eps0': 1e-10, 'think_budget': 1536,
            'answer_budget': 256, 'group': 8, 'limit': 128}


def row(i=0):
    return {'index': i, 'gold': '7', 'answer': '7', 'correct': True, 'switch_step': 100,
            'twins': [
                {'answer': '7', 'flip': False, 'switch_step': 100, 'same_switch': True,
                 'growth_rate': .02, 'log10_max_think': -8., 'log10_end_think': -11.},
                {'answer': '8', 'flip': True, 'switch_step': 90, 'same_switch': False,
                 'growth_rate': -.02, 'log10_max_think': -10., 'log10_end_think': -10.}]}


def execute():
    c = load_core()
    results = []
    def group(name, fn):
        try:
            fn(); results.append({'name': name, 'passed': True})
        except Exception as e:
            results.append({'name': name, 'passed': False, 'error': repr(e)})
    def slope_vs_endpoint():
        z = c.reduce_rows([row()], args())
        assert z['counts']['positive_fit_but_endpoint_below_eps0'] == 1
        assert z['counts']['endpoint_vs_nominal_eps0'] == {'below': 1, 'equal': 1}
        assert z['counts']['maximum_vs_nominal_eps0'] == {'above': 1, 'equal': 1}
        assert z['counts']['problem_any_flip'] == 1
        assert z['counts']['answer_flip_fraction_all_twins'] == .5
    group('positive_fitted_slope_is_not_endpoint_growth', slope_vs_endpoint)
    def strict_edges():
        r = row();r['twins'][0]['log10_end_think'] = -10. + 1e-12
        r['twins'][1]['log10_end_think'] = -10. - 1e-12
        z = c.reduce_rows([r], args())
        assert z['counts']['endpoint_vs_nominal_eps0'] == {'above': 1, 'below': 1}
        assert c.relation(-10., -10.) == 'equal'
    group('strict_nominal_threshold_and_equality', strict_edges)
    def undefined():
        r = row()
        for t in r['twins']:
            t.update(growth_rate=None, log10_max_think=None, log10_end_think=None)
        z = c.reduce_rows([r], args())
        assert z['counts']['endpoint_vs_nominal_eps0'] == {'undefined': 2}
        assert z['finite_scalar_summaries']['growth_rate']['median'] is None
        assert z['counts']['answer_flips_reconstructed'] == 1
        empty = c.reduce_rows([], args())
        assert empty['counts']['answer_flip_fraction_all_twins'] is None
    group('undefined_distance_keeps_answer_denominator', undefined)
    def nonfinite():
        r = row();r['twins'][0]['growth_rate'] = float('nan')
        r['twins'][0]['log10_end_think'] = float('inf')
        r['twins'][1]['growth_rate'] = 'bad'
        z = c.reduce_rows([r], args())
        assert z['finite_scalar_summaries']['growth_rate']['finite_count'] == 0
        assert z['counts']['diagnostic_twin_rows'] == 2
        encoded = json.dumps(c.json_safe(z), allow_nan=False)
        assert 'positive_infinity' in encoded and 'wrong_type' in encoded
    group('nonfinite_and_wrong_scalar_type_retained', nonfinite)
    def flags():
        r = row();r['correct'] = False;r['twins'][0]['flip'] = True
        r['twins'][0]['same_switch'] = False
        z = c.reduce_rows([r], args())
        assert z['counts']['base_correct_mismatches'] == 1
        assert z['counts']['flip_flag_mismatches'] == 1
        assert z['counts']['same_switch_flag_mismatches'] == 1
        assert z['counts']['answer_flips_stored'] == 2
        assert z['counts']['answer_flips_reconstructed'] == 1
    group('stored_boolean_disagreements_not_overwritten', flags)
    def duplicate():
        z = c.reduce_rows([row(4), row(4)], args())
        assert z['coverage']['records'] == 2 and z['coverage']['unique_problem_indices'] == 1
        assert z['coverage']['duplicate_indices'] == {'4': [1, 2]}
        assert z['rows'][1]['duplicate_of_lines'] == [1]
    group('duplicate_source_rows_retained', duplicate)
    def windows_floor():
        r = row();r['switch_step'] = 0
        r['twins'][0]['log10_end_think'] = -300.
        z = c.reduce_rows([r], args())
        assert 'scalar_present_for_empty_base_window' in z['rows'][0]['twins'][0]['diagnostics']
        assert z['counts']['endpoint_writer_floor'] == 1
        r = row();r['twins'][0]['log10_end_think'] = -7.
        z = c.reduce_rows([r], args())
        assert 'endpoint_exceeds_saved_maximum' in z['rows'][0]['twins'][0]['diagnostics']
        assert z['counts']['earlier_twin_switch'] == 1
    group('base_window_switch_and_floor_diagnostics', windows_floor)
    def schema():
        cases=[]
        r=row();del r['twins'][0]['growth_rate'];cases.append(r)
        r=row();r['twins'].pop();cases.append(r)
        r=row();r['switch_step']=1537;cases.append(r)
        r=row();r['index']=True;cases.append(r)
        for r in cases:
            try: c.reduce_rows([r], args())
            except ValueError: pass
            else: raise AssertionError('structural failure accepted')
    group('missing_fields_count_and_identity_refusals', schema)
    def comparison():
        z=c.compare_leaves({'x': 1., 'extra': 2}, {'x': 1.+1e-12, 'missing': 3})
        assert [v['status'] for v in z] == ['missing_published','missing_own','different']
        assert z[2]['signed_difference'] != 0
    group('all_leaf_missing_and_tiny_delta_retention', comparison)
    def projections():
        r=row()
        e=c.declared_publication_projection([r],args(),'endpoint')
        g=c.declared_publication_projection([r],args(),'fitted_slope')
        assert e['twins_grew']==0 and e['twins_shrank']==1
        assert g['twins_grew']==1 and g['twins_shrank']==1
        assert e['problems_any_flip']==1. and e['twin_flip_rate']==.5
        assert e['growth_rate_n']==2 and e['growth_rate_quartiles'][1]==0.
        assert e['different_switch']==1 and e['accuracy_base']==1.
        assert e['log10_end_think_median']==-10.5
    group('two_declared_projections_are_never_selected_by_agreement',projections)
    return {'groups': results, 'passed': all(r['passed'] for r in results), 'count': len(results)}


def manufactured_full_shape():
    c=load_core(); start=time.monotonic()
    out={m:c.reduce_rows([row(i) for i in range(n)], {**args(m), 'limit':n})
         for m,n in [('r1qwen7b',128),('r1llama8b',128),('qwq32b',32)]}
    payload=json.dumps(c.json_safe(out),allow_nan=False,sort_keys=True).encode()
    return {'model_shapes':[128,128,32], 'twins_per_problem':2,
            'seconds_through_json_serialization':time.monotonic()-start,
            'serialized_bytes':len(payload),
            'rss_KiB_observed_then':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'Manufactured allocation/serialization only; not a worst-case output proof.'}


if __name__=='__main__':
    result={'controls':execute(),'manufactured':manufactured_full_shape()}
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['controls']['passed'] else 1)
