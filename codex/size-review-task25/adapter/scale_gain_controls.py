"""Manufactured-only controls, prepared and NOT executed for this release.

No target files, producer imports, external calls, or implicit main entry point.
An independently reviewed bounded harness may later invoke run_controls().
"""
import copy
import math
import reducers as r


def scale_row(t, le=.01, period=0, entropy=2., completed=True):
    return dict(log10_temperature=t, lyapunov=le, period=period, entropy=entropy, completed=completed)


def gain_row(t, entropy, net=2., read=-1., period=0, le=.01, completed=True):
    return dict(log10_temperature=t, entropy_mean=entropy, log_g_net=net,
                log_g_read=read, period=period, lyapunov=le, boundary_term=0., completed=completed)


def source(key, rows, available=True, disposition='manufactured_completed'):
    return dict(id=key, available=available, expected_rows=len(rows), rows=rows,
                run_disposition=disposition, alpha=0., max_period=64)


def spec(key='A', family='fixture', model='small', params=1., extras=()):
    return dict(family=family, model=model, params_b=params,
                primary_sources=[key], supplemental_sources=list(extras))


def one(rows, extras=None):
    ss = {'A': source('A', rows)}
    ss.update(extras or {})
    return r._one_scale(spec(extras=tuple(extras or {})), ss)


def duplicate_chain_and_scalar_differences():
    rows = [scale_row(0., le=.01), scale_row(.75e-9, le=.02),
            scale_row(1.5e-9, le=.03, entropy=4.), scale_row(.1)]
    a = one(rows)
    j = a['primary_join']
    assert [x['source_row'] for x in j['selected']] == [0, 3]
    assert len(j['duplicates']) == 2
    assert j['duplicates'][1]['previous_id'] == j['rows'][1]['id']
    assert j['duplicates'][1]['retained_chain_id'] == j['rows'][0]['id']
    assert j['duplicates'][1]['signed_anchor_differences']['log10_temperature'] > 1e-9
    assert j['duplicates'][1]['signed_previous_differences']['entropy'] == 2.
    assert a['literal_summary']['temperatures_all'] == 2


def stable_equal_endpoint_and_conflicting_labels():
    ss = {'A': source('A', [scale_row(0., le=.01), scale_row(.1)]),
          'B': source('B', [scale_row(0., le=.02, entropy=9.), scale_row(.2)])}
    a = r._one_scale(spec(extras=['B']), ss)
    j = a['full_join']
    assert j['selected'][0]['source_id'] == 'A'
    assert j['duplicates'][0]['exact_coordinate_equal'] is True
    assert j['duplicates'][0]['signed_previous_differences']['entropy'] == 7.
    ss['B']['rows'][0]['period'] = 2
    b = r._one_scale(spec(extras=['B']), ss)
    assert b['literal_summary'] is None
    assert 'adjacent_duplicate_chaos_disagreement' in b['source_refusals']
    assert len(b['full_join']['rows']) == 4


def classification_priority_and_invalidity():
    a = one([scale_row(-1., le=.001), scale_row(-.8, period=-1),
             scale_row(-.6, period=.5), scale_row(-.4, period=65),
             scale_row(-.2, entropy=float('inf')), scale_row(0., completed=False)])
    rows = a['primary_join']['rows']
    assert rows[0]['literal_unresolved'] and not rows[0]['literal_chaotic']
    assert rows[1]['literal_chaotic'] and not rows[1]['finite_qualified']
    assert rows[2]['literal_periodic'] and not rows[2]['finite_qualified']
    assert rows[3]['literal_periodic'] and not rows[3]['finite_qualified']
    assert rows[4]['literal_chaotic'] and not rows[4]['finite_qualified']
    assert rows[5]['literal_chaotic'] and not rows[5]['finite_qualified']
    assert a['qualification']['core_qualified_count'] == 1
    assert a['literal_summary']['chaotic_core'] == 3


def nonfinite_refusal_and_discarded_nonfinite():
    a = one([scale_row(0., le=float('nan')), scale_row(.1, le=0.)])
    assert a['literal_summary'] is None
    assert 'retained_nonfinite_exponent' in a['source_refusals']
    b = one([scale_row(0., le=0.), scale_row(0., le=float('nan')), scale_row(.1, le=0.)])
    assert b['literal_summary']['chaotic_all'] == 0
    assert not b['source_refusals']
    assert not b['primary_join']['rows'][1]['finite_qualified']
    assert math.isnan(b['primary_join']['duplicates'][0]['signed_previous_differences']['lyapunov'])


def missing_optional_and_historical_failed_source():
    absent = source('B', [], available=False, disposition='unavailable')
    a = one([scale_row(-1.), scale_row(0.)], {'B': absent})
    assert a['absent_optional_sources'] == ['B'] and not a['literal_summary']['missing']
    ss = {'A': source('A', [], available=False),
          'B': source('B', [scale_row(0., le=float('nan')), scale_row(.1)])}
    sp = spec(); sp['primary_sources'] = ['A', 'B']
    b = r._one_scale(sp, ss)
    assert b['literal_summary']['missing'] and not b['source_refusals']
    assert b['primary_join']['present_row_diagnostic_refusals'] == ['retained_nonfinite_exponent']
    ss = {'A': source('A', [scale_row(-1.), scale_row(0.)], disposition='tool_exit_1_after_save')}
    c = r._one_scale(spec(), ss)
    assert not c['literal_summary']['missing']
    assert c['primary_join']['source_status'][0]['run_disposition'] == 'tool_exit_1_after_save'


def upper_only_gate_and_nonuniform_weights():
    a = one([scale_row(-1.2), scale_row(-1.), scale_row(.3), scale_row(.3000000005), scale_row(.5)])
    assert a['literal_summary']['temperatures_core'] == 3  # near-endpoint row suppressed by literal join
    assert len(a['lower_axis_check']['below_minus_one_ids']) == 1
    # A separate nonduplicate above .3 remains in the source core tolerance.
    b = one([scale_row(-1.2), scale_row(0.), scale_row(.3000000005)])
    assert b['literal_summary']['temperatures_core'] == 3
    assert len(b['lower_axis_check']['core_membership_difference_ids']) == 2
    c = one([scale_row(-1., entropy=1.), scale_row(-.5, entropy=2.), scale_row(.5, entropy=10.)])
    assert [x['spacing_weight'] for x in c['full_join']['selected']] == [.5, .75, 1.]
    assert c['literal_summary']['chaos_decades'] == 2.25
    assert c['literal_summary']['intermediate_entropy_decades'] == .75
    assert c['literal_summary']['log10T_range'] == [-1., .5]


def p1_strict_p2_ties_and_missing():
    ss, specs = {}, []
    for family, order in r.FAMILIES:
        for j, model in enumerate(order):
            key = family+model
            specs.append(spec(key, family, model, j+1.))
            ss[key] = source(key, [scale_row(-1., le=0.), scale_row(0., le=0.)])
    a = r.scale_summary(ss, specs)
    assert a['producer_projection']['P1_holds'] is False
    assert a['producer_projection']['P2_holds'] is True
    assert a['descriptive_correlations']['breadth']['rho'] == {'nonfinite': 'nan'}
    ss['Qwen2.514B'] = source('Qwen2.514B', [], available=False)
    b = r.scale_summary(ss, specs)
    assert b['producer_projection']['P1_holds'] is None
    assert b['producer_projection']['P2_holds'] is None


def average_tie_ranks_and_undefined_core():
    assert r._ranks([3., 1., 1., 2.]).tolist() == [4., 1.5, 1.5, 3.]
    assert abs(r._spearman([1., 1., 2.], [1., 1., 2.])['rho'] - 1.) < 1e-15
    assert math.isnan(r._spearman([1., 1.], [1., 2.])['rho'])
    a = one([scale_row(.5), scale_row(.6)])
    assert math.isnan(a['literal_summary']['share_core'])
    assert a['literal_summary']['temperatures_core'] == 0


def all_cell_gain_bins_and_boundaries():
    hs = [-1., 0., 1., 3., 6., 10., 99., float('nan'), float('inf')]
    rows = [gain_row(i, h, net=float(i)) for i, h in enumerate(hs)]
    rows += [gain_row(10., 2., net=100., period=2)]
    ss = {'G': source('G', rows)}
    a = r.gain_summary([dict(label='G', family='fixture', model='small', source_id='G')], ss)
    b = a['models'][0]['bins']
    assert [x['cells'] for x in b] == [1, 2, 1, 1, 1]
    assert b[1]['median_log_g_net'] == 51. and b[1]['chaotic'] == 1
    assert len(a['models'][0]['outside_all_bin_ids']) == 4
    assert a['models'][0]['rows'][7]['qualification_failures'] == ['entropy_mean_nonfinite']


def gain_empty_nonfinite_period_and_missing():
    rows = [gain_row(0., 2., net=float('nan'), period=-1), gain_row(.1, 2., net=3.)]
    a = r.gain_summary([dict(label='G', family='fixture', model='small', source_id='G')], {'G': source('G', rows)})
    b = a['models'][0]['bins']
    assert b[0]['status'] == 'empty' and b[0]['median_log_g_net'] is None
    assert '0-1' not in a['producer_projection']['G']
    assert b[1]['median_log_g_net'] == {'nonfinite': 'nan'} and b[1]['net_finite_count'] == 1
    assert not a['models'][0]['rows'][0]['literal_chaotic']
    assert not a['models'][0]['rows'][0]['finite_qualified']
    missing = r.gain_summary([dict(label='M', family='fixture', model='small', source_id='M')],
                             {'M': source('M', [], available=False)})
    assert all(x['status'] == 'missing_source' for x in missing['models'][0]['bins'])


def p3_every_comparison_and_assumption():
    specs, ss = [], {}
    for family, order in r.FAMILIES:
        for j, model in enumerate(order[:3]):
            key = family+model
            specs.append(dict(label=key, family=family, model=model, source_id=key))
            ss[key] = source(key, [gain_row(k, h, net=3.-j, read=2.-j) for k, h in enumerate((2., 4., 8.))])
    a = r.gain_summary(specs, ss)
    assert len(a['P3']) == 24 and a['P3_holds'] is True
    assert all('not a literal OLMo-2 14B test' in x['roster_interpretation'] for x in a['P3'])
    assert all(all(s['cells'] == 1 for s in x['sides']) for x in a['P3'])
    ss['OLMo-27B']['rows'][1]['entropy_mean'] = 99.
    b = r.gain_summary(specs, ss)
    assert b['P3_holds'] is None
    assert sum(x['P3_verdict'] is None for x in b['P3']) == 4


def gain_duplicate_labels_retained():
    specs = [dict(label='same', family='fixture', model='one', source_id='A'),
             dict(label='same', family='fixture', model='two', source_id='B')]
    a = r.gain_summary(specs, {'A': source('A', [gain_row(0., 2., net=1.)]),
                             'B': source('B', [gain_row(0., 2., net=3.)])})
    assert len(a['models']) == 2 and len(a['duplicate_label_ledger']) == 1
    assert a['producer_projection']['same']['1-3']['median_log_g_net'] == 3.
    assert a['duplicate_label_ledger'][0]['earlier_literal_bins']['1-3']['median_log_g_net'] == 1.


def complete_leaf_differences_and_shapes():
    a = r.compare_leaves({'x': 1.+2**-52, 'v': [True], 'extra': 4}, {'x': 1., 'v': [1, 2], 'e': {}})
    by = {x['path']: x for x in a if x['disposition'] != 'sequence_shape_mismatch'}
    assert by['/x']['signed_difference'] == 2**-52
    assert by['/v/0']['disposition'] == 'different'
    assert by['/v/1']['disposition'] == 'missing_own_leaf'
    assert by['/extra']['disposition'] == 'extra_own_key'
    assert by['/e']['disposition'] == 'empty_mapping_difference'
    b = r.compare_leaves({'nonfinite': 'nan'}, float('nan'))[0]
    assert b['disposition'] == 'different' and b['numeric_pair_finite'] is False


def strict_schema_refusals():
    good = {'A': source('A', [scale_row(0.), scale_row(.1)])}
    changes = [(('id',), 'wrong'), (('expected_rows',), 3), (('available',), 1)]
    for path, value in changes:
        bad = copy.deepcopy(good); bad['A'][path[0]] = value
        try: r._source(bad, 'A')
        except ValueError: pass
        else: raise AssertionError('invalid source schema accepted')
    bad = copy.deepcopy(good); bad['A']['rows'][0]['completed'] = 1
    try: r._one_scale(spec(), bad)
    except ValueError: pass
    else: raise AssertionError('non-Boolean completion accepted')
    for value in (True, '0', complex(1, 0)):
        try: r.number(value)
        except ValueError: pass
        else: raise AssertionError('scalar coercion accepted')


GROUPS = [duplicate_chain_and_scalar_differences, stable_equal_endpoint_and_conflicting_labels,
          classification_priority_and_invalidity, nonfinite_refusal_and_discarded_nonfinite,
          missing_optional_and_historical_failed_source, upper_only_gate_and_nonuniform_weights,
          p1_strict_p2_ties_and_missing, average_tie_ranks_and_undefined_core,
          all_cell_gain_bins_and_boundaries, gain_empty_nonfinite_period_and_missing,
          p3_every_comparison_and_assumption, gain_duplicate_labels_retained,
          complete_leaf_differences_and_shapes, strict_schema_refusals]


def run_controls():
    """Manufactured only; retain every failure rather than stopping at the first."""
    outcomes = []
    for group in GROUPS:
        try:
            group()
            outcomes.append({'group': group.__name__, 'passed': True})
        except Exception as error:
            outcomes.append({'group': group.__name__, 'passed': False,
                             'error_type': type(error).__name__, 'error': str(error)})
    return {'manufactured_only': True, 'groups': outcomes, 'all_passed': all(x['passed'] for x in outcomes)}
