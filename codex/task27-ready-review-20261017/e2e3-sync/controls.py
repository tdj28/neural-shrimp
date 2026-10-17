"""Manufactured controls only. Prepared, NOT executed. No target paths or producer imports."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e3 as e
import synchronization as s


def refuses(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('required refusal absent')


def record(i, layout, answer='1', gold='1', batch=0):
    return {'index': i, 'layout': layout, 'batch': batch, 'row': i % layout, 'prompt_len': 100,
            'answer': answer, 'gold': gold, 'correct': answer is not None and answer == gold,
            'gain': [.05, .1], 'text': 'The answer is ' + (answer or 'unknown')}


def arm(name, flip=False):
    return {'id': name, 'args': {'layouts': [8,16], 'temperature': .6, 'gain_cap': 0.,
                               'top_n': 15, 'soft_steps': 2, 'out': name},
            'records': [record(i, l, '2' if flip and l == 16 else '1', batch=i//2)
                        for i in range(4) for l in [8,16]]}


def test_plurality():
    assert e.plurality(['a','b','b','a']) == 'b'
    assert e.plurality([None,'a',None,'b']) == 'a'
    assert e.plurality([None]*8) is None


def test_extract():
    assert e.extract('1 then The answer is 1,002. then 90') == '1002'
    assert e.extract('value 1 then -2.5') == '-2.5'
    assert e.extract('The answer is unknown') is None


def test_bootstrap():
    out = e.bootstrap({'a':[0.], 'b':[1.,1.,1.]}, np.random.default_rng(2), 100)
    assert set(out['draws']) <= {0., .75, 1.} and out['cluster_counts'] == [1,3]
    assert e.bootstrap({}, np.random.default_rng(2))['interval'] is None
    assert e.mcnemar(0,0) == 1. and e.mcnemar(3,0) == .25
    assert abs(e.wilson(0,4)[0]) < 1e-15


def test_e2():
    out = e.e2([arm('base', True), arm('cap')], 'base', list(range(4)), 20)
    c = out['contrasts']['cap - base']
    assert c['flip_rate_difference'] == -1 and c['accuracy_difference'] == .5
    assert c['flip_cluster95'] == [-1.,-1.]
    assert out['contrast_evidence']['cap - base']['reduces_flips_descriptive']
    broken = arm('missing'); broken['records'].pop()
    x = e.e2([arm('base'), broken], 'base', list(range(4)), 20)
    assert x['coverage'][1]['missing'] == [[3,16]] and x['arms']['missing']['problems'] == 3


def test_schema_and_rounding():
    bad = arm('duplicate'); bad['records'].append(bad['records'][0])
    refuses(lambda: e.e2([bad], 'duplicate', list(range(4)), 10))
    bad = arm('steps'); bad['records'][0]['gain'] = [1.]
    refuses(lambda: e.e2([bad], 'steps', list(range(4)), 10))
    a = arm('cap'); a['args']['gain_cap'] = .1
    for r in a['records']:
        r['gain'] = [a['args']['gain_cap'] * 1.001, .10011]; r['temp'] = [.6,.5]
    out = e.e2([a], 'cap', list(range(4)), 10)
    assert out['arms']['cap']['steps_cap_unmet_share'] == .5
    assert out['coverage'][0]['gain_boundary_ambiguous_count'] == 8
    a['records'][0]['gain'][0] = float('nan')
    out = e.e2([a], 'cap', list(range(4)), 10)
    assert out['coverage'][0]['nonfinite_gain'] == 1 and out['arms']['cap']['mean_gain'] is None


def test_e3():
    def single(name, ans):
        return {'id': name, 'args': {'layouts':[16]},
                'records':[dict(record(i,16,ans), perturb_realized=.001) for i in range(4)]}
    soft = [single('s'+str(i), None if i == 0 else '1') for i in range(8)]
    hard = [single('h'+str(i), '2') for i in range(8)]
    out = e.e3(single('base','1'),soft,single('greedy','1'),hard,list(range(4)),20)
    assert out['projection']['accuracy']['soft_vote'] == 1.
    assert out['projection']['soft_distinct_answers_mean'] == 2.
    assert out['projection']['accuracy']['soft_single'] == .875
    soft[0]['records'].pop()
    assert e.e3(single('base','1'),soft,single('greedy','1'),hard,list(range(4)),20)['excluded_from_common'] == [3]


def test_threshold():
    out = s.threshold([.3,0.,.2,.1], [True,True,False,True])
    assert out == {'c_lo':.2,'c_hi':.3,'status':'bracketed','non_monotone':True}
    assert s.threshold([0.,.1],[True,False])['status'] == 'largest c not synchronized'
    assert s.threshold([0.,.1],[False,False])['c_hi'] is None
    assert s.threshold([0.,.1],[True,True])['c_lo'] is None
    refuses(lambda: s.threshold([.1,.1],[True,False]))


def test_runs_bursts():
    r = s.laminae([True,True,False,True,False,True,True,True])
    assert r['lengths'] == [2,1,3] and r['censored_first'] and r['censored_last']
    assert s.bursts([1e-9,1e-3,2e-3,3e-3,1e-8,2e-3,1e-9,2e-3]) == [3,8]
    assert s.slope([2]*99)['fit'] is None
    assert s.slope([2]*100)['fit'] is None  # enough runs, insufficient distinct bins


def test_slope_ols():
    lengths = [2]*200+[3]*140+[4]*100+[6]*65+[10]*30+[16]*20+[24]*10+[40]*5
    result = s.slope(lengths)
    ok = np.array(result['fit_mask'])
    x = np.log(np.array(result['midpoints'])[ok]); y = np.log(np.array(result['density'])[ok])
    oracle = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0][1]
    assert abs(result['fit']['slope'] - oracle) < 1e-12


def test_transverse():
    lam = math.log(2)
    cs = [0.,.25,.5]
    twins = np.array([[lam]*20,[0.]*20,[0.]*20])
    out = s.transverse([{'index':0}], [{'cell':0,'c':c} for c in cs], np.array([[lam]*30]), twins,{0:lam})
    assert abs(out['cells']['0']['c_star_jvp'] - .25) < 1e-15
    assert len(out['undefined_differences']) == 1 and not out['all_grid_difference_defined']
    assert out['max_absolute_measured_minus_predicted'] < 1e-14
    assert s.block_error(np.arange(23.))['trailing_unused'] == 3


def test_sweep():
    meta = {'pairs':[{'cell':0,'c':0.},{'cell':0,'c':.2}], 'steps':6, 'complete':True,
            'args':{'start':'near','record':6},'commit':{},'sweep_npz_sha256':'manufactured'}
    dist = np.array([[.1]*7,[.1,1e-9,2e-3,1e-9,2e-3,1e-9,1e-9]])
    tok = np.array([[1]*6,[2]*6,[1]*6,[1]*6])
    out = s.sweep(meta,dist,tok,tail=2)['0']
    assert out['pairs'][1]['first_below_tol'] == 1 and out['pairs'][1]['synced']
    assert out['pairs'][1]['bursts_1e-8_to_1e-3'] == 2
    assert out['tv_c0_A_vs_B_tokens'] == 1. and out['pairs'][1]['tv_A_vs_c0_tokens'] == .5
    dist[1,1] = float('nan')
    assert s.sweep(meta,dist,tok,tail=2)['0']['threshold'] is None
    refuses(lambda: s.sweep(meta,dist.T,tok,tail=2))


def test_periodic():
    row = {'sweep':'near','cell':0,'c':.1,'primitive_period':2,'return_distance_at_P':1e-13,
           'min_return':1e-14,'argmin_return':4,'pair_distance_now':.1,'top1_A':[1,2],'top1_B':[2,1]}
    doc = {'args':{'max_period':40,'tol':1e-9},'pairs':[row]}
    assert s.periodic(doc)[0]['all_scalar_gates']
    row['return_distance_at_P'] = 1e-8
    assert not s.periodic(doc)[0]['all_scalar_gates']
    doc['pairs'].append(copy.deepcopy(row))
    refuses(lambda: s.periodic(doc))


def main():
    tests = [test_plurality,test_extract,test_bootstrap,test_e2,test_schema_and_rounding,test_e3,
             test_threshold,test_runs_bursts,test_slope_ols,test_transverse,test_sweep,test_periodic]
    outcomes = []
    for fn in tests:
        try:
            fn(); outcomes.append({'name':fn.__name__,'passed':True})
        except Exception as exc:
            outcomes.append({'name':fn.__name__,'passed':False,'error':repr(exc)})
    report = {'schema':'task27-e2e3-sync-manufactured-v1','groups':outcomes,
              'passed':sum(r['passed'] for r in outcomes),'failed':sum(not r['passed'] for r in outcomes)}
    print(json.dumps(report,indent=2))
    return int(report['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
