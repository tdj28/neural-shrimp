"""Synthetic-only arithmetic, decoder binding and complete-census tests."""
from pathlib import Path
import copy
from fractions import Fraction
import math
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import mixture_readout_evaluation as m


def vector(x):
    result=np.zeros(96);result[0]=x
    return result


def define(A,B,sigma,*,one_mode_rho=None):
    return dict(complete=True,available=True,status='defined',A=A.copy(),B=B.copy(),
                sigma=float(sigma),one_mode_rho=one_mode_rho)


def envelope(p=.9,*,qualified=True,width=2e-6):
    return dict(complete=True,qualified=qualified,available=True,status='complete',
                interval=[max(0.,p-width/2),min(1.,p+width/2)],nominal=p,error=None,diagnostics={})


def probability(*args,**kwargs):return envelope()
def linear(*args,**kwargs):return envelope(.7)


def fixture(cell=None,*,one_mode=True,alternate_offset=0.):
    cell=copy.deepcopy(cell or m.cell_census()[0]);period=37.3312345;k=int(cell['window'][1:])
    times=period*(k-1+(np.arange(32)+.5)/32)
    A=vector(1e-4);B=vector(2e-5);base=np.zeros(96);h=vector(1.)
    sealed=dict(cell=cell,id=cell['id'],context_id=cell['context_id'],complete=True,status='complete',
        sigma=1e-4,epsilon=1e-4,baseline=base,h=h,times=times,period=period,definitions={})
    for recipe in m.RECIPES:
        rho=.5 if recipe=='one_mode' else None
        if recipe=='one_mode' and not one_mode:
            sealed['definitions'][recipe]=m._unavailable_definition(cell,recipe,'No real rho');continue
        baseline=base.copy()
        if recipe=='alternate':baseline+=vector(alternate_offset)
        sealed['definitions'][recipe]=dict(id=cell['id']+'--classifier--'+recipe,recipe=recipe,
            recipe_name=m.RECIPE_NAMES[recipe],available=True,complete=True,status='complete',
            definition=define(A,B,1e-4,one_mode_rho=rho),A=None if recipe=='one_mode' else A.copy(),B=B.copy(),
            baseline=baseline,rho=rho,error=None)
    scores={}
    for method in m.methods_for_cell(cell):
        rows=[]
        for first in m.SIGNS:
            for second in m.SIGNS:
                hist=f'{cell["context_id"]}--identity--s1{first:+d}--s2{second:+d}';ident=hist+'--'+method
                task=dict(id=ident,history_id=hist,context_id=cell['context_id'],setting=cell['setting'],
                    phase_index=cell['phase_index'],phase_fraction=cell['phase_fraction'],split=cell['split'],arm='identity',
                    first_sign=first,second_sign=second,epsilon=1e-4,method=method)
                mean=first*A+second*B
                rows.append(dict(task=task,task_id=ident,history_id=hist,context_id=cell['context_id'],complete=True,
                    qualified=True,status='complete',windows={cell['window']:dict(times=times.copy(),
                    activation_displacement=mean,primary=m.old._project(h,mean,first))}))
        scores[method]=rows
    return cell,sealed,scores


def run_fixture(*args,**kwargs):
    return m.evaluate_cell(*fixture(*args,**kwargs),probability_fn=probability,linear_probability_fn=linear)


def test_complete_census_is_4704_not_trajectories():
    cells=m.cell_census();slots=[s for c in cells for s in m.calculation_census(c)]
    assert len(cells)==144 and len(slots)==4704 and len({s['id'] for s in slots})==4704
    assert sum(s['family']=='finite' and s['recipe']=='primary' for s in slots)==1200
    assert sum(s['family']=='alternate' for s in slots)==576
    assert sum(s['family']=='reference' for s in slots)==1728
    assert sum(len(m.observation_census(c)) for c in cells)==1200


def test_panel_cell_has40_slots_and_separate_secondaries():
    packet=run_fixture()
    assert packet['complete'] and packet['qualified'] and packet['secondary_qualified']
    assert len(packet['calculations'])==40 and len(packet['class_rows'])==12
    assert len(packet['method_comparisons'])==16 and len(packet['reference_comparisons'])==4


def test_nonpanel_cell_has32_slots():
    packet=run_fixture(m.cell_census()[3])
    assert packet['complete'] and len(packet['calculations'])==32 and len(packet['class_rows'])==8


def test_missing_one_mode_never_suppresses_available_primary():
    packet=run_fixture(one_mode=False)
    assert packet['complete'] and packet['qualified'] and not packet['secondary_complete']
    assert not packet['secondary_qualified']
    assert all(not r['result']['complete'] for r in packet['calculations'] if r['recipe']=='one_mode')


def test_exact_alternate_translation_and_mathematical_one_mode_terms():
    cell,sealed,scores=fixture(alternate_offset=1.)
    # Deliberate cancellation stress: only the probability core may add these terms.
    scores['rk4-fine'][0]['windows'][cell['window']]['activation_displacement']=vector(1e16)
    scores['rk4-fine'][0]['windows'][cell['window']]['primary']=m.old._project(sealed['h'],vector(1e16),-1)
    sealed['baseline']=vector(1.);sealed['definitions']['primary']['baseline']=vector(1.)
    sealed['definitions']['alternate']['baseline']=vector(1e16)
    observations=m._observations(cell,sealed,scores)
    spec=next(s for s in m.calculation_census(cell) if s['family']=='alternate' and s['first_sign']==s['second_sign']==-1)
    terms=m._terms(sealed,spec,observations)
    exact=sum((Fraction.from_float(t['coefficient'])*Fraction.from_float(t['vector'][0]) for t in terms),Fraction())
    assert exact==1 and sum(t['coefficient']*t['vector'][0] for t in terms)==0.
    one=next(s for s in m.calculation_census(cell) if s['family']=='reference' and s['recipe']=='one_mode')
    terms=m._terms(sealed,one,observations)
    assert terms[0]['coefficient']==-.5
    np.testing.assert_array_equal(terms[0]['vector'],sealed['definitions']['one_mode']['B'])


def test_alternate_reference_original_linear_keeps_physical_offset():
    args=fixture(alternate_offset=3e-5);calls=[]
    def capture(h,terms,first,sigma):
        calls.append(copy.deepcopy(terms));return linear(h,terms,first,sigma)
    m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=capture)
    alternate=calls[4:8]
    assert all(len(terms)==4 for terms in alternate)
    for terms in alternate:
        np.testing.assert_array_equal(terms[-2]['vector'],args[1]['definitions']['alternate']['baseline'])
        assert terms[-2]['coefficient']==1. and terms[-1]['coefficient']==-1.


def test_control_dominance_is_four_class_mean_not_per_class():
    args=fixture();packet=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    control=[r for r in packet['calculations'] if r['family']=='reference' and r['recipe']=='primary']
    for row,p,q in zip(control,[.9,.9,.5,.5],[1.,.4,.7,.4],strict=True):
        row['result']=envelope(p);row['linear_reference']=envelope(q)
    rebuilt=m.assemble_cell(*args,packet['calculations'])
    assert rebuilt['controls']['primary']['qualified'] and rebuilt['qualified']
    assert control[0]['result']['nominal']<control[0]['linear_reference']['nominal']


def test_reference_control_genuine_average_failure_retained():
    args=fixture();packet=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    for r in packet['calculations']:
        if r['family']=='reference' and r['recipe']=='primary':r['result']=envelope(.4)
    rebuilt=m.assemble_cell(*args,packet['calculations'])
    assert rebuilt['complete'] and not rebuilt['qualified']
    assert not rebuilt['controls']['primary']['balanced_dominance_not_contradicted']


def test_comparison_uses_maximum_possible_difference_not_midpoint():
    # Nominals agree but a conservative probability difference breaches .001.
    a=envelope(.5,width=1e-4);b=envelope(.50096,width=1e-4)
    comparison=m._comparison('test',a,b)
    assert abs(a['nominal']-b['nominal'])<.001
    assert comparison['maximum_possible_difference']>.001 and not comparison['qualified']


def test_interval_averages_and_differences_round_outward():
    values=[[.1,.2],[.2,.3],[.7,.9]];interval=m._mean_interval(values)
    exact_lo=sum(Fraction.from_float(x[0]) for x in values)/3
    exact_hi=sum(Fraction.from_float(x[1]) for x in values)/3
    assert Fraction.from_float(interval[0])<=exact_lo and Fraction.from_float(interval[1])>=exact_hi
    difference=m._difference_interval(interval,[.17,.21])
    assert Fraction.from_float(difference[0])<=Fraction.from_float(interval[0])-Fraction.from_float(.21)


def test_linear_replay_failure_suppresses_qualification_not_computations():
    args=fixture();args[2]['rk4-fine'][0]['windows'][args[0]['window']]['primary']['standardized_mean']+=.1
    result=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    assert result['complete'] and not result['qualified']
    assert result['counts']['calculations_complete']==40
    assert not result['gates']['original_linear_replay']


@pytest.mark.parametrize('change',[
    lambda args:args[2]['rk4-fine'][0]['task'].update(arm='timing'),
    lambda args:args[2]['rk4-fine'][0].update(qualified=False),
    lambda args:args[2]['rk4-fine'][0]['windows'][args[0]['window']].update(activation_displacement=np.zeros(96,dtype=np.float32)),
    lambda args:args[2]['rk4-fine'].append(copy.deepcopy(args[2]['rk4-fine'][0]))])
def test_wrong_physical_task_or_array_schema_retains_full_denominator(change):
    args=fixture();change(args);p=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    assert not p['complete'] and not p['qualified'] and len(p['calculations'])==40


def test_callback_prefix_salvage_never_runs_missing_calculations():
    args=fixture();recorded=[];calls=[]
    def prob(*a):calls.append(1);return envelope()
    def callback(record):
        recorded.append(record)
        if len(recorded)==3:raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError):m.evaluate_cell(*args,probability_fn=prob,linear_probability_fn=linear,progress_fn=callback)
    assert len(calls)==3
    partial=m.assemble_cell(*args,recorded)
    assert partial['counts']['calculations_complete']==3 and len(partial['calculations'])==40
    assert not partial['complete'] and len(calls)==3


def test_single_saved_slot_replay_only_invokes_one_probability():
    args=fixture();calls=[]
    def prob(*a):calls.append(a);return envelope()
    spec=m.calculation_census(args[0])[5]
    record=m.evaluate_calculation(*args,spec,probability_fn=prob,linear_probability_fn=linear)
    assert record['id']==spec['id'] and len(calls)==1


def test_failed_linear_control_keeps_successful_mixture_packet():
    args=fixture()
    def bad(*args):raise RuntimeError('synthetic CDF error')
    result=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=bad)
    assert all(r['result']['complete'] for r in result['calculations'])
    assert all(not r['linear_reference']['complete'] for r in result['calculations'] if r['family']=='reference')
    assert not result['complete']


@pytest.mark.parametrize('corruption',[
    lambda p:p['calculations'].append(copy.deepcopy(p['calculations'][0])),
    lambda p:p['calculations'][0].update(recipe='alternate'),
    lambda p:p['calculations'][0]['result'].update(qualified=True,complete=False),
    lambda p:p['calculations'][0]['result'].update(interval=[-.1,.5])])
def test_calculation_binding_and_false_pass_rejections(corruption):
    args=fixture();p=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear);corruption(p)
    with pytest.raises(ValueError):m.assemble_cell(*args,p['calculations'])


def test_full_missing_summary_retains_all_census_and_fixed_bounds():
    s=m.summarize(m.cell_census(),{},{});c=s['counts']
    assert not s['complete'] and not s['qualified'] and s['primary'] is None
    assert c['total_new_probability_calculations']==4704 and c['finite_class_method_rows']==1200
    assert c['primary_method_comparisons']==c['secondary_method_comparisons']==624
    assert c['reference_comparisons']==576 and c['method_context_window_groups']==300
    assert c['auxiliary_linear_reference_calculations']==1728 and c['mixture_plus_auxiliary_probability_calculations']==6432
    assert c['auxiliary_reference_linear_cdfs_complete']==0
    assert s['primary_failure_bounds']['gain_interval']==[-1.,1.]


@pytest.fixture(scope='module')
def full_results():
    cells=m.cell_census();definitions={};results={}
    for cell in cells:
        args=fixture(cell);definitions[cell['id']]=args[1]
        results[cell['id']]=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    return cells,definitions,results


def test_full_144_cell_summary_has24_phase_primary_and_no_subset(full_results):
    s=m.summarize(*full_results)
    assert s['complete'] and s['qualified'] and s['primary']['phases']==24
    assert s['counts']['calculations_complete']==4704
    assert s['counts']['auxiliary_reference_linear_cdfs_complete']==1728
    assert s['counts']['mixture_plus_auxiliary_probability_calculations']==6432
    assert s['counts']['cells_qualified']==144
    assert len(s['primary_failure_bounds']['phase_rows'])==24
    assert all(len(v['means'])==3 for v in s['settings'].values())


def test_one_missing_core_cell_suppresses_whole_primary(full_results):
    cells,definitions,results=full_results;partial=dict(results);partial.pop(cells[0]['id'])
    s=m.summarize(cells,definitions,partial)
    assert not s['qualified'] and s['primary'] is None
    assert s['settings']['core']['means'] is None
    assert s['settings']['failed-near-edge']['qualified']
    assert s['primary_failure_bounds']['qualified_phases']==24


def test_summary_rejects_forged_original_probability(full_results):
    cells,definitions,results=full_results;partial=dict(results);packet=copy.deepcopy(partial[cells[0]['id']])
    packet['observations'][0]['original_probability']=.1;partial[cells[0]['id']]=packet
    with pytest.raises(ValueError,match='original probability'):m.summarize(cells,definitions,partial)


def test_summary_rejects_tampered_completion_flag(full_results):
    cells,definitions,results=full_results;partial=dict(results);packet=dict(partial[cells[0]['id']],complete=False);partial[cells[0]['id']]=packet
    with pytest.raises(ValueError,match='complete differs'):m.summarize(cells,definitions,partial)


def test_reference_adapter_uses_only_declared_alternate_and_preserves_missing_secondary(monkeypatch):
    cell,sealed,_=fixture(one_mode=False);A=vector(1e-4);B=vector(2e-5);altA=vector(3e-4);altB=vector(4e-5)
    context=dict(id=cell['context_id'],setting=cell['setting'],phase_index=cell['phase_index'],phase_fraction=cell['phase_fraction'],split=cell['split'],period=37.3312345,rho=None)
    indices=np.arange(32,dtype=np.int64)
    primary=dict(times=sealed['times'],indices=indices,segment='post',baseline=np.zeros((32,3)),h=sealed['h'],linear_predictions={'identity':dict(A=A,B=B)})
    outer=dict(context_id=cell['context_id'],status='complete',qualified=True,evaluation=dict(context_id=cell['context_id'],context=context,complete=True,qualified=True,epsilon=1e-4,sigma=1e-4,windows={cell['window']:primary}))
    sentinel={'synthetic':'alternate'};calls=[]
    monkeypatch.setattr(m.old,'_tangent',lambda packet,ctx,method:calls.append((packet,ctx,method)))
    monkeypatch.setattr(m.old,'_predict',lambda ctx,packet,window:(dict(times=sealed['times'],indices=indices,segment='post',activations=np.full((32,3),.2)),{'identity':dict(A=altA,B=altB)}))
    result=m.seal_cell(cell,outer,sentinel,define_fn=define)
    assert result['complete'] and not result['definitions']['one_mode']['available']
    assert calls==[(sentinel,context,'rk45')]
    np.testing.assert_array_equal(result['definitions']['alternate']['A'],altA)
    np.testing.assert_array_equal(result['definitions']['alternate']['baseline'],np.full(96,.2))
    np.testing.assert_array_equal(result['definitions']['primary']['baseline'],np.zeros(96))


def test_invalid_reference_seal_cannot_construct_finite_decoder():
    c=m.cell_census()[0];p=m.seal_cell(c,{},None,define_fn=lambda *args,**kwargs:pytest.fail('Unexpected construction'))
    assert not p['complete'] and set(p['definitions'])==set(m.RECIPES)


def test_classifier_wrapper_cannot_disagree_with_core_reference():
    args=fixture();args[1]['definitions']['alternate']['A'][1]=1.
    with pytest.raises(ValueError,match='core A'):m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)


def test_boolean_sign_is_not_an_integer_label():
    args=fixture();args[2]['rk4-fine'][2]['task']['first_sign']=True
    packet=m.evaluate_cell(*args,probability_fn=probability,linear_probability_fn=linear)
    assert not packet['complete'] and not packet['qualified']
    assert any('first_sign differs' in (o['error'] or '') for o in packet['observations'])


def test_invalid_core_envelope_retains_its_returned_partial_evidence():
    args=fixture()
    def malformed(*args):return dict(envelope(),interval=[float('nan'),1.],diagnostics={'sentinel':'keep'})
    packet=m.evaluate_cell(*args,probability_fn=malformed,linear_probability_fn=linear)
    first=packet['calculations'][0]['result']
    assert not first['complete'] and first['partial_result']['diagnostics']['sentinel']=='keep'
    assert math.isnan(first['partial_result']['interval'][0])


def test_contract_census_and_scope_are_explicit():
    c=m.contract()
    assert c['exact']['total_new_probability_calculations']==4704
    assert c['exact']['auxiliary_linear_reference_calculations']==1728
    assert c['exact']['mixture_plus_auxiliary_probability_calculations']==6432
    assert c['primary_requires_both_settings'] and not c['fresh_holdout']
    assert not c['secondary_failures_invalidate_primary'] and not c['bootstrap']


def test_new_classifier_failure_preserves_known_original_linear_bound():
    cells=m.cell_census()
    chosen=next(c for c in cells if c['setting']=='failed-near-edge' and c['phase_index']==3 and c['window']=='k4')
    cell,sealed,scores=fixture(chosen)
    sealed['definitions']['alternate']=m._unavailable_definition(cell,'alternate','Synthetic precision failure')
    sealed.update(complete=False,status='failed')
    packet=m.evaluate_cell(cell,sealed,scores,probability_fn=probability,linear_probability_fn=linear)
    assert not packet['complete'] and not packet['qualified']
    assert all(o['complete'] and o['replay_pass'] for o in packet['observations'])
    assert all(g['original_linear_interval'] is not None for g in packet['groups'])
    summary=m.summarize(cells,{cell['id']:sealed},{cell['id']:packet})
    assert summary['counts']['cells_complete']==0
    bound=summary['primary_failure_bounds']['phase_rows'][3]
    assert bound['original_linear_known'] and not bound['qualified']
    assert bound['gain_interval'][0]>-1. and bound['gain_interval'][1]<1.
