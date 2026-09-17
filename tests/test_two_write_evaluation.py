"""Synthetic-only EXP-016 census, sealed references, numerical gates and aggregation."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import two_write_evaluation as evaluation
import two_write_acquisition as acquisition
import run_phase_memory_recovery as codec


def flow(t, x, a, g):
    return np.array([1., 0., 0.])


def jacobian(t, x, a, g):
    return np.zeros((3,3))


def analytic_solver(fun, span, initial, *, t_eval, **kwargs):
    values = np.repeat(initial[:,None], len(t_eval), axis=1)
    values[0] += t_eval
    return SimpleNamespace(t=t_eval.copy(), y=values, success=True,
        message='fabricated analytic constant drift, no integration', nfev=0,njev=0,nlu=0)


def native_drift(alpha, gamma, initial, dt, steps, stride, count, output, final, progress):
    x=np.ctypeslib.as_array(initial,shape=(3,)).copy()
    states=np.ctypeslib.as_array(output,shape=(count*3,)).reshape(count,3)
    states[:]=x;states[:,0]+=np.arange(count)*(stride*dt)
    np.ctypeslib.as_array(final,shape=(3,))[:]=states[-1]
    np.ctypeslib.as_array(progress,shape=(2,))[:]=[steps,count]
    return count


def synthetic_context():
    phase=evaluation.phase_census()[0]
    return dict(id='core--p00',setting='core',**phase,phase_epoch=phase['phase_fraction'],
        period=1.,alpha=0.,gamma=0.,initial=np.zeros(3),flow=np.array([1.,0.,0.]),
        phase_gradient=np.array([1.,0.,0.]),nominal_event_phase_gradient=np.array([1.,0.,0.]),
        direction=np.array([0.,1.,0.]),rho=-.8,
        provenance=dict(reference_qualified=True,reference_status='complete'))


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.setattr(acquisition,'SAMPLES_PER_CYCLE',512)
    ctx=synthetic_context()
    primary=acquisition.acquire_tangent(ctx,'dop853',rhs=flow,jacobian=jacobian,solver=analytic_solver)
    alternate=acquisition.acquire_tangent(ctx,'rk45',rhs=flow,jacobian=jacobian,solver=analytic_solver)
    sealed=evaluation.seal_reference(ctx,primary,alternate)
    assert sealed['complete'] and sealed['qualified'], sealed
    return ctx,primary,alternate,sealed


def selected_task(first=1,second=-1,arm='identity',method='rk4-fine'):
    return next(t for t in evaluation.task_census() if t['context_id']=='core--p00'
        and t['first_sign']==first and t['second_sign']==second and t['arm']==arm and t['method']==method)


def trace(sealed,**kwargs):
    task=selected_task(**kwargs)
    packet=acquisition.acquire_trace(task,sealed['context'],native_fn=native_drift,rhs=flow,solver=analytic_solver)
    return task,packet


def physical(task):
    return dict({k:v for k,v in task.items() if k not in ('method','history_id')},id=task['history_id'])


def test_exact_census_and_random_phases():
    phases=evaluation.phase_census();histories=evaluation.history_census();tasks=evaluation.task_census(histories)
    assert len(phases)==24 and len(histories)==1296 and len(tasks)==2700
    assert len(evaluation.comparison_census(histories))==1404
    assert len({t['id'] for t in tasks})==2700
    assert sum(t['method']=='dop853-independent' for t in tasks)==108
    assert sum(t['split']=='development' for t in tasks)==918
    assert sum(t['split']=='heldout' for t in tasks)==1782
    np.testing.assert_array_equal([p['phase_fraction'] for p in phases],np.random.Generator(np.random.PCG64(2026091701)).random(24))
    for setting in evaluation.SETTINGS:
        assert sum(h['setting']==setting for h in histories)==648
    assert evaluation.contract()['expected']['all_jobs']==2796


@pytest.mark.parametrize('edit',[
    lambda p:p.pop(),lambda p:p[0].update(phase_index=True),
    lambda p:p[8].update(split='development'),lambda p:p[0].update(phase_fraction=np.nan),
    lambda p:p[1].update(phase_fraction=p[0]['phase_fraction']),lambda p:p[0].update(extra=1)])
def test_invalid_phase_census(edit):
    phases=evaluation.phase_census();edit(phases)
    with pytest.raises(ValueError):evaluation.history_census(phases)


def test_seal_identity_observer_shared_secondary_zero_and_immediate(reference):
    ctx,p,a,sealed=reference
    assert set(sealed['windows'])==set(evaluation.WINDOWS)
    for name,w in sealed['windows'].items():
        assert w['h'].shape==(96,) and w['baseline'].shape==(32,3)
        assert np.isclose(np.linalg.norm(w['h']),1.)
        assert w['linear_predictions']['identity']['primary_accuracy']['with_second']>=.5
        if name=='immediate':
            assert np.all(w['times']<ctx['period']/8)
            for arm in evaluation.ARMS:
                np.testing.assert_array_equal(w['linear_predictions'][arm]['B'],np.zeros(96))
                np.testing.assert_array_equal(w['linear_predictions'][arm]['A'],w['linear_predictions']['identity']['A'])
            assert 'calibration' in w['one_mode_prediction']['scope']
        else:
            assert not w['arm_observers']['timing']['available']
            assert w['arm_observers']['identity']['available']
            np.testing.assert_array_equal(w['one_mode_prediction']['A'],-.8*w['linear_predictions']['identity']['B'])
    assert sealed['arithmetic_qualified'] and sealed['reference_comparison']['normalization_max']==0


def test_rho_unavailable_is_comparator_only(reference):
    ctx,p,a,_=reference;ctx['rho']=None
    sealed=evaluation.seal_reference(ctx,p,a)
    assert sealed['complete'] and sealed['qualified']
    assert not sealed['windows']['k4']['one_mode_prediction']['available']
    assert sealed['windows']['immediate']['one_mode_prediction']['available']


@pytest.mark.parametrize('edit',[
    lambda ctx,p,a:ctx['provenance'].update(reference_qualified=False),
    lambda ctx,p,a:ctx.update(nominal_event_phase_gradient=np.array([1.000001,0.,0.])),
    lambda ctx,p,a:a['post']['states'].__setitem__((100,1),2e-6),
    lambda ctx,p,a:a['post']['phi'].__setitem__((100,2,2),1.1),
])
def test_reference_numerical_failures_are_retained(reference,edit):
    ctx,p,a,_=reference;edit(ctx,p,a)
    sealed=evaluation.seal_reference(ctx,p,a)
    assert sealed['complete'] and not sealed['qualified']
    assert any(v is False for v in sealed['gates'].values())


@pytest.mark.parametrize('edit',[
    lambda ctx,p,a:a.update(method='dop853'),
    lambda ctx,p,a:a['post'].update(phi=a['post']['phi'].astype('float32')),
    lambda ctx,p,a:a['pre']['solver'].update(success=False),
    lambda ctx,p,a:a['pre']['solver'].update(nfev=True),
    lambda ctx,p,a:a['post']['times'].__setitem__(1,np.nan),
])
def test_failed_reference_identity_dtype_or_solver(reference,edit):
    ctx,p,a,_=reference;edit(ctx,p,a)
    sealed=evaluation.seal_reference(ctx,p,a)
    assert not sealed['complete'] and not sealed['qualified'] and sealed['error']


def test_primary_zero_A_is_failed_prerequisite(reference):
    ctx,p,a,_=reference
    for packet in (p,a):
        packet['pre']['phi'][1:, :, 1]=0
    sealed=evaluation.seal_reference(ctx,p,a)
    assert not sealed['complete'] and not sealed['qualified'] and 'A is zero' in sealed['error']


def test_all_arms_and_no_first_interference(reference):
    sealed=reference[-1]
    for arm in evaluation.ARMS:
        task,packet=trace(sealed,first=0,second=1,arm=arm)
        score=evaluation.score_trace(task,packet,sealed)
        assert score['complete'] and score['qualified'],score
        for w in score['windows'].values():
            assert w['primary']['probability_correct']==.5
            assert w['zero_zero_drift_pass'] is None
        assert score['windows']['k4']['primary']['standardized_mean']>1
        assert score['windows']['immediate']['primary']['standardized_mean']==0


def test_actual_same_baseline_is_not_unknown_second_conditioned(reference):
    sealed=reference[-1]
    task,packet=trace(sealed,first=1,second=1)
    score=evaluation.score_trace(task,packet,sealed)
    task0,packet0=trace(sealed,first=1,second=0)
    absent=evaluation.score_trace(task0,packet0,sealed)
    assert score['windows']['immediate']['primary']==absent['windows']['immediate']['primary']
    assert score['windows']['k4']['primary']['projected_mean']>1.9*absent['windows']['k4']['primary']['projected_mean']


def test_zero_zero_drift_gate(reference):
    sealed=reference[-1];task,packet=trace(sealed,first=0,second=0)
    packet['post']['states'][100:,1]+=1e-5
    score=evaluation.score_trace(task,packet,sealed)
    assert score['complete'] and not score['qualified']
    assert not score['gates']['zero_zero_drift']


@pytest.mark.parametrize('edit',[
    lambda t,p:p['task'].update(first_sign=True),
    lambda t,p:p['event'].update(operation_order='write_then_retention'),
    lambda t,p:p['first_pulse']['requested_delta'].__setitem__(1,0.),
    lambda t,p:p['event']['after'].__setitem__(1,1.),
    lambda t,p:p['post']['states'].__setitem__((100,0),np.nan),
    lambda t,p:p['post'].update(states=p['post']['states'].astype('float32')),
    lambda t,p:p['post']['solver'].update(steps=1),
    lambda t,p:p.update(context_id='core--p01'),
    lambda t,p:p['post']['times'].__setitem__(5,6.),
])
def test_trace_binding_failures(reference,edit):
    sealed=reference[-1];task,packet=trace(sealed)
    edit(task,packet)
    score=evaluation.score_trace(task,packet,sealed)
    assert not score['complete'] and not score['qualified'] and score['error']


def test_saved_array_comparison_and_explicit_missing_methods(reference):
    sealed=reference[-1]
    coarse,cp=trace(sealed,method='rk4-coarse');fine,fp=trace(sealed,method='rk4-fine')
    comparison=evaluation.compare_traces(physical(fine),cp,fp,sealed)
    assert comparison['complete'] and comparison['qualified']
    assert comparison['secondary']['qualified']
    missing=evaluation.compare_traces(physical(fine),None,None,sealed,methods=('rk4-fine','dop853-independent'))
    assert not missing['complete'] and missing['id'].endswith('rk4-fine--vs--dop853-independent')


@pytest.mark.parametrize('delta,gate',[(2e-6,'states'),(2e-7,'activation')])
def test_raw_waveform_gate_cannot_be_hidden_by_accuracy(reference,delta,gate):
    sealed=reference[-1];coarse,cp=trace(sealed,method='rk4-coarse');fine,fp=trace(sealed,method='rk4-fine')
    fp['post']['states'][100,2]+=delta # orthogonal to observer, q unchanged
    result=evaluation.compare_traces(physical(fine),cp,fp,sealed)
    assert result['complete'] and not result['qualified'] and not result['gates'][gate]
    assert result['gates']['margins'] and result['gates']['accuracy']


def test_margin_and_accuracy_gates_are_separate_from_saturation(reference):
    sealed=reference[-1];coarse,cp=trace(sealed,first=1,second=-1,method='rk4-coarse');fine,fp=trace(sealed,first=1,second=-1,method='rk4-fine')
    left=evaluation.score_trace(coarse,cp,sealed);right=evaluation.score_trace(fine,fp,sealed)
    right['windows']['k4']['primary']['standardized_mean']+=.011
    result=evaluation.compare_traces(physical(fine),cp,fp,sealed,left_score=left,right_score=right)
    assert result['complete'] and not result['gates']['margins']
    assert result['gates']['accuracy']
    right=evaluation.score_trace(fine,fp,sealed)
    right['windows']['k4']['primary']['probability_correct']+=.0011
    result=evaluation.compare_traces(physical(fine),cp,fp,sealed,left_score=left,right_score=right)
    assert not result['gates']['accuracy'] and result['gates']['margins']


def test_undefined_secondary_does_not_invalidate_primary(reference):
    sealed=reference[-1];coarse,cp=trace(sealed,arm='timing',method='rk4-coarse');fine,fp=trace(sealed,arm='timing',method='rk4-fine')
    result=evaluation.compare_traces(physical(fine),cp,fp,sealed)
    assert result['complete'] and result['qualified']
    assert not result['secondary']['available'] and not result['secondary']['qualified']


def test_packet_codec_roundtrip_score_and_comparison(reference,tmp_path):
    sealed=reference[-1];task,packet=trace(sealed)
    before=evaluation.score_trace(task,packet,sealed)
    a=codec.save_packet(tmp_path,'reference',sealed);b=codec.save_packet(tmp_path,'trace',packet)
    restored_seal=codec.load_packet(a['path']);restored_trace=codec.load_packet(b['path'])
    after=evaluation.score_trace(restored_trace['task'],restored_trace,restored_seal)
    assert after['complete'] and codec.packet_digest(after)==codec.packet_digest(before)
    assert after['windows']['k4']['activation_displacement'].dtype==np.float64


def compact_census():
    histories=evaluation.history_census();tasks=evaluation.task_census(histories)
    scores={}
    for task in tasks:
        prob=.6 if task['setting']=='core' else .7
        windows={}
        for window in evaluation.WINDOWS:
            p=.5 if task['first_sign']==0 else prob
            windows[window]=dict(primary=dict(projected_mean=0.,noise_sd=1e-4,standardized_mean=0.,probability_correct=p),
                secondary=dict(available=False,score=None),linear_prediction=None,one_mode_prediction=None)
        scores[task['id']]=dict(task=task,task_id=task['id'],history_id=task['history_id'],context_id=task['context_id'],
            complete=True,qualified=True,status='complete',windows=windows,gates=dict(reference_qualified=True,zero_zero_drift=True))
    comparisons=[dict(spec,complete=True,qualified=True,status='complete',gates={},secondary=dict(available=False,qualified=False))
        for spec in evaluation.comparison_census(histories)]
    prerequisites={h['context_id']:dict(context_id=h['context_id'],complete=True,qualified=True,status='complete') for h in histories}
    return histories,scores,comparisons,prerequisites


def test_full_summary_exact_denominators_and_paired_primary():
    args=compact_census();result=evaluation.summarize(*args)
    assert result['complete'] and result['qualified']
    assert len(result['scores'])==2700 and len(result['comparisons'])==1404
    assert len(result['balanced_accuracies'])==576 and len(result['contexts'])==48
    assert result['primary']['estimate_percentage_points']==pytest.approx(10.)
    assert result['primary']['central_95_percentage_points']==pytest.approx([10.,10.])
    assert result['settings']['core']['means'] is not None
    assert result['primary_failure_bounds']['edge_minus_core_percentage_points']==pytest.approx([10.,10.])


def test_missing_scores_comparisons_and_prerequisites_keep_denominator():
    h,s,c,p=compact_census()
    removed=next(k for k,v in s.items() if v['task']['setting']=='core' and v['task']['phase_index']==8)
    del s[removed]
    c.pop();del p['failed-near-edge--p09']
    result=evaluation.summarize(h,s,c,p)
    assert not result['complete'] and not result['qualified']
    assert len(result['scores'])==2700 and len(result['comparisons'])==1404
    assert result['primary'] is None
    assert all(x['means'] is None for x in result['settings'].values())
    assert result['primary_failure_bounds']['setting_accuracy']['core']['unknown']==1
    assert result['primary_failure_bounds']['setting_accuracy']['failed-near-edge']['unknown']==2 # missing last comparison p23


def test_setting_failure_does_not_suppress_other_setting_mean():
    h,s,c,p=compact_census();p['core--p00']['qualified']=False
    result=evaluation.summarize(h,s,c,p)
    assert result['complete'] and not result['qualified']
    assert result['settings']['core']['means'] is None
    assert result['settings']['failed-near-edge']['means'] is not None
    assert result['primary'] is None
    assert result['primary_failure_bounds']['setting_accuracy']['core']['unknown']==0 # development failure, heldout blocks intact


@pytest.mark.parametrize('edit',[
    lambda h,s,c,p:h.pop(),lambda h,s,c,p:h.__setitem__(1,h[0]),
    lambda h,s,c,p:s.update(unplanned={}),lambda h,s,c,p:c.append(c[0]),
    lambda h,s,c,p:c[0].update(right_task_id='wrong'),
    lambda h,s,c,p:next(iter(s.values()))['task'].update(first_sign=True),
    lambda h,s,c,p:next(iter(s.values()))['windows']['k4']['primary'].update(probability_correct=np.nan),
])
def test_summary_rejects_extra_duplicate_or_forged_identity(edit):
    args=compact_census();edit(*args)
    with pytest.raises(ValueError):evaluation.summarize(*args)


def test_bootstrap_is_paired_phase_block_and_bounds_complete():
    core=np.linspace(.5,.8,16);edge=core+.1
    result=evaluation.paired_phase_interval(core,edge)
    assert result['central_95_percentage_points']==pytest.approx([10.,10.])
    assert result['seed']==20260917 and result['resamples']==20000
    assert evaluation.accuracy_failure_bounds([None]*16)==dict(lower=0.,upper=1.,known=0,unknown=16,denominator=16)
    bounds=evaluation.accuracy_failure_bounds([.5]*15+[None])
    assert bounds['lower']==.46875 and bounds['upper']==.53125
    with pytest.raises(ValueError):evaluation.paired_phase_interval(core[:15],edge[:15])
    with pytest.raises(ValueError):evaluation.accuracy_failure_bounds([None]*15)


def test_composed_reference_gate_catches_oblique_amplification(reference):
    ctx,p,a,_=reference
    p['pre']['derivatives'][-1,1]=1e8
    a['pre']['derivatives'][-1,1]=1e8
    a['pre']['phi'][-1,0,1]+=1e-8
    sealed=evaluation.seal_reference(ctx,p,a)
    assert sealed['complete'] and not sealed['qualified']
    assert sealed['gates']['scaled_full_tangent']
    assert not sealed['gates']['composed_state_predictions']
    assert not sealed['gates']['composed_activation_predictions']
    assert sealed['local_amplification']['Q_operator_norm']>1e7


@pytest.mark.parametrize('target',['score','comparison','prerequisite'])
def test_incomplete_flags_cannot_produce_qualified_setting(target):
    h,s,c,p=compact_census()
    row=next(iter(s.values())) if target=='score' else c[0] if target=='comparison' else next(iter(p.values()))
    row['complete']=False
    with pytest.raises(ValueError,match='Incomplete'):
        evaluation.summarize(h,s,c,p)


def test_wholly_unavailable_summary_retains_all_denominators():
    result=evaluation.summarize(evaluation.history_census(),{},[],{})
    assert not result['complete'] and not result['qualified']
    assert len(result['scores'])==2700 and len(result['comparisons'])==1404
    assert len(result['balanced_accuracies'])==576
    assert result['primary_failure_bounds']['edge_minus_core_percentage_points']==[-100.,100.]
    assert all(s['means'] is None for s in result['settings'].values())


def test_derivative_clock_has_fixed_physical_flow_orientation(reference):
    sealed=reference[-1]
    for w in sealed['windows'].values():
        observer=w['derivative_observer']
        assert observer['available'] and np.all(observer['h'][::3]>=0)
        assert np.all(observer['h'][1::3]==0) and np.all(observer['h'][2::3]==0)
        assert 'physical activation velocity' in observer['scope']
    task,packet=trace(sealed)
    score=evaluation.score_trace(task,packet,sealed)
    assert score['complete'] and all(w['derivative']['available'] for w in score['windows'].values())
    assert all(w['derivative']['score']['probability_correct']==.5 for w in score['windows'].values())
    coarse,cp=trace(sealed,method='rk4-coarse')
    comparison=evaluation.compare_traces(physical(task),cp,packet,sealed)
    assert comparison['derivative']['available'] and comparison['derivative']['qualified']
    assert 'raw_phi_all_nine' in sealed['reference_comparison']['segments']['pre']


def test_supplied_physical_event_seal_cannot_be_overwritten(reference):
    ctx,p,a,_=reference
    supplied=acquisition.seal_context(ctx,p,rhs=flow)
    p['pre']['derivatives'][-1,1]=1. # z dot f unchanged, Q would differ
    sealed=evaluation.seal_reference(supplied,p,a)
    assert not sealed['complete'] and not sealed['qualified']
    assert 'supplied sealed event_flow' in sealed['error']
