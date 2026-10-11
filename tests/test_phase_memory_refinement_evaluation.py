"""Synthetic arrays only: no target trajectories, integrations, fitting or I/O."""
import copy
from dataclasses import asdict
import math
from pathlib import Path
import sys
import numpy as np
import pytest

_checkout=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(_checkout/'tools'),str(_checkout/'tests')]
import phase_memory_refinement_evaluation as ev
import phase_memory as pm


def case(phase=0,dose=0,sign=1,setting='core',direction='phase_write',sham=False):
    stem=f'{setting}--p{phase}'
    if sham:return dict(id=stem+'--sham',setting=setting,phase_index=phase,kind='sham',dose_index=None,epsilon=None,direction=None,sign=None)
    return dict(id=f'{stem}--d{dose}--{direction}--s{sign:+d}',setting=setting,phase_index=phase,kind='pulse',dose_index=dose,epsilon=(1e-4,1e-3,1e-2)[dose],direction=direction,sign=sign)


def context(c):
    initial=np.zeros(3);delta=np.zeros(3) if c['kind']=='sham' else np.array([c['sign']*c['epsilon'],0.,0.]);period=1.;epoch=c['phase_index']/8
    pulse=dict(initial=initial+delta,requested_delta=delta.copy(),realized_delta=delta.copy(),realized_norm=float(np.linalg.norm(delta)),addition_rounding_residual=np.zeros(3))
    clocks={};predictions={}
    for delay in ev.DELAYS:
        r=pm.ClockReadout(period*(delay-1+(np.arange(32)+.5)/32),np.zeros((32,3)),np.full((32,3),1/96.),96.,epoch,period,delay)
        clocks[str(delay)]=asdict(r);gain=float(np.linalg.norm(r.weights));predictions[str(delay)]=dict(phase_only_q=0.,full_linear_q=0.,zero_response_q=0.,noise_gain=gain)
    return dict(setting=c['setting'],alpha=ev.SETTINGS[c['setting']][0],gamma=ev.SETTINGS[c['setting']][1],phase_initial=initial,phase_epoch=epoch,period=period,pulse=pulse,orientation=1,readouts=clocks,predictions=predictions)


def packet(c,method,ctx=None):
    ctx=ctx or context(c);state=np.broadcast_to(ctx['pulse']['initial'],(12801,3)).copy();solver=dict(ev.METHOD_SPECS[method])
    if solver['kind']=='DOP853':solver['nfev']=0
    return dict(task=ev.task_for(c,method),alpha=ctx['alpha'],gamma=ctx['gamma'],phase_initial=ctx['phase_initial'],phase_epoch=ctx['phase_epoch'],period=ctx['period'],**ctx['pulse'],initial_activation_displacement=np.tanh(ctx['pulse']['initial'])-np.tanh(ctx['phase_initial']),times=ev.TIMES,states=state,solver=solver,status='complete',complete=True,duration=320.,sample_dt=.025)


def fabricated_score(c,m,ctx=None,q=0.):
    ctx=ctx or context(c);rows={}
    for d in ev.DELAYS:
        r=pm.ClockReadout(**ctx['readouts'][str(d)])
        rows[str(d)]=dict(q=q,noise_gain=float(np.linalg.norm(r.weights)),clock_sha256=ev._clock_hash(r),completed_window_end=d*r.period,last_sample_time=float(r.times[-1]),phase_only_q=0.,full_linear_q=0.,zero_response_q=0.,standardized_sham_bias=q/(1e-4*np.linalg.norm(r.weights)) if c['kind']=='sham' else None)
    return dict(task=ev.task_for(c,m),case_id=c['id'],status='complete',method=m,orientation=1,period=ctx['period'],phase_epoch=ctx['phase_epoch'],readouts=rows,error=None)


@pytest.mark.parametrize('method',ev.METHODS)
def test_all_truthful_methods_score_without_any_integration(method):
    c=case();ctx=context(c);p=packet(c,method,ctx);out=ev.score_packet(p,c,ctx)
    assert out['status']=='complete' and out['method']==method and out['task']['method']==method
    assert len(out['readouts'])==4 and p['solver']['id']==method


@pytest.mark.parametrize('field,value',[('id','dop853'),('rtol',1e-12),('atol',1e-14),('max_step',.025),('success',1),('nfev',True),('method','RK4')])
def test_new_adaptive_solver_cannot_be_relabelled_or_relaxed(field,value):
    c=case();ctx=context(c);p=packet(c,ev.NEW_METHODS[-1],ctx);p['solver'][field]=value
    assert ev.score_packet(p,c,ctx)['status']=='failed'


@pytest.mark.parametrize('change',['times','float32','partial','task_sign_bool','phase_epoch','initial','pulse','clock','gain','nan'])
def test_scoring_rejects_changed_context_or_packet(change):
    c=case();ctx=context(c);p=packet(c,ev.NEW_METHODS[0],ctx)
    if change=='times':p['times']=p['times'].copy();p['times'][5]+=.001
    elif change=='float32':p['states']=p['states'].astype(np.float32)
    elif change=='partial':p['complete']=False
    elif change=='task_sign_bool':p['task']['sign']=True
    elif change=='phase_epoch':p['phase_epoch']=.25
    elif change=='initial':p['states'][0,0]+=.1
    elif change=='pulse':p['requested_delta']=p['requested_delta']+np.array([.1,0.,0.])
    elif change=='clock':ctx['readouts']['4']['phase_epoch']=.25
    elif change=='gain':ctx['predictions']['4']['noise_gain']*=2
    else:p['states'][1,0]=np.nan
    assert ev.score_packet(p,c,ctx)['status']=='failed'


def test_original_clock_score_exactly_reproduces_old_evaluator_on_fabricated_arrays():
    import test_phase_memory_runner as f
    import phase_memory_evaluation as old
    reference=f.reference('core');prop=f.propagator(reference,0)
    prepared=old.prepare_reference_outputs({'core':reference},{('core',0):prop})
    task=old.task_census()[0];c={k:v for k,v in task.items() if k!='method'};c['id']=task['id'].rsplit('--',1)[0]
    trace=f.trace(task,reference);trace['solver']['method']=trace['solver']['kind'];pc=prepared['cases'][c['id']];clock=prepared['contexts']['core--p0']
    ctx=dict(setting='core',alpha=reference['alpha'],gamma=reference['gamma'],phase_initial=clock['initial'],phase_epoch=clock['phase_epoch'],period=clock['period'],pulse=pc['pulse'],orientation=pc['orientation'],readouts=clock['readouts'],predictions=pc['readouts'])
    derivative=np.asarray([ev.diagnostics.hopfield_rhs(t,x,trace['alpha'],trace['gamma']) for t,x in zip(trace['times'],trace['states'])])
    earlier=old.score_trace(task,trace,reference,prop,state_derivatives=derivative,prepared=prepared)
    fresh=ev.score_packet(trace,c,ctx)
    assert earlier['status']==fresh['status']=='complete'
    for d in ev.DELAYS:
        for k in ('q','noise_gain','phase_only_q','full_linear_q'):
            assert earlier['readouts'][str(d)][k]==fresh['readouts'][str(d)][k]


def pair(c=None):
    c=c or case();a,b=ev.NEW_METHODS[:2];ctx=context(c)
    spec=next(r for r in ev.comparison_specs([c]) if r['pair']==[a,b])
    return spec,packet(c,a,ctx),packet(c,b,ctx),fabricated_score(c,a,ctx),fabricated_score(c,b,ctx)


def test_full_and_late_rms_max_threshold_crossings_and_final_state():
    spec,left,right,a,b=pair(case(sham=True))
    right['states'][10,0]=1e-6;right['states'][20,1]=np.nextafter(1e-6,np.inf)
    right['states'][9600,2]=2e-6;right['states'][-1,2]=3e-6
    row=ev.compare_pair(spec,left,right,a,b)
    assert row['complete'] and not row['qualified']
    assert row['state_difference']['first_threshold_crossing_time']==.5
    assert row['activation_difference']['first_threshold_crossing_time']==.25
    assert row['state_difference']['maximum_absolute']==3e-6 and row['state_difference']['maximum_time']==320.
    assert row['late_window']['samples']==3201 and row['late_window']['start']==240.
    assert row['late_window']['state_difference']['first_threshold_crossing_time']==240.
    assert row['last_state_difference']==[0.,0.,-3e-6]
    assert row['state_difference']['rms']==pytest.approx(np.sqrt(np.mean((left['states']-right['states'])**2)),rel=1e-15)


def test_readout_crossing_uses_declared_completed_windows_only():
    spec,left,right,a,b=pair();gain=a['readouts']['1']['noise_gain']
    b['readouts']['2']['q']=.009*ev.PRIMARY_NOISE*gain
    b['readouts']['4']['q']=.011*ev.PRIMARY_NOISE*gain
    row=ev.compare_pair(spec,left,right,a,b)
    assert row['complete'] and not row['qualified']
    assert row['first_completed_readout_threshold_crossing']==dict(delay_cycles=4,completed_window_end=4.,last_sample_time=3+31.5/32)
    assert row['state_pass'] and row['activation_pass']


@pytest.mark.parametrize('change',['changed_clock','changed_gain','wrong_case','wrong_order','missing_score','changed_pulse'])
def test_comparator_preserves_failure_for_identity_or_missingness(change):
    spec,left,right,a,b=pair()
    if change=='changed_clock':b['readouts']['1']['clock_sha256']='f'*64
    elif change=='changed_gain':b['readouts']['1']['noise_gain']*=2
    elif change=='wrong_case':right['task']['phase_index']=1
    elif change=='wrong_order':spec['pair'].reverse()
    elif change=='missing_score':b={}
    else:right['requested_delta']=right['requested_delta']+np.array([0.,.01,0.])
    row=ev.compare_pair(spec,left,right,a,b)
    assert not row['complete'] and not row['qualified'] and row['error']


def census():
    binary=[('core',3,'phase_write'),('core',3,'neuron_one'),('core',7,'phase_write'),('core',7,'neuron_one'),('failed-near-edge',2,'phase_write'),('failed-near-edge',2,'neuron_one'),('failed-near-edge',3,'phase_write'),('failed-near-edge',3,'neuron_one'),('failed-near-edge',4,'phase_write'),('failed-near-edge',5,'phase_write'),('failed-near-edge',5,'neuron_one'),('failed-near-edge',6,'phase_write'),('failed-near-edge',6,'neuron_one')]
    rows=[case(phase,dose,sign,setting,direction) for setting,phase,direction in binary for dose in range(3) for sign in (-1,1)]
    rows += [case(phase=phase,setting=setting,sham=True) for setting,phase in sorted({(x[0],x[1]) for x in binary})]
    return rows


def test_all_pairs_and_failure_denominators_are_retained_without_any_waveforms():
    rows=census();out=ev.summarize(rows,{},{});
    assert out['counts']==dict(physical_cases=85,methods=8,trace_rows=680,pair_rows=2380,same_method_ba_cells=3744,required_new_traces=425,required_old_traces=255)
    assert out['comparison_groups']=={'old-old':dict(required=255,complete=0,passed=0),'old-new':dict(required=1275,complete=0,passed=0),'new-new':dict(required=850,complete=0,passed=0)}
    assert not out['complete'] and not out['original_study_requalified'] and out['setting_means'] is None
    assert all(r['status']=='failed' for r in out['terminal_rows']+out['paired_readouts'])


def test_summary_rejects_omission_duplicate_extra_or_wrong_signed_contexts():
    rows=census()
    for changed in (rows[:-1],rows[:-1]+[rows[0]],rows+[rows[0]]):
        with pytest.raises(ValueError):ev.summarize(changed,{}, {})
    with pytest.raises(ValueError):ev.summarize(rows,{'unplanned':{}},{})


def test_gaussian_accuracy_keeps_below_chance_and_no_extra_sqrt_sample_factor():
    assert ev._accuracy(0.,0.,.2,1,1e-4)[0]==.5
    actual,means=ev._accuracy(-1e-5,1e-5,.2,1,1e-4)
    expected=.5*(1+math.erf(-.5/math.sqrt(2)))
    assert actual==pytest.approx(expected,abs=1e-16) and actual<.5 and means==[-.5,-.5]
    assert ev._accuracy(-1e-5,1e-5,.2,-1,1e-4)[0]>.5


def test_full_synthetic_summary_and_one_missing_packet_keep_every_cell():
    cases=census();packets={};scores={}
    for c in cases:
        ctx=context(c);base=packet(c,ev.METHODS[0],ctx)
        for method in ev.METHODS:
            solver=dict(ev.METHOD_SPECS[method])
            if solver['kind']=='DOP853':solver['nfev']=0
            name=ev.task_for(c,method)['id']
            packets[name]=dict(base,task=ev.task_for(c,method),solver=solver)
            scores[name]=fabricated_score(c,method,ctx,q=0. if c['kind']=='sham' else -c['sign']*1e-5)
    full=ev.summarize(cases,packets,scores)
    assert full['complete'] and all(r['qualified'] for r in full['comparisons'])
    assert all(r['score']==scores[r['task']['id']] for r in full['terminal_rows'])
    assert all(len(r['score']['readouts'])==4 for r in full['terminal_rows'] if r['task']['kind']=='sham')
    assert all(r['status']=='complete' and r['balanced_accuracy']<.5 for r in full['paired_readouts'])
    assert full['setting_means'] is None and full['original_study_requalified'] is False
    missing=ev.task_for(cases[0],ev.NEW_METHODS[-1])['id'];del packets[missing]
    partial=ev.summarize(cases,packets,scores)
    assert not partial['complete'] and partial['counts']==full['counts']
    assert sum(r['status']=='failed' for r in partial['terminal_rows'])==1
    assert sum(not r['complete'] for r in partial['comparisons'])==7
    assert sum(r['status']=='failed' for r in partial['paired_readouts'])==12


def test_score_boolean_task_substitution_cannot_hide_in_numeric_equality():
    spec,left,right,a,b=pair();b['task']['sign']=True
    assert ev.compare_pair(spec,left,right,a,b)['complete'] is False


def test_failed_partial_score_payload_is_not_discarded():
    cases=census();c=cases[0];method=ev.NEW_METHODS[0];name=ev.task_for(c,method)['id']
    partial=dict(task=ev.task_for(c,method),case_id=c['id'],status='failed',readouts={'1':{'q':.125}},error='synthetic later readout failure')
    result=ev.summarize(cases,{name:packet(c,method)},{name:partial})
    row=next(r for r in result['terminal_rows'] if r['task']['id']==name)
    assert row['status']=='failed' and row['score']==partial
