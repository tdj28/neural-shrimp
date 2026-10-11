"""Pure synthetic oscillator/codec tests; no neural integration or fitting."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import phase_memory as pm
import phase_memory_evaluation as ev
from run_hopfield import safe


def synthetic_inputs():
    """Exact analytic circle, radial decay1 and third-axis decay2, period4."""
    period=4.; w=2*np.pi/period
    t=np.linspace(0,period,4097); a=w*t
    x=np.column_stack((np.cos(a),np.sin(a),np.zeros(len(t))))
    dx=np.column_stack((-w*np.sin(a),w*np.cos(a),np.zeros(len(t))))
    z=dx/(w*w); dz=np.column_stack((-np.cos(a),-np.sin(a),np.zeros(len(t))))
    references={};propagators={}
    for setting,(alpha,gamma) in ev.SETTINGS.items():
        phases=[]
        for phase in range(8):
            j=512*phase
            phases.append(dict(phase_index=phase,phase_epoch=float(t[j]),initial=x[j].copy(),
                flow=dx[j].copy(),phase_gradient=z[j].copy(),**pm.pulse_directions(z[j],dx[j])))
            theta=a+phase*period/8*w
            rotation=np.zeros((len(t),3,3));rotation[:,0,0]=-np.sin(theta);rotation[:,1,0]=np.cos(theta)
            rotation[:,0,1]=np.cos(theta);rotation[:,1,1]=np.sin(theta);rotation[:,2,2]=1.
            drotation=np.zeros_like(rotation);drotation[:,0,0]=-w*np.cos(theta);drotation[:,1,0]=-w*np.sin(theta)
            drotation[:,0,1]=-w*np.sin(theta);drotation[:,1,1]=w*np.cos(theta)
            decay=np.column_stack((np.ones(len(t)),np.exp(-t),np.exp(-2*t)))
            ddecay=decay*np.array([0.,-1.,-2.])
            basis0=rotation[0].copy()
            phi=np.einsum('nij,nj,jk->nik',rotation,decay,basis0.T)
            dphi=np.einsum('nij,nj,jk->nik',drotation,decay,basis0.T)+np.einsum('nij,nj,jk->nik',rotation,ddecay,basis0.T)
            phi[0]=np.eye(3)
            propagators[(setting,phase)]=dict(setting=setting,phase_index=phase,period=period,
                initial=x[j].copy(),status='complete',qualified=True,gates={'synthetic':True},
                times=t.copy(),phi=phi,phi_derivatives=dphi,monodromy=phi[-1].copy())
        references[setting]=dict(setting=setting,alpha=alpha,gamma=gamma,period=period,status='complete',qualified=True,
            gates={'synthetic':True},orbit=dict(times=t.copy(),states=x.copy(),derivatives=dx.copy()),
            adjoint=dict(times=t.copy(),values=z.copy(),derivatives=dz.copy()),phases=phases)
    return references,propagators


def analytic_trace(task,reference):
    phase=reference['phases'][task['phase_index']];pulse=ev._pulse(task,phase)
    initial=pulse['initial'];t=ev.TIMES;w=2*np.pi/reference['period']
    r0=np.hypot(initial[0],initial[1]);angle0=np.arctan2(initial[1],initial[0]);radius=1+(r0-1)*np.exp(-t)
    dradius=-(r0-1)*np.exp(-t);theta=angle0+w*t
    state=np.column_stack((radius*np.cos(theta),radius*np.sin(theta),initial[2]*np.exp(-2*t)))
    derivative=np.column_stack((dradius*np.cos(theta)-w*radius*np.sin(theta),dradius*np.sin(theta)+w*radius*np.cos(theta),-2*initial[2]*np.exp(-2*t)))
    state[0]=initial
    if task['method']=='dop853':solver=dict(id='dop853',kind='DOP853',rtol=1e-12,atol=1e-14,max_step=.025,success=True,nfev=0)
    else:
        dt=float(task['method'].split('-')[1]);solver=dict(id=task['method'],kind='RK4',dt=dt,steps=round(320/dt))
    trace=dict(task=task.copy(),alpha=reference['alpha'],gamma=reference['gamma'],phase_initial=phase['initial'].copy(),
        **pulse,initial_activation_displacement=np.tanh(initial)-np.tanh(phase['initial']),times=t.copy(),states=state,
        status='complete',complete=True,solver=solver)
    return trace,derivative


def roundtrip(value,path):
    """Real safe JSON metadata + compressed NPZ, exact member/hash checks."""
    arrays={}
    def encode(item):
        if isinstance(item,np.ndarray):
            key=f'a{len(arrays)}';arrays[key]=item;return {'__ndarray__':key}
        if isinstance(item,dict):return {str(k):encode(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [encode(v) for v in item]
        return safe(item)
    metadata=encode(value);npz=path.with_suffix('.npz');meta=path.with_suffix('.json')
    np.savez_compressed(npz,**arrays)
    meta.write_text(json.dumps({'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'packet':metadata},allow_nan=False))
    saved=json.loads(meta.read_text());assert saved['npz_sha256']==hashlib.sha256(npz.read_bytes()).hexdigest()
    with np.load(npz,allow_pickle=False) as loaded:
        assert set(loaded.files)==set(arrays)
        def decode(item):
            if isinstance(item,dict):
                if set(item)=={'__ndarray__'}:return loaded[item['__ndarray__']].copy()
                return {k:decode(v) for k,v in item.items()}
            if isinstance(item,list):return [decode(v) for v in item]
            return item
        return decode(saved['packet'])


@pytest.fixture(scope='module')
def inputs():return synthetic_inputs()


@pytest.fixture(scope='module')
def prepared(inputs):return ev.prepare_reference_outputs(*inputs)


def test_census_and_prep_is_result_free_complete(inputs,prepared,monkeypatch):
    assert len(ev.task_census())==912 and len(set(x['id'] for x in ev.task_census()))==912
    assert prepared['qualified'] and len(prepared['contexts'])==16 and len(prepared['cases'])==304
    assert len(prepared['paired_nulls'])==1728
    assert all(x['status']=='complete' for x in prepared['paired_nulls'])
    assert all(x['diagnostic_times'][-1]<=8*x['period'] for x in prepared['contexts'].values())
    monkeypatch.setattr(ev,'score_trace',lambda *a,**k:pytest.fail('Preparation accessed target scorer'))
    replay=ev.replay_prepared_reference_outputs(prepared,*inputs)
    assert replay['match'] and replay['qualified']


def test_missing_reference_preserves_census_and_other_setting(inputs):
    references,props=inputs;reduced={k:v for k,v in references.items() if k=='core'}
    result=ev.prepare_reference_outputs(reduced,props)
    assert not result['qualified'] and len(result['contexts'])==16 and len(result['cases'])==304
    assert sum(r['status']=='complete' for r in result['cases'].values())==152
    assert sum(r['status']=='complete' for r in result['paired_nulls'])==864


@pytest.mark.parametrize('kind',['short','duplicate','wrongtype','wrongorder'])
def test_preparation_rejects_changed_task_plan(inputs,kind):
    tasks=ev.task_census()
    if kind=='short':tasks.pop()
    elif kind=='duplicate':tasks[-1]=tasks[0]
    elif kind=='wrongtype':tasks[0]['phase_index']=False
    else:tasks.reverse()
    with pytest.raises(ValueError):ev.prepare_reference_outputs(*inputs,tasks)


def test_requested_phase_null_and_realized_linear_bound_before_target(inputs,prepared):
    for case in prepared['cases'].values():
        if case['status']!='complete':continue
        context=prepared['contexts'][case['context_id']]
        assert case['requested_time_displacement']==float(context['phase_gradient']@case['pulse']['requested_delta'])
        assert case['realized_time_displacement']==float(context['phase_gradient']@case['pulse']['realized_delta'])
        assert case['pulse_used_for_phase_only']=='requested_delta'
        assert case['pulse_used_for_full_linear']=='realized_delta'
        if case['case']['direction']=='phase_null':assert abs(case['requested_time_displacement'])<1e-16


@pytest.mark.parametrize('mutation',['weight','null','linear','float32','nan','identity'])
def test_changed_prepared_packet_fails_before_trace(inputs,prepared,mutation):
    task=ev.task_census()[0];cid=ev._context_id(task);caseid=ev._case_id(task)
    changed={'schema_version':prepared['schema_version'],'required_counts':prepared['required_counts'],
        'contexts':{cid:copy.deepcopy(prepared['contexts'][cid])},'cases':{caseid:copy.deepcopy(prepared['cases'][caseid])}}
    if mutation=='weight':changed['contexts'][cid]['readouts']['1']['weights'][0,0]+=1e-8
    elif mutation=='null':changed['cases'][caseid]['readouts']['1']['phase_only_q']+=1e-8
    elif mutation=='linear':changed['cases'][caseid]['full_linear_state_delta'][3,0]+=1e-8
    elif mutation=='float32':changed['contexts'][cid]['initial']=changed['contexts'][cid]['initial'].astype(np.float32)
    elif mutation=='nan':changed['cases'][caseid]['orientation']=np.nan
    else:changed['cases'][caseid]['case']['phase_index']=True
    result=ev.score_trace(task,{},inputs[0][task['setting']],inputs[1][(task['setting'],task['phase_index'])],state_derivatives=None,prepared=changed)
    assert result['status']=='failed' and 'Prepared reference replay differs' in result['error']


@pytest.mark.parametrize('mutation',['truncated','wronggrid','float32','nan','booltask','solver','rounding','complete'])
def test_invalid_trace_retained_as_failed(inputs,prepared,mutation):
    task=ev.task_census()[0];ref=inputs[0][task['setting']];trace,derivative=analytic_trace(task,ref)
    if mutation=='truncated':trace['states']=trace['states'][:-1]
    elif mutation=='wronggrid':trace['times'][3]+=.001
    elif mutation=='float32':trace['states']=trace['states'].astype(np.float32)
    elif mutation=='nan':trace['states'][4,0]=np.nan
    elif mutation=='booltask':trace['task']['phase_index']=False
    elif mutation=='solver':trace['solver']['steps']+=1
    elif mutation=='rounding':trace['realized_delta'][0]+=1e-15
    else:trace['complete']=False
    result=ev.score_trace(task,trace,ref,inputs[1][(task['setting'],task['phase_index'])],state_derivatives=derivative,prepared=prepared)
    assert result['status']=='failed' and result['error']


def test_safe_json_npz_roundtrip_prepared_trace_score_evaluate(inputs,prepared,tmp_path):
    # Simulates saved JSON's normalization of numpy string/int scalar metadata.
    task=ev.task_census()[0];changed=copy.deepcopy(prepared)
    changed['tasks'][0]['method']=np.str_(task['method'])
    restored=roundtrip(changed,tmp_path/'prepared')
    assert ev.replay_prepared_reference_outputs(restored,*inputs)['match']
    ref=inputs[0][task['setting']];trace,d=analytic_trace(task,ref)
    trace['task']['kind']=np.str_('pulse');trace['task']['phase_index']=np.int64(0)
    trace=roundtrip(trace,tmp_path/'trace')
    score=ev.score_trace(task,trace,ref,inputs[1][('core',0)],state_derivatives=d,prepared=restored)
    assert score['status']=='complete',score['error']
    savedscore=roundtrip(score,tmp_path/'score')
    rescored=ev.score_trace(task,trace,ref,inputs[1][('core',0)],state_derivatives=d,prepared=restored)
    assert not ev.packet_differences(savedscore,rescored)
    result=ev.summarize([savedscore],[],prerequisites={s:dict(reference=True,propagators=[True]*8) for s in ev.SETTINGS})
    assert not result['qualified'] and len(result['terminal_rows'])==912
    assert len(result['paired_readouts'])==5184 and len(result['balanced_accuracy_comparisons'])==3456
    assert all(s['aggregates'] is None for s in result['settings'].values())


def test_packet_replay_normalizes_metadata_not_array_dtypes():
    assert not ev.packet_differences({'s':np.str_('ok'),'n':np.int64(1)},{'s':'ok','n':1})
    assert ev.packet_differences({'n':True},{'n':1})
    for value in (np.nan,np.inf,-np.inf):assert ev.packet_differences({'x':[value]},{'x':[value]})
    assert ev.packet_differences(np.zeros(3,dtype=np.float32),np.zeros(3))
    assert ev.packet_differences(np.array([-0.]),np.array([0.]))


@pytest.fixture(scope='module')
def scored_cohort(inputs,prepared):
    scores=[];comparisons=[];tasks=ev.task_census()
    for start in range(0,len(tasks),3):
        rows=tasks[start:start+3];packets=[];scored=[]
        for task in rows:
            ref=inputs[0][task['setting']];trace,derivative=analytic_trace(task,ref)
            score=ev.score_trace(task,trace,ref,inputs[1][(task['setting'],task['phase_index'])],state_derivatives=derivative,prepared=prepared)
            assert score['status']=='complete',score['error']
            packets.append(trace);scored.append(score);scores.append(score)
        for a,b in ((0,1),(1,2)):
            comparisons.append(ev.compare_trace_pair(rows[a],packets[a],scored[a],rows[b],packets[b],scored[b]))
    return scores,comparisons


def prerequisites():return {s:dict(reference=True,propagators=[True]*8) for s in ev.SETTINGS}


def test_full_synthetic_path_preserves_all_delays_methods_noise_denominators(scored_cohort):
    result=ev.summarize(*scored_cohort,prerequisites=prerequisites())
    assert result['qualified'],{s:v for s,v in result['settings'].items()}
    assert result['required_counts']==dict(trace_rows=912,trace_readout_scores=3648,pulse_readout_scores=3456,sham_readout_scores=192,trace_comparisons=608,trace_comparisons_per_setting=304,paired_readout_cells=5184,method_independent_null_cells=1728,signed_response_pairs=432,signed_readout_cells=1728,balanced_accuracy_comparisons=3456)
    assert len(result['trace_comparisons'])==608
    assert sum(c['status']=='complete' for c in result['paired_readouts'])==5184
    for setting in result['settings'].values():
        assert len(setting['aggregates'])==36
        assert sum(row['primary'] for row in setting['aggregates'])==1
        assert all(row['cells_per_direction']==24 for row in setting['aggregates'])
    for cell in result['paired_readouts']:
        assert cell['zero_balanced_accuracy']==.5 and 0<=cell['balanced_accuracy']<=1


@pytest.mark.parametrize('failure',['missing','duplicate','nan','sham','prerequisite','numeric','ba','booltask','negative_difference'])
def test_fail_closed_means_and_no_sham_recentering(scored_cohort,failure):
    scores,comparisons=copy.deepcopy(scored_cohort);prereq=prerequisites()
    target=next(s for s in scores if s['task']['setting']=='core' and s['task']['kind']=='sham')
    original_q=scores[0]['readouts']['4']['q']
    if failure=='missing':scores.pop(0)
    elif failure=='duplicate':scores.append(scores[0])
    elif failure=='nan':scores[0]['readouts']['4']['q']=np.nan
    elif failure=='sham':target['readouts']['8']['q']=1.
    elif failure=='prerequisite':prereq['core']['propagators'][3]=False
    elif failure=='numeric':comparisons[0]['state_difference']['maximum_absolute']=1.01e-6
    elif failure=='ba':scores[0]['readouts']['1']['q']=-1.
    elif failure=='booltask':scores[0]['task']['phase_index']=False
    else:comparisons[0]['state_difference']['maximum_absolute']=-1.
    result=ev.summarize(scores,comparisons,prerequisites=prereq)
    assert not result['qualified'] and not result['settings']['core']['qualified']
    assert result['settings']['core']['aggregates'] is None
    assert result['settings']['failed-near-edge']['qualified']
    assert len(result['terminal_rows'])==912 and len(result['paired_readouts'])==5184
    if failure=='sham':assert scores[0]['readouts']['4']['q']==original_q


def test_exact_gaussian_accuracy_and_below_chance_no_refit():
    from scipy.special import ndtr
    assert ev._accuracy(.2,-.1,2.,1,.1)==float((ndtr(1)+ndtr(.5))/2)
    assert ev._accuracy(-.2,.1,2.,1,.1)<.5
    assert ev._accuracy(0.,0.,2.,1,.1)==.5


def test_comparison_mean_recomputed_even_when_ba_saturated(scored_cohort):
    scores,comparisons=copy.deepcopy(scored_cohort)
    scores[0]['readouts']['1']['q']+=1e-8
    result=ev.summarize(scores,comparisons,prerequisites=prerequisites())
    assert not result['trace_comparisons'][0]['qualified']
    assert not result['settings']['core']['qualified']
    assert result['settings']['failed-near-edge']['qualified']


def test_unknown_unattributable_extra_record_blocks_complete_study(scored_cohort):
    scores,comparisons=copy.deepcopy(scored_cohort)
    scores.append({'task':None})
    result=ev.summarize(scores,comparisons,prerequisites=prerequisites())
    assert not result['qualified'] and len(result['invalid_input_records']['scored'])==1
    assert len(result['terminal_rows'])==912


def test_odd_even_signed_diagnostics_include_reference_not_observed_sham(inputs,prepared):
    minus=ev.task_census()[0];plus=ev.task_census()[3]
    reference=inputs[0]['core'];prop=inputs[1][('core',0)]
    tm,dm=analytic_trace(minus,reference);tp,dp=analytic_trace(plus,reference)
    sm=ev.score_trace(minus,tm,reference,prop,state_derivatives=dm,prepared=prepared)
    sp=ev.score_trace(plus,tp,reference,prop,state_derivatives=dp,prepared=prepared)
    row=ev.paired_response_diagnostics(plus,tp,sp,minus,tm,sm,prepared=prepared)
    assert row['status']=='complete' and row['samples']==1281
    assert row['state_odd']['maximum_absolute']>0
    for delay in ('1','2','4','8'):
        assert row['readouts'][delay]['q_odd']==(sp['readouts'][delay]['q']-sm['readouts'][delay]['q'])/2
        assert row['readouts'][delay]['q_even']==(sp['readouts'][delay]['q']+sm['readouts'][delay]['q'])/2
    tm['complete']=False
    failed=ev.paired_response_diagnostics(plus,tp,sp,minus,tm,sm,prepared=prepared)
    assert failed['status']=='failed' and failed['ids']==[plus['id'],minus['id']]


def test_odd_even_readout_census_and_unavailable_rows(scored_cohort):
    scores,comparisons=scored_cohort
    result=ev.summarize(scores,comparisons,prerequisites=prerequisites())
    assert len(result['signed_response_readouts'])==1728
    assert all(r['status']=='complete' for r in result['signed_response_readouts'])
    failed=ev.summarize(scores[1:],comparisons,prerequisites=prerequisites())
    assert len(failed['signed_response_readouts'])==1728
    assert sum(r['status']=='unavailable' for r in failed['signed_response_readouts'])==4
