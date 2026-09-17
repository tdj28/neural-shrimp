"""Pure saved-array diagnostics for the post-EXP014 numerical refinement.

No integration, file access, fitting, clock re-estimation, phase alignment or
method relabeling. The caller binds each compact context to the immutable
original reference seal and preserves original study qualification unchanged.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import itertools
import json
import numpy as np
from scipy.special import ndtr
import hopfield_diagnostics as diagnostics
import phase_memory as pm

SETTINGS={'core':(1.064,10.260722222222222),'failed-near-edge':(1.0661333333333334,10.297166666666667)}
OLD_METHODS=('rk4-0.0025','rk4-0.00125','dop853')
NEW_METHODS=('rk4-0.000625','rk4-0.0003125','rk4-0.00015625','dop853-1e-13-h0.0125','dop853-1e-13-h0.00625')
METHODS=OLD_METHODS+NEW_METHODS
METHOD_SPECS={m:dict(id=m,kind='RK4',dt=float(m[4:]),steps=round(320/float(m[4:]))) for m in METHODS if m.startswith('rk4-')}
METHOD_SPECS.update({'dop853':dict(id='dop853',kind='DOP853',rtol=1e-12,atol=1e-14,max_step=.025,success=True),
    **{m:dict(id=m,kind='DOP853',rtol=1e-13,atol=1e-15,max_step=h,success=True) for m,h in zip(NEW_METHODS[-2:],(.0125,.00625))}})
for _spec in METHOD_SPECS.values():_spec['method']=_spec['kind']
DELAYS=(1,2,4,8)
NOISES=(1e-5,1e-4,1e-3)
PRIMARY_NOISE=1e-4
TIMES=np.arange(12801,dtype=np.float64)*.025
TIMES[-1]=320.;TIMES.setflags(write=False)
CASE_KEYS={'id','setting','phase_index','kind','dose_index','epsilon','direction','sign'}


def validate_case(case):
    if type(case) is not dict or set(case)!=CASE_KEYS:raise ValueError('Exact physical case fields required')
    if type(case['setting']) is not str or case['setting'] not in SETTINGS or type(case['phase_index']) is not int or not 0<=case['phase_index']<8:
        raise ValueError('Fixed setting and integer phase required')
    stem=f"{case['setting']}--p{case['phase_index']}"
    if case['kind']=='sham':
        expected=stem+'--sham'
        if any(case[k] is not None for k in ('dose_index','epsilon','direction','sign')):raise ValueError('Sham metadata differs')
    elif case['kind']=='pulse':
        d=case['dose_index'];sgn=case['sign']
        if type(d) is not int or d not in (0,1,2) or type(sgn) is not int or sgn not in (-1,1):raise ValueError('Integer dose/sign required')
        if type(case['epsilon']) is not float or case['epsilon']!=(1e-4,1e-3,1e-2)[d] or case['direction'] not in ('phase_write','phase_null','neuron_one'):
            raise ValueError('Pulse dose/direction differs')
        expected=f"{stem}--d{d}--{case['direction']}--s{sgn:+d}"
    else:raise ValueError('Unknown physical case kind')
    if type(case['id']) is not str or case['id']!=expected:raise ValueError('Physical case ID differs')
    return case


def task_for(case,method):
    validate_case(case)
    if type(method) is not str or method not in METHODS:raise ValueError('Unknown numerical method')
    return dict(case,id=case['id']+'--'+method,method=method)


def _case_task(task):
    if type(task) is not dict or set(task)!=CASE_KEYS|{'method'}:raise ValueError('Exact method task required')
    method=task['method']
    if type(method) is not str or method not in METHODS or not task['id'].endswith('--'+method):raise ValueError('Method/task identity differs')
    case={k:v for k,v in task.items() if k!='method'};case['id']=task['id'][:-(len(method)+2)]
    if task!=task_for(case,method) or any(type(task[k]) is not type(task_for(case,method)[k]) for k in task):raise ValueError('Typed task identity differs')
    return case,method


def validate_packet(packet,case):
    validate_case(case);actual,method=_case_task(packet['task'])
    if actual!=case or any(type(actual[k]) is not type(case[k]) for k in case):raise ValueError('Packet physical case differs')
    if packet.get('status')!='complete' or packet.get('complete') is not True:raise ValueError('Packet incomplete')
    for key,want in [('duration',320.),('sample_dt',.025)]:
        if type(packet.get(key)) is not float or packet[key]!=want:raise ValueError('Duration/sample spacing differs')
    if tuple(packet[k] for k in ('alpha','gamma'))!=SETTINGS[case['setting']]:raise ValueError('Packet system parameters differ')
    t=pm._array(packet['times'],'trace times',TIMES.shape);x=pm._array(packet['states'],'trace states',(12801,3))
    if not np.array_equal(t,TIMES):raise ValueError('Exact full-duration saved grid required')
    initial=pm._array(packet['initial'],'trace initial',(3,))
    if not np.array_equal(x[0],initial):raise ValueError('Trace initial sample differs')
    solver=packet['solver']
    for key,want in METHOD_SPECS[method].items():
        if type(solver.get(key)) is not type(want) or solver[key]!=want:raise ValueError('Truthful solver metadata required: '+key)
    if solver['kind']=='DOP853' and (type(solver.get('nfev')) is not int or solver['nfev']<0):raise ValueError('Independent solver work count required')
    return x,method


def _clock_hash(readout):
    meta={k:getattr(readout,k) for k in ('derivative_energy','phase_epoch','period','delay_cycles')}
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
    for key in ('times','reference_activations','weights'):
        a=getattr(readout,key);h.update(key.encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def _context(case,context):
    if context['setting']!=case['setting'] or tuple(context[k] for k in ('alpha','gamma'))!=SETTINGS[case['setting']]:raise ValueError('Clock context setting differs')
    period=pm._scalar(context['period'],'period',positive=True);epoch=pm._scalar(context['phase_epoch'],'phase epoch')
    if 8*period>320. or epoch!=case['phase_index']*period/8:raise ValueError('Clock epoch/horizon differs')
    initial=pm._array(context['phase_initial'],'phase initial',(3,));pulse=context['pulse']
    requested=pm._array(pulse['requested_delta'],'requested pulse',(3,));actual_initial=pm._array(pulse['initial'],'pulse initial',(3,))
    realized=actual_initial-initial
    if not np.array_equal(actual_initial,initial+requested) or not np.array_equal(pm._array(pulse['realized_delta'],'realized pulse',(3,)),realized):raise ValueError('Exact pulse realization differs')
    if not np.array_equal(pm._array(pulse['addition_rounding_residual'],'pulse rounding',(3,)),realized-requested) or pm._scalar(pulse['realized_norm'],'realized norm')!=float(np.linalg.norm(realized)):
        raise ValueError('Pulse rounding/norm differs')
    if case['kind']=='sham':
        if not np.array_equal(requested,np.zeros(3)):raise ValueError('Nonzero sham')
    elif not np.isclose(np.linalg.norm(requested),case['epsilon'],rtol=1e-12,atol=0):raise ValueError('Requested pulse dose differs')
    orientation=context['orientation']
    if type(orientation) is not int or orientation not in (-1,1) or (case['kind']=='sham' or case['direction']=='phase_null') and orientation!=1:
        raise ValueError('Fixed classifier orientation differs')
    if set(context['readouts'])!={str(d) for d in DELAYS} or set(context['predictions'])!={str(d) for d in DELAYS}:raise ValueError('All four sealed readouts required')
    result={}
    for delay in DELAYS:
        readout=pm.ClockReadout(**context['readouts'][str(delay)])
        if readout.period!=period or readout.phase_epoch!=epoch or readout.delay_cycles!=delay:raise ValueError('Clock identity differs')
        prediction=context['predictions'][str(delay)];gain=float(np.linalg.norm(readout.weights))
        if pm._scalar(prediction['noise_gain'],'noise gain',positive=True)!=gain:raise ValueError('Sealed noise gain differs')
        for name in ('phase_only_q','full_linear_q'):pm._scalar(prediction[name],name)
        if prediction['zero_response_q']!=0.:raise ValueError('Nonzero zero-response prediction')
        result[str(delay)]=(readout,gain,prediction)
    return result


def score_packet(packet,case,context):
    """Score original or genuinely named new solver on exactly the old clock."""
    result=dict(task=packet.get('task'),case_id=case.get('id'),status='failed',readouts={},error=None)
    try:
        validate_case(case);state,method=validate_packet(packet,case);clocks=_context(case,context)
        for name,want in context['pulse'].items():
            if name=='realized_norm':
                if packet[name]!=want:raise ValueError('Trace pulse norm differs')
            elif not np.array_equal(pm._array(packet[name],name,(3,)),want):raise ValueError('Trace pulse differs: '+name)
        if not np.array_equal(packet['phase_initial'],context['phase_initial']) or packet['phase_epoch']!=context['phase_epoch'] or packet['period']!=context['period']:
            raise ValueError('Trace phase identity differs')
        if not np.array_equal(packet['initial_activation_displacement'],np.tanh(packet['initial'])-np.tanh(context['phase_initial'])):raise ValueError('Initial activation displacement differs')
        derivative=np.asarray([diagnostics.hopfield_rhs(t,x,packet['alpha'],packet['gamma']) for t,x in zip(TIMES,state,strict=True)],dtype=np.float64)
        segment=pm.HermiteSegment(TIMES,state,derivative)
        result.update(method=method,orientation=context['orientation'],period=context['period'],phase_epoch=context['phase_epoch'])
        for delay in DELAYS:
            readout,gain,prediction=clocks[str(delay)];q=readout.score(np.tanh(segment.evaluate(readout.times)))
            result['readouts'][str(delay)]=dict(q=q,noise_gain=gain,clock_sha256=_clock_hash(readout),
                completed_window_end=delay*readout.period,last_sample_time=float(readout.times[-1]),
                phase_only_q=prediction['phase_only_q'],full_linear_q=prediction['full_linear_q'],zero_response_q=0.,
                standardized_sham_bias=q/(PRIMARY_NOISE*gain) if case['kind']=='sham' else None)
        result['status']='complete'
    except (ValueError,KeyError,TypeError,AttributeError,FloatingPointError) as error:result['error']=repr(error)
    return result


def _score_identity(score,case,method):
    actual,actual_method=_case_task(score['task'])
    if score.get('status')!='complete' or actual!=case or actual_method!=method or score.get('case_id')!=case['id'] or score.get('method')!=method:raise ValueError('Missing/changed complete score identity')
    if set(score['readouts'])!={str(d) for d in DELAYS}:raise ValueError('Complete score readout census required')
    if type(score['orientation']) is not int or score['orientation'] not in (-1,1):raise ValueError('Score orientation differs')
    period=pm._scalar(score['period'],'score period',positive=True)
    if pm._scalar(score['phase_epoch'],'score epoch')!=case['phase_index']*period/8 or 8*period>320:raise ValueError('Score clock epoch differs')
    for delay in DELAYS:
        row=score['readouts'][str(delay)]
        for k in ('q','phase_only_q','full_linear_q'):pm._scalar(row[k],k)
        pm._scalar(row['noise_gain'],'noise gain',positive=True)
        if type(row['clock_sha256']) is not str or len(row['clock_sha256'])!=64:raise ValueError('Missing sealed clock binding')
        if row['completed_window_end']!=delay*period or row['last_sample_time']!=period*(delay-1+31.5/32):raise ValueError('Readout time identity differs')
    return score


def comparison_specs(cases):
    rows=[];seen=set()
    for case in cases:
        validate_case(case)
        if case['id'] in seen:raise ValueError('Duplicate physical case')
        seen.add(case['id'])
        for left,right in itertools.combinations(METHODS,2):
            group='old-old' if left in OLD_METHODS and right in OLD_METHODS else ('new-new' if left in NEW_METHODS and right in NEW_METHODS else 'old-new')
            rows.append(dict(id=case['id']+'--'+left+'--vs--'+right,case_id=case['id'],setting=case['setting'],pair=[left,right],group=group))
    return rows


def _difference(values,times,threshold):
    values=pm._array(values,'comparison differences');times=pm._array(times,'comparison times',(len(values),))
    maximum=float(np.max(np.abs(values)));scaled=values/maximum if maximum else values
    location=np.unravel_index(np.argmax(np.abs(values)),values.shape);cross=np.flatnonzero(np.any(np.abs(values)>threshold,axis=1))
    return dict(rms=float(maximum*np.sqrt(np.mean(scaled**2))),maximum_absolute=maximum,maximum_time=float(times[location[0]]),
        maximum_component=int(location[1]),value_at_maximum=float(values[location]),component_rms=(maximum*np.sqrt(np.mean(scaled**2,axis=0))).tolist(),
        threshold=threshold,first_threshold_crossing_time=float(times[cross[0]]) if len(cross) else None)


def compare_pair(spec,left_packet,right_packet,left_score,right_score):
    """Symmetric numerical differences without alignment or sham recentering."""
    result=dict(spec=spec,complete=False,qualified=False,error=None)
    try:
        case,left_method=_case_task(left_packet['task']);other,right_method=_case_task(right_packet['task'])
        if case!=other or spec not in comparison_specs([case]) or spec['pair']!=[left_method,right_method]:raise ValueError('Same-case canonical method pair required')
        left,_=validate_packet(left_packet,case);right,_=validate_packet(right_packet,case)
        a=_score_identity(left_score,case,left_method);b=_score_identity(right_score,case,right_method)
        if a['orientation']!=b['orientation'] or a['period']!=b['period'] or a['phase_epoch']!=b['phase_epoch']:raise ValueError('Readout clock/classifier changed across methods')
        for key in ('initial','phase_initial','requested_delta','realized_delta','addition_rounding_residual','initial_activation_displacement'):
            if not np.array_equal(left_packet[key],right_packet[key]):raise ValueError('Pair pulse identity differs')
        delta=left-right;activation=np.tanh(left)-np.tanh(right);late=TIMES>=240.
        state=_difference(delta,TIMES,1e-6);act=_difference(activation,TIMES,1e-7)
        readouts={};first=None
        for delay in DELAYS:
            ra,rb=a['readouts'][str(delay)],b['readouts'][str(delay)]
            for key in ('clock_sha256','noise_gain','completed_window_end','last_sample_time','phase_only_q','full_linear_q','zero_response_q'):
                if ra[key]!=rb[key]:raise ValueError('Sealed readout differs across methods: '+key)
            difference=abs(ra['q']-rb['q'])/(PRIMARY_NOISE*ra['noise_gain'])
            readouts[str(delay)]=dict(standardized_mean_difference=difference,passed=bool(difference<=.01),completed_window_end=ra['completed_window_end'],last_sample_time=ra['last_sample_time'])
            if first is None and difference>.01:first=dict(delay_cycles=delay,completed_window_end=ra['completed_window_end'],last_sample_time=ra['last_sample_time'])
        result.update(complete=True,state_difference=state,activation_difference=act,readouts=readouts,
            late_window=dict(start=240.,end=320.,samples=int(np.sum(late)),state_difference=_difference(delta[late],TIMES[late],1e-6),activation_difference=_difference(activation[late],TIMES[late],1e-7)),
            last_state_difference=delta[-1].tolist(),last_activation_difference=activation[-1].tolist(),
            first_completed_readout_threshold_crossing=first,state_pass=state['maximum_absolute']<=1e-6,activation_pass=act['maximum_absolute']<=1e-7)
        result['qualified']=bool(result['state_pass'] and result['activation_pass'] and all(r['passed'] for r in readouts.values()))
    except (ValueError,KeyError,TypeError,AttributeError,FloatingPointError) as error:result['error']=repr(error)
    return result


def _accuracy(qplus,qminus,gain,orientation,sigma):
    values=np.asarray([orientation*qplus,-orientation*qminus])/(sigma*gain)
    if not np.isfinite(values).all():raise ValueError('Nonfinite standardized class means')
    return float(np.mean(ndtr(values))),values.tolist()


def summarize(cases,packets,scores):
    """Retain every preselected case/method/pair and BA cell, including failures.

    Keys of packets/scores are task IDs. Missing rows remain explicit failures;
    no complete-case aggregate, selected solver, setting mean or requalification.
    """
    specs=comparison_specs(cases);by_case={r['id']:r for r in cases}
    if len(cases)!=85 or len(by_case)!=85:raise ValueError('Exactly 85 fixed physical cases required')
    contexts={(r['setting'],r['phase_index'],r['direction']) for r in cases if r['kind']=='pulse'}
    phase_contexts={(r['setting'],r['phase_index']) for r in cases if r['kind']=='pulse'}
    expected_pulses={(s,p,d,e,sign) for s,p,d in contexts for e in (0,1,2) for sign in (-1,1)}
    actual_pulses={(r['setting'],r['phase_index'],r['direction'],r['dose_index'],r['sign']) for r in cases if r['kind']=='pulse'}
    shams={(r['setting'],r['phase_index']) for r in cases if r['kind']=='sham'}
    if len(contexts)!=13 or len(phase_contexts)!=7 or expected_pulses!=actual_pulses or shams!=phase_contexts:raise ValueError('Complete 13 binary contexts/all doses/signs and seven shams required')
    allowed={task_for(c,m)['id'] for c in cases for m in METHODS}
    if set(packets)-allowed or set(scores)-allowed:raise ValueError('Unplanned packet/score IDs')
    terminal=[];valid={}
    for case in cases:
        for method in METHODS:
            task=task_for(case,method);name=task['id'];row=dict(task=task,status='failed',error=None,score=scores.get(name))
            try:
                validate_packet(packets[name],case);valid[name]=_score_identity(scores[name],case,method)
                row.update(status='complete',sham_pass=(all(abs(v['q']/(PRIMARY_NOISE*v['noise_gain']))<=.01 for v in valid[name]['readouts'].values()) if case['kind']=='sham' else None))
            except (KeyError,ValueError,TypeError,AttributeError) as error:row['error']=repr(error)
            terminal.append(row)
    comparisons=[compare_pair(spec,packets.get(spec['case_id']+'--'+spec['pair'][0],{}),packets.get(spec['case_id']+'--'+spec['pair'][1],{}),scores.get(spec['case_id']+'--'+spec['pair'][0],{}),scores.get(spec['case_id']+'--'+spec['pair'][1],{})) for spec in specs]
    ba=[]
    for setting,phase,direction in sorted(contexts):
        for dose in range(3):
            plus=f'{setting}--p{phase}--d{dose}--{direction}--s+1';minus=f'{setting}--p{phase}--d{dose}--{direction}--s-1'
            for method in METHODS:
                for delay in DELAYS:
                    for sigma in NOISES:
                        row=dict(setting=setting,phase_index=phase,direction=direction,dose_index=dose,method=method,delay_cycles=delay,sigma=sigma,status='failed',error=None)
                        try:
                            left,right=valid[plus+'--'+method],valid[minus+'--'+method];a,b=left['readouts'][str(delay)],right['readouts'][str(delay)]
                            if left['orientation']!=right['orientation'] or a['clock_sha256']!=b['clock_sha256'] or a['noise_gain']!=b['noise_gain']:raise ValueError('Signed pair clock/classifier differs')
                            for source,label in [('q','balanced_accuracy'),('phase_only_q','phase_only_balanced_accuracy'),('full_linear_q','full_linear_balanced_accuracy')]:
                                value,standardized=_accuracy(a[source],b[source],a['noise_gain'],left['orientation'],sigma)
                                row[label]=value
                                if source=='q':row['standardized_correct_means']=standardized
                            row.update(status='complete',q_plus=a['q'],q_minus=b['q'],noise_gain=a['noise_gain'],orientation=left['orientation'],zero_balanced_accuracy=.5)
                        except (KeyError,ValueError,TypeError,AttributeError) as error:row['error']=repr(error)
                        ba.append(row)
    counts=dict(physical_cases=85,methods=8,trace_rows=len(terminal),pair_rows=len(comparisons),same_method_ba_cells=len(ba),required_new_traces=425,required_old_traces=255)
    if (counts['trace_rows'],counts['pair_rows'],counts['same_method_ba_cells'])!=(680,2380,3744):raise AssertionError('Internal full denominator changed')
    groups={group:dict(required=sum(r['spec']['group']==group for r in comparisons),complete=sum(r['spec']['group']==group and r['complete'] for r in comparisons),passed=sum(r['spec']['group']==group and r['qualified'] for r in comparisons)) for group in ('old-old','old-new','new-new')}
    return dict(schema_version='phase-memory-refinement-diagnostics-v1',counts=counts,terminal_rows=terminal,comparisons=comparisons,paired_readouts=ba,comparison_groups=groups,
        complete=all(r['status']=='complete' for r in terminal+ba) and all(r['complete'] for r in comparisons),
        original_study_requalified=False,setting_means=None,scope='Post-run numerical diagnostic of all85 selected physical cases. All8 methods retained; unsaturated readouts accompany BA. No setting means, solver selection, alignment, sham recentering or original gate waiver.')
