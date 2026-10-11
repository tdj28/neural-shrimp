#!/usr/bin/env python3
"""Independent saved-array refinement audit, developed before its outcomes.

No producer imports, ODE calls, fitting or arithmetic-tolerance adjustments.
Exact evidence/identity/scientific-gate checks and strict independent arithmetic
replay are reported separately. SciPy Hermite, vector RHS and math.erf differ
from producer scalar RHS/manual Hermite/ndtr; rounding differences are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import independent_codec as codec
import selection_v2 as selection

OLD_METHODS=('rk4-0.0025','rk4-0.00125','dop853')
NEW_METHODS=('rk4-0.000625','rk4-0.0003125','rk4-0.00015625','dop853-1e-13-h0.0125','dop853-1e-13-h0.00625')
METHODS=OLD_METHODS+NEW_METHODS
TIMES=np.arange(12801,dtype=np.float64)*.025;TIMES[-1]=320.
DELAYS=(1,2,4,8);NOISES=(1e-5,1e-4,1e-3)
SETTINGS={'core':(1.064,10.260722222222222),'failed-near-edge':(1.0661333333333334,10.297166666666667)}
read=codec.read;sha=codec.sha;decode=codec.decode_packet


def json_default(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)


def equal(a,b):
    if isinstance(a,np.generic):a=a.item()
    if isinstance(b,np.generic):b=b.item()
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def tree_replay(saved,fresh,arithmetic,evidence,path=''):
    if path.rsplit('/',1)[-1] in {'first_threshold_crossing_time','maximum_time','completed_window_end','last_sample_time','threshold','start','end'}:
        evidence.check(equal(saved,fresh),path+' exact sampled time/threshold')
    elif isinstance(saved,np.ndarray) or isinstance(fresh,np.ndarray):arithmetic.near(saved,fresh,path,array=True)
    elif isinstance(fresh,dict):
        evidence.check(isinstance(saved,dict) and saved.keys()==fresh.keys(),path+' exact fields')
        if isinstance(saved,dict):
            for k in fresh:
                if k in saved:tree_replay(saved[k],fresh[k],arithmetic,evidence,path+'/'+k)
    elif isinstance(fresh,list):
        evidence.check(isinstance(saved,list) and len(saved)==len(fresh),path+' list census')
        if isinstance(saved,list):
            for i,(a,b) in enumerate(zip(saved,fresh)):tree_replay(a,b,arithmetic,evidence,path+f'/{i}')
    elif type(fresh) in (float,np.float64):arithmetic.near(saved,fresh,path)
    else:evidence.check(equal(saved,fresh),path+' exact value',dict(saved=saved,independent=fresh))


def rhs(states,alpha,gamma):
    matrix=np.array([[1.5,2.9,alpha],[-1.5,1.18,0.],[gamma,-22.,.47]],dtype=np.float64)
    return -states+np.tanh(states)@matrix.T


def clock_hash(row):
    h=hashlib.sha256(json.dumps({k:row[k] for k in ('derivative_energy','phase_epoch','period','delay_cycles')},sort_keys=True,separators=(',',':')).encode())
    for k in ('times','reference_activations','weights'):h.update(k.encode());h.update(row[k].tobytes(order='C'))
    return h.hexdigest()


def score(packet,case,context,evidence):
    name=packet['task']['id'];method=packet['task']['method'];task=dict(case,id=case['id']+'--'+method,method=method)
    evidence.check(equal(packet['task'],task) and method in METHODS,name+' task identity')
    evidence.check(packet['status']=='complete' and packet['complete'] is True,name+' complete trace')
    x=packet['states'];evidence.check(x.dtype==np.float64 and x.shape==(12801,3) and np.isfinite(x).all() and np.array_equal(packet['times'],TIMES),name+' finite full grid')
    evidence.check(tuple(packet[k] for k in ('alpha','gamma'))==SETTINGS[case['setting']],name+' fixed weights')
    evidence.check(packet['duration']==320. and packet['sample_dt']==.025,name+' fixed duration')
    if method.startswith('rk4-'):
        dt=float(method[4:]);required=dict(id=method,kind='RK4',method='RK4',dt=dt,steps=round(320/dt))
    else:
        rtol,atol,h=(1e-12,1e-14,.025) if method=='dop853' else (1e-13,1e-15,.0125 if method.endswith('h0.0125') else .00625)
        required=dict(id=method,kind='DOP853',method='DOP853',rtol=rtol,atol=atol,max_step=h,success=True)
        evidence.check(type(packet['solver']['nfev']) is int and packet['solver']['nfev']>=0,name+' truthful adaptive work')
    evidence.check(all(equal(packet['solver'].get(k),v) for k,v in required.items()),name+' truthful solver')
    for key,value in context['pulse'].items():evidence.check(equal(packet[key],value),name+' original pulse '+key)
    base=context['phase_initial'];evidence.check(np.array_equal(x[0],packet['initial']) and np.array_equal(packet['initial'],base+packet['requested_delta']),name+' exact initial addition')
    evidence.check(np.array_equal(packet['realized_delta'],packet['initial']-base),name+' realized pulse')
    evidence.check(np.array_equal(packet['initial_activation_displacement'],np.tanh(packet['initial'])-np.tanh(base)),name+' activation pulse')
    evidence.check(packet['phase_epoch']==context['phase_epoch'] and packet['period']==context['period'],name+' fixed clock epoch')
    interpolator=CubicHermiteSpline(TIMES,x,rhs(x,*SETTINGS[case['setting']]),axis=0)
    rows={}
    for delay in DELAYS:
        clock=context['readouts'][str(delay)];t=clock['times'];weight=clock['weights'];gain=float(np.linalg.norm(weight))
        evidence.check(np.array_equal(t,context['period']*(delay-1+(np.arange(32)+.5)/32)),name+f' exact clock{delay}')
        q=float(np.sum(weight*(np.tanh(interpolator(t))-clock['reference_activations'])))
        old=context['predictions'][str(delay)]
        rows[str(delay)]=dict(q=q,noise_gain=gain,clock_sha256=clock_hash(clock),completed_window_end=delay*clock['period'],last_sample_time=float(t[-1]),
            phase_only_q=old['phase_only_q'],full_linear_q=old['full_linear_q'],zero_response_q=0.,standardized_sham_bias=q/(1e-4*gain) if case['kind']=='sham' else None)
    return dict(task=task,case_id=case['id'],status='complete',readouts=rows,error=None,method=method,orientation=context['orientation'],period=context['period'],phase_epoch=context['phase_epoch'])


def stats(values,times,threshold):
    magnitude=np.abs(values);flat=int(np.argmax(magnitude));location=np.unravel_index(flat,values.shape);maximum=float(magnitude[location])
    scaled=values/maximum if maximum else values;cross=np.flatnonzero(np.max(magnitude,axis=1)>threshold)
    return dict(rms=float(maximum*np.sqrt(np.mean(scaled*scaled))),maximum_absolute=maximum,maximum_time=float(times[location[0]]),maximum_component=int(location[1]),value_at_maximum=float(values[location]),
        component_rms=(maximum*np.sqrt(np.mean(scaled*scaled,axis=0))).tolist(),threshold=threshold,first_threshold_crossing_time=float(times[cross[0]]) if len(cross) else None)


def specs(cases):
    result=[]
    for case in cases:
        for a,b in itertools.combinations(METHODS,2):
            group='old-old' if a in OLD_METHODS and b in OLD_METHODS else 'new-new' if a in NEW_METHODS and b in NEW_METHODS else 'old-new'
            result.append(dict(id=case['id']+'--'+a+'--vs--'+b,case_id=case['id'],setting=case['setting'],pair=[a,b],group=group))
    return result


def comparison(spec,packets,scores):
    a,b=(spec['case_id']+'--'+m for m in spec['pair'])
    if a not in packets or b not in packets or a not in scores or b not in scores:return dict(spec=spec,complete=False,qualified=False)
    delta=packets[a]['states']-packets[b]['states'];u=np.tanh(packets[a]['states'])-np.tanh(packets[b]['states']);late=TIMES>=240.
    state=stats(delta,TIMES,1e-6);activation=stats(u,TIMES,1e-7);rows={};first=None
    for delay in DELAYS:
        x,y=scores[a]['readouts'][str(delay)],scores[b]['readouts'][str(delay)];difference=abs(x['q']-y['q'])/(1e-4*x['noise_gain'])
        rows[str(delay)]=dict(standardized_mean_difference=difference,passed=bool(difference<=.01),completed_window_end=x['completed_window_end'],last_sample_time=x['last_sample_time'])
        if first is None and difference>.01:first=dict(delay_cycles=delay,completed_window_end=x['completed_window_end'],last_sample_time=x['last_sample_time'])
    sp=state['maximum_absolute']<=1e-6;ap=activation['maximum_absolute']<=1e-7
    return dict(spec=spec,complete=True,qualified=bool(sp and ap and all(x['passed'] for x in rows.values())),error=None,state_difference=state,activation_difference=activation,
        readouts=rows,late_window=dict(start=240.,end=320.,samples=int(np.sum(late)),state_difference=stats(delta[late],TIMES[late],1e-6),activation_difference=stats(u[late],TIMES[late],1e-7)),
        last_state_difference=delta[-1].tolist(),last_activation_difference=u[-1].tolist(),first_completed_readout_threshold_crossing=first,state_pass=sp,activation_pass=ap)


def accuracy(qplus,qminus,gain,orientation,sigma):
    correct=np.array([orientation*qplus,-orientation*qminus])/(sigma*gain)
    return float(sum(.5*(1+math.erf(float(z)/math.sqrt(2))) for z in correct)/2),correct.tolist()


def paired_cells(cases,scores):
    binary=sorted({(c['setting'],c['phase_index'],c['direction']) for c in cases if c['kind']=='pulse'});result=[]
    for setting,phase,direction in binary:
        for dose in range(3):
            stem=f'{setting}--p{phase}--d{dose}--{direction}'
            for method in METHODS:
                for delay in DELAYS:
                    for sigma in NOISES:
                        row=dict(setting=setting,phase_index=phase,direction=direction,dose_index=dose,method=method,delay_cycles=delay,sigma=sigma,status='failed')
                        plus=scores.get(stem+'--s+1--'+method);minus=scores.get(stem+'--s-1--'+method)
                        if plus is not None and minus is not None:
                            a,b=plus['readouts'][str(delay)],minus['readouts'][str(delay)];orientation=plus['orientation'];gain=a['noise_gain']
                            for metric,out in [('q','balanced_accuracy'),('phase_only_q','phase_only_balanced_accuracy'),('full_linear_q','full_linear_balanced_accuracy')]:
                                value,z=accuracy(a[metric],b[metric],gain,orientation,sigma);row[out]=value
                                if metric=='q':row['standardized_correct_means']=z
                            row.update(status='complete',error=None,q_plus=a['q'],q_minus=b['q'],noise_gain=gain,orientation=orientation,zero_balanced_accuracy=.5)
                        result.append(row)
    return result


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def bound_artifacts(raw,repo,evidence):
    receipt=read(raw/'receipt.json');manifest=read(raw/'manifest.json');plan=manifest['plan'];entries=[]
    for path in sorted(raw.rglob('*')):
        evidence.check(not path.is_symlink(),'no outcome symlink:'+str(path))
        if path.is_file() and path!=raw/'receipt.json':entries.append(dict(path=str(path.relative_to(raw)),bytes=path.stat().st_size,sha256=sha(path)))
    evidence.check(entries==receipt['artifacts'],'exact raw inventory')
    evidence.check(sum(x['bytes'] for x in entries)==receipt['artifact_bytes'],'receipt bytes')
    evidence.check(receipt['source_commit']==manifest['source_commit'] and receipt['plan_sha256']==manifest['plan_sha256'],'manifest receipt source')
    source=receipt['source_commit']
    for entry in plan['source_files']+[manifest['review']]:
        data=subprocess.check_output(['git','-C',str(repo),'show',f'{source}:{entry["path"]}'])
        evidence.check(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'committed source:'+entry['path'])
    content=subprocess.check_output(['git','-C',str(repo),'show',f'{source}:experiments/hopfield/EXP-014-refinement-plan.json'])
    evidence.check(hashlib.sha256(content).hexdigest()==receipt['plan_sha256'] and json.loads(content)==plan,'committed exact plan')
    old=Path(plan['original_directory']);evidence.check(sha(old/'receipt.json')==plan['original_receipt_sha256'],'original receipt unchanged')
    for entry in plan['input_files']:
        path=old/entry['path'];evidence.check(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'bound old input:'+entry['path'])
    independent=selection.select(read(old/'summary.json'));cases=[{k:v for k,v in row.items() if k!='selection_category'} for row in independent['cases']]
    evidence.check(cases==plan['selection']['cases'],'independent exact ordered85-case selection')
    return receipt,manifest,plan,cases,dict(artifacts=len(entries),bytes=sum(x['bytes'] for x in entries),frozen_sources=len(plan['source_files']),old_inputs=len(plan['input_files']))


def run(raw,repo,evidence,arithmetic):
    raw=Path(raw).resolve();repo=Path(repo).resolve();receipt,manifest,plan,cases,inventory=bound_artifacts(raw,repo,evidence);old=Path(plan['original_directory'])
    analysis=decode(raw/'analysis.json',evidence);saved=analysis['summary'];summary=read(raw/'summary.json');ledger=read(raw/'all-terminal-records.json')
    desired=[c['id']+'--'+m for c in cases for m in NEW_METHODS]
    evidence.check(len(ledger)==425 and {r['id'] for r in ledger}==set(desired),'425 exact new terminal census')
    evidence.check(summary['status_counts']==dict(Counter(r['status'] for r in ledger)),'terminal status census')
    contexts={};packets={};scores={};missing=[]
    if (raw/'input-seal.json').exists():
        contexts=decode(raw/'input-seal.json',evidence);original=decode(old/'reference-seal.json',evidence)
        evidence.check(set(contexts)=={c['id'] for c in cases},'85 sealed compact contexts')
        for case in cases:
            context=contexts[case['id']];previous=original['cases'][case['id']];clock=original['contexts'][f'{case["setting"]}--p{case["phase_index"]}']
            expected=dict(setting=case['setting'],alpha=SETTINGS[case['setting']][0],gamma=SETTINGS[case['setting']][1],phase_initial=clock['initial'],phase_epoch=clock['phase_epoch'],period=clock['period'],pulse=previous['pulse'],orientation=previous['orientation'],readouts=clock['readouts'],predictions=previous['readouts'],original_reference_semantic_sha256=read(old/'reference-seal.json')['semantic_sha256'],original_case_id=case['id'])
            evidence.check(equal(context,expected),'exact unchanged clock/pulse context:'+case['id'])
        original=None;gc.collect()
        old_ledger={r['id']:r for r in read(old/'all-terminal-records.json')['traces']};new_ledger={r['id']:r for r in ledger}
        for case in cases:
            for method in METHODS:
                name=case['id']+'--'+method;rows,base=(old_ledger,old) if method in OLD_METHODS else (new_ledger,raw);row=rows[name]
                if row['status']!='complete':missing.append(name);continue
                path=base/'traces'/row['packet_path'];evidence.check(sha(path)==row['packet_sha256'],name+' packet metadata hash');packet=decode(path,evidence)
                try:result=score(packet,case,contexts[case['id']],evidence)
                except (ValueError,KeyError,TypeError,IndexError) as error:
                    evidence.check(False,'complete trace independent score exception:'+name,repr(error));continue
                packets[name]=packet;scores[name]=result
    terminal={row['task']['id']:row for row in saved['terminal_rows']};expected_all=[c['id']+'--'+m for c in cases for m in METHODS]
    evidence.check(len(saved['terminal_rows'])==680 and set(terminal)==set(expected_all),'680 exact score rows')
    for name,row in terminal.items():
        evidence.check((row['status']=='complete')==(name in scores),name+' score availability')
        if name in scores:
            tree_replay(row['score'],scores[name],arithmetic,evidence,name+' score')
            if row['task']['kind']=='sham':
                passed=all(abs(x['q']/(1e-4*x['noise_gain']))<=.01 for x in scores[name]['readouts'].values());evidence.check(row['sham_pass']==passed,name+' sham gate')
    planned=specs(cases);recorded={r['spec']['id']:r for r in saved['comparisons']}
    evidence.check(len(saved['comparisons'])==2380 and set(recorded)=={r['id'] for r in planned},'2380 exact method comparisons')
    independent_pairs=[]
    for spec in planned:
        fresh=comparison(spec,packets,scores);oldrow=recorded[spec['id']]
        evidence.check(oldrow['complete']==fresh['complete'] and oldrow['qualified']==fresh['qualified'],spec['id']+' scientific agreement decision')
        if fresh['complete']:tree_replay(oldrow,fresh,arithmetic,evidence,spec['id']+' comparison')
        independent_pairs.append(fresh)
    cells=paired_cells(cases,scores);evidence.check(len(cells)==len(saved['paired_readouts'])==3744,'3744 complete BA denominator')
    for index,(fresh,stored) in enumerate(zip(cells,saved['paired_readouts'])):
        keys=('setting','phase_index','direction','dose_index','method','delay_cycles','sigma','status')
        evidence.check(all(equal(fresh[k],stored[k]) for k in keys),f'BA{index} fixed identity/status')
        if fresh['status']=='complete':tree_replay(stored,fresh,arithmetic,evidence,f'BA{index}')
    groups={g:dict(required=sum(r['spec']['group']==g for r in independent_pairs),complete=sum(r['spec']['group']==g and r['complete'] for r in independent_pairs),passed=sum(r['spec']['group']==g and r['qualified'] for r in independent_pairs)) for g in ('old-old','old-new','new-new')}
    evidence.check(saved['comparison_groups']==groups,'all comparison group counts')
    analysis_complete=len(scores)==680 and all(r['complete'] for r in independent_pairs) and all(r['status']=='complete' for r in cells)
    evidence.check(saved['complete']==analysis_complete and saved['setting_means'] is None and saved['original_study_requalified'] is False,'no omission/means/requalification')
    evidence.check(equal(summary['analysis'],saved),'summary versus saved analysis')
    post=read(raw/'postflight.json')
    evidence.check(summary['postflight_error']==post['error'] and summary['source_valid'] is (post['error'] is None) and post['source_bindings_unchanged'] is (post['error'] is None),'postflight source conclusion')
    expected_complete=all(r['status']=='complete' for r in ledger) and summary['error'] is None and not analysis['packet_errors'] and analysis_complete
    evidence.check(summary['complete']==expected_complete,'producer full completion')
    evidence.check(receipt['status']==('complete' if expected_complete and summary['postflight_error'] is None else 'incomplete_or_invalid'),'receipt completion')
    owners=[];remaining=[]
    stage=raw/'traces'
    if stage.exists():
        cleanup=read(stage/'cleanup.json');owners=[json.loads(x) for x in (stage/'processes.jsonl').read_text().splitlines()]
        evidence.check(cleanup['all_registered_workers_joined'] is True and cleanup['all_original_identities_absent'] is True,'recorded cleanup')
        evidence.check(len(owners)==len(cleanup['workers']) and len({x['task_id'] for x in owners})==len(owners),'ownership exact census')
        for owner,closed in zip(owners,cleanup['workers'],strict=True):
            evidence.check(all(closed[k]==v for k,v in owner.items()) and closed['original_identity_absent'] is True,'owned historical absence:'+owner['task_id'])
            if process_identity(owner['pid'])==owner['identity']:remaining.append(owner)
        index={r['task_id']:r for r in owners}
        for row in ledger:
            if row.get('pid') is not None:evidence.check(row['id'] in index and row['pid']==index[row['id']]['pid'],'terminal owned PID:'+row['id'])
        events=[json.loads(x) for x in (raw/'events.jsonl').read_text().splitlines()];seal=[x for x in events if x['event']=='original_inputs_sealed'];start=[x for x in events if x['event']=='traces_started']
        evidence.check(len(seal)==len(start)==1 and seal[0]['monotonic_ns']<start[0]['monotonic_ns'],'old sealed clock before all target dispatch')
        evidence.check(seal[0]['sha256']==sha(raw/'input-seal.json'),'original input seal event binding')
    producer_absent=process_identity(manifest['producer_pid'])!=manifest['producer_identity'];evidence.check(not remaining and producer_absent,'current namespace original owners absent')
    complete_pairs=[r for r in independent_pairs if r['complete']]
    details={}
    for g in groups:
        selected=[r for r in complete_pairs if r['spec']['group']==g]
        details[g]=dict(max_state=max((r['state_difference']['maximum_absolute'] for r in selected),default=None),max_activation=max((r['activation_difference']['maximum_absolute'] for r in selected),default=None),max_standardized_readout=max((v['standardized_mean_difference'] for r in selected for v in r['readouts'].values()),default=None),
            state_failures=sum(not r['state_pass'] for r in selected),activation_failures=sum(not r['activation_pass'] for r in selected),readout_failures=sum(any(not v['passed'] for v in r['readouts'].values()) for r in selected))
    return dict(raw_directory=str(raw),source_commit=receipt['source_commit'],receipt_sha256=sha(raw/'receipt.json'),inventory=inventory,counts=dict(physical_cases=85,expected_new_traces=425,loaded_traces=len(packets),scored_traces=len(scores),comparison_rows=2380,paired_ba_cells=3744),comparison_groups=groups,group_maxima_and_failure_counts=details,
        failing_comparisons=[r for r in independent_pairs if not r['qualified']],missing_or_incomplete_traces=missing,
        complete=analysis_complete,setting_means=None,original_study_requalified=False,cleanup=dict(recorded_workers=len(owners),remaining_original_identities=remaining,producer_original_identity_absent=producer_absent,pid_namespace=os.readlink('/proc/self/ns/pid'),hostname=platform.node(),scope='Current namespace; host invocation must be established separately'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.raw.resolve()) or a.output.resolve().is_relative_to(a.repo.resolve()):raise ValueError('New external report required')
    evidence=codec.Audit();arithmetic=codec.Audit();start=time.monotonic();result={}
    try:result=run(a.raw,a.repo,evidence,arithmetic)
    except Exception as error:evidence.check(False,'fatal audit exception',repr(error))
    result.update(evidence_and_scientific_decisions_passed=not evidence.failures,evidence_checks=evidence.checks,evidence_failures=evidence.failures,
        strict_arithmetic_replay_passed=not arithmetic.failures,strict_arithmetic_checks=arithmetic.checks,strict_arithmetic_failures=arithmetic.failures,
        maximum_scalar_arithmetic_difference=arithmetic.max_scalar_error,maximum_array_arithmetic_difference=arithmetic.max_array_error,elapsed_seconds=time.monotonic()-start,
        arithmetic_tolerances=dict(rtol=1e-10,scalar_atol=1e-12,array_atol=1e-11),scientific_thresholds=dict(state_max=1e-6,activation_max=1e-7,standardized_readout=.01),
        audit_sources=[dict(path=str(x),sha256=sha(x)) for x in (Path(__file__),Path(codec.__file__),Path(selection.__file__),Path(selection.previous.__file__))],scope='Saved evidence only, no integration/fitting. Arithmetic discrepancies are retained separately; original EXP014 strict audit and scientific conclusions remain unchanged.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,default=json_default,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('evidence_and_scientific_decisions_passed','evidence_checks','strict_arithmetic_replay_passed','strict_arithmetic_checks')}))
    raise SystemExit(0 if not evidence.failures else 1)


if __name__=='__main__':main()
