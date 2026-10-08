"""Six pinned local residual derivatives; no trajectory, corrective solve or launch default."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS: os.environ[k]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from audit_s1r_branch_outcome import entry,digest,inventory,record,read_json,loss_gradient_hessian
from neural_shrimp.period2_branch_io import Store
from neural_shrimp.s1r_gap_diagnostic import (Budget,LIMITS,TOL,finite,exact,require,evaluate_state,
    geometry,carried_secants,saved_linear_check,saved_trials,request_stop)
PLAN='docs/plans/s1r-lower-gap-inputs.json'
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'tools/audit_s1r_branch_outcome.py','src/neural_shrimp/s1r_gap_diagnostic.py',
 'tools/diagnose_s1r_lower_gap.py','tools/launch_s1r_lower_gap.py',PLAN,
 'docs/plans/s1r-lower-gap-next.md','docs/plans/s1r-lower-gap-implementation.md')


def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean checkout required')
    result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes()
        require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted source '+name)
        result[name]=dict(bytes=len(data),sha256=digest(data))
    return result


def load_inputs():
    plan=read_json(ROOT/PLAN); folder=Path(plan['collection_root'])
    require(not folder.is_symlink() and not folder.stat().st_mode&0o222 and
        all(p.is_file() and not p.is_symlink() and not p.stat().st_mode&0o222 for p in folder.iterdir()),'immutable regular input inventory')
    inventory(folder,plan['collection_manifest']['sha256'])
    for name,identity in plan['collection_metadata'].items(): require(entry(folder/name)==identity,'metadata pin '+name)
    runtime=read_json(folder/'runtime-start.json')
    require(runtime['source_commit']==plan['producer_execution_commit'],'producer execution source')
    for name,identity in runtime['sources'].items():
        for commit in (plan['producer_execution_commit'],plan['producer_reviewed_source']):
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
            require(dict(bytes=len(data),sha256=digest(data))==identity,'producer source binding '+name)
    analytic=plan['independent_analytic_model_source']
    require(entry(ROOT/analytic['file'])==analytic['identity'],'independent analytic source unchanged')
    decoded={}; encoded={}
    for name,pin in plan['selected_records'].items():
        require(entry(folder/name)==pin['json'] and entry(folder/pin['arrays_file'])==pin['arrays'],'selected input pin '+name)
        encoded[name]=read_json(folder/name)['payload']; kind,payload=record(folder,name);decoded[name]=payload
    binding=decoded['00000-bound_inputs.json']
    require(binding['model']==dict(activation='tanh',bias=False,d=8,in_scale=.5,loss_scale=1.,out_scale=.125,ridge=.001,width=8),'exact S1R model')
    x,y=finite(binding['x'],(72,8)),finite(binding['y'],(72,))
    slots=[]
    for spec in plan['six_future_fixed_slots']:
        node=decoded[spec['record']]; raw=encoded[spec['record']]
        for key in ('params','rates'): require(raw[key]==spec[key],'slot control descriptor')
        if 'native_phase' in spec:
            require(node['accepted'] and raw['pair']==spec['pair'],'native pair input')
            state=node['pair'][spec['native_phase']]
        else:
            index=0
            key='seed' if spec['role'].endswith('_seed') else 'state'
            require(raw['corrections'][index][key]==spec['state'],'failed state descriptor')
            state=node['corrections'][index][key]
        slots.append(dict(role=spec['role'],state=finite(state,(72,)),params=finite(node['params'],(2,)),
            rates=finite(node['rates'],(2,)),native_phase=spec.get('native_phase'),node=node))
    require(len(slots)==6 and len({s['role'] for s in slots})==6,'six unique roles')
    parent,last=slots[0]['node'],slots[2]['node']; edge=decoded['05002-edge.json']; failure=decoded['05267-edge.json']
    require(edge['status']=='accepted' and edge['parent']==parent['id'] and edge['direct']==last['id'],'witnessed predecessor/last edge')
    require(failure['parent']==last['id'] and failure['direct']==slots[4]['node']['id'],'nearest failed edge')
    require(edge['parent_phase'] in (0,1) and edge['parent_match']['phase'] in (0,1),'native/carry phase bits')
    exchange=edge['parent_phase']^edge['parent_match']['phase']
    secants=carried_secants(parent['pair'],last['pair'],parent['params'][0],last['params'][0],exchange)
    require(failure['parent_phase']==edge['parent_match']['phase'],'failure carries last native phase')
    for slot in slots[:4]:
        slot['secant']=secants[('parent' if slot['role'].startswith('predecessor') else 'child',slot['native_phase'])]
    for slot in slots[4:]: slot['secant']=secants[('child',failure['parent_phase'])]
    iteration=decoded[plan['saved_last_linear_equation']['record']]
    require(iteration['call']==473 and iteration['iteration']==3 and exact(iteration['state'],slots[5]['state']),'stalled linear equation state')
    for key in ('state','residual','step'):
        require(encoded[plan['saved_last_linear_equation']['record']][key]==plan['saved_last_linear_equation'][key],'linear equation descriptor')
    trials=[decoded[f'{i:05d}-line_search.json'] for i in range(5252,5265)]
    require(all(s['params'][1]==parent['params'][1] for s in slots),'fixed output control')
    return dict(plan=plan,identity=binding['identity'],x=x,y=y,model=binding['model'],slots=slots,
        iteration=iteration,trials=trials,scale=np.maximum(1.,np.abs(last['pair'][0])),
        phase_binding=dict(parent_phase=edge['parent_phase'],exchange=exchange,last_carried_phase=failure['parent_phase']))


def agreement(assembly,slot):
    if slot['native_phase'] is None:return dict(scope='nonclosed fixed-state residual geometry',qualified_endpoint=False)
    node=slot['node']; phase=slot['native_phase']; state=slot['state']; other=node['pair'][1-phase]
    normalized=lambda a,b:float(np.max(np.abs(a-b)/(TOL['closure_absolute']+TOL['closure_relative']*np.maximum(np.abs(a),np.abs(b)))))
    closure=normalized(assembly['map2'],state);transition=normalized(assembly['map1'],other)
    separation=float(np.max(np.abs(node['pair'][0]-node['pair'][1])))/max(1.,float(np.max(np.abs(node['pair']))))
    error=float(np.max(np.abs(assembly['M']-node['spectra']['phases'][phase]['matrix'])))
    passed=closure<=1 and transition<=1 and separation>=TOL['primitive_separation'] and error<=TOL['agreement_absolute']
    return dict(qualified_endpoint=bool(passed),closure_normalized=closure,transition_normalized=transition,
        primitive_separation=separation,saved_monodromy_max_absolute_error=error,
        scope='numerical closed-orbit endpoint only; no fold or new orbit acceptance')


def run(out,review_commit):
    source_inventory=sources()
    require(isinstance(review_commit,str) and re.fullmatch('[0-9a-f]{40}',review_commit),'full review commit')
    subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in THREADS),'CPU1 nice10')
    store=Store(out,LIMITS['store_bytes']); budget=Budget(store.emit); started=time.monotonic()
    result=dict(schema='s1r-lower-gap-v1',status='running',completed_roles=[],accepted_orbit=False,
        applied_correction=False,fold_verified=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        review_commit=review_commit,sources=source_inventory,limits=LIMITS,tolerances=TOL)
    try:
        store.json('runtime-start.json',dict(**result,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
            thread_env={k:os.environ[k] for k in THREADS},numpy=np.__version__,scipy=scipy.__version__,
            python=platform.python_version(),started_utc=datetime.now(timezone.utc).isoformat(),gpu=False,paid=False))
        inputs=load_inputs(); budget.check()
        store.json('input-plan.json',inputs['plan'])
        store.emit('bound_inputs',dict(plan_file='input-plan.json',plan_identity=entry(store.root/'input-plan.json'),
            identity=inputs['identity'],model=inputs['model'],
            x=inputs['x'],y=inputs['y'],phase_binding=inputs['phase_binding'],scale=inputs['scale'],
            slots=[{k:s[k] for k in ('role','state','params','rates','native_phase','secant')} for s in inputs['slots']],
            saved_iteration=inputs['iteration'],saved_trials=inputs['trials']))
        result['saved_line_search']=saved_trials(inputs['iteration'],inputs['trials'])
        store.emit('saved_line_search_reduction',result['saved_line_search'])
        evaluations={};geometries={}
        for index,slot in enumerate(inputs['slots']):
            budget.check();role=slot['role']
            rate_vector=np.r_[np.full(64,slot['rates'][0]),np.full(8,slot['rates'][1])]
            derivative=lambda theta:loss_gradient_hessian(theta,inputs['x'],inputs['y'],inputs['model'])
            assembly=evaluate_state(slot['state'],rate_vector,64,derivative,budget,role)
            check=agreement(assembly,slot);store.emit(role+'_agreement',check)
            if slot['native_phase'] is not None:require(check['qualified_endpoint'],'qualified endpoint derivative/closure disagreement')
            if index==5:
                delta=float(np.max(np.abs(assembly['R']-inputs['iteration']['residual'])))
                store.emit('stalled_residual_agreement',dict(max_absolute_error=delta,passed=delta<=TOL['agreement_absolute']))
                require(delta<=TOL['agreement_absolute'],'stalled residual disagreement')
                result['saved_linear_equation']=saved_linear_check(assembly['A'],assembly['R'],slot['state'],
                    inputs['iteration']['step'],inputs['iteration']['residual'])
                store.emit('saved_linear_equation',result['saved_linear_equation'])
            geometry_result=geometry(assembly,inputs['scale'],slot['secant'],budget,role)
            label='numerical qualified-endpoint branch tangent' if check['qualified_endpoint'] else 'local residual level-set tangent at nonclosed state'
            store.emit(role+'_complete',dict(coordinate_statuses={k:v['rank']['status'] for k,v in geometry_result.items()},
                tangent_scope=label,accepted_orbit=False,fold_verified=False))
            evaluations[role]=assembly;geometries[role]=geometry_result
            result['completed_roles'].append(role)
        # Phase propagation uses already retained J0 and partial_h_F, never a new evaluation.
        phase_checks=[]
        for prefix in ('predecessor','last'):
            zero=prefix+'_native_phase_0';one=prefix+'_native_phase_1'
            for coords in ('raw_B','scaled_B'):
                t0=geometries[zero][coords]['null_tangent'];t1=geometries[one][coords]['null_tangent']
                if t0['tangent'] is None or t1['tangent'] is None:
                    phase_checks.append(dict(endpoint=prefix,coordinates=coords,status='abstain_unqualified_tangent'));continue
                a=evaluations[zero];v=t0['tangent']
                propagated=np.r_[a['J0']@v[:-1]+a['partial_h_F']*v[-1],v[-1]]
                norm=float(np.linalg.norm(propagated))
                cosine=float(propagated@t1['tangent']/norm) if norm>0 else None
                phase_checks.append(dict(endpoint=prefix,coordinates=coords,status='descriptive_phase_transport',
                    propagated=propagated,oriented_cosine=cosine))
        store.emit('phase_tangent_transport',phase_checks)
        budget.check()
        require(budget.attempted==budget.returned==dict(maps=12,jacobians=12,svds=24),'complete fixed operation inventory')
        result['status']='diagnostic_complete'
    except BaseException as error:
        result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error));raise
    finally:
        # Arrays remain in their phase records; compact result contains summaries only.
        if 'saved_linear_equation' in result:
            result['saved_linear_equation']={k:v for k,v in result['saved_linear_equation'].items() if not isinstance(v,np.ndarray)}
        result.update(operation_attempted=budget.attempted,operation_returned=budget.returned,
            elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        store.finish(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True)
    args=p.parse_args();signal.signal(signal.SIGTERM,request_stop)
    result=run(args.out,args.review_commit)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
