"""Profile or collect the fixed S1R bootstrap association unit. Separate root authorization required."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.gdmaps import GridMap
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.period2_branch import Options,PilotStop
from neural_shrimp.period2_branch_io import Store,file_entry,identity,load_record,canonical
from neural_shrimp.s1r_bootstrap import (InterruptibleBudget,HiddenBridge,run_bootstrap,compare_common_controls,
    fixed_state_profile,OUTPUT_LOG,CENTRAL_HIDDEN,STARTS,WARM_OPTIONS,WARM_STEPS)

INPUT=Path('/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001')
INPUT_MANIFEST='45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
INPUT_SOURCE='db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
PLAN='docs/plans/s1r-historical-family-association-next.md'
SOURCES=(PLAN,'src/neural_shrimp/__init__.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/trainability.py',
 'src/neural_shrimp/models.py','src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'src/neural_shrimp/s1r_bootstrap.py','tools/explore_s1r_bootstrap_association.py','tools/explore_a3_ns_curve.py')
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:raise ValueError(message)


def sources():
    result={}
    for name in SOURCES:
        require((ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted source '+name)
        result[name]=file_entry(ROOT/name)
    return result


def check_manifest(folder,expected=None):
    if expected is not None:require(file_entry(folder/'manifest.json')['sha256']==expected,'input manifest hash')
    m=json.loads((folder/'manifest.json').read_text())
    require(set(p.name for p in folder.iterdir())==set(m['files'])|{'manifest.json'},'exact input inventory')
    for name,e in m['files'].items():
        p=folder/name;require(Path(name).name==name and not p.is_symlink() and file_entry(p)==e,'input file binding '+name)
    return m


def bound_inputs():
    check_manifest(INPUT,INPUT_MANIFEST)
    runtime=json.loads((INPUT/'runtime-start.json').read_text());require(runtime['source_commit']==INPUT_SOURCE,'saved-lineage source')
    _,inputs=load_record(INPUT,'00000-inputs.json');central=None
    for event in [json.loads(line) for line in (INPUT/'journal.jsonl').read_text().splitlines()]:
        if event['kind']=='node':
            _,node=load_record(INPUT,event['file'])
            if node['id']=='node-0004':central=node
    require(central is not None and central['accepted'],'qualified saved central pair')
    require(np.array_equal(central['params'],[CENTRAL_HIDDEN,OUTPUT_LOG]),'saved central controls')
    net,x,y,theta0=SYSTEMS['S1R'](28002)
    require(asdict(net)==inputs['model'],'model identity')
    for name,value in dict(x=x,y=y,theta0=theta0).items():
        require(identity(value)==inputs['identity'][name] and np.array_equal(value,inputs[name]),'model data identity '+name)
    return net,x,y,theta0,central,inputs['identity']


def qualified_profile(folder,source_inventory):
    m=check_manifest(folder);r=json.loads((folder/'result.json').read_text());runtime=json.loads((folder/'runtime-start.json').read_text())
    require(r['mode']=='profile' and r['status']=='profile_complete' and m['status']==r['status'],'completed profile required')
    require(runtime['sources']==source_inventory and runtime['input_manifest_sha256']==INPUT_MANIFEST,'profile source/input binding')
    require(runtime['device']=='cpu' and runtime['nice']>=10 and all(runtime['thread_env'][k]=='1' for k in THREADS),'profile resources')
    p=r['timing'];require(p['map_calls']==2000 and p['jvp_calls']==128,'profile count')
    require(all(np.isfinite(p[k]) and p[k]>0 for k in ('map_seconds','jvp_seconds')),'finite profile timings')
    estimate=2*(80000*p['map_seconds']/2000+2400*p['jvp_seconds']/128)+30
    require(np.isfinite(estimate) and estimate<=100 and estimate==p['projected_collection_seconds'] and p['fits_internal_100_seconds'],'profile requires disclosed split')
    return dict(path=str(folder),manifest_sha256=file_entry(folder/'manifest.json')['sha256'],projected_collection_seconds=estimate)


def run(mode,out,review_commit,profile=None):
    require(mode in ('profile','collect'),'mode')
    require(isinstance(review_commit,str) and re.fullmatch('[0-9a-f]{40}',review_commit) is not None,'exact review commit required')
    subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    source_inventory=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    profile_binding=None
    if mode=='collect':
        require(profile is not None,'reviewed profile is required')
        profile_binding=qualified_profile(profile,source_inventory)
    else:require(profile is None,'profile command cannot reuse a profile')
    store=Store(out,64*1024**2);started=time.monotonic();stop_requested=[False]
    def emit(kind,payload):store.emit(kind,payload,final=kind=='bootstrap_terminal')
    def stop(signum,frame):stop_requested[0]=True
    old=signal.signal(signal.SIGTERM,stop)
    budget=InterruptibleBudget(8 if mode=='profile' else 100,160,lambda:stop_requested[0])
    runtime=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       source_review_commit=review_commit,sources=source_inventory,mode=mode,input_root=str(INPUT),input_manifest_sha256=INPUT_MANIFEST,
       started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
       thread_env={k:os.environ[k] for k in THREADS},device='cpu',gpu=False,paid_compute=False,
       numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),load_average=os.getloadavg(),
       limits=dict(internal_seconds=8 if mode=='profile' else 100,external_seconds=9 if mode=='profile' else 115,
                   kill_grace_seconds=1 if mode=='profile' else 5,phase_correctors=160,link_attempts=19,raw_bytes=64*1024**2),
       profile_binding=profile_binding)
    store.json('runtime-start.json',runtime)
    result=dict(schema='s1r-bootstrap-association-v1',mode=mode,status='running',bootstraps=[],bridge=[],association=None,
        historical_family_association='unresolved',lower_start_association='not_attempted',evidence='exploratory numerical procedure reproduction; no historical full-state ancestry')
    engine=None
    try:
        net,x,y,theta0,central,data_identity=bound_inputs();budget.check()
        store.emit('bound_inputs',dict(model=asdict(net),x=x,y=y,theta0=theta0,central_pair=central['pair'],
                   central_params=central['params'],identity=data_identity,input_source=INPUT_SOURCE,input_manifest_sha256=INPUT_MANIFEST))
        grid=GridMap(net,x,y,[10.**CENTRAL_HIDDEN],[10.**OUTPUT_LOG])
        if mode=='profile':
            result['timing']=fixed_state_profile(grid,central['pair'][0],budget,store.emit)
            result['status']='profile_complete';return result
        def family(params):return GridMap(net,x,y,[10.**float(params[0])],[10.**float(params[1])]).cell_map(0)
        engine=HiddenBridge(family,Options(),budget,store.emit,19)
        qualified={}
        for start in STARTS:
            budget.check();label='start_'+str(start);case=dict(label=label,params=[start,OUTPUT_LOG],status='running',node=None)
            result['bootstraps'].append(case);store.emit('bootstrap_start',case)
            grid=GridMap(net,x,y,[10.**start],[10.**OUTPUT_LOG])
            screen,seeds=run_bootstrap(grid,theta0,budget,emit,label)
            case.update(period=int(screen['period'][0]),screen_status=int(screen['status'][0]),status='bootstrap_not_period_two')
            if seeds is not None:
                node=engine.node(np.array([start,OUTPUT_LOG]),seeds,'bootstrap:'+label)
                case.update(node=node['id'],status='qualified_pair' if node['accepted'] else 'bootstrap_qualification_failed')
                if node['accepted']:qualified[start]=node
            store.emit('bootstrap_result',case)
        if 1.1 not in qualified:
            result['status']='upper_bootstrap_unqualified';return result
        parent=engine.node(central['params'],central['pair'],'saved_central_requalification')
        result['central']=parent['id']
        if not parent['accepted']:
            result['status']='saved_central_unqualified';return result
        previous=None
        for i in range(1,13):
            target=np.array([1.1 if i==12 else CENTRAL_HIDDEN+(1.1-CENTRAL_HIDDEN)*i/12,OUTPUT_LOG])
            segment,parent,previous=engine.hidden_segment(parent,target,previous)
            segment['macro_index']=i;store.emit('bridge_segment',segment);result['bridge'].append(segment)
            if segment['status']!='connected':
                result['status']='bridge_gap';return result
        result['association']=compare_common_controls(parent,qualified[1.1]);store.emit('association',result['association'])
        result['status']='completed_common_control_comparison';return result
    except PilotStop as e:
        result.update(status='resource_stop',reason=str(e));return result
    except BaseException as e:
        result.update(status='failed_or_interrupted',reason=type(e).__name__+': '+str(e));raise
    finally:
        for case in result['bootstraps']:
            if case['status']=='running':case['status']='interrupted_before_completion'
        signal.signal(signal.SIGTERM,old)
        result.update(phase_corrector_calls=budget.calls,link_attempts=0 if engine is None else len(engine.edges),
                      elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        store.finish(result)
        for p in store.root.iterdir():p.chmod(0o444)
        store.root.chmod(0o555)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('profile','collect'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--review-commit',required=True);p.add_argument('--profile',type=Path);args=p.parse_args()
    result=run(args.mode,args.out,args.review_commit,args.profile)
    print(canonical(result).decode())
    return 0 if result['status'] in ('profile_complete','completed_common_control_comparison','upper_bootstrap_unqualified','saved_central_unqualified','bridge_gap') else 2

if __name__=='__main__':raise SystemExit(main())
