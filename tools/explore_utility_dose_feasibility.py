"""One fixed four-gradient feasibility gate; separate root launch required."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,platform,re,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.period2_branch import PilotStop
from neural_shrimp.period2_branch_io import Store,file_entry,identity
from neural_shrimp.utility_dose import (TOTAL_BYTES,STORE_BYTES,LAUNCHER_BYTES,INPUT_MANIFEST,INPUT_REVIEW,STATE_HASHES,
    require,training_data,plan_array_hash,dose_prediction)
INPUT=Path('/data2/neural-shrimp-data/audits/utility-inputs/attempt-001')
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/period2_branch.py',
 'src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/utility_dose.py','tools/explore_utility_dose_feasibility.py',
 'tools/launch_utility_dose_feasibility.py','docs/plans/utility-dose-feasibility.md')

def sources():
    result={}
    for name in SOURCES:
        require((ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted source '+name)
        result[name]=file_entry(ROOT/name)
    return result

def array_from_record(values,record):
    a=np.ascontiguousarray(values,dtype='<f8')
    require(np.isfinite(a).all(),'finite saved array')
    require(identity(a)=={k:record[k] for k in ('dtype','shape','order','sha256')},'saved array identity')
    require([format(int(v),'016x') for v in a.ravel().view('<u8')]==record['binary64_hex'],'saved array bits')
    return a

def bound_inputs():
    require(file_entry(INPUT/'manifest.json')['sha256']==INPUT_MANIFEST,'input manifest')
    manifest=json.loads((INPUT/'manifest.json').read_text())
    require(set(p.name for p in INPUT.iterdir())==set(manifest['files'])|{'manifest.json'},'exact input inventory')
    for name,e in manifest['files'].items():
        require(Path(name).name==name and not (INPUT/name).is_symlink() and file_entry(INPUT/name)==e,'bound input '+name)
    forks=json.loads((INPUT/'local-forks.json').read_text());require(len(forks)==4,'exact four forks')
    model=json.loads((INPUT/'model-identity.json').read_text())['map']
    require(model['map_id']=='S2-28011' and model['seed']==28011 and model['n_parameters']==13,'fixed development model')
    net=TwoLayerNet(**model['network']);require(net.hidden_parameter_count==9 and net.n_parameters==13,'packing9+4')
    x,y=training_data()
    require(plan_array_hash(x)==model['data_sha256']['x'] and plan_array_hash(y)==model['data_sha256']['y'],'training data exact plan hashes')
    for i,f in enumerate(forks):
        require(f['point']==i//2 and f['array_index']==[4100,i%2] and f['accepted_updates_at_fork']==8202,'fixed fork order/clock')
        f['native_state']=array_from_record(f['state_values'],f['state'])
        f['native_rates']=array_from_record(f['recovery_rate_values'],f['recovery_rates'])
        require(f['state']['sha256']==STATE_HASHES[i] and f['native_state'].shape==(13,) and f['native_rates'].shape==(2,),'fixed fork identity')
        require(f['native_rates'].tobytes()==np.array([10.**float(v) for v in f['recovery_log_rates']]).tobytes(),'recovery rates arithmetic')
    historical=json.loads((INPUT/'result.json').read_text())['historical_warmup']
    require(historical['counted_batched_cell_updates']==1232000 and historical['elapsed_seconds']==63.64927253895439,'historical timing identity')
    projection=1+1000*4*historical['elapsed_seconds']/historical['counted_batched_cell_updates']
    require(np.isfinite(projection) and projection<5,'historical cost screen')
    return net,x,y,forks,dict(projected_seconds=projection,formula='1+1000*4*63.64927253895439/1232000',
        limitation='historical batched-cell update cost, thousandfold factor and1s allowance; not a new exact-workload profile')

def run(out,review_commit):
    require(re.fullmatch('[0-9a-f]{40}',review_commit) is not None,'exact review SHA')
    subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    inventory=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    store=Store(out,STORE_BYTES);started=time.monotonic();stopped=[False];old=signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise PilotStop('external_signal')
        if time.monotonic()-started>=5:raise PilotStop('deadline')
    result=dict(schema='utility-dose-feasibility-v1',status='running',gradient_attempts=0,gradient_returns=0,forks=[],candidate_feasible=None,
                evidence='algebraic feasibility at four saved development states; no state update, pulse, outcome or utility claim')
    try:
        store.json('runtime-start.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            sources=inventory,source_review_commit=review_commit,input_review_commit=INPUT_REVIEW,input_manifest_sha256=INPUT_MANIFEST,
            started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
            thread_env={k:os.environ[k] for k in THREADS},python=platform.python_version(),numpy=np.__version__,load_average=os.getloadavg(),
            limits=dict(internal_seconds=5,external_seconds=10,kill_grace_seconds=5,gradients=4,total_bytes=TOTAL_BYTES,store_bytes=STORE_BYTES,launcher_bytes=LAUNCHER_BYTES)))
        net,x,y,forks,gate=bound_inputs();check()
        store.emit('bound_inputs',dict(model=asdict(net),x=x,y=y,cost_gate=gate,packing=dict(W0=[0,6],b0=[6,9],w1=[9,12],b1=[12,13]),
           forks=[dict(point=f['point'],origin=f['origin'],state=f['native_state'],rates=f['native_rates'],state_sha256=f['state']['sha256']) for f in forks]))
        result['cost_gate']=gate
        for i,f in enumerate(forks):
            check();state=f['native_state'].copy();original=state.copy()
            store.emit('gradient_pending',dict(index=i,point=f['point'],origin=f['origin'],state=state,rates=f['native_rates']))
            result['gradient_attempts']+=1
            gradient=net.gradient(state[None,:],x,y)[0]
            result['gradient_returns']+=1
            store.emit('gradient_return',dict(index=i,gradient=gradient),final=True)
            require(gradient.shape==(net.n_parameters,),'full flat gradient shape')
            require(state.tobytes()==original.tobytes(),'gradient mutated fork state')
            check()
            prediction=dose_prediction(gradient,f['native_rates'],net.hidden_parameter_count)
            store.emit('dose_prediction',dict(index=i,point=f['point'],origin=f['origin'],prediction=prediction))
            result['forks'].append(dict(index=i,status=prediction['status'],reason=prediction['reason']))
        result['candidate_feasible']=all(f['status']=='feasible' for f in result['forks'])
        result['status']='feasible_for_further_design' if result['candidate_feasible'] else 'fixed_candidate_infeasible'
        return result
    except PilotStop as error:
        result.update(status='resource_stop',reason=str(error));return result
    except BaseException as error:
        result.update(status='failed_or_interrupted',reason=type(error).__name__+': '+str(error));raise
    finally:
        signal.signal(signal.SIGTERM,old);result.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        store.finish(result)
        for p in store.root.iterdir():p.chmod(0o444)
        store.root.chmod(0o555)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args()
    result=run(a.out,a.review_commit);print(json.dumps(result,sort_keys=True))
    return 0 if result['status'] in ('feasible_for_further_design','fixed_candidate_infeasible') else 2
if __name__=='__main__':raise SystemExit(main())
