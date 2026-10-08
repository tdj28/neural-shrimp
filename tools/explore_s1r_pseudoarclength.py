"""Source-bound two-edge local PAL exploration; separate launch authorization required."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,json,platform,re,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from audit_s1r_branch_outcome import entry,digest,inventory,record,read_json,loss_gradient_hessian
from neural_shrimp.period2_branch_io import Store
from neural_shrimp.s1r_pseudoarclength import Engine,LIMITS,TOL,require,request_stop,PilotStop
PLAN='docs/plans/s1r-pseudoarclength-inputs.json'
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/s1r_gap_diagnostic.py',
 'src/neural_shrimp/s1r_pseudoarclength.py','tools/audit_s1r_branch_outcome.py','tools/explore_s1r_pseudoarclength.py',
 'tools/launch_s1r_pseudoarclength.py',PLAN,'docs/plans/s1r-pseudoarclength-next.md','docs/plans/s1r-pseudoarclength-implementation.md')

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean checkout required')
    bound=read_json(ROOT/PLAN)['bound_repository_files'];names=sorted(set(SOURCES)|set(bound));out={}
    for name in names:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        out[name]=dict(bytes=len(data),sha256=digest(data))
        if name in bound:require(out[name]==bound[name],'immutable design basis '+name)
    return out

def load_inputs():
    plan=read_json(ROOT/PLAN);old=plan['historical_collection'];folder=Path(old['collection_root'])
    require(not folder.is_symlink() and not folder.stat().st_mode&0o222 and all(p.is_file() and not p.is_symlink() and not p.stat().st_mode&0o222 for p in folder.iterdir()),'immutable historical inventory')
    inventory(folder,old['collection_manifest']['sha256'])
    for name,pin in old['collection_metadata'].items():require(entry(folder/name)==pin,'historical metadata')
    runtime=read_json(folder/'runtime-start.json');require(runtime['source_commit']==old['producer_execution_commit'],'historical execution')
    for name,pin in runtime['sources'].items():
        for revision in (old['producer_execution_commit'],old['producer_reviewed_source']):
            data=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT);require(pin==dict(bytes=len(data),sha256=digest(data)),'historical source')
    decoded={}
    for name,pin in plan['selected_records'].items():
        require(entry(folder/name)==pin['json'] and entry(folder/pin['arrays_file'])==pin['arrays'],'selected historical files')
        decoded[name]=record(folder,name)[1]
    binding=decoded['00000-bound_inputs.json'];pre=decoded['04820-node.json'];last=decoded['04918-node.json'];edge=decoded['05002-edge.json'];failure=decoded['05267-edge.json']
    require(binding['model']==dict(activation='tanh',bias=False,d=8,in_scale=.5,loss_scale=1.,out_scale=.125,ridge=.001,width=8),'exact S1R model')
    require(pre['accepted'] and last['accepted'] and edge['status']=='accepted' and edge['parent']==pre['id'] and edge['direct']==last['id'],'qualified starting lineage')
    carried=edge['parent_match']['phase'];oldphase=edge['parent_phase']
    require(type(carried) is int and type(oldphase) is int and carried in (0,1) and oldphase in (0,1) and failure['parent']==last['id'] and failure['parent_phase']==carried,'native carry binding')
    require(last['params'][0]==.992 and last['params'][1]==.6271428571428572,'fixed anchor controls')
    require(last['rates'].tobytes()==np.array([10.**float(v) for v in last['params']]).tobytes(),'exact anchor rate rule')
    for summary in plan['gap_diagnostic']['raw'].values():inventory(Path(summary['root']),summary['manifest']['sha256'])
    audit=plan['gap_audit'];inventory(Path(audit['root']),audit['manifest_sha256'])
    require(entry(ROOT/plan['analytic_source']['file'])==plan['analytic_source']['identity'],'unchanged analytic primitive')
    secant=np.r_[last['pair'][carried]-pre['pair'][oldphase],last['params'][0]-pre['params'][0]]
    saved=plan['initial_raw_tangent_summaries']['native_phase_'+str(carried)]['summary']['null_tangent']['h_component']
    return dict(plan=plan,model=binding['model'],identity=binding['identity'],x=binding['x'],y=binding['y'],pair=last['pair'],params=last['params'],rates=last['rates'],carried_phase=carried,secant=secant,saved_h_component=saved)

def run(out,review_commit):
    src=sources();require(re.fullmatch('[0-9a-f]{40}',review_commit) is not None,'full review SHA');subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in THREADS),'CPU1 nice10')
    store=Store(out,LIMITS['store_bytes']);started=time.monotonic();engine=None
    result=dict(schema='s1r-pseudoarclength-v1',status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),review_commit=review_commit,sources=src,limits=LIMITS,tolerances=TOL,fold_verified=False,lower_association=False,accepted_edges=[],last_witnessed=None)
    try:
        store.json('runtime-start.json',dict(**result,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),started_utc=datetime.now(timezone.utc).isoformat(),gpu=False,paid=False))
        inputs=load_inputs();store.json('input-plan.json',inputs.pop('plan'));store.emit('bound_inputs',inputs)
        screen=2*(1.1480005020275712*max(256/12,102/24)+38.103352084057406*(16/473))+5
        store.emit('cost_gate',dict(projected_seconds=screen,limit_seconds=60.,passed=bool(screen<=60.),scope='historical heuristic only'))
        require(screen<=60.,'historical cost screen')
        derivative=lambda theta:loss_gradient_hessian(theta,inputs['x'],inputs['y'],inputs['model'])
        engine=Engine(derivative,64,float(inputs['params'][1]),store.emit,seconds=max(.000001,60.-(time.monotonic()-started)))
        result.update(engine.run(inputs['pair'],float(inputs['params'][0]),inputs['carried_phase'],inputs['secant'],inputs['saved_h_component']))
    except PilotStop as error:result.update(status='stopped',error=str(error))
    except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error));raise
    finally:
        if engine is not None:result.update(operation_attempted=engine.ops.attempted,operation_returned=engine.ops.returned,stage_counts=engine.counts,last_witnessed=engine.last_witnessed,accepted_edges=engine.durable_edges)
        else:result.update(operation_attempted={},operation_returned={},stage_counts={})
        result.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat());store.finish(result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--review-commit',required=True);args=parser.parse_args();signal.signal(signal.SIGTERM,request_stop)
    result=run(args.out,args.review_commit);print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
