"""Exactly four development center gradients and eight paired one-step losses."""
import argparse,hashlib,json,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.utility_common_center import evaluate
from neural_shrimp.utility_dose_audit import inventory,bound_original_inputs,bit_equal,STATE_HASHES
from neural_shrimp.period2_branch_io import Store,load_record,file_entry,identity
RAW=Path('/data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001');PIN='4589dc9afeaa098537341adf27ae230a64110a8fdaf048baf7f8fef22c6d19be'
PAIRED=Path('/data2/neural-shrimp-data/explore/utility-paired-feasibility/attempt-001');PAIRPIN='9a3fb7c7c3f5fc9595b4f3325541a015f9c11748fb7afd9cc0927e2988e3c02d'
FILES=['src/neural_shrimp/'+n for n in ['__init__.py','controls.py','models.py','gdmaps.py','period2_branch.py','period2_branch_io.py','utility_dose_audit.py','utility_paired_geometry.py','utility_common_center.py']]+['tools/explore_utility_common_center.py','tools/launch_utility_common_center.py','tests/test_utility_common_center.py','docs/plans/utility-common-center.md']
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def require(ok,message):
    if not ok:raise ValueError(message)
def nested_inventory(folder,pin):
    require(file_entry(folder/'manifest.json')['sha256']==pin,'paired manifest pin');m=json.loads((folder/'manifest.json').read_text())['files'];require(set(m)=={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()and p!=folder/'manifest.json'},'paired exact inventory')
    for n,d in m.items():require(not Path(n).is_absolute()and '..'not in Path(n).parts and not(folder/n).is_symlink()and file_entry(folder/n)==d,'paired bytes')
    return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
    store=Store(a.out,limit=1024**2);started=time.monotonic();stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));counts=dict(gradient_attempts=0,gradient_returns=0,loss_attempts=0,loss_returns=0);result=dict(status='incomplete',rows=[],counts=counts,scope='Four consumed development forks; exact fixed one-step arithmetic only, not tails, qualified distant controls, geometry-specific utility, controller or protected assessment.')
    def check():
        if stop[0]or time.monotonic()-started>5:raise TimeoutError('5s/termination')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB cooperative worker')
    store.json('runtime-start.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={n:file_entry(ROOT/n)for n in FILES},pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k]for k in THREADS},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,started_unix=time.time()))
    try:
        model=TwoLayerNet(2,3,ridge=.001);tx=np.arange(48,dtype=float).reshape(24,2)/48-.5;ty=np.sin(np.arange(24));tt=np.linspace(-.3,.4,13);tg=np.arange(1.,14.)*.001;tr=np.array([10.,1.]);start=time.monotonic()
        for _ in range(100):evaluate(tt,tg,tr,lambda z:model.gradient(z[None],tx,ty)[0],lambda z:model.loss(z[None],tx,ty)[0],check=check)
        seconds=time.monotonic()-start;projection=40*seconds/100+1;store.emit('synthetic_profile',dict(cases=100,gradients=100,losses=200,seconds=seconds,projected_seconds=projection));require(np.isfinite(projection)and projection<5,'measured profile gate');check()
        inventory(RAW,PIN,1024**2);nested_inventory(PAIRED,PAIRPIN);x,y,forks,original_inventory=bound_original_inputs();store.emit('bound_inputs',dict(x=x,y=y,forks=forks,input_manifest=PIN,paired_manifest=PAIRPIN,original_input_inventory=original_inventory,model=vars(model)))
        for i,f in enumerate(forks):
            check();kind,p=load_record(RAW,f'{2+2*i:05d}-independent_gradient_return.json');require(kind=='independent_gradient_return'and p['index']==i and identity(p['state'])['sha256']==STATE_HASHES[i]and bit_equal(p['state'],f['state'])and bit_equal(p['rates'],f['rates']),'fixed gradient/state/rates')
            old=json.loads((PAIRED/f'fork-{i}.json').read_text());require(old['qualified']and old['gradient_identity']==identity(p['gradient'])and old['original_state_sha256']==STATE_HASHES[i],'paired input gradient binding')
            def emit(kind,value):store.emit(kind,dict(index=i,**value))
            def gradient(z):
                counts['gradient_attempts']+=1;require(counts['gradient_attempts']<=4,'four gradient cap');value=model.gradient(z[None],x,y)[0];counts['gradient_returns']+=1;return value
            def loss(z):
                counts['loss_attempts']+=1;require(counts['loss_attempts']<=8,'eight loss cap');value=model.loss(z[None],x,y)[0];counts['loss_returns']+=1;return value
            r=evaluate(f['state'],p['gradient'],f['rates'],gradient,loss,emit,check);r.update(index=i,point=f['point'],origin=f['origin']);result['rows'].append(r)
        require(counts==dict(gradient_attempts=4,gradient_returns=4,loss_attempts=8,loss_returns=8),'all four exact operations');result['status']='completed';check()
    except BaseException as e:result.update(status='failed_or_interrupted',failure=type(e).__name__+': '+str(e))
    finally:result.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);store.finish(result)
    print(json.dumps(result));return 0 if result['status']=='completed'else 2
if __name__=='__main__':raise SystemExit(main())
