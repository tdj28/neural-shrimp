"""Fixed checkpoint inference precision pilot; no training or unseen text."""
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.tiny_lm import TinyTransformer
RAW=Path('/data2/neural-shrimp-data/explore/tiny-lm-prediction/attempt-002-calibration')
PIN='a8a4166cb0b03c5f8b72b0a8039dfb99acc0e4ad1fc12a53d4d3371f0b0d4bff'
BATCH=Path('/data2/neural-shrimp-data/explore/tiny-lm/attempt-001-unit')
BATCH_PIN='ef17075eafaab531914245efadf43024aa51f135ed97bde6f3f4ecf7ac68d92c'
MODEL_PIN='27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8'
PAIRS=[(s,f) for s in (28151,28152) for f in (.8,1.6,2.5)]
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/explore_saved_lm_inference.py','tools/launch_saved_lm_inference.py','src/neural_shrimp/tiny_lm.py','tests/test_saved_lm_inference.py','docs/plans/saved-lm-inference-precision.md']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAP=16*1024**2

def require(b,s):
    if not b:raise ValueError(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return dict(sha256=sha(p),bytes=p.stat().st_size)
def total(out):return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
def save(p,x):
    b=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();require(len(b)<1024**2,'JSON cap')
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def bound_sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
    ans={}
    for n in FILES:
        b=(ROOT/n).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT),'committed source');ans[n]=hashlib.sha256(b).hexdigest()
    require(ans['src/neural_shrimp/tiny_lm.py']==MODEL_PIN,'unchanged trained model')
    return ans

def gap_metrics(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    require(a.shape==b.shape and a.ndim>=2 and a.shape[-1]>=2,'logit shape')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'finite logits')
    err=np.max(np.abs(a-b),axis=-1);ordered=np.sort(a,axis=-1);gap=ordered[...,-1]-ordered[...,-2]
    same=np.argmax(a,axis=-1)==np.argmax(b,axis=-1);sufficient=gap>2*err
    require(np.all(same[sufficient]),'measured gap inequality contradicted')
    return dict(max_logit_difference=float(err.max()),min_reference_margin=float(gap.min()),decisions=int(same.size),changed=int((~same).sum()),measured_gap_sufficient=int(sufficient.sum()),scope='computed pairwise margin screen, not interval error bound or general robustness certificate')

def first_difference(a,b):
    require(a.shape==b.shape and a.ndim==2,'step by batch tokens')
    return [int(np.flatnonzero(a[:,i]!=b[:,i])[0]) if np.any(a[:,i]!=b[:,i]) else None for i in range(a.shape[1])]

def inputs():
    require(sha(RAW/'manifest.json')==PIN,'calibration manifest')
    m=json.loads((RAW/'manifest.json').read_text());entries=m['files'];names=[e['path'] for e in entries]
    require(len(names)==len(set(names)) and set(names)=={str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p.name!='manifest.json'},'exact inventory')
    for e in entries:
        p=RAW/e['path'];require(not Path(e['path']).is_absolute() and '..' not in Path(e['path']).parts and not p.is_symlink(),'safe member')
        require(entry(p)=={k:e[k] for k in ('sha256','bytes')},'input bytes')
    data=json.loads((RAW/'data.json').read_text());require(data==json.loads((BATCH/'data.json').read_text()),'identical consumed training batch metadata')
    require(data['training_character_offsets']==[4096*j for j in range(8)] and data['context']==32,'consumed prefix restriction')
    require(sha(BATCH/'batch.npz')==BATCH_PIN,'retained training batch')
    with np.load(BATCH/'batch.npz',allow_pickle=False) as z:tokens=z['tokens'].copy()
    require(tokens.shape==(8,32) and tokens.dtype==np.int64 and np.all((tokens>=0)&(tokens<len(data['vocabulary']))),'tokens')
    cases=[]
    for seed,factor in PAIRS:
        d=RAW/f'seed-{seed}-factor-{factor:g}';summary=json.loads((d/'result.json').read_text())
        require(summary['seed']==seed and summary['factor']==factor and summary['outcome']=='completed' and summary['completed_updates']==800,'fixed complete checkpoint')
        p=d/'states-000800-000800.npy';v=np.load(p,allow_pickle=False)
        require(v.shape==(1,15776) and v.dtype==np.float32 and np.isfinite(v).all(),'final float32 checkpoint')
        cases.append((seed,factor,v[0].copy(),dict(path=str(p),**entry(p))))
    return data,tokens,cases

def run(out):
    source=bound_sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(n)=='1' for n in THREADS),'CPU1 nice10')
    require(int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))>2*1024**2,'RAM headroom')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;returned=0;result={'status':'incomplete','cases':[]}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def stop(*_):raise TimeoutError('signal deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
    def check():
        require(time.monotonic()-start<90,'90s worker cap');require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'2GiB RSS cap');require(total(out)<CAP-1024**2,'ordinary raw cap')
    def forward(model,theta,tokens,tag):
        nonlocal calls,returned
        check();calls+=1;require(calls<=780,'780 attempted forward cap')
        save(out/f'forward-{calls:04d}-attempt.json',dict(tag=tag,dtype=str(theta.dtype),attempt=calls))
        with torch.inference_mode():z=model.forward(theta,tokens)[:,-1,:].clone()
        returned+=1
        # Persist each returned output before qualification, including NaN/Inf.
        path=out/f'forward-{calls:04d}-return.npy';array=z.numpy()
        require(total(out)+array.nbytes+256<CAP-1024**2,'return prewrite cap')
        with path.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        require(bool(torch.isfinite(z).all()),'finite forward return');return z
    save(out/'runtime-start.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=source,pid=os.getpid(),started_unix=time.time(),python=sys.version,torch=torch.__version__,numpy=np.__version__,threads={n:os.environ[n] for n in THREADS},nice=os.getpriority(os.PRIO_PROCESS,0),cpu_only=True,training_calls=0))
    for n in FILES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        # Full-size synthetic input and weights; no target checkpoint before the cost gate.
        model=TinyTransformer(66);theta=model.initialize(28155);x=torch.arange(256).reshape(8,32)%66;times=[]
        for dtype in (torch.float32,torch.float64):
            for _ in range(6):
                t=time.monotonic();forward(model,theta.to(dtype),x,dict(mode='synthetic_profile',repeat=_));times.append(time.monotonic()-t)
        projected=2*max(times)*768+5;save(out/'profile.json',dict(forward_seconds=times,projected_seconds=projected,gate=projected<90,target_unopened=True));require(projected<90,'measured timing gate')
        data,prefix,cases=inputs();require(model.n_parameters==15776 and len(data['vocabulary'])==66,'same model dimensions')
        save(out/'input-binding.json',dict(calibration_manifest=PIN,batch_sha256=BATCH_PIN,prefix_offsets=data['training_character_offsets'],checkpoints=[c[3] for c in cases],scope='all six final calibration checkpoints and already consumed eight training prefixes; no source corpus opened'))
        for seed,factor,array,binding in cases:
            p32=torch.from_numpy(array);p64=p32.double();x32=torch.from_numpy(prefix.copy());x64=x32.clone();traces={k:[] for k in ('logits32','repeat32','logits64_matched','logits64_free','tokens32','tokens64','same_history')}
            for step in range(32):
                tag=dict(seed=seed,factor=factor,step=step)
                a=forward(model,p32,x32,dict(tag,mode='float32'));repeat=forward(model,p32,x32,dict(tag,mode='float32_repeat'));matched=forward(model,p64,x32,dict(tag,mode='float64_matched'));free=forward(model,p64,x64,dict(tag,mode='float64_free'))
                require(torch.equal(a,repeat),'same arithmetic repeat differs')
                ta=a.argmax(-1);tb=free.argmax(-1)
                for key,val in [('logits32',a),('repeat32',repeat),('logits64_matched',matched),('logits64_free',free),('tokens32',ta),('tokens64',tb),('same_history',(x32==x64).all(-1))]:traces[key].append(val.numpy().copy())
                x32=torch.cat([x32[:,1:],ta[:,None]],dim=1);x64=torch.cat([x64[:,1:],tb[:,None]],dim=1)
            traces={k:np.stack(v) for k,v in traces.items()};traces['prefix_tokens']=prefix
            require(sum(v.nbytes for v in traces.values())+total(out)<CAP-1024**2,'prewrite array cap')
            file=out/f'seed-{seed}-factor-{factor:g}.npz'
            with file.open('xb') as f:np.savez(f,**traces);f.flush();os.fsync(f.fileno())
            row=dict(seed=seed,factor=factor,checkpoint=binding,array=entry(file),array_file=file.name,repeat_exact=True,matched_history=gap_metrics(traces['logits32'],traces['logits64_matched']),first_greedy_divergence=first_difference(traces['tokens32'],traces['tokens64']),greedy_token_mismatches=int(np.sum(traces['tokens32']!=traces['tokens64'])),identical_prefix_decisions=int(traces['same_history'].sum()))
            result['cases'].append(row);save(out/f'case-{seed}-{factor:g}.json',row)
        result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,attempted_forward_calls=calls,returned_forward_calls=returned,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='exploratory CPU float32-versus-float64 arithmetic sensitivity at identical float32-rounded frozen weights. Finite greedy rollout, no inference chaos, training-causation, task-accuracy, generalization or shrimp-specific claim. No training/assessment/validation/controller.')
        save(out/'result.json',result);save(out/'manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result));return 0 if result['status']=='completed' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.exit(run(a.out))
