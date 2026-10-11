"""Fixed causal-horizon development comparison, with prediction sealed first."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import argparse,hashlib,io,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import explore_tiny_lm_response as B
from neural_shrimp.tiny_lm import TinyTransformer,LMProblem
from neural_shrimp.tiny_lm_horizon import ORIGINS,predict,evaluate,summarize
need=B.need
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-horizon')
RESPONSE=Path('/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-target')
RESPONSE_PIN='fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5'
RESPONSE_WORKER='cb57135497afa72437651b7eaab807e58387a304fd0f0d2609113ffcff073aad'
AUDITS=[(Path('/data2/neural-shrimp-data/audits/tiny-lm-response-independent/attempt001'),'296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948'),
        (Path('/data2/neural-shrimp-data/audits/tiny-lm-two-snapshot-independent/attempt001'),'7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d')]
ORIGINAL_SOURCE='b4c3d588da671f25a7fa22c6882e1cdcb42d2079'
ORIGINAL_EXEC='081c5dcc779c3b2f60d0a3102275f0fa7609efbf'
FILES=B.SOURCES+['src/neural_shrimp/'+n+'.py'for n in('__init__','controls','models','tiny_lm_horizon')]+[
    'tools/explore_tiny_lm_horizon.py','tools/launch_tiny_lm_horizon.py',
    'tests/test_tiny_lm_horizon.py','tests/test_tiny_lm_horizon_driver.py',
    'docs/plans/tiny-lm-causal-horizon.md','docs/plans/tiny-lm-horizon-source-provenance.json','docs/reviews/tiny-lm-horizon-profile001-failure.json']


def forecast(rows):
    need(len(rows)==2 and [r['arm']for r in rows]==[8,256],'two fixed profile arms')
    estimate=20.
    for row in rows:
        need(set(row['times'])=={'hvp','ce'},'only two measured operations')
        for kind,count in(('hvp',36),('ce',16)):
            v=row['times'][kind];need(len(v)==2 and all(type(x)is float and np.isfinite(x)and x>0 for x in v),'two finite measured costs')
            estimate+=2*count*max(v)
    return estimate


def immutable(root,pin):
    inv=B.inventory(root,pin)
    legacy=(root,pin)in B.PARENTS.values()
    need((legacy or not root.stat().st_mode&0o222)and not(root/'manifest.json').stat().st_mode&0o222 and all(not(root/n).stat().st_mode&0o222 for n in inv),'closed read-only input files and qualified container')
    return dict(root=str(root),manifest=pin,files=len(inv),bytes=sum(e['bytes']for e in inv.values()),directory_mode=oct(root.stat().st_mode&0o777),historical_writable_container=bool(legacy and root.stat().st_mode&0o222))


class Work:
    def __init__(self,out,mode):
        self.out=out;out.mkdir(parents=True,exist_ok=False);self.mode=mode;self.start=time.monotonic();self.used=0
        self.limit=60 if mode=='profile'else 180;self.cap=dict(hvp=4,ce=4)if mode=='profile'else dict(hvp=72,ce=32)
        self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0);self.events=[];self.times={k:[]for k in self.cap}
    def check(self):
        need(time.monotonic()-self.start<self.limit and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'fixed time/4GiB RSS')
    def write(self,name,data,closing=False):
        need(self.used+len(data)<(128 if closing else 124)*1024**2,'128MiB with4MiB closing reserve')
        p=self.out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        self.used+=len(data)
    def json(self,name,data,closing=False):self.write(name,(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
    def array(self,name,v):
        b=io.BytesIO();np.save(b,np.asarray(v),allow_pickle=False);self.write(name+'.npy',b.getvalue())
    def call(self,kind,problem,theta,vector=None):
        self.check();need(self.attempted[kind]<self.cap[kind],'fixed native operation cap')
        index=self.attempted[kind]+1;prefix=f'operations/{kind}-{index:03d}';self.array(prefix+'-theta',theta)
        if kind=='hvp':self.array(prefix+'-vector',vector)
        self.attempted[kind]=index;event=dict(operation=kind,index=index,returned=False)
        self.events.append(event);self.json(prefix+'-attempt.json',event);t=time.monotonic();x=torch.as_tensor(theta,dtype=torch.float64)
        value=problem.hvp(x,torch.as_tensor(vector,dtype=torch.float64)).detach().numpy().copy()if kind=='hvp'else float(problem.cross_entropy(x))
        elapsed=time.monotonic()-t;self.array(prefix+'-return',value);need(np.isfinite(value).all(),'finite native return')
        self.returned[kind]+=1;self.times[kind].append(elapsed);event.update(returned=True,seconds=elapsed)
        self.json(prefix+'-result.json',event);self.check();return value


def finite_array(path,shape):
    a=np.load(path,allow_pickle=False);need(a.shape==shape and a.dtype==np.float64 and np.isfinite(a).all(),'complete actual float64 retained array '+str(path));return a


def source(w):
    identity={n:B.entry(ROOT/n)for n in FILES}
    for n,e in identity.items():
        data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==e['sha256'],'exact committed source '+n);w.write('source/'+n,data)
    need(identity['src/neural_shrimp/tiny_lm.py']['sha256']==B.NATIVE_SHA,'unchanged original native LM')
    provenance=B.load(ROOT/'docs/plans/tiny-lm-horizon-source-provenance.json')
    need(provenance['original_source_commit']==ORIGINAL_SOURCE and provenance['files']=={n:identity[n]for n in B.SOURCES},'literal original dependency provenance')
    for n,e in provenance['files'].items():need(hashlib.sha256(subprocess.check_output(['git','show',ORIGINAL_SOURCE+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'historical executed dependency')
    return identity


def bind_response(w,identity):
    inv=[immutable(RESPONSE,RESPONSE_PIN),immutable(RESPONSE/'run',RESPONSE_WORKER)]
    old=B.load(RESPONSE/'run/result.json')
    need(old['passed']and old['status']=='response_complete'and old['source_commit']==ORIGINAL_EXEC
         and old['operation_attempted']==old['operation_returned']==dict(gradient=580,hvp=256,ce=96)
         and old['source_identity']=={n:identity[n]for n in B.SOURCES}
         and old['torch']==torch.__version__ and old['numpy']==np.__version__,'complete native/source/library-qualified response')
    for root,pin in AUDITS:
        inv.append(immutable(root,pin));need(B.load(root/'run/result.json')['passed'],'independent prior outcome audit')
    w.json('response-binding.json',dict(inventories=inv,source=ORIGINAL_EXEC,scope='Consumed development only; no protected arrays or new trajectories.'))


def build_predictions(w,parent):
    records=[]
    for seed in(28161,28162):
        for arm in(8,256):
            theta,x,y,eta,binding=B.select(parent,'target',seed,arm,600);label=f'seed-{seed}-arm-{arm}';folder=RESPONSE/'run'/label
            reference=finite_array(folder/'reference.npy',(17,15776));gradients=finite_array(folder/'reference-gradients.npy',(17,15776))
            need(np.array_equal(reference[0],theta),'same promoted initial state');need(np.array_equal(reference[1:],reference[:-1]-eta*gradients[:-1]),'entire source-authenticated reference recurrence')
            original=B.load(folder/'inputs.json');need(binding==original,'actual data/rate/coordinate binding')
            w.json(label+'/inputs.json',binding);problem=LMProblem(TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
            tangent=[finite_array(folder/f'direction-{i}-transported.npy',(17,15776))for i in range(2)]
            frozen0=[finite_array(folder/f'direction-{i}-frozen.npy',(17,15776))for i in range(2)]
            for t in ORIGINS:
                # Only these two local states/gradients are passed to native predictors.
                early_states=reference[t:t+2].copy();early_gradients=gradients[t:t+2].copy()
                losses=[w.call('ce',problem,v)+.0005*float(np.dot(v,v))for v in early_states]
                for direction in range(2):
                    v=tangent[direction];scale=float(np.linalg.norm(v[t]));need(np.isfinite(scale)and scale>0,'nonzero actual current tangent')
                    u,v1,a2=[q/scale for q in v[t:t+3]]
                    frozen=[u.copy(),v1.copy()]
                    if t==0:
                        frozen=list(frozen0[direction][:5]/scale);need(np.array_equal(frozen[0],u)and np.array_equal(frozen[1],v1),'literal original frozen first action')
                    else:
                        for _ in range(3):frozen.append(frozen[-1]-eta*w.call('hvp',problem,early_states[0],frozen[-1]))
                    frozen=np.asarray(frozen);need(frozen.shape==(5,15776)and np.isfinite(frozen).all(),'complete frozen local forecast')
                    rowlabel=label+f'/origin-{t:02d}-direction-{direction}'
                    early=np.array([u,v1,a2,frozen[2],early_gradients[0],early_gradients[1]])
                    w.array(rowlabel+'/early-vectors',early);w.array(rowlabel+'/early-states',early_states);w.array(rowlabel+'/frozen',frozen)
                    prediction=predict(*early,*losses)
                    record=dict(seed=seed,arm=arm,origin=t,direction=direction,label=rowlabel,normalization=scale,
                        losses=losses,prediction=prediction,causal_inputs='state/gradient indices t,t+1; tangent t,t+1,t+2; the latter uses only Jacobian at state t+1; frozen Jacobian only at t')
                    w.json(rowlabel+'/prediction.json',record);records.append(record)
    need(len(records)==32 and w.attempted==w.returned==w.cap,'all fixed predictions and exact72HVP/32CE completed')
    closed={r['label']:{n:B.entry(w.out/r['label']/n)for n in('prediction.json','early-vectors.npy','early-states.npy','frozen.npy')}for r in records}
    w.json('predictions-closed.json',dict(count=32,files=closed,evaluation_arrays_unopened=True))
    return records


def evaluate_predictions(w,records):
    need((w.out/'predictions-closed.json').exists(),'all predictors sealed before evaluation')
    rows=[]
    for record in records:
        w.check();seed,arm,direction,t=[record[k]for k in('seed','arm','direction','origin')]
        folder=RESPONSE/'run'/f'seed-{seed}-arm-{arm}';label=record['label'];scale=record['normalization']
        eps=B.load(folder/'outcome.json')['epsilon_values'];need(len(eps)==2 and all(np.isfinite(x)and x>0 for x in eps)and eps[0]==2*eps[1],'two original perturbation amplitudes')
        responses=[]
        for k in range(2):
            plus=finite_array(folder/f'direction-{direction}-epsilon-{k}-plus.npy',(17,15776));minus=finite_array(folder/f'direction-{direction}-epsilon-{k}-minus.npy',(17,15776))
            responses.append((plus[t:t+5]-minus[t:t+5])/(2*eps[k]*scale))
        ordered=finite_array(folder/f'direction-{direction}-transported.npy',(17,15776))[t:t+5]/scale
        frozen=finite_array(w.out/label/'frozen.npy',(5,15776));u=finite_array(w.out/label/'early-vectors.npy',(6,15776))[0]
        answer=evaluate(record['prediction'],u,frozen,ordered,*responses)
        for name,arr in [('response-large',responses[0]),('response-small',responses[1])]:w.array(label+'/'+name,arr)
        row=dict(**record,evaluation=answer,ordered_source=dict(path=str(folder/f'direction-{direction}-transported.npy'),identity=B.entry(folder/f'direction-{direction}-transported.npy'),slice=[t,t+5],normalization=scale));w.json(label+'/evaluation.json',row);rows.append(row)
    return rows


def main(args):
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
    need(args.out==BASE/f'attempt002-{args.mode}'/'run','fixed exclusive output')
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    w=Work(args.out,args.mode);result=dict(passed=False,status='incomplete',mode=args.mode,protected_outcomes_evaluated=False)
    def stop(*_):raise TimeoutError('fixed horizon owned deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,w.limit)
    try:
        identity=source(w);runtime=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=identity,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pid=os.getpid(),dtype='float64',device='cpu',threads=dict.fromkeys(THREADS,'1'),operation_caps=w.cap)
        w.json('runtime.json',runtime);failed=BASE/'attempt001-profile';immutable(failed,'b5475b36a5e22d5381473fe490bedfbfcf422e01ab8880bccf40d6e1891925a7');failure=B.load(failed/'run/result.json');need(not failure['passed']and failure['error']=='ValueError: closed immutable input'and failure['operation_attempted']==failure['operation_returned']==dict(hvp=0,ce=0),'literal preserved zero-native legacy-container failure');w.json('prior-failure.json',dict(root=str(failed),outer='b5475b36a5e22d5381473fe490bedfbfcf422e01ab8880bccf40d6e1891925a7'));parent,pin=B.PARENTS[args.mode];w.json('parent-binding.json',immutable(parent,pin))
        if args.mode=='profile':
            costs=[]
            for arm in(8,256):
                theta,x,y,eta,binding=B.select(parent,'profile',28160,arm,50);w.json(f'arm-{arm}-inputs.json',binding)
                problem=LMProblem(TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
                direction=np.random.default_rng(28166).standard_normal(15776);times={k:[]for k in('hvp','ce')}
                for _ in range(2):
                    for k in('hvp','ce'):w.call(k,problem,theta,direction if k=='hvp'else None);times[k].append(w.times[k][-1])
                costs.append(dict(arm=arm,times=times))
            estimate=forecast(costs);need(estimate<180,'fixed measured180s target admission')
            result.update(passed=True,status='profile_complete',profile_costs=costs,forecast_seconds=estimate)
        else:
            profile=BASE/'attempt002-profile';need(args.profile_outer and args.profile_worker,'both completed profile manifests')
            immutable(profile,args.profile_outer);immutable(profile/'run',args.profile_worker);prior=B.load(profile/'run/result.json');pr=B.load(profile/'run/runtime.json');la=B.load(profile/'launcher-result.json')
            need(prior['passed']and prior['status']=='profile_complete'and la['exit_code']==0 and not la['owned_live_pids']and la['worker_pid_absent'],'complete owned profile')
            need(pr['sources']==identity and all(pr[k]==runtime[k]for k in('python','numpy','torch','dtype','device','threads')),'same source/runtime profile')
            need(prior['operation_attempted']==prior['operation_returned']==dict(hvp=4,ce=4)and forecast(prior['profile_costs'])==prior['forecast_seconds']<180,'exact recomputed measured gate')
            for pid in(pr['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'profile owned process absent')
            w.json('profile-binding.json',dict(root=str(profile),outer=args.profile_outer,worker=args.profile_worker,forecast_seconds=prior['forecast_seconds']))
            bind_response(w,identity);records=build_predictions(w,parent);rows=evaluate_predictions(w,records)
            result.update(passed=True,status='causal_horizon_complete',rows=rows,summary=summarize(rows))
        need(w.attempted==w.returned==w.cap,'complete fixed operation counts');w.check()
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=w.cap,operation_attempted=w.attempted,operation_returned=w.returned)
        w.json('events.json',w.events,True);w.json('result.json',result,True);w.json('manifest.json',dict(files={str(p.relative_to(w.out)):B.entry(p)for p in w.out.rglob('*')if p.is_file()}),True)
    print(json.dumps({k:v for k,v in result.items()if k!='rows'}));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
