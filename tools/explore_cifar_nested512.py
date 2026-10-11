"""One nested larger-budget comparison at historical018; no step applied."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,math,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
import explore_cifar_reorthogonalized as P
import explore_cifar_stationary_continuation as E
import diagnose_cifar_krylov as D
from neural_shrimp.cifar_reorthogonalized import orthogonalize,SETTINGS
from neural_shrimp.cifar_nested_krylov import extend,qualify
from neural_shrimp.cifar_restart_diagnostic import Store
B=E.B;U=D.U;need=U.need;read=U.read;entry=U.entry
import review_cifar_reorthogonalized as A
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/nested512001')
CARD='docs/plans/cifar-nested512-inputs.json'
CAPS=dict(action=0,j2vp=0,hvp=258,direct_gradient=0)
LIMITS=dict(seconds=300,rss_kib=4*1024**2,as_bytes=12*1024**3,raw_bytes=1024**3)
EXTRA=['tools/explore_cifar_nested512.py','tools/launch_cifar_nested512.py','src/neural_shrimp/cifar_nested_krylov.py','tests/test_cifar_nested_krylov.py','tests/test_cifar_nested512_driver.py','docs/plans/cifar-nested512.md',CARD,'docs/reviews/cifar-reorthogonalized-independent.json']
def install():P.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+tuple(A.FILES)+tuple(EXTRA)))


class Budget:
    def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
    def check(self):
        need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds'],'owned300s deadline')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'owned4GiB RSS ceiling')
    def call(self,kind,function,*args,**kw):
        self.check();need(self.attempted[kind]<CAPS[kind],'fixed native operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]))
        v=function(*args,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));self.check();return v

def check_card(card):
    need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS and card['operation_caps']==CAPS,'fixed reviewed nested card')
    need(card['prior_comparison']==read(ROOT/P.CARD),'literal prior256 comparison card')
    need(card['prefix']==dict(root=str(A.PARENT),source=A.EXEC,outer=A.PINS[0],worker=A.PINS[1]),'sole immutable qualified256 prefix')
    need(card['columns']==dict(old=256,new=256,total=512)and card['state']=='state-018.npy','fixed historical nested count')

def bind(card,inventory,runtime,budget,store):
    parent,index,old,audit,old_events=P.bind(card['prior_comparison'],inventory,runtime,budget,store)
    prefix=card['prefix'];root=Path(prefix['root']);U.inventory(root,prefix['outer'],budget.check);inv=U.inventory(root/'run',prefix['worker'],budget.check);prior=read(root/'run/result.json');U.source(root,prior,prefix['source'])
    need(prior['status']=='reorthogonalized_comparison_complete'and prior['comparison_qualified']and prior['basis']['actions']==prior['basis']['rank']==256 and not prior['direction_applied'],'qualified whole256 basis')
    for n,h in prior['source_sha256'].items():need(inventory[n]==h,'unchanged entire256 native source '+n)
    need(set(prior['runtime'])==set(runtime)and all(prior['runtime'][k]==runtime[k]for k in runtime if k!='pid'),'unchanged native runtime except current PID')
    for k in('data_identity','eta','lam','model'):need(prior['input_identity'][k]==old['input_identity'][k],'unchanged native objective '+k)
    owner=read(root/'launcher-result.json');need(owner['source_commit']==prefix['source']and owner['exit_code']==0 and owner['scientific_complete']and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed256 ownership')
    for k in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(owner[k])).exists(),'actual256 owner absent')
    ip=card['independent'];iroot=Path(ip['root']);U.inventory(iroot,ip['outer'],budget.check);U.inventory(iroot/'run',ip['worker'],budget.check);ir=read(iroot/'run/runtime.json');io=read(iroot/'run/outcome.json');ib=read(iroot/'run/input-binding.json')
    need(ir['source_commit']==ip['source']and io['passed']and io['native_calls']==0 and io['numerical_action_rank']==256 and io['comparison']['qualified'],'independent256 qualification')
    need(ib['producer']==str(root)and ib['producer_source']==prefix['source']and ib['producer_pins']==A.PINS and ib['all_at_historical_state018'],'same independently qualified prefix')
    for n,v in ir['files'].items():need(entry(iroot/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',ip['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'independent committed source '+n)
    owner=read(iroot/'launcher-result.json');need(owner['source_commit']==ip['source']and owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed independent owner')
    for k in('launcher_pid','owned_group_pid',):need(not Path('/proc',str(owner[k])).exists(),'actual independent owner absent')
    events=[json.loads(v)for v in(root/'run/events.jsonl').read_text().splitlines()];counts,calls,rows=A.events_bound(events,inv)
    need(counts==dict(hvp=257,direct_gradient=1)and all(c['state']=='state-000.npy'for c in calls.values()),'all prefix actions same point')
    store.json('nested-binding.json',dict(card=entry(ROOT/CARD),prefix=prefix,independent=ip,counts=counts,old_state='state-018.npy',columns=card['columns'],no_direction_applied=True))
    return parent,index,old,root,calls,rows,prior,events

def forecasts(orth_seconds,algebra_seconds,native_maximum,elapsed,archive_bytes,profile_peak):
    array_bytes=98666*8+128;raw=524*array_bytes+128*1024**2+archive_bytes
    seconds=elapsed+2*(256*orth_seconds+algebra_seconds+258*native_maximum)+30+raw/(50*1024**2)
    need(all(math.isfinite(v)and v>0 for v in(orth_seconds,algebra_seconds,native_maximum)),'finite positive full-size measured costs')
    rss=profile_peak+1024**2
    return dict(passed=seconds<LIMITS['seconds']and raw<LIMITS['raw_bytes']and rss<LIMITS['rss_kib'],forecast_seconds=seconds,forecast_raw_bytes=raw,forecast_peak_rss_kib=rss,maximum_native_seconds=native_maximum,max_width_twopass_seconds=orth_seconds,gram_svd_assembly_seconds=algebra_seconds,profile_and_binding_seconds=elapsed,safety_factor=2,raw_array_count_ceiling=524,write_bytes_per_second=50*1024**2,target_arrays_unopened=True)

def profile(events,budget,store,archive_bytes):
    n,m=98666,512;rng=np.random.default_rng(52512)
    store.json('profile-input.json',dict(seed=52512,rows=n,columns=m,recipe='Fortran512 V=normal/sqrt(n), W=linspace(.5,2,n)*V, w=normal; three full-widthCGS2 then Gram/fullW SVD/Vc/Wc. Also allocate/copy two256 prefixes as in actual extension.',target_arrays_unopened=True))
    V=np.asfortranarray(rng.standard_normal((n,m))/np.sqrt(n));W=np.asfortranarray(np.linspace(.5,2,n)[:,None]*V);pv=V[:,:256].copy(order='F');pw=W[:,:256].copy(order='F');w=rng.standard_normal(n);samples=[]
    for _ in range(3):
        budget.check();t=time.monotonic();q,c=orthogonalize(V,w);samples.append(time.monotonic()-t);need(np.isfinite(q).all(),'finite512 synthetic orthogonalization')
    t=time.monotonic();gram=V.T@V;coef,_,rank,singular=np.linalg.lstsq(W,-V[:,0],rcond=SETTINGS['svd_rcond']);p=V@coef;hp=W@coef;algebra=time.monotonic()-t;budget.check()
    maximum=max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='hvp')
    report=forecasts(max(samples),algebra,maximum,time.monotonic()-budget.start,archive_bytes,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);report.update(samples=samples,synthetic_rank=int(rank),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);store.json('profile.json',report);need(report['passed'],'full-size nested512 admission')
    del V,W,pv,pw,w,q,c,gram,coef,p,hp,singular;gc.collect();return report


def run(review):
    inventory=B.sources();runtime=B.environment();store=Store(BASE/'run',limit=LIMITS['raw_bytes'],reserve=4*1024**2);budget=Budget(store.event)
    def stop(*_):budget.stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,LIMITS['seconds'])
    result=dict(status='running',source_commit=review,source_sha256=inventory,runtime=runtime,settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,direction_applied=False,comparison_qualified=False)
    try:
        need(review==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'exact execution source');card=read(ROOT/CARD);check_card(card);store.json('runtime-start.json',result)
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz')as tar:
            for n in inventory:
                data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',review+':'+n],cwd=ROOT)).hexdigest()==inventory[n],'literal committed own source');i=tarfile.TarInfo(n);i.size=len(data);i.mtime=0;tar.addfile(i,io.BytesIO(data))
        store.write('source.tar.gz',archive.getvalue());parent,index,old,prefix,calls,rows,prior,events=bind(card,inventory,runtime,budget,store);result['profile']=profile(events,budget,store,len(archive.getvalue()))
        folder=prefix/'run';load=lambda n:U.vector(folder/n);x=load('state-000.npy');g=load(calls['direct_gradient',1]['result'])
        need(B.byte_equal(x,U.vector(parent/'run/state-018.npy'))and B.byte_equal(g,U.vector(parent/'run'/index['direct_gradient'][19]['output'])),'literal historical state018 and gradient')
        V=np.empty((98666,256),order='F');W=np.empty_like(V)
        for i in range(256):
            budget.check();V[:,i]=load(calls['hvp',i+1]['direction']);W[:,i]=load(calls['hvp',i+1]['result'])
        first=U.vector(parent/'run'/index['hvp'][774]['direction']);need(B.byte_equal(V[:,0],first),'same literal first native direction')
        q,cs=orthogonalize(V,W[:,-1]);need(cs==rows[-1]['coefficients']and float(np.linalg.norm(q))==rows[-1]['residual_norm'],'literal retained terminal prefix residual');store.json('prefix-terminal-replay.json',dict(coefficients=cs,residual_norm=float(np.linalg.norm(q)),passed=True))
        baseline=float(np.linalg.norm(g+load(calls['hvp',257]['result'])))/float(np.linalg.norm(g));U.close(baseline,prior['comparison']['true_forcing'],'same qualified256 baseline')
        arrays,model,_,_,identity=B.inputs()
        for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'unchanged original objective '+k)
        store.json('input-identity.json',dict(identity=identity,state='state-018.npy',prefix=card['prefix'],baseline_forcing=baseline,direction_applied=False));store.array('historical-gradient',g)
        problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=E.Native(problem,store,budget)
        identity_hp=native.hvp(x,first);need(B.byte_equal(identity_hp,W[:,0]),'byte-exact historical first action replay')
        p,pred,gram,report=extend(g,V,W,lambda v:native.hvp(x,v),max_columns=512,event=lambda k,v:store.event(dict(kind=k,**v)),tick=budget.check)
        for name,v in [('candidate-direction',p),('predicted-action',pred),('final-gram',gram)]:store.array(name,v)
        store.json('basis-report.json',report);actual=native.hvp(x,p);result['comparison']=qualify(g,pred,actual,report,baseline);store.json('comparison.json',result['comparison'])
        need(budget.attempted==budget.returned and budget.returned['hvp']==report['new_basis_actions']+2 and budget.returned['direct_gradient']==0,'all inclusive nested native actions');budget.check();result.update(status='nested512_comparison_complete',comparison_qualified=result['comparison']['qualified'],basis=report,input_identity=identity,historical_actions_reused=256,new_hvp_acquisitions=budget.returned['hvp'],cumulative_actual_hvp_acquisitions=257+budget.returned['hvp'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-budget.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
    print(json.dumps(result));return 0 if result['status']=='nested512_comparison_complete'else 2

def main():
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);return run(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
if __name__=='__main__':raise SystemExit(main())
