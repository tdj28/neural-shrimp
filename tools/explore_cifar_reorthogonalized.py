"""One equal-action comparison at the exact consumed state018; no step applied."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,math,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_stationary_continuation as E
import diagnose_cifar_krylov as D
from neural_shrimp.cifar_reorthogonalized import solve,qualify,orthogonalize,SETTINGS
from neural_shrimp.cifar_restart_diagnostic import Store
B=E.B;U=D.U;need=U.need;read=U.read;entry=U.entry
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/reorthogonalized001')
CARD='docs/plans/cifar-reorthogonalized-inputs.json'
CAPS=dict(action=0,j2vp=0,hvp=257,direct_gradient=1)
LIMITS=dict(seconds=300,rss_kib=3*1024**2,as_bytes=8*1024**3,raw_bytes=768*1024**2)
EXTRA=['tools/explore_cifar_reorthogonalized.py','tools/launch_cifar_reorthogonalized.py','src/neural_shrimp/cifar_reorthogonalized.py','tests/test_cifar_reorthogonalized.py','tests/test_cifar_reorthogonalized_driver.py','docs/plans/cifar-reorthogonalized-comparison.md',CARD,'docs/reviews/cifar-krylov-conditioning001.json']
def install():E.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+tuple(D.FILES)+tuple(EXTRA)))

class Budget:
    def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
    def check(self):
        need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds'],'owned300s deadline')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'owned3GiB RSS ceiling')
    def call(self,kind,function,*args,**kw):
        self.check();need(self.attempted[kind]<CAPS[kind],'fixed native operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]))
        v=function(*args,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));self.check();return v

def check_card(card):
    need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS and card['operation_caps']==CAPS,'complete fixed reviewed comparison card')
    need(card['operator']==dict(state='state-018.npy',gradient=19,first_hvp=774,true_hvp=1030),'sole historical state018 operator')
    need(card['history']==read(ROOT/D.CARD),'literal complete original consumed four-operator authority')

def bind(card,inventory,runtime,budget,store):
    parent,index,audit=D.bind(card['history'],budget.check,store.json)
    old=read(parent/'run/result.json')
    for n,h in old['source_sha256'].items():need(inventory[n]==h,'unchanged native and historical source '+n)
    for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(runtime[k]==old['runtime'][k],'same measured native runtime '+k)
    diag=card['diagnostic'];path=Path(diag['root']);need(path==D.OUT.parent and diag['outer']=='c3922ddd385d4c18246de92b8f76f89d9e5e0be380f105a54f0c62d3cd51652c','literal saved conditioning authority')
    U.inventory(path,diag['outer'],budget.check);U.inventory(path/'run',diag['worker'],budget.check)
    rt=read(path/'run/runtime.json');out=read(path/'run/outcome.json');binding=read(path/'run/input-binding.json')
    need(rt['source_commit']==diag['source']and out['passed']and out['native_calls']==0 and binding['producer']==card['history']['producer']and binding['independent']==card['history']['independent'],'completed exact saved diagnosis')
    for n,v in rt['files'].items():need(entry(path/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',diag['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'literal saved diagnostic source '+n)
    owner=read(path/'launcher-result.json');need(owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed diagnostic ownership')
    for k in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(owner[k])).exists(),'actual historical process absent')
    need(out['units'][3]['state']=='state-018.npy'and out['units'][3]['direction_action_qualified'],'fixed saved negative operator authority')
    events=[json.loads(s)for s in(parent/'run/events.jsonl').read_text().splitlines()]
    store.json('comparison-binding.json',dict(card=entry(ROOT/CARD),operator=card['operator'],diagnostic=diag,original_input_identity=old['input_identity'],baseline_forcing=audit['steps'][3]['forcing'],saved_span_forcing=out['units'][3]['span_predicted_forcing']))
    return parent,index,old,audit,events

def forecasts(orth_seconds,algebra_seconds,native_maxima,elapsed,archive_bytes):
    # Charge the maximum 256-column two-pass cost for every new action.
    native=sum(CAPS[k]*native_maxima[k]for k in('hvp','direct_gradient'))
    array_bytes=98666*8+128;raw=524*array_bytes+64*1024**2+archive_bytes
    forecast=elapsed+2*(256*orth_seconds+algebra_seconds+native)+30+raw/(50*1024**2)
    need(all(math.isfinite(v)and v>0 for v in [orth_seconds,algebra_seconds,*native_maxima.values()]),'finite positive measured full costs')
    return dict(passed=forecast<LIMITS['seconds']and raw<LIMITS['raw_bytes'],forecast_seconds=forecast,forecast_raw_bytes=raw,maximum_native_seconds=native_maxima,max_width_twopass_seconds=orth_seconds,gram_svd_assembly_seconds=algebra_seconds,profile_and_binding_seconds=elapsed,safety_factor=2,raw_array_count_ceiling=524,write_bytes_per_second=50*1024**2,target_arrays_unopened=True)

def profile(events,budget,store,archive_bytes):
    n,m=98666,256;rng=np.random.default_rng(52018)
    store.json('profile-input.json',dict(seed=52018,rows=n,columns=m,recipe='Fortran V=normal/sqrt(n), W=linspace(.5,2,n)*V, w=normal; three full-width CGS2 then Gram, lstsq(W,-V0), Vc and Wc.',target_arrays_unopened=True))
    V=np.asfortranarray(rng.standard_normal((n,m))/np.sqrt(n));W=np.asfortranarray(np.linspace(.5,2,n)[:,None]*V);w=rng.standard_normal(n);samples=[]
    for _ in range(3):
        budget.check();t=time.monotonic();q,c=orthogonalize(V,w);samples.append(time.monotonic()-t);need(np.isfinite(q).all(),'finite synthetic two-pass return')
    t=time.monotonic();gram=V.T@V;coef,_,rank,singular=np.linalg.lstsq(W,-V[:,0],rcond=SETTINGS['svd_rcond']);p=V@coef;hp=W@coef;algebra=time.monotonic()-t;budget.check()
    maxima={k:max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k)for k in('hvp','direct_gradient')}
    report=forecasts(max(samples),algebra,maxima,time.monotonic()-budget.start,archive_bytes);report.update(samples=samples,synthetic_rank=int(rank),synthetic_gram_error=float(np.linalg.norm(gram-np.eye(m),ord=np.inf)),synthetic_forcing=float(np.linalg.norm(V[:,0]+hp))/float(np.linalg.norm(V[:,0])),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['forecast_peak_rss_kib']=report['peak_rss_kib']+1024**2;report['passed']=bool(report['passed']and report['forecast_peak_rss_kib']<LIMITS['rss_kib']);store.json('profile.json',report);need(report['passed'],'complete measured equal-action admission before target arrays')
    del V,W,w,q,c,gram,coef,p,hp,singular;gc.collect();return report

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
        store.write('source.tar.gz',archive.getvalue());parent,index,old,audit,events=bind(card,inventory,runtime,budget,store);result['profile']=profile(events,budget,store,len(archive.getvalue()))
        op=card['operator'];folder=parent/'run';x=U.vector(folder/op['state']);oldg=U.vector(folder/index['direct_gradient'][19]['output']);first=U.vector(folder/index['hvp'][774]['direction']);oldfirst=U.vector(folder/index['hvp'][774]['output']);oldhp=U.vector(folder/index['hvp'][1030]['output'])
        baseline=float(np.linalg.norm(oldg+oldhp))/float(np.linalg.norm(oldg));U.close(baseline,audit['steps'][3]['forcing'],'same exact old physical forcing')
        arrays,model,_,_,identity=B.inputs()
        for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'unchanged original objective '+k)
        store.json('input-identity.json',dict(identity=identity,operator=op,baseline_forcing=baseline,parent=card['history']['producer'],direction_applied=False));store.array('historical-point',x);store.array('historical-gradient',oldg)
        problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=E.Native(problem,store,budget);g=native.grad(x);need(B.byte_equal(g,oldg),'byte-exact same original gradient')
        def action(v):
            h=native.hvp(x,v)
            if budget.returned['hvp']==1:need(B.byte_equal(h,oldfirst),'byte-exact literal original first Hessian action')
            return h
        p,pred,gram,report=solve(g,first,action,event=lambda k,v:store.event(dict(kind=k,**v)),tick=budget.check)
        for name,v in [('candidate-direction',p),('predicted-action',pred),('final-gram',gram)]:store.array(name,v)
        store.json('basis-report.json',report);actual=native.hvp(x,p);result['comparison']=qualify(g,pred,actual,report,baseline);store.json('comparison.json',result['comparison'])
        need(budget.attempted==budget.returned and budget.returned['hvp']==report['actions']+1 and budget.returned['direct_gradient']==1,'complete inclusive native accounting');budget.check();result.update(status='reorthogonalized_comparison_complete',comparison_qualified=result['comparison']['qualified'],basis=report,input_identity=identity)
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-budget.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
    print(json.dumps(result));return 0 if result['status']=='reorthogonalized_comparison_complete'else 2

def main():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);return run(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
if __name__=='__main__':raise SystemExit(main())
