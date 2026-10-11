"""One bounded stationary/eigenpair seed; the old optimizer state is preserved."""
import argparse,hashlib,inspect,io,json,math,os,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_projected as Q
from neural_shrimp.cifar_stationary import stationary,positive_mode,SETTINGS,minres
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
B,P,A=Q.B,Q.P,Q.A
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/attempt001')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/projected001')
PIN='89eb660c4f1ba722381db36eda561c8c826a642efa704b5c0c7ba9f7e9c4803e'
EXEC='d9c74b034913e7891def3919548ae276f9d92944'
CAPS=dict(action=0,j2vp=0,hvp=1063,direct_gradient=59)
EXTRA=tuple('src/neural_shrimp/'+n+'.py'for n in('__init__','controls','models','cifar_amplitude','cifar_stationary'))+('tools/explore_cifar_stationary.py','tools/launch_cifar_stationary.py','tests/test_cifar_stationary.py','tests/test_cifar_stationary_driver.py','docs/plans/cifar-stationary-seed.md','docs/reviews/cifar-projected001.json')
def install():Q.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+EXTRA))

class Budget(BaseBudget):
    def __init__(self,event):super().__init__(event,seconds=400.)
    def call(self,kind,function,*a,**kw):
        self.check();B.require(self.attempted[kind]<CAPS[kind],'stationary native cap '+kind)
        self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]))
        v=function(*a,**kw);self.returned[kind]+=1
        self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def measured_gate(events):
    maxima={k:max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k)for k in('hvp','direct_gradient')}
    forecast=2*sum(CAPS[k]*v for k,v in maxima.items())+20+3072/50
    B.require(all(math.isfinite(v)and v>0 for v in maxima.values())and forecast<400,'complete measured400s admission')
    return dict(forecast_seconds=forecast,maximum_operation_seconds=maxima,safety_factor=2,overhead_seconds=20,raw_write_allowance_seconds=3072/50)

def check_parent(row):
    B.require(row['source_commit']==EXEC and row['status']=='projected_pilot_complete'and row['comparison_qualified']and row['qualification']['passed']
        and not row['accepted_orbit']and not row['spectrum_requested']and not row['trials']and row['accepted_index']is None
        and row['linear']['dimension']==4 and row['linear']['action_linearity_pass']and not row['linear']['forcing_pass']
        and row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=12,direct_gradient=14),'complete fixed negative original state')

class Native:
    def __init__(self,problem,store,budget):self.problem,self.store,self.budget=problem,store,budget;self.states={}
    def state(self,x):
        x=np.asarray(x,dtype=np.float64);B.require(x.shape==(98666,)and np.isfinite(x).all(),'full original coordinate state')
        key=hashlib.sha256(x.tobytes()).hexdigest()
        if key not in self.states:
            name=f'state-{len(self.states):03d}';record=self.store.array(name,x);self.states[key]=dict(file=name+'.npy',identity=record)
        return self.states[key]
    def call(self,kind,x,v=None):
        self.budget.check();state=self.state(x);index=self.budget.attempted[kind]+1;name=f'native-{kind}-{index:04d}'
        row=dict(kind='native_input',operation=kind,index=index,state=state)
        if v is not None:
            v=np.asarray(v,dtype=np.float64);B.require(v.shape==(98666,)and np.isfinite(v).all(),'full finite native direction')
            row['direction']=dict(file=name+'-vector.npy',identity=self.store.array(name+'-vector',v))
        self.store.event(row)
        tx=B.cifar.as_tensor(x)
        result=(self.problem.grad(tx)if kind=='direct_gradient'else self.problem.hvp(tx,B.cifar.as_tensor(v))).numpy()
        ret=self.store.array(name+'-return',result);B.require(result.shape==(98666,)and np.isfinite(result).all(),'full finite native output')
        self.store.event(dict(kind='native_output',operation=kind,index=index,result=dict(file=name+'-return.npy',identity=ret)))
        self.budget.check();return result
    def grad(self,x):return self.call('direct_gradient',x)
    def hvp(self,x,v):return self.call('hvp',x,v)

def run(out,review):
    B.require(out==BASE/'run','single fixed output');inventory=B.sources();runtime=B.environment()
    store=Store(out,limit=3*1024**3,reserve=4*1024**2);started=time.monotonic();budget=Budget(store.event)
    result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,
        settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,mode=None,comparison_qualified=False,candidate_rate_scope='2/tau is only a proposal after qualified stationarity and eigenpair; no optimizer run at that rate')
    try:
        B.require(review==result['source_commit'],'exact reviewed execution source');store.json('runtime-start.json',result);store.write('minres-source.py',inspect.getsource(minres).encode())
        for root,pin in[(PARENT,PIN),(Q.CORR,Q.CORR_PIN),(Q.GEOM,Q.GEOM_PIN),(P.DIAGNOSTIC,P.DIAG_PIN)]:P.verify_diagnostic(root,pin,budget.check)
        old=json.loads((PARENT/'run/result.json').read_text());check_parent(old)
        for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):B.require(runtime[k]==old['runtime'][k],'same measured runtime '+k)
        for n,h in old['source_sha256'].items():B.require(inventory[n]==h==hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest(),'same source-bound original kernel '+n)
        with tarfile.open(PARENT/'run/source.tar.gz','r:gz')as tar:
            members=tar.getmembers();B.require({m.name for m in members}==set(old['source_sha256'])and all(m.isfile()and m.size<2*1024**2 for m in members),'complete literal old archive')
            for m in members:B.require(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==inventory[m.name],'old archived bytes')
        result['profile']=measured_gate([json.loads(l)for l in(PARENT/'run/events.jsonl').read_text().splitlines()]);store.json('profile.json',result['profile'])
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz')as tar:
            for n in inventory:
                data=(ROOT/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(data);i.mtime=0;tar.addfile(i,io.BytesIO(data))
        store.write('source.tar.gz',archive.getvalue())
        def read(n):
            v=np.load(PARENT/'run'/n,allow_pickle=False);B.require(v.shape==(98666,)and v.dtype==np.float64 and np.isfinite(v).all(),'exact complete original phases');return v
        x,y=read('initial-theta.npy'),read('initial-map1.npy');m=(x+y)/2;delta=(x-y)/2
        arrays,model,_,_,identity=B.inputs()
        for k in('data_identity','eta','lam','model'):B.require(identity[k]==old['input_identity'][k],'same original physical objective '+k)
        B.require(identity['eta']==.00827753 and identity['lam']==.01,'old reference rate/ridge')
        result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,original_rate_scope='reference only; new seed eta=2/tau is separately reported')
        store.json('input-identity.json',result['input_identity']);store.array('original-phase0',x);store.array('original-phase1',y);store.array('initial-midpoint',m);store.array('initial-half-displacement',delta)
        problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=Native(problem,store,budget)
        B.require(np.isfinite(np.linalg.norm(delta))and np.linalg.norm(delta)>0,'nonzero original full phase displacement')
        g=native.grad(m);q=delta/np.linalg.norm(delta);v=np.random.default_rng(28167).standard_normal(len(m));v/=np.linalg.norm(v)
        hq,hv=native.hvp(m,q),native.hvp(m,v)
        symmetry=abs(float(np.dot(q,hv)-np.dot(v,hq)))/max(float(np.linalg.norm(hq)),float(np.linalg.norm(hv)),1e-12)
        B.require(symmetry<=1e-10,'actual midpoint Hessian symmetric pairing');errors=[]
        for i,h in enumerate(max(1.,float(np.linalg.norm(m)))*np.array([1e-3,1e-4,1e-5])):
            gm,gp=native.grad(m-h*q),native.grad(m+h*q);fd=(gp-gm)/(2*h);store.array(f'qualification-fd-{i}',fd);errors.append(dict(h=float(h),relative_error=P.fd_error(hq,fd)))
        B.require(errors[-1]['relative_error']<=5e-6,'actual midpoint Hessian finite difference')
        result['qualification']=dict(passed=True,symmetry_error=symmetry,finite_differences=errors);store.json('qualification.json',result['qualification'])
        result['current_profile']=measured_gate([json.loads(l)for l in(out/'events.jsonl').read_text().splitlines()]);store.json('current-profile.json',result['current_profile'])
        point,grad,station=stationary(m,g,native.grad,native.hvp,event=store.event);store.array('stationary-point',point);store.array('stationary-gradient',grad);result['stationary']=station;store.json('stationary.json',station)
        if station['passed']:
            q,hq,mode=positive_mode(point,delta,native.hvp,event=store.event);store.array('positive-mode',q);store.array('positive-mode-action',hq);result['mode']=mode;store.json('mode.json',mode)
        budget.check();result.update(status='stationary_seed_complete',comparison_qualified=True,seed_qualified=bool(station['passed']and result['mode']and result['mode']['passed']))
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1)
    print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
