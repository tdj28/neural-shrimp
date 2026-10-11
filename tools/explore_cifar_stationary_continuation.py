"""Four bounded current-state stationary units; global original trust ball retained."""
import argparse,hashlib,inspect,io,json,math,os,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_projected as Q
import explore_cifar_stationary_saved_merit as M
from neural_shrimp.cifar_stationary_continuation import stationary_from_seed,SETTINGS as CONTINUATION_SETTINGS
from neural_shrimp.cifar_stationary import positive_mode,SETTINGS,minres
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
B,P,A=Q.B,Q.P,Q.A
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/continuation001')
PARENT=M.BASE
PIN='3d77085cf744c2bb60e1e5d905765e20e6bf2e1cb10abd28415fa20139593300'
EXEC='86ab814cbcfbc8465b8a313e711a69314c7406e6'
AUDIT=Path('/data2/neural-shrimp-data/audits/cifar-stationary-saved-merit/attempt001')
AUDIT_PIN='364f7147e20590d0299cf5174d3631d02c3f64434d0bfa7c0b4fd63a3f013297'
AUDIT_SOURCE='5c41b3cf7040fae5d4bbbb47b1a15c65b6a2af9a'

CAPS=dict(action=0,j2vp=0,hvp=1063,direct_gradient=59)
EXTRA=('src/neural_shrimp/cifar_stationary_continuation.py','tools/explore_cifar_stationary_continuation.py','tools/launch_cifar_stationary_continuation.py','tests/test_cifar_stationary_continuation.py','tests/test_cifar_stationary_continuation_driver.py','docs/plans/cifar-stationary-continuation.md','docs/reviews/cifar-stationary-merit-independent.json')
def install():M.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+EXTRA))

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
    B.require(row['source_commit']==EXEC and row['status']=='stationary_saved_merit_complete'and row['comparison_qualified']
        and not row['accepted_orbit']and not row['rigorous_certificate']and not row['stationarity_qualified']
        and row['accepted_index']==3 and len(row['trials'])==4 and [v['accepted']for v in row['trials']]==[False,False,False,True]
        and row['direction_gate']['passed']and not row['direction_gate']['old_forcing_pass']
        and row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=1,direct_gradient=5),'complete audited saved merit step')


def audit_parent(tick):
    P.verify_diagnostic(AUDIT,AUDIT_PIN,tick)
    rt=json.loads((AUDIT/'run/runtime.json').read_text());out=json.loads((AUDIT/'run/outcome.json').read_text());bind=json.loads((AUDIT/'run/input-binding.json').read_text())
    B.require(rt['source_commit']==AUDIT_SOURCE and out['passed']and out['native_calls']==0 and out['analysis']['accepted_index']==3 and not out['analysis']['stationarity_qualified'],'exact independent saved acceptance audit')
    B.require(bind['parent']==str(PARENT)and bind['pins']['parent']==PIN and bind['producer_source']==EXEC,'audit of literal selected parent')
    for n,v in rt['files'].items():
        b=(AUDIT/'run/source'/n).read_bytes();B.require(hashlib.sha256(b).hexdigest()==v['sha256']==hashlib.sha256(subprocess.check_output(['git','show',AUDIT_SOURCE+':'+n],cwd=ROOT)).hexdigest(),'independent historical audit source')
    return dict(root=str(AUDIT),outer=AUDIT_PIN,source=AUDIT_SOURCE)

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
        settings=CONTINUATION_SETTINGS,mode_settings={k:SETTINGS[k]for k in ['power_steps','eigen_residual']},accepted_orbit=False,rigorous_certificate=False,mode=None,comparison_qualified=False,candidate_rate_scope='2/tau is only a proposal after qualified stationarity and eigenpair; no optimizer run at that rate')
    try:
        B.require(review==result['source_commit'],'exact reviewed execution source');store.json('runtime-start.json',result);store.write('minres-source.py',inspect.getsource(minres).encode())
        for root,pin in[(PARENT,PIN),(M.PARENT,M.PIN)]:P.verify_diagnostic(root,pin,budget.check)
        result['independent_audit']=audit_parent(budget.check)
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
        m,oldg=read('final-point.npy'),read('final-gradient.npy')
        B.require(B.byte_equal(m,read('trial-03-point.npy'))and B.byte_equal(oldg,read('trial-03-gradient.npy')),'exact audited accepted state and gradient')
        origin=read('initial-point.npy')
        original=np.load(M.PARENT/'run/initial-midpoint.npy',allow_pickle=False);delta=np.load(M.PARENT/'run/initial-half-displacement.npy',allow_pickle=False)
        B.require(B.byte_equal(origin,original)and delta.shape==(98666,)and delta.dtype==np.float64 and np.isfinite(delta).all(),'original unchanged global trust origin and mode seed')
        B.require(float(np.linalg.norm(m-origin))/max(1.,float(np.linalg.norm(origin)))<=.05,'accepted seed remains in original5% ball')
        arrays,model,_,_,identity=B.inputs()
        for k in('data_identity','eta','lam','model'):B.require(identity[k]==old['input_identity'][k],'same original physical objective '+k)
        B.require(identity['eta']==.00827753 and identity['lam']==.01,'old reference rate/ridge')
        result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,original_rate_scope='reference only; new seed eta=2/tau is separately reported')
        store.json('input-identity.json',result['input_identity']);store.array('global-original-midpoint',origin);store.array('initial-point',m);store.array('initial-half-displacement',delta)
        problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=Native(problem,store,budget)
        B.require(np.isfinite(np.linalg.norm(delta))and np.linalg.norm(delta)>0,'nonzero original full phase displacement')
        g=native.grad(m);B.require(B.byte_equal(g,oldg),'byte-exact current gradient replay');q=delta/np.linalg.norm(delta);v=np.random.default_rng(28167).standard_normal(len(m));v/=np.linalg.norm(v)
        hq,hv=native.hvp(m,q),native.hvp(m,v)
        symmetry=abs(float(np.dot(q,hv)-np.dot(v,hq)))/max(float(np.linalg.norm(hq)),float(np.linalg.norm(hv)),1e-12)
        B.require(symmetry<=1e-10,'actual midpoint Hessian symmetric pairing');errors=[]
        for i,h in enumerate(max(1.,float(np.linalg.norm(m)))*np.array([1e-3,1e-4,1e-5])):
            gm,gp=native.grad(m-h*q),native.grad(m+h*q);fd=(gp-gm)/(2*h);store.array(f'qualification-fd-{i}',fd);errors.append(dict(h=float(h),relative_error=P.fd_error(hq,fd)))
        B.require(errors[-1]['relative_error']<=5e-6,'actual midpoint Hessian finite difference')
        result['qualification']=dict(passed=True,symmetry_error=symmetry,finite_differences=errors);store.json('qualification.json',result['qualification'])
        result['current_profile']=measured_gate([json.loads(l)for l in(out/'events.jsonl').read_text().splitlines()]);store.json('current-profile.json',result['current_profile'])
        point,grad,station=stationary_from_seed(m,g,origin,native.grad,native.hvp,event=store.event);store.array('stationary-point',point);store.array('stationary-gradient',grad);result['stationary']=station;store.json('stationary.json',station)
        if station['passed']:
            q,hq,mode=positive_mode(point,delta,native.hvp,event=store.event);store.array('positive-mode',q);store.array('positive-mode-action',hq);result['mode']=mode;store.json('mode.json',mode)
        budget.check();result.update(status='stationary_continuation_complete',comparison_qualified=True,seed_qualified=bool(station['passed']and result['mode']and result['mode']['passed']))
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1)
    print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
