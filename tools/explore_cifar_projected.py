"""One fixed <=4D physical correction at the unchanged saved CIFAR state."""
import argparse,hashlib,io,json,math,os,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_phase_scaled as P
from neural_shrimp.cifar_projected import basis,proposal,qualify
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_merit_lsmr import trial
B,A=P.B,P.A
CORR=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/corrector001')
CORR_PIN='966225d5358a9f81479de0519db856c635398d43b8615aee8705a87e87b6a2ea'
CORR_EXEC='9bc60ebaea7334a72903d71761e085f99f7bdd1a'
GEOM=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/geometry001')
GEOM_PIN='f7dcd7afffe8a01108695b1308d1a44eeb2d4465497b76499d2ea2e152f64ae7'
CAPS=dict(action=0,j2vp=0,hvp=12,direct_gradient=40)
EXTRA=('src/neural_shrimp/cifar_projected.py','tools/explore_cifar_projected.py',
       'tools/launch_cifar_projected.py','tests/test_cifar_projected.py',
       'tests/test_cifar_projected_driver.py','docs/plans/cifar-projected-corrector.md')


def install():P.install();B.SOURCES=tuple(B.SOURCES)+EXTRA


class Budget(BaseBudget):
    def __init__(self,event):super().__init__(event,seconds=120.)
    def call(self,kind,function,*args,**kw):
        self.check();B.require(self.attempted[kind]<CAPS[kind],'fixed operation cap '+kind)
        self.attempted[kind]+=1;t=time.monotonic()
        self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]))
        v=function(*args,**kw);self.returned[kind]+=1
        self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t))
        return v


def measured_gate(events):
    costs={k:max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k)for k in('hvp','direct_gradient')}
    forecast=2*sum(CAPS[k]*v for k,v in costs.items())+20
    B.require(all(math.isfinite(v)and v>0 for v in costs.values())and forecast<120,'measured120s fixed-unit admission')
    return dict(forecast_seconds=forecast,maximum_operation_seconds=costs,safety_factor=2,overhead_seconds=20)


def check_parent(row):
    B.require(row['source_commit']==CORR_EXEC and row['status']=='phase_scaled_pilot_complete'
              and row['qualification']['passed'] and row['accepted_index']is None
              and not row['trials'] and not row['linear']['forcing_pass']
              and row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=818,direct_gradient=14),
              'exact completed negative fixed parent')


def run(out,review):
    inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True)
    store=Store(out,limit=160*1024**2);started=time.monotonic();budget=Budget(store.event)
    result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=inventory,review_commit=review,runtime=B.environment(),accepted_orbit=False,
        spectrum_requested=False,comparison_qualified=False,trials=[],accepted_index=None,
        method='Fixed ordered p,q-p,phase displacement,residual span; fresh physical actions and original forcing/Armijo gates')
    try:
        store.json('runtime-start.json',result)
        P.verify_diagnostic(CORR,CORR_PIN,budget.check);P.verify_diagnostic(GEOM,GEOM_PIN,budget.check)
        old=json.loads((CORR/'run/result.json').read_text());check_parent(old)
        geometry=json.loads((GEOM/'run/outcome.json').read_text())
        B.require(geometry['passed']and geometry['native_calls']==0 and geometry['parent']['outer']==CORR_PIN
                  and not geometry['analysis']['line']['forcing_point5_reachable'],'completed fixed geometry diagnosis')
        for n,h in old['source_sha256'].items():
            B.require(inventory[n]==h==hashlib.sha256(subprocess.check_output(['git','show',CORR_EXEC+':'+n],cwd=ROOT)).hexdigest(),'exact prior executed source '+n)
        with tarfile.open(CORR/'run/source.tar.gz','r:gz')as tar:
            members=tar.getmembers();B.require({m.name for m in members}==set(old['source_sha256'])and all(m.isfile()and m.size<2*1024**2 for m in members),'complete historical source archive')
            for m in members:B.require(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==inventory[m.name],'historical source bytes')
        result['profile']=measured_gate([json.loads(v)for v in(CORR/'run/events.jsonl').read_text().splitlines()])
        store.json('profile.json',result['profile'])
        archive=io.BytesIO()
        with tarfile.open(fileobj=archive,mode='w:gz')as tar:
            for n in inventory:
                data=(ROOT/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
        store.write('source.tar.gz',archive.getvalue())
        def read(name,shape):
            v=np.load(CORR/'run'/f'{name}.npy',allow_pickle=False)
            B.require(v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all(),'complete same-state binary64 vector')
            return v
        n=98666;theta,one,two,oldp,oldap=[read(k,(n,))for k in('initial-theta','initial-map1','initial-map2','direction','actual-action')]
        z=read('shooting-direction',(2*n,));B.require(np.array_equal(z[:n],oldp),'literal saved physical p')
        arrays,model,_,_,identity=B.inputs()
        for key in ('data_identity','eta','lam','model'):B.require(identity[key]==old['input_identity'][key],'same exact physical map '+key)
        B.require(identity['eta']==.00827753 and identity['lam']==.01,'fixed development rate/ridge')
        result['input_identity']=dict(identity,parent_root=str(CORR),parent_manifest=CORR_PIN,parent_execution=CORR_EXEC,
            parent_state_sha256=A.L.entry(CORR/'run/initial-theta.npy')['sha256'])
        store.json('input-identity.json',result['input_identity']);store.array('initial-theta',theta)
        problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;eta=model['eta']
        def map_one(name,x):
            store.array(name+'-input',x);y=problem.update(B.cifar.as_tensor(x),eta).numpy();store.array(name+'-return',y);return y
        first=map_one('replay-phase1',theta);B.require(B.byte_equal(first,one),'native same-state phase1 replay')
        second=map_one('replay-phase2',first);B.require(B.byte_equal(second,two),'native same-state phase2 replay')
        store.array('initial-map1',first);store.array('initial-map2',second);r=second-theta;store.array('initial-residual',r)
        def action(name,v):
            store.array(name+'-input',v)
            hx=problem.hvp(B.cifar.as_tensor(theta),B.cifar.as_tensor(v)).numpy();store.array(name+'-Hx',hx)
            jx=v-eta*hx;store.array(name+'-Jx',jx)
            hy=problem.hvp(B.cifar.as_tensor(first),B.cifar.as_tensor(jx)).numpy();store.array(name+'-Hy',hy)
            value=jx-eta*hy-v;store.array(name+'-return',value);return value
        replay=action('saved-p-replay',oldp)
        replay_error=P.fd_error(oldap,replay);B.require(replay_error<=5e-11,'actual saved p action re-evaluation')
        unit=oldp/np.linalg.norm(oldp);actual=replay/np.linalg.norm(oldp);errors=[]
        for k,h in enumerate([max(1.,np.linalg.norm(theta))*s for s in(1e-3,1e-4,1e-5)]):
            values=[]
            for sign in(-1,1):
                x=theta+sign*h*unit;y=map_one(f'fd-{k}-{sign}-phase1',x);w=map_one(f'fd-{k}-{sign}-phase2',y);values.append(w-x)
            fd=(values[1]-values[0])/(2*h);store.array(f'fd-{k}-difference',fd);errors.append(dict(h=float(h),relative_error=P.fd_error(actual,fd)))
        B.require(errors[-1]['relative_error']<=5e-6,'fresh actual full-period finite difference')
        result['qualification']=dict(passed=True,saved_p_action_relative_error=replay_error,finite_differences=errors,fd_threshold=5e-6)
        store.json('qualification.json',result['qualification'])
        result['current_profile']=measured_gate([json.loads(v)for v in(out/'events.jsonl').read_text().splitlines()]);store.json('current-profile.json',result['current_profile'])
        candidates=[oldp,z[n:]-oldp,first-theta,r]
        for i,v in enumerate(candidates):store.array(f'candidate-{i}',v)
        U,records=basis(candidates);store.array('basis',U);store.json('basis-records.json',records)
        actions=np.column_stack([action(f'column-{i}',U[:,i])for i in range(U.shape[1])]);store.array('actions',actions)
        p,pred,linear=proposal(U,actions,r);store.array('direction',p);store.array('assembled-action',pred)
        ap=action('fresh-proposal',p);store.array('actual-action',ap);linear.update(qualify(r,pred,ap));result['linear']=linear;store.json('linear.json',linear)
        B.require(linear['action_linearity_pass'],'fresh physical action equals projected action')
        result.update(stop='linear_forcing_or_descent_failed',last_accepted_state='initial-theta.npy')
        if linear['forcing_pass']and linear['usable_descent']:
            result['stop']='backtracking_exhausted'
            for j in range(13):
                budget.check();alpha=2.**(-j);x=theta+alpha*p;y=map_one(f'trial-{j}-phase1',x);w=map_one(f'trial-{j}-phase2',y)
                for name,v in [('theta',x),('map1',y),('map2',w)]:store.array(f'trial-{j}-'+name,v)
                metrics=trial(x,y,w,r,ap,alpha);metrics['index']=j;result['trials'].append(metrics)
                if metrics['accepted_local_step']:
                    theta,first,second=x,y,w;result.update(accepted_index=j,last_accepted_state=f'trial-{j}-theta.npy',stop='one_step_accepted');break
        budget.check();result.update(status='projected_pilot_complete',comparison_qualified=True,
            final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),
            final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))))
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args()
    install();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);signal.signal(signal.SIGTERM,request_stop)
    B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
