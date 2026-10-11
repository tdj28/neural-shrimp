"""Consumed-data numerical flip and primitive36 exploration (not a proof)."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.numerical_flip import correct,qualify,refine_flip,crossing,doubled
RAW=Path('/data2/neural-shrimp-data')
PARENT=RAW/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
CERTS=RAW/'raw/EXP-028/attempt-002/stage3/certificates.json'
RAYS=RAW/'explore/a2-survey/rays.json'
PINS={str(PARENT):'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',str(CERTS):'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8',str(RAYS):'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4'}
CARD='docs/plans/s2-period18-flip-pilot.md'
RESUME=RAW/'explore/s2-period18-flip/attempt-001-target'
RESUME_MANIFEST='7bfeb8fcacdef4a12dc4c0b3fda500361a0a4b66dbb41812c1d9c4d2d0d6ad34'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/numerical_flip.py','tools/explore_s2_period18_flip.py','tests/test_numerical_flip.py',CARD]
CAP=32*1024**2
ARTIFACT_ROOT=None

def require(ok,why):
    if not ok:raise ValueError(why)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'sha256':digest(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_bytes())
def exact(x):return str(Fraction(float(x)))
def clean(x):
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':repr(x)}
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def total_bytes():
    return 0 if ARTIFACT_ROOT is None else sum(p.stat().st_size for p in ARTIFACT_ROOT.rglob('*') if p.is_file())
def save(p,obj,final=False):
    payload=(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    limit=(CAP+1024**2 if p.name in ('launcher-result.json','manifest.json') and (p.parent/'run').exists() else CAP) if final else 28*1024**2
    if total_bytes()+len(payload)>limit:
        if not final:
            save(ARTIFACT_ROOT/'oversized-artifact.json',{'path':str(p),'attempted_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
                'written':False,'reason':'aggregate ordinary output cap; full numerical evaluations remain in journal'},final=True)
        raise OSError('aggregate artifact limit')
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def inventory(out):return {str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
def now():return datetime.now(timezone.utc).isoformat()
def source_binding():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed source')
    for p in SOURCES:require((ROOT/p).read_bytes()==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed source '+p)
    return {p:digest(ROOT/p) for p in SOURCES}


def prepare():
    for p,h in PINS.items():require(digest(Path(p))==h,'pinned input '+p)
    consumed=read(PARENT);row=read(CERTS)[111];window=read(RAYS)['S2-28011-zoom:587'];ray=window['rays'][13]
    require(row['period']==18 and row['component']==587 and row['map_id']=='S2-28011-zoom' and ray['type']=='flip','fixed target')
    ell=[float(t) for t in window['center']];direction=[float(a)*float(b) for a,b in zip(window['scale'],ray['direction'])]
    require(all(t!=0 for t in direction),'both rates vary')
    family={'schema':'neural-shrimp-exact-log-ray-v1','ell':[exact(t) for t in ell],'direction':[exact(t) for t in direction],
        'ell_float':ell,'direction_float':direction,'historical_scale':window['scale'],'historical_unit_direction':ray['direction'],'formula':'eta_i(s)=exp(log(10)*(ell_i+direction_i*s)); ell,direction exact binary64 dyadics',
        'numerical_convention':'scalar10.0**float(ell_i+direction_i*s); proposal for the exact-real exponential family',
        'center_certificate_relation':'Exact exponential rates at s=0 need not equal the earlier binary64 dyadic center rates; numerical carrying is not a validated connecting tube.'}
    return {'schema':'neural-shrimp-period18-flip-input-v1','consumed_map':{k:consumed[k] for k in ('schema','network','x','y')},
        'family':family,'input_pins':PINS,'row_index':111,'ray_index':13,'period':18,'ray':ray,
        'historical_orbit':row['certificate']['orbit'],'historical_rates':[row['lr_hidden'],row['lr_output']]}


class Experiment:
    def __init__(self,inputs,out):
        self.inputs=inputs;self.out=out;self.start=time.monotonic();self.points=0;self.returns=0;self.events=0;self.bytes=0;self.stopped=False
        self.log=(out/'evaluations.jsonl').open('x');self.progress=(out/'events.jsonl').open('x')
        model=inputs['consumed_map'];spec=model['network'].copy()
        for k in ('in_scale','out_scale','loss_scale','ridge'):spec[k]=float(Fraction(spec[k]))
        self.net=TwoLayerNet(**spec);self.x=np.array([[float(Fraction(t)) for t in row] for row in model['x']]);self.y=np.array([float(Fraction(t)) for t in model['y']])
        self.ell=inputs['family']['ell_float'];self.direction=inputs['family']['direction_float']
    def check(self):
        if self.stopped or time.monotonic()-self.start>100:raise TimeoutError('100s internal deadline')
        if self.points>=40000:raise TimeoutError('40000 joint map/Jacobian/parameter products')
        if self.bytes>28*1024**2:raise OSError('ordinary32MiB output reserve')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS2GiB')
    def append(self,handle,obj):
        text=json.dumps(clean(obj),sort_keys=True,allow_nan=False)+'\n'
        if total_bytes()+len(text.encode())>28*1024**2:
            save(self.out/'overflow-return.json',{'journal':handle.name,'returned_record':obj,'reason':'ordinary aggregate cap'},final=True)
            raise OSError('aggregate journal/artifact cap')
        handle.write(text);handle.flush();os.fsync(handle.fileno());self.bytes+=len(text.encode())
    def emit(self,kind,obj):self.append(self.progress,{'event':self.events,'kind':kind,'record':obj});self.events+=1
    def evaluate(self,theta,s,period):
        self.check();theta=np.asarray(theta,dtype=float);z=theta.copy();n=len(z);m=np.eye(n);b=np.zeros(n);orbit=[z.tolist()];checkpoints={};self.returns+=1
        rates=np.array([10.0**float(a+float(s)*d) for a,d in zip(self.ell,self.direction)])
        rv=self.net.rate_vector([rates[0]],[rates[1]])[0];dv=self.net.rate_vector([rates[0]*np.log(10.)*self.direction[0]],[rates[1]*np.log(10.)*self.direction[1]])[0]
        record={'id':self.returns,'theta':theta.tolist(),'s':float(s),'period':period,'rates':rates.tolist(),'rates_exact':[exact(t) for t in rates]}
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                for k in range(period):
                    self.check();self.points+=1
                    grad=self.net.gradient(z[None,:],self.x,self.y)[0];h=self.net.hessian(z,self.x,self.y)
                    j=np.eye(n)-rv[:,None]*h
                    b=j@b-dv*grad;m=j@m;z=z-rv*grad
                    orbit.append(z.tolist())
                    if k+1 in (period//2,period):checkpoints[str(k+1)]={'state':z.tolist(),'monodromy':m.tolist(),'parameter_derivative':b.tolist()}
                    require(np.isfinite(z).all() and np.isfinite(m).all() and np.isfinite(b).all(),'finite joint return')
            record.update(orbit=orbit,monodromy=m.tolist(),parameter_derivative=b.tolist(),residual=(z-theta).tolist(),checkpoints=checkpoints,completed=True)
        except BaseException as exc:
            record.update(orbit=orbit,monodromy=m.tolist(),parameter_derivative=b.tolist(),checkpoints=checkpoints,completed=False,error=type(exc).__name__+': '+str(exc));self.append(self.log,record);raise
        self.append(self.log,record);return record
    def close(self):self.log.close();self.progress.close()


def run(out,profile=False):
    global ARTIFACT_ROOT
    ARTIFACT_ROOT=out
    bound=source_binding();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();exp=None
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'started_utc':now(),
        'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'internal_seconds':100,'joint_map_jacobian_cap':40000,'raw_bytes':CAP,'monitored_rss_bytes':2*1024**3,'evidence_rung':0,'cpu_only':True,'profile':profile}
    save(out/'runtime-start.json',runtime)
    for p in SOURCES:
        q=out/'source'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
    result={'qualified':False,'flip_qualified':False,'crossing_qualified':False,'doubled_qualified_count':0,'evidence_rung':0,'status':'started'}
    try:
        if profile:
            inputs={'consumed_map':{'network':{'d':2,'width':3,'activation':'tanh','bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},
                'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)]},
                'family':{'ell_float':[-2.,-2.],'direction_float':[.05,.03]},'synthetic':True}
        else:inputs=prepare()
        save(out/'inputs.json',inputs);exp=Experiment(inputs,out)
        def stop(*_):exp.stopped=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,100)
        resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
        if profile:
            times=[]
            for _ in range(5):
                then=time.monotonic();exp.evaluate(np.arange(13)/128.,0.,18);times.append(time.monotonic()-then)
            result.update(qualified=True,status='synthetic_profile_complete',scientific_claim=False,seconds_per18_evaluation=times,
                          projected40000_joint_step_seconds=max(times)*40000/18)
            return 0
        require(digest(RESUME/'manifest.json')==RESUME_MANIFEST,'retained prefix manifest')
        for name,e in read(RESUME/'manifest.json')['files'].items():require(entry(RESUME/name)==e,'retained prefix artifact '+name)
        require(read(RESUME/'run/inputs.json')==inputs,'unchanged exact input/family')
        nodes=read(RESUME/'run/path.json');require(len(nodes)==22 and all(n['qualified'] for n in nodes),'qualified retained prefix')
        previous=nodes[-1];x=np.array(previous['state']);targets=np.linspace(0.,.999*inputs['ray']['s_boundary'],33)
        require(previous['s']==float(targets[len(nodes)-1]),'exact common-parameter resume')
        save(out/'resume.json',{'path':str(RESUME),'manifest_sha256':RESUME_MANIFEST,'prefix_nodes':len(nodes),'common_s':previous['s']})
        save(out/'path.json',nodes)
        for s in targets[len(nodes):]:
            node=correct(exp.evaluate,x,float(s),18);node['s']=float(s);nodes.append(node);save(out/'path.json',nodes);exp.emit('path_node',node)
            require(node['qualified'],'forward path correction/qualification')
            new=np.array(node['state'])
            if previous is not None:
                reverse=correct(exp.evaluate,new,previous['s'],18);scale=max(1.,float(np.max(abs(x))))
                error=float(np.max(abs(np.asarray(reverse['state'])-x))/scale)
                pair_error=float(np.max(abs(np.asarray(node['data']['orbit'][:-1])-np.asarray(previous['data']['orbit'][:-1])))/scale)
                edge={'from_s':previous['s'],'to_s':float(s),'reverse':reverse,'reverse_error':error,'full_native_phase_step':pair_error,
                      'qualified':bool(reverse['qualified'] and error<1e-7 and pair_error<.1)}
                exp.emit('path_edge',edge);require(edge['qualified'],'native phase/reverse link unresolved')
            # Adaptation: continue qualified primitive orbits through instability; retain every stability label.
            x=new;previous=node
        transitions=[{'from_s':a['s'],'to_s':b['s'],'from_stability':a['qualification']['stability'],'to_stability':b['qualification']['stability'],'from_multipliers':a['qualification']['multipliers'],'to_multipliers':b['qualification']['multipliers']} for a,b in zip(nodes,nodes[1:]) if a['qualification']['stability']!=b['qualification']['stability']]
        save(out/'stability-transitions.json',transitions)
        point=refine_flip(exp.evaluate,x,float(nodes[-1]['s']),18,exp.emit);save(out/'flip.json',point)
        result['flip_qualified']=point['qualified'];require(point['qualified'],'flip point unqualified')
        seed={'schema':'neural-shrimp-flip-proof-seed-v1','period':18,'consumed_map':inputs['consumed_map'],'family':inputs['family'],
            'theta':[exact(t) for t in point['state']],'parameter':exact(point['s']),'right_vector':[exact(t) for t in point['q']],
            'numeric':{'theta':point['state'],'parameter':point['s'],'right_vector':point['q']},'source_commit':runtime['source_commit'],'inputs_sha256':digest(out/'inputs.json')}
        save(out/'flip-proof-seed.json',seed)
        crossed=crossing(exp.evaluate,point,18,exp.emit);save(out/'crossing.json',crossed);result['crossing_qualified']=crossed['qualified']
        # Independent native starting phase: transport q/p with the actual first map derivative.
        try:
            first=exp.evaluate(np.asarray(point['state']),point['s'],1);j0=np.asarray(first['monodromy'])
            q0=np.asarray(point['q']);p0=np.asarray(point['left']);k=float(np.linalg.norm(j0@q0))
            q1=j0@q0/k;p1=k*np.linalg.solve(j0.T,p0);theta1=first['orbit'][-1]
            native_data=exp.evaluate(np.asarray(theta1),point['s'],18)
            native_point={'state':theta1,'s':point['s'],'q':q1.tolist(),'left':p1.tolist(),'data':native_data}
            native_crossing=crossing(exp.evaluate,native_point,18,exp.emit)
            save(out/'adjacent-phase.json',{'transport_norm':k,'point':native_point,'qualification':qualify(native_data),
                'crossing':native_crossing,'scope':'Numerical native-phase and crossing check; cubic coefficient not computed in this pilot.'})
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
            save(out/'adjacent-phase.json',{'qualified':False,'error':type(exc).__name__+': '+str(exc),'scope':'Optional native-phase diagnostic unresolved; primary evidence unchanged.'})
        scale=max(1.,max(abs(np.array(point['state']))));branches=[];last=None
        for exponent in (12,11,10):
            amp=(2.**-exponent)*scale;seed=None
            if last is not None and last['qualified']:
                ratio=amp/last['amplitude_requested'];seed=np.r_[np.array(point['state'])+ratio*(np.array(last['state'])-np.array(point['state'])),point['s']+ratio**2*last['delta_s']]
            try:
                branch=doubled(exp.evaluate,point,18,amp,seed,exp.emit)
                opposite=exp.evaluate(np.asarray(branch['data']['orbit'][18]),branch['s'],36)
                branch['opposite_phase']=opposite;branch['opposite_qualification']=qualify(opposite)
                branch['gates']['opposite_phase_qualified']=branch['opposite_qualification']['qualified'];branch['qualified']=all(branch['gates'].values())
                last=branch
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                branch={'qualified':False,'amplitude_requested':amp,'error':type(exc).__name__+': '+str(exc)}
            branches.append(branch);save(out/'doubled.json',branches)
        result['doubled_qualified_count']=sum(b['qualified'] for b in branches)
        result['doubled_attracting_count']=sum(b['qualified'] and b['qualification']['stability']=='attracting' for b in branches)
        result.update(qualified=bool(point['qualified'] and crossed['qualified'] and result['doubled_qualified_count']>0),
                      status='numerical_pilot_complete',evidence_rung=1 if point['qualified'] and crossed['qualified'] else 0,
                      state=point['state'],s=point['s'],q=point['q'],full_phase_path_nodes=len(nodes),continuous_branch_proof=False)
    except BaseException as exc:
        result.update(status='incomplete_or_failed',error=type(exc).__name__+': '+str(exc));save(out/'failure.json',{'error':result['error']},final=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if exp is not None:exp.close();result.update(joint_map_jacobian_evaluations=exp.points,return_evaluations=exp.returns,events=exp.events)
        result.update(elapsed_seconds=time.monotonic()-start,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(out/'outcome.json',result,final=True);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values())},final=True)
    print(json.dumps(clean(result)),flush=True);return 0 if result['qualified'] else 2


def launch(out,profile=False):
    global ARTIFACT_ROOT
    ARTIFACT_ROOT=out
    source_binding();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=None
    meta={'launcher_pid':os.getpid(),'started_utc':now(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':115,'kill_grace':5}
    cmd=['timeout','--signal=TERM','--kill-after=5s','115s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--out',str(out/'run')]
    if profile:cmd.append('--profile')
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=p.pid;save(out/'launcher-start.json',meta);code=p.wait(timeout=122)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc);code=2
        finally:
            if p is not None:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
    remaining=[]
    if p is not None:
        for q in Path('/proc').iterdir():
            if not q.name.isdigit():continue
            try:
                fields=(q/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==p.pid and fields[0]!='Z':remaining.append(int(q.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    meta.update(exit_code=code,owned_live_pids_remaining=remaining,finished_utc=now(),elapsed_seconds=time.monotonic()-start);save(out/'launcher-result.json',meta,final=True)
    files=inventory(out);size=sum(e['bytes'] for e in files.values());save(out/'manifest.json',{'files':files,'total_bytes':size,'within_limit':size<CAP+1024**2},final=True)
    for q in out.rglob('*'):
        if q.is_file():q.chmod(0o444)
    for q in sorted([q for q in out.rglob('*') if q.is_dir()],reverse=True):q.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code if not remaining and size<CAP+1024**2 else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--profile',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.out,a.profile) if a.execute else launch(a.out,a.profile))
