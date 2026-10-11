"""Two fixed remaining historical flip rays: carried18 and doubled36, numerical only."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,importlib.util,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prior_flip_driver',ROOT/'tools/explore_s2_period18_flip.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from neural_shrimp.numerical_flip import correct,qualify,refine_flip,crossing,doubled
MAP=old.RAW/'explore/a2-survey/maps/S2-28011-zoom-c587/map.npz'
MAP_SHA='500e4e0250e8d08d844e302698e1531b3593bdefda84ea0cb5fcf35c56cf3c1b'
CARD='docs/plans/s2-period18-other-flips.md'
SOURCES=old.SOURCES+['tools/explore_s2_period18_other_flips.py','tests/test_s2_period18_other_flips.py',CARD]
CAP=64*1024**2;ORDINARY=56*1024**2;INTERNAL=60;EXTERNAL=75;CALLS=80000
ROOT_OUT=None
RESUME=old.RAW/'explore/s2-period18-other-flips/attempt-001'
RESUME_SHA='949636d923030117bc714f43f528e9ac5c8ebb87ba1d552c6eb6a0fe84d3c828'

def size():return sum(p.stat().st_size for p in ROOT_OUT.rglob('*') if p.is_file())
def save(p,obj,final=False):
    payload=(json.dumps(old.clean(obj),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    limit=CAP if final else ORDINARY
    if size()+len(payload)>limit:raise OSError('aggregate64MiB output cap; returned evaluations remain journaled')
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def binding():
    old.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed source')
    for p in SOURCES:old.require((ROOT/p).read_bytes()==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed source '+p)
    return {p:old.digest(ROOT/p) for p in SOURCES}

def select_rays(window,saved):
    """Exactly reconstruct the original coarse adjacency; not orbit qualification."""
    period,status=saved['period'],saved['status'];lh,lo=saved['log10_hidden'],saved['log10_output']
    dh,do=lh[1]-lh[0],lo[1]-lo[0];center=np.asarray(window['center']);scale=np.asarray(window['scale']);rows=[]
    for i,ray in enumerate(window['rays']):
        if ray['type']!='flip' or ray['s_boundary'] is None:continue
        direction=np.asarray(ray['direction'])*scale;first=None
        for s in np.linspace(0,1,4000):
            point=center+s*direction;r,c=int(round((point[0]-lh[0])/dh)),int(round((point[1]-lo[0])/do))
            if not(0<=r<len(lh) and 0<=c<len(lo)):break
            if not(status[r,c]==2 and period[r,c]==window['period']):
                first={'s':float(s),'row':r,'column':c,'status':int(status[r,c]),'period':int(period[r,c])};break
        rows.append({'ray_index':i,'historical_boundary':ray['s_boundary'],'first_other_cell':first})
    chosen=[r['ray_index'] for r in rows if r['ray_index']!=13 and r['first_other_cell'] is not None and (r['first_other_cell']['status'],r['first_other_cell']['period'])==(2,36)][:2]
    old.require(len(chosen)==2,'two fixed remaining adjacent36 rays available')
    return {'selected':chosen,'all_flip_rows':rows,'rule':'first two ray indices excluding13 whose original nearest-cell first non18 label is status2/period36; not a qualified36 association'}

def prepare(index,window):
    inp=old.prepare();ray=window['rays'][index];ell=[float(x) for x in window['center']];direction=[float(a)*float(b) for a,b in zip(window['scale'],ray['direction'])]
    f=inp['family'];f.update(ell=[old.exact(x) for x in ell],direction=[old.exact(x) for x in direction],ell_float=ell,direction_float=direction,historical_scale=window['scale'],historical_unit_direction=ray['direction'])
    inp.update(ray_index=index,ray=ray,map_sha256=MAP_SHA);return inp

class Experiment(old.Experiment):
    def __init__(self,inputs,out,budget):super().__init__(inputs,out);self.budget=budget
    def check(self):
        if self.budget['stopped'] or time.monotonic()-self.budget['start']>INTERNAL:raise TimeoutError('shared60s deadline')
        if self.budget['completed_points']+self.points>=CALLS:raise TimeoutError('shared80000 joint-call cap')
        if size()>ORDINARY:raise OSError('ordinary aggregate output cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS2GiB')
    def append(self,handle,obj):
        b=(json.dumps(old.clean(obj),sort_keys=True,allow_nan=False)+'\n').encode()
        if size()+len(b)>ORDINARY:
            save(self.out/'overflow-return.json',{'journal':handle.name,'returned_record':obj,'reason':'ordinary aggregate cap'},final=True);raise OSError('ordinary journal cap')
        handle.write(b.decode());handle.flush();os.fsync(handle.fileno());self.bytes+=len(b)

    def evaluate(self,theta,s,period):
        self.check();theta=np.asarray(theta,dtype=float);z=theta.copy();n=len(z);m=np.eye(n);b=np.zeros(n);orbit=[z.tolist()];checkpoints={};self.returns+=1
        record={'id':self.returns,'theta':theta.tolist(),'s':float(s),'period':period}
        try:
            rates=np.array([10.0**float(a+float(s)*d) for a,d in zip(self.ell,self.direction)])
            rv=self.net.rate_vector([rates[0]],[rates[1]])[0]
            # Rates are positive; their derivative along a ray is signed.
            direction=np.r_[np.full(self.net.hidden_parameter_count,self.direction[0]),np.full(n-self.net.hidden_parameter_count,self.direction[1])]
            dv=rv*np.log(10.)*direction
            old.require(np.isfinite(dv).all(),'finite signed parameter-rate derivative')
            record.update(rates=rates.tolist(),rates_exact=[old.exact(t) for t in rates],rate_derivative=dv.tolist())
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                for k in range(period):
                    self.check();self.points+=1
                    grad=self.net.gradient(z[None,:],self.x,self.y)[0];h=self.net.hessian(z,self.x,self.y)
                    j=np.eye(n)-rv[:,None]*h;b=j@b-dv*grad;m=j@m;z=z-rv*grad;orbit.append(z.tolist())
                    if k+1 in (period//2,period):checkpoints[str(k+1)]={'state':z.tolist(),'monodromy':m.tolist(),'parameter_derivative':b.tolist()}
                    old.require(np.isfinite(z).all() and np.isfinite(m).all() and np.isfinite(b).all(),'finite joint return')
            record.update(orbit=orbit,monodromy=m.tolist(),parameter_derivative=b.tolist(),residual=(z-theta).tolist(),checkpoints=checkpoints,completed=True)
        except BaseException as exc:
            record.update(orbit=orbit,monodromy=m.tolist(),parameter_derivative=b.tolist(),checkpoints=checkpoints,completed=False,error=type(exc).__name__+': '+str(exc));self.append(self.log,record);raise
        self.append(self.log,record);return record


def one_ray(inp,out,budget,source):
    out.mkdir();save(out/'inputs.json',inp);exp=Experiment(inp,out,budget);result={'ray_index':inp['ray_index'],'status':'started','flip_qualified':False,'doubled_qualified_count':0,'doubled_attracting_count':0,'continuous_branch_proof':False};nodes=[];transitions=[]
    try:
        x=np.asarray(inp['historical_orbit'][0]);previous=None
        for i,s in enumerate(np.linspace(0,.999*inp['ray']['s_boundary'],65)):
            node=correct(exp.evaluate,x,float(s),18);node['s']=float(s);save(out/f'node-{i:03d}.json',node)
            brief={k:node[k] for k in ('qualified','state','s')};brief['qualification']=node.get('qualification');nodes.append(brief);save(out/'path.json',nodes)
            old.require(node['qualified'],'forward correction unresolved')
            new=np.asarray(node['state'])
            if previous is not None:
                reverse=correct(exp.evaluate,new,previous['s'],18);scale=max(1.,float(np.max(abs(x))))
                error=float(np.max(abs(np.asarray(reverse['state'])-x))/scale)
                movement=float(np.max(abs(np.asarray(node['data']['orbit'][:-1])-np.asarray(previous['data']['orbit'][:-1])))/scale)
                edge={'from_s':previous['s'],'to_s':float(s),'reverse':reverse,'reverse_error':error,'full_native_phase_step':movement,'qualified':bool(reverse['qualified'] and error<1e-7 and movement<.1)}
                save(out/f'edge-{i:03d}.json',edge);old.require(edge['qualified'],'phase/reverse link unresolved')
                if previous['qualification']['stability']!=node['qualification']['stability']:
                    transitions.append({'from_s':previous['s'],'to_s':float(s),'from_qualification':previous['qualification'],'to_qualification':node['qualification']});save(out/'stability-transitions.json',transitions)
            x=new;previous=node
        point=refine_flip(exp.evaluate,x,nodes[-1]['s'],18,exp.emit);save(out/'flip.json',point);result['flip_qualified']=point['qualified'];old.require(point['qualified'],'selected flip unqualified')
        # A root found far from the selected historical boundary is a different target.
        old.require(abs(point['s']-inp['ray']['s_boundary'])<.01,'selected historical boundary proximity')
        seed={'schema':'neural-shrimp-flip-proof-seed-v1','period':18,'consumed_map':inp['consumed_map'],'family':inp['family'],'theta':[old.exact(x) for x in point['state']],'parameter':old.exact(point['s']),'right_vector':[old.exact(x) for x in point['q']],'source_commit':source,'inputs_sha256':old.digest(out/'inputs.json'),'ray_index':inp['ray_index']}
        save(out/'flip-proof-seed.json',seed)
        crossed=crossing(exp.evaluate,point,18,exp.emit);save(out/'crossing.json',crossed);result['crossing_qualified']=crossed['qualified']
        scale=max(1.,max(abs(np.asarray(point['state']))));branches=[];last=None
        for exponent in (12,11,10):
            amp=2.**-exponent*scale;seed=None
            if last is not None and last['qualified']:
                ratio=amp/last['amplitude_requested'];seed=np.r_[np.asarray(point['state'])+ratio*(np.asarray(last['state'])-np.asarray(point['state'])),point['s']+ratio**2*last['delta_s']]
            try:
                branch=doubled(exp.evaluate,point,18,amp,seed,exp.emit)
                opposite=exp.evaluate(np.asarray(branch['data']['orbit'][18]),branch['s'],36);branch['opposite_phase']=opposite;branch['opposite_qualification']=qualify(opposite)
                branch['gates']['opposite_phase_qualified']=branch['opposite_qualification']['qualified'];branch['qualified']=all(branch['gates'].values());last=branch
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:branch={'qualified':False,'amplitude_requested':amp,'error':type(exc).__name__+': '+str(exc)}
            branches.append(branch);save(out/'doubled.json',branches)
        result.update(status='numerical_pilot_complete',flip_s=point['s'],doubled_qualified_count=sum(b['qualified'] for b in branches),doubled_attracting_count=sum(b['qualified'] and b['qualification']['stability']=='attracting' and b['opposite_qualification']['stability']=='attracting' for b in branches),first_observed_transition=transitions[0] if transitions else None,transitions=transitions)
    except BaseException as exc:result.update(status='incomplete_or_failed',error=type(exc).__name__+': '+str(exc))
    finally:
        exp.close();budget['completed_points']+=exp.points;result.update(path_nodes=len(nodes),joint_calls=exp.points,returned_evaluations=exp.returns,events=exp.events,elapsed_seconds=time.monotonic()-exp.start);save(out/'outcome.json',result,final=True)
    return result

def execute(out):
    global ROOT_OUT
    ROOT_OUT=out;bound=binding();old.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required');out.mkdir(parents=True,exist_ok=False);budget={'start':time.monotonic(),'completed_points':0,'stopped':False};source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();result={'status':'started','rays':[]}
    runtime={'source_commit':source,'source_sha256':bound,'started_utc':old.now(),'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'cpu_only':True,'numpy':np.__version__,'scipy':old.scipy.__version__,'python':old.platform.python_version(),'internal_seconds':INTERNAL,'joint_call_cap':CALLS,'raw_bytes':CAP,'monitored_rss_bytes':2*1024**3}
    save(out/'runtime-start.json',runtime)
    for p in SOURCES:q=out/'source'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
    def stop(*_):budget['stopped']=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        old.require(old.digest(MAP)==MAP_SHA,'fixed original grid identity');window=old.read(old.RAYS)['S2-28011-zoom:587']
        with np.load(MAP,allow_pickle=False) as saved:selection=select_rays(window,saved)
        save(out/'selection.json',dict(selection,map=old.entry(MAP),rays=old.entry(old.RAYS)))
        old.require(old.digest(RESUME/'manifest.json')==RESUME_SHA,'retained first attempt manifest')
        manifest=old.read(RESUME/'manifest.json');old.require(old.inventory(RESUME)==manifest['files'],'complete retained first attempt inventory')
        previous=old.read(RESUME/'run/outcome.json');old.require(selection['selected']==[16,51] and [r['ray_index'] for r in previous['rays']]==[16,51],'same fixed selected rays')
        old.require(previous['rays'][0]['status']=='numerical_pilot_complete' and previous['rays'][1]['joint_calls']==0,'ray16 retained and ray51 unexecuted')
        save(out/'resume.json',{'path':str(RESUME),'manifest_sha256':RESUME_SHA,'retained_ray16':previous['rays'][0],'only_rerun_ray':51})
        for index in [51]:
            if budget['stopped'] or time.monotonic()-budget['start']>INTERNAL:raise TimeoutError('shared deadline before next ray')
            result['rays'].append(one_ray(prepare(index,window),out/f'ray-{index:02d}',budget,source));save(out/'progress.json',result,final=True)
        result['status']='fixed_ray51_recorded'
    except BaseException as exc:result.update(status='failed_or_interrupted',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-budget['start'],joint_calls=budget['completed_points'],max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=old.now());save(out/'outcome.json',result,final=True);files=old.inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(x['bytes'] for x in files.values())},final=True)
    print(json.dumps(old.clean(result)),flush=True);return 0 if result['status']=='fixed_ray51_recorded' else 2

def launch(out):
    global ROOT_OUT
    ROOT_OUT=out;bound=binding();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL}s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--out',str(out/'run')]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':bound,'argv':cmd,'launcher_pid':os.getpid(),'started_utc':old.now(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL,'kill_grace':5};save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=EXTERNAL+7)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    remaining=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                f=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(f[2])==proc.pid and f[0]!='Z':remaining.append(int(p.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    if remaining or size()>CAP:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_live_pids_remaining=remaining,finished_utc=old.now());save(out/'launcher-result.json',meta,final=True);files=old.inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(x['bytes'] for x in files.values())},final=True)
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in out.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':old.entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((execute if a.execute else launch)(a.out))
