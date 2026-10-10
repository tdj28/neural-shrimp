"""One fixed common-gauge correlated-cocycle stability pilot."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint,numpy,scipy
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_arclength_slices as io
import analyze_s2_period18_slice_budget as budget
from neural_shrimp.rigorous_correlated_cocycle import correlated_slice
from neural_shrimp.rigorous_folds import FoldTanhMap
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,midpoint_matrix,canonical_bytes
from flint import arb,arb_mat
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-correlated-cocycle.md'
SOURCES=list(dict.fromkeys(budget.SOURCES+['src/neural_shrimp/rigorous_correlated_cocycle.py','tools/certify_s2_period18_correlated_cocycle.py','tests/test_rigorous_correlated_cocycle.py','tests/test_s2_period18_correlated_cocycle.py',CARD]))
CAP=256*1024**2
PARENT=budget.PARENT
PIN=budget.PIN
EXECUTION=budget.EXECUTION
BUDGET=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-slice-budget-2026-10-10/attempt-002/run')
BUDGET_PIN='301b5a433201f4349dfd8fb319f1e269c1edb7d31eb4ab21981a0207fc1e14cc'
BUDGET_EXECUTION='f991e22571f25f7f972564b37d7e919fefb3c01a'
ENDPOINTS=['719/2048','45/128']

class CountedMixedMap(FoldTanhMap):
    def __init__(self,inputs,max_calls=18):
        require(type(max_calls) is int and max_calls==18,'fixed18 mixed calls')
        self.attempted=0;self.returned=0;self.max_calls=max_calls;self.precision=ctx.prec
        super().__init__(inputs)
    def jet_step(self,theta,hidden_rate):
        require(ctx.prec==self.precision and self.attempted<self.max_calls,'mixed precision/cap')
        self.attempted+=1
        result=super().jet_step(theta,hidden_rate);self.returned+=1;return result
    def counts(self):return {'attempted':self.attempted,'returned':self.returned,'cap':self.max_calls}

def authenticate(root,pin,execution):
    require(entry(root/'manifest.json')['sha256']==pin,'pinned input manifest')
    verify_inventory(root,read(root/'manifest.json')['files'])
    rt=read(root/'runtime-start.json');outcome=read(root/'outcome.json')
    require(rt['source_commit']==execution and rt['precision']==256 and outcome['exit_code']==0,'completed fixed input execution')
    for name,digest in rt['source_sha256'].items():
        require(entry(ROOT/name)['sha256']==digest and sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT))==digest,'unchanged input source '+name)
    return rt,outcome

def joined_inputs(inp,saved,probe,start,final,report):
    require(inp['chart']==saved['chart'] and inp['model_inputs']==saved['model_inputs'],'identical native parent and model')
    require(probe['role']=='probe' and probe['stage_indices']==list(range(7,14)),'complete fixed probe stages')
    require(probe['localization']['exact_endpoints']==start['exact_endpoints']==report['exact_endpoints']==ENDPOINTS,'same exact common-gauge slice')
    require(start['normal']==inp['chart']['normal'] and start['origin']==inp['chart']['origin'],'same gauge normal/origin')
    require(start['preconditioner']==inp['chart']['preconditioner'] and start['inherited_image']==inp['chart']['image_correction'],'same fixed C and inherited root')
    require(final['iteration']==2 and final['retained_error']==probe['localization']['final_error'],'same complete third intersection')
    # Redo only the inexpensive saved 235D Neumann arithmetic; no new derivative.
    tangent=budget.tangent_bound(restore_matrix(start['preconditioner']),restore_matrix(final['jacobian']),[exact(v) for v in inp['chart']['exact_weights']])
    require(tangent==report['implicit_tangent'] and tangent['passed'],'exact saved uniform implicit tangent reproduction')
    whole=probe['localization']['final_state_box']
    derivative=restore_matrix(tangent['derivative_box'])
    require(len(whole)==derivative.nrows()==235 and derivative.ncols()==1,'full native state and tangent')
    inp.update(whole_box=whole,whole_tangent=[tangent['derivative_box'][i][0] for i in range(235)],
      exact_endpoints=ENDPOINTS,uniform_tangent=tangent,
      derivative_dependency={'domain':final['derivative_domain'],'jacobian':final['jacobian'],
        'scope':'Uniform derivative Uprime=A(U)^-1 e_last on this same closed gauge interval. The last localization derivative domain contains every exact root; q<1 gives invertibility and the three fixed intersections retain all derivatives. Predictor slope is not substituted.'})
    return inp

def target_input():
    rt,outcome=authenticate(PARENT,PIN,EXECUTION)
    require(rt['mode']=='target' and outcome['completed_cases']==2,'completed floor and probe acquisition')
    brt,bout=authenticate(BUDGET,BUDGET_PIN,BUDGET_EXECUTION)
    require(brt['parent_manifest']==PIN and bout['status']=='saved_slice_budgets_completed','same saved derivative budget parent')
    inp=io.partial_input();saved=read(PARENT/'inputs.json.gz')
    result=joined_inputs(inp,saved,read(PARENT/'probe-result.json.gz'),read(PARENT/'stage-007-slice_start.json.gz'),read(PARENT/'stage-010-slice_intersection.json.gz'),read(BUDGET/'probe-budget.json'))
    result['binding']={'root':str(PARENT),'manifest':PIN,'execution':EXECUTION,'budget_root':str(BUDGET),'budget_manifest':BUDGET_PIN,'budget_execution':BUDGET_EXECUTION}
    return result


def save(path,obj):
    data=canonical_bytes(obj);root=path.parent
    payload={'uncompressed_bytes':len(data),'uncompressed_sha256':sha256(data)}
    require(len(data)<256*1024**2,'uncompressed artifact cap')
    if path.suffix=='.gz':data=gzip.compress(data,compresslevel=6,mtime=0)
    require(len(data)<60*1024**2,'single stored artifact cap')
    require(total(root)+len(data)<CAP,'aggregate raw cap')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    return payload
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        result[name]=sha256(b)
    return result

def profile_input():
    inp=io.profile_input();inp['exact_endpoints']=['0','1/2048']
    # Zero-state ridge model: root theta=0 for all actual h; exact Uprime=e_h.
    inp['whole_box']=vector_json([arb(0)]*234+[arb(exact('1025/4096'),exact('1/4096'))])
    inp['whole_tangent']=vector_json([arb(0)]*234+[arb(1)])
    return inp


def profile_gate(path,pin,outer_pin):
    outer=path.parent
    require(entry(outer/'manifest.json')['sha256']==outer_pin,'pinned profile launcher manifest')
    verify_inventory(outer,read(outer/'manifest.json')['files'])
    launcher=read(outer/'launcher-result.json')
    require(launcher['mode']=='profile' and launcher['source_sha256']==sources() and launcher['exit_code']==0 and launcher['owned_live_pids']==[],'successful unchanged profile launcher and cleanup')
    elapsed_outer=launcher['elapsed_seconds']
    require(type(elapsed_outer) in (int,float) and math.isfinite(elapsed_outer) and 0<elapsed_outer<42,'bounded positive profile launcher time')
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':90,'unit_derivative_cap':180,'mixed_call_cap':18,'internal_seconds':30,'tick_cap':500}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':90,'map_returned':90,'unit_attempted':180,'unit_returned':180,'map_cap':90,'unit_cap':180},'profile all90 native calls')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<30,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<rss<2*1024**2,'bounded profile RSS')
    require(result['completed_intersections']==3 and result['completed_cases']==1 and result['completed_physical_phases']==18 and result['completed_products']==18 and result['mixed_counts']=={'attempted':18,'returned':18,'cap':18},'complete profile localization and physical metrics')
    require(result['cases']==[{'role':'probe','passed':True,'point_passed':True,'interval_passed':True}], 'profile must exercise all positive metric paths')
    require(2*rss<2*1024**2,'profile doubled RSS projection')
    require(4*elapsed+10<90,'profile projected target time')
    estimate=max(6*total(path),128*1024**2)
    require(estimate<224*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'outer_manifest_sha256':outer_pin,'projected_seconds':4*elapsed+10,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=30 if mode=='profile' else 90
    calls=90;tick_cap=500
    ticks=0;serial=0;model=None;jet_model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started','completed_intersections':0,'completed_cases':0,'completed_physical_phases':0,'completed_products':0}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':calls,'unit_derivative_cap':2*calls,'mixed_call_cap':18,'internal_seconds':limits,'tick_cap':tick_cap,
      'raw_cap':CAP,'rss_cap_kib':2*1024**2,'parent_manifest':PIN if mode=='target' else None,'cpu_only':True}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limits)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if stopped or ticks>tick_cap or time.monotonic()-start>=limits:raise TimeoutError('fixed time/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('RSS cap')
        if total(out)>224*1024**2:raise OSError('ordinary raw cap')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload);serial+=1
        if kind=='slice_intersection':outcome['completed_intersections']+=1
        if kind=='correlated_phase_model':outcome['completed_physical_phases']+=1
        if kind=='correlated_product':outcome['completed_products']+=1
        tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin,profile_outer_pin))
        results=[]
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else target_input()
            payload=save(out/'inputs.json.gz',inp);save(out/'inputs-payload.json',payload);tick()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=calls)
            jet_model=CountedMixedMap(inp['model_inputs'])
            require(jet_model.n==model.n==13 and jet_model.hidden==model.hidden,'same native model layout')
            role='probe';left,right=inp['exact_endpoints']
            save(out/'probe-domain.json',{'role':role,'exact_endpoints':[left,right]});begin=serial
            result=correlated_slice(model,jet_model,inp['chart'],inp['whole_box'],inp['whole_tangent'],left,right,
              precision=256,synthetic_profile=mode=='profile',tick=tick,observe=observe)
            local=result['center_localization'];local.pop('start');local.pop('intersections')
            result['stage_indices']=list(range(begin,serial));result['role']=role
            payload=save(out/'probe-result.json.gz',result);save(out/'probe-result-payload.json',payload)
            results.append({'role':role,'passed':result['passed'],'point_passed':result['metric']['point_D_positive']['passed'],
              'interval_passed':result['metric']['interval_D_positive']['passed']})
            outcome['completed_cases']+=1;tick()
        # Preserve the one fixed case even when a sufficient stability gate fails.
        code=0
        outcome.update(passed=mode=='target' and all(v['passed'] for v in results),cases=results,
            status='fixed_correlated_slice_pilot_completed' if mode=='target' else 'synthetic_profile_completed',scientific_claim=mode=='target')
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        if jet_model is not None:outcome['mixed_counts']=jet_model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);save(out/'manifest.json',{'files':inventory(out),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=35 if mode=='profile' else 105
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--profile-outer-manifest',profile_outer_pin)]:
        if value is not None:cmd.extend([flag,str(value)])
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=seconds+7)
        except BaseException as exc:
            meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=[]
    if proc is not None:
        for q in Path('/proc').iterdir():
            if not q.name.isdigit():continue
            try:
                fields=(q/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(q.name))
            except (OSError,ValueError):pass
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_live_pids=live)
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);require(total(out)<CAP+1024**2,'total launcher plus worker cap')
    save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for q in sorted(out.rglob('*'),reverse=True):q.chmod(0o555 if q.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'target requires pinned profile')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest)
if __name__=='__main__':raise SystemExit(main())
