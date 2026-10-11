"""One fixed known-root floor and endpoint-side gauge-slice stability pilot."""
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
import certify_s2_period18_fold_small_march as io
import analyze_s2_period18_near_fold_gauge as parent
from neural_shrimp.rigorous_arclength_slices import slice_stability
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,midpoint_matrix
from flint import arb,arb_mat
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-arclength-slice-stability.md'
SOURCES=list(dict.fromkeys(io.SOURCES+parent.SOURCES+['src/neural_shrimp/rigorous_arclength_slices.py','src/neural_shrimp/rigorous_periodic_lyapunov.py','tools/certify_s2_period18_arclength_slices.py','tests/test_rigorous_arclength_slices.py','tests/test_s2_period18_arclength_slices.py',CARD]))
CAP=256*1024**2
PARENT=parent.PARENT
PIN=parent.PIN
EXECUTION=parent.EXECUTION
def partial_input():
    inp=parent.partial_input()
    chart=read(PARENT/'chart-009.json.gz')
    require(entry(PARENT/'chart-009.json.gz')['sha256']=='99e5505a4d9ec31c27138b2bc125ee10727ab148a6a362411bacb55d0d8de7f5','exact qualified156 chart')
    require(compact_chart(chart)==inp['previous'],'complete parent image/predictor identity')
    require(chart['passed'] and all(chart['gates'].values()),'all inherited parent gates')
    inp['chart']=chart
    return inp


def domains(chart,mode):
    lo,hi=map(Fraction,chart['exact_parameter_endpoints'])
    require(lo<hi,'positive parent gauge width')
    probe=(str(hi-(hi-lo)/16),str(hi))
    return [('probe',*probe)] if mode=='profile' else [('floor',str(hi),str(hi)),('probe',*probe)]


def save(path,obj):
    data=io.canonical_bytes(obj);root=path.parent
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
    inp=io.historical.synthetic();inp['model_inputs']['lr_hidden']=inp['model_inputs']['lr_output']='1/4'
    # A proposal-only exact sparse border for the zero-state ridge model.
    # No target state or target derivative is used for this engineering profile.
    mat=arb_mat(235,235)
    for phase in range(18):
        for i in range(13):
            row=13*phase+i;mat[row,row]=exact('1/2' if i==12 else '3/4')
            mat[row,13*((phase+1)%18)+i]-=1
    mat[234,234]=1
    previous={'passed':False,'gates':{'uniform_root':False,'primitive_period':False,'anchor_identified':False},
      'statement':{'period':18,'phase_dimension':13,'shooting_dimension':235},'normal':['0']*234+['1'],
      'origin':['0']*234+['1/4'],'parameter_center':'0','exact_parameter_endpoints':['-1/64','1/64'],
      'center':vector_json([arb(0)]*234+[exact('1/4')]),
      'predictor_slope':matrix_json(arb_mat([[int(i==234)] for i in range(235)])),
      'predictor_curvature':matrix_json(arb_mat(235,1)),
      'preconditioner':matrix_json(midpoint_matrix(mat.inv())),
      'image_correction':matrix_json(arb_mat([[arb(0,arb((1,-16)))] for _ in range(235)]))}
    return {'model_inputs':inp['model_inputs'],'chart':previous,'provenance':inp['provenance']}


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':90,'unit_derivative_cap':180,'internal_seconds':20,'tick_cap':400}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':90,'map_returned':90,'unit_attempted':180,'unit_returned':180,'map_cap':90,'unit_cap':180},'profile all90 native calls')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'bounded profile RSS')
    require(result['completed_intersections']==3 and result['completed_cases']==1 and result['completed_physical_phases']==18,'complete profile localization and physical metrics')
    require(result['cases']==[{'role':'probe','passed':True,'point_passed':True,'interval_phases_passed':18}], 'profile must exercise all positive metric paths')
    require(2*rss<2*1024**2,'profile doubled RSS projection')
    require(4*elapsed+10<90,'profile projected target time')
    estimate=max(6*total(path),128*1024**2)
    require(estimate<224*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'projected_seconds':4*elapsed+10,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=20 if mode=='profile' else 90
    calls=90 if mode=='profile' else 180;tick_cap=400 if mode=='profile' else 800
    ticks=0;serial=0;model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started','completed_intersections':0,'completed_cases':0,'completed_physical_phases':0}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':calls,'unit_derivative_cap':2*calls,'internal_seconds':limits,'tick_cap':tick_cap,
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
        if kind=='slice_physical_jacobians':outcome['completed_physical_phases']+=len(row['phase_jacobians'])
        tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        results=[]
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else partial_input()
            if mode=='target':require(inp['chart']['exact_parameter_endpoints']==['11/32','45/128'],'exact last qualified chart')
            payload=save(out/'inputs.json.gz',inp);save(out/'inputs-payload.json',payload);tick()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=calls)
            for role,left,right in domains(inp['chart'],mode):
                save(out/f'{role}-domain.json',{'role':role,'exact_endpoints':[left,right]});begin=serial
                result=slice_stability(model,inp['chart'],left,right,iterations=3,precision=256,
                                      synthetic_profile=mode=='profile',tick=tick,observe=observe)
                local=result['localization'];local.pop('start');local.pop('intersections')
                result['stage_indices']=list(range(begin,serial));result['role']=role
                payload=save(out/(role+'-result.json.gz'),result);save(out/(role+'-result-payload.json'),payload)
                results.append({'role':role,'passed':result['passed'],'point_passed':result['stability']['point_passed'],
                  'interval_phases_passed':sum(v.get('interval_D_positive',{}).get('passed',False) for v in result['stability']['phases'])})
                outcome['completed_cases']+=1;tick()
        # Preserve both fixed cases even if a sufficient stability gate fails.
        code=0
        outcome.update(passed=mode=='target' and all(v['passed'] for v in results),cases=results,
            status='fixed_slice_pilot_completed' if mode=='target' else 'synthetic_profile_completed',scientific_claim=mode=='target')
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);save(out/'manifest.json',{'files':inventory(out),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 105
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin)]:
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
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires pinned profile')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest)
if __name__=='__main__':raise SystemExit(main())
