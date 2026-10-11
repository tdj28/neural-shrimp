"""Bounded containment refinement of the existing exact-gauge endpoint."""
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
import certify_s2_period18_fold_march as io
from neural_shrimp.rigorous_arclength_localization import localize_endpoint
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json
from flint import arb
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-fold-endpoint-recovery.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['src/neural_shrimp/rigorous_arclength_localization.py','tools/certify_s2_period18_fold_endpoint.py','tests/test_s2_period18_fold_endpoint.py',CARD]))
CAP=128*1024**2
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-march-2026-10-10/unit-001/run')
PIN='df78f5d1f4f1b358f8cdd3a9a6ccf3496949d5ae2568a9ff0471ca11232b6eb9'
EXECUTION='c33834a52bd260321d8bc4186210afdb26329ff9'

def validate_prefix(inp,rows,load,expected=15):
    """Read all durable native joins; never use the unqualified terminal state."""
    require(len(rows)==expected,'exact qualified prefix count')
    previous=inp['previous'];finals=[];initials=[]
    for index,row in enumerate(rows):
        name=f'chart-{index:03d}.json.gz'
        require(row['index']==index and row['artifact']['path']==name,'ordered durable chart index')
        c=load(name,row['artifact']);io.joined(previous,c)
        require(c['passed']==row['passed'] and c['gates']==row['gates'],'mirrored durable gates')
        require(not row['known_fold_identified'] and not any(a['contained_in_qualified_chart'] for a in c['known_fold_membership']['all_cyclic_alignments']),'not already joined to fold')
        require(row['reboxed']==('reboxing' in c),'mirrored rebox status')
        if row['reboxed']:
            ident=c['reboxing']['initial_artifact'];require(ident['path']==f'initial-{index:03d}.json.gz','exact retained initial name')
            old=load(ident['path'],ident)
            require(old['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'root-only initial failure')
            for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):
                require(c[key]==old[key],'unchanged rebox predictor '+key)
            initials.append(ident['path'])
        finals.append(name);previous=io.compact_chart(c)
    return previous,finals,initials


def partial_input():
    require(entry(PARENT/'manifest.json')['sha256']==PIN,'exact incomplete-parent manifest')
    verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    runtime=read(PARENT/'runtime-start.json');io.bound_sources(PARENT,runtime,EXECUTION)
    require(runtime['executed_runner']==io.CURRENT and runtime['source_sha256'].get(io.CURRENT)==entry(ROOT/io.CURRENT)['sha256'],'exact parent runner')
    require(not (PARENT/'batch.json').exists(),'preserved absent terminal batch')
    outcome=read(PARENT/'outcome.json')
    require(outcome['exit_code']==2 and not outcome['passed'] and outcome['error']=={'type':'ValueError','message':'nonnegative finite radii proposal'},'exact retained terminal proposal failure')
    inp=read(PARENT/'inputs.json');original,endpoint=io.initial_anchor()
    require(inp['model_inputs']==original['model_inputs'] and inp['previous']==endpoint,'exact initial new-gauge anchor and model')
    require(inp['provenance']['parent_root']==str(io.INITIAL) and inp['provenance']['parent_manifest']==io.INITIAL_PIN and inp['provenance']['parent_execution']==io.INITIAL_EXECUTION,'exact initial dependency')
    rows=[read(q) for q in sorted(PARENT.glob('stage-*-chart_committed.json'))]
    def load(name,identity):
        require(entry(PARENT/name)=={k:identity[k] for k in ('sha256','bytes')},'bound retained artifact '+name)
        return read(PARENT/name)
    previous,finals,initials=validate_prefix(inp,rows,load)
    require(outcome['last_durable_chart']['index']==14 and outcome['last_durable_chart']['identity']==rows[-1]['artifact'],'last durable successful chart')
    terminal=outcome['last_durable_initial'];require(terminal['index']==15 and terminal['identity']['path']=='initial-015.json.gz','terminal failed initial identity')
    failed=load('initial-015.json.gz',terminal['identity'])
    require(not failed['passed'] and failed['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'unqualified tail stays excluded')
    require(failed['anchor']['state_box']==previous['endpoint_root_enclosures'][1] and failed['anchor']['exact_parameter']==previous['exact_parameter_endpoints'][1]=='33/128','failed tail references qualified endpoint only')
    require({p.name for p in PARENT.glob('chart-*.json.gz')}==set(finals),'exact final chart set')
    require({p.name for p in PARENT.glob('initial-*.json.gz')}==set(initials+['initial-015.json.gz']),'exact retained initial set')
    return {'model_inputs':inp['model_inputs'],'previous':previous,'provenance':{'parent_root':str(PARENT),'parent_manifest':PIN,'parent_execution':EXECUTION,'qualified_prefix':15,'last_chart':rows[-1]['artifact'],'failed_tail':terminal,'original_anchor':inp['provenance']['original_anchor'],'initial_new_gauge_manifest':io.INITIAL_PIN}}


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
    radius=arb((1,-16));box=[arb(0,radius) for _ in range(234)]+[arb(exact('1/4'),radius)]
    previous={'passed':False,'gates':{'uniform_root':False,'primitive_period':False,'anchor_identified':False},
      'statement':{'period':18,'phase_dimension':13,'shooting_dimension':235},'normal':['0']*234+['1'],
      'origin':['0']*234+['1/4'],'exact_parameter_endpoints':['-1/64','0'],'endpoint_root_enclosures':[[],vector_json(box)]}
    return {'model_inputs':inp['model_inputs'],'previous':previous,'provenance':inp['provenance']}


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':72,'unit_derivative_cap':144,'internal_seconds':20,'tick_cap':400}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':72,'map_returned':72,'unit_attempted':144,'unit_returned':144,'map_cap':72,'unit_cap':144},'profile all72 native calls')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'bounded profile RSS')
    require(result['completed_intersections']==3,'all three profile intersections')
    require(2*elapsed+5<30,'profile projected target time')
    estimate=max(4*total(path),80*1024**2)
    require(estimate<112*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'projected_seconds':2*elapsed+5,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=20 if mode=='profile' else 30
    ticks=0;serial=0;model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started','completed_intersections':0}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':72,'unit_derivative_cap':144,'internal_seconds':limits,'tick_cap':400,
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
        if stopped or ticks>400 or time.monotonic()-start>=limits:raise TimeoutError('fixed time/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('RSS cap')
        if total(out)>112*1024**2:raise OSError('ordinary raw cap')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload);serial+=1
        if kind=='intersection':outcome['completed_intersections']+=1
        tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else partial_input()
            if mode=='target':require(inp['previous']['exact_parameter_endpoints'][1]=='33/128','exact last qualified endpoint')
            save(out/'inputs.json',inp);tick()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=72)
            result=localize_endpoint(model,inp['previous'],iterations=3,precision=256,synthetic_profile=mode=='profile',tick=tick,observe=observe)
        # Stages already durably retain every dense matrix; compact result links them.
        result.pop('start');result.pop('intersections')
        result['stage_files']=[f'stage-{i:03d}-'+('localization_start' if i==0 else 'intersection')+'.json.gz' for i in range(4)]
        save(out/'result.json',result);tick();code=0
        outcome.update(passed=mode=='target',status='known_endpoint_localized' if mode=='target' else 'synthetic_profile_completed',scientific_claim=mode=='target')
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
    seconds=25 if mode=='profile' else 45
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
