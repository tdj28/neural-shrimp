"""One fresh doubled-width known-root stability slice, with synthetic profile."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,json,math,platform,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_correlated_cocycle as io
import certify_s2_period18_norm_cocycle as norm
from neural_shrimp.rigorous_stability_slice import stability_slice
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
CARD='docs/plans/s2-period18-doubled-stability-slice.md'
SOURCES=list(dict.fromkeys(norm.SOURCES+['src/neural_shrimp/rigorous_stability_slice.py','tools/certify_s2_period18_stability_slice.py','tools/launch_s2_period18_stability_slice.py','tests/test_rigorous_stability_slice.py','tests/test_s2_period18_stability_slice.py',CARD]))
ENDPOINTS=['359/1024','45/128']
CAP=384*1024**2
io.CAP=CAP

class DurabilityError(RuntimeError):pass

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name);result[name]=io.sha256(b)
    return result

def input_for(mode):
    if mode=='profile':
        value=io.io.profile_input();value['chart']['exact_weights']=['1']*235
        value['exact_endpoints']=['-1/2048','1/2048']
    else:
        value=io.io.partial_input()
        require(value['chart']['exact_parameter_endpoints']==['11/32','45/128'],'same exact qualified156 chart')
        value['exact_endpoints']=ENDPOINTS
    return value

def profile_gate(path,pin,outer_pin):
    outer=path.parent
    require(io.entry(outer/'manifest.json')['sha256']==outer_pin,'pinned profile launcher manifest')
    io.verify_inventory(outer,io.read(outer/'manifest.json')['files'])
    launch=io.read(outer/'launcher-result.json')
    require(launch['mode']=='profile' and launch['source_sha256']==sources() and launch['exit_code']==0 and launch['owned_live_pids']==[],'profile source/exit/ownership')
    for key,cap in [('elapsed_seconds',42),('max_group_rss_kib',2*1024**2)]:
        v=launch[key];require(type(v) in (int,float) and math.isfinite(v) and 0<v<cap,'bounded positive profile launcher '+key)
    require(io.entry(path/'manifest.json')['sha256']==pin,'pinned profile worker manifest')
    io.verify_inventory(path,io.read(path/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json')
    require(rt['mode']=='profile' and rt['source_sha256']==sources() and out['exit_code']==0 and out['status']=='synthetic_profile_completed','same complete profile')
    settings={'python':platform.python_version(),'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,
      'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,'precision':256,'map_cap':162,'unit_derivative_cap':324,'mixed_call_cap':18,'internal_seconds':30,'tick_cap':800}
    require(all(rt.get(k)==v for k,v in settings.items()),'exact profile libraries/settings')
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile threads/nice')
    require(out['adapter_counts']=={'map_attempted':162,'map_returned':162,'unit_attempted':324,'unit_returned':324,'map_cap':162,'unit_cap':324},'all162 C1 calls')
    require(out['mixed_counts']=={'attempted':18,'returned':18,'cap':18},'all18 mixed calls')
    require(out['completed_intersections']==6 and out['completed_physical_phases']==18 and out['completed_products']==18 and out['stages']==85,'complete85-stage profile')
    require(out['result_status']=='complete_slice_bounds' and out['passed'],'all positive profile gates')
    elapsed=out['elapsed_seconds'];rss=out['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<30,'positive profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<rss<1024**2,'doubled profile RSS projection')
    require(4*elapsed+10<90,'profile time projection')
    estimate=max(6*io.total(path),192*1024**2);require(estimate<336*1024**2,'profile raw projection')
    return {'worker_manifest':pin,'outer_manifest':outer_pin,'projected_seconds':4*elapsed+10,'projected_bytes':estimate,'scope':'engineering admission only'}

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();limit=30 if mode=='profile' else 90
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'mode':mode,
      'pid':os.getpid(),'ppid':os.getppid(),'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1'),
      'python':platform.python_version(),'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'precision':256,'map_cap':162,'unit_derivative_cap':324,'mixed_call_cap':18,'tick_cap':800,'internal_seconds':limit,'rss_cap_kib':2*1024**2,'raw_cap':CAP,
      'exact_endpoints':ENDPOINTS if mode=='target' else ['-1/2048','1/2048'],'scientific_claim':False}
    io.save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    stopped=False;serial=0;ticks=0;model=None;jet=None;code=2
    outcome={'status':'started','passed':False,'completed_intersections':0,'completed_physical_phases':0,'completed_products':0,'scientific_claim':False}
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if stopped or ticks>800 or time.monotonic()-started>=limit:raise TimeoutError('fixed time/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('fixed RSS cap')
        if io.total(out)>336*1024**2:raise OSError('ordinary raw cap')
    def observe(kind,row):
        nonlocal serial
        try:
            name=f'stage-{serial:03d}-{kind}';payload=io.save(out/(name+'.json.gz'),row);io.save(out/(name+'-payload.json'),payload);serial+=1
        except Exception as exc:raise DurabilityError('failed durable stage: '+str(exc))from exc
        if kind.endswith('slice_intersection'):outcome['completed_intersections']+=1
        if kind=='phase_derivatives':outcome['completed_physical_phases']+=1
        if kind=='norm_product':outcome['completed_products']+=1
        tick()
    try:
        if mode=='target':io.save(out/'cost-gate.json',profile_gate(profile,profile_pin,profile_outer_pin))
        with ctx.workprec(256):
            inp=input_for(mode);payload=io.save(out/'inputs.json.gz',inp);io.save(out/'inputs-payload.json',payload);tick()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=162);jet=io.CountedMixedMap(inp['model_inputs'])
            require(jet.n==model.n==13 and jet.hidden==model.hidden,'identical native model dimensions')
            result=stability_slice(model,jet,inp['chart'],*inp['exact_endpoints'],synthetic_profile=mode=='profile',tick=tick,observe=observe)
            payload=io.save(out/'slice-result.json.gz',result);io.save(out/'slice-result-payload.json',payload)
            outcome.update(passed=result['passed'],result_status=result['status'],exact_endpoints=result['exact_endpoints'])
        code=0;outcome['status']='synthetic_profile_completed' if mode=='profile' else 'fixed_slice_bounds_completed'
        outcome['scope']='Producer strict interval gates only, pending independent outcome review; no full-branch/neutral-fold/geometry claim.'
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});io.save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        if jet is not None:outcome['mixed_counts']=jet.counts()
        outcome.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stages=serial,ticks=ticks,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);io.save(out/'manifest.json',{'files':io.inventory(out)})
    print(json.dumps(outcome),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only for target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
