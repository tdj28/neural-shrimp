"""Seven uncomputed exact slices complete chart156, conditional on strict gates."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import argparse,json,math,platform,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_refined_tangent_cocycle as previous
io=previous.io
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_refined_stability_slice import refined_stability_slice
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
CARD='docs/plans/s2-period18-chart-stability.md'
SOURCES=list(dict.fromkeys(previous.SOURCES+['src/neural_shrimp/rigorous_refined_stability_slice.py','tools/certify_s2_period18_chart_stability.py','tools/launch_s2_period18_chart_stability.py','tests/test_rigorous_refined_stability_slice.py','tests/test_s2_period18_chart_stability.py',CARD]))
REUSED=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-refined-tangent-cocycle-2026-10-10/attempt-001-target/run')
REUSED_PIN='37a1b50124970d35ea11cf6c6815da80a194f6a81d6abf2484ca87fabb88c42a'
REUSED_EXEC='a42bc33fba39fbf3b1c99440e13c81a804ec46fd'
CAP=512*1024**2;ORDINARY=448*1024**2;io.CAP=CAP

class DurabilityError(RuntimeError):pass

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(b)
    return result

def partition():
    # Descend away from the completed endpoint-side slice, no repeated prefix.
    return [[str(Fraction(k,1024)),str(Fraction(k+1,1024))] for k in range(358,351,-1)]

def coverage(rows,reused):
    all_ends=sorted([[Fraction(v) for v in x] for x in rows+[reused]])
    require(len(all_ends)==8 and all_ends[0][0]==Fraction(11,32) and all_ends[-1][1]==Fraction(45,128),'exact eight-slice coverage endpoints')
    require(all(b-a==Fraction(1,1024) for a,b in all_ends) and all(all_ends[i][1]==all_ends[i+1][0] for i in range(7)),'no coverage gaps/overlaps')
    return [str(all_ends[0][0]),str(all_ends[-1][1])]

def reused_result(inp):
    rt,out=io.authenticate(REUSED,REUSED_PIN,REUSED_EXEC)
    value=io.read(REUSED/'result.json.gz');old=io.read(REUSED/'inputs.json.gz')
    require(rt['mode']=='target' and out['passed'] and out['stages']==75 and out['mixed_counts']=={'attempted':18,'returned':18,'cap':18},'qualified reused slice')
    require(value['passed'] and value['metric']['passed'] and value['exact_endpoints']==['359/1024','45/128'],'complete unchanged endpoint slice')
    require(old['model_inputs']==inp['model_inputs'],'identical exact native map')
    # The reused acquisition independently binds the same full original chart.
    old_rt,old_out=io.authenticate(previous.prior.PARENT,previous.prior.PIN,previous.prior.EXECUTION)
    old_input=io.read(previous.prior.PARENT/'inputs.json.gz')
    require(old_input['chart']==inp['chart'] and old_input['model_inputs']==inp['model_inputs'],'identical inherited chart/gauge/phase')
    return {'root':str(REUSED),'manifest':REUSED_PIN,'source_commit':REUSED_EXEC,'exact_endpoints':value['exact_endpoints'],'passed':True,'scope':'Producer bound reused once; independent review remains separate.'}

def input_for(mode):
    if mode=='profile':
        value=io.io.profile_input();value['chart']['exact_weights']=['1']*235
        value['slices']=[['-1/2048','1/2048']];value['reused']=None
    else:
        value=io.io.partial_input();require(value['chart']['exact_parameter_endpoints']==['11/32','45/128'],'exact qualified chart156')
        value['slices']=partition();value['reused']=reused_result(value);coverage(value['slices'],value['reused']['exact_endpoints'])
    return value

def profile_gate(path,pin,outer_pin):
    require(io.entry(path.parent/'manifest.json')['sha256']==outer_pin,'profile outer pin');io.verify_inventory(path.parent,io.read(path.parent/'manifest.json')['files'])
    launch=io.read(path.parent/'launcher-result.json')
    require(launch['mode']=='profile' and launch['source_sha256']==sources() and launch['exit_code']==0 and launch['owned_live_pids']==[],'profile source/exit/ownership')
    for key,cap in [('elapsed_seconds',47),('max_group_rss_kib',2*1024**2)]:
        v=launch[key];require(type(v) in (int,float) and math.isfinite(v) and 0<v<cap,'positive bounded launcher '+key)
    require(io.entry(path/'manifest.json')['sha256']==pin,'profile worker pin');io.verify_inventory(path,io.read(path/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json')
    require(rt['mode']=='profile' and rt['source_sha256']==sources() and out['exit_code']==0 and out['passed'] and out['completed_slices']==1 and out['stages']==161,'complete positive161-stage profile')
    settings={'python':platform.python_version(),'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,'precision':256,'internal_seconds':30,'per_slice_C1_cap':162,'per_slice_mixed_cap':36,'tick_cap':6400}
    require(all(rt.get(k)==v for k,v in settings.items()),'exact libraries/settings')
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'CPU1 nice10')
    require(out['counts']=={'map_attempted':162,'map_returned':162,'unit_attempted':324,'unit_returned':324,'mixed_attempted':36,'mixed_returned':36},'full synthetic arithmetic coverage')
    elapsed=out['elapsed_seconds'];rss=out['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<30 and 21*elapsed+15<300,'threefold seven-slice measured time')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<rss<1024**2,'doubled peak RSS')
    estimate=max(42*io.total(path),336*1024**2);require(estimate<ORDINARY,'sixfold seven-slice measured raw')
    return {'worker_manifest':pin,'outer_manifest':outer_pin,'projected_seconds':21*elapsed+15,'projected_bytes':estimate,'scope':'engineering admission only'}

def stream(inp,mode,construct,run,retain,tick=lambda:None):
    rows=[]
    for index,ends in enumerate(inp['slices']):
        tick();models=construct(index)
        result=run(index,models,ends)
        retain(index,result);tick()
        rows.append({'index':index,'exact_endpoints':ends,'passed':bool(result['passed']),'status':result['status']})
        if not result['passed']:break
    complete=len(rows)==len(inp['slices']) and all(r['passed'] for r in rows)
    return {'passed':complete,'rows':rows,'complete_coverage':coverage([r['exact_endpoints'] for r in rows],inp['reused']['exact_endpoints']) if complete and mode=='target' else None}

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();limit=30 if mode=='profile' else 300;serial=0;ticks=0;stopped=False;code=2;models=[]
    rt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),
      'started_utc':io.now(),'precision':256,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1'),'python':platform.python_version(),'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,
      'internal_seconds':limit,'per_slice_C1_cap':162,'per_slice_mixed_cap':36,'slice_cap':1 if mode=='profile' else 7,'tick_cap':6400,'rss_cap_kib':2*1024**2,'raw_cap':CAP}
    io.save(out/'runtime-start.json',rt);outcome={'status':'started','passed':False,'scientific_claim':False,'completed_slices':0};counts=dict.fromkeys(('map_attempted','map_returned','unit_attempted','unit_returned','mixed_attempted','mixed_returned'),0)
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
    def tick():
        nonlocal ticks
        ticks+=1
        if stopped or ticks>6400 or time.monotonic()-started>=limit:raise TimeoutError('fixed time/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('fixed RSS cap')
        if io.total(out)>ORDINARY:raise OSError('ordinary raw cap')
    def observe(index,kind,row):
        nonlocal serial
        try:
            name=f'slice-{index:02d}-stage-{serial:04d}-{kind}';payload=io.save(out/(name+'.json.gz'),row);io.save(out/(name+'-payload.json'),payload);serial+=1
        except Exception as exc:raise DurabilityError('durable stage failed: '+str(exc))from exc
        tick()
    def flush_counts():
        nonlocal models
        if models:
            for key,v in models[0].counts().items():
                if key in counts:counts[key]+=v
            for model in models[1:]:
                d=model.counts();counts['mixed_attempted']+=d['attempted'];counts['mixed_returned']+=d['returned']
            models=[]
    try:
        if mode=='target':io.save(out/'cost-gate.json',profile_gate(profile,profile_pin,profile_outer_pin))
        with ctx.workprec(256):
            inp=input_for(mode);payload=io.save(out/'inputs.json.gz',inp);io.save(out/'inputs-payload.json',payload);tick()
            def construct(index):
                nonlocal models
                flush_counts();require(ctx.prec==256,'constructor precision');models=[ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=162),io.CountedMixedMap(inp['model_inputs']),io.CountedMixedMap(inp['model_inputs'])]
                require(all(v.n==13 and v.hidden==9 for v in models),'same native model');io.save(out/f'slice-{index:02d}-start.json',{'index':index,'exact_endpoints':inp['slices'][index],'precision':ctx.prec});return models
            def run(index,items,ends):return refined_stability_slice(*items,inp['chart'],*ends,synthetic_profile=mode=='profile',tick=tick,observe=lambda k,r:observe(index,k,r))
            def retain(index,result):
                payload=io.save(out/f'slice-{index:02d}-result.json.gz',result);io.save(out/f'slice-{index:02d}-result-payload.json',payload);outcome['completed_slices']+=1
            summary=stream(inp,mode,construct,run,retain,tick);io.save(out/'coverage.json',summary)
            outcome.update(passed=summary['passed'],complete_coverage=summary['complete_coverage'],rows=summary['rows'])
        code=0;outcome['status']='fixed_chart_slices_completed'
    except BaseException as exc:outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);flush_counts();outcome.update(counts=counts,elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stages=serial,ticks=ticks,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);io.save(out/'manifest.json',{'files':io.inventory(out)})
    print(json.dumps(outcome));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
