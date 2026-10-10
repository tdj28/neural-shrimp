"""Fixed floor/probe experiments localizing the already proved mean-value branch."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gc,importlib.util,json,math,platform,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
import flint,numpy,scipy
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_orbits import exact,ball_json,matrix_json,midpoint_matrix,canonical_bytes
from neural_shrimp.rigorous_connection import require,shooting
from neural_shrimp.rigorous_gradient_enclosure import MeanValueHiddenRateMap
from neural_shrimp.rigorous_metric_slices import derivative_slice
from neural_shrimp.rigorous_periodic_lyapunov import periodic_witness
from neural_shrimp.rigorous_root_localization import localized_stability
spec=importlib.util.spec_from_file_location('frozen_meanvalue_driver',ROOT/'tools/certify_s2_period18_connection_meanvalue.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-meanvalue-2026-10-09/attempt-001-target/run')
PARENT_SHA='dbb6ea5e57606788cbf1a7659ebcff2b4e91ebad2d11ca7779b88c32e2dcc9b7'
PARENT_EXEC='916c76d35c1786b906991bebcc16406864e62c25'
CARD='docs/plans/s2-period18-localized-roots-pilot.md'
SOURCES=list(dict.fromkeys(common.SOURCES+[
 'src/neural_shrimp/rigorous_metric_slices.py','src/neural_shrimp/rigorous_periodic_lyapunov.py',
 'src/neural_shrimp/rigorous_root_localization.py','tools/certify_s2_period18_localized_roots.py',
 'tests/test_rigorous_root_localization.py',CARD,
 'docs/reviews/s2-period18-connection-meanvalue-outcome.json','docs/reviews/s2-period18-meanvalue-independent-review.json']))
CAP=128*1024**2
common.CAP=CAP
entry,read,save,now,total,inventory,verify_inventory=(getattr(common,n) for n in ('entry','read','save','now','total','inventory','verify_inventory'))
LIMITS={'profile':(20,25,126,252,300),'floor':(60,75,216,432,560),'probe':(60,75,180,360,400)}


def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        result[name]=common.sha256(data)
    return result


def environment():
    return {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'threads':dict.fromkeys(THREADS,'1')}


def completed_bundle(path,pin,mode):
    require(entry(path/'manifest.json')['sha256']==pin,'approved worker manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    outer=path.parent;verify_inventory(outer,read(outer/'manifest.json')['files'])
    launch=read(outer/'launcher-result.json');run=read(path/'runtime-start.json');out=read(path/'outcome.json')
    require(launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'successful completed owned process')
    for key,value in environment().items():require(run[key]==value,'library/thread equality '+key)
    require(run['mode']==out['mode']==mode and run['nice']>=10 and launch['nice']>=10,'mode/nice identity')
    internal,external,_,_,_=LIMITS[mode]
    for value,cap in ((out['elapsed_seconds'],internal),(launch['elapsed_seconds'],external+5),(out['max_rss_kib'],2*1024**2)):
        require(type(value) in (int,float) and math.isfinite(value) and 0<=value<cap,'finite bounded resource measurement')
    require(run['source_sha256']==sources(),'exact profiled source inventory')
    return out,run,launch


def profile_gate(path,pin):
    out,_,_=completed_bundle(path,pin,'profile')
    require(out['status']=='synthetic_profile_completed' and out['counts']['map_returned']==126 and out['counts']['unit_returned']==252,'complete profile coverage')
    require(len(out['probes'])==1 and out['probes'][0]['localized_passed'],'profile exercised full metric path')
    projected=3*out['elapsed_seconds']+5;size=3*total(path)
    require(projected<55 and size<120*1024**2,'profile projection gate')
    return {'manifest_sha256':pin,'projected_seconds':projected,'projected_bytes':size,
            'scope':'engineering safety factor only; scientific gates independent'}


def prepare_parent():
    require(entry(PARENT/'manifest.json')['sha256']==PARENT_SHA,'fixed reviewed parent manifest')
    verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    bundle=read(PARENT/'certificate.json.gz');runtime=bundle['runtime'];result=bundle['result']
    require(runtime['source_commit']==PARENT_EXEC,'fixed parent execution')
    for name,digest in runtime['source_sha256'].items():require(entry(ROOT/name)['sha256']==digest,'unchanged parent dependency '+name)
    require(result['gates']['connected_primitive_branch'] and len(result['charts'])==2,'reviewed connected parent')
    original=common.prepare();saved=read(PARENT/'inputs.json')
    require(canonical_bytes(original)==canonical_bytes(saved),'original model/data/rates/native anchor identity')
    keys=('gates','statement','center','predictor_slope','preconditioner','image_correction','exact_parameter_endpoints','parameter_center')
    charts=[{key:chart[key] for key in keys} for chart in result['charts']]
    require(all(c['gates']['uniform_root'] and c['gates']['primitive_period'] for c in charts),'both primitive parent roots')
    require(charts[0]['exact_parameter_endpoints'][1]==charts[1]['exact_parameter_endpoints'][0],'unchanged exact join')
    require(charts[0]['parameter_center']==original['rate'],'same original center')
    del bundle,result;gc.collect()
    return original,charts


def synthetic_parent(model,tick):
    # For y=0, theta=0 is an exact fixed point for every rate. Repeating it
    # eighteen times tests 234D arithmetic, with primitive-period explicitly false.
    zero=[[arb(0)]*13 for _ in range(18)];h=exact('1/128')
    _,jac,_,_=shooting(model,zero,h,tick)
    pre=midpoint_matrix(jac.inv())
    return {'gates':{'uniform_root':True,'primitive_period':False},
        'statement':{'period':18,'phase_dimension':13},'parameter_center':'1/128',
        'exact_parameter_endpoints':['7/1024','9/1024'],'center':[[ball_json(v) for v in row] for row in zero],
        'predictor_slope':matrix_json(arb_mat(234,1)),'preconditioner':matrix_json(pre),
        'image_correction':matrix_json(arb_mat([[arb(0,exact('1/524288'))] for _ in range(234)]))}


def domain(chart,index,mode):
    if mode in ('profile','floor'):return chart['parameter_center'],chart['parameter_center']
    # Fixed outer slices maximize displacement from the respective chart center.
    lo,hi=map(Fraction,chart['exact_parameter_endpoints']);width=Fraction(1,32768)
    require(lo+width<=hi,'probe width fits')
    return (str(lo),str(lo+width)) if index==0 else (str(hi-width),str(hi))


def summarize(index,ends,baseline,result):
    loc=result['localization'];last=loc['refinements'][-1]
    phases=result['stability']['phases'];margins=[p['interval_D_positive']['row_margins'] for p in phases if 'interval_D_positive' in p]
    return {'chart':index,'exact_endpoints':list(ends),'baseline_passed':None if baseline is None else baseline['passed'],
        'localized_passed':result['passed'],'refinement_count':len(loc['refinements']),
        'initial_correction':loc['initial_correction'],'final_correction':loc['final_correction'],
        'final_radius_upper':last['max_correction_radius'],'phase_decrease_margins':margins,
        'point_metric_passed':result['stability']['point_passed']}


def execute(out,mode,profile=None,profile_pin=None,floor=None,floor_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    seconds,external,mapcap,unitcap,tickcap=LIMITS[mode]
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0;models=[];stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
        'nice':os.getpriority(os.PRIO_PROCESS,0),**environment(),'internal_seconds':seconds,'external_seconds':external,
        'map_cap':mapcap,'unit_cap':unitcap,'tick_cap':tickcap,'raw_cap':CAP,'cpu_only':True,'paid':False,
        'parent_manifest_sha256':None if mode=='profile' else PARENT_SHA,'precision':256,'refinements':3}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,seconds)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>tickcap or stopped or time.monotonic()-start>=seconds:raise TimeoutError('tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('RSS cap')
        if total(out)>=120*1024**2:raise OSError('ordinary storage cap')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}';meta=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),meta);serial+=1;tick()
    outcome={'mode':mode,'status':'started','passed':False,'probes':[]};code=2
    try:
        if mode!='profile':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        if mode=='probe':
            prior,_,_=completed_bundle(floor,floor_pin,'floor')
            require(len(prior['probes'])==2 and all(v['localized_passed'] for v in prior['probes']),'both localized floors must pass')
            save(out/'floor-gate.json',{'manifest_sha256':floor_pin,'passed':True})
        with ctx.workprec(256):
            if mode=='profile':
                inputs=common.synthetic();model=MeanValueHiddenRateMap(inputs['model_inputs'],max_calls=126);models.append(model)
                charts=[synthetic_parent(model,tick)]
            else:inputs,charts=prepare_parent()
            save(out/'inputs.json',inputs);save(out/'parent-extract.json',{'charts':charts,'parent_manifest_sha256':runtime['parent_manifest_sha256']})
            for index,chart in enumerate(charts):
                if mode!='profile':model=MeanValueHiddenRateMap(inputs['model_inputs'],max_calls=108 if mode=='floor' else 90);models.append(model)
                ends=domain(chart,index,mode);baseline=None
                if mode in ('profile','floor'):
                    blocks,dom=derivative_slice(model,chart,*ends,tick=tick);observe('baseline_domain',dom)
                    baseline=periodic_witness(blocks,tick=tick,observe=observe);observe('baseline_metric',baseline)
                result=localized_stability(model,chart,*ends,tick=tick,observe=observe)
                observe('completed_probe',{'chart':index,'baseline':baseline,'result':result})
                outcome['probes'].append(summarize(index,ends,baseline,result));save(out/'progress.json',outcome)
                del result,baseline;gc.collect()
        passed=all(row['localized_passed'] for row in outcome['probes'])
        outcome.update(passed=passed and mode!='profile',status='synthetic_profile_completed' if mode=='profile' else ('selected_probes_passed' if passed else 'inconclusive'),
            scope='selected exact subintervals only; no new full-coverage or boundary-connection claim',scientific_claim=passed and mode!='profile')
        code=0 if passed else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        counts={key:sum(m.counts()[key] for m in models) for key in ('map_attempted','map_returned','unit_attempted','unit_returned')}
        outcome.update(counts=counts,model_counts=[m.counts() for m in models],elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=now(),exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps({'status':outcome['status'],'exit_code':code,'elapsed_seconds':outcome['elapsed_seconds'],'counts':outcome['counts']}),flush=True)
    return code


def launch(out,mode,profile=None,profile_pin=None,floor=None,floor_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=LIMITS[mode][1];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    require(int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))>3*1024**2,'at least3GiB available before launch')
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
        'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'load_average':os.getloadavg(),
        'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s',sys.executable,str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--floor',floor),('--floor-manifest',floor_pin)]:
        if value is not None:cmd.extend([flag,str(value)])
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=seconds+7)
        except BaseException as exc:meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=[]
    if proc is not None:
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(path.name))
            except (OSError,ValueError):pass
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_live_pids=live)
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);require(total(out)<CAP+1024**2,'worker plus launcher cap')
    save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for path in sorted(out.rglob('*'),reverse=True):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=list(LIMITS),required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--floor',type=Path);p.add_argument('--floor-manifest')
    a=p.parse_args();require((a.mode!='profile')==(a.profile is not None and a.profile_manifest is not None),'target modes require pinned profile')
    require((a.mode=='probe')==(a.floor is not None and a.floor_manifest is not None),'finite probes require pinned floor')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.floor,a.floor_manifest)
if __name__=='__main__':raise SystemExit(main())
