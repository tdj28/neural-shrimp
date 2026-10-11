"""Fold-directed smaller-chart continuation after one saved fresh-domain repair."""
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
from analyze_s2_period18_march_failure import fold_alignments
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,check_orbit
from neural_shrimp.rigorous_connection import HiddenRateMap,require
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_arclength_rebox import stream_reboxed
from neural_shrimp.rigorous_folds import FoldTanhMap
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
import certify_s2_period18_arclength_reboxing as historical
import certify_s2_period18_fold_saved_rebox as anchor_driver
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import fraction_mid
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json,matrix_json
from flint import arb,arb_mat
CARD='docs/plans/s2-period18-fold-small-march.md'
SOURCES=list(dict.fromkeys(anchor_driver.SOURCES+historical.SOURCES+['tools/certify_s2_period18_fold_small_march.py','tests/test_s2_period18_fold_small_march.py',CARD]))
SETTINGS=dict(precision=256,halfwidth_exponent=-8,radius_exponent=-21,newton_steps=2)
CAP=384*1024**2
INITIAL=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-saved-rebox-2026-10-10/attempt-001-target/run')
INITIAL_PIN='68abff9e1ccdb5c4b8170acd174c4e5e8f27e585698ea1c5436281d89bc56d2d'
INITIAL_EXECUTION='bf688606f5afb5daf6bdfb20a02fd458c1ee7d94'
CURRENT='tools/certify_s2_period18_fold_small_march.py'
CountedJetMap=historical.CountedJetMap
known_fold=historical.known_fold

def entry(path):
    b=path.read_bytes();return {'sha256':sha256(b),'bytes':len(b)}
def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as f: data=f.read(256*1024**2+1)
        require(len(data)<=256*1024**2,'decompressed artifact cap')
        if path.name.endswith('.json.gz'):
            meta=json.loads(path.with_name(path.name.removesuffix('.json.gz')+'-payload.json').read_bytes())
            require(len(data)==meta['uncompressed_bytes'] and sha256(data)==meta['uncompressed_sha256'],'decompressed payload identity')
        return json.loads(data)
    return json.loads(path.read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def total(out):return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
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
def inventory(out):return {str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
def verify_inventory(out,members):
    require(isinstance(members,dict) and bool(members),'inventory required')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    require(actual==set(members),'exact inventory')
    for name,e in members.items():
        p=out/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'safe member')
        require(entry(p)==e,'file identity '+name)
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        result[name]=sha256(b)
    return result

def qualified(chart):
    require(chart['passed'] and all(chart['gates'].values()) and all(chart['coordinate_inclusions']) and chart['anchor_check']['passed'],'all strict chart gates')
    require(chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'native18x13 chart')
    require([r['divisor'] for r in chart['divisors'] if r['excluded']]==[1,2,3,6,9],'primitive18 exclusions')
    st=chart['settings'];require(st['precision']==256 and st['halfwidth_exponent']==-8 and st['newton_steps']==2 and -100<=st['radius_exponent']<=-10,'fixed halfwidth/precision and bounded adapted radii')
    ww=[exact(v) for v in chart['exact_weights']];require(len(ww)==235 and all(v.is_exact() and v.is_finite() and 0<v<=1 for v in ww),'positive exact weights')


def joined(previous,chart):
    qualified(chart)
    require(chart['normal']==previous['normal'] and chart['origin']==previous['origin'],'same exact gauge')
    require(chart['anchor']['state_box']==previous['endpoint_root_enclosures'][1] and chart['anchor']['exact_parameter']==previous['exact_parameter_endpoints'][1]==chart['exact_parameter_endpoints'][0],'complete native exact shared endpoint')


def bound_sources(parent,rt,execution):
    require(rt['source_commit']==execution and rt['mode']=='target','exact target execution')
    for name,digest in rt['source_sha256'].items():
        require(entry(parent/'source'/name)['sha256']==digest and sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT))==digest,'bound parent source '+name)
    for name in anchor_driver.SOURCES:require(rt['source_sha256'].get(name)==entry(ROOT/name)['sha256'],'unchanged reviewed anchor/source '+name)


def initial_anchor():
    require(entry(INITIAL/'manifest.json')['sha256']==INITIAL_PIN,'fixed repaired manifest');verify_inventory(INITIAL,read(INITIAL/'manifest.json')['files'])
    rt=read(INITIAL/'runtime-start.json');bound_sources(INITIAL,rt,INITIAL_EXECUTION)
    outcome=read(INITIAL/'outcome.json');inp=read(INITIAL/'inputs.json.gz');chart=read(INITIAL/'certificate.json.gz')
    require(outcome['passed'] and outcome['exit_code']==0 and outcome['completed_reboxes']==1,'completed one-shot repaired chart');qualified(chart)
    original=historical.original_anchor();require(inp['model_inputs']==original['model_inputs'] and inp['provenance']['original_anchor']==original['provenance'],'original exact model/data/anchor')
    failed=inp['chart'];require(not failed['passed'] and failed['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'saved root-only failure')
    require(chart['saved_rebox_provenance']==inp['provenance'] and inp['provenance']['manifest']==anchor_driver.PIN and inp['provenance']['chart_index']==1,'exact retained witness identity')
    for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):
        require(chart[key]==failed[key],'unchanged repaired predictor '+key)
    require(chart['reboxing']['proposal']==read(INITIAL/'radii-proposal.json'),'exact repaired proposal')
    require(chart['exact_parameter_endpoints']==['17/64','35/128'],'fixed repaired coverage')
    require(chart['anchor']['exact_parameter']=='17/64' and chart['anchor_check']['passed'],'qualified exact native start')
    require(not any(a['contained_in_qualified_chart'] for a in chart['known_fold_membership']['all_cyclic_alignments']),'not already joined to fold')
    return inp,compact_chart(chart)


def prepare(parent,pin,execution):
    import re
    require(parent is not None and re.fullmatch('[0-9a-f]{64}',pin or '') and re.fullmatch('[0-9a-f]{40}',execution or ''),'explicit exact predecessor pins')
    initial,first_endpoint=initial_anchor()
    if parent==INITIAL:
        require(pin==INITIAL_PIN and execution==INITIAL_EXECUTION,'fixed initial predecessor identity')
        return {'model_inputs':initial['model_inputs'],'previous':first_endpoint,'provenance':{'parent_root':str(parent),'parent_manifest':pin,'parent_execution':execution,'original_anchor':initial['provenance']['original_anchor'],'initial_repaired_manifest':INITIAL_PIN}}
    require(entry(parent/'manifest.json')['sha256']==pin,'exact later march manifest');verify_inventory(parent,read(parent/'manifest.json')['files'])
    rt=read(parent/'runtime-start.json');bound_sources(parent,rt,execution)
    require(rt.get('executed_runner')==CURRENT and rt['source_sha256'].get(CURRENT)==entry(ROOT/CURRENT)['sha256'],'unchanged actual fold-directed runner')
    for name in ('src/neural_shrimp/rigorous_arclength_rebox.py','src/neural_shrimp/rigorous_reboxing_gradient.py'):
        require(rt['source_sha256'].get(name)==entry(ROOT/name)['sha256'],'same reboxing kernel')
    inp=read(parent/'inputs.json');batch=read(parent/'batch.json');out=read(parent/'outcome.json')
    require(out['passed'] and out['exit_code']==0 and batch['passed'] and not batch['known_fold_identified'],'only complete not-yet-fold predecessor')
    require(batch['completed_charts']==batch['qualified_charts']==batch['requested_charts'] and 1<=batch['requested_charts']<=16,'complete bounded unit')
    require(inp['model_inputs']==initial['model_inputs'] and inp['provenance']['initial_repaired_manifest']==INITIAL_PIN,'same exact repaired native lineage')
    previous=inp['previous'];require(previous['passed'] and all(previous['gates'].values()) and previous['normal']==first_endpoint['normal'] and previous['origin']==first_endpoint['origin'],'inherited qualified new gauge')
    for index,row in enumerate(batch['charts']):
        require(row['index']==index and row['artifact']['path']==f'chart-{index:03d}.json.gz','ordered exact chart inventory')
        name=row['artifact']['path'];require(entry(parent/name)=={k:row['artifact'][k] for k in ('sha256','bytes')},'bound final chart')
        c=read(parent/name);require(c['passed']==row['passed'] and c['gates']==row['gates'],'mirrored chart gates');joined(previous,c)
        if 'reboxing' in c:
            identity=c['reboxing']['initial_artifact'];path=f'initial-{index:03d}.json.gz'
            require(identity['path']==path and entry(parent/path)=={k:identity[k] for k in ('sha256','bytes')},'retained initial witness')
            trial=read(parent/path);require(trial['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'only root-bound failure reboxed')
            for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):
                require(c[key]==trial[key],'unchanged predictor/Taylor/anchor '+key)
        previous=compact_chart(c)
    return {'model_inputs':inp['model_inputs'],'previous':previous,'provenance':{'parent_root':str(parent),'parent_manifest':pin,'parent_execution':execution,'original_anchor':initial['provenance']['original_anchor'],'initial_repaired_manifest':INITIAL_PIN}}


def plan_count(previous,fold):
    normal=[exact(v) for v in previous['normal']];origin=[exact(v) for v in previous['origin']]
    box=[restore_ball(v) for row in fold['orbit'] for v in row]+[restore_ball(fold['rate'])]
    require(len(box)==len(normal)==len(origin)==235,'whole native0 fold box')
    projection=sum((normal[i]*(box[i]-origin[i]) for i in range(235)),arb(0))
    delta=Fraction(2)**previous['settings']['halfwidth_exponent'];end=Fraction(previous['exact_parameter_endpoints'][1])
    target=fraction_mid(projection.upper())+delta;remaining=target-end
    require(remaining>0,'native0 fold neighborhood already covered; stop for diagnosis')
    ratio=remaining/(2*delta);needed=(ratio.numerator+ratio.denominator-1)//ratio.denominator
    return {'requested_charts':min(16,needed),'nominal_remaining_charts':needed,'whole_native0_fold_gauge':ball_json(projection),
      'target_right_gauge':str(target),'current_right_gauge':str(end),
      'scope':'deterministic workload target from full native0 fold projection; every chart still checks all18 alignments and strict membership; no proximity or monotonicity claim'}


def profile_input():
    old=historical.synthetic();old['model_inputs']['lr_hidden']=old['model_inputs']['lr_output']='1/4';center=[arb(0)]*234+[exact('1/4')];delta=arb((1,-8))
    previous={'passed':False,'gates':{'uniform_root':True,'primitive_period':False,'anchor_identified':True},'statement':{'period':18,'phase_dimension':13},'settings':SETTINGS,
      'normal':['0']*234+['1'],'origin':['0']*234+['1/4'],'center':vector_json(center),'parameter_center':'0','exact_parameter_endpoints':['-1/256','1/256'],
      'predictor_slope':matrix_json(arb_mat([[int(i==234)] for i in range(235)])),'predictor_curvature':matrix_json(arb_mat(235,1)),
      'exact_weights':historical.prepare_weights()+['1'],'endpoint_root_enclosures':[vector_json(center[:-1]+[center[-1]+sign*delta]) for sign in (-1,1)]}
    return {'model_inputs':old['model_inputs'],'previous':previous,'provenance':old['provenance']}


def profile_environment(runtime,result):
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite nonnegative bounded profile time')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'finite nonnegative bounded profile RSS')
    require(result['exit_code']==0 and runtime['mode']=='profile','successful profile worker')
    expected={'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'internal_seconds':20,'tick_cap':400,'map_cap':180,'unit_derivative_cap':360,'jet_cap':108}
    require(all(runtime.get(k)==v for k,v in expected.items()),'profile library/settings equality')
    require(runtime['threads']==dict.fromkeys(THREADS,'1') and runtime['nice']>=10,'profile threads/nice')


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'approved profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    result=read(path/'outcome.json');runtime=read(path/'runtime-start.json')
    profile_environment(runtime,result)
    charts=read(path/'batch.json')['charts']
    require(len(charts)==2 and all(c['gates']['uniform_root'] and c['reboxed'] for c in charts),'profile covers both initial and reboxed root stages')
    require(result['adapter_counts']=={'map_attempted':180,'map_returned':180,'unit_attempted':360,'unit_returned':360,'map_cap':180,'unit_cap':360},'complete C1 derivative profile'
    )
    require(result['jet_counts']=={'attempted':108,'returned':108,'cap':108},'complete path-jet profile')
    require(result['status']=='synthetic_profile_completed' and result['elapsed_seconds']<20,'completed profile timing')
    require(runtime['source_sha256']==sources(),'same profiled source bytes')
    require(16*result['elapsed_seconds']+5<180,'conservative profile projection below180 seconds')
    require(max(total(path)*16,16*20*1024**2)<352*1024**2,'storage estimate includes real parent compression cost')
    return {'manifest_sha256':pin,'elapsed_seconds':result['elapsed_seconds'],
            'projected_seconds':16*result['elapsed_seconds']+5,'profile_bytes':total(path),'projected_bytes':max(total(path)*16,16*20*1024**2),
            'scope':'conservative engineering gate, not a proof or exact runtime prediction'}

def execute(out,mode,profile=None,profile_pin=None,parent=None,parent_pin=None,parent_execution=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    limits=20 if mode=='profile' else 180
    run_count=2 if mode=='profile' else 16;map_cap=90*run_count;jet_cap=54*run_count;tick_cap=400 if mode=='profile' else 3200
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0;model=None;jet_model=None
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':bound,'executed_runner':str(Path(__file__).resolve().relative_to(ROOT)),'argv':sys.argv,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
      'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
      'predecessor':{'root':str(parent) if parent is not None else None,'manifest_sha256':parent_pin,'execution':parent_execution},
      'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'numpy':numpy.__version__,'scipy':scipy.__version__,
      'internal_seconds':limits,'map_cap':map_cap,'unit_derivative_cap':2*map_cap,'jet_cap':jet_cap,'tick_cap':tick_cap,'raw_cap':CAP,'cpu_only':True,'paid':False}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limits)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>tick_cap or stopped or time.monotonic()-start>limits:raise TimeoutError('tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('RSS cap')
        if total(out)>352*1024**2:raise OSError('ordinary raw allowance')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        if kind=='chart_completed':
            payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload)
        else:save(out/(name+'.json'),row)
        serial+=1;tick()
    outcome={'passed':False,'mode':mode,'status':'started'};code=2
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else prepare(parent,parent_pin,parent_execution)
        save(out/'inputs.json',inp)
        fold=known_fold(inp['model_inputs']) if mode=='target' else None
        if fold is not None:
            save(out/'fold-input-identity.json',fold['identity'])
            with ctx.workprec(256):planned=plan_count(inp['previous'],fold)
            run_count=planned['requested_charts'];save(out/'fold-directed-plan.json',planned)
        def emit_chart(index,row):
            if fold is not None:
                row['known_fold_membership']={'identity':fold['identity'],'all_cyclic_alignments':fold_alignments(row,fold['orbit'],fold['rate']),
                  'scope':'full correlated fold-root containment in this qualified chart; no attraction or whole geometry'}
            name=f'chart-{index:03d}.json.gz';payload=save(out/name,row)
            save(out/f'chart-{index:03d}-payload.json',payload)
            identity={'path':name,**entry(out/name),'payload':payload}
            outcome['last_durable_chart']={'index':index,'passed':row['passed'],'identity':identity}
            return identity
        def emit_initial(index,row):
            name=f'initial-{index:03d}.json.gz';payload=save(out/name,row);save(out/f'initial-{index:03d}-payload.json',payload)
            identity={'path':name,**entry(out/name),'payload':payload}
            outcome['last_durable_initial']={'index':index,'identity':identity};return identity
        with ctx.workprec(SETTINGS['precision']):
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=map_cap);jet_model=CountedJetMap(inp['model_inputs'],jet_cap)
            result=stream_reboxed(model,jet_model,inp['previous'],count=run_count,synthetic_profile=mode=='profile',force_profile_rebox=mode=='profile',tick=tick,observe=observe,emit_initial=emit_initial,emit_chart=emit_chart)
        save(out/'batch.json',result)
        profile_complete=mode=='profile'
        outcome.update(passed=bool(result['passed']) and not profile_complete,
            status='synthetic_profile_completed' if profile_complete else ('joined_intervals_certified' if result['passed'] else 'inconclusive'),
            gates={'connected_primitive_branch':result['passed'],'requested_charts':run_count,'completed_charts':len(result['charts']),'attraction_evaluated':False,'known_fold_identified':result.get('known_fold_identified',False)},scientific_claim=result['passed'] and not profile_complete)
        code=0 if profile_complete or result['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        if jet_model is not None:outcome['jet_counts']=jet_model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out)
        save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code


def launch(out,mode,profile=None,profile_pin=None,parent=None,parent_pin=None,parent_execution=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 195
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--parent',parent),('--parent-manifest',parent_pin),('--parent-execution',parent_execution)]:
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
    p.add_argument('--parent',type=Path);p.add_argument('--parent-manifest');p.add_argument('--parent-execution')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires fixed profile pin')
    require((a.mode=='target')==(a.parent is not None and a.parent_manifest is not None and a.parent_execution is not None),'target requires exact separate predecessor pins')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.parent,a.parent_manifest,a.parent_execution)
if __name__=='__main__':raise SystemExit(main())
