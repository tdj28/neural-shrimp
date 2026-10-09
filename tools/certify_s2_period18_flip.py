"""Single bounded exact-log-ray flip pilot, synthetic resource profile, or replay.

The target mode requires the parent's explicit post-kernel-review launch decision.
All modes retain an exclusive source/input-bound bundle, including failed attempts.
"""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS: os.environ[name]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import json,math,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_flips import FlipTanhMap,augmented,check_flip
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,dyadic,exact,vector_json,matrix_json
from flint import arb,ctx
DATA=Path('/data2/neural-shrimp-data')
PARENT=DATA/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
CERTS=DATA/'raw/EXP-028/attempt-002/stage3/certificates.json'
RAYS=DATA/'explore/a2-survey/rays.json'
PINS={str(PARENT):'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',
      str(CERTS):'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8',
      str(RAYS):'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4'}
CARD='docs/plans/s2-period18-flip-certificate.md'
SOURCES=['src/neural_shrimp/rigorous_flips.py','src/neural_shrimp/rigorous_orbits.py',
         'src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
         'tools/certify_s2_period18_flip.py','tests/test_rigorous_flips.py',CARD]
NUMERIC_SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/numerical_flip.py','tools/explore_s2_period18_flip.py','tests/test_numerical_flip.py','docs/plans/s2-period18-flip-pilot.md']
SETTINGS=dict(precision=256,radius_exponent=-100,newton_steps=5,max_power=4096)
CAP=16*1024**2
INTERNAL_SECONDS=120
EXTERNAL_SECONDS=135

def require(value,why):
    if not value: raise ValueError(why)
def entry(path):
    b=path.read_bytes();return {'sha256':sha256(b),'bytes':len(b)}
def read(path):return json.loads(path.read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def total(out):return sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
def save(path,obj,cap=CAP):
    b=canonical_bytes(obj)
    require(len(b)<cap,'single artifact limit')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def inventory(out):return {str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}
def verify_inventory(out,manifest):
    require(isinstance(manifest,dict) and bool(manifest),'empty inventory')
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    require(actual==set(manifest),'complete inventory')
    for name,e in manifest.items():
        p=out/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'unsafe artifact name')
        require(entry(p)==e,'artifact identity '+name)
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    out={}
    for p in SOURCES:
        b=(ROOT/p).read_bytes()
        require(b==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'uncommitted source '+p)
        out[p]=sha256(b)
    return out

def assemble(consumed,row,window,numeric_inputs,point,seed):
    """Pure identity reconstruction, separately exercised by synthetic controls."""
    require((row['map_id'],row['component'],row['period'])==('S2-28011-zoom',587,18),'fixed row111')
    require([i for i,r in enumerate(window['rays']) if r['type']=='flip'][0]==13,'lowest-index historical flip ray13')
    ray=window['rays'][13]
    require(numeric_inputs['row_index']==111 and numeric_inputs['ray_index']==13 and numeric_inputs['period']==18,'numeric target selection')
    require(numeric_inputs['ray']==ray and numeric_inputs['historical_orbit']==row['certificate']['orbit'],'historical native orbit/ray binding')
    model={k:consumed[k] for k in ('schema','network','x','y')}
    require(numeric_inputs['consumed_map']==seed['consumed_map']==model,'same consumed exact model/data')
    require(seed['schema']=='neural-shrimp-flip-proof-seed-v1' and seed['period']==18 and point['qualified'] is True,'qualified numerical proposal')
    ell=[float(x) for x in window['center']]
    direction=[float(a)*float(b) for a,b in zip(window['scale'],ray['direction'])]
    require(len(ell)==len(direction)==2 and all(math.isfinite(x) for x in ell+direction) and all(x!=0 for x in direction),'finite both-rate direction')
    family={'schema':'neural-shrimp-exact-log-ray-v1','ell':[dyadic(x) for x in ell],'direction':[dyadic(x) for x in direction]}
    require(seed['family']==numeric_inputs['family'],'same numerical/seed family record')
    sf=seed['family'];require(sf['schema']==family['schema'],'family schema')
    require(sf['historical_scale']==window['scale'] and sf['historical_unit_direction']==ray['direction'],'original scale/unit direction binding')
    for name in ('ell','direction'):
        require([Fraction(x) for x in sf[name]]==[Fraction(x) for x in family[name]],'exact historical family '+name)
        require([Fraction(float(x)) for x in sf[name+'_float']]==[Fraction(x) for x in family[name]],'numerical historical family '+name)
    require(len(seed['theta'])==len(seed['right_vector'])==len(point['state'])==len(point['q'])==13,'complete13D seed')
    require(0<=float(point['s'])<=1 and math.isfinite(float(point['s'])),'bounded historical ray proposal')
    for key,values in (('theta',point['state']),('right_vector',point['q'])):
        require([Fraction(x) for x in seed[key]]==[Fraction(float(x)) for x in values],'exact numerical proposal '+key)
    require(Fraction(seed['parameter'])==Fraction(float(point['s'])),'exact numerical proposal s')
    # Base-class rate fields are explicitly unused; both actual rates come from family.
    model.update(lr_hidden='1/1',lr_output='1/1')
    return {'schema':'neural-shrimp-flip-input-v1','period':18,'map_inputs':model,
            'rate_family':family,'approximation':seed['theta']+[seed['parameter']]+seed['right_vector']}


def prepare(numerical,numerical_manifest):
    require(numerical is not None and numerical_manifest is not None,'explicit numerical bundle pin required')
    require(numerical.is_dir() and not numerical.is_symlink(),'safe numerical bundle')
    require(entry(numerical/'manifest.json')['sha256']==numerical_manifest,'externally pinned numerical manifest')
    require(not any(p.is_symlink() for p in numerical.rglob('*')),'no numerical artifact symlinks')
    verify_inventory(numerical,read(numerical/'manifest.json')['files'])
    raw=numerical/'run';verify_inventory(raw,read(raw/'manifest.json')['files'])
    for name,h in PINS.items():require(entry(Path(name))['sha256']==h,'pinned historical input '+name)
    runtime=read(raw/'runtime-start.json');launcher=read(numerical/'launcher-result.json')
    require(runtime['source_commit']==launcher['source_commit'],'numeric launcher/source join')
    require(launcher['owned_live_pids_remaining']==[],'numeric producer cleanup')
    require(set(runtime['source_sha256'])==set(NUMERIC_SOURCES),'complete numerical source inventory')
    for name,h in runtime['source_sha256'].items():
        require(sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+name],cwd=ROOT))==h,'numeric committed source '+name)
        require(entry(raw/'source'/name)['sha256']==h,'numeric retained source '+name)
    num=read(raw/'inputs.json');point=read(raw/'flip.json');seed=read(raw/'flip-proof-seed.json')
    require(seed['source_commit']==runtime['source_commit'] and seed['inputs_sha256']==entry(raw/'inputs.json')['sha256'],'numerical seed provenance')
    require(num['input_pins']==PINS,'numeric historical pins')
    row=read(CERTS)[111];window=read(RAYS)['S2-28011-zoom:587']
    result=assemble(read(PARENT),row,window,num,point,seed)
    result['provenance']={'historical_input_pins':PINS,'numeric_root':str(numerical),
        'numeric_manifest_sha256':numerical_manifest,'numeric_run_manifest_sha256':entry(raw/'manifest.json')['sha256'],
        'numeric_seed_sha256':entry(raw/'flip-proof-seed.json')['sha256'],'numeric_execution_commit':runtime['source_commit'],
        'selection':'row111 S2-28011-zoom:587, lowest-index historical flip ray13',
        'family_convention':'eta_i(s)=exp(log(10)*(ell_i+d_i*s)); ell_i and rounded scale_i*direction_i interpreted as exact dyadics. Both rates vary.',
        'center_scope':'Exact exponential rates at s=0 are not equated with the earlier dyadic center-certificate rates; no connecting tube claimed.'}
    return result


def synthetic():
    return {'schema':'neural-shrimp-flip-input-v1','period':18,
        'map_inputs':{'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},
            'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],
            'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)],'lr_hidden':'1/1','lr_output':'1/1'},
        'rate_family':{'schema':'neural-shrimp-exact-log-ray-v1','ell':['-2','-2'],'direction':['1/20','3/100']},
        'approximation':[str(Fraction(i-6,64)) for i in range(13)]+['1/7']+[str(Fraction(i+1,64)) for i in range(13)],
        'provenance':{'synthetic':True,'purpose':'two18-step third-directional evaluations; resource-only, no flip claim'}}


def profile_gate(profile,pin):
    require(profile is not None and pin is not None,'pinned measured profile required')
    require(entry(profile/'manifest.json')['sha256']==pin,'external profile manifest pin')
    verify_inventory(profile,read(profile/'manifest.json')['files']);verify_inventory(profile/'run',read(profile/'run/manifest.json')['files'])
    runtime=read(profile/'run/runtime-start.json');outcome=read(profile/'run/outcome.json');launcher=read(profile/'launcher-result.json')
    require(runtime['mode']==launcher['mode']=='profile' and outcome['status']=='synthetic_profile_complete' and outcome['scientific_claim'] is False and outcome['passed'] is True,'completed resource-only profile')
    expected=sources();require(runtime['source_sha256']==launcher['source_sha256']==expected,'unchanged bound profile source')
    require(runtime['source_commit']==launcher['source_commit'],'profile execution commit join')
    for name,h in expected.items():
        require(sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+name],cwd=ROOT))==h,'profile historical source')
        require(entry(profile/'run/source'/name)['sha256']==h,'profile retained source')
    for directory,cap in ((profile/'run',CAP),(profile,CAP+1024**2)):
        manifest=read(directory/'manifest.json');require(manifest['total_bytes']==sum(v['bytes'] for v in manifest['files'].values()) and manifest['within_limit'] is True and total(directory)<cap,'profile artifact totals/cap')
    require(runtime['python_flint']==flint.__version__ and runtime['flint']==flint.__FLINT_VERSION__,'same profile arithmetic library')
    require(launcher['exit_code']==0 and launcher['owned_leader_exited'] and launcher['owned_live_pids_remaining']==[],'profile clean exit')
    return profile_resource_gates(runtime,outcome,launcher,profile,pin)


def profile_resource_gates(runtime,outcome,launcher,profile,pin):
    rows=outcome['timings'];require(len(rows)==2 and [r['evaluation'] for r in rows]==['center','box'],'two fixed profile evaluations')
    times=[r['seconds'] for r in rows]
    require(all(type(t)in(int,float) and math.isfinite(t) and 0<t<=INTERNAL_SECONDS for t in times),'finite positive measured times')
    projected=7*max(times)+10
    require(outcome['projected_seven_evaluation_seconds_plus10']==projected and projected<=60,'measured conservative cost gate')
    elapsed=outcome['elapsed_seconds'];rss=outcome['max_rss_kib']
    require(type(elapsed)in(int,float) and math.isfinite(elapsed) and max(times)<=elapsed<=INTERNAL_SECONDS,'profile elapsed cap')
    require(type(rss)is int and 0<rss<=2*1024**2,'profile RSS cap')
    require(type(launcher['elapsed_seconds'])in(int,float) and math.isfinite(launcher['elapsed_seconds']) and elapsed<=launcher['elapsed_seconds']<=EXTERNAL_SECONDS+7,'profile external cap')
    require(runtime['threads']=={k:'1' for k in THREADS} and runtime['nice']>=10,'profile CPU1/nice contract')
    return {'profile_root':str(profile),'profile_manifest_sha256':pin,'projected_seconds':projected,'gate_seconds':60,'passed':True}


def whole_box_proximity(result,inp):
    def pair(x):return Fraction(x[0])*Fraction(2)**x[1]
    seed=[Fraction(x) for x in inp['approximation']]
    require(len(result['box'])==len(seed)==27,'full selected box dimension')
    distances=[];den=max([Fraction(1)]+[abs(x) for x in seed])
    for ball,x in zip(result['box'],seed):
        require('mid' in ball and 'rad' in ball,'finite whole-box proximity')
        mid,rad=pair(ball['mid']),pair(ball['rad']);require(rad>=0,'nonnegative radius')
        distances.append(max(abs(mid-rad-x),abs(mid+rad-x))/den)
    d=max(distances)
    return {'passed':bool(result['gates']['augmented_root_unique'] and d<=Fraction(1,10**7)),
            'normalized_max_distance':str(d),'display_only':float(d),'cap':'1/10000000',
            'scope':'Entire normalized27D root box near the saved numerical flip proposal; not seed inclusion, center-orbit connection or branch ancestry.'}


def replay(bundle,expected,observe,tick):
    require(entry(bundle/'manifest.json')['sha256']==expected,'pinned replay manifest')
    verify_inventory(bundle,read(bundle/'manifest.json')['files'])
    envelope=read(bundle/'certificate.json');inp=read(bundle/'inputs.json')
    require(envelope['schema']=='neural-shrimp-flip-certificate-v1','certificate schema')
    require(envelope['inputs_sha256']==entry(bundle/'inputs.json')['sha256'],'input binding')
    p=inp['provenance'];expected_inputs=prepare(Path(p['numeric_root']),p['numeric_manifest_sha256'])
    require(canonical_bytes(inp)==canonical_bytes(expected_inputs),'exact original target reconstruction')
    require(envelope['runtime']['source_sha256']=={p:sha256((ROOT/p).read_bytes()) for p in SOURCES},'replay source equality')
    require(envelope['runtime']['python_flint']==flint.__version__ and envelope['runtime']['flint']==flint.__FLINT_VERSION__,'replay arithmetic library versions')
    result=check_flip(inp,**envelope['result']['settings'],tick=tick,observe=observe)
    require(canonical_bytes(result)==canonical_bytes(envelope['result']),'byte-identical mathematical replay')
    association=whole_box_proximity(result,inp)
    require(association==envelope['whole_box_proximity'],'exact association replay')
    return inp,{'schema':'neural-shrimp-flip-replay-v1','passed':bool(result['passed'] and association['passed']),'whole_box_proximity':association,'identical':True,'bundle':str(bundle),'manifest_sha256':expected,'result':result}


def execute(out,mode,bundle=None,manifest=None,numerical=None,numeric_pin=None,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();count=0
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
        'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
        'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'internal_seconds':INTERNAL_SECONDS,'tick_cap':160,'raw_cap':CAP,'cpu_only':True,'paid':False,
        'numeric_root':None if numerical is None else str(numerical),'numeric_manifest_sha256':numeric_pin,
        'profile_root':None if profile is None else str(profile),'profile_manifest_sha256':profile_pin}
    save(out/'runtime-start.json',runtime)
    for p in SOURCES:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL_SECONDS)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal count
        count+=1
        if count>160 or stopped or time.monotonic()-start>INTERNAL_SECONDS:raise TimeoutError('flip tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS above2GiB')
        if total(out)>14*1024**2:raise OSError('ordinary output budget')
    serial=0
    def observe(kind,record):
        nonlocal serial
        save(out/f'evaluation-{serial:03d}-{kind}.json',record);serial+=1;tick()
    outcome={'passed':False,'mode':mode,'status':'started'};code=2
    try:
        if mode=='replay':
            inp,result=replay(bundle,manifest,observe,tick);save(out/'inputs.json',inp);save(out/'replay.json',result)
            outcome.update(passed=result['passed'],status='replayed');code=0 if result['passed'] else 2
        else:
            if mode=='target':save(out/'profile-eligibility.json',profile_gate(profile,profile_pin))
            inp=synthetic() if mode=='profile' else prepare(numerical,numeric_pin);save(out/'inputs.json',inp)
            if mode=='profile':
                with ctx.workprec(SETTINGS['precision']):
                    model=FlipTanhMap(inp['map_inputs'],inp['rate_family']);center=[exact(x) for x in inp['approximation']];rows=[]
                    for tag,z in [('center',center),('box',[arb(x,arb((1,SETTINGS['radius_exponent']))) for x in center])]:
                        then=time.monotonic();h,j,data=augmented(model,z,18,tick)
                        require(all(x.is_finite() for x in h+data['third_directional']) and all(j[i,k].is_finite() for i in range(j.nrows()) for k in range(j.ncols())),'finite synthetic profile return')
                        observe('profile-'+tag,{'h':vector_json(h),'jacobian':matrix_json(j),'third_directional':vector_json(data['third_directional'])})
                        rows.append({'evaluation':tag,'seconds':time.monotonic()-then})
                outcome.update(passed=True,status='synthetic_profile_complete',timings=rows,
                    projected_seven_evaluation_seconds_plus10=7*max(r['seconds'] for r in rows)+10,scientific_claim=False);code=0
            else:
                result=check_flip(inp,**SETTINGS,tick=tick,observe=observe)
                association=whole_box_proximity(result,inp)
                save(out/'certificate.json',{'schema':'neural-shrimp-flip-certificate-v1','inputs_sha256':entry(out/'inputs.json')['sha256'],'runtime':runtime,'result':result,'whole_box_proximity':association})
                outcome.update(passed=bool(result['passed'] and association['passed']),mathematical_proof_passed=result['passed'],association_passed=association['passed'],status=result['status'],gates=result['gates'],criticality=result['criticality'],doubled_side_in_s=result['doubled_side_in_s']);code=0 if outcome['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,ticks=count,evaluations_retained=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code


def launch(out,mode,bundle=None,manifest=None,numerical=None,numeric_pin=None,profile=None,profile_pin=None):
    bound=sources();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL_SECONDS,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL_SECONDS}s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in (('--bundle',bundle),('--manifest-sha256',manifest),('--numerical-root',numerical),('--numerical-manifest-sha256',numeric_pin),('--profile-root',profile),('--profile-manifest-sha256',profile_pin)):
        if value is not None:cmd+=[flag,str(value)]
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=EXTERNAL_SECONDS+7)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc);code=2
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
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':remaining.append(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_leader_exited=proc is not None and proc.poll() is not None,owned_live_pids_remaining=remaining)
    if total(out)>CAP+1024**2 or remaining:code=2;meta['exit_code']=2;meta['failure']='resource/inventory qualification failed'
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([q for q in out.rglob('*') if q.is_dir()],reverse=True):p.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True);return code


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target','replay'],required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--bundle',type=Path);p.add_argument('--manifest-sha256');p.add_argument('--numerical-root',type=Path);p.add_argument('--numerical-manifest-sha256');p.add_argument('--profile-root',type=Path);p.add_argument('--profile-manifest-sha256');a=p.parse_args()
    require((a.mode=='replay')==(a.bundle is not None and a.manifest_sha256 is not None),'replay bundle and pin')
    require(a.mode!='target' or all(x is not None for x in (a.numerical_root,a.numerical_manifest_sha256,a.profile_root,a.profile_manifest_sha256)),'target requires numerical and profile bundles/pins')
    return (execute if a.execute else launch)(a.out,a.mode,a.bundle,a.manifest_sha256,a.numerical_root,a.numerical_manifest_sha256,a.profile_root,a.profile_manifest_sha256)
if __name__=='__main__':raise SystemExit(main())
