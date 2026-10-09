"""Single bounded augmented-fold pilot, synthetic resource profile, or replay.

The target mode requires the parent's explicit post-kernel-review launch decision.
All modes retain an exclusive source/input-bound bundle, including failed attempts.
"""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS: os.environ[name]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import json,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_folds import FoldTanhMap,augmented,check_fold
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,dyadic,exact,vector_json,matrix_json
from flint import arb,ctx
DATA=Path('/data2/neural-shrimp-data')
PARENT=DATA/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
NUMERIC=DATA/'explore/s2-period18-fold/attempt-001'
PINS={str(PARENT):'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',
      str(NUMERIC/'manifest.json'):'3a44977630bd3d9f329bad8e6369c4d024fc4a379ee6ebe593c85d9b913cb620',
      str(NUMERIC/'outcome.json'):'ddd90afa59622d1bc3da59bfc1eb424afcb16570abbcf8dee8f3eefeadd4ec13'}
CARD='docs/plans/s2-period18-fold-certificate.md'
SOURCES=['src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py',
         'src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
         'tools/certify_s2_period18_fold.py','tests/test_rigorous_folds.py',CARD]
SETTINGS=dict(precision=256,radius_exponent=-80,newton_steps=5,max_power=4096)
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

def prepare():
    for name,h in PINS.items():require(entry(Path(name))['sha256']==h,'pinned input '+name)
    verify_inventory(NUMERIC,read(NUMERIC/'manifest.json'))
    old=read(PARENT);numerical=read(NUMERIC/'outcome.json');native=read(NUMERIC/'inputs.json')
    require(numerical['accepted'] and numerical['period']==18 and numerical['target']=='S2-28011-zoom:587' and numerical['ray_index']==0,'fixed numerical fold')
    require(native['network']==old['network'] and native['row_index']==111 and native['ray_index']==0,'same exact model/selection')
    hlog,olog=numerical['parameter_log10']
    # Match the scalar Python power convention used by the numerical producer.
    rh,ro=10.0**float(hlog),10.0**float(olog)
    require(ro.hex()=='0x1.a461eed55764cp-1','exact historical output-rate identity')
    theta=numerical['diagnostics']['state'];v=numerical['diagnostics']['right_null']
    require(len(theta)==len(v)==13,'full augmented seed')
    model={k:old[k] for k in ('schema','network','x','y')}
    model.update(lr_hidden=dyadic(rh),lr_output=dyadic(ro))
    return {'schema':'neural-shrimp-fold-input-v1','period':18,'map_inputs':model,
            'approximation':[dyadic(x) for x in theta]+[dyadic(rh)]+[dyadic(x) for x in v],
            'provenance':{'input_pins':PINS,'numeric_source_commit':read(NUMERIC/'runtime-start.json')['source_commit'],
                'numeric_parameter_log10':[hlog,olog],'hidden_rate_seed_hex':rh.hex(),'output_rate_hex':ro.hex(),
                'output_rate_matches_row111':ro.hex()=='0x1.a461eed55764cp-1',
                'parameter_semantics':'Hidden rate is an exact-real variable; scalar binary64 powers only create the proposal and fixed output constant. No log derivative enters proof.',
                'selection':'Existing qualified ray0 fold of S2-28011-zoom:587; no search/retry.',
                'scope':'Same consumed exact model/data as C1; local certificate only, not connection to the center orbit.'}}

def synthetic():
    net={'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2',
         'out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'}
    return {'schema':'neural-shrimp-fold-input-v1','period':18,
        'map_inputs':{'schema':'neural-shrimp-c1-input-v1','network':net,
            'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],
            'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)],'lr_hidden':'1/128','lr_output':'1/128'},
        'approximation':[str(Fraction(i-6,64)) for i in range(13)]+['1/128']+[str(Fraction(i+1,64)) for i in range(13)],
        'provenance':{'synthetic':True,'purpose':'two 18-step mixed-jet evaluations, no target and no fold claim'}}

def replay(bundle,expected,observe,tick):
    require(entry(bundle/'manifest.json')['sha256']==expected,'pinned replay manifest')
    manifest=read(bundle/'manifest.json');verify_inventory(bundle,manifest['files'])
    envelope=read(bundle/'certificate.json');inp=read(bundle/'inputs.json')
    require(envelope['schema']=='neural-shrimp-fold-certificate-v1','certificate schema')
    require(envelope['inputs_sha256']==entry(bundle/'inputs.json')['sha256'],'input binding')
    require(canonical_bytes(inp)==canonical_bytes(prepare()),'exact original target input reconstruction')
    require(envelope['runtime']['source_sha256']=={p:sha256((ROOT/p).read_bytes()) for p in SOURCES},'replay source equality')
    require(envelope['runtime']['python_flint']==flint.__version__ and envelope['runtime']['flint']==flint.__FLINT_VERSION__,'replay arithmetic library versions')
    result=check_fold(inp,**envelope['result']['settings'],tick=tick,observe=observe)
    require(canonical_bytes(result)==canonical_bytes(envelope['result']),'byte-identical mathematical replay')
    return inp,{'schema':'neural-shrimp-fold-replay-v1','passed':bool(result['passed']),
                'identical':True,'bundle':str(bundle),'manifest_sha256':expected,'result':result}

def execute(out,mode,bundle=None,manifest=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();count=0
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
        'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
        'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'internal_seconds':INTERNAL_SECONDS,'tick_cap':160,'raw_cap':CAP,'cpu_only':True,'paid':False}
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
        if count>160 or stopped or time.monotonic()-start>INTERNAL_SECONDS:raise TimeoutError('fold tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS above 2GiB')
        if total(out)>14*1024**2:raise OSError('ordinary output budget')
    serial=0
    def observe(kind,record):
        nonlocal serial
        save(out/f'evaluation-{serial:03d}-{kind}.json',record);serial+=1
        # Returned full evaluation is durable before the budget check.
        tick()
    outcome={'passed':False,'mode':mode,'status':'started'}
    code=2
    try:
        if mode=='replay':
            inp,result=replay(bundle,manifest,observe,tick);save(out/'inputs.json',inp);save(out/'replay.json',result)
            outcome.update(passed=result['passed'],status='replayed');code=0 if result['passed'] else 2
        else:
            inp=synthetic() if mode=='profile' else prepare();save(out/'inputs.json',inp)
            if mode=='profile':
                with ctx.workprec(SETTINGS['precision']):
                    model=FoldTanhMap(inp['map_inputs']);center=[exact(x) for x in inp['approximation']]
                    rows=[]
                    for tag,z in [('center',center),('box',[arb(x,arb((1,-80))) for x in center])]:
                        then=time.monotonic();g,j,_=augmented(model,z,18,tick)
                        observe('profile-'+tag,{'g':vector_json(g),'jacobian':matrix_json(j)})
                        rows.append({'evaluation':tag,'seconds':time.monotonic()-then})
                outcome.update(passed=True,status='synthetic_profile_complete',timings=rows,
                    projected_seven_evaluation_seconds=7*max(r['seconds'] for r in rows),scientific_claim=False);code=0
            else:
                result=check_fold(inp,**SETTINGS,tick=tick,observe=observe)
                save(out/'certificate.json',{'schema':'neural-shrimp-fold-certificate-v1',
                    'inputs_sha256':entry(out/'inputs.json')['sha256'],'runtime':runtime,'result':result})
                outcome.update(passed=result['passed'],status=result['status'],gates=result['gates']);code=0 if result['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)})
        save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=count,evaluations_retained=serial,
                       finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out)
        save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,bundle=None,manifest=None):
    bound=sources();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),
        'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),
        'external_seconds':EXTERNAL_SECONDS,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL_SECONDS}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    if bundle is not None:cmd+=['--bundle',str(bundle),'--manifest-sha256',manifest]
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=EXTERNAL_SECONDS+7)
        except BaseException as exc:
            meta['failure']=type(exc).__name__+': '+str(exc);code=2
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_leader_exited=proc is not None and proc.poll() is not None)
    remaining=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                stat=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(stat[2])==proc.pid and stat[0]!='Z':remaining.append(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    meta['owned_live_pids_remaining']=remaining
    if total(out)>CAP+1024**2 or remaining:
        code=2;meta['exit_code']=2;meta['failure']='resource/inventory qualification failed'
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([q for q in out.rglob('*') if q.is_dir()],reverse=True):p.chmod(0o555)
    out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code if not remaining else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['profile','target','replay'],required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--bundle',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args()
    require((a.mode=='replay')==(a.bundle is not None and a.manifest_sha256 is not None),'replay needs exact bundle and manifest pin')
    return (execute if a.execute else launch)(a.out,a.mode,a.bundle,a.manifest_sha256)
if __name__=='__main__':raise SystemExit(main())
