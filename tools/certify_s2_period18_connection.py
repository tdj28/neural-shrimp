"""Exclusive measured prototype of one center-anchored exact-real rate interval."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,check_orbit
from neural_shrimp.rigorous_connection import HiddenRateMap,tube,require
RECEIPT='certificates/s2-period18-receipt.json'
RECEIPT_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
CARD='docs/plans/s2-period18-connection-pilot.md'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/rigorous_orbits.py',
 'src/neural_shrimp/rigorous_connection.py','tools/certify_s2_period18_connection.py',
 'tests/test_rigorous_connection.py',CARD,RECEIPT]
SETTINGS=dict(precision=256,halfwidth_exponent=-20,radius_exponent=-28,newton_steps=2,max_power=4096)
CAP=64*1024**2

def entry(path):
    b=path.read_bytes();return {'sha256':sha256(b),'bytes':len(b)}
def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as f: data=f.read(256*1024**2+1)
        require(len(data)<=256*1024**2,'decompressed artifact cap')
        if path.name=='certificate.json.gz':
            meta=json.loads(path.with_name('certificate-payload.json').read_bytes())
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

def prepare():
    receipt_path=ROOT/RECEIPT;require(entry(receipt_path)['sha256']==RECEIPT_SHA,'fixed canonical anchor')
    receipt=read(receipt_path);base=Path(receipt['external_root'])
    verify_inventory(base,receipt['files'])
    envelope=read(base/'certificate.json');anchor=envelope['result'];inp=read(base/'inputs.json')
    require(receipt['passed'] and anchor['passed'] and receipt['statement']==anchor['exact_statement'],'certified anchor statement')
    require(inp['period']==18 and anchor['exact_statement']['dimension']==13,'fixed 18x13 state order')
    require(inp['lr_hidden']=='1254985102288567/17592186044416' and inp['lr_output']=='1848862009810323/2251799813685248','original exact layer rates')
    require(entry(base/'inputs.json')['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','exact consumed model/data')
    require(entry(ROOT/'src/neural_shrimp/rigorous_orbits.py')['sha256']==receipt['source_sha256']['src/neural_shrimp/rigorous_orbits.py'],'unchanged anchor kernel')
    phases=anchor['orbit_enclosures'][:18]
    initial=[[str(Fraction(v['mid'][0])*Fraction(2)**v['mid'][1]) for v in row] for row in phases]
    return {'model_inputs':inp,'initial':initial,'rate':inp['lr_hidden'],
            'anchor':{'rate':inp['lr_hidden'],'phase_boxes':phases},
            'provenance':{'receipt_sha256':RECEIPT_SHA,'receipt':RECEIPT,'certificate':entry(base/'certificate.json'),
                          'inputs':entry(base/'inputs.json'),'order':'native phases0..17 of unchanged C1 certificate; no phase permutation',
                          'model_identity':'exact same model_inputs used for every phase and rate; only hidden rate varies',
                          'anchor_source_commit':receipt['source_commit']}}

def synthetic():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,
         'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'1/1000'},
         'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],
         'y':['0']*24,'lr_hidden':'1/128','lr_output':'1/128'}
    return {'model_inputs':inp,'initial':[['0']*13 for _ in range(18)],'rate':'1/128','anchor':None,
            'provenance':{'synthetic':True,'scientific_claim':False,'purpose':'same18x13 dense shooting algebra and24-sample tanh derivatives; no target inputs'}}

def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'approved profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    result=read(path/'outcome.json');runtime=read(path/'runtime-start.json')
    require(result['status']=='synthetic_profile_completed' and result['elapsed_seconds']<20,'completed profile timing')
    require(runtime['source_sha256']==sources(),'same profiled source bytes')
    require(4*result['elapsed_seconds']+2<100,'conservative profile projection below100 seconds')
    require(total(path)*3<60*1024**2,'three-times profile storage fits ordinary allowance')
    return {'manifest_sha256':pin,'elapsed_seconds':result['elapsed_seconds'],
            'projected_seconds':4*result['elapsed_seconds']+2,'profile_bytes':total(path),
            'scope':'conservative engineering gate, not a proof or exact runtime prediction'}

def execute(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    limits=20 if mode=='profile' else 120
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
      'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
      'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'internal_seconds':limits,'tick_cap':240,'raw_cap':CAP,'cpu_only':True,'paid':False}
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
        if ticks>240 or stopped or time.monotonic()-start>limits:raise TimeoutError('tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('RSS cap')
        if total(out)>60*1024**2:raise OSError('ordinary raw allowance')
    def observe(kind,row):
        nonlocal serial
        save(out/f'stage-{serial:03d}-{kind}.json',row);serial+=1;tick()
    outcome={'passed':False,'mode':mode,'status':'started'};code=2
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        inp=synthetic() if mode=='profile' else prepare()
        if mode=='replay':
            require(entry(bundle/'manifest.json')['sha256']==bundle_pin,'fixed replay manifest')
            verify_inventory(bundle,read(bundle/'manifest.json')['files'])
            old=read(bundle/'certificate.json.gz')
            require(old['runtime']['source_sha256']==bound,'replay source equality')
            require(old['runtime']['python_flint']==flint.__version__ and old['runtime']['flint']==flint.__FLINT_VERSION__,'same arithmetic libraries')
            require(canonical_bytes(inp)==canonical_bytes(read(bundle/'inputs.json')),'original anchor input reconstruction')
            anchor_root=Path(read(ROOT/RECEIPT)['external_root'])
            original=read(anchor_root/'certificate.json')['result']
            require(canonical_bytes(check_orbit(inp['model_inputs'],**original['settings']))==canonical_bytes(original),'unchanged anchor proof replay')
        save(out/'inputs.json',inp)
        with ctx.workprec(SETTINGS['precision']):
            result=tube(HiddenRateMap(inp['model_inputs']),inp['initial'],inp['rate'],anchor=inp['anchor'],**SETTINGS,tick=tick,observe=observe)
        payload=save(out/'certificate.json.gz',{'schema':'neural-shrimp-connection-certificate-v1','inputs_sha256':entry(out/'inputs.json')['sha256'],
            'runtime':runtime,'result':result})
        save(out/'certificate-payload.json',payload)
        if mode=='replay':require(canonical_bytes(result)==canonical_bytes(old['result']),'byte-identical tube replay')
        profile_complete=mode=='profile'
        outcome.update(passed=bool(result['passed']) and not profile_complete,
            status='synthetic_profile_completed' if profile_complete else ('anchored_interval_certified' if result['passed'] else 'inconclusive'),
            gates=result['gates'],scientific_claim=result['passed'] and not profile_complete)
        code=0 if profile_complete or result['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out)
        save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code


def launch(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 135
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--bundle',bundle),('--manifest-sha256',bundle_pin)]:
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
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target','replay'],required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--bundle',type=Path);p.add_argument('--manifest-sha256')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires fixed profile pin')
    require((a.mode=='replay')==(a.bundle is not None and a.manifest_sha256 is not None),'replay requires fixed bundle pin')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest,a.bundle,a.manifest_sha256)
if __name__=='__main__':raise SystemExit(main())
