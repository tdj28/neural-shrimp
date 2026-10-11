"""Exclusive measured prototype of saved-matrix periodic Lyapunov stability witness."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint,numpy,scipy
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_orbits import canonical_bytes,sha256,check_orbit
from neural_shrimp.rigorous_connection import HiddenRateMap,require
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_periodic_lyapunov import periodic_witness
RECEIPT='certificates/s2-period18-receipt.json'
RECEIPT_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
CARD='docs/plans/s2-period18-periodic-metric-pilot.md'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/rigorous_orbits.py',
 'src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py',
 'tools/certify_s2_period18_connection_frames.py','tests/test_rigorous_connection.py',
 'tests/test_rigorous_connection_frames.py','docs/plans/s2-period18-connection-framed-pilot.md',
 'src/neural_shrimp/rigorous_periodic_lyapunov.py','tests/test_rigorous_periodic_lyapunov.py',
 'tools/certify_s2_period18_periodic_metric.py','docs/reviews/s2-period18-connection-framed-outcome.json',CARD,RECEIPT]
SETTINGS=dict(precision=256,halfwidth_exponent=-12,radius_exponent=-20,newton_steps=2,max_power=4096)
CAP=16*1024**2

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
    require(len(data)<15*1024**2,'single stored artifact cap')
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

PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-framed-2026-10-09/attempt-002-target/run')
PARENT_MANIFEST='b7fcf410a9f806f21dfa3d1bd650bb0f93782b0e30294cba0ed4b4a1c2efd270'
PARENT_CERTIFICATE='2df4a3c51362350bb53dce07f50dd5fc0ec796a82bb4de0156e33eea992a1a58'


def prepare():
    require(entry(PARENT/'manifest.json')['sha256']==PARENT_MANIFEST,'fixed parent inventory')
    verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    require(entry(PARENT/'certificate.json.gz')['sha256']==PARENT_CERTIFICATE,'fixed parent certificate')
    parent=read(PARENT/'certificate.json.gz')
    require(parent['runtime']['source_commit']=='5b4ce6d1eed6cc30e000ec4860e8efab61855eef','parent source commit')
    for name,pin in parent['runtime']['source_sha256'].items():
        require(entry(ROOT/name)['sha256']==pin,'unchanged parent proof source '+name)
    result=parent['result'];require(result['gates']['connected_primitive_branch'] and len(result['charts'])==2,'proved joined primitive branch')
    for chart in result['charts']:
        require(all(chart['gates'][k] for k in ('uniform_root','primitive_period','anchor_identified')),'root prerequisites')
        require(chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'fixed native dimensions')
        require(chart['root_image_stability']['attempted'],'proved narrowed boxes')
    return result


def execute(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=0;serial=0;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),
        'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),
        'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'internal_seconds':20,'tick_cap':200,'raw_cap':CAP,
        'cpu_only':True,'paid':False,'parent_manifest':PARENT_MANIFEST,'parent_certificate':PARENT_CERTIFICATE}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,20)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>200 or stopped or time.monotonic()-start>20:raise TimeoutError('tick/wall budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('RSS cap')
        if total(out)>15*1024**2:raise OSError('ordinary raw allowance')
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}';payload=save(out/(name+'.json.gz'),row)
        save(out/(name+'-payload.json'),payload);serial+=1;tick()
    outcome={'passed':False,'status':'started'};code=2
    try:
        parent=prepare();save(out/'parent-binding.json',{'path':str(PARENT),'manifest':entry(PARENT/'manifest.json'),
            'certificate':entry(PARENT/'certificate.json.gz'),'connected_parameter_endpoints':parent['connected_parameter_endpoints']})
        proofs=[]
        for i,chart in enumerate(parent['charts']):
            with ctx.workprec(256):
                blocks=[restore_matrix(m) for m in chart['root_image_stability']['phase_jacobians']]
                proof=periodic_witness(blocks,precision=256,tick=tick,observe=observe)
            observe('chart_completed',{'chart':i,'proof':proof});proofs.append(proof)
        result={'schema':'neural-shrimp-periodic-metric-connection-v1','passed':all(p['passed'] for p in proofs),
            'proofs':proofs,'parent_certificate_sha256':PARENT_CERTIFICATE,
            'connected_parameter_endpoints':parent['connected_parameter_endpoints'],
            'scope':'Uniform full-state attraction on the two parent primitive18 charts only if both periodic witnesses pass; parent existence/primitive/join proof remains a separate dependency.'}
        payload=save(out/'certificate.json.gz',{'runtime':runtime,'result':result});save(out/'certificate-payload.json',payload)
        outcome.update(passed=result['passed'],status='periodic_metric_certified' if result['passed'] else 'inconclusive',
            chart_gates=[{'point':p['point_passed'],'interval':p['passed']} for p in proofs])
        code=0 if result['passed'] else 2
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values())})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=30
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
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--execute',action='store_true')
    a=p.parse_args();return (execute if a.execute else launch)(a.out,a.mode)
if __name__=='__main__':raise SystemExit(main())
