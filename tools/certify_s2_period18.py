"""One consumed-data adapter and owned launcher for the unchanged C1 checker."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import certify_c1
from neural_shrimp.rigorous_orbits import canonical_bytes,dyadic,sha256
RAW=Path('/data2/neural-shrimp-data')
PARENT=RAW/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
CERTS=RAW/'raw/EXP-028/attempt-002/stage3/certificates.json'
PINS={str(PARENT):'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',str(CERTS):'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8',str(RAW/'explore/a2-survey/rays.json'):'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4',str(RAW/'explore/a2-survey/survey.json'):'49aee09a7ec81c6df953267d035c8185a93502b5585b83904f09b9666d46a38b'}
ROW_HASH='3cc3cd1fccc72fff3b0f1a81204cc761421d5a335b70e60a76ea199be78d6883'
SETTINGS=dict(precision=256,radius_exponent=-80,newton_steps=5,max_power=4096)
CARD='docs/plans/s2-period18-certificate.md'
SOURCES=certify_c1.SOURCES+['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/certify_s2_period18.py','tests/test_s2_period18_certificate.py',CARD,'experiments/trainability/EXP-028-plan.json']
MAX_BYTES=8*1024**2

def require(value,message):
    if not value:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def entry(p):
    b=Path(p).read_bytes();return dict(bytes=len(b),sha256=sha256(b))
def save(p,obj):
    b=canonical_bytes(obj);temp=p.with_name(p.name+'.tmp')
    with temp.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
    os.replace(temp,p)
def shape_hash(a):return sha256(repr(a.shape).encode()+np.ascontiguousarray(a).tobytes())
def objhash(obj):return sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

def convert(parent,row,spec):
    require(row['map_id']=='S2-28011-zoom' and row['component']==587 and row['period']==18,'fixed target')
    c=row['certificate'];require(c['verified'] and c['attracting'] and c['primitive_period']==18,'historical primitive attractor')
    model=parent['network'].copy()
    for k in ('in_scale','out_scale','loss_scale','ridge'):model[k]=float(Fraction(model[k]))
    require(spec['seed']==28011 and spec['system']=='S2' and model==spec['network'],'same consumed model')
    def decoded(values):
        a=np.asarray(values,dtype=object)
        nums=[Fraction(v) for v in a.flat];out=np.array([float(v) for v in nums]).reshape(a.shape)
        require(np.isfinite(out).all() and all(Fraction(float(v))==q for v,q in zip(out.flat,nums)),'exact dyadic binary64 input')
        return out
    x,y=decoded(parent['x']),decoded(parent['y'])
    require(x.shape==(24,2) and y.shape==(24,) and shape_hash(x)==spec['data_sha256']['x'] and shape_hash(y)==spec['data_sha256']['y'],'consumed training hash')
    orbit=np.asarray(c['orbit'],dtype=np.float64);require(orbit.shape==(18,13) and np.isfinite(orbit).all(),'selected full orbit')
    rates=[float(row[k]) for k in ('lr_hidden','lr_output')];require(all(np.isfinite(v) and v>0 for v in rates),'actual positive rates')
    return dict(schema='neural-shrimp-c1-input-v1',network=parent['network'],x=parent['x'],y=parent['y'],lr_hidden=dyadic(rates[0]),lr_output=dyadic(rates[1]),period=18,approximation=[dyadic(v) for v in orbit[0]],provenance=dict(map_id=row['map_id'],component=587,row_index=111,phase=0,data_sha256={k:spec['data_sha256'][k] for k in ('x','y')},parent_consumed_inputs=str(PARENT),parent_inputs_sha256=PINS[str(PARENT)],certificate_file=str(CERTS),certificate_sha256=PINS[str(CERTS)],row_canonical_sha256=ROW_HASH,selection='Exploratory choice of the existing strong classical-geometry candidate S2-28011-zoom:587; no new parameter or rank search.',semantics='Binary64 saved model/data/rates/phase interpreted as exact dyadic constants; tanh and GD are exact real. No proof of boundaries, ancestry, chaos or floating implementation.'))

def prepare():
    for p,h in PINS.items():require(entry(p)['sha256']==h,'pinned source input '+p)
    row=read(CERTS)[111];require(objhash(row)==ROW_HASH,'row111 exact record')
    spec=next(m for m in read(ROOT/'experiments/trainability/EXP-028-plan.json')['maps'] if m['map_id']=='S2-28011')
    inputs=convert(read(PARENT),row,spec);inputs['provenance']['plan_sha256']=entry(ROOT/'experiments/trainability/EXP-028-plan.json')['sha256']
    rays=read(RAW/'explore/a2-survey/rays.json')['S2-28011-zoom:587'];summary=next(r for r in read(RAW/'explore/a2-survey/survey.json') if r['key']=='S2-28011-zoom:587')
    require(summary['period']==18 and rays['period']==18 and summary['shrimp_consistent_draft'],'same historical geometry candidate')
    return inputs,dict(summary=summary,ray_record_sha256=objhash(rays),input_pins=PINS,scope='Historical floating ray/map anatomy only; geometry and ancestry not certified by this attempt.')

def pair_fraction(pair):
    require(isinstance(pair,list) and len(pair)==2 and all(type(v) is int for v in pair),'integer dyadic pair');return Fraction(pair[0])*Fraction(2)**pair[1]
def proximity(inputs,box):
    seed=[Fraction(v) for v in inputs['approximation']];require(len(seed)==len(box)==13,'full state proximity');scale=max(Fraction(1),*(abs(v) for v in seed));rows=[];upper=Fraction(0)
    for i,(s,b) in enumerate(zip(seed,box)):
        require(b.get('finite',True) is not False and 'mid' in b and 'rad' in b,'finite exact enclosure')
        m,r=pair_fraction(b['mid']),pair_fraction(b['rad']);require(r>=0,'positive radius');d=abs(m-s)+r;upper=max(upper,d)
        rows.append(dict(coordinate=i,seed=str(s),midpoint=str(m),radius=str(r),distance_upper=str(d)))
    value=upper/scale;threshold=Fraction(1,10_000_000)
    return dict(passed=value<=threshold,normalized_whole_box_distance_upper=str(value),threshold=str(threshold),normalization=str(scale),coordinates=rows,scope='Exact rational whole-box distance to historical row111 phase0 only; not seed membership, branch identity, boundary or full-window certification.')

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    out={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name);out[name]=sha256(b)
    return out

def deadline(*_):raise TimeoutError('one bounded period18 attempt deadline')
def execute(out):
    binding=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(status='started',passed=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=binding,settings=SETTINGS,pid=os.getpid(),ppid=os.getppid(),started_utc=datetime.now(timezone.utc).isoformat(),threads={k:os.environ[k] for k in THREADS},nice=os.getpriority(os.PRIO_PROCESS,0),gpu=False,paid=False)
    save(out/'adapter-start.json',result);signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGTERM,deadline);signal.setitimer(signal.ITIMER_REAL,60)
    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_BYTES,MAX_BYTES))
    try:
        inputs,geometry=prepare();save(out/'selected-inputs.json',inputs);save(out/'geometry-binding.json',geometry)
        result['inputs_sha256']=sha256(canonical_bytes(inputs));print(json.dumps(dict(stage='proof_start',pid=os.getpid(),inputs_sha256=result['inputs_sha256'])),flush=True)
        code=certify_c1.run(inputs,out/'certificate',SETTINGS)
        cert=read(out/'certificate/certificate.json');near=proximity(inputs,cert['result']['box']);save(out/'whole-box-proximity.json',near)
        result.update(status='certificate_passed' if code==0 else 'certificate_inconclusive',passed=code==0,whole_box_proximity_passed=near['passed'],normalized_whole_box_distance_upper=near['normalized_whole_box_distance_upper'],certificate_receipt=read(out/'certificate/receipt.json'))
    except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error));save(out/'failure.json',dict(error=result['error']))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=datetime.now(timezone.utc).isoformat());save(out/'adapter-result.json',result)
        files={str(p.relative_to(out)):entry(p) for p in out.rglob('*') if p.is_file()};save(out/'manifest.json',dict(files=files,status=result['status'],total_bytes=sum(v['bytes'] for v in files.values()),within_limit=sum(v['bytes'] for v in files.values())<MAX_BYTES))
    return 0 if result['passed'] else 2

def owned_live_pids(group):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[2])==group and f[0]!='Z':rows.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows

def launch(out):
    sources();require(not out.exists(),'exclusive attempt');out.mkdir(parents=True);start=time.monotonic();owned=None
    meta=dict(started_utc=datetime.now(timezone.utc).isoformat(),launcher_pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),load_average=os.getloadavg(),meminfo=Path('/proc/meminfo').read_text(),cpu_only=True,external_seconds=75,kill_grace=5,raw_cap_bytes=MAX_BYTES)
    cmd=['timeout','--signal=TERM','--kill-after=5s','75s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--out',str(out/'proof')];meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            owned=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=owned.pid;save(out/'launcher-start.json',meta);code=owned.wait(timeout=81)
        except BaseException as error:
            meta['failure']=type(error).__name__+': '+str(error)
            if owned is not None:
                try:os.killpg(owned.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                owned.wait()
            code=2
    if owned is not None:
        try:os.killpg(owned.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    meta['owned_live_pids_remaining']=[] if owned is None else owned_live_pids(owned.pid)
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),owned_group_exited=owned is not None and owned.poll() is not None);save(out/'launcher-result.json',meta)
    files={str(p.relative_to(out)):entry(p) for p in out.rglob('*') if p.is_file()};save(out/'manifest.json',dict(files=files,total_bytes=sum(v['bytes'] for v in files.values()),within_limit=sum(v['bytes'] for v in files.values())<MAX_BYTES))
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in out.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    out.chmod(0o555);print(json.dumps(dict(exit_code=code,out=str(out),manifest=entry(out/'manifest.json'),elapsed_seconds=meta['elapsed_seconds'])),flush=True);return code

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();return execute(a.out) if a.execute else launch(a.out)
if __name__=='__main__':raise SystemExit(main())
