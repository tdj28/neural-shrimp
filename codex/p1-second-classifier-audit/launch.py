"""Own exactly one finite audit worker; no Git, models, or retry loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PYTHON='/data2/neural-shrimp/.venv/bin/python'
ROOT=Path('/data2/neural-shrimp-data/codex/p1-second-classifier-audit')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
START_AVAILABLE=20*1024**3
RUN_AVAILABLE=16*1024**3


def process(pid):
    text=Path(f'/proc/{pid}/stat').read_text()
    fields=text[text.rfind(')')+2:].split()
    return {'pid':pid,'starttime':int(fields[19]),'pgrp':int(fields[2]),'session':int(fields[3])}


def rss(pid):
    data=Path(f'/proc/{pid}/status').read_text().splitlines()
    return next((int(x.split()[1])*1024 for x in data if x.startswith('VmRSS:')),0)


def available():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                if x.startswith('MemAvailable:'))


def available_gate(value,running=False):
    floor=RUN_AVAILABLE if running else START_AVAILABLE
    if value<floor:
        raise MemoryError('protective MemAvailable floor '+str(floor))


def dump(path,obj):
    with path.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)


def stop_owned(child,identity):
    if child.poll() is not None: return
    actual=process(child.pid)
    if actual!=identity or identity['pgrp']!=child.pid or identity['session']!=child.pid:
        raise RuntimeError('ownership uncertain; no signal sent')
    child.send_signal(signal.SIGTERM)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process(child.pid)!=identity: raise RuntimeError('ownership changed before kill')
        child.kill(); child.wait(timeout=5)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--card',type=Path,required=True); a=ap.parse_args()
    card=json.loads(a.card.read_text())
    if card['status']!='accepted_by_root' or not card.get('acceptance_reference'):
        raise ValueError('prepared cards cannot launch')
    if card['cpu_threads']!=1 or card['nice']!=10 or card['as_bytes']!=4*1024**3:
        raise ValueError('resource policy mismatch')
    if card['raw_bytes']!=64*1024**2: raise ValueError('fixed64MiB raw envelope')
    ceilings={'profile':120,'descriptive':300,'l8':300}
    if not 0<card['seconds']<=ceilings[card['mode']]: raise ValueError('finite mode deadline violated')
    for rel,want in card['source_sha256'].items():
        p=HERE/rel
        if p.resolve().parent!=HERE or hashlib.sha256(p.read_bytes()).hexdigest()!=want:
            raise ValueError('pinned source mismatch')
    if 'launch.py' not in card['source_sha256']: raise ValueError('launcher pin missing')
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10: raise ValueError('exact nice10 required')
    if nice<10: os.nice(10-nice)
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<card['as_bytes']: raise ValueError('hard AS too small')
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,card['as_bytes']))
    available_gate(available())
    out=Path(card['output'])
    if out.resolve().parent!=ROOT or out.exists(): raise ValueError('fresh direct-child output required')
    owner=ROOT/(out.name+'-owner'); owner.mkdir(parents=True,exist_ok=False)
    dump(owner/'card.json',card)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',**{k:'1' for k in THREADS})
    command=[PYTHON,'-I',str(HERE/'run.py'),'--card',str((owner/'card.json').resolve())]
    start=time.monotonic(); child=None; identity=None; result={'completed':False,'command':command}
    def interrupted(signum,frame): raise InterruptedError('supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        with (owner/'stdout').open('xb') as stdout,(owner/'stderr').open('xb') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            # Popen ownership retained immediately; birth identity obtained before polling/cleanup.
            identity=process(child.pid)
            dump(owner/'owned-process.json',{'launcher':process(os.getpid()),'worker':identity,'command':command})
            peak=0
            while child.poll() is None:
                try:
                    used=rss(os.getpid())+rss(child.pid)
                except FileNotFoundError:
                    if child.poll() is not None: break
                    raise RuntimeError('worker status vanished while direct child remains live')
                peak=max(peak,used)
                if time.monotonic()-start>card['seconds']+10:
                    raise TimeoutError('outer finite deadline')
                if used>card['as_bytes']:
                    raise MemoryError('sampled aggregate RSS watchdog')
                available_gate(available(),running=True)
                raw=sum(p.stat().st_size for root in (out,owner) if root.exists() for p in root.iterdir() if p.is_file())
                if raw>card['raw_bytes']-2*1024**2:
                    raise OSError('raw budget reached with2MiB closing reserve')
                time.sleep(.1)
            result.update(worker_exit=child.returncode,peak_sampled_aggregate_rss=peak)
            worker=json.loads((out/'result.json').read_text())
            result['completed']=child.returncode==0 and worker['completed']
    except BaseException as e:
        result['error']=repr(e)
    finally:
        if child is not None and child.poll() is None:
            try:
                if identity is None:
                    # Direct unreaped child PID is still owned, but absent birth registration is a failure.
                    identity=process(child.pid)
                stop_owned(child,identity)
            except BaseException as e: result['cleanup_error']=repr(e)
        result.update(seconds=time.monotonic()-start,owned_worker_exited=child is None or child.poll() is not None)
        result['raw_bytes_before_closing']=sum(p.stat().st_size for root in (out,owner) if root.exists()
            for p in root.iterdir() if p.is_file())
        if result['raw_bytes_before_closing']>card['raw_bytes']-2*1024**2: result['completed']=False
        if not result['owned_worker_exited'] or result['seconds']>card['seconds']+20: result['completed']=False
        result['files']=[{'name':p.name,'bytes':p.stat().st_size,
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(owner.iterdir())]
        dump(owner/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['completed'] else 2


if __name__=='__main__': raise SystemExit(main())
