"""Single source-bound saved-lower-gap audit launcher; no target QR/model operations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,selectors,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/audits/s1r-lower-gap-independent')
def require(value,message):
    if not value:raise ValueError(message)
def canonical(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    data=v if isinstance(v,bytes) else canonical(v)
    require(sum(x.stat().st_size for x in p.parent.iterdir())+len(data)<=1024*1024,'launcher byte cap')
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def group_status(group):
    rows=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group:rows.append(dict(pid=int(path.name),state=fields[0]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows

def cleanup_owned(group):
    try:os.killpg(group,signal.SIGKILL)
    except ProcessLookupError:pass
    deadline=time.monotonic()+1.
    while True:
        live=[r['pid'] for r in group_status(group) if r['state']!='Z']
        if not live or time.monotonic()>=deadline:return live
        time.sleep(.01)

def main():
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    source,pins_path,pins_sha=sys.argv[1:];pins_path=Path(pins_path)
    require(len(source)==40 and len(pins_sha)==64 and entry(pins_path)['sha256']==pins_sha,'full source and pinned metadata')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source,'audit execution HEAD')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit checkout')
    require(Path(__file__).read_bytes()==subprocess.check_output(['git','show',source+':tools/reviews/launch_s1r_lower_gap_audit.py'],cwd=ROOT),'committed launcher bytes')
    out=BASE/'attempt-001';launcher=BASE/'attempt-001-launcher'
    require(not out.exists() and not launcher.exists(),'exclusive attempt')
    env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')});env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
    argv=['timeout','--signal=TERM','--kill-after=5s','20s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python',
      'tools/reviews/audit_s1r_lower_gap_outcome.py','--out',str(out),'--pins',str(pins_path),'--pins-sha256',pins_sha,'--source-commit',source]
    launcher.mkdir(parents=True,exist_ok=False);write(launcher/'launcher-source.py',Path(__file__).read_bytes())
    runtime=dict(source_commit=source,pins_file=entry(pins_path),pins= json.loads(pins_path.read_bytes()),argv=argv,cwd=str(ROOT),launcher_pid=os.getpid(),
      started_utc=datetime.now(timezone.utc).isoformat(),nice=os.getpriority(os.PRIO_PROCESS,0),load_average=os.getloadavg(),thread_env={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
      out=str(out),log_limit=128*1024,launcher_byte_limit=1024*1024,supervisor_seconds=25)
    write(launcher/'runtime-start.json',runtime);process=None;start=time.monotonic();result=dict(status='pending',exit_code=None,error=None,log_truncated=False,owned_live_pids_remaining=[]);seen=set()
    try:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        write(launcher/'process.json',dict(timeout_pid=process.pid));selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ);size=0
        with (launcher/'stdout.log').open('xb') as log:
            while selector.get_map() or process.poll() is None:
                for key,_ in selector.select(.1):
                    data=os.read(key.fd,65536)
                    if not data:selector.unregister(key.fileobj);continue
                    block=data[:max(0,128*1024-size)];log.write(block);log.flush();os.fsync(log.fileno());size+=len(block)
                    if len(block)!=len(data):
                        result['log_truncated']=True
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                seen.update(r['pid'] for r in group_status(process.pid))
                if time.monotonic()-start>=25:
                    result['error']='supervisor_timeout_or_inherited_pipe'
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    for key in list(selector.get_map().values()):selector.unregister(key.fileobj)
                    process.stdout.close()
                    break
            selector.close()
        result.update(status='finished',exit_code=process.wait(timeout=1))
    except BaseException as e:
        result.update(status='wrapper_failure',error=type(e).__name__+': '+str(e))
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            result['exit_code']=process.wait(timeout=1)
    finally:
        if process is not None:
            seen.update(r['pid'] for r in group_status(process.pid))
            result['owned_live_pids_remaining']=cleanup_owned(process.pid)
            if process.poll() is None:result['exit_code']=process.wait(timeout=1)
        result['observed_owned_pids']=sorted(seen)
        result.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),timeout_pid=None if process is None else process.pid)
        result['raw_files']={} if not out.exists() else {p.name:entry(p) for p in sorted(out.iterdir()) if p.is_file()}
        result['raw_bytes']=sum(v['bytes'] for v in result['raw_files'].values());write(launcher/'receipt.json',result)
        write(launcher/'manifest.json',{p.name:entry(p) for p in sorted(launcher.iterdir())})
        for folder in (out,launcher):
            if folder.exists():
                for file in folder.iterdir():file.chmod(0o444)
                folder.chmod(0o555)
        require(sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=8*1024**2,'aggregate audit cap; retain all evidence')
        print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},sort_keys=True))
    return 0 if result['exit_code']==0 and result['error'] is None and not result['log_truncated'] and not result['owned_live_pids_remaining'] else 2
if __name__=='__main__':raise SystemExit(main())
