"""Single source-bound saved-data audit launcher; no target QR/model operations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,selectors,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/audits/cifar-input-span-independent')
def canonical(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
    data=v if isinstance(v,bytes) else canonical(v)
    assert sum(x.stat().st_size for x in p.parent.iterdir())+len(data)<=512*1024
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def main():
    source,pins_path,pins_sha=sys.argv[1:];pins_path=Path(pins_path)
    assert len(source)==40 and len(pins_sha)==64 and entry(pins_path)['sha256']==pins_sha
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==source
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert Path(__file__).read_bytes()==subprocess.check_output(['git','show',source+':tools/reviews/launch_cifar_input_span_audit.py'],cwd=ROOT)
    out=BASE/'attempt-001';launcher=BASE/'attempt-001-launcher'
    assert not out.exists() and not launcher.exists()
    env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')});env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'))
    argv=['timeout','--signal=TERM','--kill-after=5s','20s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python',
      'tools/reviews/audit_cifar_input_span_outcome.py','--out',str(out),'--pins',str(pins_path),'--pins-sha256',pins_sha,'--source-commit',source]
    launcher.mkdir(parents=True,exist_ok=False);write(launcher/'launcher-source.py',Path(__file__).read_bytes())
    runtime=dict(source_commit=source,pins_file=entry(pins_path),pins= json.loads(pins_path.read_bytes()),argv=argv,cwd=str(ROOT),launcher_pid=os.getpid(),
      started_utc=datetime.now(timezone.utc).isoformat(),load_average=os.getloadavg(),thread_env={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
      out=str(out),log_limit=128*1024,launcher_byte_limit=512*1024,supervisor_seconds=25)
    write(launcher/'runtime-start.json',runtime);process=None;start=time.monotonic();result=dict(status='pending',exit_code=None,error=None,log_truncated=False)
    try:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        write(launcher/'process.json',dict(timeout_pid=process.pid));selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ);size=0
        with (launcher/'stdout.log').open('xb') as log:
            while selector.get_map():
                for key,_ in selector.select(.1):
                    data=os.read(key.fd,65536)
                    if not data:selector.unregister(key.fileobj);continue
                    block=data[:max(0,128*1024-size)];log.write(block);log.flush();os.fsync(log.fileno());size+=len(block)
                    if len(block)!=len(data):
                        result['log_truncated']=True
                        try:os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                if time.monotonic()-start>=25:
                    result['error']='supervisor_timeout_or_inherited_pipe'
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    for key in list(selector.get_map().values()):selector.unregister(key.fileobj)
                    process.stdout.close()
            selector.close()
        result.update(status='finished',exit_code=process.wait())
    except BaseException as e:
        result.update(status='wrapper_failure',error=type(e).__name__+': '+str(e))
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            result['exit_code']=process.wait()
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),timeout_pid=None if process is None else process.pid)
        result['raw_files']={} if not out.exists() else {p.name:entry(p) for p in sorted(out.iterdir()) if p.is_file()}
        result['raw_bytes']=sum(v['bytes'] for v in result['raw_files'].values());write(launcher/'receipt.json',result)
        write(launcher/'manifest.json',{p.name:entry(p) for p in sorted(launcher.iterdir())})
        for folder in (out,launcher):
            if folder.exists():
                for file in folder.iterdir():file.chmod(0o444)
                folder.chmod(0o555)
        assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<=4*1024**2,'aggregate audit cap; retain all evidence'
        print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},sort_keys=True))
    return 0 if result['exit_code']==0 and result['error'] is None and not result['log_truncated'] else 2
if __name__=='__main__':raise SystemExit(main())
