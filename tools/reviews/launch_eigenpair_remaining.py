"""One owned-group bounded saved-arithmetic review launch."""
import hashlib,json,os,signal,subprocess,sys,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/audits/eigenpair-remaining-locus')
def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    OUT=BASE/'unit000-attempt001'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))>4*1024**2
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    def save(n,v):
        with(OUT/n).open('xb')as f:f.write((json.dumps(v,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
    command=['timeout','--signal=TERM','--kill-after=5s','195s',sys.executable,str(ROOT/'tools/reviews/review_eigenpair_remaining.py'),'--out',str(OUT/'run'),'--spec',str(ROOT/'docs/plans/eigenpair-remaining-unit000.json')]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'launcher_pid':os.getpid(),'argv':command,'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'started_unix':time.time(),'internal_seconds':180,'external_seconds':195,'kill_grace_seconds':5}
    save('launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes())
    with(OUT/'process.log').open('xb')as log:
        try:proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=202)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    live=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                f=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(f[2])==proc.pid and f[0]!='Z':live.append(int(p.name))
            except(OSError,ValueError):pass
    if live or sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())>25*1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
    inv={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv,'total_bytes':sum(e['bytes']for e in inv.values())})
    for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    OUT.chmod(0o555);print(json.dumps({'root':str(OUT),'exit_code':code,'manifest':entry(OUT/'manifest.json')}),flush=True);return code
if __name__=='__main__':raise SystemExit(main())
