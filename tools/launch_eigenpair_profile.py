#!/usr/bin/env python3
"""Own-process-only bounded launcher for the synthetic eigenpair cost profile."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-profile')
def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))>3*1024**2
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    def save(name,value):
        with(OUT/name).open('xb') as f:f.write((json.dumps(value,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
    cmd=['timeout','--signal=TERM','--kill-after=5s','135s',sys.executable,str(ROOT/'tools/profile_eigenpair_locus.py'),str(OUT/'run')]
    meta={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'argv':cmd,'launcher_pid':os.getpid(),'mode':'synthetic_only','started_unix':time.time(),'meminfo':Path('/proc/meminfo').read_text(),'load':os.getloadavg()}
    save('launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes())
    with(OUT/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=142)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    live=[]
    if proc:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(p.name))
            except(OSError,ValueError):pass
    if live or sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>257*1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
    inv={str(p.relative_to(OUT)):entry(p) for p in sorted(OUT.rglob('*')) if p.is_file()};save('manifest.json',{'files':inv})
    for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    OUT.chmod(0o555);print(json.dumps({'root':str(OUT),'exit_code':code,'manifest':entry(OUT/'manifest.json')}));return code
if __name__=='__main__':raise SystemExit(main())
