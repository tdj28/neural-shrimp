"""Exclusive saved-only review; sampled resource caps and owned-group cleanup."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import audit_chart156_stability_saved as a

def group(pid):
    found=[];rss=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[2])!=pid or stat[0]=='Z':continue
            found.append(int(p.name));rss+=next(int(s.split()[1]) for s in (p/'status').read_text().splitlines() if s.startswith('VmRSS:'))
        except (OSError,ValueError,StopIteration):pass
    return found,rss

def main(out):
    a.need(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10')
    a.need(not subprocess.check_output(['git','status','--porcelain'],cwd=a.ROOT,text=True).strip(),'clean review source')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();proc=None;code=1;maxrss=0
    cmd=['timeout','--signal=TERM','--kill-after=5s','80s',sys.executable,str(Path(__file__).with_name('audit_chart156_stability_saved.py')),'--out',str(out/'run')]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.ROOT,text=True).strip(),'launcher_pid':os.getpid(),'command':cmd,'nice':os.getpriority(os.PRIO_PROCESS,0),'meminfo':Path('/proc/meminfo').read_text(),'load':os.getloadavg(),'internal_seconds':60,'external_seconds':80,'grace_seconds':5,'rss_cap_kib':2*1024**2,'raw_cap_bytes':8*1024**2}
    a.put(out/'launch-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=a.ROOT);meta['owned_group']=proc.pid;a.put(out/'launch-start.json',meta)
            while proc.poll() is None:
                live,rss=group(proc.pid);maxrss=max(maxrss,rss)
                a.need(rss<2*1024**2 and time.monotonic()-started<87,'sampled memory/time cap')
                a.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<7*1024**2,'ordinary raw cap')
                time.sleep(.1)
            code=proc.returncode
        except BaseException as e:meta['error']={'type':type(e).__name__,'message':str(e)}
        finally:
            if proc:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait(timeout=5)
    live=group(proc.pid)[0] if proc else []
    if live:code=1
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-started,peak_group_rss_kib=maxrss,owned_live_pids=live);a.put(out/'launch-end.json',meta)
    files={str(p.relative_to(out)):{'sha256':a.sha(p.read_bytes()),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
    a.put(out/'manifest.json',{'files':files});a.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<8*1024**2,'final aggregate cap')
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o444 if p.is_file() else 0o555)
    out.chmod(0o555);print(json.dumps(meta));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
