#!/usr/bin/env python3
"""Own-group supervisor for one fixed, separately authorized two-chart target."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-003-quadratic-target')
RSS_CAP_KIB=2*1024**2

def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def owned(group):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':
                rss=int(next((line.split()[1] for line in (p/'status').read_text().splitlines() if line.startswith('VmRSS:')),'0'))
                rows.append({'pid':int(p.name),'rss_kib':rss})
        except(OSError,ValueError):pass
    return rows

def stop_group(group,sig):
    try:os.killpg(group,sig)
    except ProcessLookupError:pass

def command():
    return ['timeout','--signal=TERM','--kill-after=5s','135s',sys.executable,str(ROOT/'tools/certify_eigenpair_locus_quadratic.py'),str(OUT/'run')]

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))>3*1024**2
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2;peak=0;seen=set();reason=None;term_time=None
    def save(name,value):
        with(OUT/name).open('xb') as f:f.write((json.dumps(value,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
    meta={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'argv':command(),'launcher_pid':os.getpid(),'mode':'target','started_unix':time.time(),'meminfo':Path('/proc/meminfo').read_text(),'load':os.getloadavg(),'monitored_group_rss_cap_kib':RSS_CAP_KIB,'rss_sampling_seconds':.1,'scope':'only this owned process group; sampled RSS not hard kernel resident-memory bound'}
    save('launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes())
    with(OUT/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(command(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('owned-process.json',{'group_pid':proc.pid})
            while proc.poll() is None:
                rows=owned(proc.pid);seen.update(r['pid'] for r in rows);rss=sum(r['rss_kib'] for r in rows);peak=max(peak,rss)
                if term_time is None:
                    if rss>=RSS_CAP_KIB:reason='owned group RSS cap'
                    elif time.monotonic()-start>=135:reason='supervisor deadline'
                    elif (OUT/'process.log').stat().st_size>=128*1024:reason='log cap'
                    if reason:stop_group(proc.pid,signal.SIGTERM);term_time=time.monotonic()
                if term_time is not None and time.monotonic()-term_time>=5:stop_group(proc.pid,signal.SIGKILL)
                if time.monotonic()-start>=142:stop_group(proc.pid,signal.SIGKILL);reason='absolute supervisor deadline';break
                time.sleep(.1)
            code=proc.wait(timeout=2)
            if reason:code=2
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:stop_group(proc.pid,signal.SIGKILL);proc.wait()
    live=[] if proc is None else owned(proc.pid)
    if live or sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>385*1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live,observed_owned_pids=sorted(seen),peak_monitored_group_rss_kib=peak,stop_reason=reason)
    save('launcher-result.json',meta)
    inv={str(p.relative_to(OUT)):entry(p) for p in sorted(OUT.rglob('*')) if p.is_file()};save('manifest.json',{'files':inv})
    for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    OUT.chmod(0o555);print(json.dumps({'root':str(OUT),'exit_code':code,'manifest':entry(OUT/'manifest.json')}));return code
if __name__=='__main__':raise SystemExit(main())
