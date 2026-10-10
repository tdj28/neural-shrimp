"""CPU1 owned-group capped launcher for one source-bound stability slice."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,json,signal,subprocess,sys,time
from pathlib import Path
import certify_s2_period18_preceding_quarter_stability as worker
io=worker.io;require=worker.require


def owned(group):
    pids=[];rss=0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            stat=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[2])!=group or stat[0]=='Z':continue
            pids.append(int(path.name))
            rss+=next(int(line.split()[1]) for line in (path/'status').read_text().splitlines() if line.startswith('VmRSS:'))
        except (OSError,ValueError,StopIteration):continue
    return pids,rss


def launch(out,mode,profile=None,profile_pin=None,profile_outer_pin=None,index=None,previous=None,previous_pin=None,previous_outer_pin=None):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10');bound=worker.sources()
    seconds=40 if mode=='profile' else 390
    source=Path(__file__).read_bytes();out.mkdir(parents=True,exist_ok=False);(out/'launcher-source.py').write_bytes(source)
    cmd=['timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',sys.executable,str(worker.ROOT/'tools/certify_s2_period18_preceding_quarter_stability.py'),'--out',str(out/'run'),'--mode',mode]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin),('--profile-outer-manifest',profile_outer_pin),('--index',index),('--previous',previous),('--previous-manifest',previous_pin),('--previous-outer-manifest',previous_outer_pin)]:
        if value is not None:cmd.extend([flag,str(value)])
    started=time.monotonic();proc=None;maximum=0
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=worker.ROOT,text=True).strip(),'source_sha256':bound,'mode':mode,
      'launcher_pid':os.getpid(),'argv':cmd,'started_utc':io.now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),
      'external_seconds':seconds,'kill_grace_seconds':5,'rss_cap_kib':2*1024**2,'aggregate_raw_cap':worker.CAP+2*1024**2,'launcher_source_sha256':io.sha256(source)}
    io.save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=worker.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=proc.pid;io.save(out/'launcher-start.json',meta)
            while proc.poll() is None:
                _,rss=owned(proc.pid);maximum=max(maximum,rss)
                require(rss<2*1024**2,'owned-group RSS cap')
                require(io.total(out)<worker.CAP+512*1024 and (out/'process.log').stat().st_size<128*1024,'raw/log cap')
                require(time.monotonic()-started<seconds+7,'external cleanup deadline')
                time.sleep(.1)
            code=proc.returncode
        except BaseException as exc:meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=owned(proc.pid)[0] if proc is not None else []
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-started,max_group_rss_kib=maximum,finished_utc=io.now(),owned_live_pids=live)
    io.save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():io.verify_inventory(out/'run',io.read(out/'run/manifest.json')['files'])
    io.save(out/'manifest.json',{'files':io.inventory(out),'within_raw_cap':io.total(out)<worker.CAP+2*1024**2})
    for path in sorted(out.rglob('*'),reverse=True):path.chmod(0o444 if path.is_file() else 0o555)
    out.chmod(0o555);require(io.total(out)<worker.CAP+2*1024**2,'aggregate raw cap')
    print(json.dumps({'exit_code':code,'elapsed_seconds':meta['elapsed_seconds'],'manifest':io.entry(out/'manifest.json')}));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--index',type=int);p.add_argument('--previous',type=Path);p.add_argument('--previous-manifest');p.add_argument('--previous-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only for target')
    raise SystemExit(launch(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest,a.index,a.previous,a.previous_manifest,a.previous_outer_manifest))
