"""Exclusive bounded owned-process launch; no GPU or foreign-process actions."""
import argparse,hashlib,json,os,selectors,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def group_members(group):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            f=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[2])==group and f[0]!='Z':found.append((int(path.name),int(f[21])*os.sysconf('SC_PAGE_SIZE')))
        except (OSError,ValueError):pass
    return found

def launch(args):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source required')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('launcher nice10 required')
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('one thread env required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();wall=20 if args.mode=='profile' else 75
    def save(name,obj):
        with(out/name).open('xb')as f:f.write((json.dumps(obj,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
    command=['timeout','--signal=TERM','--kill-after=5s',str(wall)+'s',sys.executable,str(ROOT/'tools/explore_response_spectrum.py'),'--mode',args.mode,'--out',str(out/'run')]
    if args.profile:command+=['--profile',args.profile,'--profile-manifest',args.profile_manifest]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'argv':command,'cwd':str(ROOT),'launcher_pid':os.getpid(),'started_unix':time.time(),'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'environment':{k:os.environ.get(k)for k in THREADS},'external_seconds':wall,'kill_grace_seconds':5,'monitored_group_rss_bytes':480*1024**2}
    save('launcher-start.json',meta);(out/'launcher.py').write_bytes(Path(__file__).read_bytes());proc=None;code=2;failures=[];maxrss=0;logbytes=0;observed=set()
    try:
        proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        save('owned-process.json',{'process_group_pid':proc.pid});selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        with(out/'process.log').open('xb')as log:
            while True:
                members=group_members(proc.pid);observed.update(p for p,_ in members);rss=sum(r for _,r in members);maxrss=max(maxrss,rss)
                failure=None
                if rss>480*1024**2:failure='group RSS monitor cap'
                if time.monotonic()-start>wall+6:failure='supervisor timeout including inherited pipe'
                if failure:
                    failures.append(failure)
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    break
                for key,_ in selector.select(.05):
                    chunk=os.read(key.fileobj.fileno(),65536)
                    if not chunk:selector.unregister(key.fileobj);continue
                    remain=128*1024-logbytes;log.write(chunk[:remain]);log.flush();logbytes+=min(remain,len(chunk))
                    if len(chunk)>remain:
                        failures.append('log cap');os.killpg(proc.pid,signal.SIGKILL)
                if proc.poll() is not None and not selector.get_map():break
            log.flush();os.fsync(log.fileno())
        code=proc.wait(timeout=2)
    except BaseException as exc:failures.append(type(exc).__name__+': '+str(exc))
    finally:
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();live=group_members(proc.pid)
        else:live=[]
        if live:failures.append('owned group remains')
        total=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
        if total>17*1024**2:failures.append('aggregate byte cap')
        if failures:code=2
        meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,max_group_rss_bytes=maxrss,observed_owned_pids=sorted(observed),owned_live_pids=live,failures=failures,log_bytes=logbytes,output_bytes_before_receipt=total)
        save('launcher-result.json',meta);inv={str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv,'total_bytes':sum(v['bytes']for v in inv.values())})
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json')}),flush=True);return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','panel'],required=True);p.add_argument('--out',required=True);p.add_argument('--profile');p.add_argument('--profile-manifest');a=p.parse_args()
    if a.mode=='panel'and(not a.profile or not a.profile_manifest):p.error('pinned profile required')
    raise SystemExit(launch(a))
