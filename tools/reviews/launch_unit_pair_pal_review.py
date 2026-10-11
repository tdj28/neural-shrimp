"""Dedicated owned-process launcher for one saved independent705 audit."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/native705-independent/attempt001')

def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def group(g):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(a[2])==g and a[0]!='Z':rows.append((int(p.name),int(a[21])*os.sysconf('SC_PAGE_SIZE')))
        except(OSError,ValueError):pass
    return rows

def main():
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10')
    mem=Path('/proc/meminfo').read_text();available=int(next(x.split()[1]for x in mem.splitlines()if x.startswith('MemAvailable:')))*1024
    if available<8*1024**3:raise MemoryError('8GiB headroom')
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=['timeout','--signal=TERM','--kill-after=5s','260s',sys.executable,str(ROOT/'tools/reviews/review_unit_pair_pal_driver.py')]
    def save(n,v):
        with(OUT/n).open('x')as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    rec=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=command,meminfo=mem,load=os.getloadavg(),started_unix=time.time(),peak_group_rss_bytes=0)
    save('launcher-start.json',rec);start=time.monotonic();child=None;code=2
    try:
        with(OUT/'process.log').open('xb')as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',dict(pid=child.pid))
            while child.poll()is None:
                rss=sum(v[1]for v in group(child.pid));rec['peak_group_rss_bytes']=max(rec['peak_group_rss_bytes'],rss)
                if rss>4*1024**3 or time.monotonic()-start>267 or(OUT/'process.log').stat().st_size>1024**2:raise RuntimeError('owned resource cap')
                time.sleep(.1)
            code=child.returncode
    except BaseException as exc:rec['failure']=type(exc).__name__+': '+str(exc)
    finally:
        if child is not None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
        live=[]if child is None else group(child.pid);rec.update(exit_code=code,seconds=time.monotonic()-start,owned_live_pids=[v[0]for v in live]);save('launcher-result.json',rec)
        files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()};save('manifest.json',dict(files=files))
        for p in sorted(OUT.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        OUT.chmod(0o555)
    print(json.dumps(dict(root=str(OUT),code=code,owned_live_pids=rec['owned_live_pids'],manifest=entry(OUT/'manifest.json'))));return 0 if code==0 and not live and sum(v['bytes']for v in files.values())<65*1024**2 else 2
if __name__=='__main__':raise SystemExit(main())
