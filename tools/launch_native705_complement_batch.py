"""Four fixed independent owned complementary workers; no root acquisitions."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/batch001')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')

def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def group(g):
    live=[];rss=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            f=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[2])==g and f[0]!='Z':live.append(int(p.name));rss+=int(next((s.split()[1]for s in(p/'status').read_text().splitlines()if s.startswith('VmRSS:')),'0'))
        except(OSError,ValueError):pass
    return live,rss

def rss():return int(next(s.split()[1]for s in Path('/proc/self/status').read_text().splitlines()if s.startswith('VmRSS:')))
def main():
    if os.getpriority(os.PRIO_PROCESS,0)<10 or subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip():raise ValueError('clean nice10 committed source')
    available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))
    if available<24*1024**2:raise MemoryError('24GiB available required for reserved16GiB batch plus headroom')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,6*1024**3));BASE.mkdir(parents=True,exist_ok=False);start=time.monotonic();jobs=[];failure=None;peak=0;parent_peak=0
    def save(name,value):
        with(BASE/name).open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('launcher-start.json',dict(source_commit=source,launcher_pid=os.getpid(),units=[1,2,3,4],mem_available_kib=available,load=os.getloadavg(),started_unix=time.time(),per_worker_seconds=400,external_seconds=415,kill_grace_seconds=5,per_worker_rss_kib=4*1024**2,total_rss_kib=16*1024**2,parent_rss_kib=256*1024,per_worker_raw_bytes=128*1024**2,total_raw_bytes=520*1024**2,threads=dict.fromkeys(THREADS,'1')))
    (BASE/'launcher.py').write_bytes(Path(__file__).read_bytes());env=dict(os.environ,**dict.fromkeys(THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
    try:
        for unit in range(1,5):
            folder=BASE/f'chart{unit:03d}';folder.mkdir();log=(folder/'process.log').open('xb');cmd=[sys.executable,str(ROOT/'tools/probe_native705_complement_batch.py'),'--unit',str(unit)];proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(dict(unit=unit,folder=folder,log=log,proc=proc,peak=0,command=cmd));save(f'chart{unit:03d}/owned-process.json',dict(group_pid=proc.pid,command=cmd))
        while any(j['proc'].poll()is None for j in jobs):
            total=0
            for j in jobs:
                live,used=group(j['proc'].pid);j['peak']=max(j['peak'],used);total+=used
                if used>4*1024**2:raise MemoryError('owned individual4GiB cap')
            peak=max(peak,total);parent_peak=max(parent_peak,rss())
            if total>16*1024**2 or parent_peak>256*1024:raise MemoryError('combined16GiB/parent256MiB cap')
            if time.monotonic()-start>415:raise TimeoutError('owned415s batch cap')
            if sum(p.stat().st_size for p in BASE.rglob('*')if p.is_file())>520*1024**2:raise OSError('combined520MiB raw cap')
            time.sleep(.25)
    except BaseException as e:failure=type(e).__name__+': '+str(e)
    finally:
        # Only these four recorded process groups can be signalled.
        for j in jobs:
            if j['proc'].poll()is None:
                try:os.killpg(j['proc'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and any(j['proc'].poll()is None for j in jobs):time.sleep(.1)
        for j in jobs:
            try:os.killpg(j['proc'].pid,signal.SIGKILL)
            except ProcessLookupError:pass
            j['proc'].wait();j['log'].close()
    records=[]
    for j in jobs:
        p=j['folder']/'run/outcome.json';out=json.loads(p.read_text())if p.exists()else None;live,_=group(j['proc'].pid);record=dict(unit=j['unit'],group_pid=j['proc'].pid,exit_code=j['proc'].returncode,owned_live_pids=live,worker_pid_absent=not Path('/proc',str(j['proc'].pid)).exists(),peak_rss_kib=j['peak'],outcome=out);records.append(record);save(f'chart{j["unit"]:03d}/launcher-result.json',record)
    complete=len(records)==4 and not failure and all(x['exit_code']==0 and not x['owned_live_pids']and x['worker_pid_absent']and x['outcome']and x['outcome']['passed']and x['outcome']['native_calls']==216 for x in records)
    result=dict(passed=bool(complete),source_commit=source,elapsed_seconds=time.monotonic()-start,peak_group_rss_kib=peak,launcher_sampled_peak_rss_kib=parent_peak,failure=failure,workers=records,owned_live_pids=[p for x in records for p in x['owned_live_pids']],scope='Exactly four independently proved same-family charts. Per-chart complete complementary bounds only; no joining inference beyond authenticated independent parent proof, no automatic retry or subdivision.')
    save('launcher-result.json',result);files={str(p.relative_to(BASE)):entry(p)for p in sorted(BASE.rglob('*'))if p.is_file()};save('manifest.json',dict(files=files,total_bytes=sum(v['bytes']for v in files.values())))
    for p in sorted(BASE.rglob('*'),key=lambda x:len(x.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    BASE.chmod(0o555);print(json.dumps(dict(root=str(BASE),passed=complete,manifest=entry(BASE/'manifest.json'))));return 0 if complete else 2
if __name__=='__main__':raise SystemExit(main())
