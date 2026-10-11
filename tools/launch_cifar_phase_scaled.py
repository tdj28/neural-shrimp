"""One bounded native CIFAR phase-balanced corrector."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/corrector001')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def save(name,value):
    with(OUT/name).open('xb')as f:f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
def owned(group):
    live=[];rss=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=group or fields[0]=='Z':continue
            live.append(int(p.name));rss+=int(next((v.split()[1]for v in(p/'status').read_text().splitlines()if v.startswith('VmRSS:')),'0'))
        except(OSError,ValueError):pass
    return live,rss
def current_rss_kib():return int(next(v.split()[1]for v in Path('/proc/self/status').read_text().splitlines()if v.startswith('VmRSS:')))
def scientific_complete(row):
    caps=dict(action=0,j2vp=0,hvp=818,direct_gradient=40)
    a,b=row.get('operation_attempted',{}),row.get('operation_returned',{})
    if not(row.get('status')=='phase_scaled_pilot_complete'and row.get('comparison_qualified')is True and row.get('accepted_orbit')is False and row.get('spectrum_requested')is False and row.get('operation_caps')==caps and set(a)==set(b)==set(caps)and all(type(a[k])is int and 0<=b[k]==a[k]<=caps[k]for k in caps)and row.get('qualification',{}).get('passed')is True):return False
    linear=row.get('linear',{});trials=row.get('trials',[]);index=row.get('accepted_index')
    if not(0<=linear.get('iterations',401)<=400 and linear.get('scaled_actions',403)<=402 and linear.get('extra_phase_actions')==4 and len(trials)<=13):return False
    if trials and not(linear.get('forcing_pass')and linear.get('usable_descent')):return False
    if index is not None:return type(index)is int and 0<=index<len(trials)and index==len(trials)-1 and trials[index]['accepted_local_step']and row['last_accepted_state']==f'trial-{index}-theta.npy'
    return not any(v['accepted_local_step']for v in trials)and row['last_accepted_state']=='initial-theta.npy'

def main():
    assert not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')));assert available>10*1024**2
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,8*1024**3))
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;failure=None;peak=0;parent_peak=0
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();command=[sys.executable,str(ROOT/'tools/explore_cifar_phase_scaled.py'),'--out',str(OUT/'run'),'--review-commit',source]
    meta={'source_commit':source,'launcher_pid':os.getpid(),'argv':command,'load':os.getloadavg(),'mem_available_kib':available,'started_unix':time.time(),'internal_seconds':300,'external_seconds':320,'kill_grace_seconds':5,'worker_rss_kib':2*1024**2,'worker_as_bytes':8*1024**3,'launcher_rss_kib':256*1024,'output_cap_bytes':258*1024**2,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)}
    save('launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes());env=dict(os.environ,**dict.fromkeys(THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
    with(OUT/'process.log').open('xb')as log:
        try:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid})
            while proc.poll()is None:
                _,rss=owned(proc.pid);peak=max(peak,rss);parent_peak=max(parent_peak,current_rss_kib())
                if peak>2*1024**2 or parent_peak>256*1024:raise MemoryError('owned2GiB/launcher256MiB cap')
                if time.monotonic()-start>320:raise TimeoutError('external320s cap')
                if sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())>258*1024**2:raise OSError('combined258MiB output cap')
                time.sleep(.25)
        except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                if proc.poll()is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    live=[] if proc is None else owned(proc.pid)[0];code=2 if proc is None else proc.returncode
    if failure or live or sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())>258*1024**2:code=2
    completed=False
    try:completed=scientific_complete(json.loads((OUT/'run/result.json').read_text()))
    except (OSError,ValueError,KeyError,TypeError):pass
    try:
        inv=json.loads((OUT/'run/manifest.json').read_text())['files'];actual={p.name:entry(p)for p in (OUT/'run').iterdir()if p.is_file()and p.name!='manifest.json'}
        completed=completed and inv==actual and all(p.is_file()and not p.is_symlink()for p in(OUT/'run').iterdir())
    except(OSError,ValueError,KeyError,TypeError):completed=False
    if not completed:code=2
    meta.update(scientific_complete=completed,exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live,worker_pid_absent=proc is None or not Path('/proc',str(proc.pid)).exists(),maximum_group_rss_kib=peak,launcher_peak_sampled_rss_kib=parent_peak,failure=failure);save('launcher-result.json',meta)
    files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()};save('manifest.json',{'files':files,'total_bytes':sum(e['bytes']for e in files.values())})
    for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    OUT.chmod(0o555);print(json.dumps({'root':str(OUT),'exit_code':code,'manifest':entry(OUT/'manifest.json')}),flush=True);return code
if __name__=='__main__':raise SystemExit(main())
