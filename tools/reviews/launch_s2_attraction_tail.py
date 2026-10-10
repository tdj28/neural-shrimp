"""One fixed two-chart batch with separate owned groups and combined caps."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/s2-attraction-remaining/batch004')
CHARTS=(155,156)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')

def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def save(root,name,value):
    with (root/name).open('xb') as f:f.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
def processes(groups):
    answer={g:{'live':[],'rss_kib':0} for g in groups}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split();g=int(fields[2])
            if g not in answer or fields[0]=='Z':continue
            rss=int(next((v.split()[1] for v in (p/'status').read_text().splitlines() if v.startswith('VmRSS:')),'0'))
            answer[g]['live'].append(int(p.name));answer[g]['rss_kib']+=rss
        except (OSError,ValueError):pass
    return answer

def current_rss_kib(status=None):
    # Linux getrusage historical peak can survive exec of a larger tool parent.
    status=Path('/proc/self/status').read_text() if status is None else status
    value=int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))
    if value<0:raise ValueError('negative actual RSS')
    return value

def freeze(root):
    files={str(p.relative_to(root)):entry(p) for p in sorted(root.rglob('*')) if p.is_file()}
    save(root,'manifest.json',{'files':files,'total_bytes':sum(v['bytes']for v in files.values())})
    for p in sorted(root.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)

def main():
    assert not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
    assert available>6*1024**2
    # Preserve the reviewed3GiB hard ceiling so workers can set their own soft limit.
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,3*1024**3))
    OUT.mkdir(parents=True,exist_ok=False);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();start=time.monotonic();jobs=[];failure=None;peak=0;launcher_peak=0
    meta={'source_commit':source,'launcher_pid':os.getpid(),'charts':list(CHARTS),'started_unix':time.time(),'mem_available_kib':available,'load':os.getloadavg(),'workers':2,'internal_seconds_each':240,'external_seconds_each':260,'kill_grace_seconds':5,'worker_rss_kib_each':2*1024**2,'combined_worker_rss_kib':4*1024**2,'launcher_rss_kib':256*1024,'aggregate_output_bytes':35*1024**2,'threads_each':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)}
    save(OUT,'launcher-start.json',meta);(OUT/'launcher.py').write_bytes(Path(__file__).read_bytes());env=dict(os.environ,**dict.fromkeys(THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
    try:
        for chart in CHARTS:
            root=OUT/f'chart{chart}';root.mkdir();cmd=[sys.executable,str(ROOT/'tools/reviews/review_s2_attraction_tail.py'),'--out',str(root/'run'),'--chart',str(chart)]
            log=(root/'process.log').open('xb');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job={'chart':chart,'root':root,'process':p,'log':log,'start':time.monotonic(),'peak':0};jobs.append(job)
            save(root,'launcher-start.json',{'source_commit':source,'parent_launcher_pid':os.getpid(),'argv':cmd,'owned_group_pid':p.pid,'threads':dict.fromkeys(THREADS,'1'),'nice':meta['nice']})
        while any(j['process'].poll()is None for j in jobs):
            launcher_peak=max(launcher_peak,current_rss_kib());now=time.monotonic();groups=processes([j['process'].pid for j in jobs]);total=sum(g['rss_kib']for g in groups.values());peak=max(peak,total)
            for j in jobs:
                j['peak']=max(j['peak'],groups[j['process'].pid]['rss_kib'])
                if j['peak']>2*1024**2:raise MemoryError('per-owned-group2GiB cap')
                if j['process'].poll()is None and now-j['start']>260:raise TimeoutError('per-worker260s cap')
            if total>4*1024**2 or launcher_peak>256*1024:raise MemoryError('combined4GiB/launcher256MiB cap')
            for j in jobs:
                if sum(p.stat().st_size for p in j['root'].rglob('*') if p.is_file())>17*1024**2:raise OSError('per-chart17MiB output cap')
            if now-start>265:raise TimeoutError('batch265s cap')
            time.sleep(.25)
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
    finally:
        for j in jobs:
            if j['process'].poll()is None:
                try:os.killpg(j['process'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+5
        while any(j['process'].poll()is None for j in jobs) and time.monotonic()<deadline:time.sleep(.05)
        for j in jobs:
            try:os.killpg(j['process'].pid,signal.SIGKILL)
            except ProcessLookupError:pass
            j['process'].wait();j['log'].close()
    final=processes([j['process'].pid for j in jobs]);rows=[]
    for j in jobs:
        p=j['process'];live=final[p.pid]['live'];code=p.returncode
        if live or sum(v.stat().st_size for v in j['root'].rglob('*') if v.is_file())>17*1024**2:code=2
        row={'source_commit':source,'chart':j['chart'],'exit_code':code,'owned_group_pid':p.pid,'owned_live_pids':live,'worker_pid_absent':not Path('/proc',str(p.pid)).exists(),'elapsed_seconds':time.monotonic()-j['start'],'max_group_rss_kib':j['peak'],'batch_failure':failure}
        save(j['root'],'launcher-result.json',row);freeze(j['root']);rows.append({'root':str(j['root']),'outer_manifest':entry(j['root']/'manifest.json'),**row})
    passed=len(rows)==2 and failure is None and all(v['exit_code']==0 and not v['owned_live_pids']and v['worker_pid_absent']for v in rows)
    meta.update(passed=passed,elapsed_seconds=time.monotonic()-start,maximum_combined_worker_rss_kib=peak,launcher_peak_sampled_rss_kib=launcher_peak,inherited_historical_rusage_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,failure=failure,rows=rows)
    save(OUT,'launcher-result.json',meta)
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>35*1024**2:raise OSError('batch35MiB cap')
    freeze(OUT);print(json.dumps({'root':str(OUT),'passed':passed,'outer_manifest':entry(OUT/'manifest.json')}),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
