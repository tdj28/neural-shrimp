"""One serial, bounded saved-audit lane. No producer code import or automatic retry."""
import argparse, hashlib, json, os, pathlib, resource, shutil, signal, subprocess, sys, time
P=pathlib.Path
GIB=1024**3
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def pin(p):
    p=P(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def verify(e):
    a=pin(e['path'])
    assert all(a[k]==e[k] for k in ('bytes','sha256')),e['path']
    return a
def write(p,v):
    with P(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def ident(pid):
    p=P('/proc')/str(pid)
    try:
        s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
        status=(p/'status').read_text();uid=[x for x in status.splitlines() if x.startswith('Uid:')][0]
        return {'pid':pid,'starttime':int(v[19]),'pgid':int(v[2]),'sid':int(v[3]),'state':v[0],
          'nice':int(v[16]),'threads':int(v[17]),'uid_line':uid,'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
          'limits':(p/'limits').read_text()}
    except FileNotFoundError:return None
    except ProcessLookupError:return None
def members(group):
    found=[]
    for p in P('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split()
            if int(v[2])==group or int(v[3])==group:found.append({'pid':int(p.name),'starttime':int(v[19]),'pgid':int(v[2]),'sid':int(v[3]),'state':v[0]})
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return found
def capacity(path):
    mem={k:int(v.split()[0])*1024 for k,v in (x.split(':',1) for x in P('/proc/meminfo').read_text().splitlines())}
    return {'available_ram_bytes':mem['MemAvailable'],'disk_free_bytes':shutil.disk_usage(path).free,'loadavg':list(os.getloadavg())}
def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--lane',required=True);ap.add_argument('--source',required=True);a=ap.parse_args()
    plan=json.loads(P(a.plan).read_text());assert plan['status']=='reviewed_for_execution'
    lane=plan['lanes'][a.lane];repo=P(plan['repository']);out=P(lane['output']);sup=P(lane['supervision'])
    assert not out.exists() and not sup.exists(),'fresh output required; no retry'
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
    assert git('status','--porcelain')=='','owned source checkout must be clean'
    assert git('rev-parse','HEAD')==a.source==git('rev-parse','origin/main'),'private checkpoint equality'
    blobs=[]
    for e in plan['source_files']:
        verify(e);rel=str(P(e['path']).relative_to(repo));b=subprocess.check_output(['git','show',a.source+':'+rel],cwd=repo)
        assert hashlib.sha256(b).hexdigest()==e['sha256'];blobs.append({'relative_path':rel,'sha256':e['sha256'],'bytes':len(b)})
    for e in lane['preflight_pins']:verify(e)
    if lane.get('input_inventory'):
        inv=json.loads(P(lane['input_inventory']['path']).read_text())
        for e in inv['files']:verify(e)
    # A serial run's prior receipt must show closure; unknown jobs are never touched.
    prior=[]
    for p in lane.get('prior_receipts',[]):
        v=json.loads(P(p).read_text());assert v['worker_reaped'] and v['worker_group_session_absent'];prior.append(pin(p))
    base=P(plan['raw_root']);base.mkdir(parents=True,exist_ok=True)
    cap=capacity(base);assert cap['available_ram_bytes']>=28*GIB and cap['disk_free_bytes']>=16*GIB,cap
    os.setpriority(os.PRIO_PROCESS,0,10);resource.setrlimit(resource.RLIMIT_AS,(GIB,12*GIB))
    sup.mkdir();out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[plan['python'],'-I','-B',lane['entry'],*lane['arguments']]
    env=os.environ.copy();env.update({k:'2' for k in THREADS});env['PYTHONDONTWRITEBYTECODE']='1'
    cpus=set(sorted(os.sched_getaffinity(0))[:2]);start=time.monotonic();proc=None;reason=None;observed=None
    write(sup/'admission.json',{'source_commit':a.source,'plan':pin(a.plan),'lane':a.lane,'argv':cmd,'capacity':cap,'supervisor':ident(os.getpid()),'source_blobs':blobs,'prior_receipts':prior,'limits':lane['limits'],'threads':{k:env[k] for k in THREADS}})
    def limits():
        os.setpriority(os.PRIO_PROCESS,0,10);os.sched_setaffinity(0,cpus)
        resource.setrlimit(resource.RLIMIT_AS,(lane['limits']['as_bytes'],lane['limits']['as_bytes']))
        resource.setrlimit(resource.RLIMIT_CPU,(lane['limits']['cpu_seconds'],lane['limits']['cpu_seconds']))
        resource.setrlimit(resource.RLIMIT_FSIZE,(lane['limits']['output_bytes'],lane['limits']['output_bytes']))
        write(sup/'worker-startup.json',{'identity':ident(os.getpid()),'planned_argv':cmd,'affinity':sorted(os.sched_getaffinity(0)),
          'address_space':resource.getrlimit(resource.RLIMIT_AS),'cpu_seconds':resource.getrlimit(resource.RLIMIT_CPU),'file_size':resource.getrlimit(resource.RLIMIT_FSIZE)})
    try:
        with (sup/'stdout.log').open('xb') as stdout,(sup/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
            while proc.poll() is None:
                if observed is None:
                    x=ident(proc.pid)
                    if x and lane['entry'] in x['cmdline']:observed=x
                elapsed=time.monotonic()-start;capnow=capacity(base);used=size(out)+(sup/'stdout.log').stat().st_size+(sup/'stderr.log').stat().st_size
                if elapsed>lane['limits']['wall_seconds']:reason='wall deadline'
                elif used>lane['limits']['output_bytes']:reason='sampled output cap'
                elif capnow['available_ram_bytes']<16*GIB:reason='runtime available RAM floor'
                elif capnow['disk_free_bytes']<16*GIB:reason='runtime disk floor'
                if reason:break
                time.sleep(1)
            if reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            else:proc.wait()
        if members(proc.pid):
            # Parent is already reaped: numeric group alone does not authorize signals.
            # These workers must not spawn children. Preserve unexpected membership.
            reason=reason or 'unexpected group/session members after parent exit; retained for exact ownership review'
        worker=json.loads((sup/'worker-startup.json').read_text())
        assert worker['identity']['nice']==10 and worker['identity']['threads']==1
        assert worker['address_space']==[lane['limits']['as_bytes']]*2 and worker['cpu_seconds']==[lane['limits']['cpu_seconds']]*2
        for e in plan['source_files']:verify(e)
        for e in lane['preflight_pins']:verify(e)
        if lane.get('input_inventory'):
            for e in inv['files']:verify(e)
    except BaseException as e:
        reason=reason or type(e).__name__+': '+str(e)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
        raise
    finally:
        files=[pin(p) for p in sorted(out.rglob('*')) if p.is_file()] if out.exists() else []
        rc=None if proc is None else proc.returncode
        write(sup/'receipt.json',{'lane':a.lane,'source_commit':a.source,'argv':cmd,'returncode':rc,'failure':reason,
          'external_elapsed_seconds':time.monotonic()-start,'worker_reaped':proc is not None and rc is not None,
          'worker_observed_after_exec':observed,'worker_pid':None if proc is None else proc.pid,
          'worker_identity_absent':proc is not None and ident(proc.pid) is None,
          'worker_group_session_absent':proc is not None and not members(proc.pid),'supervisor':ident(os.getpid()),
          'output_files':files,'actual_output_bytes':sum(x['bytes'] for x in files),
          'children_ru_maxrss_kib_at_reap':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
          'scope':'External window includes startup, pre/post checks and serialization; source/native producer execution is not attested. Scientific claims require separate saved-result review.'})
    if reason or proc.returncode!=0:raise SystemExit(1)
if __name__=='__main__':main()
