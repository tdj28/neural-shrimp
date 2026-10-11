#!/usr/bin/env python3
"""Fixed8 later-chart, at-most64-cell owned complementary-proof scheduler."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from neural_shrimp.complement_partition import cell,children,next_action,coverage
import launch_native705_complement_halves as L
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/later001')
CARD='docs/plans/native705-later-complement-inputs.json'
LIMITS=dict(attempts=64,concurrent=4,worker_seconds=400,owned_seconds=415,grace_seconds=5,worker_rss_kib=4*1024**2,total_rss_kib=16*1024**2,parent_rss_kib=256*1024,worker_raw_bytes=384*1024**2,total_raw_bytes=24*1024**3,total_seconds=10800,available_kib=24*1024**2,depth=3)
entry=L.entry

def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(p.read_text())
def save(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def raw_bytes(root):return sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
def available():return int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))
def seal(root):
    files={str(p.relative_to(root)):entry(p)for p in sorted(root.rglob('*'))if p.is_file()};save(root,'manifest.json',dict(files=files,total_bytes=sum(v['bytes']for v in files.values())))
    for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    root.chmod(0o555)
def verify_worker(root):
    m=read(root/'manifest.json');actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p.name!='manifest.json'}
    need(actual==set(m['files']),'complete literal worker inventory')
    for n,v in m['files'].items():need(entry(root/n)==v,'literal retained worker '+n)
    return entry(root/'manifest.json')
def make_assignment(index,row,c,source):
    return dict(attempt=index,ordinal=row['ordinal'],cell=c,source_commit=source,root=str(BASE/f'chart{index:03d}/run'),input_index=entry(ROOT/CARD),bounds={k:LIMITS[k]for k in ['worker_seconds','worker_rss_kib','worker_raw_bytes']}|dict(native_calls=360))
def classify(done,job,owner):
    need(done['assignment']==job and done['unit']==job['attempt']and owner['exit_code']==(0 if done['passed']else 2),'same complete worker assignment and mathematical exit')
    return next_action(done,job['cell'])
def stop(j):
    p=j['proc']
    if p.poll()is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait();j['log'].close()
def close(j,source):
    stop(j);folder=j['folder'];run=folder/'run';live,_=L.group(j['proc'].pid)
    owner=dict(group_pid=j['proc'].pid,exit_code=j['proc'].returncode,owned_live_pids=live,worker_pid_absent=not Path('/proc',str(j['proc'].pid)).exists(),peak_rss_kib=j['peak'],seconds=time.monotonic()-j['start'])
    save(folder,'launcher-result.json',owner);decision=dict(assignment=j['job'],owner=owner,action='stop')
    try:
        need(not live and owner['worker_pid_absent'],'exact owned worker gone')
        decision['worker']=verify_worker(run);done=read(run/'outcome.json');runtime=read(run/'runtime.json');need(runtime['source_commit']==source and runtime['assignment']==j['job'],'same exact worker source/card')
        need(all(entry(ROOT/n)==v and entry(run/'source'/n)==v for n,v in runtime['sources'].items()),'unchanged full archived worker source')
        decision['outcome']=done;decision['action']=classify(done,j['job'],owner)
        if decision['action']in ['accept','split']:
            need(read(run/'profile.json')['passed']and len(list((run/'native').glob('*-input.json')))==360 and len(list((run/'native').glob('*-return.json')))==360,'complete measured admission and360 retained actions')
            binding=read(run/'bindings.json');proof=read(run/'complement.json');need(binding['assignment']==j['job']and binding['cell']==j['job']['cell']and proof['passed']==done['passed']and proof['metric']['passed']==done['metric_passed'],'literal full11 theorem result')
            decision['proof']=entry(run/'complement.json');decision['binding']=entry(run/'bindings.json')
    except BaseException as e:decision.update(action='stop',error=type(e).__name__+': '+str(e))
    save(folder,'decision.json',decision);seal(folder);decision['outer']=entry(folder/'manifest.json');return decision

def main():
    need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();card=read(ROOT/CARD);need([r['ordinal']for r in card['charts']]==list(range(31,39)),'fixed8 later original chart index')
    need(card['limits']==LIMITS and card['status']=='reviewed-packet','fixed committed adaptive caps');need(available()>=LIMITS['available_kib'],'24GiB current availability for16GiB workers plus headroom')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,6*1024**3));BASE.mkdir(parents=True,exist_ok=False);start=time.monotonic();active=[];decisions=[];sealed=0;peak=0;parentpeak=0;failure=None;index=0
    queue=[(r['ordinal'],cell(r['producer']['settings']['parameter'],str(F(2)**r['producer']['settings']['halfwidth_exponent']),r['initial_depth'],i))for r in card['charts']for i in range(2**r['initial_depth'])];accepted={r['ordinal']:[]for r in card['charts']};blocked=False
    save(BASE,'launcher-start.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,input_index=entry(ROOT/CARD),mem_available_kib=available(),load=os.getloadavg(),threads=dict.fromkeys(L.THREADS,'1'),started_unix=time.time()));(BASE/'launcher.py').write_bytes(Path(__file__).read_bytes());save(BASE,'fixed-index.json',card)
    env=dict(os.environ,**dict.fromkeys(L.THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
    def halt(*_):raise TimeoutError('owned adaptive packet interrupted')
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGALRM,halt);signal.setitimer(signal.ITIMER_REAL,10800)
    def check():
        nonlocal peak,parentpeak
        total=0
        for j in active:
            _,used=L.group(j['proc'].pid);j['peak']=max(j['peak'],used);total+=used
            need(used<LIMITS['worker_rss_kib']and time.monotonic()-j['start']<415,'individual4GiB/415s owned cap')
            need((j['folder']/'process.log').stat().st_size<128*1024,'128KiB worker log')
        peak=max(peak,total);parentpeak=max(parentpeak,L.rss());need(total<16*1024**2 and parentpeak<256*1024,'aggregate16GiB/parent256MiB')
        need(time.monotonic()-start<10800,'10800s total bound');need(sealed+sum(raw_bytes(j['folder'])for j in active)<24*1024**3-64*1024**2,'24GiB raw with64MiB closing reserve')
    try:
        while active or(queue and not blocked and index<64):
            check()
            while queue and not blocked and len(active)<4 and index<64:
                if time.monotonic()-start>10800-430 or sealed+(len(active)+1)*388*1024**2>=24*1024**3-64*1024**2:blocked=True;failure='remaining resource reserve reached';break
                # Available memory already excludes the RSS of running workers;
                # reserve their unused envelopes before another4GiB acquisition.
                reserved=sum(max(0,4*1024**2-L.group(j['proc'].pid)[1])for j in active)
                need(available()>=reserved+10*1024**2,'fresh remaining reservation and6GiB headroom')
                ordinal,c=queue.pop(0);index+=1;row=card['charts'][ordinal-31];job=make_assignment(index,row,c,source);folder=BASE/f'chart{index:03d}';folder.mkdir();save(folder,'attempt-card.json',job)
                log=(folder/'process.log').open('xb');cmd=[sys.executable,str(ROOT/'tools/probe_native705_later_complement.py'),'--attempt',str(index)]
                try:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:log.close();raise
                j=dict(proc=p,folder=folder,log=log,job=job,start=time.monotonic(),peak=0);active.append(j);save(folder,'owned-process.json',dict(group_pid=p.pid,command=cmd))
            completed=[j for j in active if j['proc'].poll()is not None]
            for j in completed:
                d=close(j,source);active.remove(j);sealed+=raw_bytes(j['folder']);decisions.append(d);ordinal=j['job']['ordinal'];c=j['job']['cell']
                if d['action']=='accept':accepted[ordinal].append(c)
                elif d['action']=='split':
                    r=card['charts'][ordinal-31];queue[0:0]=[(ordinal,cell(r['producer']['settings']['parameter'],str(F(2)**r['producer']['settings']['halfwidth_exponent']),dep,idx))for dep,idx in children(c)]
                else:blocked=True;failure=f'complete retained attempt{j["job"]["attempt"]:03d} requires diagnosis'
            if active:time.sleep(.25)
    except BaseException as e:failure=type(e).__name__+': '+str(e);blocked=True
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        for j in active:
            d=close(j,source);decisions.append(d);sealed+=raw_bytes(j['folder'])
            if d['action']=='accept':accepted[j['job']['ordinal']].append(j['job']['cell'])
    covers={str(r['ordinal']):coverage(r['producer']['settings']['parameter'],str(F(2)**r['producer']['settings']['halfwidth_exponent']),accepted[r['ordinal']])for r in card['charts']};passed=all(v['complete']for v in covers.values())and failure is None
    result=dict(passed=passed,source_commit=source,launched=index,attempts=decisions,coverage=covers,complete_charts=sum(v['complete']for v in covers.values()),pending=queue,failure=failure,seconds=time.monotonic()-start,peak_group_rss_kib=peak,parent_sampled_peak_rss_kib=parentpeak,owned_live_pids=[p for d in decisions for p in d['owner']['owned_live_pids']],native_call_ceiling=index*360,scope='Producer full11 complementary stability only on accepted closed cells of8 previously independently connected quadratic/cubic705 roots. All attempts retained; exact per-chart unions and original joins, no cross-gauge scalar interpolation. Independent arithmetic replay remains required; no full-window or critical-pair attraction claim.')
    save(BASE,'launcher-result.json',result);seal(BASE);print(json.dumps(dict(root=str(BASE),passed=passed,attempts=index,complete_charts=result['complete_charts'],manifest=entry(BASE/'manifest.json'))));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
