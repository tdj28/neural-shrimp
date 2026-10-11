"""Bounded independent saved-only fixed-gauge endpoint intersection audit."""
import argparse,gc,json,math,os,platform,resource,signal,subprocess,sys,time,hashlib
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from neural_shrimp.rigorous_orbits import canonical_bytes
import review_arclength_connection as O
import review_endpoint_localization_math as R
M=R.M
BASE=Path('/data2/neural-shrimp-data/audits/s2-period18-endpoint-localization')
RAW=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-endpoint-localization-2026-10-10/attempt-001-target')
OUTER='b3f04834c3ca63faacd8914b066521f604f6cf6f34103548ea12bc6012c210bb'
WORKER='78da8e09a2e55d23a6b5f09a0bc5430a74e20bbd534cfe08de54b51a074a1ce8'
EXECUTION='f2df07b026117e66e6ad726394fdbca7893e1ef5'
PRIOR=Path('/data2/neural-shrimp-data/audits/s2-period18-reboxed-chain/unit-001')
PRIOR_SHA='c1bae707c0912e6a1ef3243aebc641f6dc367d6e305dafb3599f34d77d677d72'
FILES=list(dict.fromkeys(O.FILES+['tools/reviews/review_localization_math.py','tools/reviews/review_endpoint_localization_math.py','tools/reviews/review_s2_endpoint.py','tests/reviews/test_endpoint_localization_review.py','docs/plans/s2-endpoint-independent-review.md']))

def bind(tick):
 O.verify(RAW,OUTER);O.verify(RAW/'run',WORKER);O.verify(PRIOR,PRIOR_SHA);tick()
 prior=O.read(PRIOR/'run/outcome.json');M.need(prior['passed'] and prior['total_qualified_charts']==127,'audited127-chart ancestry')
 state=O.read(PRIOR/'run/endpoint.json');run=RAW/'run';rt=O.read(run/'runtime-start.json');done=O.read(run/'outcome.json');la=O.read(RAW/'launcher-result.json')
 M.need(rt['source_commit']==la['source_commit']==EXECUTION and la['source_sha256']==rt['source_sha256'],'exact committed producer')
 for n,h in rt['source_sha256'].items():M.need(O.entry(run/'source'/n)['sha256']==h and hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+n],cwd=ROOT)).hexdigest()==h,'source identity '+n)
 M.need(rt['python']==platform.python_version() and rt['python_flint']==flint.__version__ and rt['flint']==flint.__FLINT_VERSION__,'arithmetic versions')
 M.need(rt['threads']==dict.fromkeys(O.THREADS,'1') and rt['nice']>=10 and rt['cpu_only'] and rt['precision']==256,'runtime scope')
 M.need(la['exit_code']==done['exit_code']==0 and done['passed'] and done['status']=='known_endpoint_localized' and not la['owned_live_pids'],'completed exact target')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
 M.need(done['completed_intersections']==3 and done['stages']==4 and done['adapter_counts']==dict(map_attempted=72,map_returned=72,unit_attempted=144,unit_returned=144,map_cap=72,unit_cap=144),'fixed complete acquisition')
 for v,cap in((done['elapsed_seconds'],30),(done['max_rss_kib'],2*1024**2),(la['elapsed_seconds'],52)):M.need(type(v)in(int,float) and math.isfinite(v) and 0<=v<cap,'resource gate')
 cost=O.read(run/'cost-gate.json');profile=RAW.parent/'attempt-001-profile/run';O.verify(profile,cost['manifest_sha256']);pr=O.read(profile/'runtime-start.json');pd=O.read(profile/'outcome.json')
 M.need(pr['source_sha256']==rt['source_sha256'] and pr['mode']=='profile' and pd['exit_code']==0 and pd['completed_intersections']==3,'completed identical-source profile')
 M.need(cost['projected_seconds']==2*pd['elapsed_seconds']+5<30 and cost['projected_bytes']<112*1024**2,'profile cost admission')
 inp=O.read(run/'inputs.json');M.need(inp['previous']==state['original'],'whole audited predecessor unchanged')
 model=O.read(Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-reboxing-2026-10-10/unit-001/run/inputs.json'))['model_inputs'];M.need(inp['model_inputs']==model,'unchanged native model/data')
 return run,state

def run_worker(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
 M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1 nice10')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False}
 def save(n,v):
  b=canonical_bytes(v);M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<16*1024**2,'16MiB cap');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stop[0] and time.monotonic()-start<45 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'time/RSS cap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,45)
 save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n)for n in FILES},'pid':os.getpid(),'threads':dict.fromkeys(O.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  run,state=bind(tick);initial=O.read(run/'stage-000-localization_start.json.gz');r=O.read(run/'result.json')
  M.need(r['stage_files']==['stage-000-localization_start.json.gz']+[f'stage-{i:03d}-intersection.json.gz'for i in range(1,4)],'exact stage inventory')
  def rows():
   for i in range(1,4):
    row=O.read(run/f'stage-{i:03d}-intersection.json.gz');tick();yield row;del row;gc.collect();tick()
  for prec in(256,384):
   v=R.audit(state['original'],initial,rows(),r,prec,state['independent384_endpoint'] if prec==384 else None);tick();save(f'reduction-{prec}.json',v)
  result.update(passed=True,producer_outer=OUTER,prior_audit=PRIOR_SHA,exact_gauge=r.get('exact_gauge','253/64'),final_max_radius=r['final_max_radius'],scope='same primitive18 endpoint; original256 arithmetic identities and384 inherited-root intersection reduction; no derivative reacquisition, attraction, extension or fold identity')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
 print(json.dumps(result),flush=True);return 0 if result['passed']else 2

def launch():
 out=BASE/'attempt-001';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
 M.need(int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))>3*1024**2,'host headroom')
 def save(n,v):
  with(out/n).open('xb')as f:f.write(canonical_bytes(v));f.flush();os.fsync(f.fileno())
 command=['timeout','--signal=TERM','--kill-after=5s','60s',sys.executable,str(Path(__file__).resolve()),'--worker','--out',str(out/'run')]
 meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'launcher_pid':os.getpid(),'argv':command,'started_unix':time.time(),'load':os.getloadavg(),'internal_seconds':45,'external_seconds':60,'kill_grace_seconds':5}
 save('launcher-start.json',meta)
 with(out/'process.log').open('xb')as log:
  try:proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=67)
  except BaseException as e:meta['failure']=type(e).__name__+': '+str(e)
  finally:
   if proc is not None:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
 live=[]
 if proc is not None:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    f=(p/'stat').read_text().rsplit(')',1)[1].split()
    if int(f[2])==proc.pid and f[0]!='Z':live.append(int(p.name))
   except(OSError,ValueError):pass
 if live or sum(p.stat().st_size for p in out.rglob('*')if p.is_file())>17*1024**2:code=2
 meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
 inv={str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv})
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps({'root':str(out),'exit_code':code,'manifest':O.entry(out/'manifest.json')}),flush=True);return code


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args()
 raise SystemExit(run_worker(a.out)if a.worker else launch())
