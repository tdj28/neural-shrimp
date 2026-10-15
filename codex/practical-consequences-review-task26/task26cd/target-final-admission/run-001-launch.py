"""One Task26cd saved-data calculation; prepared supervisor."""
from pathlib import Path
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,sys,time,traceback
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
B=Path('/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26cd')
OUT=B/'run-001';SUP=OUT.with_name(OUT.name+'-supervision')
PREFIX='codex/practical-consequences-review-task26/task26cd/'
WRAPPER=PREFIX+'run.py';STATS=PREFIX+'statistics.py';MAP=PREFIX+'map_evidence.py';CARD=PREFIX+'run-001-card.json'
WRAPPER_SHA='ae8984f0ede44a33a47b8e805139dfaa4de90989fb4806b965f0a36b2fdcd826';STATS_SHA='ba48cbca6df70be8581a161a1fa6fc57a884dc92bad126cf24ddb2bd0e7ba904';MAP_SHA='7c288d51d8b6b28156a119d7618e37e9b6d6c16af0fcf0d3c2f164e5d4f68a1b'
ADMISSION='reviewed';CARD_SHA='78b32dacebc9d5ecbc7038c0eb753d14dae79bf51f7e14a12f11922401dd5112'  # Coordinator binds only the exact reviewed card after private preparation.
GIB=1024**3;WALL=600;MAX_OUTPUT=128*1024**2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
 with p.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def identity(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split();s=dict(x.split(':',1)for x in(p/'status').read_text().splitlines()if ':'in x)
 return {'pid':pid,'start':int(f[19]),'pgid':int(f[2]),'sid':int(f[3]),'uid':int(s['Uid'].split()[0]),'nice':int(f[16]),'threads':int(s['Threads']),'limits':(p/'limits').read_text(),'command':[os.fsdecode(x)for x in(p/'cmdline').read_bytes().split(b'\0')if x]}
def groups_absent(row):
 assert not Path('/proc',str(row['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert int(f[2])!=row['pgid'] and int(f[3])!=row['sid'],('audit group remains',p.name)
def child_limits():
 resource.setrlimit(resource.RLIMIT_AS,(4*GIB,)*2);resource.setrlimit(resource.RLIMIT_CPU,(580,580))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);args=ap.parse_args()
 assert ADMISSION=='reviewed' and all(isinstance(pin,str) and len(pin)==64 and all(c in '0123456789abcdef'for c in pin)for pin in (CARD_SHA,)),'prepared Task26cd saved-data supervisor cannot run'
 assert len(args.source)==40 and all(c in '0123456789abcdef'for c in args.source)
 os.nice(10-os.getpriority(os.PRIO_PROCESS,0));resource.setrlimit(resource.RLIMIT_AS,(4*GIB,)*2)
 assert not OUT.exists() and not SUP.exists()
 def git(*words):return subprocess.check_output(['git','-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m',*words],cwd=R,timeout=30)
 assert git('rev-parse','HEAD').decode().strip()==args.source
 assert not git('status','--porcelain','--untracked-files=all').strip()
 remote=git('ls-remote','--exit-code','origin','refs/heads/main').decode().split()[0];git('merge-base','--is-ancestor',args.source,remote)
 sources={}
 for name,pin in [(WRAPPER,WRAPPER_SHA),(STATS,STATS_SHA),(MAP,MAP_SHA),(CARD,CARD_SHA)]:
  data=git('show',args.source+':'+name);assert hashlib.sha256(data).hexdigest()==pin==sha(R/name),name;sources[name]=data
 own=Path(__file__);launch_rel='codex/practical-consequences-review-task26/task26cd/run-001-launch.py';assert git('show',args.source+':'+launch_rel)==own.read_bytes();sources[launch_rel]=own.read_bytes()
 card=json.loads(sources[CARD]);assert card['admission']=='reviewed' and card['source_sha256']=={'run.py':WRAPPER_SHA,'statistics.py':STATS_SHA,'map_evidence.py':MAP_SHA}
 assert card['resources']=={'worker_cpu_slots':1,'supervisor_cpu_slots':1,'nice':10,'address_space_bytes':4*GIB,'wall_seconds':600,'cpu_seconds':580,'max_output_bytes':MAX_OUTPUT,'startup_mem_available_bytes':20*GIB,'runtime_mem_available_bytes':16*GIB}
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'));assert available>=20*GIB and shutil.disk_usage('/data2').free>=16*GIB
 SUP.mkdir();src=SUP/'source'
 for name,data in sources.items():p=src/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o444)
 command=['/data2/neural-shrimp/.venv/bin/python','-I','-B','-u',str(src/WRAPPER),'--card',str(src/CARD),'--out',str(OUT)]
 save(SUP/'admission.json',{'source_commit':args.source,'private_remote_ancestor':remote,'sources':{n:hashlib.sha256(d).hexdigest()for n,d in sources.items()},'available_bytes':available,'disk_free_bytes':shutil.disk_usage('/data2').free,'load_average':os.getloadavg(),'cpu_affinity_count':len(os.sched_getaffinity(0)),'scope':'One Task26cd independent saved-data calculation on the four frozen original MECH partitions and original24-cell XO. Exact5353 alpha and6837 bias cells, all eight1000-step stored arrays, original1500-step summaries kept distinct. Shared seed0 order: R1 network/readout, six R2 fields, four original XO contrasts. R4 same-temperature factor rows are descriptive. Complete cell/window/exclusion ledgers and all producer discrepancies retained; computational completion is independent of scientific scores. No model/derivative/producer execution, later XO7 or new experiment. CPU1 plus supervisor,4GiB hard AS,580CPU/600s wall plus5s TERM/5s KILL cleanup,128MiB sampled output,20/16GiB available RAM and16GiB disk. One attempt, no retry/extension. Closed synthetic profile supports the measured resource card; target decompression/final comparisons are bounded additional work, not a worst-case guarantee. Prepared admission/missing card pin refuses before capture or launch.','command':command,'seconds':WALL,'memory_gib':4,'cpu_worker':1,'supervisor_slots':1,'output_cap_bytes':MAX_OUTPUT,'supervisor':identity(os.getpid())})
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
 child=None;birth=None;failure=None;code=None;timed_out=False;closed=False;start=time.monotonic();max_bytes=0
 def stopped(signum,frame):raise InterruptedError('owned audit supervisor interrupted')
 old_handlers={s:signal.signal(s,stopped)for s in (signal.SIGTERM,signal.SIGINT)}
 def size():
  return (sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())if OUT.exists()else 0)+(SUP/'stdout.log').stat().st_size+(SUP/'stderr.log').stat().st_size
 try:
  with(SUP/'stdout.log').open('xb')as out,(SUP/'stderr.log').open('xb')as err:
   child=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=child_limits)
   birth=identity(child.pid);assert birth['pgid']==birth['sid']==child.pid
   # Popen returns after exec; exact argv may become visible shortly afterward.
   deadline=min(start+WALL,time.monotonic()+5)
   while birth['command']!=command:
    assert time.monotonic()<deadline;time.sleep(.01);fresh=identity(child.pid);assert fresh['start']==birth['start'];birth=fresh
   assert birth['uid']==os.getuid() and birth['nice']==10 and birth['threads']==1
   assert '4294967296' in birth['limits']
   save(SUP/'worker.json',birth);print(json.dumps({'worker':birth,'output':str(OUT)}),flush=True)
   while child.poll() is None:
    if time.monotonic()-start>=WALL:timed_out=True;raise TimeoutError('600s saved-data calculation deadline')
    assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=16*GIB,'shared RAM floor'
    count=size();max_bytes=max(max_bytes,count)
    if count>MAX_OUTPUT:raise RuntimeError('saved audit output cap')
    time.sleep(.5)
   code=child.wait();groups_absent(birth);closed=True
   count=size();max_bytes=max(max_bytes,count);assert count<=MAX_OUTPUT,'final audit output cap'
   assert code==0,'saved audit nonzero exit'
   expected={'amap-a-cells.jsonl','amap-b-cells.jsonl','bmap-a-cells.jsonl','bmap-b-cells.jsonl','report.json','manifest.json'}
   assert OUT.exists() and set(card['expected_output_files'])==expected
   assert {p.name for p in OUT.iterdir()}==expected and all(p.is_file() for p in OUT.iterdir())
   result=json.loads((OUT/'report.json').read_text());assert result['computational_status']=='complete'
   assert result['coverage']=={'alpha_cells':5353,'bias_cells':6837,'XO_cells':24,'XO_windows_per_cell_per_arm':4,'map_stored_steps_per_cell':1000,'map_full_record_steps':1500}
   manifest=json.loads((OUT/'manifest.json').read_text());rows=manifest['files']
   assert len(rows)==5 and {x['path'] for x in rows}==expected-{'manifest.json'}
   for row in rows:assert (OUT/row['path']).stat().st_size==row['bytes'] and sha(OUT/row['path'])==row['sha256']
   for name,pin in [(WRAPPER,WRAPPER_SHA),(STATS,STATS_SHA),(MAP,MAP_SHA),(CARD,CARD_SHA)]:assert sha(R/name)==pin
 except BaseException as exc:
  for signum in old_handlers:signal.signal(signum,signal.SIG_IGN)
  failure={'error':repr(exc),'traceback':traceback.format_exc()}
  if child is not None and child.poll() is None:
   # This source spawns no descendants; retain its fresh session/group identity.
   child.terminate()
   try:code=child.wait(timeout=5)
   except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=5)
  if child is not None:
   code=child.returncode
   if birth is None:raise
   groups_absent(birth);closed=True
 finally:
  report={'exit_code':code,'timed_out':timed_out,'seconds':time.monotonic()-start,'worker':birth,'owned_worker_and_groups_closed':closed,'failure':failure,'maximum_output_bytes_sampled':max_bytes,'source_commit':args.source,'wrapper_sha256':WRAPPER_SHA,'card_sha256':CARD_SHA,'certificate_passed':False,'experimental_claims':False}
  save(SUP/'closure.json',report)
  if child is None or closed:
   for base in [OUT,SUP]:
    if not base.exists():continue
    rows=[{'path':str(p.relative_to(base)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(base.rglob('*'))if p.is_file()]
    save(base/'final-manifest.json',{'files':rows})
    for p in base.rglob('*'):
     if p.is_file():p.chmod(0o444)
  print(json.dumps(report),flush=True)
  for signum,handler in old_handlers.items():signal.signal(signum,handler)
 return 0 if failure is None else 1
if __name__=='__main__':sys.exit(main())
