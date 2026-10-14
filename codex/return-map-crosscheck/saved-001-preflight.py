"""One saved-reducer source/input-byte/closed-ownership preflight; no numerical payload decode."""
import argparse,hashlib,importlib.util,json,os,resource,shutil,subprocess,time,traceback
from pathlib import Path
REPO=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
OWNED=Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
T18=Path('/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002')
CLOSURES=[(T18.with_name('run-002-closure-001')/'report.json','2d3a679a6dabde4379f95c56c66b303d9a82a5e1ddad6b76f4a3d7296c66b047'),(OWNED/'trajectory-001-closure-001/report.json','f30cae9c4ed82b6ed4c581995a46a1cf5b2825a3ee3549948f5853caca91a3b9')]
LAUNCHER_SHA='259e056e695dbe3aa5b9112a793e9272360371d4efd5f1833d12cdaec56a546c'
AUTHORITY_SHA='d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split();s=dict(x.split(':',1)for x in(p/'status').read_text().splitlines()if ':'in x)
 return dict(pid=pid,start=int(f[19]),ppid=int(f[1]),pgid=int(f[2]),sid=int(f[3]),uid=int(s['Uid'].split()[0]),nice=int(f[16]),threads=int(s['Threads']),state=f[0],rss_kib=int(s.get('VmRSS','0 kB').split()[0]),address_space_limits=next(x.split()[-3:-1]for x in(p/'limits').read_text().splitlines()if x.startswith('Max address space')),command=[os.fsdecode(x)for x in(p/'cmdline').read_bytes().split(b'\0')if x])
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--card-sha256',required=True);a=p.parse_args()
started=time.monotonic();os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));assert os.getpriority(os.PRIO_PROCESS,0)==10
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(60,65))
out=OWNED/'saved-001-preflight';out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'executed-preflight.py')
try:
 assert not(OWNED/'saved-001').exists() and not(OWNED/'saved-001-supervision').exists()
 assert not subprocess.check_output(['git','-C',str(REPO),'-c','core.preloadIndex=false','status','--porcelain','--untracked-files=all'],timeout=20)
 launcher=REPO/'codex/return-map-crosscheck/launch.py';assert sha(launcher)==LAUNCHER_SHA
 assert sha(REPO/'codex/return-map-crosscheck/saved-001-card.md')==a.card_sha256
 spec=importlib.util.spec_from_file_location('task19_reviewed_launcher',launcher);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 args=argparse.Namespace(attempt='saved-001',source=a.source,plan='saved-001-plan.json',plan_sha256=a.plan_sha256)
 m.validate_arguments(args);plan,blobs,remote=m.snapshot(REPO,args,started+55,[]);assert len(blobs)==10
 auth=OWNED/'input-authority-001/manifest.json';assert sha(auth)==AUTHORITY_SHA==plan['input_manifest_sha256']
 metadata=json.loads(auth.read_text());assert len(metadata['files'])==10
 input_rows=[]
 for row in metadata['files']:
  q=auth.parent/row['path'];assert q.is_relative_to(auth.parent) and not q.is_symlink();assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256'] and not q.stat().st_mode&0o222
  input_rows.append({k:row[k]for k in ['path','sha256','bytes']})
 closed=[]
 for path,pin in CLOSURES:
  assert sha(path)==pin and not path.stat().st_mode&0o222
  c=json.loads(path.read_text());assert c['passed']is True and c['original_waiter_launcher_worker_pids_groups_sessions_absent']is True
  assert c['uncertain_ownership']==[]
  for old in c['original_identities'].values():assert not Path('/proc',str(old['pid'])).exists()
  closed.append(dict(path=str(path),sha256=pin,source_commit=c['source_commit'],receipt_sha256=c['receipt_sha256'],tool_session_reaped_exit_code=c['tool_session_reaped_exit_code'],original_pids_absent=True))
 # Matching names never grant process-control authority; duplicates merely refuse admission.
 ancestors=set();pid=os.getpid()
 while pid>1:
  ancestors.add(pid)
  try:pid=proc(pid)['ppid']
  except OSError:break
 duplicates=[]
 for q in Path('/proc').iterdir():
  if not q.name.isdigit() or int(q.name) in ancestors:continue
  try:
   cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if any(s in cmd for s in ['llm-return-map-crosscheck/trajectory-001','llm-return-map-crosscheck/saved-001','llm-loop-gain-crosscheck/run-002/source/']):duplicates.append(dict(pid=int(q.name),command=cmd))
  except OSError:pass
 assert not duplicates
 mem=m.available();disk=shutil.disk_usage(OWNED).free;assert mem>=20*1024**3 and disk>=16*1024**3
 gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 result=dict(passed=True,source_commit=a.source,remote_main=remote,plan_sha256=a.plan_sha256,card_sha256=a.card_sha256,launcher_sha256=LAUNCHER_SHA,source_files=[dict(path=k,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v))for k,v in blobs.items()],input_authority_sha256=AUTHORITY_SHA,input_files=input_rows,input_values_decoded=False,original_closed_authorities=closed,no_duplicate_task19_processes=True,mem_available_bytes=mem,disk_available_bytes=disk,load=os.getloadavg(),gpu_readonly=dict(exit_code=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),reservation=dict(task18_cpu=0,task19_cpu=2,total_cpu=2,auxiliary_cpu=2,task18_as_gib=0,task19_as_gib=4,minimum_runtime_headroom_gib=16),envelope=plan['envelope'],seconds=time.monotonic()-started,model_or_numerical_execution=False)
except BaseException as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started,model_or_numerical_execution=False)
with(out/'report.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
for q in out.iterdir():q.chmod(0o444)
print(json.dumps(dict(path=str(out/'report.json'),sha256=sha(out/'report.json'),passed=result['passed'])))
raise SystemExit(0 if result['passed'] else 1)
