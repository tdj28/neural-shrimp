"""Second fixed stress-cell source/input-byte/comparator-and-cell04-closure preflight; no numerical payload decode."""
import argparse,hashlib,importlib.util,json,os,resource,shutil,subprocess,time,traceback
from pathlib import Path
REPO=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
OWNED=Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
COMPARISON=Path('/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-comparison-001-authority/report.json')
COMPARISON_SHA='398388244c818e7d341ab2c8de5ed14bbc29354d06a81083ebae9c76e3498a6b'
LAUNCHER_SHA='6e355b855f0a083f7e6da4acb5836cb3b726cd4114c60c457cb3041af0a15605'
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
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(60,65))
out=OWNED/'stress-cell13-001-preflight';out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'executed-preflight.py')
try:
 assert not(OWNED/'stress-cell13-001').exists() and not(OWNED/'stress-cell13-001-supervision').exists()
 assert not subprocess.check_output(['git','-C',str(REPO),'-c','core.preloadIndex=false','status','--porcelain','--untracked-files=all'],timeout=20)
 launcher=REPO/'codex/return-map-crosscheck/stress_launch.py';assert sha(launcher)==LAUNCHER_SHA
 assert sha(REPO/'codex/return-map-crosscheck/stress-cell13-001-card.md')==a.card_sha256
 spec=importlib.util.spec_from_file_location('task19_reviewed_launcher',launcher);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 args=argparse.Namespace(attempt='stress-cell13-001',source=a.source,plan='stress-cell13-001-plan.json',plan_sha256=a.plan_sha256)
 m.validate_arguments(args);plan,blobs,remote=m.snapshot(REPO,args,started+55,[]);assert len(blobs)==11
 auth=OWNED/'input-authority-001/manifest.json';assert sha(auth)==AUTHORITY_SHA==plan['input_manifest_sha256']
 metadata=json.loads(auth.read_text());assert len(metadata['files'])==10
 input_rows=[]
 for row in metadata['files']:
  q=auth.parent/row['path'];assert q.is_relative_to(auth.parent) and not q.is_symlink();assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256'] and not q.stat().st_mode&0o222
  input_rows.append({k:row[k]for k in ['path','sha256','bytes']})
 assert sha(COMPARISON)==COMPARISON_SHA and not COMPARISON.stat().st_mode&0o222
 c=json.loads(COMPARISON.read_text());assert c['passed'] is True and c['tool_session_reaped_exit_code']==0 and c['original_audit_supervisor_worker_groups_sessions_absent'] is True
 assert c['source_commit']=='2e1fbc3db2d5043f76dd64a5ec6478ada38f9141' and c['report_sha256']=='5d1e63c58275c5f8500c0f242f81555b03921bf63d92dcfa98833a2bec6dbd5d' and c['captured_git_source_blobs_verified']==4
 for old in c['identities']:assert not Path('/proc',str(old['pid'])).exists()
 closed=[dict(path=str(COMPARISON),sha256=COMPARISON_SHA,source_commit=c['source_commit'],report_sha256=c['report_sha256'],tool_session_reaped_exit_code=c['tool_session_reaped_exit_code'],original_pids_absent=True,all_original_groups_sessions_absent_in_accepted_authority=True)]
 # Exact computational closure of cell04 and its accepted saved review; no science decoded.
 prior=OWNED/'stress-cell04-001-closure-001/report.json';prior_sha='b2dbd356258a0a66d38c79a43911f6cffbbaf91f7422cf1f315f5dd44ad994be'
 assert sha(prior)==prior_sha and not prior.stat().st_mode&0o222
 d=json.loads(prior.read_text());assert d['passed'] is True and d['tool_session_id']==85698 and d['tool_session_reaped_exit_code']==0 and d['original_waiter_launcher_worker_pids_groups_sessions_absent'] is True
 assert d['source_commit']=='6b946a069420390b773eb6a7327dbf2d7323477b' and d['plan_sha256']=='3cd70f0d2f186730da64e73cb811b4993cedf0518b3dac6e1695e0a6b63d8e7c' and d['receipt_sha256']=='0023017e561b9cc477b39cf82567064c511c7ef2ce411aed2dfba33bec610168'
 receipt=OWNED/'stress-cell04-001/receipt.json';assert sha(receipt)==d['receipt_sha256'] and not receipt.stat().st_mode&0o222
 rd=json.loads(receipt.read_text());assert rd['status']=='return_map_completed' and rd['exit_code']==0 and rd['timed_out'] is False and rd['worker_termination_verified'] is True and rd['uncertain_ownership']==[]
 assert {k:[v['pid'],v['start']]for k,v in d['original_identities'].items()}=={'worker':[1510289,191406298],'launcher':[1510259,191406147],'waiter':[1510256,191406144]}
 for old in d['original_identities'].values():assert not Path('/proc',str(old['pid'])).exists()
 closed.append(dict(path=str(prior),sha256=prior_sha,receipt_sha256=d['receipt_sha256'],source_commit=d['source_commit'],tool_session_reaped_exit_code=0,original_pids_absent=True,all_original_groups_sessions_absent_in_accepted_authority=True))
 review=OWNED/'stress-cell04-001-review-001-authority/report.json';review_sha='4e220d4b05b2c9a2f4eb2a4d5cead46d8129795e0e7cc2e92aa5bb1d038a9eb0'
 assert sha(review)==review_sha and not review.stat().st_mode&0o222
 d=json.loads(review.read_text());assert d['passed'] is True and d['tool_session_id']==33694 and d['tool_session_reaped_exit_code']==0 and d['original_audit_supervisor_worker_groups_sessions_absent'] is True
 assert d['source_commit']=='70bd350a40bbcfb16725fa1d235aa528402b56b4' and d['report_sha256']=='65a2e62c1c206a17623e8d830d98d161497695a694dc314152353fbee3af8322' and d['captured_git_source_blobs_verified']==3
 assert sorted((x['pid'],x['start'])for x in d['identities'])==[(1537988,191538696),(1538030,191538839)]
 for old in d['identities']:assert not Path('/proc',str(old['pid'])).exists()
 closed.append(dict(path=str(review),sha256=review_sha,source_commit=d['source_commit'],report_sha256=d['report_sha256'],tool_session_reaped_exit_code=0,original_pids_absent=True,all_original_groups_sessions_absent_in_accepted_authority=True))
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
   if any(s in cmd for s in ['llm-return-map-crosscheck/stress-cell04-001/source/','llm-return-map-crosscheck/stress-cell13-001/source/','llm-return-map-crosscheck/stress-cell04-001-review-001-supervision/source/','llm-loop-gain-crosscheck/run-002-comparison-001-supervision/source/']):duplicates.append(dict(pid=int(q.name),command=cmd))
  except OSError:pass
 assert not duplicates
 mem=m.available();disk=shutil.disk_usage(OWNED).free;assert mem>=36*1024**3 and disk>=16*1024**3
 gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
 result=dict(passed=True,source_commit=a.source,remote_main=remote,plan_sha256=a.plan_sha256,card_sha256=a.card_sha256,launcher_sha256=LAUNCHER_SHA,source_files=[dict(path=k,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v))for k,v in blobs.items()],input_authority_sha256=AUTHORITY_SHA,input_files=input_rows,input_values_decoded=False,original_closed_authorities=closed,no_duplicate_task19_processes=True,mem_available_bytes=mem,disk_available_bytes=disk,load=os.getloadavg(),gpu_readonly=dict(exit_code=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),reservation=dict(task18_cpu=0,task19_cpu=3,total_cpu=3,auxiliary_cpu=3,task18_as_gib=0,task19_as_gib=12,minimum_runtime_headroom_gib=16),envelope=plan['envelope'],seconds=time.monotonic()-started,model_or_numerical_execution=False)
except BaseException as exc:result=dict(passed=False,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started,model_or_numerical_execution=False)
with(out/'report.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False)
for q in out.iterdir():q.chmod(0o444)
print(json.dumps(dict(path=str(out/'report.json'),sha256=sha(out/'report.json'),passed=result['passed'])))
raise SystemExit(0 if result['passed'] else 1)
