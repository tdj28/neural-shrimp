"""Task16 original output/source/process closure only; no scientific decoding."""
from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
resource.setrlimit(resource.RLIMIT_CPU,(60,65))
os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
B=Path('/data2/neural-shrimp-data/codex/steering-controls-crosscheck')
O=B/'run-001/run';S=B/'run-001-supervision';X=B/'run-001-dispatch';D=B/'run-001-closure-001'
D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin,str(p);return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=25265,tool_session_reaped_exit_code=0,tool_exit_observation='Inference write_stdin terminal result; foreground exec supervisor with persistent outer logs; no separate waiter',numerical_values_decoded=False,numerical_replay=False)
try:
 rows=[]
 for base,pin,count in [(O,'c6242ae732d7959b82bf17e9c703d968e3216ca62b66cc38abfc5d86c02cdcbf',51),(S,'0d58dc96557e6b4b0e9b749a9100d6e22c16b2d69a319d4360b87d936dee725a',16)]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'}
  assert len(expected)==len(m['files'])==count and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256'],name
  assert not(base/'final-manifest.json').stat().st_mode&0o222
  rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual),bytes=sum(x['bytes']for x in expected.values())))
 a=pinned(S/'admission.json','8de4529ed1342c7c1878745b864ef3f488b81826b7aa48252d649affc0966273')
 c=pinned(S/'closure.json','2c210be28006b0dd251e60ea3e019b0761c42852f0965949fe73a0a5c462fcb9')
 w=pinned(S/'worker.json','ae4c66078e72be071e45401468ddb89fdf1259eb39c6a88bbf5d8d033a8a1296')
 q=pinned(O/'status.json','21579f825b201b443ebbbbb56f6b491703868ea75370828464baaeb70ec0ceba')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='9b97f9f6e0d55bc5f02e3dc8676b398ab5812541'
 assert q['computational_completed']is True and q['all_datasets_admitted']is True
 assert q['source_commit']==c['scientific_source_commit']=='dc6cd5c6f3e04c6f17302509035288dcad2e4e77'
 assert q['plan_sha256']==c['plan_sha256']==a['sources']['codex/steering-controls-crosscheck/run-001-plan.json']
 assert q['sources']['run.py']==c['wrapper_sha256']==a['sources']['codex/steering-controls-crosscheck/run.py']
 assert q['sources']['tabulate.py']==a['sources']['codex/steering-controls-crosscheck/tabulate.py']
 assert c['card_sha256']==a['sources']['codex/steering-controls-crosscheck/run-001-card.md']
 assert len(a['sources'])==11
 source_files=[]
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10)
  assert blob==p.read_bytes()and sha(p)==pin;source_files.append(dict(path=name,sha256=pin,bytes=p.stat().st_size))
 ids=[a['supervisor'],w]
 assert [(x['pid'],x['start'])for x in ids]==[(2757544,199251816),(2757585,199251964)]
 for x in ids:assert x['uid']==os.getuid()and x['pgid']==x['sid']==x['pid']and not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 assert {p.name for p in X.iterdir()}=={'stdout.log','stderr.log'}
 outer=[]
 for p in sorted(X.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 assert (S/'stderr.log').stat().st_size==(X/'stderr.log').stat().st_size==0
 report.update(passed=True,source_commit=a['source_commit'],scientific_source_commit=q['source_commit'],plan_sha256=c['plan_sha256'],card_sha256=c['card_sha256'],wrapper_sha256=c['wrapper_sha256'],report_sha256=sha(O/'report.json'),status_sha256=sha(O/'status.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=11,source_files=source_files,original_supervisor_worker_groups_sessions_absent=True,identities=ids,outer_logs=outer,computational_completed=True,worker_elapsed_seconds=q['elapsed_seconds'],worker_peak_rss_kib=q['peak_rss_kib'],supervisor_elapsed_seconds=c['seconds'],scope='Original saved output/source/closure authentication only; scientific saved-field check and report acceptance remain separate')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;report['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
