from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/steering-controls-crosscheck');O=B/'run-001-review-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=50359,tool_session_reaped_exit_code=0,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,'5bbc25b8cfe584c64e05b9a4b3dfbaa5add6aa2304a7fe0e2065de39d37c3ed6'),(S,'c9e93332320e43a7bf81035bc921b8f24a83b50bfa1932c467ddc02e2fc1d8bd')]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json','78317e9be7a7cf32dcd39db448e31d2c2b1ecb4372ad5e5efc2ef4141d602854');c=pinned(S/'closure.json','92027136add3be8e065575f1ffab07c89ca1ff38a35e655fe206dd0a7c430de8');w=read(S/'worker.json');q=pinned(O/'report.json','f0544574ddd02f4f3329d52cc528559bc6b1a5c6aa69e75df6f1a6530de5e297')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='7a607602e98de099c9c610171035b2e34f7f8fcf';assert q['passed']is True and q['completed']is True
 assert q['auditor_sha256']==c['wrapper_sha256']==a['sources']['codex/steering-controls-crosscheck/saved-audit.py'];assert q['audit_card_sha256']==c['card_sha256']==a['sources']['codex/steering-controls-crosscheck/run-001-review-card.json']
 assert len(a['sources'])==3
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==[(2768013,199317540),(2768030,199317674)]
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log'};outer_rows=[]
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'report.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=3,original_audit_supervisor_worker_groups_sessions_absent=True,identities=ids,scope='Saved source/output/closure authentication only; no numerical checks repeated')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
