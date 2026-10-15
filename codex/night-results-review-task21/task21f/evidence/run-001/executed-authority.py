from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21f');O=B/'run-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=54893,tool_session_reaped_exit_code=0,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,'2d8c9affc0219f485c164587c02c929f38db3e7400f8419dfab57970f77c9212'),(S,'9ac96a34c1ac3f07c4e1a869ac85d49c041f540a3716fde73f80ad0c0a33273a')]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json','e8df8b28eb73f1c16a21bb16def217138cdf2c8d4a0934cf1f8ffa61f6b22242');c=pinned(S/'closure.json','aadb89224f0be7d40f0855393d2c9a94046125026059f0feddc4c98b073c879f');w=read(S/'worker.json');q=pinned(O/'report.json','b7347ffd2ef4c61e69d3cab25be276cc515ce8fb57ead609ec0bff159a9e1c31')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='7d75414c291a4ada09c4e99fa85f84ae3d3b8165';assert q['computational_completed']is True and q['scientific_claims_accepted']is False
 assert c['wrapper_sha256']==a['sources']['codex/night-results-review-task21/task21f/run.py'];assert c['card_sha256']==a['sources']['codex/night-results-review-task21/task21f/run-001-card.json']
 assert len(a['sources'])==4
 assert dict((x['name'],x['sha256'])for x in q['actual_source_pins'])['run.py']==c['wrapper_sha256'] and q['card_sha256']==c['card_sha256']
 for name,pin in [(x['name'],x['sha256'])for x in q['actual_source_pins']]:assert a['sources']['codex/night-results-review-task21/task21f/'+name]==pin
 assert len(q['input_authentication'])==44
 for x in q['input_authentication']:
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==[(3075287,201579523),(3075306,201579661)]
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log','preflight.json','preflight.py','dispatch.json'};outer_rows=[]
 assert sha(outer/'preflight.json')=='92bacfc2104bf877889ede311fc44e5fd5d946a12c247696e6d8906513fd6c13'
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'report.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=4,original_input_pin_rows_verified=44,original_supervisor_worker_groups_sessions_absent=True,identities=ids,scope='Saved source/input/output/closure authentication only; no numerical checks repeated. Original session54893 reaped0; exact captured supervisor/worker births checked absent. No separate historical dispatch-waiter identity is inferred.')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
