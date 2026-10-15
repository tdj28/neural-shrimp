from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21g');O=B/'run-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=71929,tool_session_reaped_exit_code=0,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,'9a924e6261df99e4a8470641814a64ce0b4219cb25ce9348b74c35f44c5b1f4d'),(S,'d0d1064036ef1f55c687599ee96b197b88cd3ec9e62ced11cefa94f28e8e67a1')]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json','a0f4ae1d297e98d42d69b78dec128239ffe6a9bb1314398a542d2c1e436b1547');c=pinned(S/'closure.json','4083520cf02ea6cae13f83be899cbd39581a87c553a3cd3af424ba5259cfcde7');w=read(S/'worker.json');q=pinned(O/'report.json','83a09773d414e7efe735f768424d90e7a19c9f250b6df3a6281f8b3096b80d26')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='34654dac0c03d26b2f52742310bd0fa552e9fd69';assert q['computational_completed']is True and q['scientific_claims_accepted']is False
 assert c['wrapper_sha256']==a['sources']['codex/night-results-review-task21/task21g/run.py'];assert c['card_sha256']==a['sources']['codex/night-results-review-task21/task21g/run-001-card.json']
 assert len(a['sources'])==4
 assert dict((x['name'],x['sha256'])for x in q['actual_source_pins'])['run.py']==c['wrapper_sha256'] and q['card_sha256']==c['card_sha256']
 for name,pin in [(x['name'],x['sha256'])for x in q['actual_source_pins']]:assert a['sources']['codex/night-results-review-task21/task21g/'+name]==pin
 assert len(q['input_authentication'])==15
 for x in q['input_authentication']:
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==[(3080323,201667580),(3080347,201667692)]
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log','preflight.json','preflight.py','dispatch.json'};outer_rows=[]
 assert sha(outer/'preflight.json')=='301a17f478ad127ffd8ffb517dad3b967ea6c7ee3bbef464df5c9243ddb2482d'
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'report.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=4,original_input_pin_rows_verified=15,original_supervisor_worker_groups_sessions_absent=True,identities=ids,scope='Saved source/input/output/closure authentication only; no numerical checks repeated. Original session71929 reaped0; exact captured supervisor/worker births checked absent. No separate historical dispatch-waiter identity is inferred.')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
