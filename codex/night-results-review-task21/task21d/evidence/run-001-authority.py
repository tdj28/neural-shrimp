from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21d');O=B/'run-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=2433,tool_session_reaped_exit_code=0,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,'bb2ee1e81bbca6c1c8489cf90761f9f9768b7e6eff2278b8d230ad04599d1d21'),(S,'7e13ea4833a8b3240c095d5d1557f80f7a3cafbcf3e8cb3f7945905ee9b79d78')]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json','c1793fde4211e0ed86c40a7cc52a2ca63307c8ee213b633500d091261e3b7d81');c=pinned(S/'closure.json','324e1ab6fde0cd9c9aa3ff9e769277ad2a301a5a004ddd3b48d80553702f2e59');w=read(S/'worker.json');q=pinned(O/'report.json','f3451894ae6c24174d70cb6f1022978b5d1fd0176ee2607bbc6dfff60615a684')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='ff5a792e19149d4134a36dc1ce00dbf2a648f94c';assert q['computational_completed']is True and q['scientific_claims_accepted']is False
 assert c['wrapper_sha256']==a['sources']['codex/night-results-review-task21/task21d/run.py'];assert c['card_sha256']==a['sources']['codex/night-results-review-task21/task21d/run-001-card.json']
 assert len(a['sources'])==5
 assert q['source_sha256']['run.py']==c['wrapper_sha256'] and q['card_sha256']==c['card_sha256']
 for name,pin in q['source_sha256'].items():assert a['sources']['codex/night-results-review-task21/task21d/'+name]==pin
 assert len(q['input_pins'])==16
 for x in q['input_pins']:
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==[(3019852,201072986),(3019874,201073097)]
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log'};outer_rows=[]
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'report.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=5,original_input_pin_rows_verified=16,original_supervisor_worker_groups_sessions_absent=True,identities=ids,scope='Saved source/input/output/closure authentication only; no numerical checks repeated. Original session2433 reaped0; exact captured supervisor/worker births checked absent. No separate historical dispatch-waiter identity is inferred.')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
