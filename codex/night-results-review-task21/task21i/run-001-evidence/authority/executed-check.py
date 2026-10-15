from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
PINS={'output_manifest': 'd778763f7f3fba93dd4a9a5207b163fce9da01c08cd91d01df2a51dc03e41d45', 'supervision_manifest': 'ee8d2783d6aa016febe9a5a1e2f301ccc77f55371cdce9e89e10ad8c2659589d', 'admission': 'd5f4a1d6f75e4992934fa0e588d7e018d90deab9ede50a4fbbe8262325c5b69e', 'closure': 'f9b913d2a3927c15d5a8eaca0bc12e1bef23d8d477e6b65bbb865e86dd083c51', 'report': '3fb8d17b3976e8b7feaa4b2e520ce4b54928de279f7d98097a31cc28daad1d48', 'source_commit': 'f5fa2fd151de4f500d74b83c7176ccc9525360fb', 'preflight': 'a80416dda44d2978034b49ef07de124b99bc75d015907e8e5e778484c3ddbb22'}
SESSION_ID=26780; SESSION_REAPED_EXIT=0; EXPECTED_IDENTITIES=[(3139358, 202258763), (3139370, 202258878)]
assert all(isinstance(v,str) and len(v)==(40 if k=='source_commit' else 64) for k,v in PINS.items()) and type(SESSION_ID)is int and SESSION_REAPED_EXIT==0 and EXPECTED_IDENTITIES is not None, 'prepared authority: actual pins/session/identities not supplied'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21i');O=B/'run-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=SESSION_ID,tool_session_reaped_exit_code=SESSION_REAPED_EXIT,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,PINS['output_manifest']),(S,PINS['supervision_manifest'])]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json',PINS['admission']);c=pinned(S/'closure.json',PINS['closure']);w=read(S/'worker.json');q=pinned(O/'report.json',PINS['report'])
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']==PINS['source_commit'];assert q['schema']=='task21i-independent-saved-behavior-v1' and q['computational_status']=='complete'
 assert c['wrapper_sha256']==a['sources']['codex/night-results-review-task21/task21i/run.py'];assert c['card_sha256']==a['sources']['codex/night-results-review-task21/task21i/run-001-card.json']
 assert len(a['sources'])==5
 assert set(q['source_sha256'])=={'run.py','statistics.py','adjustment.py'} and q['source_sha256']['run.py']==c['wrapper_sha256']
 for name,pin in q['source_sha256'].items():assert a['sources']['codex/night-results-review-task21/task21i/'+name]==pin
 assert len(q['input_pins'])==145
 for x in q['input_pins']:
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==EXPECTED_IDENTITIES
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log','preflight.json','preflight.py','dispatch.json'};outer_rows=[]
 assert sha(outer/'preflight.json')==PINS['preflight']
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'report.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=5,original_input_pin_rows_verified=145,original_supervisor_worker_groups_sessions_absent=True,identities=ids,scope=f'Saved source/input/output/closure authentication only; no numerical checks repeated. Original session{SESSION_ID} reaped0; exact captured supervisor/worker births checked absent. No separate historical dispatch-waiter identity is inferred.')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
