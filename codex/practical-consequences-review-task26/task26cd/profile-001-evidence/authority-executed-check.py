from pathlib import Path
import json,hashlib,resource,os,subprocess,shutil,time,traceback
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26cd');O=B/'profile-001';S=O.with_name(O.name+'-supervision');D=O.with_name(O.name+'-authority');D.mkdir(exist_ok=False);shutil.copyfile(__file__,D/'executed-check.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pinned(p,pin):assert sha(p)==pin;return read(p)
start=time.monotonic();report=dict(passed=False,tool_session_id=52883,tool_session_reaped_exit_code=0,numerical_replay=False)
try:
 rows=[]
 for base,pin in [(O,'d0a30298dd997067161625b78fc871847b02137d5c860f4f5fe656855090215c'),(S,'e7cd9695b8156d0394e403fdeef5cf4e1ea2ed7a788b53a943580773741594b9')]:
  m=pinned(base/'final-manifest.json',pin);expected={x['path']:x for x in m['files']};actual={p.relative_to(base).as_posix():p for p in base.rglob('*')if p.is_file()and p!=base/'final-manifest.json'};assert len(expected)==len(m['files'])and actual.keys()==expected.keys()
  for name,p in actual.items():assert not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==expected[name]['bytes']and sha(p)==expected[name]['sha256']
  assert not(base/'final-manifest.json').stat().st_mode&0o222;rows.append(dict(root=str(base),manifest_sha256=pin,files=len(actual)))
 a=pinned(S/'admission.json','90be0341b50c99e43639e0314a708ef2d9268a65ca07f049c25ecfb42c05e153');c=pinned(S/'closure.json','5265b7d1ca1b2976e891cf7c0b84d1f71d8ef9303b1a4c992278e303fe5433f9');w=read(S/'worker.json');q=pinned(O/'receipt.json','07ccabba738fb14803be40a7f3c5a16d7dfa2587214cdef3c96cf57aea5f0578')
 assert c['exit_code']==0 and c['timed_out']is False and c['failure']is None and c['owned_worker_and_groups_closed']is True and c['worker']==w
 assert a['source_commit']==c['source_commit']=='f395d4d0a4f245444f4578d474f26df7eae1b809';assert q['map_cells']==[5353,6837] and q['XO_cells']==24 and q['draws']==10000 and len(q['files'])==5
 assert c['wrapper_sha256']==a['sources']['codex/practical-consequences-review-task26/task26cd/synthetic_profile.py'];assert c['card_sha256']==a['sources']['codex/practical-consequences-review-task26/task26cd/synthetic-profile-card.json']
 assert len(a['sources'])==6
 for name,pin in a['sources'].items():
  p=S/'source'/name;blob=subprocess.check_output(['git','-C',str(R),'-c','core.preloadIndex=false','-c','core.packedGitWindowSize=32m','-c','core.packedGitLimit=64m','show',a['source_commit']+':'+name],timeout=10);assert blob==p.read_bytes()and sha(p)==pin
 ids=[a['supervisor'],w];assert [(x['pid'],x['start'])for x in ids]==[(2922630,200230831),(2922644,200230945)]
 for x in ids:assert not Path('/proc',str(x['pid'])).exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:f=(p/'stat').read_text().rsplit(')',1)[1].split()
  except(FileNotFoundError,ProcessLookupError):continue
  assert all(int(f[2])!=x['pgid']and int(f[3])!=x['sid']for x in ids)
 outer=O.with_name(O.name+'-dispatch');assert {p.name for p in outer.iterdir()}=={'stdout.log','stderr.log'};outer_rows=[]
 for p in sorted(outer.iterdir()):
  assert p.is_file()and not p.is_symlink();p.chmod(0o444);outer_rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 report.update(passed=True,outer_logs=outer_rows,source_commit=a['source_commit'],report_sha256=sha(O/'receipt.json'),closure_sha256=sha(S/'closure.json'),manifests=rows,captured_git_source_blobs_verified=6,original_supervisor_worker_groups_sessions_absent=True,identities=ids,scope='Saved source/output/closure authentication only; no numerical checks repeated')
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc())
report['seconds']=time.monotonic()-start;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(D.iterdir())]),indent=2)+'\n')
for p in D.iterdir():p.chmod(0o444)
print(json.dumps(dict(passed=report['passed'],path=str(D/'report.json'),report_sha256=sha(D/'report.json'),manifest_sha256=sha(D/'manifest.json'),seconds=report['seconds'],error=report.get('error'))));raise SystemExit(0 if report['passed']else 1)
