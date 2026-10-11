"""Saved-only floating forcing/defect diagnosis; proposals never certify a root."""
import hashlib,json,math,os,resource,signal,subprocess,time,traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/critical-organizer-bridge/attempt001')
OUT=PARENT.with_name('attempt001-diagnosis')/'run'
SOURCE='7e31da607e67190c2ee0dee413db47b21e8c2174'
PINS=['6dbedfc759ae4c6c493bf29cee1d3b54076ec0d1f74ecc740b2f2005ab6aa57b','f996a071d39c738d66e8157df576ad61a66ba0be8282c2fd0be22065a54501f2']
FILES=['tools/diagnose_clark_critical_bridge.py','tools/launch_clark_critical_bridge_diagnosis.py','tests/test_clark_critical_bridge_diagnosis.py','docs/plans/clark-critical-bridge-diagnosis.md']
def need(v,msg):
 if not v:raise ValueError(msg)
def read(p):return json.loads(p.read_bytes())
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def exact_pair(pair):return F(pair[0])*F(2)**pair[1]
def upper_abs(ball):return abs(exact_pair(ball['mid']))+exact_pair(ball['rad'])
def upper_float(x):return np.nextafter(float(x),math.inf)if x else 0.
def reduce(B,f,r):
 B=np.asarray(B,float);f=np.asarray(f,float);r=np.asarray(r,float);n=len(r)
 need(B.shape==(n,n)and f.shape==(n,)and n>=2 and np.isfinite(B).all()and np.isfinite(f).all()and np.isfinite(r).all()and(B>=0).all()and(f>=0).all()and(r>0).all(),'finite nonnegative complete bounds')
 row=B@r/r;res=f/r;image=res+row;radius=float(max(abs(np.linalg.eigvals(B))))
 sol=np.linalg.solve(np.eye(n)-B,f+2.**-40)
 need(np.isfinite(sol).all()and(sol>0).all(),'positive floating resolvent proposal')
 proposed=np.array([2.**math.ceil(math.log2(2*x))for x in sol]);need(np.isfinite(proposed).all()and(proposed>0).all(),'finite dyadic proposed radii')
 return dict(dimension=n,old_weighted_row_sums=row.tolist(),old_residual_ratios=res.tolist(),old_image_ratios=image.tolist(),old_strict_inclusions=int(sum(image<1)),spectral_radius_proposal=radius,positive_resolvent=sol.tolist(),proposed_radii=proposed.tolist(),proposed_exponents=[int(math.log2(v))for v in proposed],proposed_to_original=(proposed/r).tolist(),frozen_old_domain_weighted_defect=float(max(B@proposed/proposed)),frozen_old_domain_image_ratio=float(max((f+B@proposed)/proposed)),dominant_coordinates=np.argsort(-image).tolist(),absolute_defect=B.tolist(),absolute_forcing=f.tolist(),scope='Floating saved-bound diagnosis only. Enlarged radii leave original acquisition domains; new whole-domain evaluations and all strict gates are mandatory. No proof or known-root join inferred.')
def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<10 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<448*1024,'10s/448MiB cap')
 def save(name,value,closing=False):
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used[0]+len(b)<2*1024**2-(0 if closing else 128*1024),'2MiB with closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
 def halt(*_):raise TimeoutError('10s saved-only cap')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,10)
 try:
  need(os.getpriority(os.PRIO_PROCESS,0)>=10 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean nice10 source');commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();sources={n:entry(ROOT/n)for n in FILES}
  save('runtime.json',dict(source_commit=commit,sources=sources,pid=os.getpid(),native_calls=0,numpy=np.__version__,limits=dict(seconds=10,rss_bytes=448*1024**2,as_bytes=2*1024**3,raw_bytes=2*1024**2)))
  for n,pin in sources.items():
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);used[0]+=len(raw)
  for parent,pin in zip([PARENT,PARENT/'run'],PINS):
   need(entry(parent/'manifest.json')['sha256']==pin,'literal original inventory');files=read(parent/'manifest.json')['files'];actual={str(p.relative_to(parent)):p for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'};need(set(files)==set(actual),'complete original inventory')
   for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable original bytes');tick()
  rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');old=read(PARENT/'run/outcome.json');need(rt['source_commit']==la['source_commit']==SOURCE and la['exit_code']==2 and not la['owned_live_pids']and old['error']=='ValueError: uniform full26 root tube','retained exact failure')
  for n,pin in rt['sources'].items():need(entry(PARENT/'run/source'/n)==pin and hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'historical source');tick()
  t=read(PARENT/'run/tube.json');need(t['dimension']==26 and not t['passed']and len(t['coordinate_inclusions'])==26,'exact full failed tube')
  B=[[upper_float(upper_abs(v))for v in row]for row in t['defect']];f=[upper_float(upper_abs(row[0]))for row in t['preconditioned_residual']];r=[float(exact_pair(v['mid']))for v in t['radii']];need(all(exact_pair(v['rad'])==0 for v in t['radii']),'exact radii')
  answer=reduce(B,f,r);save('diagnosis.json',answer);save('input-binding.json',dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],tube=entry(PARENT/'run/tube.json'),original_calls=old['calls']));tick();result.update(passed=True,status='saved_floating_diagnosis_complete',original_strict_inclusions=sum(t['coordinate_inclusions']),native_calls=0,scope=answer['scope'])
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
