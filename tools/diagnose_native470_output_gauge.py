"""One source-bound saved output-gauge error budget; zero native evaluations."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native470_output_gauge as D
from neural_shrimp.signed_correlated_conditioning import budget,pair,reconstruct
from neural_shrimp.rigorous_orbits import canonical_bytes
need,read,entry=D.need,D.read,D.entry
OUT=D.OUT.parents[1]/'diagnosis001/run';CARD='docs/plans/native470-output-diagnosis-inputs.json';RECEIPT='docs/reviews/native470-output-gauge001.json'
LIMITS=dict(seconds=90,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=32*1024**2,native_calls=0)
FILES=sorted(set(D.FILES+[CARD,RECEIPT,'src/neural_shrimp/unit_pair_conditioning.py','src/neural_shrimp/signed_correlated_conditioning.py','tools/diagnose_native470_output_gauge.py','tools/launch_native470_output_diagnosis.py','tests/test_signed_correlated_conditioning.py','tests/test_native470_output_diagnosis.py','docs/plans/native470-output-diagnosis.md']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def check(self):need(time.monotonic()-self.start<90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'90s/4GiB saved-only diagnostic')
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);need(self.used+len(raw)<32*1024**2-(0 if closing else 2*1024**2),'32MiB/2MiB closing reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.check()

def bind(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['limits']==LIMITS and card['producer_receipt']==entry(ROOT/RECEIPT),'fixed reviewed saved diagnostic');spec=read(ROOT/RECEIPT);p,rt=D.C.history(w,spec,2);out=read(p/'run/outcome.json');need(spec['source']=='72266c49e9a5fa6ea02f249a4cc6388fa21dd217'and spec['root']==str(D.OUT.parent)and out==spec['outcome']and not out['passed']and out['native_calls']==432 and out['ticks']==450 and out['endpoint_refined'],'literal complete output-gauge failure')
 for n,v in spec['files'].items():need(entry(p/'run'/n)==v,'literal failed field '+n)
 for n in ['src/neural_shrimp/rigorous_signed_correlated.py','src/neural_shrimp/rigorous_signed_gradient_path.py','src/neural_shrimp/rigorous_polynomial_inverse.py']:need(rt['sources'][n]==entry(ROOT/n),'unchanged original proof algebra '+n)
 D.bind(D.M.Prefix(w,'prior-'),read(ROOT/D.CARD),True);w.save('input-binding.json',dict(card=card,source=spec,scope='Complete immutable failure, including passed conditional endpoint. Saved floating full-domain coefficients only; changed-E requires new acquisition.'))
 return p

def gradients(p,w):
 paths=sorted((p/'run/events').glob('*-target-*.json.gz'));events=[(f.name.split('-target-')[1].removesuffix('.json.gz'),read(f))for f in paths];ins=[];rets=[];native_in=[];native_out=[];domain=[]
 for i,(k,v)in enumerate(events):
  if k=='native_input':native_in.append(v['index'])
  if k=='native_return':native_out.append(v['index'])
  if k=='signed_correlated_domain':domain.append(v)
  if k=='signed_gradient_path_input':
   seq=events[i+1:i+4];need([kind for kind,_ in seq]==['native_input','native_return','signed_gradient_path_return'],'literal nested gradient acquisition order');a,b,r=[z for _,z in seq]
   need(a['index']==b['index']and v['phase']==r['phase']and[{k:x for k,x in z.items()if k!='kind'}for z in a['inputs']]==v['inputs']and[{k:x for k,x in z.items()if k!='kind'}for z in b['returns']]==r['returns'],'actual complete native gradient input-return identity');need(all(z['kind']=='gradient_path'for z in a['inputs']+b['returns']),'actual gradient jet type');ins.append(v);rets.append(r)
 need(native_in==native_out==list(range(144))and len(ins)==len(rets)==36 and len(domain)==1,'every original144 target pair/36gradient phases');w.check();return ins,rets,domain[0]

def profile(w):
 n=470;start=time.monotonic();I=np.eye(n);Z=np.zeros((n,n));J=[(.99*I,.00001*np.ones((n,n))),(Z,Z),(Z,Z)];R=[(np.ones((n,1))*1e-9,np.zeros((n,1)))for _ in range(4)];C=[(I,Z),(Z,Z)];answer=budget(C,J,R,np.ones(n),2**-12,2**-15);elapsed=time.monotonic()-start;need(answer['dimension']==470,'full470 synthetic polynomial profile');forecast=time.monotonic()-w.start+10*elapsed+60;ok=forecast<90;w.save('profile.json',dict(passed=ok,seconds=elapsed,forecast_seconds=forecast,dimension=n,native_calls=0,target_arrays_unopened=True));need(ok,'full470 measured arithmetic before target arrays')

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('90s owned saved diagnostic')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact diagnostic source');f=OUT/'source'/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw);w.used+=len(raw)
  card=read(ROOT/CARD);need(card['status']=='reviewed'and card['limits']==LIMITS,'source admitted profile');profile(w);p=bind(w,card);chart=read(p/'run/target-certificate.json.gz');inp=read(p/'run/target-inputs.json');ins,rets,domain=gradients(p,w)
  need(chart['shooting_dimension']==470 and chart['sigma']==-1 and chart['weights']==inp['weights']and chart['normal']==inp['normal']==['0']*469+['1']and chart['center']==domain['center']and chart['slope']==domain['slope']and chart['curvature']==domain['curvature']and chart['whole_box']==domain['whole_box'],'actual full correction domain and output gauge');A=reconstruct(rets[:18],ins[:18],chart['normal']);B=reconstruct(rets[18:],ins[18:],chart['normal']);J=[A[0],A[1],B[2]];del A,B,ins,rets;gc.collect();C=[]
  for rows in chart['inverse_polynomial_exact']:
   m=np.array([[float(F(v[0])*F(2)**v[1])for v in row]for row in rows]);need(m.shape==(470,470)and np.isfinite(m).all(),'full finite exact inverse proposal conversion');C.append((m,np.zeros_like(m)))
  R=[pair(v)for v in chart['residual_coefficients']];weights=[float(F(v))for v in chart['weights']];report=budget(C,J,R,weights,2.**chart['settings']['radius_exponent'],2.**chart['settings']['halfwidth_exponent']);w.save('diagnostic.json',report);ans.update(passed=True,summary={k:report[k]for k in ('original_max_weighted_row','original_max_image_ratio','dominant_rows','proposal_same_old_domain_max_row','proposal_same_old_domain_max_image','forcing_order_max_ratios','defect_order_max_rows')},scope=report['scope'])
 except BaseException as e:ans.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<90 and peak<4*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
