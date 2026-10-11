"""Saved-only same-root total crossing refinement; no model/derivative calls."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_second_unit_flip as C
from neural_shrimp.rigorous_unit_flip_crossing import validated_inverse,refined_transverse_crossing
from neural_shrimp.rigorous_orbits import eye,matrix_json,canonical_bytes
from neural_shrimp.rigorous_eigenpair_march import decode
need,read,entry=C.need,C.read,C.entry
OUT=C.OUT.parents[1]/'crossing001/run';CARD='docs/plans/second-unit-flip-crossing-inputs.json'
LIMITS=dict(seconds=30,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=32*1024**2,native_calls=0,precision=384)
FILES=sorted(set(C.FILES+[CARD,'docs/plans/second-unit-flip-crossing.md','docs/reviews/second-unit-flip-rigorous001.json','src/neural_shrimp/rigorous_unit_flip_crossing.py','tools/refine_second_unit_flip_crossing.py','tools/launch_second_unit_flip_crossing.py','tests/test_second_unit_flip_crossing.py']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def check(self):need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'30s1GiB')
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);need(self.used+len(raw)<(32 if closing else 28)*1024**2,'32MiB4MiBclosing');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.check()
def profile(w):
 A=arb_mat([[arb(int(i==j))+arb((1,-30))for j in range(55)]for i in range(55)]);t=time.monotonic();used=w.used;X=validated_inverse(A);w.save('profile-inverse.json',dict(input=matrix_json(A),inverse=matrix_json(X)));dt=time.monotonic()-t;forecast=dict(seconds=time.monotonic()-w.start+50*dt+10,bytes=w.used+10*(w.used-used)+4*1024**2,rss_kib=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024);ok=forecast['seconds']<30 and forecast['bytes']<32*1024**2 and forecast['rss_kib']<1024**2;w.save('profile.json',dict(passed=ok,forecast=forecast,target_arrays_unopened=True,native_calls=0));need(ok,'full55 measured saved reduction before arrays')
def bind(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['limits']==LIMITS,'fixed reviewed same-root crossing');s=card['root'];p,rt=C.C.history(w,s,2);o=read(p/'run/outcome.json');need(p==C.OUT.parent and o==s['outcome']and not o['passed']and o['root_passed']and o['complement_passed']and not o['crossing_passed']and o['native_calls']==o['native_inputs']==432,'literal completed root/spectrum and unresolved crossing')
 need(rt['settings']==C.SETTINGS and rt['limits']==C.LIMITS,'unchanged exact original root settings')
 for n in ('root.json','input-binding.json','same-root-complement.json','same-root-crossing.json'):need(entry(p/'run'/n)==s['files'][n],'exact saved actual root file')
 if metadata_only:return p
 root=read(p/'run/root.json');ib=read(p/'run/input-binding.json');need(root['passed']and root['coordinate_inclusions']==[True]*55 and all(root['gates'].values())and decode(root['contraction'])<1 and [v['divisor']for v in root['divisors']]==[1,2,3,6,9]and all(v['excluded']for v in root['divisors']),'complete same55 root authority')
 for n,v in ib['native_source'].items():need(entry(ROOT/n)==rt['sources'][n]==v,'identical actual native equations')
 files=list((p/'run/events').glob('*-target-box_evaluation.json.gz'));need(len(files)==1,'literal whole box acquisition');e=read(files[0]);need(e['box']==root['box']and e['jacobian']==root['whole_jacobian']and e['monodromy']==root['monodromy'],'entire same acquired55 domain/J/M')
 w.save('input-binding.json',dict(card=card,root=entry(p/'run/root.json'),native_source=ib['native_source'],same_box_event=entry(files[0]),map_inputs=ib['map_inputs']));return root

def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0,target_acquired=False)
 def stop(*_):raise TimeoutError('30s owned same-root crossing')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   profile(w);root=bind(w,read(ROOT/CARD));ans['target_acquired']=True;box=[decode(v)for v in root['box']];J=arb_mat([[decode(v)for v in row]for row in root['whole_jacobian']]);M=arb_mat([[decode(v)for v in row]for row in root['monodromy']]);result=refined_transverse_crossing(J,M,box[42:],13);w.save('crossing.json',result);ans.update(passed=result['passed'],crossing=result['multiplier_hidden_rate_derivative'],status='same_root_crossing_qualified'if result['passed']else'saved_bound_unresolved',scope='Same already proved55 root, full saved derivative domain and stable10 quotient; tighter validated inverses only. No new root, native calls, full signed connection or normal-form claim.')
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<30 and peak<1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
