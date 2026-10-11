#!/usr/bin/env python3
"""Fixed independent512 same-root Clark generalized-flip outcome audit."""
import hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
import review_clark_organizer_math as M
from neural_shrimp import rigorous_clark_generalized_flip as A
from neural_shrimp.rigorous_orbits import ball_json,canonical_bytes
need=M.need
OUT=Path('/data2/neural-shrimp-data/audits/clark-generalized-flip-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-generalized-flip/attempt001')
OUTER='5bb53bac3155b56c20e2365c640b9e080b842691aac04ce44962ba19f88b95d5';WORKER='a1ddbef5dfef44058b5eb980aa276a4a4c7a10f0f9873f2d40f51ac0e4b786b1';EXEC='a65cadec1f55d97671e25675a163a349a9b2b45a'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAPS=dict(augmented=4,path5=10,native_phases=54,events=160)
SHARED=['src/neural_shrimp/'+n+'.py'for n in('__init__','controls','models','rigorous_orbits','rigorous_clark_generalized_flip')]
FILES=SHARED+['tools/reviews/review_clark_organizer_math.py','tools/reviews/review_clark_organizer.py','tools/launch_clark_organizer_review.py','tests/test_clark_organizer_review_math.py','tests/test_clark_organizer_review_driver.py','docs/plans/clark-organizer-independent.md','docs/reviews/clark-rnn-generalized-flip-outcome.json']

def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def serial(x):
 if isinstance(x,arb):return ball_json(x)
 if isinstance(x,arb_mat):return [[serial(v)for v in row]for row in x.tolist()]
 if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v)for v in x]
 return x

def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable manifest');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(actual),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable '+n);tick()
 return dict(root=str(root),manifest=pin,files=len(files),bytes=sum(v['bytes']for v in files.values()))

def root_binding(root,center_attempt,center_return,whole_attempt,whole_return):
 need(root['dimension']==26 and len(root['center'])==len(root['box'])==26,'actual26-coordinate root')
 need(center_attempt['z']==root['center']and center_return['value']==root['center_value']and whole_attempt['z']==root['box']and whole_return['value']==root['whole_value']and whole_return['jacobian']==root['whole_jacobian'],'literal center/whole acquired inputs and values')
 need(len(root['whole_jacobian'])==26 and all(len(row)==26 for row in root['whole_jacobian']),'complete saved full Jacobian')

def bind(w):
 inv=[inventory(PARENT,OUTER,w.tick),inventory(PARENT/'run',WORKER,w.tick)];run=PARENT/'run';rt=read(run/'runtime.json');la=read(PARENT/'launcher-result.json');out=read(run/'outcome.json');receipt=read(ROOT/'docs/reviews/clark-rnn-generalized-flip-outcome.json')
 need(rt['source_commit']==la['source_commit']==EXEC and rt['precision']==384 and la['exit_code']==0 and la['worker_pid_absent']and not la['owned_live_pids'],'completed exact producer')
 need(receipt['source']==EXEC and receipt['root']==str(PARENT)and receipt['outcome']==out and receipt['inventories'][0]['manifest_sha256']==OUTER and receipt['inventories'][1]['manifest_sha256']==WORKER,'literal tracked producer receipt')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'producer owned PID absent')
 for n,e in rt['sources'].items():need(entry(run/'source'/n)==e and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'exact historical acquired source '+n)
 for n in SHARED:need(entry(ROOT/n)==rt['sources'][n],'identical shared native kernel '+n)
 events=sorted((run/'events').glob('*.json'));need(len(events)==146 and [int(p.name.split('-')[0])for p in events]==list(range(1,147)),'complete original inclusive event inventory')
 need(sum('augmented_attempt' in p.name for p in events)==9 and sum('generalized_flip_return' in p.name for p in events)==9 and sum('path5_attempt' in p.name for p in events)==10,'actual original completed operations')
 r=read(run/'root.json');root_binding(r,read(run/'events/093-augmented_attempt.json'),read(run/'events/099-generalized_flip_return.json'),read(run/'events/100-augmented_attempt.json'),read(run/'events/106-generalized_flip_return.json'))
 inp=read(run/'inputs.json');p=Path(inp['matrix_path']);need(entry(p)==inp['matrix']and inp['matrix']['sha256']=='1be92c769bfd21bf23a2a7ff79aca118cd58454b311e4bb095db117c865f265f','literal original fixed matrix')
 with np.load(p,allow_pickle=False)as f:j=f['j'].copy()
 need(j.shape==(12,12)and j.dtype==np.float64 and np.isfinite(j).all(),'full fixed matrix');rational=[[str(Fraction.from_float(float(v)))for v in row]for row in j];need(rational==inp['map']['j0']and inp['map']['schema']=='exact-clark-generalized-flip-v1','actual original exact dyadic map')
 jj=[[arb(v)for v in row]for row in rational];c=[M.decode(v)for v in r['center']];old=[M.decode(v)for v in r['box']];need(all(v.is_exact()for v in c),'literal exact retained center')
 w.save('input-binding.json',dict(inventories=inv,source=EXEC,matrix=entry(p),saved_root=entry(run/'root.json'),source_identical=SHARED,derivative_events=[entry(run/'events'/n)for n in('093-augmented_attempt.json','099-generalized_flip_return.json','100-augmented_attempt.json','106-generalized_flip_return.json')],map=inp['map'],shared_dependencies='Native augmented Dual kernel and python-flint/Arb; proof reducers independent.'))
 return jj,c,old,r,inp

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s/1GiB cap')
 def save(self,n,row,closing=False):
  raw=canonical_bytes(serial(row));need(self.used+len(raw)<(128 if closing else 124)*1024**2,'128MiB with4MiB reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def charge(self,k):self.tick();need(self.calls[k]<CAPS[k],'fixed '+k+' ceiling');self.calls[k]+=1
 def emit(self,k,row):
  self.charge('events')
  if k in('native_augmented_polynomial','independent_native_path','independent_native_phase'):self.charge('native_phases')
  self.save(f'events/{self.calls["events"]:03d}-{k}.json',row);self.tick()
 def augmented(self,j,z):self.charge('augmented');self.emit('shared_augmented_attempt',dict(z=z));return A.augmented(j,z,self.emit)
 def path(self,h,j,g,d):self.charge('path5');self.emit('independent_path_attempt',dict(coefficients=h,gain=g,leak=d));return M.signed_path(h,j,g,d,self.emit)

def synthetic(w):
 n=12;q=[arb(int(i==0))for i in range(n)];j=[[arb(int(i==k))*(1 if i==0 else arb(1)/2)for k in range(n)]for i in range(n)];z=[arb(0)]*n+[arb(3)/4]+q+[arb(1)];times={};sizes={}
 def measure(k,f):
  t=time.monotonic();used=w.used;r=f();times.setdefault(k,[]).append(time.monotonic()-t);sizes.setdefault(k,[]).append(w.used-used);return r
 for zz in(z,[arb(v,arb((1,-94)))for v in z]):measure('augmented',lambda:w.augmented(j,zz))
 h=[[arb(i+k+1)/100 for i in range(n)]for k in range(6)]
 for _ in range(5):measure('path',lambda:w.path(h,j,arb(7)/5,arb(3)/5))
 def algebra():
  M.krawczyk([arb(0)]*26,[arb(0,arb((1,-94)))]*26,M.col([0]*26),M.eye(26),w.emit)
  try:M.native(j,z,w.emit)
  except ValueError as e:need(str(e)=='all native6 proper divisors','only intended native profile failure')
  else:raise ValueError('synthetic fixedpoint admitted as primitive6')
  a=arb_mat([[(-1 if i==0 else arb(1)/4)*int(i==k)for k in range(n)]for i in range(n)])
  def poly(h):return [[-h[k][0]+(3*h[1][0]**5 if k==5 else 0)]+[h[k][i]/4 for i in range(1,n)]for k in range(6)]
  M.normal_form([arb(0)]*n,a,q,q,poly,w.emit);J=arb_mat(26,26)
  for i in range(n):J[i,i]=a[i,i]-1;J[n+i,n+1+i]=a[i,i]+1
  J[12,12]=-3;J[12,25]=-2;J[24,13]=1;J[25,12]=5;J[25,25]=7;M.unfolding(J,a,q,w.emit)
 measure('algebra',algebra);t={k:max(v)for k,v in times.items()};b={k:max(v)for k,v in sizes.items()};forecast=time.monotonic()-w.start+2*(2*t['augmented']+5*t['path']+t['algebra'])+20;raw=w.used+2*(2*b['augmented']+5*b['path']+b['algebra'])+8*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 out=dict(passed=bool(forecast<60 and raw<128*1024**2 and peak<1024**3),times=times,bytes=sizes,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,calls=dict(w.calls),target_arrays_unopened=True);w.save('profile.json',out);need(out['passed'],'measured full12D independent audit admission')

def same_root(old,fresh):
 need(len(old)==len(fresh)==26 and all(b.contains_interior(a)for a,b in zip(old,fresh)),'strict complete original-root box in fresh unique box');return dict(all26_original_boxes_strictly_inside=True,authority='Independently verified original root lies in fresh box; fresh strict contraction supplies uniqueness for identical augmented equations.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed audit');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,60);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=512,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'exact own source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):
   synthetic(w);j,c,old,saved,inp=bind(w);endpoints=[(arb(tuple(v['mid']))-arb(tuple(v['rad'])),arb(tuple(v['mid']))+arb(tuple(v['rad'])))for v in saved['box']];oldproof=M.krawczyk(c,old,M.matrix(saved['center_value']),M.matrix(saved['whole_jacobian']),w.emit,exact_bounds=endpoints);w.save('saved-root.json',oldproof)
   fresh=[arb(v,arb((1,-94)))for v in c];h0,_,_=w.augmented(j,c);h,J,_=w.augmented(j,fresh);proof=M.krawczyk(c,fresh,h0,J,w.emit);w.save('fresh-root.json',dict(**proof,whole_value=h,whole_jacobian=J,radius_exponent=-94,newton_steps=0));join=same_root(old,fresh);need(fresh[-1]>arb(inp['gain_bracket'][0])and fresh[-1]<arb(inp['gain_bracket'][1]),'same original fixed gain bracket');w.save('same-root.json',join)
   A0,p,context=M.native(j,fresh,w.emit);w.save('native-context.json',context);q=fresh[13:25];normal=M.normal_form(fresh[:12],A0,q,p,lambda h:w.path(h,j,fresh[-1],fresh[12]),w.emit);w.save('quintic.json',normal);unfold=M.unfolding(J,A0,p,w.emit);w.save('unfolding.json',unfold);w.tick()
   result.update(passed=True,status='independent_same_root_generalized_flip_pass',gain=fresh[-1],leak=fresh[12],quintic=normal['quintic'],physical_determinant=unfold['physical_determinant'],multiplier_leak_derivative=unfold['multiplier_leak_derivative'],critical_cubic_gain_derivative=unfold['critical_cubic_gain_derivative'],primitive_period=6,complement_dimension=11,power_bounds=context['power_bounds'],same_root=join,scope='Independent512 saved and fresh zeroNewton exact same26D root reductions. Shared augmented Dual/Arb kernel; independent native cosh-Jacobian and exp-series quintic arithmetic. Signed G=-F3 generalized flip, native F6 critical+1 symmetry-breaking degeneracy; no interval global fold/cascade connection or full window.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(serial(result)));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
