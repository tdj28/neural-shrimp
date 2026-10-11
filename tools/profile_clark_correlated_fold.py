#!/usr/bin/env python3
"""One synthetic-only full51 path/mode/base cost profile; no target reader."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_clark_correlated_path as P
from neural_shrimp import rigorous_clark_correlated_tube as T
from neural_shrimp import rigorous_clark_explicit_fold as E
from neural_shrimp import rigorous_clark_amplitude_tube as A
from neural_shrimp.rigorous_orbits import canonical_bytes,vector_json,matrix_json,ball_json,midpoint_matrix
need=P.require
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-profile/attempt002/run')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SETTINGS=dict(precision=384,seconds=90,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=256*1024**2,closing_reserve=4*1024**2)
CAPS=dict(path=6,mixed=2,context=1,algebra=2,native_phases=30,events=128)
FILES=sorted(set(['tools/profile_clark_correlated_fold.py','tools/launch_clark_correlated_fold_profile.py','tests/test_clark_correlated_profile.py','tests/test_rigorous_clark_correlated_path.py','tests/test_rigorous_clark_correlated_tube.py','docs/plans/clark-correlated-explicit-path.md','docs/plans/clark-correlated-profile.md','docs/reviews/clark-correlated-fold-attempt001.json']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while chunk:=f.read(1024**2):h.update(chunk)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<SETTINGS['seconds']and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<SETTINGS['rss_bytes'],'fixed90s/2GiB profile cap')
 def charge(self,k):self.tick();need(self.calls[k]<CAPS[k],'fixed inclusive '+k+' ceiling');self.calls[k]+=1
 def save(self,name,obj,closing=False):
  raw=canonical_bytes(obj)
  if name.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<SETTINGS['raw_bytes']-(0 if closing else SETTINGS['closing_reserve']),'256MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def emit(self,k,obj):
  self.charge('events')
  if k in('clark_gradient_path_return','native_even_odd_phase','paired_native_phase'):self.charge('native_phases')
  self.save(f'events/{self.calls["events"]:04d}-{k}.json.gz',obj);self.tick()
 def path(self,j,c,s,a,error,b,d):
  self.charge('path');self.emit('path_attempt',dict(matrix=matrix_json(arb_mat(j)),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),error=vector_json(error),parameter=ball_json(b),d=ball_json(d)));return P.coefficients(j,c,s,a,error,b,d,tick=self.tick,observe=self.emit)
 def mixed(self,j,u,b):
  self.charge('mixed');self.emit('mixed_attempt',dict(matrix=matrix_json(arb_mat(j)),input=vector_json(u),parameter=ball_json(b)));return E.augmented(j,u,b,self.emit)


def fixture():
 n=12;j=[[arb(int(i==k))/4+arb((-1)**(i+k))/256 for k in range(n)]for i in range(n)];c=[arb(i+1)/128 for i in range(n)]+[arb(5)/8]+[arb(int(i==0))for i in range(n)]+[arb(4)]+[arb(i-12)/64 for i in range(25)];s=[arb((-1)**i*(i+1))/128 for i in range(51)];a=[arb((-1)**i)/32 for i in range(51)];error=[arb(0,arb(1)/256 if i<26 else arb(4))for i in range(51)]
 return j,c,s,a,error,arb(1)/8192,arb(1)/8192


def synthetic(w):
 j,c,s,a,error,b,h=fixture();w.save('synthetic-inputs.json',dict(matrix=matrix_json(arb_mat(j)),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),error=vector_json(error),parameter=ball_json(b),halfwidth=ball_json(h)));times={};sizes={}
 def measure(k,f):
  begin=time.monotonic();used=w.used;answer=f();times.setdefault(k,[]).append(time.monotonic()-begin);sizes.setdefault(k,[]).append(w.used-used);return answer
 zero=[arb(0)]*51;point=None
 for index,(err,d)in enumerate([(zero,arb(0)),(zero,arb(0,h)),(error,arb(0)),(error,arb(0,h)),(error,arb(0)),(error,arb(0,h))]):
  ans=measure('path',lambda:w.path(j,c,s,a,err,b,d))
  if index==0:point=ans
 for index in range(2):
  box=c if index==0 else[c[i]+error[i]for i in range(51)];ans=measure('mixed',lambda:w.mixed(j,box,b))
  if index==0:
   need(all((point[0][i,0]-ans[0][i,0]).contains(0)for i in range(51))and all((point[1][0][i,k]-ans[1][i,k]).contains(0)for i in range(51)for k in range(51)),'full51 independent MixedJet point value/J agreement')
 def context():
  w.charge('context');jj=[[arb(int(i==k==0))for k in range(12)]for i in range(12)];z=[arb(0)]*12+[arb(1)/2]+[arb(int(i==0))for i in range(12)]+[arb(1)];w.emit('context_attempt',dict(matrix=matrix_json(arb_mat(jj)),input=vector_json(z),parameter=ball_json(arb(0)),tangent=matrix_json(arb_mat(25,1))));out=A.conditional_native(jj,z,arb(0),arb_mat(25,1),observe=w.emit);need(not out['passed']and all(x['witness_power']==1 for x in out['contexts']),'full native context negative primitive control');return out
 measure('context',context)
 for repeat in range(2):
  def algebra():
   w.charge('algebra');n=51;J0=arb_mat([[arb(int(i==k))+arb((i+1)*(k+1))/2**20 for k in range(n)]for i in range(n)]);J1=arb_mat([[arb((-1)**(i+k))/512 for k in range(n)]for i in range(n)]);J2=arb_mat(n,n);C0=midpoint_matrix(J0.inv());C1=midpoint_matrix(-C0*J1*C0);R=[arb_mat(n,1)for _ in range(4)];data=T.reduce([C0,C1],[J0,J1,J2],R,[arb(1)/256]*n,h,tick=w.tick);need(data['passed']and all(data['coordinate_inclusions']),'full51 correlated reduction control');w.emit('synthetic_correlated_reduction',dict(C=[matrix_json(C0),matrix_json(C1)],J=[matrix_json(J0),matrix_json(J1),matrix_json(J2)],R=[matrix_json(x)for x in R],radii=vector_json([arb(1)/256]*n),halfwidth=ball_json(h),result=T.serialize(data)))
  measure('algebra',algebra)
 need(all(w.calls[k]==CAPS[k]for k in('path','mixed','context','algebra','native_phases')),'exact completed full-size profile calls')
 return dict(passed=True,settings=SETTINGS,caps=CAPS,times=times,bytes=sizes,max_seconds={k:max(v)for k,v in times.items()},max_bytes={k:max(v)for k,v in sizes.items()},calls=dict(w.calls),target_arrays_unopened=True,scope='Only unrelated full12/native51 acquisition and reduction costs. No actual target arrays, root, or target admission; subsequent fixed caller needs explicit costed caps and source/card review.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(SETTINGS['as_bytes'],)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,SETTINGS['seconds']);result=dict(passed=False,status='incomplete',target_arrays_unopened=True)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,settings=SETTINGS,caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<SETTINGS['raw_bytes']-SETTINGS['closing_reserve'],'source archive closing reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):report=synthetic(w);w.save('profile.json',report)
  result.update(passed=True,status='complete_unrelated_full51_cost_profile')
 except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
