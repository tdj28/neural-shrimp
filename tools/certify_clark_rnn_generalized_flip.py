#!/usr/bin/env python3
"""One fixed interval Clark26D cubic-zero/quintic/unfolding pilot."""
import gc,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_clark_generalized_flip as A
from neural_shrimp import rigorous_clark_generalized_flip_proof as P
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,canonical_bytes
need=A.require
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-generalized-flip/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-quintic-unfolding-001')
OUTER='48ec71730acbc3d53574d554f45768360ea3e2bd63655c4a5b9d41e1d0db79f7';WORKER='4d7ed2394fd3a921b8ddf1f10bbf67a3d229b3350b032dff2c8d985a32984f09';EXECUTION='276986186b3ce6bac756a7141c4b01105a77f7fc'
MATRIX=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-p6-refinement-001/run/inputs.npz')
# Keep the literal historic input hash independently visible.
MATRIX_PIN='1be92c769bfd21bf23a2a7ff79aca118cd58454b311e4bb095db117c865f265f'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAPS=dict(augmented=9,path5=10,normal=2,context=2,unfolding=2,native_phase_returns=69,events=256)
FILES=['src/neural_shrimp/'+n+'.py'for n in('__init__','controls','models','rigorous_orbits','rigorous_clark_generalized_flip','rigorous_clark_generalized_flip_proof')]+['tools/certify_clark_rnn_generalized_flip.py','tools/launch_clark_rnn_generalized_flip.py','tests/test_rigorous_clark_generalized_flip.py','tests/test_rigorous_clark_generalized_flip_proof.py','tests/test_clark_generalized_flip_driver.py','docs/plans/clark-rnn-n12-generalized-flip.md','docs/plans/clark-rnn-n12-generalized-flip-attempt.md','docs/reviews/clark-rnn-n12-quintic-outcome.json']


def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def exact_float(x):return arb(str(F.from_float(float(x))))
def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable manifest');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(actual)==set(files),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable '+n);tick()
 return dict(root=str(root),manifest=pin,files=len(files),bytes=sum(v['bytes']for v in files.values()))


def point_record(tag):
 paths=[p for p in(PARENT/'run').glob('*-quintic_point.json')if read(p)['tag']==tag];need(len(paths)==1,'literal unique saved '+tag);p=paths[0];md=read(p);a=p.with_name(md['array_file']);need(entry(a)['sha256']==md['array_sha256'],'actual saved full phase array')
 with np.load(a,allow_pickle=False)as f:arrays={n:f[n].copy()for n in f.files}
 need(arrays['physical'].shape==(25,)and arrays['cycle'].shape==(6,12)and arrays['left'].shape==(12,)and all(np.isfinite(v).all()for v in arrays.values()),'complete finite saved full native point')
 return md,arrays,dict(metadata=entry(p),arrays=entry(a))


def parent_gate(out):
 need(out['status']=='completed_fixed_native_quintic_unfolding'and len(out['points'])==9 and len(out['associations'])==8 and out['unfolding']['candidate']and out['unfolding']['strict_cubic_bracket'],'complete fixed numerical organizer candidate')
 need(all(v['passed']and all(v['gates'].values())for v in out['points'])and all(v['reverse_native_passed']and v['reverse']['passed']and v['forward']['passed']and v['reverse_q_error']<1e-7 for v in out['associations']),'full old native and reverse qualifications')


def bind(tick):
 rows=[inventory(PARENT,OUTER,tick),inventory(PARENT/'run',WORKER,tick)];rt=read(PARENT/'run/runtime-start.json');la=read(PARENT/'launch-end.json');out=read(PARENT/'run/result.json');parent_gate(out)
 need(rt['source']['commit']==la['source_commit']==EXECUTION and la['exit_code']==0 and la['owned_group_empty']and la['child_pid_absent'],'exact completed parent')
 for pid in(rt['pid'],la['launcher_pid'],la['child_pid']):need(not Path('/proc',str(pid)).exists(),'old owned PID absent')
 for n,h in rt['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+n],cwd=ROOT)).hexdigest()==h,'historical exact source '+n)
 receipt=read(ROOT/'docs/reviews/clark-rnn-n12-quintic-outcome.json');need(receipt['root']==str(PARENT)and receipt['outer']==OUTER and receipt['worker']==WORKER and receipt['source']==EXECUTION and receipt['outcome']==out,'literal tracked numerical receipt')
 binding=read(PARENT/'run/input-binding.json');need(binding['old_global_phase_failure_preserved']and binding['prior']['prior']['matrix_input_sha256']==MATRIX_PIN and entry(MATRIX)['sha256']==MATRIX_PIN,'original native matrix and retained failed global join')
 with np.load(MATRIX,allow_pickle=False)as f:j=f['j'].copy()
 need(j.shape==(12,12)and j.dtype==np.float64 and np.isfinite(j).all(),'literal full binary64 fixed matrix')
 pts={k:point_record(k)for k in('center','left','right')}
 for tag,(md,arrays,_)in pts.items():
  old=next(v for v in out['points']if v['tag']==tag);need(all(md[k]==v for k,v in old.items())and md['passed'],'literal qualified '+tag);need(float(arrays['physical'][12])==md['delta']and np.array_equal(arrays['cycle'][0],arrays['physical'][:12]),'literal actual native phase')
 left,right=pts['left'][0]['gain'],pts['right'][0]['gain'];need(left<pts['center'][0]['gain']<right and right-left<3e-9 and pts['left'][0]['normal_cubic']>0>pts['right'][0]['normal_cubic'],'original tiny strict cubic bracket')
 md,arrays,pins=pts['center'];seed=[exact_float(v)for v in arrays['physical']]+[exact_float(md['gain'])];jj=[[exact_float(v)for v in row]for row in j]
 return jj,seed,arrays,[exact_float(left),exact_float(right)],dict(inventories=rows,parent_source=EXECUTION,matrix=entry(MATRIX),matrix_path=str(MATRIX),seed_pins={k:v[2]for k,v in pts.items()},gain_bracket=[str(F.from_float(left)),str(F.from_float(right))],map=dict(schema='exact-clark-generalized-flip-v1',j0=[[str(F.from_float(float(v)))for v in row]for row in j]),seed=[str(F.from_float(float(v)))for v in list(arrays['physical'])+[md['gain']]],scope='Same fixed numerical native phase and cubic bracket. Historical continuation remains numerical; local exact proof independently establishes only this selected root.')


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'180s/2GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(256 if closing else 252)*1024**2,'256MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed '+key+' call ceiling');self.calls[key]+=1
 def emit(self,kind,row):
  self.charge('events')
  if kind in('native_augmented_polynomial','native_polynomial','native_context_phase'):self.charge('native_phase_returns')
  self.save(f'events/{self.calls["events"]:03d}-{kind}.json',row);self.tick()
 def augmented(self,j,z):
  self.charge('augmented');self.emit('augmented_attempt',dict(z=vector_json(z)));return A.augmented(j,z,self.emit)
 def path(self,h,j,g,d):
  self.charge('path5');self.emit('path5_attempt',dict(coefficients=[vector_json(v)for v in h],gain=ball_json(g),leak=ball_json(d)));return A.signed_path(h,j,g,d,self.emit)
 def normal(self,x,mat,q,p,evaluate):self.charge('normal');return P.conditional_normal_form(x,mat,q,p,evaluate,self.emit)
 def context(self,j,z,mat,q):self.charge('context');return P.conditional_native_context(j,z,mat,q,observe=self.emit)
 def unfolding(self,J,mat,p):self.charge('unfolding');return P.conditional_unfolding(J,mat,p,self.emit)


def synthetic(w):
 n=12;q=[arb(int(i==0))for i in range(n)];j=[[arb(int(i==k))*(1 if i==0 else arb(1)/2)for k in range(n)]for i in range(n)];z=[arb(0)]*n+[arb(3)/4]+q+[arb(1)];times={};sizes={}
 def measured(k,f):
  old=w.used;t=time.monotonic();answer=f();times.setdefault(k,[]).append(time.monotonic()-t);sizes.setdefault(k,[]).append(w.used-old);return answer
 measured('augmented',lambda:w.augmented(j,z));box=[arb(v,arb((1,-96)))for v in z];measured('augmented',lambda:w.augmented(j,box))
 other=[[arb(int(i==k))/5+arb(1)/200 for k in range(n)]for i in range(n)];zz=[arb(i+1)/50 for i in range(n)]+[arb(3)/5]+q+[arb(7)/5];measured('augmented',lambda:w.augmented(other,zz))
 h=[[arb(i+k+1)/100 for i in range(n)]for k in range(6)]
 for _ in range(5):measured('path5',lambda:w.path(h,other,arb(7)/5,arb(3)/5))
 mat=arb_mat([[(-1 if i==0 else -arb(1)/8)*int(i==k)for k in range(n)]for i in range(n)])
 def polynomial(h):
  x=[v[0]for v in h];x2=P.multiply(x,x);x5=P.multiply(P.multiply(x2,x2),x)
  return [[-h[k][0]+2*x5[k]]+[-h[k][i]/8 for i in range(1,n)]for k in range(6)]
 normal=measured('normal',lambda:w.normal([arb(0)]*n,mat,q,q,polynomial));need(normal['passed'],'known full12D degenerate polynomial coefficient')
 context=measured('context',lambda:w.context([[arb(int(i==k==0))for k in range(n)]for i in range(n)],[arb(0)]*n+[arb(1)/2]+q+[arb(1)],mat,q));need(not context['passed']and context['witness_power']==1 and not any(v['excluded']for v in context['divisors']),'full12D fixedpoint/quotient profile control')
 J=arb_mat(26,26)
 for i in range(n):J[i,i]=mat[i,i]-1;J[n+i,n+1+i]=mat[i,i]+1
 J[12,12]=-3;J[12,25]=-2;J[24,13]=1;J[25,12]=5;J[25,25]=7
 unfold=measured('unfolding',lambda:w.unfolding(J,mat,q));need(unfold['passed'],'full25D critical-curve inverse profile')
 t={k:max(v)for k,v in times.items()};b={k:max(v)for k,v in sizes.items()};forecast=time.monotonic()-w.start+2*(6*t['augmented']+5*t['path5']+t['normal']+t['context']+t['unfolding'])+30;raw=w.used+2*(6*b['augmented']+5*b['path5']+b['normal']+b['context']+b['unfolding'])+8*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 report=dict(passed=bool(forecast<180 and raw<256*1024**2 and peak<2*1024**3),times=times,bytes=sizes,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True,calls=dict(w.calls));w.save('profile.json',report);need(report['passed'],'measured full12D interval proof admission');return report


def exact_root_gate(root,box,cubic):
 need(root['passed']and len(root['coordinate_inclusions'])==26 and all(root['coordinate_inclusions'])and len(box)==26,'exact full26D root before any conditional claim');need(box[-1]>0 and 0<box[12]<1 and cubic.contains(0),'actual same-root physical domain and cubic consistency')


def association(box,context,arrays,bracket):
 need(len(box)==26 and box[-1]>bracket[0]and box[-1]<bracket[1],'exact organizer gain inside original strict numerical bracket');old=[exact_float(v)for v in arrays['physical']];scale=max([arb(1)]+[abs(v).upper()for v in old[:12]]);dist=max(abs(box[i]-old[i]).upper()for i in range(12))/scale;mode=sum(((box[13+i]-old[13+i])**2 for i in range(12)),arb(0)).sqrt().upper();leak=abs(box[12]-old[12]).upper();orbit=[[arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))for v in row]for row in context['orbit_enclosures']]
 errors=[max(abs(orbit[k][i]-exact_float(arrays['cycle'][k,i])).upper()for i in range(12))/scale for k in range(6)]
 need(dist<arb('1e-6')and mode<arb('1e-6')and leak<arb('1e-8')and all(v<arb('1e-6')for v in errors),'literal same-phase whole box/cycle/mode numerical association')
 return dict(passed=True,state_error=ball_json(dist),mode_error=ball_json(mode),leak_error=ball_json(leak),native_phase_errors=vector_json(errors),scope='Interval bounds on proximity to exact dyadic saved numerical states, with no phase/sign relabeling. This does not promote the earlier numerical fold/flip branch to an interval connection.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,180);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'exact source archive');need(w.used+len(raw)<252*1024**2,'source cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   synthetic(w);j,seed,arrays,bracket,binding=bind(w.tick);w.save('inputs.json',binding)
   box,J,data,root=P.root_enclosure(lambda z:w.augmented(j,z),seed,newton_steps=4,radius_exponent=-96,observe=w.emit,tick=w.tick);w.save('root.json',root);exact_root_gate(root,box,data['cubic'])
   q=box[13:25];context=w.context(j,box,data['monodromy'],q);w.save('native-context.json',context);need(context['passed'],'exact native primitive6 and stable11 complement');join=association(box,context,arrays,bracket);w.save('association.json',join)
   normal=w.normal(box[:12],data['monodromy'],q,data['left'],lambda h:w.path(h,j,box[-1],box[12]));w.save('quintic.json',normal);need(normal['passed'],'same exact cubic-zero nonzero quintic')
   unfold=w.unfolding(J,data['monodromy'],data['left']);w.save('unfolding.json',unfold);need(unfold['passed'],'same exact physical two-rate transversality');w.tick()
   result.update(passed=True,status='primitive6_signed_generalized_flip_with_transverse_physical_unfolding',dimension=12,augmented_dimension=26,primitive_period=6,critical_signed_multiplier=-1,critical_native_multiplier=1,complement_dimension=11,gain=ball_json(box[-1]),leak=ball_json(box[12]),quintic=normal['quintic'],physical_determinant=unfold['physical_determinant'],scope='Exact-real local G=-F^3 cubic-zero generalized flip, nonzero quintic/physical two-parameter unfolding, primitive native6 and stable11-mode complement. Same saved native phase by bounded numerical proximity; no rigorous global fold/cascade connection or full shrimp-window claim. Independent outcome audit pending.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
