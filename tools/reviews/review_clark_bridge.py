"""One fixed independent local Clark bridge replay; shared family, own reductions."""
import hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import numpy as np
from flint import arb,arb_mat,ctx
import flint
import review_clark_bridge_math as B
M=B.M;need=M.need
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/critical-organizer-bridge/attempt002')
SOURCE='60d2cdb8b3575ed9034c090f07f121a2edc6ed74'
PINS=['d26b3603f6df5db6b054347244e9293b2bb13b44cc3cf055844b4223df5cb617','5c173ff6dab3aac842bea80cdd05f254d79231eb8b28a4370b3b2d4ff26011ec']
OUT=Path('/data2/neural-shrimp-data/audits/clark-critical-bridge-independent/attempt001/run')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/reviews/review_clark_bridge.py','tools/reviews/review_clark_bridge_math.py','tools/reviews/review_clark_organizer_math.py','tools/launch_clark_bridge_review.py','tests/test_clark_bridge_review.py','tests/test_clark_bridge_review_driver.py','docs/plans/clark-critical-bridge-independent.md','docs/reviews/clark-critical-bridge-outcome.json']
EXPECTED={'native_polynomial':36,'native_context_phase':12,'family_attempt':12,'native_transport_phase':12,'critical_family_return':12,'center_newton_proposal':2,'validated_inverse':2,'native_context_result':2,'critical_tube_return':2,'native_transport_return':2,'newton_proposal':2,'root_enclosure':1}
def read(p):return json.loads(p.read_bytes())
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def serial(v):
 if isinstance(v,arb):return dict(mid=[int(x)for x in v.mid().man_exp()],rad=[int(x)for x in v.rad().man_exp()],display=v.str(22))
 if isinstance(v,arb_mat):return [[serial(x)for x in row]for row in v.tolist()]
 if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [serial(x)for x in v]
 return v
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=0;self.events=0
 def tick(self):need(not self.stopped and time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'30s/1GiB cap')
 def save(self,n,v,closing=False):
  raw=(json.dumps(serial(v),sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(self.used+len(raw)<(32 if closing else 30)*1024**2,'32MiB with2MiB closing reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);self.used+=len(raw)
 def emit(self,k,v):
  self.tick();self.events+=1;need(self.events<=128,'128event cap')
  if k=='independent_native_phase':self.calls+=1;need(self.calls<=18,'18inclusive independent native phases')
  self.save(f'events/{self.events:03d}-{k}.json',v)
def inventory(root,pin,w):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable inventory');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(actual),'complete original inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'actual immutable bytes '+n);w.tick()
def archive(root,source,w):
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');need(rt['source_commit']==la['source_commit']==source and la['exit_code']==0 and not la['owned_live_pids'],'complete literal successful source')
 for pid in [rt['pid'],la['launcher_pid'],la['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'old owned PID absent')
 actual={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};need(set(actual)==set(rt['sources']),'complete source archive')
 for n,pin in rt['sources'].items():need(entry(actual[n])==pin and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'historical source '+n);w.tick()
 return rt

def family_record(run,start,end):
 a=read(run/f'events/{start:03d}-family_attempt.json');z=read(run/f'events/{end:03d}-critical_family_return.json');need(z['input']==a['input']and z['parameter']==a['parameter'],'literal acquisition input/return pairing')
 names=[run/f'events/{k:03d}-native_polynomial.json'for k in range(start+1,end)];need(end-start==4 and all(p.is_file()for p in names),'all three native signed phases');return z

def bind(w):
 for r,pin in zip([PARENT,PARENT/'run'],PINS):inventory(r,pin,w)
 rt=archive(PARENT,SOURCE,w);run=PARENT/'run';out=read(run/'outcome.json');need(out['passed']and out['calls']==dict(context=2,events=97,family=12,native_phase_returns=60,transport=2),'complete original inclusive scientific inventory')
 events=sorted((run/'events').iterdir());need(len(events)==97 and [int(p.name[:3])for p in events]==list(range(1,98))and Counter(p.stem.split('-',1)[1]for p in events)==EXPECTED,'exact all97 acquisition events')
 for start in [1,6,11,32,38,44,49,54,68,74,80,85]:family_record(run,start,start+4)
 t=read(run/'tube.json');ep=read(run/'endpoint-root.json');need(len(ep['center'])==len(ep['box'])==26 and t['dimension']==26,'full26 fixed dimensions');need(t==read(run/'events/059-critical_tube_return.json')and ep==read(run/'events/090-root_enclosure.json'),'literal retained proof objects')
 for start,end,key,mat in [(44,48,'center','center_jacobian'),(49,53,'predictor_box','predictor_jacobian'),(54,58,'whole_box','whole_jacobian')]:
  a=family_record(run,start,end);need(a['input']==t[key]and a['jacobian']==t[mat],'actual family proof domain/value binding')
  if key=='center':need(a['value']==t['center_value']and a['parameter']==t['parameter_center'],'exact tube center residual/gain')
 need(family_record(run,49,53)['parameter']==family_record(run,54,58)['parameter']==t['predictor_box'][-1],'same entire actual gain domain')
 for start,end,key in [(80,84,'center'),(85,89,'box')]:
  a=family_record(run,start,end);need(a['parameter']==ep['center'][-1],'same exact fixed endpoint gain');need(a['input']==ep[key]and a['value']==ep['center_value'if key=='center'else'whole_value'],'exact endpoint value/root domain')
  if key=='box':need(a['jacobian']==ep['whole_jacobian'],'exact endpoint wholeJ')
 binding=read(run/'input-binding.json');card=binding['card']
 for row in binding['inventories']:inventory(Path(row['root']),row['manifest'],w)
 weighted=read(run/'weighted-radius-binding.json')
 for row in weighted['inventories']:inventory(Path(row['root']),row['manifest'],w)
 independent=card['independent'];ir=Path(independent['root']);archive(ir,independent['source'],w);old=read(ir/'run/outcome.json');same=read(ir/'run/same-root.json');fresh=read(ir/'run/fresh-root.json')
 need(old['passed']and old['primitive_period']==6 and old['complement_dimension']==11 and same['all26_original_boxes_strictly_inside']and fresh['passed']and len(fresh['coordinate_inclusions'])==26 and all(fresh['coordinate_inclusions']),'independently established exact organizer authority')
 need(read(ir/'run/input-binding.json')['map']==binding['map'],'identical physical map and matrix')
 known=[M.decode(v)for v in fresh['box']];need(len(known)==len(binding['organizer_box'])==26 and all(B.covered(a,b)for a,b in zip(binding['organizer_box'],known)),'same complete exact organizer domain')
 # Every original rate/matrix entry is dyadic from the immutable binary64 source.
 j=[[arb(v)for v in row]for row in binding['map']['j0']];need(len(j)==12 and all(len(row)==12 and all(v.is_exact()for v in row)for row in j),'exact full original12D matrix')
 matrix_path=Path(card['matrix']['path']);need(entry(matrix_path)==card['matrix']['entry'],'literal original matrix bytes')
 with np.load(matrix_path,allow_pickle=False)as f:matrix=f['j'].copy()
 need([[str(F.from_float(float(v)))for v in row]for row in matrix]==binding['map']['j0'],'all binary64 matrix entries identical')
 numeric=Path(card['numerical']['root'])/'run';ap=numeric/card['numerical']['array'];need(entry(ap)==binding['saved_array']and entry(ap)['sha256']==card['numerical']['array_sha256'],'literal original six native phases')
 with np.load(ap,allow_pickle=False)as f:arrays={k:f[k].copy()for k in f.files}
 need(arrays['cycle'].shape==(6,12)and arrays['physical'].shape==(25,)and all(a.dtype==np.float64 and np.isfinite(a).all()for a in arrays.values()),'full finite native association arrays')
 w.save('input-binding.json',dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],card=card,original_events=97,shared_family_derivative_source=rt['sources'],tube=entry(run/'tube.json'),endpoint=entry(run/'endpoint-root.json'),known_organizer=independent,native_derivative_scope='Own cosh native6 derivatives; source-authenticated augmented family acquisitions and Arb shared.'))
 return t,ep,known,j,arrays,card

def synthetic(w):
 n=26;c=[arb(1)]*n;v=[arb(2)]*(n-1)+[arb(1)];r=[arb('1/32')]*(n-1)+[arb('1/1024')];h=arb('1/16');d=arb(0,h);E=[arb(0,x)for x in r];p=[c[i]+v[i]*d for i in range(n)];whole=[p[i]+E[i]for i in range(n)]
 def jac(g):return arb_mat([[int(i==k)-2*g*int(k==n-1)for k in range(n)]for i in range(n-1)]+[[int(k==n-1)for k in range(n)]])
 t=serial(dict(center=c,slope=v,radii=r,parameter_center=arb(1),halfwidth=h,actual_error=E,predictor_box=p,whole_box=whole,center_value=arb_mat(n,1),center_jacobian=jac(arb(1)),predictor_jacobian=jac(p[-1]),whole_jacobian=jac(whole[-1])))
 begin=time.monotonic();used=w.used;B.uniform(t,w.emit);algebra=time.monotonic()-begin
 n=12;j=[[arb(int(i==k==0))for k in range(n)]for i in range(n)];z=[arb(0)]*n+[arb('1/2')]+[arb(int(i==0))for i in range(n)]+[arb(1)];begin=time.monotonic()
 try:M.native(j,z,w.emit)
 except ValueError as exc:need(str(exc)=='all native6 proper divisors','expected unrelated fixed-point negative control')
 else:raise ValueError('negative primitive profile unexpectedly accepted')
 native=time.monotonic()-begin;forecast=6*algebra+4*native+10;raw=6*(w.used-used)+4*1024**2;peak=4*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+64*1024**2
 out=dict(passed=bool(forecast<30 and raw<32*1024**2 and peak<1024**3),algebra_seconds=algebra,native6_seconds=native,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,native_calls=w.calls,target_arrays_unopened=True);w.save('profile.json',out);need(out['passed']and w.calls==6,'measured full26/native12 admission')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS)and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed CPU1nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=512,python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=10,native_phase_cap=18,shared_augmented_calls=0))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):
   synthetic(w);t,ep,known,j,arrays,card=bind(w);uniform=B.uniform(t,w.emit);w.save('uniform.json',uniform);organizer=B.join(known,uniform);w.save('organizer-join.json',organizer);need(organizer['passed']and organizer['coordinates']==26,'exact independently known organizer identity')
   center=[M.decode(v)for v in ep['center']];box=[M.decode(v)for v in ep['box']];point=M.krawczyk(center,box,M.matrix(ep['center_value']),M.matrix(ep['whole_jacobian']),w.emit,exact_bounds=[B.endpoints(v)for v in ep['box']]);w.save('endpoint-root.json',point)
   _,_,context=M.native(j,uniform['whole'],w.emit);w.save('whole-native.json',context);_,_,native=M.native(j,box,w.emit);w.save('endpoint-native.json',native)
   direct=B.both_orientations(box,uniform);returned,norm=B.return_box(box,native);back=B.both_orientations(returned,uniform);sign=B.choose(direct,back);w.save('endpoint-joins.json',dict(direct=direct,native_return=back,positive_norm=norm,orientation=sign))
   exact=lambda v:arb(str(F.from_float(float(v))));scale=max([arb(1)]+[abs(exact(v)).upper()for v in arrays['physical'][:12]]);errors=[max(abs(native['orbit'][k][i]-exact(arrays['cycle'][k,i])).upper()for i in range(12))/scale for k in range(6)];qerr=sum(((box[13+i]-exact(arrays['physical'][13+i]))**2 for i in range(12)),arb(0)).sqrt().upper();leak=abs(box[12]-exact(arrays['physical'][12])).upper();need(all(e<arb('1e-6')for e in errors)and qerr<arb('1e-6')and leak<arb('1e-8'),'literal full native phases/mode/leak association');w.save('native-association.json',dict(phase_errors=errors,mode_error=qerr,leak_error=leak));need(w.calls==18,'all18 independent native phases')
   result.update(passed=True,status='independent_exact_local_bridge_pass',strict_inclusions=26,organizer_join=26,endpoint_join=26,native_return_join=26,mode_orientation=sign,contraction=uniform['contraction'],whole_quotient_power=context['power_bounds'][-1],primitive_period=6,critical_signed_multiplier=-1,critical_native_multiplier=1,scope='Independent512 uniform local critical curve linked to same independently exact organizer and original saved-phase endpoint root. Shared source-authenticated augmented derivative acquisitions/Arb; independent reductions and cosh native orbit/derivatives. No rigorously connected remote fold/cascade or full shrimp window.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.calls,events=w.events,shared_augmented_calls=0);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(serial(result)));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
