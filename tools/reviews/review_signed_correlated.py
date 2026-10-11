"""Fixed saved independent512 correlated signed470 root and full known-root join."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_signed_correlated_math as D
import review_signed470_advancing as A
import review_correlated705 as K
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,vector_json
H,R,M,U,need=D.H,D.R,D.H.M,D.U,D.need
read,entry,inventory,archive=A.read,A.entry,A.inventory,A.archive
OUT=Path('/data2/neural-shrimp-data/audits/signed470-correlated-independent/attempt001/run');CARD='docs/plans/native470-correlated-independent-inputs.json'
LIMITS=dict(seconds=300,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=768*1024**2,precision=512,native_calls=0)
FILES=sorted(set(A.FILES+K.FILES+[CARD,'docs/plans/native470-correlated-independent.md','tools/reviews/review_signed_correlated.py','tools/reviews/review_signed_correlated_math.py','tools/launch_signed_correlated_review.py','tests/reviews/test_signed_correlated_review.py','tests/reviews/test_signed_correlated_driver.py']))
class Work(K.Work):
 def tick(self):need(not self.stopped and time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'300s6GiB')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(768 if closing else 764)*1024**2,'768MiB4MiBclosing');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 # Byte-identical streaming implementation in K uses its stricter636MiB inner
 # bound, retained conservatively; this caller enforces the outer768MiB cap.
 def matrix(self,name,a):
  old=K.OUT;K.OUT=OUT
  try:return super().matrix(name,a)
  finally:K.OUT=old

def synthetic(w):
 n=470;den=arb((1,270));B=arb_mat([[arb(((i+1)*6364136223846793005+(j+1)*1442695040888963407)**4%2**250)/den for j in range(n)]for i in range(n)]);I=M.ident(n);C=I+B;J=[I-B,B/8,B/32];res=[arb_mat([[arb((1,-80))*(i+1)/n]for i in range(n)])]+[arb_mat(n,1)for _ in range(3)];rho=arb(1)/4096;h=arb(1)/8;t=time.monotonic();used=w.used;r=H.polynomial_root(C,J,res,[arb(1)]*n,rho,h,[arb(0,rho)]*n,w.tick,w.emit);need(r['passed'],'full470 unrelated polynomial proof');dt=time.monotonic()-t;size=w.used-used;forecast=dict(seconds=time.monotonic()-w.start+3*dt+80,bytes=w.used+3*size+48*1024**2,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+2*1024**2);ok=forecast['seconds']<300 and forecast['bytes']<636*1024**2 and forecast['rss_kib']<6*1024**2;w.save('profile.json',dict(passed=ok,seconds=dt,bytes=size,forecast=forecast,target_arrays_unopened=True,native_calls=0));need(ok,'full470 measured arithmetic+storage profile before arrays');del B,C,J,res;gc.collect()

def bind(card,w,metadata_only=False):
 need(card['status']=='reviewed'and card['limits']==LIMITS,'literal reviewed fixed audit');known=card['known'];ap=Path(known['root']);spec=card['producer'];p=Path(spec['root'])
 for root,s in((ap,known),(p,spec)):
  inventory(root,s['outer']['sha256'],w);inventory(root/'run',s['worker']['sha256'],w);archive(root,s['source'],w);la=read(root/'launcher-result.json');need(la['exit_code']==0 and la['failure']is None and la['worker_pid_absent'],'successful owned execution')
 ai=read(ap/'run/input-binding.json');ao=read(ap/'run/outcome.json');need(ao==known['outcome']and ao['passed']and ao['root_coordinates']==ao['anchor_coordinates']==470 and ao['native_calls']==0,'entire independent known-root authority')
 class Prefix:
  def tick(self):w.tick()
  def save(self,n,v,closing=False):w.save('prior-'+n,v,closing)
 A.binding(ai['card'],Prefix())
 need(entry(ap/'run/connected-left-endpoint.json')==known['endpoint']and entry(ap/'run/independent512.json')==known['proof'],'literal independent root endpoint/proof')
 o=read(p/'run/outcome.json');need(o==spec['outcome']and o['passed']and o['strict_inclusions']==o['anchor_inclusions']==470 and o['native_calls']==324 and o['ticks']==336 and o['events']==1096,'complete correlated target/profiles')
 for n,e in spec['files'].items():need(entry(p/'run'/n)==e,'literal acquired file '+n)
 ib=read(p/'run/input-binding.json');inp=read(p/'run/target-inputs.json');rt=read(p/'run/runtime.json');need(ib['independent']==known and ib['map_inputs']==inp['map_inputs']==ai['model']and ib['native_source']==ai['native_source'],'same model and complete original known-root authority')
 for n,e in ai['native_source'].items():need(rt['sources'][n]==e,'literal unchanged native map '+n)
 need(inp['settings']==dict(period=18,sigma=-1,halfwidth_exponent=-3,radius_exponent=-12,newton_steps=2,precision=256,parameter='-49/256',known_parameter='-17/256')and inp['normal']==['0']*468+['1','0'],'fixed complete correlated geometry')
 w.save('input-binding.json',dict(card=card,model=inp['map_inputs'],native_source=ai['native_source'],exact_known_parameter='-17/256',scope='Full independently proved known endpoint under literal identical native gauge. No projected-scalar shortcut across changed gauges.'))
 if metadata_only:return p,ap
 endpoint=read(ap/'run/connected-left-endpoint.json');proof=read(ap/'run/independent512.json');need(endpoint['box']==proof['recomputed_endpoints'][0]and endpoint['exact_parameter']=='-17/256'and endpoint['normal']==inp['normal']and endpoint['origin']==inp['origin']and endpoint['sigma']==-1 and endpoint['period']==18 and endpoint['dimension']==13,'whole independent root and exact same gauge parameter');return p,endpoint,inp

def acquisitions(p):
 events=[(f.name.split('-target-')[1].removesuffix('.json.gz'),read(f))for f in sorted((p/'run/events').glob('*-target-*.json.gz'))]
 by={}
 for k,v in events:by.setdefault(k,[]).append(v)
 expected=dict(native_input=144,native_return=144,phase_return=54,path_phase_return=54,signed_gradient_path_return=36,signed_gradient_path_input=36,signed_proposal_input=3,path_assembly_return=3,signed_proposal_return=3,signed_gradient_path_assembly_return=2,signed_affine_proposal_input=1,signed_predictor_return=1,signed_correlated_domain=1)
 need({k:len(v)for k,v in by.items()}==expected,'every target event retained')
 need([v['index']for v in by['native_input']]==[v['index']for v in by['native_return']]==list(range(144)),'all literal144 native input/return indices')
 gradients=[];paths=[]
 for i,(kind,v)in enumerate(events):
  if kind=='signed_gradient_path_input':
   seq=events[i+1:i+4];need([k for k,_ in seq]==['native_input','native_return','signed_gradient_path_return'],'literal nested gradient acquisition order');ni,no,ret=[r for _,r in seq];need(ni['index']==no['index']and v['phase']==ret['phase'],'gradient phase/index')
   need([{k:x for k,x in z.items()if k!='kind'}for z in ni['inputs']]==v['inputs']and[{k:x for k,x in z.items()if k!='kind'}for z in no['returns']]==ret['returns']and all(z['kind']=='gradient_path'for z in ni['inputs']+no['returns']),'outer gradient tied to actual native payload');gradients.append(dict(**v,returns=ret['returns']))
  if kind=='path_phase_return':
   need(i>=2 and [k for k,_ in events[i-2:i]]==['native_input','native_return'],'literal native path acquisition order');ni,no=[r for _,r in events[i-2:i]];need(ni['index']==no['index']and all(z['kind']=='path'for z in ni['inputs']+no['returns'])and[[z['a'],z['b']]for z in ni['inputs']]==v['inputs']and[[z['a'],z['b']]for z in no['returns']]==v['returns'],'path acquisition exact payload');paths.append(v)
 return by['signed_correlated_domain'][0],by['signed_predictor_return'][0],gradients,paths[-36:]

def replay(card,w):
 p,known,inp=bind(card,w);domain,proposal,grad,paths=acquisitions(p);chart=read(p/'run/target-certificate.json.gz');need(chart['schema']=='neural-shrimp-correlated-signed-eigenpair-v1'and chart['sigma']==-1 and chart['shooting_dimension']==470 and chart['settings']==dict(halfwidth_exponent=-3,radius_exponent=-12)and chart['parameter_endpoints']==['-81/256','-17/256'],'literal fixed original correlated claim')
 need(chart['center']==domain['center']==proposal['center']and chart['slope']==domain['slope']==proposal['slope']and chart['curvature']==domain['curvature']==proposal['curvature']and chart['normal']==inp['normal']and chart['origin']==inp['origin']and chart['weights']==inp['weights'],'entire predictor and phase/rate/gauge proposal')
 c,s,a=[[R.exact(M.point(v))for v in chart[k]]for k in('center','slope','curvature')];normal=[R.exact(F(v))for v in chart['normal']];origin=[R.exact(F(v))for v in chart['origin']];weights=[R.exact(F(v))for v in chart['weights']];h=arb(1)/8;rho=arb(1)/4096;d=arb(0,h);error=[R.original_ball(v)for v in domain['error']];need(all(len(v)==470 for v in(c,s,a,normal,origin,weights,error))and all(w>0 for w in weights),'all470 actual coordinates')
 for old,weight in zip(domain['error'],weights,strict=True):
  lo,hi=M.interval(old);rr=M.point(ball_json(rho*weight));need(lo==-hi and lo<=-rr and rr<=hi,'actual full correction E covers each requested radius')
 whole=[c[i]+s[i]*d+a[i]*d*d/2+error[i]for i in range(470)]
 for outer,inner in zip(domain['whole_box'],whole,strict=True):U.contains(outer,ball_json(inner),'actual fullE curved tube')
 need(domain['whole_box']==chart['whole_box']and domain['parameter']==chart['parameter_center']==inp['parameter']=='-49/256'and domain['sigma']==-1 and whole[-2]>0 and whole[-1]>0,'same physical domains/gauge')
 zero=D.jacobian(c,s,a,error,arb(0),grad[:18],13,18,-1,normal,w.tick);wide=D.jacobian(c,s,a,error,d,grad[18:],13,18,-1,normal,w.tick);J=[zero[0],zero[1],wide[2]];del zero,wide;gc.collect()
 coeff=D.residual(c,s,a,arb(0),paths[:18],13,18,-1,normal,origin,R.exact(F('-49/256')),w.tick);wholecoeff=D.residual(c,s,a,d,paths[18:],13,18,-1,normal,origin,R.exact(F('-49/256')),w.tick);res=[arb_mat([[coeff[i,k]if k<3 else wholecoeff[i,3]]for i in range(470)])for k in range(4)]
 raw=chart['inverse_polynomial_exact'][0];need(chart['inverse_polynomial_encoding']=='exact-dyadic-pairs'and raw==proposal['preconditioner_exact']and len(raw)==470 and all(len(row)==470 for row in raw),'exact original C0 finite proposal');C0=arb_mat([[arb(tuple(v))for v in row]for row in raw]);need(all(v.is_exact()for row in C0.tolist()for v in row),'all original inverse entries exact');w.emit('acquired_J0',J[0]);w.emit('acquired_J1',J[1]);w.emit('acquired_J2',J[2]);proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit)
 ac=D.anchor(known['box'],c,s,a,normal,origin,R.exact(F('-49/256')),h,[rho*v for v in weights],exact_parameter='-17/256');proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[26*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];gates=dict(root=proof['passed'],known_root=ac['passed']and ac['coordinates']==470,primitive=all(v['coordinates']for v in proper));ends=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  dd=sign*h;ends.append(vector_json([c[i]+s[i]*dd+a[i]*dd*dd/2+R.original_ball(corr[i])for i in range(470)]))
 w.save('independent512.json',dict(passed=all(gates.values()),gates=gates,root=proof,anchor=ac,primitive=proper,recomputed_endpoints=ends));w.save('connected-left-endpoint.json'if all(gates.values())else'unqualified-endpoint.json',dict(box=ends[0],normal=chart['normal'],origin=chart['origin'],exact_parameter='-81/256',dimension=13,period=18,sigma=-1));return dict(passed=all(gates.values()),root_coordinates=sum(proof['coordinate_inclusions']),anchor_coordinates=ac['coordinates'],contraction=proof['contraction'],native_calls=0,scope='Independent saved512 fullE native Jacobian/path assembly and polynomial proof plus exact-gauge full470 known-root join. Shared native derivatives/Arb; no complement/genericity/global closure.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('300s owned correlated470 saved audit')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact current source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):synthetic(w);w.stage='target';ans=replay(read(ROOT/CARD),w)
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<300 and peak<6*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
