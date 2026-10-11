#!/usr/bin/env python3
"""Finite adaptive squared-amplitude fold bridge from one exact organizer."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
import certify_clark_critical_bridge as B
from neural_shrimp import rigorous_clark_amplitude_fold as A
from neural_shrimp import rigorous_clark_amplitude_tube as T
from neural_shrimp import rigorous_clark_generalized_flip_proof as P
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,canonical_bytes,midpoint_matrix
need=A.require;entry=B.entry;read=B.read;decode=B.decode
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/amplitude-fold-bridge/attempt001/run')
CARD='docs/plans/clark-amplitude-fold-inputs.json';THREADS=B.THREADS
SETTINGS=dict(precision=384,seconds=180,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=512*1024**2,attempts=12,accepted_charts=8,initial_halfwidth_exponent=-13,minimum_halfwidth_exponent=-17,last_amplitude_squared='1/2048',center_newton_steps=2,endpoint_newton_steps=3,endpoint_radius_exponent=-80,radius_floor_exponent=-40,radius_ceiling_exponent=-8)
CAPS=dict(family=93,context=14,native_phase_returns=642,events=2048)
FILES=sorted(set(B.FILES+[CARD,'docs/plans/clark-desingularized-fold-continuation.md','docs/reviews/clark-critical-bridge-independent.json','tools/certify_clark_amplitude_fold.py','tools/launch_clark_amplitude_fold.py','tests/test_clark_amplitude_fold_driver.py','tests/test_rigorous_clark_amplitude_fold.py','tests/test_rigorous_clark_amplitude_tube.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'180s/2GiB cap')
 def save(self,name,value,closing=False):
  raw=canonical_bytes(value)
  if name.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(512 if closing else 508)*1024**2,'512MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed inclusive '+key+' cap');self.calls[key]+=1
 def emit(self,kind,row):
  self.charge('events')
  if kind in('native_even_odd_phase','paired_native_phase'):self.charge('native_phase_returns')
  self.save(f'events/{self.calls["events"]:04d}-{kind}.json.gz',row);self.tick()
 def family(self,j,z,b):
  self.charge('family');self.emit('family_attempt',dict(input=vector_json(z),squared_amplitude=ball_json(b)))
  return A.augmented(j,z,b,self.emit)
 def context(self,j,z,b,v,support=None):
  self.charge('context');self.emit('context_attempt',dict(input=vector_json(z),squared_amplitude=ball_json(b),paired_tangent=matrix_json(v),support=None if support is None else vector_json(support)))
  return T.conditional_native(j,z,b,v,support=support,observe=self.emit)


def synthetic(w):
 n=12;q=[arb(int(i==0))for i in range(n)];j=[[arb(int(i==k))/4+arb(1)/256 for k in range(n)]for i in range(n)];z=[arb(i+1)/50 for i in range(n)]+[arb(3)/5]+q+[arb(7)/5];times={};sizes={}
 def measure(k,f):
  before=time.monotonic();old=w.used;answer=f();times.setdefault(k,[]).append(time.monotonic()-before);sizes.setdefault(k,[]).append(w.used-old);return answer
 for i in range(3):
  box=z if i==0 else[arb(v,arb((1,-12 if i==1 else -10)))for v in z];b=arb(0)if i==0 else arb('1/4096',arb((1,-13)))
  measure('family',lambda:w.family(j,box,b))
 jj=[[arb(int(i==k==0))for k in range(n)]for i in range(n)];zz=[arb(0)]*n+[arb('1/2')]+q+[arb(1)];negative=measure('context',lambda:w.context(jj,zz,arb(0),arb_mat(25,1)))
 need(not negative['passed']and all(c['witness_power']==1 for c in negative['contexts']),'full12 fixedpoint negative native profile')
 def quadratic(z,b):return arb_mat([[x-b*b]for x in z]),arb_mat([[int(i==k)for k in range(26)]for i in range(26)]),arb_mat([[-2*b]for _ in z]),None
 trial=measure('algebra',lambda:T.tube(quadratic,[arb(1)]*26,arb(1),arb('1/16'),[arb('1/64')]*26,observe=w.emit,tick=w.tick));need(trial[-1]['passed'],'full26 explicit-parameter tube algebra')
 maxt={k:max(v)for k,v in times.items()};maxb={k:max(v)for k,v in sizes.items()};forecast=time.monotonic()-w.start+2*(90*maxt['family']+13*maxt['context']+48*maxt['algebra'])+20;raw=w.used+2*(90*maxb['family']+13*maxb['context']+48*maxb['algebra'])+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 report=dict(passed=bool(forecast<180 and raw<512*1024**2 and peak<2*1024**3),times=times,bytes=sizes,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True,calls=dict(w.calls));w.save('profile.json',report);need(report['passed'],'measured full12/26 whole-packet admission before target arrays')


def bind(w):
 card=read(ROOT/CARD);need(entry(ROOT/B.CARD)==card['base_card'],'literal already reviewed organizer genealogy');j,known,critical_seed,critical_arrays=B.bind(w)
 base=read(ROOT/B.CARD);ip=Path(base['independent']['root'])/'run';unfold=read(ip/'unfolding.json');quintic=read(ip/'quintic.json')
 authority=read(ip/'outcome.json');need(authority['passed']and unfold['multiplier_leak_derivative']==authority['multiplier_leak_derivative']and quintic['quintic']==authority['quintic']and decode(unfold['multiplier_leak_derivative']).abs_lower()>0 and decode(quintic['quintic'])>0,'same exact cubic-zero/simple-minus root with strict physical leak derivative and positive quintic')
 # Both old and independently recomputed critical-curve endpoint are authority
 # for the same physical gain/phase; the new fold root remains separate.
 pin=card['bridge'];root=Path(pin['root']);invs=[B.inventory(root,pin['outer'],w.tick),B.inventory(root/'run',pin['worker'],w.tick)];B.historical(root,pin['source']);out=read(root/'run/outcome.json');ib=read(root/'run/input-binding.json')
 need(out['passed']and out['strict_inclusions']==26 and out['organizer_join']==26 and out['endpoint_join']==26 and out['native_return_join']==26 and out['mode_orientation']==1,'independent exact critical-curve connection')
 rec=read(ROOT/'docs/reviews/clark-critical-bridge-independent.json');need(rec['root']==pin['root']and rec['source']==pin['source']and rec['outcome']==out,'literal independent bridge receipt')
 endpoint=read(root/'run/endpoint-root.json');need(endpoint['passed']and len(endpoint['box'])==26 and all(endpoint['coordinate_inclusions']),'exact old critical endpoint');oldgain=decode(endpoint['box'][-1]);need(oldgain.contains(critical_seed[-1]),'same physical endpoint gain')
 spec=card['numerical_fold'];num=Path(base['numerical']['root'])/'run';p=num/spec['point'];need(entry(p)==spec['point_identity'],'literal unchanged native fold record');md=read(p);old=read(num/'result.json')['points'][-1]['fold'];need(all(md[k]==v for k,v in old.items())and md['tag']=='fold16'and md['passed']and all(md['gates'].values())and md['gain']==spec['gain']==4.0756,'fixed full native numerical fold')
 ap=num/spec['array'];need(md['array_file']==spec['array']and entry(ap)==spec['array_identity']and md['array_sha256']==spec['array_identity']['sha256'],'actual complete fold array bytes')
 with np.load(ap,allow_pickle=False)as f:arrays={k:f[k].copy()for k in f.files}
 shapes=dict(physical=(25,),paired_physical=(25,),partner_mode=(12,),cycle_x=(6,12),cycle_y=(6,12),projection=(12,));need(set(arrays)==set(shapes)and all(a.shape==shapes[k]and a.dtype==np.float64 and np.isfinite(a).all()for k,a in arrays.items()),'complete literal both-partner native arrays')
 p=arrays['paired_physical'];need(np.array_equal(p[:12],arrays['physical'][:12])and np.array_equal(p[:12],arrays['cycle_x'][0])and np.array_equal(p[12:24],arrays['cycle_y'][0])and p[24]==arrays['physical'][12]==md['delta'],'same actual native phase/leak')
 w.save('fold-input-binding.json',dict(card=card,independent_bridge=invs,organizer_unfolding=entry(ip/'unfolding.json'),organizer_quintic=entry(ip/'quintic.json'),symbolic_anchor='At exact b=0 with the same normalized q and phase, c3 + lambda_leak*leak_b = 0. Existing exact c3=0 and lambda_leak!=0 imply all26 new equations. No zero-containment identity inference.',fold_record=spec['point_identity'],fold_arrays=spec['array_identity'],old_critical_gain=ball_json(oldgain),scope='Literal prior root/source/matrix/phase authority; numerical fold arrays are a seed and later association only.'))
 return j,known,critical_seed[-1],arrays


def dyadic_radius(value):
 need(value.is_finite()and value>=0,'finite nonnegative radius proposal');r=arb((1,-40))
 while r<=value:r*=2
 need(r<=arb((1,-8)),'fixed dyadic proposal radius ceiling');return r


def prepare(w,j,seed,known,left,center_b,h):
 c=[x.mid()for x in seed]
 for k in range(2):
  f,J,_,_=w.family(j,c,center_b);step=midpoint_matrix(J).inv()*f;new=[(c[i]-step[i,0]).mid()for i in range(26)];w.emit('center_newton_proposal',dict(index=k,before=vector_json(c),parameter=ball_json(center_b),value=matrix_json(f),jacobian=matrix_json(J),step=matrix_json(step),after=vector_json(new)));c=new
 f,J,fb,_=w.family(j,c,center_b);C=midpoint_matrix(midpoint_matrix(J).inv());v=[x.mid()for x in T.vec(-C*fb)];d=arb(0,h);p=[c[i]+v[i]*d for i in range(26)];_,Jp,hp,_=w.family(j,p,center_b+d);residual=-C*f-C*(Jp*T.col(v)+hp)*d;anchor=[known[i]-c[i]-v[i]*(left-center_b)for i in range(26)]
 r=[dyadic_radius(max((8*abs(residual[i,0])).upper(),(4*abs(anchor[i])).upper()))for i in range(26)]
 w.emit('radius_proposal',dict(center=vector_json(c),slope=vector_json(v),parameter=ball_json(center_b),halfwidth=ball_json(h),known_parameter=ball_json(left),preconditioned_residual=matrix_json(residual),known_projection=vector_json(anchor),radii=vector_json(r),scope='Deterministic finite dyadic proposals only: strictly above max(8*residual,4*known-root projection), floor2^-40, ceiling2^-8. Complete fresh predictor+E derivative follows; no expanded-domain inference.'))
 return c,v,r


def reducible(error):return str(error)in('strict full paired inverse bound','fixed dyadic proposal radius ceiling')


def march(w,j,known):
 left=arb(0);limit=arb('1/2048');h=arb((1,-13));minimum=arb((1,-17));attempts=[];accepted=[]
 _,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in T.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known];initial=known
 for attempt in range(1,13):
  if left==limit:break
  if len(accepted)>=8:break
  h=min(h,(limit-left)/2);need(h>=minimum,'fixed minimum step reached');bc=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(26)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record)
  try:
   c,v,r=prepare(w,j,candidate,known,left,bc,h);whole,v2,tangent,Jw,hb,ends,proof=T.tube(lambda z,b:w.family(j,z,b),c,bc,h,r,observe=w.emit,tick=w.tick);need(v==v2,'same exact predictor proposal');w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=T.join(known,left,c,v,bc,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==26,'all26 exact predecessor join')
    native=w.context(j,whole,bc+arb(0,h),tangent,support=[left,right]);w.save(tag+'/native.json.gz',native);curvature=T.fold_curvature(Jw,hb,tangent,12,bc+arb(0,h));w.save(tag+'/curvature.json',curvature);bb=decode(curvature['leak_bb'])
    if not native['passed']or not bb.is_finite()or bb.contains(0):record.update(status='complete_native_or_curvature_bound_unresolved')
    else:
     record.update(passed=True,status='qualified_connected_native6_fold_tube',contraction=proof['contraction'],leak_bb=curvature['leak_bb'],coordinates=26,primitive_period=6,complement_dimension=11);accepted.append(dict(attempt=attempt,center=c,slope=v,parameter=bc,halfwidth=h,radii=r,right=right,endpoint=ends[1]));left=right;known=ends[1];seed=[x.mid()for x in known];slope=v
  except ValueError as error:
   if not reducible(error):raise
   record.update(status='retained_interval_bound_unresolved',error=type(error).__name__+': '+str(error))
  w.save(tag+'/decision.json',record);attempts.append(record)
  if not record['passed']:
   h/=2
   if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(0)),target_parameter=ball_json(limit),scope='Only passed full26 tubes with strict predecessor identity and whole native6 primitive/simple+1/complement/nonzero curvature are promoted. Initial b=0 is the authenticated exact organizer; all b>0 within accepted intervals are nondegenerate folds.');w.save('continuation.json',result);need(result['passed'],'bounded connected fold range incomplete');return accepted


def endpoint(w,j,gain,arrays,known,charts):
 p=arrays['paired_physical'];m=(p[:12]+p[12:24])/2;difference=(p[:12]-p[12:24])/2;a=float(np.linalg.norm(difference));need(np.isfinite(a)and a>0,'positive finite numerical amplitude seed');q=difference/a
 # Choose exactly one sign against the authenticated organizer, swapping both
 # partners together if needed. This is a recorded seed orientation only.
 alignment=float(q@np.array([float(x.mid())for x in known[13:25]]));need(abs(alignment)>.5,'unambiguous same continued mode seed');orientation=1 if alignment>0 else -1;q=q*orientation
 seed=[B.exact_float(x)for x in m]+[B.exact_float(p[-1])]+[B.exact_float(x)for x in q]+[B.exact_float(a*a)];w.save('endpoint-seed.json',dict(seed=vector_json(seed),fixed_gain=ball_json(gain),partner_orientation=orientation))
 box,_,data,proof=P.root_enclosure(lambda u:T.fixed_gain(lambda z,b:w.family(j,z,b),u,gain),seed,newton_steps=3,radius_exponent=-80,observe=w.emit,tick=w.tick);w.save('endpoint-root.json',proof);need(proof['passed']and all(proof['coordinate_inclusions']),'full26 fixed-gain daughter fold root');z=data['z'];b=data['b'];need(b>0,'strict positive actual squared amplitude')
 joins=[dict(attempt=c['attempt'],join=T.join(z,b,c['center'],c['slope'],c['parameter'],c['halfwidth'],c['radii']))for c in charts];w.save('endpoint-joins.json',dict(rows=joins));need(any(x['join']['passed']and x['join']['coordinates']==26 for x in joins),'exact fold endpoint belongs to connected branch')
 native=w.context(j,z,b,data['paired_tangent']);w.save('endpoint-native.json.gz',native);curvature=T.fold_curvature(data['original_jacobian'],data['parameter_derivative'],data['paired_tangent'],12,b);w.save('endpoint-curvature.json',curvature);need(native['passed']and native['positive_amplitude']and curvature['passed'],'actual native primitive6 nondegenerate fold and11 stable complements')
 associations=[]
 for partner in(0,1):
  original=partner if orientation==1 else 1-partner;cycle=arrays['cycle_x'if original==0 else'cycle_y'];ctx0=next(x for x in native['contexts']if x['partner']==partner and x['shift']==0);orbit=[[decode(x)for x in row]for row in ctx0['orbit']];scale=max(1.,float(np.max(np.abs(cycle))));errors=[max(abs(orbit[k][i]-B.exact_float(cycle[k,i])).upper()for i in range(12))/B.exact_float(scale)for k in range(6)]
  oldq=arrays['physical'][13:]if original==0 else arrays['partner_mode'];actual=[decode(x)for x in ctx0['mode']];scores=[sum((abs(actual[i]-sign*B.exact_float(oldq[i])).upper()*abs(actual[i]-sign*B.exact_float(oldq[i])).upper()for i in range(12)),arb(0)).sqrt().upper()for sign in(1,-1)];matching=[s for s,e in zip((1,-1),scores)if e<arb('1e-6')];passed=all(x<arb('1e-6')for x in errors)and len(matching)==1 and abs(z[12]-B.exact_float(p[-1])).upper()<arb('1e-8');associations.append(dict(partner=partner,original_partner=original,passed=bool(passed),phase_errors=vector_json(errors),mode_errors=vector_json(scores),mode_orientation=matching,leak_error=ball_json(abs(z[12]-B.exact_float(p[-1])).upper())))
 w.save('numerical-fold-association.json',dict(passed=all(x['passed']for x in associations),rows=associations,scope='Literal all6 native phases at both partners and unique normalized native-mode sign, relative to the already qualified numerical fold. This numerical association is separate from the exact root/tube uniqueness join.'));need(all(x['passed']for x in associations),'unchanged actual numerical fold native phase/mode association');return proof,native,curvature,joins


def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,180);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS,settings=SETTINGS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<508*1024**2,'source storage cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   card=read(ROOT/CARD);need(card['settings']==SETTINGS and card['caps']==CAPS,'literal bounded design');synthetic(w);j,known,gain,arrays=bind(w);charts=march(w,j,known);root,native,curvature,joins=endpoint(w,j,gain,arrays,known,charts)
   result.update(passed=True,status='exact_organizer_to_native6_fold_branch',accepted_charts=len(charts),squared_amplitude_range=['0','1/2048'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Connected finite nondegenerate native6 fold branch born at the exact signed-G generalized flip organizer, with a certified fixed-gain root and literal same-phase numerical fold association. Native12/24 descendants and complete shrimp/window geometry remain separate; independent outcome reduction required.')
 except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
