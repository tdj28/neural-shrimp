#!/usr/bin/env python3
"""Fixed bounded explicit51 native critical curve with a generic fold endpoint."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
import certify_clark_amplitude_fold as B
from neural_shrimp import rigorous_clark_explicit_fold as E
from neural_shrimp import rigorous_clark_explicit_tube as T
from neural_shrimp import rigorous_clark_amplitude_fold as A
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,midpoint_matrix
need,entry,read,decode=B.need,B.entry,B.read,B.decode
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/explicit-fold-bridge/attempt001/run')
CARD='docs/plans/clark-explicit-fold-inputs.json';THREADS=B.THREADS
SETTINGS=dict(B.SETTINGS,dimension=51,v_radius_ceiling_exponent=2)
CAPS=dict(explicit=93,anchor=2,implicit=2,context=14,native_phase_returns=381,events=2048)
FILES=sorted(set(B.FILES+[CARD,'docs/plans/clark-explicit-amplitude-fold.md','docs/reviews/clark-amplitude-fold-attempt001.json','docs/reviews/clark-amplitude-inverse-diagnosis.json','tools/certify_clark_explicit_fold.py','tools/launch_clark_explicit_fold.py','tests/test_clark_explicit_fold_driver.py','tests/test_rigorous_clark_explicit_fold.py','tests/test_rigorous_clark_explicit_tube.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

class Work(B.Work):
 def __init__(self):super().__init__();self.calls=dict.fromkeys(CAPS,0)
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed inclusive '+key+' cap');self.calls[key]+=1
 def family(self,j,u,b):
  self.charge('explicit');self.emit('explicit_family_attempt',dict(input=vector_json(u),squared_amplitude=ball_json(b)));return E.augmented(j,u,b,self.emit)
 def implicit(self,j,z,b):
  self.charge('implicit');self.emit('implicit_endpoint_attempt',dict(input=vector_json(z),squared_amplitude=ball_json(b)));return A.augmented(j,z,b,self.emit)


def synthetic(w):
 n=12;q=[arb(int(i==0))for i in range(n)];j=[[arb(int(i==k))/4+arb(1)/256 for k in range(n)]for i in range(n)];z=[arb(i+1)/50 for i in range(n)]+[arb(3)/5]+q+[arb(7)/5];u=z+[arb(0)]*25;times={};sizes={}
 def measure(k,f):
  start=time.monotonic();used=w.used;ans=f();times.setdefault(k,[]).append(time.monotonic()-start);sizes.setdefault(k,[]).append(w.used-used);return ans
 for i in range(3):
  box=u if i==0 else[arb(v,arb((1,-12 if i==1 else -10)))for v in u];b=arb(0)if i==0 else arb('1/4096',arb((1,-13)))
  measure('explicit',lambda:w.family(j,box,b))
 measure('implicit',lambda:w.implicit(j,z,arb(0)))
 w.charge('anchor')
 def anchor_family(z,b):return [z[i]+b*(arb(0 if i==12 else i+1)/32) for i in range(25)]
 measure('anchor',lambda:E.conditional_organizer_anchor(anchor_family,[arb(0)]*25+[arb(1)],12,arb(1),w.emit))
 jj=[[arb(int(i==k==0))for k in range(n)]for i in range(n)];zz=[arb(0)]*n+[arb('1/2')]+q+[arb(1)];negative=measure('context',lambda:w.context(jj,zz,arb(0),arb_mat(25,1)))
 need(not negative['passed']and all(c['witness_power']==1 for c in negative['contexts']),'full12 fixedpoint negative context')
 def quadratic(z,b):return T.col([x-b*b for x in z]),T.eye(51),T.col([-2*b for x in z]),None
 trial=measure('algebra',lambda:T.tube(quadratic,[arb(1)]*51,arb(1),arb(1)/16,[arb(1)/64]*51,observe=w.emit,tick=w.tick));need(trial[-1]['passed'],'full51 predictor/whole tube synthetic')
 maxt={k:max(v)for k,v in times.items()};maxb={k:max(v)for k,v in sizes.items()};remaining=dict(explicit=90,implicit=1,anchor=1,context=13,algebra=48)
 forecast=time.monotonic()-w.start+2*sum(remaining[k]*maxt[k]for k in remaining)+20;raw=w.used+2*sum(remaining[k]*maxb[k]for k in remaining)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
 report=dict(passed=bool(forecast<180 and raw<512*1024**2 and peak<2*1024**3),times=times,bytes=sizes,remaining=remaining,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True,calls=dict(w.calls));w.save('profile.json',report);need(report['passed'],'full51 explicit/implicit/context measured admission')


def bind(w):
 card=read(ROOT/CARD);need(card['settings']==SETTINGS and card['caps']==CAPS and card['base_card']==entry(ROOT/B.CARD),'exact amended full51 card')
 failures=[]
 for key in('failed_implicit','diagnosis'):
  spec=card[key];root=Path(spec['root']);failures.extend([B.B.inventory(root,spec['outer'],w.tick),B.B.inventory(root/'run',spec['worker'],w.tick)]);B.B.historical(root,spec['source']);out=read(root/'run/outcome.json')
  if key=='failed_implicit':need(not out['passed']and out['calls']['native_phase_returns']==135,'preserved original zero-tube implicit failure')
  else:need(out['passed']and out['native_calls']==0,'preserved exact saved inverse diagnosis')
 j,organizer,gain,arrays=B.bind(w);base=read(ROOT/B.B.CARD);ip=Path(base['independent']['root'])/'run';unfold=read(ip/'unfolding.json');lambda_leak=decode(unfold['multiplier_leak_derivative']);w.charge('anchor')
 known,anchor=E.conditional_organizer_anchor(lambda z,b:A.paired_family(j,z,b,w.emit),organizer,12,lambda_leak,w.emit)
 need(len(known)==51,'complete exact organizer plus unique paired tangent');w.save('explicit-anchor.json',dict(known=vector_json(known),proof=anchor,failures=failures,organizer_authority=base['independent'],scope='Same authenticated exact c3zero normalized organizer and nonzero lambda_leak imply v_leak=0; tiny paired inverse encloses all v. No zero-containment existence inference.'));return j,known,gain,arrays


def radius(value,i):
 need(value.is_finite()and value>=0,'finite radius proposal');r=arb((1,-40));ceiling=arb((1,-8 if i<26 else 2))
 while r<=value:r*=2
 need(r<=ceiling,'fixed coordinate radius ceiling');return r


def prepare(w,j,seed,known,left,center_b,h):
 c=[x.mid()for x in seed]
 for k in range(2):
  f,J,_,_=w.family(j,c,center_b);step=midpoint_matrix(J).inv()*f;new=[(c[i]-step[i,0]).mid()for i in range(51)];w.emit('center_newton_proposal',dict(index=k,before=vector_json(c),parameter=ball_json(center_b),value=matrix_json(f),jacobian=matrix_json(J),step=matrix_json(step),after=vector_json(new)));c=new
 f,J,fb,_=w.family(j,c,center_b);C=midpoint_matrix(midpoint_matrix(J).inv());v=[x.mid()for x in T.vec(-C*fb)];d=arb(0,h);p=[c[i]+v[i]*d for i in range(51)];_,Jp,hp,_=w.family(j,p,center_b+d);residual=-C*f-C*(Jp*T.col(v)+hp)*d;anchor=[known[i]-c[i]-v[i]*(left-center_b)for i in range(51)]
 radii=[radius(max((8*abs(residual[i,0])).upper(),(4*abs(anchor[i])).upper()),i)for i in range(51)];w.emit('explicit_radius_proposal',dict(center=vector_json(c),slope=vector_json(v),parameter=ball_json(center_b),halfwidth=ball_json(h),known_parameter=ball_json(left),preconditioned_residual=matrix_json(residual),known_projection=vector_json(anchor),radii=vector_json(radii),scope='Finite dyadic proposals only; floor2^-40, z-coordinate ceiling2^-8 and tangent-coordinate ceiling4. Complete fresh predictor+E acquisition follows.'))
 return c,v,radii


def march(w,j,known):
 left=arb(0);limit=arb(1)/2048;h=arb((1,-13));minimum=arb((1,-17));attempts=[];accepted=[]
 _,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in T.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known]
 for attempt in range(1,13):
  if left==limit or len(accepted)>=8:break
  h=min(h,(limit-left)/2);need(h>=minimum,'fixed minimum step reached');bc=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(51)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record)
  try:
   c,v,r=prepare(w,j,candidate,known,left,bc,h);whole,v2,data,Jw,hb,ends,proof=T.tube(lambda u,b:w.family(j,u,b),c,bc,h,r,observe=w.emit,tick=w.tick);need(v==v2,'same exact predictor');w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=T.join(known,left,c,v,bc,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 same actual predecessor')
    native=w.context(j,whole[:26],bc+arb(0,h),data['paired_tangent'],support=[left,right]);w.save(tag+'/native.json.gz',native);oriented=all(decode(x)>0 for x in native['partner_orientation'])
    if not native['passed']or not oriented or (left>0 and not native['positive_amplitude']):record.update(status='complete_native_or_orientation_bound_unresolved')
    else:
     record.update(passed=True,status='qualified_connected_native6_critical_tube',contraction=proof['contraction'],coordinates=51,primitive_period=6,complement_dimension=11,includes_degenerate_zero=bool(left==0),generic_fold_uniformly_proved=False);accepted.append(dict(attempt=attempt,center=c,slope=v,parameter=bc,halfwidth=h,radii=r,right=right,endpoint=ends[1]));left=right;known=ends[1];seed=[x.mid()for x in known];slope=v
  except ValueError as error:
   if str(error)!='fixed coordinate radius ceiling':raise
   record.update(status='retained_radius_proposal_bound',error=type(error).__name__+': '+str(error))
  w.save(tag+'/decision.json',record);attempts.append(record)
  if not record['passed']:
   h/=2
   if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(0)),target_parameter=ball_json(limit),scope='Only full51 tubes with strict predecessor and literal native6 primitive/simple+1/complement/orientation are promoted. b0 remains degenerate; H_U invertibility and genericfold are not inferred uniformly.');w.save('continuation.json',result);need(result['passed'],'bounded connected critical range incomplete');return accepted

def endpoint(w,j,gain,arrays,known,charts):
 p=arrays['paired_physical'];m=(p[:12]+p[12:24])/2;difference=(p[:12]-p[12:24])/2;a=float(np.linalg.norm(difference));need(np.isfinite(a)and a>0,'positive finite numerical amplitude seed');q=difference/a
 # Choose exactly one sign against the authenticated organizer, swapping both
 # partners together if needed. This is a recorded seed orientation only.
 alignment=float(q@np.array([float(x.mid())for x in known[13:25]]));need(abs(alignment)>.5,'unambiguous same continued mode seed');orientation=1 if alignment>0 else -1;q=q*orientation
 seed=[B.B.exact_float(x)for x in m]+[B.B.exact_float(p[-1])]+[B.B.exact_float(x)for x in q]+[arb(0)]*25+[B.B.exact_float(a*a)];w.save('endpoint-seed.json',dict(seed=vector_json(seed),fixed_gain=ball_json(gain),partner_orientation=orientation))
 box,_,data,proof=T.root_enclosure(lambda u:T.fixed_gain(lambda z,b:w.family(j,z,b),u,gain,12),seed,newton_steps=3,radius_exponent=-80,observe=w.emit,tick=w.tick);w.save('endpoint-root.json',proof);need(proof['passed']and all(proof['coordinate_inclusions']),'full51 fixed-gain daughter critical root');z=data['z'];b=data['b'];need(b>0,'strict positive actual squared amplitude')
 joins=[dict(attempt=c['attempt'],join=T.join(data['unknown'],b,c['center'],c['slope'],c['parameter'],c['halfwidth'],c['radii']))for c in charts];w.save('endpoint-joins.json',dict(rows=joins));need(any(x['join']['passed']and x['join']['coordinates']==51 for x in joins),'exact fold endpoint belongs to connected branch')
 native=w.context(j,z,b,data['paired_tangent']);w.save('endpoint-native.json.gz',native);w.charge('implicit');curvature=T.conditional_endpoint_fold(j,data['unknown'],b,w.emit);w.save('endpoint-curvature.json',curvature);need(native['passed']and native['positive_amplitude']and curvature['passed'],'actual native primitive6 nondegenerate fold and11 stable complements')
 associations=[]
 for partner in(0,1):
  original=partner if orientation==1 else 1-partner;cycle=arrays['cycle_x'if original==0 else'cycle_y'];ctx0=next(x for x in native['contexts']if x['partner']==partner and x['shift']==0);orbit=[[decode(x)for x in row]for row in ctx0['orbit']];scale=max(1.,float(np.max(np.abs(cycle))));errors=[max(abs(orbit[k][i]-B.B.exact_float(cycle[k,i])).upper()for i in range(12))/B.B.exact_float(scale)for k in range(6)]
  oldq=arrays['physical'][13:]if original==0 else arrays['partner_mode'];actual=[decode(x)for x in ctx0['mode']];scores=[sum((abs(actual[i]-sign*B.B.exact_float(oldq[i])).upper()*abs(actual[i]-sign*B.B.exact_float(oldq[i])).upper()for i in range(12)),arb(0)).sqrt().upper()for sign in(1,-1)];matching=[s for s,e in zip((1,-1),scores)if e<arb('1e-6')];passed=all(x<arb('1e-6')for x in errors)and len(matching)==1 and abs(z[12]-B.B.exact_float(p[-1])).upper()<arb('1e-8');associations.append(dict(partner=partner,original_partner=original,passed=bool(passed),phase_errors=vector_json(errors),mode_errors=vector_json(scores),mode_orientation=matching,leak_error=ball_json(abs(z[12]-B.B.exact_float(p[-1])).upper())))
 w.save('numerical-fold-association.json',dict(passed=all(x['passed']for x in associations),rows=associations,scope='Literal all6 native phases at both partners and unique normalized native-mode sign, relative to the already qualified numerical fold. This numerical association is separate from the exact root/tube uniqueness join.'));need(all(x['passed']for x in associations),'unchanged actual numerical fold native phase/mode association');return proof,native,curvature,joins


def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);B.OUT=OUT;w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,180);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS,settings=SETTINGS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<508*1024**2,'source storage cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   card=read(ROOT/CARD);need(card['settings']==SETTINGS and card['caps']==CAPS,'literal bounded design');synthetic(w);j,known,gain,arrays=bind(w);charts=march(w,j,known);root,native,curvature,joins=endpoint(w,j,gain,arrays,known,charts)
   result.update(passed=True,status='exact_organizer_connected_native6_critical_curve_with_generic_endpoint',accepted_charts=len(charts),squared_amplitude_range=['0','1/2048'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Connected finite native6 critical curve born at the exact signed-G generalized flip organizer, with single+1 and11stable complementary modes. Only the selected positive fixed-gain endpoint has separate paired inverse/nonzero physical curvature and same-phase numerical fold association. b0 remains degenerate; uniform genericfold, native12/24 and complete window geometry remain separate. Independent outcome reduction required.')
 except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
