#!/usr/bin/env python3
"""Measured finite correlated51 fold continuation from the same exact organizer."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
import certify_clark_explicit_fold as OLD
import profile_clark_correlated_fold as PROFILE
from neural_shrimp import rigorous_clark_correlated_path as P
from neural_shrimp import rigorous_clark_correlated_tube as CT
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json,canonical_bytes,midpoint_matrix
B,T,A=OLD.B,OLD.T,OLD.A
need,read,entry,decode=OLD.need,OLD.read,OLD.entry,OLD.decode
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-bridge/attempt001/run')
CARD='docs/plans/clark-correlated-fold-inputs.json';THREADS=OLD.THREADS
SETTINGS=dict(OLD.SETTINGS,attempts=8,accepted_charts=6)
CAPS=dict(explicit=32,path=40,implicit=1,anchor=1,context=8,algebra=24,native_phase_returns=270,cache_reuses=16,events=1600)
FILES=sorted(set(OLD.FILES+PROFILE.FILES+[CARD,'docs/plans/clark-correlated-fold-target.md','docs/reviews/clark-correlated-fold-profile.json','docs/reviews/clark-explicit-fold-attempt002.json','docs/reviews/clark-explicit-radius-diagnosis.json','tools/certify_clark_correlated_fold.py','tools/launch_clark_correlated_fold.py','tests/test_clark_correlated_fold_driver.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

class Work(OLD.Work):
 def __init__(self):super().__init__();self.calls=dict.fromkeys(CAPS,0);self.last_path=None
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed inclusive '+key+' cap');self.calls[key]+=1
 def emit(self,kind,row):
  if kind=='clark_gradient_path_return':self.charge('native_phase_returns')
  super().emit(kind,row)
  if kind=='clark_correlated_path_return':self.last_path=f'events/{self.calls["events"]:04d}-{kind}.json.gz'
 def path(self,j,c,s,a,error,b,d,**ignored):
  self.charge('path');self.emit('path_attempt',dict(matrix=matrix_json(arb_mat(j)),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),error=vector_json(error),parameter=ball_json(b),d=ball_json(d)));answer=P.coefficients(j,c,s,a,error,b,d,tick=self.tick,observe=self.emit);need(self.last_path is not None,'completed saved path acquisition');return answer


def key(j,c,s,a,error,b,d):return canonical_bytes(dict(matrix=matrix_json(arb_mat(j)),center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),error=vector_json(error),parameter=ball_json(b),d=ball_json(d)))

class ResidualCache:
 """Exactly two retained zero-error acquisitions, no model or domain widening."""
 def __init__(self,w):self.w=w;self.entries={};self.used=set()
 def add(self,j,c,s,a,error,b,d,answer):
  need(all(x==0 for x in error),'only zero-error residual may be reused');k=key(j,c,s,a,error,b,d);need(k not in self.entries and len(self.entries)<2,'two unique center/whole residual inputs');path=self.w.last_path;p=OUT/path;row=json.loads(gzip.decompress(p.read_bytes()));f,J=answer
  need(row['center']==vector_json(c)and row['slope']==vector_json(s)and row['curvature']==vector_json(a)and row['error']==vector_json(error)and row['parameter']==ball_json(b)and row['d']==ball_json(d)and row['value_coefficients']==matrix_json(f)and row['jacobian_coefficients']==[matrix_json(v)for v in J],'literal acquired arrays/input association')
  self.entries[k]=(answer,path,entry(p),hashlib.sha256(canonical_bytes(dict(F=matrix_json(f),J=[matrix_json(v)for v in J]))).hexdigest())
 def acquire(self,j,c,s,a,error,b,d,**kwargs):
  k=key(j,c,s,a,error,b,d)
  if all(x==0 for x in error):
   need(k in self.entries and k not in self.used,'literal unique cached residual input');answer,path,pin,digest=self.entries[k];f,J=answer
   need(entry(OUT/path)==pin and hashlib.sha256(canonical_bytes(dict(F=matrix_json(f),J=[matrix_json(v)for v in J]))).hexdigest()==digest,'immutable same cached full arrays');self.w.charge('cache_reuses');self.w.emit('cached_residual_reuse',dict(file=path,identity=pin,input_sha256=hashlib.sha256(k).hexdigest(),array_sha256=digest));self.used.add(k);return answer
  return self.w.path(j,c,s,a,error,b,d,**kwargs)


def forecast(profile):
 need(profile['passed']and profile['target_arrays_unopened']and profile['settings']==PROFILE.SETTINGS and profile['caps']==PROFILE.CAPS and profile['calls']==dict(path=6,mixed=2,context=1,algebra=2,native_phases=30,events=72),'complete literal unrelated profile')
 need(set(profile['times'])==set(profile['bytes'])==set(profile['max_seconds'])==set(profile['max_bytes'])=={'path','mixed','context','algebra'},'all measured categories')
 for k,count in [('path',6),('mixed',2),('context',1),('algebra',2)]:need(len(profile['times'][k])==len(profile['bytes'][k])==count and all(x>0 for x in profile['times'][k]+profile['bytes'][k])and max(profile['times'][k])==profile['max_seconds'][k]and max(profile['bytes'][k])==profile['max_bytes'][k],'actual measured extrema '+k)
 # Dense target algebra and late native matrix powers receive additional
 # fixed fourfold/eightfold floors before the overall factor two.
 counts=dict(path=40,mixed=34,context=8*8,algebra=24*4);seconds=2*sum(counts[k]*profile['max_seconds'][k]for k in counts)+20;raw=2*sum(counts[k]*profile['max_bytes'][k]for k in counts)+16*1024**2
 return dict(passed=bool(seconds<180 and raw<508*1024**2),seconds=seconds,bytes=raw,charged_categories=counts)


def bind_profile(w,card):
 need(card['schema']=='clark-correlated-explicit51-v1'and card['settings']==SETTINGS and card['caps']==CAPS and card['old_card']==entry(ROOT/OLD.CARD),'literal finite correlated design')
 spec=card['profile'];p=Path(spec['root']);rows=[B.B.inventory(p,spec['outer'],w.tick),B.B.inventory(p/'run',spec['worker'],w.tick)];rt=OLD.historical(p,spec['source'],0);out=read(p/'run/outcome.json');profile=read(p/'run/profile.json');need(out['passed']and out['target_arrays_unopened']and out['calls']==profile['calls']and rt['precision']==384 and rt['python_flint']==flint.__version__ and rt['flint']==flint.__FLINT_VERSION__ and rt['python']==platform.python_version(),'same actual profile runtime')
 need(rt['sources']==spec['sources'],'literal complete isolated profile source inventory')
 for n in rt['sources']:need(entry(ROOT/n)==rt['sources'][n],'byte-identical complete profiled acquisition/reduction '+n)
 f=forecast(profile);f['peak_bytes']=2*out['peak_rss_kib']*1024+256*1024**2;need(f['passed']and f['peak_bytes']<2*1024**3,'same-source measured target admission');w.save('profile-binding.json',dict(profile=spec,inventories=rows,forecast=f,target_arrays_unopened=True));return f


def bind(w,card):
 for k in('failed_explicit','diagnosis'):
  spec=card[k];p=Path(spec['root']);rows=[B.B.inventory(p,spec['outer'],w.tick),B.B.inventory(p/'run',spec['worker'],w.tick)];OLD.historical(p,spec['source'],2 if k=='failed_explicit'else 0);out=read(p/'run/outcome.json')
  if k=='failed_explicit':need(not out['passed']and out['calls']['native_phase_returns']==87 and len(read(p/'run/continuation.json')['attempts'])==5,'retained allfive first-chart radius failures')
  else:need(out['passed']and out['native_calls']==0,'retained predictor-correlation saved diagnosis')
  w.save(k+'-binding.json',dict(spec=spec,inventories=rows,outcome=out))
 return OLD.bind(w)


def prepare(w,j,seed,known,left,b,h):
 c=[x.mid()for x in seed]
 for k in range(2):
  f,J,_,_=w.family(j,c,b);step=midpoint_matrix(J).inv()*f;new=[(c[i]-step[i,0]).mid()for i in range(51)];w.emit('correlated_center_newton_proposal',dict(index=k,before=vector_json(c),parameter=ball_json(b),value=matrix_json(f),jacobian=matrix_json(J),step=matrix_json(step),after=vector_json(new)));c=new
 f,J,fb,_=w.family(j,c,b);C=midpoint_matrix(midpoint_matrix(J).inv());s=[x.mid()for x in CT.vec(-C*fb)];zero=[arb(0)]*51;linear,_=w.path(j,c,s,zero,zero,b,arb(0));w.charge('algebra');a=[x.mid()for x in CT.vec(-2*C*CT.col([linear[i,2]for i in range(51)]))];w.emit('quadratic_predictor_proposal',dict(center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),parameter=ball_json(b)))
 cache=ResidualCache(w);fc,jc=w.path(j,c,s,a,zero,b,arb(0));cache.add(j,c,s,a,zero,b,arb(0),(fc,jc));fw,jw=w.path(j,c,s,a,zero,b,arb(0,h));cache.add(j,c,s,a,zero,b,arb(0,h),(fw,jw))
 C0=midpoint_matrix(midpoint_matrix(jc[0]).inv());C1=midpoint_matrix(-C0*midpoint_matrix(jc[1])*C0);R=[CT.col([fc[i,k]if k<3 else fw[i,3]for i in range(51)])for k in range(4)];w.charge('algebra')
 forcing=[arb_mat(51,1)for _ in range(5)]
 for k in range(2):
  for l in range(4):forcing[k+l]-=[C0,C1][k]*R[l]
 residual=sum((forcing[k]*CT.powers(h)[k]for k in range(5)),arb_mat(51,1));d=left-b;anchor=[known[i]-c[i]-s[i]*d-a[i]*d*d/2 for i in range(51)];r=[OLD.radius(max((8*abs(residual[i,0])).upper(),(4*abs(anchor[i])).upper()),i)for i in range(51)]
 w.emit('correlated_radius_proposal',dict(center=vector_json(c),slope=vector_json(s),curvature=vector_json(a),parameter=ball_json(b),halfwidth=ball_json(h),known_parameter=ball_json(left),forcing_coefficients=[matrix_json(x)for x in forcing],preconditioned_residual=matrix_json(residual),known_projection=vector_json(anchor),radii=vector_json(r),scope='Finite proposals only; all fullE J0/J1/J2 are freshly acquired after these radii are fixed. Unchanged state2^-8/tangent4 ceilings.'))
 return c,s,a,r,cache


def march(w,j,known):
 left=arb(0);limit=arb(1)/2048;h=arb((1,-13));minimum=arb((1,-17));attempts=[];accepted=[];_,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in CT.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known]
 for attempt in range(1,9):
  if left==limit or len(accepted)>=6:break
  h=min(h,(limit-left)/2);need(h>=minimum,'fixed minimum step reached');b=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(51)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record)
  try:
   c,s,a,r,cache=prepare(w,j,candidate,known,left,b,h);w.charge('algebra');whole,ends,data,proof=CT.tube(j,c,s,a,b,h,r,acquire=cache.acquire,observe=w.emit,tick=w.tick);need(len(cache.used)==2,'both exact saved residuals reused once');w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=CT.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 curved exact predecessor root');native=w.context(j,whole[:26],b+arb(0,h),CT.col(whole[26:]),support=[left,right]);w.save(tag+'/native.json.gz',native);oriented=all(decode(x)>0 for x in native['partner_orientation'])
    if not native['passed']or not oriented or(left>0 and not native['positive_amplitude']):record.update(status='complete_native_or_orientation_bound_unresolved')
    else:
     record.update(passed=True,status='qualified_connected_native6_correlated_critical_tube',contraction=proof['contraction'],coordinates=51,primitive_period=6,complement_dimension=11,includes_degenerate_zero=bool(left==0),generic_fold_uniformly_proved=False);accepted.append(dict(attempt=attempt,center=c,slope=s,curvature=a,parameter=b,halfwidth=h,radii=r,right=right,endpoint=ends[1]));left=right;known=ends[1];seed=[x.mid()for x in known];slope=[(s[i]+a[i]*h).mid()for i in range(51)]
  except ValueError as error:
   if str(error)!='fixed coordinate radius ceiling':raise
   record.update(status='retained_radius_proposal_bound',error=type(error).__name__+': '+str(error))
  w.save(tag+'/decision.json',record);attempts.append(record)
  if not record['passed']:
   h/=2
   if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(0)),target_parameter=ball_json(limit),scope='Only strict full51 curved roots with exact predecessor and literal native6 partner/phase/mode/primitive/complement gates advance. b0 is degenerate; uniform H_U/genericity remain separate.');w.save('continuation.json',result);need(result['passed'],'bounded correlated critical range incomplete');return accepted

def endpoint(w,j,gain,arrays,known,charts):
 p=arrays['paired_physical'];m=(p[:12]+p[12:24])/2;difference=(p[:12]-p[12:24])/2;a=float(np.linalg.norm(difference));need(np.isfinite(a)and a>0,'positive finite numerical amplitude seed');q=difference/a
 # Choose exactly one sign against the authenticated organizer, swapping both
 # partners together if needed. This is a recorded seed orientation only.
 alignment=float(q@np.array([float(x.mid())for x in known[13:25]]));need(abs(alignment)>.5,'unambiguous same continued mode seed');orientation=1 if alignment>0 else -1;q=q*orientation
 seed=[B.B.exact_float(x)for x in m]+[B.B.exact_float(p[-1])]+[B.B.exact_float(x)for x in q]+[arb(0)]*25+[B.B.exact_float(a*a)];w.save('endpoint-seed.json',dict(seed=vector_json(seed),fixed_gain=ball_json(gain),partner_orientation=orientation))
 box,_,data,proof=T.root_enclosure(lambda u:T.fixed_gain(lambda z,b:w.family(j,z,b),u,gain,12),seed,newton_steps=3,radius_exponent=-80,observe=w.emit,tick=w.tick);w.save('endpoint-root.json',proof);need(proof['passed']and all(proof['coordinate_inclusions']),'full51 fixed-gain daughter critical root');z=data['z'];b=data['b'];need(b>0,'strict positive actual squared amplitude')
 joins=[dict(attempt=c['attempt'],join=CT.join(data['unknown'],b,c['center'],c['slope'],c['curvature'],c['parameter'],c['halfwidth'],c['radii']))for c in charts];w.save('endpoint-joins.json',dict(rows=joins));need(any(x['join']['passed']and x['join']['coordinates']==51 for x in joins),'exact fold endpoint belongs to connected branch')
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
   card=read(ROOT/CARD);bind_profile(w,card);j,known,gain,arrays=bind(w,card);charts=march(w,j,known);root,native,curvature,joins=endpoint(w,j,gain,arrays,known,charts)
   result.update(passed=True,status='exact_organizer_connected_native6_correlated_critical_curve_with_generic_endpoint',accepted_charts=len(charts),squared_amplitude_range=['0','1/2048'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Connected exact native6 critical curve from signed-G cubic-zero organizer; only selected positive endpoint has separate H_U inverse/physical curvature. Literal all-phase/mode ancestry retained. No uniform genericity, native12/24 or full-window claim; independent outcome reductions required.')
 except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
