"""Bounded adaptive signed470 continuation in the literal output-rate gauge."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,json,platform,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native470_correlated as C
from neural_shrimp.signed470_march_policy import next_state
import probe_native470_output_weighted as W
from neural_shrimp.rigorous_orbits import canonical_bytes,exact
from neural_shrimp.rigorous_eigenpair_locus import exact_mid
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
need,read,entry=C.need,C.read,C.entry
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/signed-flip-output-march001/attempt001/run');CARD='docs/plans/native470-output-march-inputs.json'
POLICY=dict(max_attempts=24,initial_width=-18,minimum_width=-24,maximum_width=-12,strong_contraction='1/5',strong_passes=2,radius_exponent=-12,newton_steps=2,precision=256,period=18,sigma=-1,output_stop='1689/2000')
LIMITS=dict(seconds=6000,unit_seconds=240,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=6*1024**3,unit_raw_bytes=768*1024**2,native_calls=3636,ticks=3786)
FILES=sorted(set(W.FILES+[CARD,'docs/plans/native470-output-march.md','src/neural_shrimp/signed470_march_policy.py','tools/march_native470_output.py','tools/launch_native470_output_march.py','tests/test_native470_output_march.py','docs/reviews/native470-output-weighted002.json']))
class Work:
 def __init__(self):self.start=time.monotonic();self.unit_start=self.start;self.used=0;self.unit_used=0;self.ticks=0;self.native=0;self.serial=0;self.stage='profile';self.prefix=''
 def check(self):need(time.monotonic()-self.start<6000 and time.monotonic()-self.unit_start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'6000s global/240s unit/4GiB')
 def tick(self):self.ticks+=1;need(self.ticks<=3786,'186+24*150 ticks');self.check()
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);data=gzip.compress(raw,mtime=0)if n.endswith('.gz')else raw
  need(self.used+len(data)<6*1024**3-(0 if closing else 16*1024**2),'6GiB/16MiB closing reserve');need(self.unit_used+len(data)<768*1024**2-(0 if closing else 4*1024**2),'768MiB perunit/4MiB reserve');p=OUT/self.prefix/n;p.parent.mkdir(parents=True,exist_ok=True)
  temporary=p.with_name(p.name+'.part')if n=='unit-start.json'else p
  with temporary.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  if temporary!=p:os.link(temporary,p);temporary.unlink()
  self.used+=len(data);self.unit_used+=len(data)
  if not closing:self.check()
 def observe(self,k,v):
  if k=='native_input':self.native+=1;need(self.native<=3636,'180+24*144 native calls')
  if k=='signed_correlated_chart_return':self.save('target-certificate.json.gz'if self.stage=='target'else'profile-certificate.json.gz',v)
  else:self.save(f'events/{self.serial:04d}-{self.stage}-{k}.json.gz',v);self.serial+=1
 def unit(self,index):self.prefix=f'attempt{index:03d}';self.unit_start=time.monotonic();self.unit_used=0;self.serial=0;self.stage='target';self.save('unit-start.json',dict(index=index,started_monotonic=self.unit_start,started_unix=time.time(),maximum_seconds=240))

class Prefix:
 def __init__(self,w,prefix):self.w=w;self.prefix=prefix
 def check(self):self.w.check()
 @property
 def start(self):return self.w.start
 def save(self,n,v,closing=False):return self.w.save(self.prefix+n,v,closing)

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['policy']==POLICY and card['limits']==LIMITS and card['independent']is not None,'fixed reviewed output continuation with independent authority')
 p,rt=C.history(w,card['producer']);a,ar=C.history(w,card['independent']);o=read(p/'run/outcome.json');ao=read(a/'run/outcome.json');ib=read(a/'run/input-binding.json');oldcard=read(ROOT/W.CARD)
 need(o==card['producer']['outcome']and o['passed']and o['native_calls']==324 and o['ticks']==336 and o['strict_inclusions']==o['anchor_inclusions']==470,'complete original scaled output proof')
 need(ao==card['independent']['outcome']and ao['passed']and ao['native_calls']==0 and ao['root_coordinates']==ao['anchor_coordinates']==470 and len(ao['refinement_rows'])==3 and all(x['coordinates']==470 for x in ao['refinement_rows']),'independent refinement/root/full470 projected connection')
 need(ib['card']['producer']==card['producer']and ib['card']['known']==oldcard['independent'],'literal complete exact same-branch genealogy')
 W.bind(Prefix(w,'preflight-'if metadata_only else'ancestry-'),oldcard,True)
 for n,e in ib['native_source'].items():need(rt['sources'][n]==entry(ROOT/n)==e,'unchanged full native map '+n)
 for n in ['src/neural_shrimp/rigorous_signed_proposal.py','src/neural_shrimp/rigorous_signed_correlated.py','src/neural_shrimp/rigorous_signed_gradient_path.py','src/neural_shrimp/rigorous_polynomial_inverse.py']:
  need(rt['sources'][n]==entry(ROOT/n),'unchanged full-domain proof kernel '+n)
 for n,k in [('connected-right-endpoint.json','endpoint'),('independent512.json','proof')]:need(entry(a/'run'/n)==card['independent'][k],'literal independently recomputed root '+n)
 for n,e in card['producer']['files'].items():need(entry(p/'run'/n)==e,'literal complete acquired source/input '+n)
 if metadata_only:return p,a
 known=read(a/'run/connected-right-endpoint.json');proof=read(a/'run/independent512.json');chart=read(p/'run/target-certificate.json.gz');inp=read(p/'run/target-inputs.json')
 need(known['box']==proof['recomputed_endpoints'][1]and len(known['box'])==470 and proof['passed']and all(proof['gates'].values()),'entire independently recomputed right endpoint')
 need(known['exact_parameter']==chart['parameter_endpoints'][1]and known['normal']==chart['normal']==inp['normal']==['0']*469+['1']and known['origin']==chart['origin']==inp['origin']and known['period']==18 and known['sigma']==-1 and known['dimension']==13,'literal phase/sign/output gauge')
 need(chart['weights']==inp['weights']and len(chart['weights'])==470 and all(F(v)>0 for v in chart['weights'])and F(chart['weights'][-2])==1 and F(chart['weights'][-1])==F(1,1048576)and inp['map_inputs']==ib['model'],'same full model and actual finite weights')
 w.save('input-binding.json',dict(card=card,known=known,model=ib['model'],native_source=ib['native_source'],weights=chart['weights'],scope='Same literal output gauge with exact previously proved endpoint scalar; no phase/sign transform.'))
 return known,chart,ib['model'],chart['weights']

def next_parameter(known,width):
 need(type(width)is int and -24<=width<=-12,'bounded exact output width')
 left=F(known);h=F(2)**width;return str(left+h),str(left),str(left+2*h)

def admission(w,card):
 pr=read(OUT/'profile.json');old=card['producer']['outcome'];total=sum(v['bytes']for v in read(Path(card['producer']['root'])/'run/manifest.json')['files'].values());unit_seconds=max(pr['forecast_seconds'],2*old['seconds']+15);unit_bytes=max(pr['forecast_bytes'],2*total+32*1024**2);rss=max(pr['forecast_rss_kib'],2*old['peak_rss_kib']+256*1024);f=dict(seconds=time.monotonic()-w.start+24*unit_seconds,raw_bytes=w.used+24*unit_bytes+16*1024**2,rss_kib=rss,unit_seconds=unit_seconds,unit_bytes=unit_bytes)
 ok=f['seconds']<6000 and f['raw_bytes']<6*1024**3 and rss<4*1024**2 and unit_seconds<240 and unit_bytes<768*1024**2;w.save('whole-packet-admission.json',dict(passed=ok,forecast=f,profile=pr,actual_prior=old,native_calls=3636,ticks=3786));need(ok,'measured entire24-attempt allowance before target arrays')

def finite_initial(chart,parameter):
 d=exact(str(F(parameter)-F(chart['parameter_center'])));return [exact_mid(original_ball(c)+original_ball(s)*d+original_ball(a)*d*d/2)for c,s,a in zip(chart['center'],chart['slope'],chart['curvature'],strict=True)]

def thin(chart):return {k:chart[k]for k in('center','slope','curvature','parameter_center','normal','origin','weights')}

def march(w,card):
 known,old,model,weights=binding(w,card);old=thin(old);width=-18;strong=0;accepted=[];attempts=[];reason='attempt_allowance_exhausted'
 for index in range(1,25):
  w.unit(index);parameter,left,right=next_parameter(known['exact_parameter'],width);initial=finite_initial(old,parameter)
  with ctx.workprec(512):box=[original_ball(v)for v in known['box']]
  anchor=dict(box=box,exact_parameter=left,period=18,sigma=-1,dimension=13);settings=dict(halfwidth_exponent=width,radius_exponent=-12)
  w.save('inputs.json',dict(initial=initial,known=known,normal=known['normal'],origin=known['origin'],parameter=parameter,parameter_endpoints=[left,right],weights=weights,settings=settings,model=model,source_card=entry(ROOT/CARD),previous_accepted=accepted[-1]if accepted else None,scope='Finite quadratic extrapolation only; all fresh native domains and complete470 root/primitive/join gates remain required.'))
  before_calls=w.native;before_ticks=w.ticks;start=time.monotonic()
  try:
   r=C.chart(C.Retained(C.C.TwoRateCuspMap(model),w.observe),initial,known['normal'],known['origin'],parameter,weights,anchor,settings,w)
   need(w.native-before_calls==144 and w.ticks-before_ticks==150,'complete one144-native/150-tick attempt')
   g=r['gates'];q=original_ball(r['bound']['contraction']);upper=F(int(q.upper().man_exp()[0]))*F(2)**int(q.upper().man_exp()[1])
   need(len(r['bound']['coordinate_inclusions'])==len(r['anchor_check']['coordinate_inclusions'])==470,'entire native root/join coordinates')
   decision=next_state(width,strong,uniform=g['uniform_normalized_eigenpair_locus'],primitive=g['primitive_period'],joined=g['anchor_identified'],contraction=upper,minimum=-24,maximum=-12)
   row=dict(index=index,passed=r['passed'],gates=g,contraction=r['bound']['contraction'],strict_inclusions=sum(r['bound']['coordinate_inclusions']),anchor_inclusions=sum(r['anchor_check']['coordinate_inclusions']),parameter_endpoints=[left,right],physical_endpoints=[v[-2:]for v in r['endpoint_root_boxes']],decision=decision,native_calls=144,ticks=150,seconds=time.monotonic()-start,certificate=entry(OUT/w.prefix/'target-certificate.json.gz'))
   w.save('outcome.json',row);attempts.append(row)
   if decision['action']=='advance':
    need(r['passed']and row['strict_inclusions']==row['anchor_inclusions']==470,'no advancement without all full gates');known=dict(box=r['endpoint_root_boxes'][1],normal=r['normal'],origin=r['origin'],exact_parameter=right,dimension=13,period=18,sigma=-1);old=thin(r);accepted.append(index);w.save('connected-right-endpoint.json',known)
    if original_ball(known['box'][-1]).lower()>=exact(POLICY['output_stop']):reason='physical_output_stop_reached';break
   elif decision['action']=='stop':reason=decision['reason'];break
   width,strong=decision['width'],decision['strong'];del r;gc.collect()
  except BaseException as e:
   row=dict(index=index,passed=False,status='incomplete_or_guard_failure',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),native_calls=w.native-before_calls,ticks=w.ticks-before_ticks,seconds=time.monotonic()-start);w.save('incomplete.json',row,True);attempts.append(row);reason='incomplete_or_guard_failure';break
 w.prefix='';w.unit_start=time.monotonic();w.unit_used=0
 if accepted:w.save('connected-right-endpoint.json',known)
 return dict(passed=bool(accepted)and reason in('attempt_allowance_exhausted','physical_output_stop_reached'),connected=bool(accepted),reason=reason,accepted_indices=accepted,attempts=attempts,last_known_parameter=known['exact_parameter'],last_known_physical_rates=known['box'][-2:],full_window=False,scope='Retained bounded exact-gauge primitive18 signed470 connected prefix. No complementary stability, generic flip or second-junction/full-window connection.')

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('6000s owned signed470 march')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,6000)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),policy=POLICY,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw);w.unit_used+=len(raw)
  with ctx.workprec(256):
   card=read(ROOT/CARD);need(card['status']=='reviewed'and card['policy']==POLICY and card['limits']==LIMITS,'source-admitted exact policy before any profile');C.profile(w);binding(w,card,True);admission(w,card);ans['target_acquired']=True;ans.update(march(w,card))
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<6000 and peak<4*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak,native_calls=w.native,ticks=w.ticks);w.prefix='';w.unit_used=0;w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
