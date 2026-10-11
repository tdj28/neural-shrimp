"""Fresh nested480-bit second junction with independently authored arithmetic."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_second_unit_flip as C
import refine_second_unit_flip_crossing as B
from neural_shrimp import junction_audit as A
from neural_shrimp.rigorous_unit_pair_flip import augmented,root_proof
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.native470_first_components import RetainedMap
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json,canonical_bytes
need,read,entry=C.need,C.read,C.entry
OUT=Path('/data2/neural-shrimp-data/audits/second-unit-flip-independent/attempt002/run');CARD='docs/plans/second-unit-flip-independent-inputs.json'
SETTINGS=dict(period=18,precision=480,newton_steps=0,radius_exponent=-96)
LIMITS=dict(seconds=60,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=128*1024**2,native_calls=162,profile_calls=54,target_calls=108)
FILES=sorted(set(B.FILES+[CARD,'docs/reviews/second-unit-flip-independent001-failure.json','docs/plans/second-unit-flip-independent.md','docs/reviews/second-unit-flip-crossing001.json','docs/reviews/second-unit-flip-audit-provenance.json','src/neural_shrimp/junction_audit.py','tests/test_junction_audit.py','tools/audit_second_unit_flip_nested.py','tools/launch_second_unit_flip_nested.py','tests/test_second_unit_flip_nested.py']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.calls=0;self.native=0;self.serial=0;self.stage='profile'
 def check(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'60s2GiB')
 def tick(self):self.calls+=1;need(self.calls<=162,'54profile+108target');self.check()
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);data=gzip.compress(raw,mtime=0)if n.endswith('.gz')else raw;need(self.used+len(data)<(128 if closing else 124)*1024**2,'128MiB4MiBclosing');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  self.used+=len(data)
  if not closing:self.check()
 def observe(self,k,v):
  if k=='native_input':self.native+=1;need(self.native<=162,'inclusive retained native attempts')
  self.save(f'events/{self.serial:04d}-{self.stage}-{k}.json.gz',v);self.serial+=1

class Prefix:
 def __init__(self,w,prefix):self.w=w;self.prefix=prefix
 def check(self):self.w.check()
 def save(self,n,v,closing=False):return self.w.save(self.prefix+n,v,closing)

def profile(w):
 inp,z=C.N.C.C.L.K.fixture();wv=[arb(i+1)for i in range(13)];scale=sum((v*v for v in wv),arb(0)).sqrt();seed=[v.mid()for v in z[:41]]+[arb(1)/100]+[(v/scale).mid()for v in wv];box=[arb(v.mid(),arb((1,-96)))for v in seed];w.save('profile-inputs.json',dict(map_inputs=inp,box=vector_json(box)));model=RetainedMap(TwoRateCuspMap(inp),w.observe);start=time.monotonic();before=w.used;v,J,_=augmented(model,box,18,w.tick);w.save('profile-evaluation.json.gz',dict(value=matrix_json(v),jacobian=matrix_json(J)));elapsed=time.monotonic()-start;size=w.used-before
 need(all(v[i,0].is_finite()for i in range(55))and all(J[i,j].is_finite()for i in range(55)for j in range(55)),'full finite55 profile');forecast=dict(seconds=time.monotonic()-w.start+6*elapsed+15,raw_bytes=w.used+6*size+16*1024**2,rss_kib=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024);ok=forecast['seconds']<60 and forecast['raw_bytes']<128*1024**2 and forecast['rss_kib']<2*1024**2;w.save('profile.json',dict(passed=ok,seconds=elapsed,bytes=size,forecast=forecast,native_calls=w.calls,target_arrays_unopened=True));need(ok and w.calls==w.native==54,'complete unrelated profile before arrays')

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'fixed reviewed nested audit')
 for n,e in card['independent_reducer']['files'].items():need(entry(ROOT/n)==e,'unchanged independent arithmetic source')
 prior,_=C.C.history(w,card['prior_failure'],2);failure=read(prior/'run/outcome.json');need(prior==OUT.parents[1]/'attempt001'and failure==card['prior_failure']['outcome']and not failure['passed']and not failure['target_acquired']and failure['native_calls']==54 and failure['error'].startswith('FileExistsError:')and read(prior/'run/profile.json')['passed']and not list((prior/'run/events').glob('*-target-*')),'retained namespace failure before target')
 B.bind(w,card['root_card'],True);C.binding(w,card['original_card'],True)
 need(card['root_card']==read(ROOT/B.CARD)and card['original_card']==read(ROOT/C.CARD),'literal original source cards')
 failed,rt=C.C.history(w,card['crossing_failure'],2);o=read(failed/'run/outcome.json');need(failed==B.OUT.parent and o==card['crossing_failure']['outcome']and not o['passed']and o['native_calls']==0 and o['status']=='saved_bound_unresolved','complete retained saved crossing failure')
 if metadata_only:return C.OUT.parent
 root=B.bind(Prefix(w,'old-root-'),card['root_card']);ib=read(C.OUT/'input-binding.json');model,_,original=C.binding(Prefix(w,'numerical-'),card['original_card']);need(original==ib,'same complete candidate/map/bracket identity');proposal=read(C.OUT/'same-root-complement.json');need(proposal['passed'],'original stable10 same-root finite basis proposal')
 need(root['settings']==dict(newton_steps=4,radius_exponent=-80)and all(A.rat(v['rad'])==0 for v in root['center']),'exact original384 center')
 # Independently recompute the original existence proof; its unresolved crossing
 # is retained, not used as an audit prerequisite or silently relabelled.
 old=A.root_arithmetic(root);w.save('saved-root-arithmetic.json',dict(inclusions=old['inclusions'],contraction=ball_json(old['contraction']),divisors=old['divisors']))
 w.save('input-binding.json',dict(card=card,old_root=entry(C.OUT/'root.json'),map_inputs=ib['map_inputs'],native_source=ib['native_source'],same_center=root['center'],targets=ib['targets'],scope='Fresh nested same root, independently authored reductions; shared native derivative kernel/Arb.'))
 return model,root,proposal,ib

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0,target_acquired=False)
 def stop(*_):raise TimeoutError('60s owned nested55 audit')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed full source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   profile(w);model,old,proposal,ib=binding(w,read(ROOT/CARD));w.stage='target';ans['target_acquired']=True
   center=[A.decode(v)for v in old['center']];need(all(v.is_exact()for v in center),'old center restored without rounding');w.save('target-domain.json',dict(center=old['center'],radius_exponent=-96,newton_steps=0,map_inputs=ib['map_inputs']))
   fresh,_=root_proof(RetainedMap(model,w.observe),center,18,radius_exponent=-96,newton_steps=0,tick=w.tick,observe=w.observe);w.save('fresh-root.json',fresh);need(fresh['center']==old['center'],'literal exact55 same center')
   # Existence/uniqueness comes from the independent reduction, not the producer flag.
   a=A.root_arithmetic(fresh);w.save('fresh-root-arithmetic.json',dict(inclusions=a['inclusions'],contraction=ball_json(a['contraction']),divisors=a['divisors']));same=A.strict_nested(old['box'],fresh['box']);w.save('root-identity.json',dict(fresh_inside_original=same,old_box=old['box'],fresh_box=fresh['box']));C.bracket([A.decode(v)for v in fresh['box']],ib['targets']);s=A.spectrum_crossing(fresh,proposal);w.save('fresh-spectrum-crossing.json',dict(complement_bounds=[ball_json(v)for v in s['complement_bounds']],crossing=ball_json(s['crossing'])))
   need(w.calls==w.native==162,'exact54profile108target complete retained phases');ans.update(passed=True,same_unique_root=True,coordinates=55,contraction=ball_json(a['contraction']),complement_bounds=[ball_json(v)for v in s['complement_bounds']],crossing=ball_json(s['crossing']),rates=fresh['box'][13:15],angle_squared=fresh['box'][41],scope='Independent arithmetic plus fresh480 zero-Newton native derivatives in a strictly nested55 box. Primitive18, simple unit pair/minus1, stable10 and transverse physical crossing. No complete signed connection or generic junction normal form.')
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<60 and peak<2*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak,native_calls=w.calls,native_inputs=w.native,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
