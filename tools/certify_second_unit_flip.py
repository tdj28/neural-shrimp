"""One measured full55 interval qualification of the second numerical junction."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_second_unit_flip as N
import probe_native470_correlated as C
import certify_native_unit_pair_flip as I
from neural_shrimp.rigorous_unit_pair_flip import augmented,root_proof
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json,matrix_json,eye,midpoint_matrix,canonical_bytes
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.native470_first_components import RetainedMap
need,read,entry=N.need,N.read,N.entry
OUT=N.OUT.parents[1]/'rigorous001/attempt001/run';CARD='docs/plans/second-unit-flip-rigorous-inputs.json'
SETTINGS=dict(period=18,precision=384,newton_steps=4,radius_exponent=-80)
LIMITS=dict(seconds=120,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=256*1024**2,native_calls=432,profile_calls=108,target_calls=324)
FILES=sorted(set(N.FILES+I.SOURCES+C.FILES+[CARD,'docs/plans/second-unit-flip-rigorous.md','docs/reviews/second-unit-flip-localization001.json','tools/certify_second_unit_flip.py','tools/launch_second_unit_flip_rigorous.py','tests/test_second_unit_flip_rigorous.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.calls=0;self.serial=0;self.stage='profile';self.native=0
 def check(self):need(time.monotonic()-self.start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'120s/2GiB cap')
 def tick(self):self.calls+=1;need(self.calls<=432,'108profile+324target native call cap');self.check()
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);data=gzip.compress(raw,mtime=0)if n.endswith('.gz')else raw;need(self.used+len(data)<(256 if closing else 248)*1024**2,'256MiB/8MiBclosing');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  self.used+=len(data)
  if not closing:self.check()
 def observe(self,k,v):
  if k=='native_input':self.native+=1;need(self.native<=432,'inclusive native retention')
  self.save(f'events/{self.serial:04d}-{self.stage}-{k}.json.gz',v);self.serial+=1

def profile(w):
 inp,z=N.C.C.L.K.fixture();wvec=[exact(str(i+1))for i in range(13)];scale=sum((v*v for v in wvec),arb(0)).sqrt();seed=[v.mid()for v in z[:41]]+[exact('1/100')]+[(v/scale).mid()for v in wvec];model=RetainedMap(TwoRateCuspMap(inp),w.observe);w.save('profile-inputs.json',dict(map_inputs=inp,seed=vector_json(seed)));times=[];sizes=[]
 for i in range(2):
  box=[arb(v.mid(),arb((1,-80)))for v in seed];start=time.monotonic();used=w.used;w.save(f'profile-domain-{i}.json',dict(box=vector_json(box)));v,J,d=augmented(model,box,18,w.tick);pre=midpoint_matrix(J.inv());D=eye(55)-pre*J;image=pre*v;w.save(f'profile-{i}.json.gz',dict(value=matrix_json(v),jacobian=matrix_json(J),preconditioner=matrix_json(pre),defect=matrix_json(D),correction=matrix_json(image)));times.append(time.monotonic()-start);sizes.append(w.used-used)
 forecast=dict(seconds=time.monotonic()-w.start+12*max(times)+20,raw_bytes=w.used+12*max(sizes)+16*1024**2,rss_kib=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024)
 ok=forecast['seconds']<120 and forecast['raw_bytes']<256*1024**2 and forecast['rss_kib']<2*1024**2
 w.save('profile.json',dict(passed=ok,times=times,sizes=sizes,forecast=forecast,native_calls=w.calls,target_arrays_unopened=True));need(ok and w.calls==w.native==108,'complete full55 measured profile before target')

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'literal reviewed second interval55 packet');N.binding(w,card['numerical_card'],metadata_only=True)
 s=card['numerical'];p,rt=C.history(w,s);need(p==N.OUT.parent and rt['settings']==N.SETTINGS,'literal second numerical source/settings');o=read(p/'run/outcome.json');need(o==s['outcome']and o['passed']and o['status']=='second_numerical_unitpair_flip_localized'and o['counts']==dict(augmented=4,section=7,adjacent=1)and o['native_calls']==900,'complete actual second candidate')
 need(read(p/'run/source'/N.CARD)==card['numerical_card']==read(ROOT/N.CARD),'entire numerical prior card source binding')
 for n,row in rt['sources'].items():need(entry(ROOT/n)==row,'unchanged actual numerical/native source '+n)
 for n in ('candidate.json','input-binding.json'):need(entry(p/'run'/n)==s['files'][n],'literal saved candidate file')
 if metadata_only:return p
 c=read(p/'run/candidate.json');ib=read(p/'run/input-binding.json');need(c['passed']and len(c['y'])==55 and c['native_diagnostics']['passed']and c['phase0_spectrum']['passed']and c['phase1_spectrum']['passed']and c['crossing_derivative']['passed'],'full candidate/native qualification')
 need([r['target']for r in c['bracket_reverse']]==[22,23]and all(r['passed']and r['coordinates']==470 and r['relative_error']<1e-7 for r in c['bracket_reverse']),'two literal full470 numerical joins')
 need(c['y'][13:15]==o['rates']and c['y'][41]==o['angle_squared']and ib['previous']==card['numerical_card']['previous'],'literal seed and parent bracket');seed=[exact(dyadic(v))for v in c['y']];inp=ib['map_inputs'];model=TwoRateCuspMap(inp);need(model.n==13 and len(inp['x'])==24,'native13D24sample model')
 b=dict(numerical=s,numerical_card=card['numerical_card'],map_inputs=inp,seed=vector_json(seed),targets=ib['targets'],candidate=entry(p/'run/candidate.json'),native_source={n:entry(ROOT/n)for n in C.C.L.K.NATIVE},scope='Isolated same-map second junction; numerical signed route is not a rigorous full boundary connection.');w.save('input-binding.json',b);return model,seed,b

def bracket(box,targets):
 need(len(box)==55 and len(targets)==2,'whole55 bracket input')
 for i,key in ((13,'hidden'),(14,'output')):
  low,high=sorted(exact(dyadic(v[key]))for v in targets);need(low<box[i]and box[i]<high,'whole proved root strictly within both actual physical-rate brackets')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('120s owned interval second55')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   profile(w);model,seed,b=binding(w,read(ROOT/CARD));w.stage='target';ans['target_acquired']=True;m=RetainedMap(model,w.observe)
   def observe(k,v):
    w.observe(k,v)
    if 'center'in v:need(all(abs(decode(a)-s)<arb((1,-20))*max(arb(1),abs(s))for a,s in zip(v['center'],seed,strict=True)),'all55 original candidate proximity')
   proof,_=root_proof(m,seed,18,radius_exponent=-80,newton_steps=4,tick=w.tick,observe=observe);w.save('root.json',proof);need(proof['passed'],'strict full55 root/primitive/rates/angle');box=[decode(v)for v in proof['box']];bracket(box,b['targets']);spectrum,crossing=I.qualify_same_root(proof);w.save('same-root-complement.json',spectrum);w.save('same-root-crossing.json',crossing)
   need(w.calls==w.native==432,'complete108profile+324target native inputs/returns');ans.update(passed=spectrum['passed']and crossing['passed'],root_passed=True,complement_passed=spectrum['passed'],crossing_passed=crossing['passed'],rates=proof['box'][13:15],angle_squared=proof['box'][41],contraction=proof['contraction'],crossing=crossing['multiplier_hidden_rate_derivative'],status='qualified_second_native_unitpair_flip'if spectrum['passed']and crossing['passed']else'complete_second_qualification_unresolved',scope='Exact local primitive18 second unit-pair/flip with conditional same-root stable10 complement/crossing. Entire signed470 connection, normal-form criticality and full window remain separate.')
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<120 and peak<2*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak,native_calls=w.calls,native_inputs=w.native,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
