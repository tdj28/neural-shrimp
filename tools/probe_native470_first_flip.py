"""One measured strict native470 sigma=-1 tube through the known junction."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native470_junction_saved as L
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube,path_coefficients
from neural_shrimp.rigorous_eigenpair_locus import phase,exact_mid
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,canonical_bytes
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
from neural_shrimp.native470_first_components import SyntheticFlip,synthetic_fixture,RetainedMap
need,read,entry=L.need,L.read,L.entry
PARENT=L.OUT.parent
OUT=PARENT.parents[1]/'signed-flip-first001/attempt001/run'
CARD='docs/plans/native470-first-flip-inputs.json'
SETTINGS=dict(period=18,sigma=-1,halfwidth_exponent=-8,radius_exponent=-14,newton_steps=2,precision=256)
FILES=[CARD,'docs/plans/native470-first-flip.md','tools/probe_native470_first_flip.py','tools/launch_native470_first_flip.py','tests/test_native470_first_flip.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
def settings():return dict(chart=SETTINGS,seconds=240,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=768*1024**2,profile_ticks=151,target_ticks=127,native_calls=276)
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.ticks=0;self.serial=0;self.stage='profile';self.native=0
 def check(self):need(time.monotonic()-self.start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
 def tick(self):self.ticks+=1;need(self.ticks<=278,'151profile+127target inclusive ticks');self.check()
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);payload=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw;need(self.used+len(payload)<(768 if closing else 760)*1024**2,'768MiB with8MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(payload);f.flush();os.fsync(f.fileno())
  self.used+=len(payload)
  if not closing:self.check()
 def observe(self,kind,row):
  if kind=='native_input':self.native+=1;need(self.native<=276,'all inclusive native acquisitions')
  if kind=='chart_return':self.save(self.stage+'-certificate.json.gz',row)
  else:self.save(f'events/{self.serial:04d}-{self.stage}-{kind}.json.gz',row);self.serial+=1

def profile(w):
 ini,kw=synthetic_fixture();w.save('profile-fixture.json',dict(initial=ini,settings={k:v for k,v in kw.items()if k!='anchor'},scope='Repeated primitive2 native sigma−1, not primitive18'))
 start=time.monotonic();used=w.used;model=RetainedMap(SyntheticFlip(),w.observe);r,_=quadratic_tube(model,ini,**kw,tick=w.tick,observe=w.observe);chart_time=time.monotonic()-start;need(r['gates']['uniform_normalized_eigenpair_locus']and r['gates']['anchor_identified']and not r['gates']['primitive_period'],'unrelated full470 signed root and false primitive18 control');del r
 inp,z=L.K.fixture();n=13;x=z[:n];q=z[n+2:2*n+2];rates=z[n:n+2];w.save('profile-native-fixture.json',dict(map_inputs=inp,state=vector_json(x),mode=vector_json(q),rates=vector_json(rates)));native=RetainedMap(TwoRateCuspMap(inp),w.observe);initial=x+q+rates;slope=[exact(str(F(i-14,128)))for i in range(28)];curve=[exact(str(F(7-i,256)))for i in range(28)];normal=[arb(0)]*26+[arb(1),arb(0)];delta=arb(0,arb((1,-14)));mt=[];pt=[]
 for k in range(12):
  w.tick();t=time.monotonic();v,j,e=phase(native,x,q,rates);w.save(f'profile-native-mixed-{k}.json.gz',dict(value=matrix_json(v),block=matrix_json(j),rates=matrix_json(e)));mt.append(time.monotonic()-t)
  t=time.monotonic();p=path_coefficients(native,initial,slope,curve,delta,1,-1,normal,initial,arb(0),w.tick,w.observe);w.save(f'profile-native-path-{k}.json.gz',dict(coefficients=matrix_json(p)));pt.append(time.monotonic()-t)
 elapsed=time.monotonic()-w.start;seconds=elapsed+2*(chart_time+72*max(mt)+54*max(pt))+15;raw=4*(w.used-used)+32*1024**2;rss=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024;ok=seconds<240 and raw<768*1024**2 and rss<4*1024**2
 w.save('profile.json',dict(passed=ok,chart_seconds=chart_time,mixed_seconds=mt,path_seconds=pt,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True,profile_ticks=w.ticks,native_calls=w.native));need(ok and w.ticks==151 and w.native==150,'complete measured signed470 profile before target')

def binding(w,card):
 need(card['status']=='reviewed'and card['settings']==settings(),'fixed reviewed one-chart design')
 spec=card['anchor'];need(spec['root']==str(PARENT),'literal signed junction parent')
 for b,pin in((PARENT,spec['outer']),(PARENT/'run',spec['worker'])):L.K.inventory(b,pin,w)
 rt=read(PARENT/'run/runtime.json');out=read(PARENT/'run/outcome.json');owner=read(PARENT/'launcher-result.json');known=read(PARENT/'run/known470.json');ib=read(PARENT/'run/input-binding.json')
 need(rt['source_commit']==owner['source_commit']==spec['source']and rt['settings']==L.settings()and out['passed']and out['native_calls']==0 and out['saved_phase_blocks']==54 and out['status']=='same_known_junction_lifted_to_signed470','complete signed same-junction lift')
 need(owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent']and all(not Path('/proc',str(owner[k])).exists()for k in('launcher_pid','owned_group_pid')),'all original signed owners gone')
 need(entry(PARENT/'run/known470.json')==spec['known470']and known['dimension']==13 and known['period']==18 and known['sigma']==-1 and known['shooting_dimension']==470 and len(known['box'])==470 and all(known['sanity'].values()),'whole exact known signed470 root')
 for n,row in rt['sources'].items():need(entry(PARENT/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical signed source '+n);w.check()
 for n in L.K.NATIVE:need(entry(ROOT/n)==rt['sources'][n]==ib['native_sources'][n],'identical actual native map '+n)
 need(ib['source']==L.SOURCE and ib['outer']==L.PINS[0]and ib['worker']==L.PINS[1],'literal original705 and55 authority')
 w.save('binding.json',dict(anchor=spec,source_archive=rt['sources'],same55=ib['independent55'],scope='Same known root, exact phase/sign/rates. No full-region complementary stability inherited.'));return known,ib['independent55']['map_inputs']

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('240s owned worker')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=commit,pid=os.getpid(),settings=settings(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads={k:os.environ[k]for k in THREADS}))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'exact committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(256):
   profile(w);card=read(ROOT/CARD);known,inp=binding(w,card)
   # Restore512-bit inherited balls exactly before proposing a256-bit finite center.
  with ctx.workprec(512):box=[original_ball(v)for v in known['box']]
  with ctx.workprec(256):
   initial=[exact_mid(v+arb(0))for v in box];normal=['0']*468+['1','0'];anchor=dict(box=box,period=18,sigma=-1,dimension=13);w.stage='target';w.save('target-inputs.json',dict(map_inputs=inp,known=known,initial=initial,normal=normal,origin=initial,parameter='0',settings=SETTINGS,weights=['1']*470));result['target_acquired']=True
   r,_=quadratic_tube(RetainedMap(TwoRateCuspMap(inp),w.observe),initial,normal=normal,origin=initial,parameter='0',anchor=anchor,**SETTINGS,tick=w.tick,observe=w.observe)
   need(w.ticks==278 and w.native==276,'complete profile and one native470 tube');result.update(passed=r['passed'],status='anchored_signed470_tube_passed'if r['passed']else'complete_bound_failure',gates=r['gates'],contraction=r['contraction'],strict_inclusions=sum(r['coordinate_inclusions']),anchor_inclusions=sum(r['anchor_check']['coordinate_inclusions']),anchor_parameter_inside=r['anchor_check']['parameter_inside'],physical_endpoints=[v[-2:]for v in r['endpoint_root_boxes']],scope='Connected local primitive18 sigma−1 locus through the same certified junction; no full-chart complement or generic flip normal form follows.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=w.ticks,native_calls=w.native,events=w.serial);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
