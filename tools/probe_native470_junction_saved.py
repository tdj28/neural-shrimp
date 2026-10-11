"""Fixed saved-only signed470 lift of the already proved same junction."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp.rigorous_signed_junction_lift import saved_flip_lift,identical
from neural_shrimp.rigorous_orbits import ball_json,vector_json,canonical_bytes
import probe_native705_junction_lift as K
need,read,entry=K.need,K.read,K.entry
PARENT=K.OUT.parent
SOURCE='0ba9605de2eb98862bda5c09f436b9d4a15f93ee'
PINS=['ebfb53346cfc47589ecd24dc14ef8e29c25e5ba62cbb616ceda97176cc4eadb4','d2c869aa01524a394efe65ba033c62b656c3f206e56bd331db799be7293f7411']
OUT=PARENT.parents[1]/'signed-junction-lift001/attempt001/run'
FILES=['tools/probe_native470_junction_saved.py','tools/launch_native470_junction_saved.py','tests/test_signed_junction_lift.py','tests/test_native470_junction_saved.py','docs/plans/native470-junction-saved.md','docs/reviews/native705-junction-lift003.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
def settings():return dict(precision=512,period=18,profile_phase_blocks=36,target_phase_blocks=18,native_calls=0,seconds=30,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=40*1024**2)
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.serial=0;self.blocks=0;self.stage='profile'
 def check(self):need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'30s/2GiB cap')
 def tick(self):self.blocks+=1;need(self.blocks<=54,'36synthetic+18saved target blocks');self.check()
 def save(self,name,value,closing=False):
  raw=canonical_bytes(value);need(self.used+len(raw)<(40 if closing else 38)*1024**2,'40MiB with2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def observe(self,kind,row):self.save(f'events/{self.serial:03d}-{self.stage}-{kind}.json',row);self.serial+=1;self.check()

def synthetic():
 n=13;p=18;b=lambda v:ball_json(v if isinstance(v,arb)else arb(v));x=[b(0)]*n;q=[b(int(i==1))for i in range(n)];r=[b(int(i==2))for i in range(n)];w=[b(int(i==0))for i in range(n)];rates=[b(1),b(2)];root=x+rates+q+r+[b(arb(1)/8)]+w;unit=(x+q+r)*p+rates+[b(arb(1)/8)];records=[]
 for k in range(p):
  B=[[b(arb((-1 if k==0 else 1)if i==j==0 else int(i==j))+arb((i*31+j+1,-450))+arb(0,arb((1,-160))))for j in range(2*n)]for i in range(2*n)]
  for name,mode in [('q',q),('r',r)]:records.append(dict(phase=k,direction=name,input=x+mode+rates,value=[[v]for v in x+mode],block=B))
 return root,unit,records

def profile(w):
 times=[];sizes=[]
 for k in range(2):
  w.stage=f'profile{k}';start=time.monotonic();used=w.used;root,unit,records=synthetic();w.save(f'profile{k}-input.json',dict(root=root,unit=unit,records=records));flat,result=saved_flip_lift(root,unit,records,n=13,period=18,tick=w.tick,observe=w.observe);w.save(f'profile{k}-result.json',result);need(len(flat)==470 and all(result['sanity'].values()),'full470 unrelated signed algebra');times.append(time.monotonic()-start);sizes.append(w.used-used)
 seconds=4*max(times)+10;raw=4*max(sizes)+8*1024**2;rss=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024;passed=seconds<30 and raw<40*1024**2 and rss<2*1024**2
 w.save('profile.json',dict(passed=passed,seconds=times,bytes=sizes,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True));need(passed and w.blocks==36,'full54-block measured admission')

def bind(w):
 for base,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):K.inventory(base,pin,w)
 rt=read(PARENT/'run/runtime.json');owner=read(PARENT/'launcher-result.json');o=read(PARENT/'run/outcome.json');ib=read(PARENT/'run/input-binding.json');ref=read(PARENT/'run/known-root-refinement.json');lift=read(PARENT/'run/native705-lift.json')
 need(rt['source_commit']==owner['source_commit']==SOURCE and rt['settings']==K.settings()and o['passed']and o['native_calls']==108 and o['events']==228 and o['status']=='same_independent_junction_lifted_to_native705','exact complete same55-to705 execution')
 need(owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent']and all(not Path('/proc',str(owner[k])).exists()for k in('launcher_pid','owned_group_pid')),'all original owners gone')
 need(ib['independent_root']==str(K.PARENT)and ib['source']==K.SOURCE and ib['outer']==K.PINS[0]and ib['worker']==K.PINS[1],'same independently certified local55 junction authority')
 for n,row in rt['sources'].items():need(entry(PARENT/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'entire original source '+n);w.check()
 for n in K.NATIVE:need(entry(ROOT/n)==rt['sources'][n]==ib['native_source'][n],'identical actual native equations '+n)
 need(ref['passed']and ref['dimension']==55 and lift['shooting_dimension']==705 and lift['native_calls']==36 and lift['closure_contains_zero']and lift['normalization_contains_zero'],'same whole refined root and completed literal lift')
 events=sorted((PARENT/'run/events').glob('*.json'));need(len(events)==228 and [int(p.name.split('-')[0])for p in events]==list(range(228)),'complete original acquisition sequence');records=[]
 for k in range(18):
  for j,direction in enumerate(('q','r')):
   inp=read(events[156+4*k+2*j]);ret=read(events[157+4*k+2*j]);need(inp['phase']==ret['phase']==k and inp['direction']==ret['direction']==direction,'exact original phase ordering');need(identical(ret['input'],inp['state']+inp['mode']+inp['rates']),'actual pre-call and returned input identity');records.append(ret)
 need(all(identical(inp,old)for inp,old in zip(lift['box'][:13],ref['box'][:13],strict=True)),'same state root')
 w.save('input-binding.json',dict(parent=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],independent55=ib,refinement=entry(PARENT/'run/known-root-refinement.json'),lift=entry(PARENT/'run/native705-lift.json'),native_sources={n:rt['sources'][n]for n in K.NATIVE}));return ref['box'],lift['box'],records

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,target_reduced=False,native_calls=0)
 def stop(*_):raise TimeoutError('30s owned worker')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=settings(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):
   profile(w);root,unit,records=bind(w);w.stage='target';result['target_reduced']=True;flat,proof=saved_flip_lift(root,unit,records,n=13,period=18,tick=w.tick,observe=w.observe);w.save('known470.json',proof);need(w.blocks==54 and all(proof['sanity'].values())and len(flat)==470,'complete conditional signed lift');result.update(passed=True,status='same_known_junction_lifted_to_signed470',physical_rates=vector_json(flat[-2:]),scope='Conditional exact signed lift of the same independently proved junction. No finite flip curve or extended complementary spectrum follows before separate strict470 root/tube membership.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,saved_phase_blocks=w.blocks,events=w.serial);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
