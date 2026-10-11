"""One fixed saved independent512 signed470 root and same-junction join."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_signed470_locus_math as E
import review_signed470_anchor as A
need=E.need
OUT=Path('/data2/neural-shrimp-data/audits/signed470-independent/attempt001/run');CARD='docs/plans/native470-independent-inputs.json'
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
 else:b=p.read_bytes()
 need(len(b)<=256*1024**2,'bounded immutable read');return json.loads(b)
FILES=sorted(set([CARD,'docs/plans/native470-independent.md','tools/reviews/review_signed470.py','tools/launch_signed470_review.py','tests/reviews/test_signed470_review.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s4GiB')
 def save(self,name,row,closing=False):
  b=(json.dumps(row,sort_keys=True,indent=2)+'\n').encode();need(self.used+len(b)<(64 if closing else 62)*1024**2,'64MiB with2MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)
  if not closing:self.tick()
def inventory(p,pin,w):
 need(entry(p/'manifest.json')['sha256']==pin,'literal immutable manifest');m=read(p/'manifest.json')['files'];need(set(m)=={str(f.relative_to(p))for f in p.rglob('*')if f.is_file()and f!=p/'manifest.json'},'complete immutable file set')
 for n,v in m.items():need(entry(p/n)==v and not(p/n).stat().st_mode&0o222,'complete immutable member');w.tick()
def archive(p,source,w):
 rt=read(p/'run/runtime.json');la=read(p/'launcher-result.json');need(rt.get('source_commit',rt.get('commit'))==la.get('source_commit',la.get('commit'))==source and not la['owned_live_pids'],'literal source/owner');need(not Path('/proc',str(rt['pid'])).exists(),'owned worker gone')
 for k in ('launcher_pid','owned_group_pid'):
  if k in la:need(not Path('/proc',str(la[k])).exists(),'owned group/launcher gone')
 for n,v in rt.get('sources',rt.get('files')).items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'entire historical source');w.tick()
 return rt

def binding(card,w):
 need(card['status']=='reviewed'and card['schema']=='independent-signed470-v1'and card['precision']==512,'fixed source-reviewed audit');s=card['producer'];p=Path(s['root']);k=card['known705'];a=Path(k['root'])
 for spec,base in [(s,p),(k,a)]:
  inventory(base,spec['outer']['sha256'],w);inventory(base/'run',spec['worker']['sha256'],w);archive(base,spec['source'],w)
 o=read(p/'run/outcome.json');need(o==s['outcome']and o['passed']and o['native_calls']==276 and o['ticks']==278 and o['events']==842 and o['strict_inclusions']==o['anchor_inclusions']==470,'complete fresh weighted root')
 need(entry(p/'run/target-certificate.json.gz')==s['certificate']and entry(p/'run/target-inputs.json')==s['inputs'],'actual weighted acquired arrays')
 ao=read(a/'run/outcome.json');ib=read(a/'run/input-binding.json');need(ao==k['outcome']and ao['passed']and ao['coordinates']==705 and ao['native_calls']==0 and ib==k['binding']and entry(a/'run/independent705.json')==k['box'],'independent rational full55 refinement/native705 authority')
 parent=Path(ib['producer'])
 for b,pin in zip((parent,parent/'run'),ib['producer_pins'],strict=True):inventory(b,pin,w)
 pr=archive(parent,ib['producer_source'],w);inp=read(p/'run/target-inputs.json');physical=read(parent/'run/input-binding.json')['map_inputs'];need(inp['map_inputs']==physical,'same exact training map/data')
 current=read(p/'run/runtime.json')
 for n,v in ib['native_source'].items():need(v==pr['sources'][n]==current['sources'][n],'same exact native derivative sources')
 # Authenticate the separate failed domain and saved proposal, but never use its derivatives.
 extra=read(p/'run/weighted-binding.json')
 for key in ('failed','diagnosis'):
  r=extra[key];b=Path(r['root']);inventory(b,r['outer']['sha256'],w);inventory(b/'run',r['worker']['sha256'],w);archive(b,r['source'],w)
 need(inp['weights']==extra['weights']and len(inp['weights'])==470 and inp['settings']==dict(period=18,sigma=-1,halfwidth_exponent=-8,radius_exponent=-14,newton_steps=2,precision=256),'actual exact weighted settings')
 w.save('input-binding.json',dict(card=card,model=physical,native_source=ib['native_source'],old_derivative_reused=False));return p,parent,inp

def profile(w,card):
 n=470;start=time.monotonic();C=arb_mat([[arb(int(i==j))+arb((1,-30))for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))+arb((1,-31))for j in range(n)]for i in range(n)]);H=arb_mat(n,4);W=arb_mat(n,4)
 for i in range(n):H[i,0]=arb((1,-70));W[i,3]=arb((1,-65))
 r=E.reduction(C,J,H,W,['1']*n,F(1,2**14),F(1,256));need(r['root'],'full470 unrelated arithmetic profile');elapsed=time.monotonic()-start;old=card['producer']['outcome'];forecast=elapsed+2*old['seconds']+30;peak=max(2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024,2*old['peak_rss_kib']+256*1024);w.save('profile.json',dict(passed=forecast<300 and peak<4*1024**2,seconds=elapsed,forecast_seconds=forecast,forecast_rss_kib=peak,formula='Dense C=I+2^-30 ones, J=I+2^-31 ones,H0=2^-70,W3=2^-65; full470 weighted polynomial reduction. No native map or target arrays.',target_arrays_unopened=True));need(forecast<300 and peak<4*1024**2,'measured and actual cost admission')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('300s owned')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),precision=512,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,sources={n:entry(ROOT/n)for n in FILES},threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(hashlib.sha256(b).digest()==hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).digest(),'own exact source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):
   card=read(ROOT/CARD);profile(w,card);p,parent,inp=binding(card,w);ref=read(parent/'run/known-root-refinement.json')['box'];unit=read(Path(card['known705']['root'])/'run/independent705.json')['box'];paths=sorted((parent/'run/events').glob('*.json'));need(len(paths)==228,'entire earlier lift events');returns=[read(paths[157+4*k+2*j])for k in range(18)for j in range(2)];anchor=A.negative_lift(ref,unit,returns,tick=w.tick);w.save('independent470-anchor.json',dict(box=anchor,coordinates=470,sigma=-1,scope='Exact rational interval negative-mode products conditional on independently certified same55 root/refinement.'))
   c=read(p/'run/target-certificate.json.gz');need(c['sigma']==-1 and c['weights']==inp['weights']and c['normal']==inp['normal']and c['origin']==inp['origin'],'actual weighted/gauge native certificate');events=[]
   for path in sorted((p/'run/events').glob('*-target-*.json.gz')):
    kind=path.name.split('-target-',1)[1].removesuffix('.json.gz')
    if kind in ('phase_return','path_phase_return','path_assembly_return'):events.append((kind,read(path)))
   proof=E.check_chart(c,dict(box=anchor),events,512,expected_settings={k:inp['settings'][k]for k in ('precision','newton_steps','halfwidth_exponent','radius_exponent')},expected_sigma=-1);w.save('independent512.json',proof);need(proof['passed']and proof['inclusions']==proof['anchor_coordinates']==470,'all470 independent root/junction join');ans.update(passed=True,contraction=proof['contraction'],root_coordinates=470,anchor_coordinates=470,scope='Independent saved512 full470 native/path assembly, uniform weighted root and projected same-junction membership. Shared analytic native returns/Arb dependency; no full-chart complement or complete window claim.')
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);ans.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
