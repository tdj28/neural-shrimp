"""Fixed saved signed470 budget; no native acquisitions or new root proof."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,flint
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native470_first_flip as C
from neural_shrimp.signed_locus_conditioning import budget
need,read,entry=C.need,C.read,C.entry
PARENT=C.OUT.parent;OUT=PARENT.parent/'diagnosis001/run'
FILES=sorted(set(C.FILES+['src/neural_shrimp/unit_pair_conditioning.py','src/neural_shrimp/signed_locus_conditioning.py','tools/diagnose_native470_first_flip.py','tools/launch_native470_first_diagnosis.py','tests/test_signed_locus_conditioning.py','docs/plans/native470-first-diagnosis.md','docs/reviews/native470-first-flip001.json']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def check(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'60s/4GiB')
 def save(self,name,row,closing=False):
  raw=(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(self.used+len(raw)<(16 if closing else 15)*1024**2,'16MiB reserve');q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.check()

def binding(w):
 spec=read(ROOT/'docs/reviews/native470-first-flip001.json');need(spec['root']==str(PARENT)and spec['source']=='08892068bf9ef775cca988a06331170b5316e290','literal completed signed failure')
 for b,pin in((PARENT,spec['outer']),(PARENT/'run',spec['worker'])):C.L.K.inventory(b,pin['sha256'],w)
 rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');o=read(PARENT/'run/outcome.json');need(rt['source_commit']==la['source_commit']==spec['source']and not o['passed']and o['status']=='complete_bound_failure'and o['native_calls']==276 and o['ticks']==278 and o['events']==842 and o['anchor_inclusions']==470 and o['gates']['primitive_period'],'complete immutable native failure')
 need(la['exit_code']==2 and la['failure']is None and not la['owned_live_pids']and all(not Path('/proc',str(la[k])).exists()for k in ('launcher_pid','owned_group_pid')),'owned failure terminated')
 for n,v in rt['sources'].items():need(entry(PARENT/'run/source'/n)==v and __import__('hashlib').sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'original exact archive');w.check()
 need(entry(PARENT/'run/target-certificate.json.gz')==spec['certificate'],'exact retained certificate');w.save('input-binding.json',spec)

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('60s fixed diagnostic')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(C.THREADS,'1'),native_calls=0))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'own committed closure');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);w.used+=len(raw)
  binding(w)
  with gzip.open(PARENT/'run/target-certificate.json.gz','rb')as f:raw=f.read(256*1024**2+1)
  need(len(raw)<=256*1024**2,'bounded decompression');chart=json.loads(raw);del raw
  need(chart['shooting_dimension']==470 and chart['sigma']==-1 and chart['period']==18 and chart['gates']=={'anchor_identified':True,'primitive_period':True,'uniform_normalized_eigenpair_locus':False}and chart['settings']=={k:C.SETTINGS[k]for k in ('precision','newton_steps','halfwidth_exponent','radius_exponent')},'exact fixed failed full470 chart')
  report=budget(chart);w.save('diagnostic.json',report);ans.update(passed=True,summary={k:report[k]for k in ('forcing_order_max_ratio','defect_mid_max_row','defect_rad_max_row','absolute_row_max','image_max_ratio','scaled_image_max_ratio')},collatz={k:v for k,v in report['scaling'].items()if k!='weights'})
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);ans.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
