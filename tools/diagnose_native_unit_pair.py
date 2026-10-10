#!/usr/bin/env python3
"""Fixed immutable saved-domain conditioning diagnostic; zero native calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair as C
from neural_shrimp.unit_pair_conditioning import diagnose
BASE=C.BASE;PARENT=BASE/'attempt001-target';OUT=BASE/'attempt001-diagnosis/run'
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/unit_pair_conditioning.py','tools/diagnose_native_unit_pair.py','tools/launch_native_unit_pair_diagnosis.py','tests/test_unit_pair_conditioning.py','docs/plans/native-unit-pair-diagnosis.md']))
def main():
 C.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');C.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 def stop(*_):raise TimeoutError('60s saved diagnostic')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def save(name,row):
  raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();C.require(used[0]+len(raw)<16*1024**2,'16MiB diagnostic cap')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:C.entry(ROOT/n)for n in SOURCES},'threads':dict.fromkeys(C.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'model_calls':0})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  C.inventory(PARENT,'232e3c9f1cd1aa020b4e600b90e7ee5ddc044ba4caa80776275b732cee24fd6a');C.inventory(PARENT/'run','24021ba19143cc3a952e5f03c980f32bb008e0f9dfdf779f35f0ccf67bcbe318')
  rt=C.read(PARENT/'run/runtime.json');la=C.read(PARENT/'launcher-result.json');out=C.read(PARENT/'run/outcome.json')
  C.require(rt['commit']==la['commit']=='a223df018d0263644ca626f476a7da2d68aacd17'and la['exit_code']==2 and not la['owned_live_pids']and la['stop_reason']is None,'complete immutable failed source')
  C.require(out['status']=='completed_bounds_failed'and out['ticks']==217,'complete fixed failed domain')
  with gzip.open(PARENT/'run/certificate.json.gz','rt')as f:chart=json.load(f)
  answer=diagnose(chart);save('diagnostic.json',answer);result.update(passed=True,scaling={k:v for k,v in answer['scaling'].items()if k!='weights'},groups=answer['groups'],resolvent_available='resolvent_proposal'in answer)
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if result['seconds']>=60 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):C.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
