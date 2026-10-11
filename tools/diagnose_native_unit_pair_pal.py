#!/usr/bin/env python3
"""Fixed immutable saved-domain conditioning diagnostic; zero native calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_pal as C
from neural_shrimp.unit_pair_pal_conditioning import diagnose
BASE=C.BASE;PARENT=BASE/'attempt002-target';OUT=BASE/'attempt002-diagnosis/run'
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/unit_pair_pal_conditioning.py','tools/diagnose_native_unit_pair_pal.py','tools/launch_native_unit_pair_pal_diagnosis.py','tests/test_unit_pair_pal_conditioning.py','docs/plans/native-unit-pair-pal-diagnosis.md']))
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
  C.inventory(PARENT,'9fe4a84dddb1e1d2f2782d357795c5d45ae2ec61258996772c6498378ca73bba');C.inventory(PARENT/'run','c1d6754f456b10dd22450737d0525e910125f18d8a1ade390d3bc98bf5b0eb21')
  rt=C.read(PARENT/'run/runtime.json');la=C.read(PARENT/'launcher-result.json');out=C.read(PARENT/'run/outcome.json')
  C.require(rt['commit']==la['commit']=='9a395b70ff469f0cd690b5b098153643c6413346'and la['exit_code']==2 and not la['owned_live_pids']and la['stop_reason']is None,'complete immutable failed source')
  C.require(out['status']=='completed_bounds_failed'and out['ticks']==253,'complete fixed failed domain')
  C.archive_bound(PARENT,rt['commit'])
  C.require(rt['settings']==C.SETTINGS and rt['threads']==dict.fromkeys(C.THREADS,'1'),'exact failed settings')
  C.require(out['events']=={'direction_return':144,'pal_chart_return':1,'pal_newton_proposal':2,'pal_path_assembly_return':3,'pal_path_phase_return':108,'pal_preconditioner_return':1},'complete253 acquired calls')
  with gzip.open(PARENT/'run/certificate.json.gz','rt')as f:chart=json.load(f)
  C.require(chart['parameter_center']=='63/4096'and chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-6,'radius_exponent':-10},'exact failed domain')
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
