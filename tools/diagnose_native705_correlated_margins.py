#!/usr/bin/env python3
"""One immutable saved-only correlated705 margin decomposition."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native705_correlated_inverse as C
from neural_shrimp.correlated_pal_margins import margins
BASE=C.BASE;PARENT=C.TARGET.parent;OUT=BASE/'attempt001-margins/run'
PIN='28393e38f8d3fadfa1355f759439d574639cb4255ca20e1d6971085caeb5971c';WORKER='da963cfe0011d6d5325a13ad055333511d47b9f9083f2909a02dd552089afbdb';SOURCE='99b6f71e61b85cc8e201fa0b0f198a15ebc3a5c2'
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/correlated_pal_margins.py','tools/diagnose_native705_correlated_margins.py','tools/launch_native705_correlated_margins.py','tests/test_correlated_pal_margins.py','docs/plans/native705-correlated-margins.md']))
def main():
 C.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');C.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
 def stop(*_):raise TimeoutError('90s saved margin cap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 def check():
  C.require(time.monotonic()-start<90,'90s saved margin cap');C.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'6GiB saved margin RSS')
 def save(name,row):
  raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();C.require(used[0]+len(raw)<16*1024**2,'16MiB raw cap')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:C.entry(ROOT/n)for n in SOURCES},'native_calls':0,'worker_seconds':90,'rss_kib':6*1024**2,'hard_as_bytes':10*1024**3})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  C.inventory(PARENT,PIN);C.inventory(PARENT/'run',WORKER);C.W.C.archive_bound(PARENT,SOURCE);la=C.read(PARENT/'launcher-result.json');done=C.read(PARENT/'run/outcome.json')
  C.require(la['exit_code']==0 and not la['owned_live_pids']and done['passed']and done['connected']and done['strict_inclusions']==done['anchor_inclusions']==705 and done['ticks']==148,'complete actual connected wider source');check()
  cert=PARENT/'run/certificate.json.gz';C.require(C.entry(cert)=={'bytes':184464912,'sha256':'c28c7a95867b1aa1df3dcfc76867e2aff0d36e1fec08225d4294f1a9799dd06a'},'fixed actual dense certificate')
  proof=C.read(cert);check();answer=margins(proof,check);save('margins.json',answer);save('binding.json',{'root':str(PARENT),'outer':PIN,'worker':WORKER,'source':SOURCE,'certificate':C.entry(cert)});result.update(passed=True,dimension=705)
 except BaseException as exc:result['failure']=type(exc).__name__+': '+str(exc)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 if result['seconds']>=90 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):C.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
 for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 OUT.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
