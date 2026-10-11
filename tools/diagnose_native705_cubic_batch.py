#!/usr/bin/env python3
"""One immutable saved-only failed new-center cubic705 diagnosis."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native705_correlated_inverse as C
import gc
from neural_shrimp.cubic_batch_diagnostics import diagnose
BASE=C.BASE.parent/'cubic-continuation001';PARENT=BASE/'batch001';OUT=BASE/'attempt001-diagnosis/run'
PIN='a3675ba78c61b9c5b24b500ca049aa4bb70b3ba813db5e44e5729377c2356b49';SOURCE='0f5955721f5219c9ca9b2f672f490675c61f9c2f'
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/cubic_batch_diagnostics.py','tools/diagnose_native705_cubic_batch.py','tools/launch_native705_cubic_batch_diagnosis.py','tests/test_cubic_batch_diagnostics.py','docs/plans/native705-cubic-batch-diagnosis.md','docs/reviews/native705-cubic-batch001-failure.json']))

def main():
 C.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');C.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 def stop(*_):raise TimeoutError('90s saved margin cap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 def check():
  C.require(time.monotonic()-start<90,'90s saved margin cap');C.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'4GiB saved margin RSS')
 def save(name,row):
  raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();C.require(used[0]+len(raw)<14*1024**2,'16MiB raw cap')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:C.entry(ROOT/n)for n in SOURCES},'native_calls':0,'python':sys.version,'numpy':C.np.__version__,'python_flint':C.flint.__version__,'flint_library':C.flint.__FLINT_VERSION__,'threads':{k:os.environ[k]for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'worker_seconds':90,'rss_kib':4*1024**2,'hard_as_bytes':8*1024**3})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  C.inventory(PARENT,PIN);done=C.read(PARENT/'outcome.json');receipt=C.read(ROOT/'docs/reviews/native705-cubic-batch001-failure.json')
  C.require(not done['passed']and done['connected_count']==0 and not done['owned_live_pids']and done['source']==SOURCE and receipt['outcome']==done and receipt['outer']==C.entry(PARENT/'manifest.json'),'literal complete failed four-worker batch');answers=[]
  for index,row in enumerate(receipt['workers'],1):
   root=PARENT/f'chart{index:03d}/run';C.require(str(root)==row['path'],'exact four indexed raw roots');C.inventory(root,row['manifest']['sha256']);C.W.C.archive_bound(root.parent,SOURCE);out=C.read(root/'outcome.json');C.require(out==row['outcome']and not out['passed']and out['ticks']==331 and out['status']=='completed_bounds_failed','complete native failed worker')
   C.require(C.entry(root/'certificate.json.gz')==row['certificate'],'exact failed coefficient source');proof=C.read(root/'certificate.json.gz');check();files=sorted((root/'events').glob('*-cubic_proposal_return.json.gz'));C.require(len(files)==3,'three retained center corrections');answer=diagnose(proof,[C.read(p)for p in files]);save(f'chart-{index}.json',answer);answers.append({'unit':index,'physical_center':answer['physical_center'],'gauge_alignment':answer['gauge_alignment_cosine'],'final_correction_radii':answer['newton_steps'][-1]['weighted_correction_in_radii'],'old_contraction':answer['old_contraction'],'old_max_image_ratio':answer['old_max_image_ratio']});del proof;gc.collect();check()
  save('binding.json',{'root':str(PARENT),'outer':PIN,'source':SOURCE,'receipt':C.entry(ROOT/'docs/reviews/native705-cubic-batch001-failure.json')});result.update(passed=True,dimension=705,charts=answers)

 except BaseException as exc:result['failure']=type(exc).__name__+': '+str(exc)
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 if result['seconds']>=90 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
 save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):C.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
 for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 OUT.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
