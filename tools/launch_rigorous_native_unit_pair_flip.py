#!/usr/bin/env python3
"""Exclusive owned-process-group launch of the numerical native NS scout."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair')
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def main(args):
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():raise ValueError('clean source')
 if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10')
 mem=Path('/proc/meminfo').read_text()
 if int(next(v.split()[1]for v in mem.splitlines()if v.startswith('MemAvailable:')))<4*1024**2:raise MemoryError('4GiB current headroom')
 out=BASE/'flip'/'rigorous'/f'attempt002-{args.mode}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
 def save(name,value):
  with(out/name).open('xb')as f:f.write((json.dumps(value,sort_keys=True,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
 argv=['timeout','--signal=TERM','--kill-after=5s','75s',sys.executable,str(ROOT/'tools/certify_native_unit_pair_flip.py'),'--mode',args.mode]
 if args.mode=='target':argv+=['--profile-outer',args.profile_outer,'--profile-worker',args.profile_worker]
 meta={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'mode':args.mode,'argv':argv,'launcher_pid':os.getpid(),'start_unix':time.time(),'meminfo':mem,'load':os.getloadavg(),'worker_seconds':60,'external_seconds':75,'kill_grace_seconds':5}
 save('launcher-start.json',meta);(out/'launcher.py').write_bytes(Path(__file__).read_bytes())
 with(out/'process.log').open('xb')as log:
  try:proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=82)
  except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
  finally:
   if proc is not None:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
 live=[]
 if proc is not None:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    row=(p/'stat').read_text().rsplit(')',1)[1].split()
    if int(row[2])==proc.pid and row[0]!='Z':live.append(int(p.name))
   except(OSError,ValueError):pass
 if live or sum(p.stat().st_size for p in out.rglob('*')if p.is_file())>33*1024**2:code=2
 meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
 save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps({'root':str(out),'exit_code':code,'manifest':entry(out/'manifest.json')}),flush=True);return code
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
