import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.utility_paired_geometry import paired
from neural_shrimp.utility_dose_audit import inventory,STATE_HASHES
from neural_shrimp.period2_branch_io import load_record,identity,file_entry
RAW=Path('/data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001');PIN='4589dc9afeaa098537341adf27ae230a64110a8fdaf048baf7f8fef22c6d19be'
FILES=['src/neural_shrimp/utility_paired_geometry.py','src/neural_shrimp/utility_dose_audit.py','src/neural_shrimp/period2_branch_io.py','tools/explore_utility_paired_feasibility.py','tests/test_utility_paired_geometry.py','docs/plans/utility-paired-feasibility.md']
def main(out):
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip();assert os.getpriority(os.PRIO_PROCESS,0)>=10
 out.mkdir(exist_ok=False,parents=True);start=time.monotonic();result={'passed':False,'forks':[],'model_calls':0,'pulse_applications':0}
 def save(n,v):
  def default(x):
   if isinstance(x,np.ndarray):return x.tolist()
   raise TypeError(type(x).__name__)
  b=json.dumps(v,default=default,sort_keys=True,indent=2,allow_nan=False).encode();assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<2*1024**2
  with (out/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 save('runtime.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_files={n:file_entry(ROOT/n) for n in FILES},pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  t=time.monotonic()
  for _ in range(1000):assert paired(np.arange(1.,14.),np.array([10.,1.]))['qualified']
  elapsed=time.monotonic()-t;projection=4*10*elapsed/1000+5;save('synthetic-profile.json',dict(iterations=1000,seconds=elapsed,projected_seconds=projection));assert projection<30
  inventory(RAW,PIN,1024**2);assert all(not p.stat().st_mode&0o222 for p in RAW.iterdir())
  for i in range(4):
   kind,p=load_record(RAW,f'{2+2*i:05d}-independent_gradient_return.json');assert kind=='independent_gradient_return' and p['index']==i and identity(p['state'])['sha256']==STATE_HASHES[i]
   row=paired(p['gradient'],p['rates']);row.update(index=i,original_state_sha256=STATE_HASHES[i],gradient_identity=identity(p['gradient']),rates=p['rates']);save(f'fork-{i}.json',row);result['forks'].append({k:v for k,v in row.items() if not isinstance(v,np.ndarray)})
   assert time.monotonic()-start<25 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024
  result.update(passed=all(r['qualified'] for r in result['forks']),input_manifest=PIN,scope='All saved forks; algebra only. No behavioral or geometry utility conclusion; distant controls remain unqualified.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):file_entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
  out.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
