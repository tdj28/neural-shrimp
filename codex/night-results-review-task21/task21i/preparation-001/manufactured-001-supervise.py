"""Sole authorized manufactured-only Task21i controls; bounded local supervision."""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,subprocess,time,shutil

def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def ident(pid):
 p=Path('/proc')/str(pid)
 try:
  st=(p/'stat').read_text();q=st.rsplit(')',1)[1].split()
  return {'pid':pid,'starttime':int(q[19]),'pgrp':int(q[2]),'session':int(q[3]),'uid':p.stat().st_uid,'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
 except FileNotFoundError:return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);args=ap.parse_args();here=Path(__file__).resolve().parent
 plan=json.loads((here/'manufactured-001-plan.json').read_text());assert plan['status']=='reviewed_manufactured_only';repo=Path(plan['repo']);out=Path(plan['output']);sup=Path(plan['supervision']);assert not out.exists()and not sup.exists()
 assert subprocess.check_output(['git','status','--porcelain'],cwd=repo)==b''
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()==args.source
 subprocess.run(['git','merge-base','--is-ancestor',args.source,'origin/main'],cwd=repo,check=True)
 available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 assert available>=20*1024**3 and shutil.disk_usage('/tmp').free>=16*1024**3
 relbase=here.relative_to(repo);blobs={}
 for n in ['statistics.py','run.py','controls.py','manufactured.py','manufactured-001-plan.json',Path(__file__).name]:
  b=(here/n).read_bytes();assert subprocess.check_output(['git','show',args.source+':'+str(relbase/n)],cwd=repo)==b
  blobs[n]=pin(here/n)
 for n,sha in plan['source_hashes'].items():assert blobs[n]['sha256']==sha
 sup.mkdir();cap=sup/'source';cap.mkdir()
 for n in blobs:shutil.copyfile(here/n,cap/n)
 save(sup/'admission.json',{'source_commit':args.source,'source_blobs':blobs,'available_bytes':available,'supervisor':ident(os.getpid()),'plan':plan})
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLBACKEND='Agg')
 def limits():
  os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(55,55));resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 cmd=['/data2/neural-shrimp/.venv/bin/python','-I',str(cap/'manufactured.py'),'--out',str(out)];started=time.monotonic();failure=None;maximum=0
 with (sup/'stdout.txt').open('xb')as so,(sup/'stderr.txt').open('xb')as se:
  worker=subprocess.Popen(cmd,stdout=so,stderr=se,env=env,start_new_session=True,preexec_fn=limits);identity=ident(worker.pid);assert identity and identity['pgrp']==worker.pid and identity['session']==worker.pid
  save(sup/'worker.json',identity)
  while worker.poll()is None:
   elapsed=time.monotonic()-started;size=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())if out.exists()else 0
   size+=(sup/'stdout.txt').stat().st_size+(sup/'stderr.txt').stat().st_size;maximum=max(maximum,size)
   avail=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
   if elapsed>60 or size>128*1024**2 or avail<16*1024**3:
    failure='deadline'if elapsed>60 else'output_cap'if size>128*1024**2 else'available_memory_floor'
    assert ident(worker.pid)['starttime']==identity['starttime'];os.killpg(worker.pid,signal.SIGTERM)
    try:worker.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=5)
    break
   time.sleep(.05)
  code=worker.wait()
 elapsed=time.monotonic()-started
 for n in blobs:assert pin(cap/n)['sha256']==blobs[n]['sha256']and pin(here/n)['sha256']==blobs[n]['sha256']
 outputs=[pin(p)for p in sorted(out.rglob('*'))if p.is_file()]if out.exists()else[]
 # The direct worker is reaped. Its fresh process group is checked without acting on any other owner.
 try:os.killpg(worker.pid,0);group_absent=False
 except ProcessLookupError:group_absent=True
 receipt={'returncode':code,'failure':failure,'external_seconds':elapsed,'maximum_sampled_output_bytes':maximum,'worker':identity,'worker_after_reap':ident(worker.pid),'owned_group_absent':group_absent,'outputs':outputs,'source_commit':args.source,'manufactured_only':True,'scientific_claims_accepted':False,'command':cmd,'output_cap_scope':'Only manufactured output directory plus stdout/stderr; source and receipts excluded.'}
 save(sup/'receipt.json',receipt);print(json.dumps(receipt))
 assert code==0 and failure is None and group_absent
if __name__=='__main__':main()
