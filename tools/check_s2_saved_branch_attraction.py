"""Fixed saved-Jacobian periodic-metric pilot on12 audited native charts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gc,gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_s2_small_march as B
from neural_shrimp.rigorous_periodic_lyapunov import periodic_witness
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import canonical_bytes
from flint import ctx
O=B.O;need=B.M.need
PRIOR=Path('/data2/neural-shrimp-data/audits/s2-period18-small-march/attempt-001');PIN='d6888862e0cc416e7aa7beb5301b76ec8188735d0cf2de09a1870a6e6009cd16'
FILES=list(dict.fromkeys(B.FILES+['src/neural_shrimp/rigorous_periodic_lyapunov.py','tests/test_rigorous_periodic_lyapunov.py','tools/check_s2_saved_branch_attraction.py','docs/plans/s2-saved-branch-attraction.md']))
def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'completed':False,'charts':[]};resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
 def save(name,v):
  b=canonical_bytes(v);b=gzip.compress(b,mtime=0) if name.endswith('.gz') else b;need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<128*1024**2,'raw128MiB');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'60s/1.5GiBcap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,60)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:O.entry(ROOT/n) for n in FILES},'threads':dict.fromkeys(O.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'native_model_calls':0})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  O.verify(PRIOR,PIN);need(O.read(PRIOR/'run/outcome.json')['passed'],'independent exact native branch review')
  roots=[B.bind(spec)[0] for spec in B.SPECS];tick()
  def process(c,label):
   need(c['passed'] and all(c['gates'][k] for k in ('uniform_root','primitive_period','anchor_identified')),'proved primitive native chart')
   need(c['statement']['period']==18 and c['statement']['phase_dimension']==13 and len(c['phase_jacobians'])==18,'complete native18 by13 derivatives')
   with ctx.workprec(256):
    blocks=[restore_matrix(v) for v in c['phase_jacobians']];need(all(m.nrows()==m.ncols()==13 for m in blocks),'physical derivatives only')
    proof=periodic_witness(blocks,precision=256,tick=tick,observe=lambda k,v:save(label+'-'+k+'.json.gz',v));save(label+'-proof.json.gz',proof)
   result['charts'].append({'label':label,'endpoints':c['exact_parameter_endpoints'],'whole_interval_attracting':proof['passed'],'point_metric_passed':proof['point_passed'],'interval_phases_passed':sum(v.get('interval_D_positive',{}).get('passed',False) for v in proof['phases'])});save(label+'-result.json',result['charts'][-1]);tick()
  c=O.read(roots[0]/'certificate.json.gz')['charts'][0];process(c,'rechart-first');del c;gc.collect()
  c=O.read(roots[1]/'certificate.json.gz');process(c,'saved-repair');del c;gc.collect()
  for j in range(10):
   c=O.read(roots[2]/f'chart-{j:03d}.json.gz');process(c,f'march-{j}');del c;gc.collect()
  result.update(completed=True,all_intervals_attracting=all(v['whole_interval_attracting'] for v in result['charts']),scope='Whole supplied native-J boxes, including interval phase correlations lost conservatively. Parent audit supplies existence, primitive period and joins. A failed metric is inconclusive, not instability. No earlier branch gaps or complete shrimp organization supplied.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['completed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
