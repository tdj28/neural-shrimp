"""Attraction of the independently localized full native156th endpoint."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews'),str(ROOT/'tools')]
import review_s2_near_fold_endpoint as B
import certify_s2_period18_near_fold_endpoint as P
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_periodic_lyapunov import periodic_witness
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_orbits import canonical_bytes,matrix_json,vector_json
from flint import ctx
O=B.O;need=B.M.need;PRIOR=B.BASE/'attempt-001';PIN='220b77b44a053a598bdbdbc264458f33a77887d27517256e60ab3c65f6789cb9'
FILES=list(dict.fromkeys(B.FILES+P.SOURCES+['src/neural_shrimp/rigorous_periodic_lyapunov.py','tests/test_rigorous_periodic_lyapunov.py','tools/certify_s2_near_fold_endpoint_attraction.py','docs/plans/s2-near-fold-endpoint-attraction.md']))
def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'completed':False,'passed':False};model=None;resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
 def save(n,v):
  b=canonical_bytes(v);b=gzip.compress(b,mtime=0) if n.endswith('.gz') else b;need(len(b)+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<32*1024**2,'32MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<40 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'40s/1.5GiBcap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,40)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:O.entry(ROOT/n) for n in FILES},'threads':dict.fromkeys(O.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'parent_independent_manifest':PIN})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  O.verify(PRIOR,PIN);review=O.read(PRIOR/'run/outcome.json');known=O.read(PRIOR/'run/reduction-256.json');need(review['passed'] and known['passed'] and known['exact_gauge']=='45/128','independent known primitive endpoint')
  O.verify(B.RAW,B.OUTER);O.verify(B.RAW/'run',B.WORKER);inp=O.read(B.RAW/'run/inputs.json')
  for name,digest in O.read(B.RAW/'run/runtime-start.json')['source_sha256'].items():need(O.entry(ROOT/name)['sha256']==digest,'unchanged source '+name)
  need(len(known['final_state_box'])==235,'full native known state/rate');save('inputs.json',{'model_inputs':inp['model_inputs'],'root_box':known['final_state_box'],'exact_gauge':'45/128','independent_parent':PIN})
  with ctx.workprec(256):
   model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=18)
   box=[restore_ball(v) for v in known['final_state_box']];need(box[-1]>0,'positive enclosed actual hidden rate');blocks=[]
   for j in range(18):
    tick();state=box[j*13:(j+1)*13];image,jac,partial=model.step(state,box[-1]);save(f'phase-{j:02d}.json',{'phase':j,'state':vector_json(state),'rate':vector_json([box[-1]]),'image':vector_json(image),'jacobian':matrix_json(jac),'partial':matrix_json(partial)});blocks.append(jac)
   proof=periodic_witness(blocks,precision=256,tick=tick,observe=lambda k,v:save(k+'.json.gz',v));save('proof.json.gz',proof);tick()
  result.update(completed=True,passed=proof['passed'],point_passed=proof['point_passed'],interval_phases_passed=sum(v.get('interval_D_positive',{}).get('passed',False) for v in proof['phases']),exact_gauge='45/128',hidden_rate=known['final_state_box'][-1],scope='Local attraction of the inherited primitive18 full-state endpoint at its enclosed actual hidden rate and fixed exact output rate, if all cyclic interval metric gates pass. No neighboring interval or full-branch attraction inferred.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,counts=None if model is None else model.counts());save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['completed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
