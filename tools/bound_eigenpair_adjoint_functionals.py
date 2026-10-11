#!/usr/bin/env python3
"""Fixed scalar residual functionals of saved root-conditioned adjoints."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import qualify_eigenpair_shooting_fold as P
from neural_shrimp.rigorous_adjoint_functional import functional
from neural_shrimp.rigorous_orbits import ball_json
from flint import arb
from neural_shrimp.rigorous_shooting_adjoint import need
RAW=P.BASE/'attempt-001-shooting-fold-target'
OUTER='35661bb57aa3afd53d3cec77f3429a10cff166e4387337a00e74915c5ecbf7ee'
RUN='a5c578cf6f437d662654bf1a159b8a903d1104e8f7f6a5eec94abdba34a7fd3a'
EXEC='501f356e65d6dfb5e88d748c8c74c70f304c3f16'
FILES=list(dict.fromkeys(P.FILES+['src/neural_shrimp/rigorous_adjoint_refinement.py','tools/refine_eigenpair_shooting_adjoint.py','tools/launch_eigenpair_adjoint_refinement.py','tests/test_rigorous_adjoint_refinement.py','docs/plans/eigenpair-adjoint-refinement.md','docs/reviews/eigenpair-shooting-fold-outcome.json']))

REF=P.BASE/'attempt-001-adjoint-refinement'
REF_OUTER='d16ad5278f7bbbfafa7b6e890ae7e2a675ee2fef0fca53b93d6fdc824ea11c3a'
REF_RUN='919baf17f247261fa32aa17bd30e5301956f54132a98a7e6d1206f310045ffc2'
REF_EXEC='85aa258fd0029be811c65b4915abf9b14bf227b8'
FILES=list(dict.fromkeys(FILES+['src/neural_shrimp/rigorous_adjoint_functional.py','tools/bound_eigenpair_adjoint_functionals.py','tools/launch_eigenpair_adjoint_functionals.py','tests/test_rigorous_adjoint_functional.py','docs/plans/eigenpair-adjoint-functionals.md','docs/reviews/eigenpair-adjoint-refinement-outcome.json']))


def matrix(rows):return arb_mat([[P.restore(v) for v in row] for row in rows])

def bind():
 binds={'outer':P.inventory(RAW,OUTER),'run':P.inventory(RAW/'run',RUN)}
 rt=P.read(RAW/'run/runtime.json');la=P.read(RAW/'launcher-result.json');out=P.read(RAW/'run/outcome.json');res=P.read(RAW/'run/result.json')
 need(rt['commit']==la['commit']==EXEC and rt['mode']==la['mode']=='target' and la['exit_code']==2 and la['stop_reason'] is None and not la['owned_live_pids'],'fixed prior gate failure')
 need(out['status']=='conditional_fold_gate_unresolved' and out['native_calls']==0 and P.finite_bound(out['elapsed_seconds'],60) and P.finite_bound(out['peak_rss_kib'],1536*1024),'prior finite resources')
 need(res['border_solve']['passed'] and res['gates']['simple_native_plus_one'] and not res['gates']['nonzero_quadratic'] and not res['gates']['nonzero_rate_transversality'],'fixed partial qualification')
 for n,v in rt['sources'].items():
  need(P.entry(ROOT/n)==P.entry(RAW/'run/source'/n)==v,'source parity')
  need(P.hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'source Git')
 binding=P.read(RAW/'run/input-binding.json');need(binding['execution']==P.PARENT_EXEC and binding['parameter_endpoints']==['147/128','151/128'] and binding['outer']['manifest']['sha256']==P.PARENT_OUTER,'same authenticated locus premise')
 for k,v in {'precision':256,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(P.THREADS,'1')}.items():need(rt[k]==v,'unchanged environment '+k)
 solution=[P.restore(v) for v in res['border_solve']['solution']]
 defect=matrix(P.read(RAW/'run/events/003-defect_return.json.gz')['matrix'])
 sr=P.read(RAW/'run/events/005-solve_residual_return.json.gz');proposal=matrix(sr['proposal']);forcing=matrix(sr['forcing'])
 ci=P.read(RAW/'run/events/000-coefficient_inputs.json.gz');rates=matrix(ci['rates']);second=[P.restore(v) for v in ci['second_contractions']]
 binds.update(execution=EXEC,parent=binding,prior_contractions=res['border_solve']['contractions'],same_source_git_verified=True)
 return solution,proposal,forcing,defect,rates,second,binds


def bind_refinement():
    binding={'outer':P.inventory(REF,REF_OUTER),'run':P.inventory(REF/'run',REF_RUN)}
    rt=P.read(REF/'run/runtime.json');la=P.read(REF/'launcher-result.json');out=P.read(REF/'run/outcome.json');r=P.read(REF/'run/result.json')
    need(rt['commit']==la['commit']==REF_EXEC and rt['mode']==la['mode']=='target' and la['exit_code']==2 and la['stop_reason'] is None and not la['owned_live_pids'],'fixed refinement execution')
    need(out['status']=='coefficient_exclusions_unresolved' and out['native_calls']==out['new_inverses']==0 and P.finite_bound(out['elapsed_seconds'],30) and P.finite_bound(out['peak_rss_kib'],1024**2),'refinement finite scope')
    need(rt['intersections']==5 and rt['precision']==256,'fixed refinement settings')
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(P.THREADS,'1')}.items():need(rt[k]==v,'refinement environment '+k)
    for n,v in rt['sources'].items():
        need(P.entry(ROOT/n)==P.entry(REF/'run/source'/n)==v,'refinement source parity')
        need(P.hashlib.sha256(subprocess.check_output(['git','show',REF_EXEC+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'refinement source Git')
    old=P.read(REF/'run/bindings.json');need(old['execution']==EXEC and old['outer']['manifest']['sha256']==OUTER and old['run']['manifest']['sha256']==RUN,'original adjoint dependency')
    y=[P.restore(v) for v in r['left_shooting']]+[arb(0)]
    return y,binding


def main(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];total=[0];serial=[0];cap=8*1024**2;reserve=2*1024**2
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
 def save(name,obj,final=False):
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  # Largest item is one235D saved matrix, below bounded input size.
  data=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  if name.endswith('.gz'):data=gzip.compress(data,mtime=0)
  need(total[0]+len(data)<(cap if final else cap-reserve),'aggregate output cap')
  with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  total[0]+=len(data)
 def check():
  if stop[0] or time.monotonic()-start>=20:raise TimeoutError('20s saved-only cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('1GiB monitoredRSS')
 def observe(kind,value):save(f'events/{serial[0]:03d}-{kind}.json.gz',value);serial[0]+=1;check()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();sources={n:P.entry(ROOT/n) for n in FILES}
 save('runtime.json',{'commit':commit,'pid':os.getpid(),'mode':'target','sources':sources,'precision':256,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(P.THREADS,'1'),'native_calls':0,'new_inverses':0,'functionals':3,'seconds_cap':20,'rss_cap_kib':1024**2,'raw_cap_bytes':cap})
 for n in FILES:
  p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
 result={'passed':False,'status':'incomplete','native_calls':0,'new_inverses':0}
 try:
  with ctx.workprec(256):
   s,y0,f,d,rr,ss,bindings=bind();y,refbinding=bind_refinement();bindings['refinement']=refbinding;save('bindings.json',bindings);check()
   border=matrix(P.read(RAW/'run/events/001-border.json.gz')['matrix']);pre=matrix(P.read(RAW/'run/events/002-preconditioner_return.json.gz')['matrix'])
   labels=['actual_hidden_rate','actual_output_rate','second_derivative_contraction'];vectors=[[rr[i,j] for i in range(rr.nrows())]+[arb(0)] for j in range(2)]+[ss+[arb(0)]]
   rows=[];values=[]
   for label,b in zip(labels,vectors):
    observe('functional_start',{'label':label});row,value=functional(border,pre,y,b,observe);row['label']=label;rows.append(row);values.append(value)
   gates={'nonzero_rate_transversality':any(not v.contains(0) for v in values[:2]),'nonzero_quadratic':not values[2].contains(0)}
   r={'passed':all(gates.values()),'gates':gates,'functionals':rows,'actual_rate_coefficients':[ball_json(v) for v in values[:2]],'second_derivative_contraction':ball_json(values[2]),'quadratic_coefficient':ball_json(values[2]/2),'scope':'Conditional scalar residual bounds on same saved root family; inherits parent and simple+1 proof, no new derivative/inverse or other-mode exclusion/attraction/geometry.'};save('result.json',r);check()
   result.update(passed=r['passed'],status='conditional_fold_coefficients_passed' if r['passed'] else 'coefficient_exclusions_unresolved',gates=r['gates'],scope=r['scope'])
 except BaseException as e:result.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(e).__name__,'message':str(e)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,events=serial[0])
 if not P.finite_bound(result['elapsed_seconds'],20) or not P.finite_bound(result['peak_rss_kib'],1024**2):result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):P.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.out))
