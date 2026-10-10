#!/usr/bin/env python3
"""Saved numerical scout conditioning/gauge diagnostics, no native acquisition."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native_unit_pair as S
need=S.need;entry=S.entry;read=S.read
PARENT=S.BASE/'attempt002-target';OUT=S.BASE/'attempt001-diagnosis/run'
PINS={'outer':'4941c57ff6a37a7574f7300b17b5980aafad35bd27e6091ba0f99bdf724403f0','worker':'dde339494bc7b5ff77eda74cfde8d38cc9a250db24d15ffe57187a327582dddd','execution':'3c456e977f3b9ea3a34e05efcee3b8e141c76fad'}
SOURCES=list(dict.fromkeys(S.SOURCES+['tools/diagnose_native_unit_pair_scout.py','tools/launch_native_unit_pair_scout_diagnosis.py','tests/test_native_unit_pair_scout_diagnosis.py','docs/plans/native-unit-pair-scout-diagnosis.md']))

def diagnostic(z,t,J,ht):
 z=np.asarray(z);J=np.asarray(J);ht=np.asarray(ht);n=(len(z)-2)//3
 need(J.shape==(len(z),len(z))and ht.shape==(len(z),)and np.isfinite(J).all(),'full finite saved Jacobian')
 q=z[n+2:2*n+2];r=z[2*n+2:];sv=np.linalg.svd(J,compute_uv=False);tan=np.linalg.solve(J,-ht)
 return {'q_norm_squared':float(q@q),'r_norm_squared':float(r@r),'q_dot_r':float(q@r),
         'ellipticity_gauge_factor':float(q@q-t*(r@r)),'jacobian_smallest_singular_value':float(sv[-1]),
         'jacobian_condition_estimate':float(sv[0]/sv[-1]),'tangent':tan.tolist(),'actual_rate_tangent':tan[n:n+2].tolist(),
         'solve_relative_residual':float(np.linalg.norm(J@tan+ht)/max(1.,np.linalg.norm(J)*np.linalg.norm(tan),np.linalg.norm(ht))),
         'scope':'Floating saved-array diagnostics only. Zero or sign changes are proposals, not certified regularity, singularities or event brackets.'}

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 def stop(*_):raise TimeoutError('30s saved diagnosis cap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30);used=0
 def save(name,row):
  nonlocal used
  b=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(used+len(b)<8*1024**2,'8MiB cap');used+=len(b)
  with(OUT/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in SOURCES},'pid':os.getpid(),'native_calls':0})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  S.inventory(PARENT,PINS['outer']);S.inventory(PARENT/'run',PINS['worker']);rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');survey=read(PARENT/'run/survey.json')
  need(rt['commit']==la['commit']and rt['commit']==PINS['execution']and la['exit_code']==0 and la['owned_live_pids']==[],'exact completed scout')
  need(out['passed']and out['status']=='corrector_limit'and out['accepted_points']==len(survey['points'])==20 and out['evaluations']==84 and out['native_calls']==3024,'complete exact scout outcome')
  for n,row in rt['sources'].items():need(entry(PARENT/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical scout source')
  returns=[read(p)for p in sorted((PARENT/'run/events').glob('*-evaluation_return.json'))];need(len(returns)==84,'all completed numerical returns')
  rows=[]
  for point in survey['points']:
   matches=[r for r in returns if r['z']==point['z']and r['t']==point['t']];need(len(matches)==1,'unique exact accepted full evaluation')
   r=matches[0];row={'index':point['index'],'t':point['t'],'rates':point['diagnostics']['rates'],'complement_max_modulus':point['diagnostics']['complement_max_modulus'],**diagnostic(r['z'],r['t'],r['jacobian'],r['parameter_derivative'])};rows.append(row)
  save('accepted-diagnostics.json',rows)
  attempted=[]
  for r in returns:
   if r['t']>rows[-1]['t']:attempted.append({'evaluation':r['evaluation'],'t':r['t'],'residual_inf':float(max(abs(np.array(r['value'])))),**diagnostic(r['z'],r['t'],r['jacobian'],r['parameter_derivative'])})
  save('failed-attempt-diagnostics.json',attempted)
  turns=[{'left':a['index'],'right':b['index'],'t':[a['t'],b['t']]}for a,b in zip(rows,rows[1:])if a['actual_rate_tangent'][0]*b['actual_rate_tangent'][0]<0]
  result.update(passed=True,accepted=20,failed_evaluations=len(attempted),hidden_rate_turn_proposals=turns,last=rows[-1],scope='Saved floating gauge/parameterization diagnostics, no new trajectory, proof, contact or branch identity.')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if result['seconds']>=30 or result['peak_rss_kib']>=1024**2:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
