"""Bounded Newton and full-monodromy classification of one saved edge candidate."""
import argparse,importlib.util,json,sys,time,resource
from pathlib import Path
import numpy as np

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();c=json.loads(Path(args.card).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);q=load('saddle_map',c['map_source']);num=load('saddle_numerics',c['numerics_source'])
 for e in c['inputs']:assert q.sha256(e['path'])==e['sha256']
 model=q.ManualQwenSoftMap.from_pinned_files();alpha,lt=c['alpha'],c['log10T'];cache=model.prefix_cache(alpha);p=c['period'];assert type(p) is int and 1<=p<=128
 x=np.load(c['candidate'],allow_pickle=False);assert x.shape==(896,) and np.isfinite(x).all();start=time.monotonic();counts={'step':0,'jvp':0};events=(out/'events.jsonl').open('x')
 def record():events.write(json.dumps(dict(elapsed=time.monotonic()-start,counts=counts.copy()))+'\n');events.flush()
 def step(x):
  counts['step']+=1;assert counts['step']<=c['max_steps']
  with torch.no_grad():return model(torch.from_numpy(x),alpha,lt,cache).numpy()
 def jvp(x,v):
  counts['jvp']+=1;assert counts['jvp']<=c['max_jvps']
  y=model.jvp(torch.from_numpy(x),torch.from_numpy(v),alpha,lt,cache=cache).detach().numpy()
  if counts['jvp']%20==0:record()
  return y
 res=num.newton_cycle(step,jvp,x,p,tol=1e-11,max_newton=8,linear_rtol=1e-9,restart=20,max_linear_iterations=2,max_backtracks=5)
 state=res.pop('state');np.save(out/'candidate-refined.npy',state);report={'newton':res,'period':p,'saddle_qualified':False}
 (out/'newton.json').write_text(json.dumps(report,indent=2)+'\n')
 if res['converged']:
  # Dense columns are formed by the own map's batched forward AD implementation.
  matrix=model.jacobian(torch.from_numpy(state),alpha,lt,period=p,chunk_size=16,cache=cache).detach().numpy();np.save(out/'monodromy.npy',matrix)
  points=num.cycle_points(step,state,p);np.save(out/'orbit.npy',points);returns=np.array([np.linalg.norm(v-state)/np.linalg.norm(state) for v in points[1:]]);divisors=[d for d in range(1,p) if p%d==0];primitive=all(returns[d-1]>1e-6 for d in divisors)
  eig=num.spectrum(matrix);vals=eig['eigenvalues'];np.savez(out/'spectrum.npz',values=vals,vectors=eig['eigenvectors'],residuals=eig['absolute_eigenpair_residuals'])
  d=np.arange(1,897,dtype=float);d/=np.linalg.norm(d);ad=model.jvp(torch.from_numpy(state),torch.from_numpy(d),alpha,lt,period=p,cache=cache).detach().numpy();den=np.linalg.norm(ad);assert den>0
  dense_error=float(np.linalg.norm(matrix@d-ad)/den);fd=[]
  with torch.no_grad():
   for h in [.01,.003,.001]:
    plus=model.period_map(torch.from_numpy(state+h*d),alpha,lt,p,cache);minus=model.period_map(torch.from_numpy(state-h*d),alpha,lt,p,cache);err=float(np.linalg.norm((plus-minus).numpy()/(2*h)-ad)/den);fd.append({'h':h,'relative_error':err})
  unstable=np.flatnonzero(np.abs(vals)>1+1e-6);neutral=np.flatnonzero(abs(np.abs(vals)-1)<=1e-6)
  checks={'closure':returns[-1]<1e-11,'primitive':primitive,'eigen_residual':float(np.max(eig['absolute_eigenpair_residuals']))<1e-8,'dense_jvp':dense_error<1e-8,'fd_convergence':fd[-1]['relative_error']<fd[0]['relative_error']/10 and fd[-1]['relative_error']<1e-3,'one_real_unstable':len(unstable)==1 and abs(vals[unstable[0]].imag)<1e-8,'no_neutral':len(neutral)==0}
  report.update(checks={k:bool(v) for k,v in checks.items()},returns=returns.tolist(),proper_divisors=divisors,dense_jvp_relative_error=dense_error,fd=fd,multipliers=[[float(v.real),float(v.imag)] for v in vals],unstable_count=len(unstable),neutral_count=len(neutral),max_eigen_residual=float(np.max(eig['absolute_eigenpair_residuals'])),saddle_qualified=all(checks.values()))
  if report['saddle_qualified']:report['predicted_escape_slope']=-p/float(np.log(abs(vals[unstable[0]])))
 for e in c['inputs']:assert q.sha256(e['path'])==e['sha256']
 report.update(completed=True,elapsed=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,counts=counts,scope='Float64 numerical orbit and complete dense spectrum, not an interval certificate. Escape slope requires a separate residence experiment; converged sinks are not boundary saddles.')
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
