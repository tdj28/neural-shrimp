"""Post-outcome independent saved arithmetic, without importing model/solver code."""
import argparse,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/explore/cifar/large-space/attempt-002')
RAW_PIN='814d5ebb4b4422c971eeb48425435080ce21f58aa9d2fa83c44f200cdcbb0f3b'
LAUNCH_PIN='2abe1f4d4d48f5a97827f2afe3bb9e01df64a6564087da549925d912827e16de'
EXEC='bcba5eea358640fb3f8b6672f7fdbe88aef298c6'
def require(x,s):
 if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(p,pin):
 require(p.is_dir() and not p.is_symlink() and sha(p/'manifest.json')==pin,'manifest pin')
 m=json.loads((p/'manifest.json').read_text())['files'];require(set(m)=={q.name for q in p.iterdir()if q.name!='manifest.json'},'complete inventory')
 for n,d in m.items():
  q=p/n;require(Path(n).name==n and q.is_file()and not q.is_symlink(),'safe member');require(q.stat().st_size==d['bytes']and sha(q)==d['sha256'],'member bytes')
 return m
def dot(x,y):return math.fsum(float(a)*float(b)for a,b in zip(x,y,strict=True))
def norm(x):return math.sqrt(dot(x,x))
def close(x,y):require(math.isfinite(float(x))and math.isfinite(float(y))and math.isclose(float(x),float(y),rel_tol=3e-12,abs_tol=1e-22),'saved scalar mismatch')
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def load(name):
 a=np.load(RAW/(name+'.npy'),allow_pickle=False);require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite native vector');return a

def audit():
 inv=verify(RAW,RAW_PIN);launch=RAW.with_name(RAW.name+'-launcher');verify(launch,LAUNCH_PIN)
 r=json.loads((RAW/'result.json').read_text());lr=json.loads((launch/'receipt.json').read_text());require(r['source_commit']==lr['source_commit']==EXEC and lr['successful']and r['comparison_qualified'],'completed source binding')
 for p,d in r['source_sha256'].items():require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+p],cwd=ROOT)).hexdigest()==d,'historical source')
 theta,one,two,res,step,ap,e=[load(n)for n in ('initial-theta','initial-map1','initial-map2','initial-residual','direction','applied-direction','true-linear-residual')]
 require(equal(two-theta,res)and equal(ap+res,e),'saved difference identities')
 phi=dot(res,res)/2;slope=dot(res,ap);forcing=norm(e)/norm(res);lin=r['linear'];close(forcing,lin['true_forcing_ratio']);close(slope,lin['merit_directional_derivative']);require(forcing<=.5 and slope<0 and not lin['requested_tolerance_met']and lin['gmres_info']==1,'linear scope')
 base=Path(r['input_identity']['comparison200_root']);verify(base,r['input_identity']['comparison200_manifest']);baseline=json.loads((base/'linear.json').read_text())['estimated_relative_residual_history'];history=lin['estimated_relative_residual_history'];require(len(history)==400 and np.allclose(history[:200],baseline,rtol=1e-10,atol=1e-12),'prefix200')
 events=[json.loads(s)for s in(RAW/'events.jsonl').read_text().splitlines()];counts={}
 for kind in ('action','j2vp','hvp','direct_gradient'):
  a=[x['index']for x in events if x['kind']=='operation_started'and x['operation']==kind];b=[x['index']for x in events if x['kind']=='operation_returned'and x['operation']==kind]
  require(a==list(range(1,r['operation_attempted'][kind]+1))and b==list(range(1,r['operation_returned'][kind]+1))and a==b,'counter joins');counts[kind]=len(a)
 rows=[]
 for i,alpha in enumerate((1.,.5,.25,.125)):
  t,m1,m2,tr=[load(f'trial-{i}-'+n)for n in ('theta','map1','map2','residual')];require(equal(t,theta+alpha*step)and equal(tr,m2-t),'trial bytes')
  merit=dot(tr,tr)/2;upper=phi+1e-4*alpha*slope;relative=norm(tr)/max(1,norm(t));sep=norm(m1-t)/max(1,norm(t));saved=r['trials'][i]
  for x,y in [(merit,saved['merit']),(merit/phi,saved['merit_ratio']),(upper,saved['armijo_upper']),(relative,saved['relative_residual']),(sep,saved['relative_separation'])]:close(x,y)
  require(saved['armijo_pass']==(merit<=upper)and saved['closure_screen']==(relative<1e-8)and saved['primitive_screen']==(sep>=1e-7)and not saved['inexact_step_candidate'],'trial decisions')
  rows.append(dict(alpha=alpha,merit_ratio=merit/phi,armijo_pass=merit<=upper,relative_residual=relative,separation=sep))
 require(not r['accepted_orbit']and not r['applied_to_continuation']and not r['spectrum_requested'],'no orbit claim')
 gradients=[e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='direct_gradient'];require(len(gradients)==11 and all(math.isfinite(x)and x>=0 for x in gradients),'measured gradient times')
 return dict(passed=True,producer_execution=EXEC,raw_manifest=RAW_PIN,launcher_manifest=LAUNCH_PIN,files=len(inv),forcing=forcing,slope=slope,initial_merit=phi,counts=counts,trials=rows,max_saved_gradient_seconds=max(gradients),smaller_trial_cost_projection=2*20*max(gradients)+5,scope='post-outcome saved scalar/byte/counter/source review; no model, derivative, GMRES or trajectory; does not independently recompute Ap or map values')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={}
 def stop(*_):raise TimeoutError('15s audit cap')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,15)
 try:result=audit()
 except BaseException as e:result=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=sha(Path(__file__)),seconds=time.monotonic()-start,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0));(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(sha256=sha(a.out/'result.json'),bytes=(a.out/'result.json').stat().st_size)}},indent=2)+'\n')
 for p in a.out.iterdir():p.chmod(0o444)
 a.out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
