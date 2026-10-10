"""Post-outcome saved arithmetic with independent fsum reductions; no model import."""
import argparse,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/explore/cifar/damped-inexact/attempt-001')
PIN='a17661b9f202574d675bf0d1c4dcdd2ab8d5f80abda9502c349d159eddbe7c1a'
LPIN='aff643aa997c679e541f7d4e703ee890c219b4d4d5b2894b62b9567d5c2c9de1'
EXEC='87156e9ea3aa178405ecc4d23d1332e9c756f3cf'
def require(x,s):
 if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(p,pin):
 require(sha(p/'manifest.json')==pin,'pin');m=json.loads((p/'manifest.json').read_text())['files'];require(set(m)=={f.name for f in p.iterdir()if f.name!='manifest.json'},'full inventory')
 for n,d in m.items():
  f=p/n;require(Path(n).name==n and not f.is_symlink()and f.is_file()and f.stat().st_size==d['bytes']and sha(f)==d['sha256']and not(f.stat().st_mode&0o222),'immutable member')
 return m
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def norm(a):return math.sqrt(dot(a,a))
def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def close(a,b):require(math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,rel_tol=3e-12,abs_tol=1e-22),'scalar equality')
def load(n,root=RAW):
 a=np.load(root/(n+'.npy'),allow_pickle=False);require(a.dtype==np.float64 and a.shape==(98666,)and np.isfinite(a).all(),'finite native vector');return a
def decisions(theta,one,two,phi,slope,alpha):
 rr=two-theta;merit=dot(rr,rr)/2;relative=norm(rr)/max(1.,norm(theta));sep=norm(one-theta)/max(1.,norm(theta));upper=phi+1e-4*alpha*slope
 return dict(merit=merit,merit_ratio=merit/phi,armijo_upper=upper,armijo_pass=merit<=upper,strict_merit_decrease=merit<phi,relative_residual=relative,relative_separation=sep,closure_screen=relative<1e-8,primitive_screen=sep>=1e-7,accepted_local_step=bool(merit<phi and merit<=upper and sep>=1e-7))
def audit():
 inv=verify(RAW,PIN);launch=RAW.with_name(RAW.name+'-launcher');verify(launch,LPIN);r=json.loads((RAW/'result.json').read_text());lr=json.loads((launch/'receipt.json').read_text());require(r['source_commit']==lr['source_commit']==EXEC and lr['successful']and r['status']=='diagnostic_complete','complete/source')
 for n,d in r['source_sha256'].items():require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'historical source')
 parent=Path(r['input_identity']['parent_root']);verify(parent,r['input_identity']['parent_manifest']);theta,one,two,res=[load(n)for n in('initial-theta','initial-map1','initial-map2','initial-residual')]
 for n,a in [('theta',theta),('map1',one),('map2',two)]:require(equal(a,load('trial-1-'+n,parent)),'parent replay')
 require(equal(res,two-theta),'initial difference');initial_phi=dot(res,res)/2;rows=[];accepted=0;state='initial-theta.npy'
 for i,s in enumerate(r['steps']):
  require(s['iteration']==i and s['base_state']==state,'sequential base');tag=f'newton-{i}';p,ap,e=[load(tag+'-'+n)for n in('direction','applied-direction','linear-error')];res=two-theta;require(equal(e,ap+res),'linear error identity');phi=dot(res,res)/2;forcing=norm(e)/norm(res);slope=dot(res,ap);lin=s['linear'];close(forcing,lin['true_forcing_ratio']);close(slope,lin['merit_directional_derivative']);require(lin['forcing_test_pass']==(forcing<=.5)and lin['descent_direction']==(slope<0),'linear decision');require(lin['inner_iterations']==400 and lin['matvec_calls']==402 and len(lin['estimated_relative_residual_history'])==400,'fixed solves');require(lin['gmres_info']==1 and not lin['requested_tolerance_met'],'strict scope');local=[]
  for j,t in enumerate(s['trials']):
   alpha=2.**(-j);require(t['index']==j and t['alpha']==alpha and forcing<=.5 and slope<0,'fixed permitted trials');prefix=tag+f'-trial-{j}';x,f,g,rr=[load(prefix+'-'+n)for n in('theta','map1','map2','residual')];require(equal(x,theta+alpha*p)and equal(rr,g-x),'trial bytes');d=decisions(x,f,g,phi,slope,alpha)
   for k,v in d.items():
    if isinstance(v,bool):require(t[k]==v,'boolean '+k)
    else:close(v,t[k])
   local.append(dict(alpha=alpha,**d));require(not t['accepted_orbit'],'no orbit')
   if d['accepted_local_step']:
    require(j==len(s['trials'])-1 and s['accepted_index']==j,'first passing accepted');marker=json.loads((RAW/(tag+'-accepted.json')).read_text());require(marker['state']==prefix+'-theta.npy'and marker['alpha']==alpha,'durable marker');theta,one,two=x,f,g;state=prefix+'-theta.npy';accepted+=1
  if not(forcing<=.5 and slope<0):require(not s['trials']and s['accepted_index']is None and i==len(r['steps'])-1,'failed direction terminal')
  rows.append(dict(iteration=i,forcing=forcing,slope=slope,accepted_index=s['accepted_index'],trials=local))
 require(len(rows)<=2 and accepted==r['accepted_steps']and state==r['last_accepted_state'],'accepted state');relative=norm(two-theta)/max(1.,norm(theta));close(relative,r['final_relative_residual']);close(norm(one-theta)/max(1.,norm(theta)),r['final_relative_separation']);events=[json.loads(x)for x in(RAW/'events.jsonl').read_text().splitlines()];counts={}
 for k,cap in dict(action=804,j2vp=804,hvp=1608,direct_gradient=54).items():
  a=[e['index']for e in events if e['kind']=='operation_started'and e['operation']==k];b=[e['index']for e in events if e['kind']=='operation_returned'and e['operation']==k];require(a==b==list(range(1,r['operation_returned'][k]+1))and len(a)==r['operation_attempted'][k]<=cap,'operation counters');counts[k]=len(a)
 require(not r['accepted_orbit']and not r['applied_to_continuation']and not r['spectrum_requested'],'scope');require(r['stop']=='true_forcing_or_descent_failed'and rows[-1]['forcing']>.5,'stop');require(lr['process_exit']==0 and not lr['owned_live_pids_remaining']and all(not Path('/proc/'+str(p)).exists()for p in lr['observed_owned_pids']),'process exit')
 return dict(passed=True,producer_execution=EXEC,raw_manifest=PIN,launcher_manifest=LPIN,files=len(inv),steps=rows,counts=counts,accepted_steps=accepted,final_relative_residual=relative,total_merit_ratio=dot(two-theta,two-theta)/(2*initial_phi),scope='post-outcome separately written saved arithmetic by producer agent; shared map/Ap returns, no independent derivative/solver replay; no orbit claim')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice');a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('15s cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=sha(Path(__file__)),seconds=time.monotonic()-start,pid=os.getpid());(a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(a.out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(bytes=(a.out/'result.json').stat().st_size,sha256=sha(a.out/'result.json'))}},indent=2)+'\n')
 for f in a.out.iterdir():f.chmod(0o444)
 a.out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
