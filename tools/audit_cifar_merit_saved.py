#!/usr/bin/env python3
"""Independent saved-only merit acceptance arithmetic; never imports native model code."""
import argparse,hashlib,json,math,os,resource,signal,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/merit-lsmr/attempt-001')
PIN='b5a555e1247f20dc2c2f1da62a70c7c256eb43d694386308388ab40ed8e799e6'
LAUNCH_PIN='9185c62b26882c2de2e062069ccc2828c79a3b98c792487d8bed271329aa1c74'
EXEC='6a6362d3bdb0929655dd5e2202de6473fd308a37'
OUT=Path('/data2/neural-shrimp-data/audits/cifar-merit-independent/attempt001/run')
SOURCES=['tools/audit_cifar_merit_saved.py','tools/launch_cifar_merit_saved_audit.py','tests/test_cifar_merit_saved_audit.py','docs/plans/cifar-merit-saved-audit.md']

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dot(x,y):return math.fsum(float(a)*float(b)for a,b in zip(x,y,strict=True))
def norm(x):return math.sqrt(dot(x,x))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=5e-12,abs_tol=1e-25)
def metrics(base,first,second,residual,action,alpha):
 phi=dot(residual,residual)/2;slope=dot(residual,action);rr=second-base
 merit=dot(rr,rr)/2;prediction=phi-dot(residual+alpha*action,residual+alpha*action)/2
 actual=phi-merit;ratio=actual/prediction if prediction>0 else None;scale=max(1.,norm(base));sep=norm(first-base)/scale
 accepted=slope<0 and prediction>0 and actual>0 and ratio>=.1 and merit<=phi+1e-4*alpha*slope and sep>=1e-7
 return dict(initial_merit=phi,merit=merit,slope=slope,predicted_reduction=prediction,actual_reduction=actual,ratio=ratio,relative_residual=norm(rr)/scale,relative_separation=sep,accepted_local_step=bool(accepted),closure_screen=bool(norm(rr)/scale<1e-8),primitive_screen=bool(sep>=1e-7))

def verify(root,pin,check):
 need(sha(root/'manifest.json')==pin,'fixed manifest')
 inv=read(root/'manifest.json')['files'];need(set(inv)=={str(p.relative_to(root))for p in root.iterdir()if p.is_file()and p.name!='manifest.json'},'complete flat inventory')
 for name,d in inv.items():
  check();p=root/name;need(Path(name).name==name and not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==d['bytes']and sha(p)==d['sha256'],'immutable inventory bytes')
 return {'root':str(root),'manifest':pin,'files_including_manifest':len(inv)+1,'bytes_including_manifest':sum(v['bytes']for v in inv.values())+(root/'manifest.json').stat().st_size}

def main(out):
 need(out==OUT,'fixed exclusive output');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']),'CPU1nice10')
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];used=[0]
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
 def check():
  if stopped[0]or time.monotonic()-start>30:raise TimeoutError('30s audit cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB audit RSS')
 def save(name,value):
  raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();need(used[0]+len(raw)<2*1024**2,'2MiB audit raw');(out/name).write_bytes(raw);used[0]+=len(raw)
 identity={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:sha(ROOT/n)for n in SOURCES}}
 save('runtime.json',dict(identity,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),native_calls=0));result={'passed':False,'steps':[]}
 try:
  check();result['input_inventory']=verify(PARENT,PIN,check);result['launcher_inventory']=verify(PARENT.with_name(PARENT.name+'-launcher'),LAUNCH_PIN,check)
  source=read(PARENT/'result.json');launch=read(PARENT.with_name(PARENT.name+'-launcher')/'receipt.json')
  need(source['source_commit']==EXEC==launch['source_commit']and launch['process_exit']==0 and launch['failure']is None and launch['owned_live_pids_remaining']==[],'fixed completed owned producer')
  for pid in launch['observed_owned_pids']:need(not Path('/proc',str(pid)).exists(),'producer PID absent')
  for name,digest in source['source_sha256'].items():
   check();need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+name],cwd=ROOT)).hexdigest()==digest,'exact historical source '+name)
  need(source['status']=='merit_pilot_complete'and source['accepted_steps']==6 and len(source['steps'])==6 and source['accepted_orbit']is False and source['spectrum_requested']is False,'fixed six-step exploratory outcome')
  started=Counter();returned=Counter()
  for event in map(json.loads,(PARENT/'events.jsonl').read_text().splitlines()):
   if event['kind']=='operation_started':started[event['operation']]+=1;need(event['index']==started[event['operation']],'sequential attempted count')
   if event['kind']=='operation_returned':returned[event['operation']]+=1;need(event['index']==returned[event['operation']],'sequential returned count')
  expected={'action':1212,'adjoint':1206,'j2vp':1212,'hvp':4836,'direct_gradient':14}
  need(dict(started)==dict(returned)==source['operation_attempted']==source['operation_returned']==expected,'independent actual operation counts')
  def array(name):
   check();a=np.load(PARENT/name,allow_pickle=False);need(a.dtype==np.float64 and a.shape==(98666,)and np.isfinite(a).all(),'finite native array');return a
  theta,one,two=[array('initial-'+n+'.npy')for n in['theta','map1','map2']]
  parent=Path(source['input_identity']['parent_root']);need(sha(parent/'manifest.json')==source['input_identity']['parent_manifest']and sha(parent/'initial-theta.npy')==source['input_identity']['parent_state_sha256'],'qualified parent state pin')
  need(np.array_equal(theta,np.load(parent/'initial-theta.npy',allow_pickle=False)),'unchanged qualified initial state')
  initial_closure=norm(two-theta)/max(1,norm(theta));initial_merit=dot(two-theta,two-theta)/2;damp=.001;last='initial-theta.npy'
  for k,step in enumerate(source['steps']):
   check();need(step['index']==k and step['base_state']==last and step['accepted_index']==0 and len(step['trials'])==1,'fixed actual fullstep chain')
   r=array(f'outer-{k}-residual.npy');p=array(f'outer-{k}-direction.npy');ap=array(f'outer-{k}-actual-action.npy');lin=read(PARENT/f'outer-{k}-linear.json')
   need(lin==step['linear']and np.array_equal(r,two-theta),'saved actual linearization residual')
   forcing=norm(r+ap)/norm(r);slope=dot(r,ap);pred=dot(r,r)/2-dot(r+ap,r+ap)/2;radius=.05*max(1,norm(theta))
   need(close(lin['true_forcing'],forcing)and close(lin['true_slope'],slope)and close(lin['full_predicted_reduction'],pred)and close(lin['radius'],radius),'independent true forcing slope prediction radius')
   need(lin['old_newton_forcing_pass']==(forcing<=.5)and lin['damp']==damp and norm(p)<=radius*(1+1e-12),'forcing classification damping trust radius')
   need(lin['lsmr_iterations']==200 and lin['lsmr_stop']==7 and lin['calls']=={'forward':202,'adjoint':201},'retained capped LSMR result')
   x,y,z=[array(f'outer-{k}-trial-0-'+n+'.npy')for n in['theta','map1','map2']];saved=step['trials'][0];alpha=saved['alpha']
   need(alpha==1 and np.array_equal(x,theta+alpha*p),'actual full-step bytes')
   fresh=metrics(x,y,z,r,ap,alpha)
   for key,value in fresh.items():need(saved[key]is value if isinstance(value,bool)else close(saved[key],value),'independent trial metric '+key)
   need(fresh['accepted_local_step']and saved['accepted_orbit']is False,'actual saved step acceptance and no orbit')
   result['steps'].append(dict(index=k,true_forcing=forcing,old_newton_forcing_pass=forcing<=.5,damp=damp,direction_norm=norm(p),radius=radius,**fresh));save(f'step{k}.json',result['steps'][-1])
   if fresh['ratio']>.75:damp=max(1e-9,damp/4)
   elif fresh['ratio']<.25:damp=min(1.,damp*4)
   theta,one,two=x,y,z;last=f'outer-{k}-trial-0-theta.npy'
  final=norm(two-theta)/max(1,norm(theta));sep=norm(one-theta)/max(1,norm(theta));merit=dot(two-theta,two-theta)/2
  need(close(source['final_relative_residual'],final)and close(source['final_relative_separation'],sep)and source['last_accepted_state']==last and source['final_damp']==damp,'independent final norms/state/damping')
  result.update(passed=True,initial_relative_residual=initial_closure,final_relative_residual=final,relative_residual_fractional_reduction=1-final/initial_closure,initial_merit=initial_merit,final_merit=merit,final_relative_separation=sep,closure_screen=final<1e-8,old_forcing_passes=sum(x['old_newton_forcing_pass']for x in result['steps']),operation_counts=expected)
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e))
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=0,scope='Independent saved-array acceptance/forcing/norm/count audit only. Does not rerun native maps, adjoints or LSMR and does not certify an orbit.')
 save('result.json',result);save('manifest.json',{'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in out.iterdir()if p.is_file()}})
 return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
