"""Post-outcome saved right-scaling and fsum arithmetic; no model/solver replay."""
import hashlib,importlib.util,json,math,os,signal,subprocess,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sp=importlib.util.spec_from_file_location('saved_arithmetic',ROOT/'tools/reviews/audit_cifar_damped_saved.py');H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
RAW=Path('/data2/neural-shrimp-data/explore/cifar/diagonal-preconditioner/attempt-001');PIN='0734d5900c9240dab0819fa958fa1d608e8beb1bdb14e10f5d2f4d48928d515a';LPIN='512f1a9097f3da25f9e59dadb80f7fb7d5003720c98ca858eb57aa9dd95bc6b2';EXEC='bd458b253a99b7304d82dbd5097b1df91ac88047'
def read(n):
 a=np.load(RAW/(n+'.npy'),allow_pickle=False);H.require(a.dtype==np.float64 and np.isfinite(a).all(),'finite array');return a
def audit():
 inv=H.verify(RAW,PIN);launcher=RAW.with_name(RAW.name+'-launcher');H.verify(launcher,LPIN);r=json.loads((RAW/'result.json').read_text());l=json.loads((launcher/'receipt.json').read_text());H.require(r['source_commit']==l['source_commit']==EXEC and l['successful']and r['status']=='diagnostic_complete','complete source identity')
 for n,d in r['source_sha256'].items():H.require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'historical source')
 parent=Path(r['input_identity']['parent_root']);H.verify(parent,r['input_identity']['parent_manifest']);theta,one,two,res=[read('initial-'+n)for n in('theta','map1','map2','residual')]
 for n,v in [('theta',theta),('map1',one),('map2',two)]:H.require(H.equal(v,H.load('newton-0-trial-5-'+n,parent)),'exact parent parity')
 H.require(H.equal(two-theta,res),'residual identity');eta=r['input_identity']['eta'];lam=r['input_identity']['lam'];q=1-Fraction.from_float(eta)*Fraction.from_float(lam);floor=abs(q*q-1);H.require(str(floor)==r['scaling']['ridge_floor_exact'],'exact ridge floor')
 d0,d1=[read('phase'+str(i)+'-diagonal')for i in(0,1)];proxy=(1-eta*d1)*(1-eta*d0)-1;scale=1/np.maximum(np.abs(proxy),float(floor));H.require(H.equal(proxy,read('approximate-residual-diagonal'))and H.equal(scale,read('right-inverse-scaling')),'right scaling bytes')
 # Reconstruct the bias/output blocks from saved activation and hidden-curvature returns.
 for i,diag in enumerate((d0,d1)):
  z=read(f'phase{i}-activation');c=read(f'phase{i}-hidden_curvature');tail=np.r_[c.mean(axis=0)+lam,np.repeat((z*z).mean(axis=0),10)+lam,np.full(10,1+lam)];H.require(np.allclose(tail,diag[3072*32:],rtol=2e-13,atol=1e-16),'saved diagonal tail')
 y=read('solver-direction');p=read('physical-direction');ap=read('actual-direction-action');e=read('actual-linear-error');H.require(H.equal(scale*y,p)and H.equal(ap+res,e),'physical direction and actual residual bytes');forcing=H.norm(e)/H.norm(res);slope=H.dot(res,ap);H.close(forcing,r['linear']['true_forcing_ratio']);H.close(slope,r['linear']['merit_directional_derivative'])
 H.require(forcing>.5 and slope<0 and not r['linear']['forcing_test_pass']and r['stop']=='true_forcing_or_descent_failed'and r['accepted_steps']==0 and not r['trials'],'failed forcing, no trial')
 H.require(r['linear']['gmres_info']==1 and r['linear']['inner_iterations']==400 and r['linear']['matvec_calls']==402 and len(r['linear']['estimated_relative_residual_history'])==400,'fixed one400 solve');events=[json.loads(x)for x in(RAW/'events.jsonl').read_text().splitlines()];counts={}
 for k,n in dict(action=402,j2vp=402,hvp=804,direct_gradient=2,analytic_diagonal=4).items():
  starts=[e['index']for e in events if e['kind']=='operation_started'and e['operation']==k];returns=[e['index']for e in events if e['kind']=='operation_returned'and e['operation']==k];H.require(starts==returns==list(range(1,n+1))and r['operation_returned'][k]==r['operation_attempted'][k]==n,'counter joins');counts[k]=n
 H.require(not r['accepted_orbit']and not r['applied_to_continuation']and not r['spectrum_requested']and all(not Path('/proc/'+str(p)).exists()for p in l['observed_owned_pids']),'scope/process');H.close(H.norm(res)/max(1.,H.norm(theta)),r['final_relative_residual'])
 return dict(passed=True,producer_execution=EXEC,worker_manifest=PIN,launcher_manifest=LPIN,actual_forcing=forcing,actual_slope=slope,counts=counts,scaling_range=[float(scale.min()),float(scale.max())],scope='post-outcome separately written producer-agent saved arithmetic; no new model/derivative/action/solve; hidden-weight diagonal and actual operator rely on bound returns plus prior synthetic formula tests; independent personnel audit pending')
def main():
 H.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');out=Path('/data2/neural-shrimp-data/audits/cifar-diagonal-saved/attempt-001');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('15s cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(seconds=time.monotonic()-start,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=H.sha(Path(__file__)),helper_sha256=H.sha(ROOT/'tools/reviews/audit_cifar_damped_saved.py'));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(bytes=(out/'result.json').stat().st_size,sha256=H.sha(out/'result.json'))}},indent=2)+'\n')
 for f in out.iterdir():f.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
