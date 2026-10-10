"""Post-outcome saved right-scaling and fsum arithmetic; no model/solver replay."""
import hashlib,importlib.util,json,math,os,signal,subprocess,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sp=importlib.util.spec_from_file_location('saved_arithmetic',ROOT/'tools/reviews/audit_cifar_damped_saved.py');H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
RAW=Path('/data2/neural-shrimp-data/explore/cifar/input-gram-preconditioner/attempt-001');PIN='9a7aa4c59fc5b24cbf89f62a6d5a69a94e66de5f94bd6bad9c029c13cb7604f4';LPIN='4130d35360d93ec17031e7e583639419eab752fd444c178a76c6186af4c7996c';EXEC='b5e961a820da458c6a47627794e21bc21fd69954'
def read(n):
 a=np.load(RAW/(n+'.npy'),allow_pickle=False);H.require(a.dtype==np.float64 and np.isfinite(a).all(),'finite array');return a
def audit():
 inv=H.verify(RAW,PIN);launcher=RAW.with_name(RAW.name+'-launcher');H.verify(launcher,LPIN);r=json.loads((RAW/'result.json').read_text());l=json.loads((launcher/'receipt.json').read_text());H.require(r['source_commit']==l['source_commit']==EXEC and l['successful']and r['status']=='diagnostic_complete','complete source identity')
 for n,d in r['source_sha256'].items():H.require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'historical source')
 parent=Path(r['input_identity']['parent_root']);H.verify(parent,r['input_identity']['parent_manifest']);theta,one,two,res=[read('initial-'+n)for n in('theta','map1','map2','residual')]
 for n,v in [('theta',theta),('map1',one),('map2',two)]:H.require(H.equal(v,H.load('newton-0-trial-5-'+n,parent)),'exact parent parity')
 H.require(H.equal(two-theta,res),'residual identity');eta=r['input_identity']['eta'];lam=r['input_identity']['lam'];q=1-Fraction.from_float(eta)*Fraction.from_float(lam);floor=abs(q*q-1);H.require(str(floor)==r['scaling']['ridge_floor_exact'],'exact ridge floor')
 d0,d1=[read('phase'+str(i)+'-diagonal')for i in(0,1)];u=read('target-gram-basis-vectors');values=read('target-gram-eigh-values');evec=read('target-gram-eigh-vectors');q=read('target-gram-qr-q');rr=read('target-gram-qr-r');small=read('target-gram-eigh-matrix');H.require(u.shape==(3073,512)and values.shape==(512,)and np.all(np.diff(values)>=0),'full ordered basis')
 checks=dict(q_orth=float(np.linalg.norm(q.T@q-np.eye(512))),v_orth=float(np.linalg.norm(evec.T@evec-np.eye(512))),matrix_identity=float(np.linalg.norm(small-rr@rr.T/512))/np.linalg.norm(small),eigen_residual=float(np.linalg.norm(small@evec-evec*values))/np.linalg.norm(small),basis_identity=float(np.linalg.norm(u-q@evec))/np.linalg.norm(u));H.require(max(checks.values())<1e-10,'saved Gram products and eigenpairs');H.require(H.equal(np.maximum(values,0),read('target-gram-basis-eigenvalues')),'roundoff handling');means=[]
 for i in (0,1):
  m=read(f'phase{i}-hidden_curvature').mean(axis=0);H.require(H.equal(m,read(f'target-scaling-phase{i}_mean_curvature')),'phase means');means.append(m)
 a,b=[lam+np.maximum(values,0)[:,None]*m for m in means];proxy=(1-eta*b)*(1-eta*a)-1;modes=1/np.maximum(abs(proxy),float(floor));op=(1-eta*d1[98336:])*(1-eta*d0[98336:])-1;output=1/np.maximum(abs(op),float(floor));H.require(H.equal(proxy,read('target-scaling-hidden_mode_proxy'))and H.equal(modes,read('target-scaling-hidden_mode_inverse'))and H.equal(output,read('target-scaling-output_inverse')),'mode proxies and inverses')
 y=read('solver-direction');p=read('physical-direction');hidden=y[:98336].reshape(3073,32);coords=u.T@hidden;reconstructed=np.r_[((hidden-u@coords)/float(floor)+u@(modes*coords)).ravel(),output*y[98336:]];direction_error=H.norm(reconstructed-p)/max(H.norm(p),1e-300);H.require(direction_error<1e-10,'independent projector right action');ap=read('actual-direction-action');e=read('actual-linear-error');H.require(H.equal(ap+res,e),'actual error bytes');forcing=H.norm(e)/H.norm(res);slope=H.dot(res,ap);H.close(forcing,r['linear']['true_forcing_ratio']);H.close(slope,r['linear']['merit_directional_derivative'])
 H.require(forcing>.5 and slope<0 and not r['linear']['forcing_test_pass']and r['stop']=='true_forcing_or_descent_failed'and r['accepted_steps']==0 and not r['trials'],'failed forcing, no trial')
 H.require(r['linear']['gmres_info']==1 and r['linear']['inner_iterations']==400 and r['linear']['matvec_calls']==402 and len(r['linear']['estimated_relative_residual_history'])==400,'fixed one400 solve');events=[json.loads(x)for x in(RAW/'events.jsonl').read_text().splitlines()];counts={}
 for k,n in dict(action=402,j2vp=402,hvp=804,direct_gradient=2,analytic_diagonal=4,qr=2,eigh=2,right_application=407).items():
  starts=[e['index']for e in events if e['kind']=='operation_started'and e['operation']==k];returns=[e['index']for e in events if e['kind']=='operation_returned'and e['operation']==k];H.require(starts==returns==list(range(1,n+1))and r['operation_returned'][k]==r['operation_attempted'][k]==n,'counter joins');counts[k]=n
 H.require(not r['accepted_orbit']and not r['applied_to_continuation']and not r['spectrum_requested']and all(not Path('/proc/'+str(p)).exists()for p in l['observed_owned_pids']),'scope/process');H.close(H.norm(res)/max(1.,H.norm(theta)),r['final_relative_residual'])
 return dict(passed=True,producer_execution=EXEC,worker_manifest=PIN,launcher_manifest=LPIN,actual_forcing=forcing,actual_slope=slope,counts=counts,saved_gram_checks=checks,right_action_relative_error=direction_error,scope='post-outcome separately written producer-agent saved arithmetic; no new model/derivative/action/solve; raw input-to-QR factorization and actual operator rely on source-bound returns; saved Gram and spectral residuals checked without a new decomposition; independent personnel audit pending')
def main():
 H.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');out=Path('/data2/neural-shrimp-data/audits/cifar-gram-saved/attempt-001');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('15s cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(seconds=time.monotonic()-start,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=H.sha(Path(__file__)),helper_sha256=H.sha(ROOT/'tools/reviews/audit_cifar_damped_saved.py'));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(bytes=(out/'result.json').stat().st_size,sha256=H.sha(out/'result.json'))}},indent=2)+'\n')
 for f in out.iterdir():f.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
