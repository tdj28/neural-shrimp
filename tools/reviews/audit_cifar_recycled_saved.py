"""Post-outcome saved SVD/reduction check; no new decomposition/model/solve."""
import hashlib,importlib.util,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sp=importlib.util.spec_from_file_location('saved_arithmetic',ROOT/'tools/reviews/audit_cifar_damped_saved.py');H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
RAW=Path('/data2/neural-shrimp-data/explore/cifar/recycled-direction/attempt-001');PIN='1a5b898db341ddd4106a3188833d82590922312a438bfdd0c16c5d68ce791a3a';LPIN='eacecdcfe8b15198191eea7a27373eff6fe87b11c4577cc6402580b2cfc4fcd8';EXEC='ac0323ab28976c5c0025f987bc05775ad612ac9b'
def read(n):
 a=np.load(RAW/(n+'.npy'),allow_pickle=False);H.require(a.dtype==np.float64 and np.isfinite(a).all(),'finite arrays');return a
def near(a,b):H.require(a.shape==b.shape and np.allclose(a,b,rtol=1e-11,atol=1e-18),'array equality')
def audit():
 inv=H.verify(RAW,PIN);launcher=RAW.with_name(RAW.name+'-launcher');H.verify(launcher,LPIN);r=json.loads((RAW/'result.json').read_text());l=json.loads((launcher/'receipt.json').read_text());H.require(r['source_commit']==l['source_commit']==EXEC and l['successful']and r['status']=='diagnostic_complete','complete identity')
 for n,d in r['source_sha256'].items():H.require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'historical source')
 parent=Path(r['input_identity']['parent_root']);H.verify(parent,r['input_identity']['parent_manifest']);old=Path(r['input_identity']['old400_root']);H.verify(old,r['input_identity']['old400_manifest']);p=read('historical-directions');a=read('historical-actions-at-current-state');theta,one,two,res=[read('initial-'+n)for n in('theta','map1','map2','residual')]
 H.require(H.equal(p[:,0],H.load('direction',old))and H.equal(p[:,1],H.load('newton-0-direction',parent))and H.equal(p[:,2],H.load('newton-1-direction',parent)),'fixed historical columns')
 for n,v in [('theta',theta),('map1',one),('map2',two)]:H.require(H.equal(v,H.load('newton-0-trial-5-'+n,parent)),'exact parent parity')
 H.require(H.equal(two-theta,res),'residual identity')
 for i in range(2):H.require(H.equal(a[:,i],read(f'fresh-historical-action-{i}')),'fresh action column')
 H.require(H.equal(a[:,2],H.load('newton-1-applied-direction',parent)),'actual current action')
 scales=np.array([H.norm(p[:,i])for i in range(3)]);near(scales,read('fit-scales-column_norms'));pn=p/scales;an=a/scales;near(pn,read('fit-normalized-directions'));near(an,read('fit-normalized-actions'))
 u,s,vh=[read('fit-svd-'+n)for n in('u','s','vh')];H.require(u.shape==(98666,3)and s.shape==(3,)and vh.shape==(3,3)and np.all(s>=0)and np.all(np.diff(s)<=0),'complete ordered triplet');reconstruction=H.norm((an-(u*s)@vh).ravel())/H.norm(an.ravel());uo=H.norm((u.T@u-np.eye(3)).ravel());vo=H.norm((vh@vh.T-np.eye(3)).ravel());H.require(max(reconstruction,uo,vo)<=1e-10,'saved decomposition residuals');rank=int(sum(x>1e-10*s[0]for x in s));condition=float(s[0]/s[-1]);H.require(rank==r['rank']['rank']==3 and condition<=1e8,'rank gate');H.close(condition,r['rank']['condition'])
 rhs=np.array([-H.dot(u[:,i],res)/s[i]for i in range(3)]);coeff=np.array([H.dot(vh[:,i],rhs)for i in range(3)]);near(coeff,read('fit-fit-normalized_coefficients'));near(coeff/scales,read('fit-fit-original_coefficients'));pred=an@coeff;direction=pn@coeff;near(pred,read('fit-fit-predicted_action'));near(direction,read('combined-direction'));H.require(H.equal(read('combined-direction'),read('fit-fit-direction')),'direction returned identity')
 actual=read('actual-combined-action');error=actual+res;H.require(H.equal(error,read('actual-combined-error')),'actual error identity');forcing=H.norm(error)/H.norm(res);mismatch=H.norm(actual-read('fit-fit-predicted_action'))/max(H.norm(res),H.norm(actual),H.norm(read('fit-fit-predicted_action')));slope=H.dot(res,actual)
 for x,y in [(forcing,r['linear']['true_forcing_ratio']),(mismatch,r['linear']['relative_action_mismatch']),(slope,r['linear']['merit_directional_derivative'])]:H.close(x,y)
 H.require(mismatch<=1e-8 and forcing>.5 and slope<0 and not r['linear']['qualified']and r['stop']=='combined_true_linear_gate_failed'and not r['trials']and r['accepted_steps']==0,'failed forcing scope')
 events=[json.loads(x)for x in(RAW/'events.jsonl').read_text().splitlines()];counts={}
 for k,n in dict(action=3,j2vp=3,hvp=6,direct_gradient=2).items():
  starts=[e['index']for e in events if e['kind']=='operation_started'and e['operation']==k];returns=[e['index']for e in events if e['kind']=='operation_returned'and e['operation']==k];H.require(starts==returns==list(range(1,n+1))and r['operation_returned'][k]==r['operation_attempted'][k]==n,'counter joins');counts[k]=n
 H.require(not r['accepted_orbit']and not r['applied_to_continuation']and not r['spectrum_requested']and all(not Path('/proc/'+str(p)).exists()for p in l['observed_owned_pids']),'scope/process')
 return dict(passed=True,producer_execution=EXEC,worker_manifest=PIN,launcher_manifest=LPIN,condition=condition,rank=rank,svd_residual=reconstruction,u_orthogonality=uo,v_orthogonality=vo,actual_forcing=forcing,actual_slope=slope,actual_predicted_mismatch=mismatch,original_coefficients=(coeff/scales).tolist(),counts=counts,scope='post-outcome separately written producer-agent saved arithmetic; no new SVD/solve/model/action; shared raw derivative values, independent personnel audit pending')
def main():
 H.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');out=Path('/data2/neural-shrimp-data/audits/cifar-recycled-saved/attempt-001');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('15s cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(seconds=time.monotonic()-start,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=H.sha(Path(__file__)),helper_sha256=H.sha(ROOT/'tools/reviews/audit_cifar_damped_saved.py'));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(bytes=(out/'result.json').stat().st_size,sha256=H.sha(out/'result.json'))}},indent=2)+'\n')
 for f in out.iterdir():f.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
