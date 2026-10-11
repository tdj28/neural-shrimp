"""Post-outcome saved-only arm identities/dot products; no model derivatives."""
import hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
from neural_shrimp.period2_branch_io import load_record,identity
ROOT=Path(__file__).resolve().parents[2];RAW=Path('/data2/neural-shrimp-data/explore/utility-common-center/attempt-001');PIN='739737a4db7d153e546b717b68bdb64ea704750dd5d55e60229a8b34054deb35'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
 need(sha(RAW/'manifest.json')==PIN,'external pin');m=json.loads((RAW/'manifest.json').read_text())['files'];need(set(m)=={str(p.relative_to(RAW))for p in RAW.rglob('*')if p.is_file()and p!=RAW/'manifest.json'},'exactinventory')
 for n,d in m.items():
  p=RAW/n;need(not p.is_symlink()and sha(p)==d['sha256']and p.stat().st_size==d['bytes']and not p.stat().st_mode&0o222,'inputbytes')
 run=RAW/'run';rt=json.loads((run/'runtime-start.json').read_text());la=json.loads((RAW/'launch-end.json').read_text());r=json.loads((run/'result.json').read_text());need(rt['source_commit']==la['source_commit']=='35d5d9c12654b2f66dee35ced6f550eadb79797d','execution');need(la['exit_code']==1 and la['owned_group_empty']and all(not Path('/proc/'+str(la[k])).exists()for k in ('launcher_pid','child_pid')),'preserved process failure/exits')
 for n,d in rt['sources'].items():need(hashlib.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+n],cwd=ROOT)).hexdigest()==d['sha256'],'historicalsource')
 journal=[json.loads(x)for x in (run/'journal.jsonl').read_text().splitlines()];need(len(journal)==34 and [j['sequence']for j in journal]==list(range(34)),'complete eventsequence');stages=[load_record(run,j['file'])for j in journal];need([k for k,v in stages]==['synthetic_profile','bound_inputs']+['paired_states','operation_start','operation_return','operation_start','operation_return','operation_start','operation_return','contrast_reduction']*4,'exact four operation triples');need(r['status']=='completed'and r['counts']==dict(gradient_attempts=4,gradient_returns=4,loss_attempts=8,loss_returns=8),'complete scientific returns')
 rows=[]
 for i in range(4):
  group=stages[2+8*i:10+8*i];state=group[0][1];need(all(v['index']==i for k,v in group),'fixedrow');t,g,z,plus,minus,v=(state[k]for k in ['original','original_gradient','center','plus','minus','contrast']);rates=state['rates'];s=np.r_[rates[0]*g[:9],rates[1]*g[9:]];need(np.array_equal(z,t-(7/8)*s)and np.array_equal(plus,z+v)and np.array_equal(minus,z-v),'actual common center arms');need(np.array_equal(group[1][1]['state'],z)and np.array_equal(group[3][1]['state'],plus)and np.array_equal(group[5][1]['state'],minus),'evaluatedstates');gc=group[2][1]['value'];lp=float(group[4][1]['value']);lm=float(group[6][1]['value']);dot=lambda a,b:math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True));delta=math.fsum([lp,-lm]);pred=2*dot(gc,v);old=2*dot(g,v);res=math.fsum([lp,-lm,-pred]);saved=group[7][1]
  for name,value in [('loss_difference_plus_minus',delta),('center_first_order_prediction',pred),('original_state_first_order_prediction',old),('centered_residual',res),('residual_fraction_of_contrast',abs(res)/abs(delta))]:need(saved[name]==r['rows'][i][name]==value,'saved scalar '+name)
  rows.append(dict(index=i,delta=delta,center_prediction=pred,original_prediction=old,residual=res,fraction=abs(res)/abs(delta),center_sign_correct=bool(delta*pred>0),original_sign_correct=bool(delta*old>0)))
 return dict(passed=True,rows=rows,raw_manifest=PIN,scientific_acquisition_complete=True,external_exit_code=1,console_failure='NumPy boolean json.dumps after immutable scientific result and manifest creation',scope='Post-outcome producer-agent independently written arm/dot/fsum reductions; no fresh loss/gradient/model calls, no independent personnel outcome replay; float64 descriptive residual only.')
def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'cleansource');out=Path('/data2/neural-shrimp-data/audits/utility-common-center/attempt-001');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('5s auditcap')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,5)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(seconds=time.monotonic()-start,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=sha(Path(__file__)));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(sha256=sha(out/'result.json'),bytes=(out/'result.json').stat().st_size)}},indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
