"""One externally pinned utility saved-outcome audit; separate launch approval required."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import platform,re,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.period2_branch import PilotStop
from neural_shrimp.period2_branch_io import Store,file_entry
from neural_shrimp.utility_dose_audit import audit_saved,need,STORE_BYTES,TOTAL_BYTES,LAUNCHER_BYTES
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/utility_dose_audit.py','tools/audit_utility_dose_feasibility.py','tools/launch_utility_dose_audit.py','tests/test_utility_dose_outcome_audit.py','docs/plans/utility-dose-outcome-audit.md')
OUT=Path('/data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001')
def sources():
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit checkout')
    inventory={}
    for name in SOURCES:
        need((ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed audit source '+name);inventory[name]=file_entry(ROOT/name)
    return inventory

def run(out,execution,raw_pin,launcher_pin,review):
    need(out.resolve()==OUT,'exclusive fixed audit path')
    for value,n in [(execution,40),(review,40),(raw_pin,64),(launcher_pin,64)]:need(re.fullmatch('[0-9a-f]{'+str(n)+'}',value) is not None,'full external identities')
    subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True)
    inventory=sources();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 audit')
    store=Store(out,STORE_BYTES);start=time.monotonic();stopped=[False];old=signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]:raise PilotStop('external_signal')
        if time.monotonic()-start>=5:raise PilotStop('deadline')
    result=dict(schema='utility-dose-independent-outcome-v1',status='running',reference_gradient_attempts=0,reference_gradient_returns=0,fixed_candidate_feasible=None,
      scope=dict(model_updates=0,trajectories=0,controls_acquired=0,hvps=0,pulses=0,protected_data=False))
    try:
        store.json('runtime-start.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=inventory,audit_review_commit=review,producer_execution_commit=execution,raw_manifest_sha256=raw_pin,launcher_manifest_sha256=launcher_pin,started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},python=platform.python_version(),numpy=np.__version__,load_average=os.getloadavg(),limits=dict(internal_seconds=5,external_seconds=10,kill_grace_seconds=5,max_reference_gradients=4,total_bytes=TOTAL_BYTES,store_bytes=STORE_BYTES,launcher_bytes=LAUNCHER_BYTES)))
        outcome=audit_saved(raw_pin,launcher_pin,execution,store,check,result);check();result.update(outcome)
    except PilotStop as error:result.update(status='audit_resource_stop',reason=str(error))
    except BaseException as error:result.update(status='audit_failed_or_interrupted',reason=type(error).__name__+': '+str(error))
    finally:
        signal.signal(signal.SIGTERM,old);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        store.finish(result)
        for p in store.root.iterdir():p.chmod(0o444)
        store.root.chmod(0o555)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--execution-commit',required=True);p.add_argument('--raw-manifest',required=True);p.add_argument('--launcher-manifest',required=True);p.add_argument('--review-commit',required=True);a=p.parse_args()
    r=run(a.out,a.execution_commit,a.raw_manifest,a.launcher_manifest,a.review_commit);print(r);raise SystemExit(0 if r['status']=='audit_passed' else 2)
