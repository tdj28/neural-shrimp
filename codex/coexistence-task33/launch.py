"""Snapshot a privately pushed certificate pilot and bound its owned process."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
ROOT=Path('/data2/neural-shrimp-data/codex/coexistence')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def ensure_nice10():
    """Lower our priority to the exact envelope once; never add ten twice."""
    current=os.getpriority(os.PRIO_PROCESS,0)
    if current>10:raise RuntimeError('exact nice10 required; refusing stronger inherited nice value')
    if current<10:os.nice(10-current)
    if os.getpriority(os.PRIO_PROCESS,0)!=10:raise RuntimeError('could not establish exact nice10')

def ensure_launcher_address_space():
    """Give Git/snapshot work the same existing bound as the future worker."""
    limit=18*1024**3
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<limit:
        raise RuntimeError('inherited hard address-space limit is below 18GiB')
    resource.setrlimit(resource.RLIMIT_AS,(limit,limit))

def main():
    # Establish the policy before Git/source-snapshot subprocesses as well.
    ensure_nice10()
    ensure_launcher_address_space()
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--precision',type=int,default=1024)
    p.add_argument('--driver',choices=['derivative_shard'],default='derivative_shard')
    p.add_argument('--plan')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--skip-box',action='store_true');a=p.parse_args()
    if not 1<=a.threads<=4 or a.driver!='center_seed' and a.threads!=1:raise ValueError('only numerical seed driver may use up to four CPU threads')
    if a.skip_box and a.driver!='pilot':raise ValueError('skip-box applies only to original point pilot')
    if (a.driver in ('signed_pilot','center_refine','center_resume','box_derivative_pilot','streamed_pilot','proposal_budget','center_secant','derivative_shard','center_continue','gate_budget','assembly')) != bool(a.plan):raise ValueError('signed/refinement/derivative pilots require an explicit committed plan; other drivers accept none')
    if a.driver=='derivative_shard' and a.seconds!=3600:raise ValueError('final derivative shard requires exactly a 3600-second watchdog')
    if a.driver=='assembly' and a.seconds!=3600:raise ValueError('final conditional assembly requires exactly a 3600-second watchdog')
    if a.driver in ('center_secant','center_continue') and a.seconds!=1800:raise ValueError('one-candidate secant requires exactly an 1800-second watchdog')
    if a.driver=='gate_budget' and a.seconds!=1800:raise ValueError('synthetic gate budget requires exactly an 1800-second watchdog')
    if a.driver=='proposal_budget' and a.seconds!=120:raise ValueError('proposal budget requires exactly a 120-second watchdog')
    if a.plan and (Path(a.plan).name!=a.plan or not a.plan.endswith('.json')):raise ValueError('plan must name a committed certificate JSON file')
    if not a.attempt or '/' in a.attempt or a.attempt.startswith('.') or not 1<=a.seconds<=3600 or not 128<=a.precision<=2048:
        raise ValueError('invalid owned attempt or resource envelope')
    repo=Path(__file__).resolve().parents[2]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    head=git('rev-parse','HEAD')
    files=git('ls-tree','-r','--name-only',head,'codex/certify-llm','codex/coexistence-task33','codex/crosscheck/qwen_map.py').splitlines()
    blobs={}
    for name in files:
        if not name.endswith(('.py','.md','.json')):continue
        data=subprocess.check_output(['git','show',head+':'+name],cwd=repo)
        if (repo/name).read_bytes()!=data:raise ValueError('uncommitted source '+name)
        blobs[name]=data
    if 'codex/certify-llm/pilot.py' not in blobs:raise ValueError('pilot not committed')
    subprocess.run(['git','fetch','origin','main'],cwd=repo,check=True)
    subprocess.run(['git','merge-base','--is-ancestor',head,'origin/main'],cwd=repo,check=True)
    attempt=ROOT/a.attempt;attempt.mkdir(parents=True,exist_ok=False)
    source=attempt/'source'
    for name,data in blobs.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    driver_name='codex/coexistence-task33/'+a.driver+'.py'
    if driver_name not in blobs:raise ValueError('driver not committed')
    command=[sys.executable,'-I','-B','-u',str(source/driver_name),'--output',str(attempt/'run')]
    if a.driver=='pilot':
        command.extend(['--precision',str(a.precision),'--threads','1'])
        if a.skip_box:command.append('--skip-box')
    elif a.driver=='center_seed':command.extend(['--threads',str(a.threads)])
    else:
        plan_name='codex/coexistence-task33/'+a.plan
        if plan_name not in blobs:raise ValueError('plan not committed')
        command.extend(['--plan',str(source/plan_name)])
    manifest=dict(source_commit=head,remote_main=git('rev-parse','origin/main'),remote_ancestry_verified_at=datetime.now(timezone.utc).isoformat(),
                  command=command,threads=a.threads,memory_gib=18,timeout_seconds=a.seconds,
                  source=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()])
    if a.driver=='center_continue':
        manifest.update(parallel_workers=4,worker_threads=1,aggregate_rss_watchdog_gib=32,
            sampled_rss_is_hard_cap=False,minimum_available_gib=12,worker_timeout_seconds=600)
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),NUMEXPR_NUM_THREADS=str(a.threads),
            CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CERT_SOURCE_COMMIT=head)
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    def limits():
        ensure_nice10();resource.setrlimit(resource.RLIMIT_AS,(18*1024**3,)*2)
    start=time.monotonic();timed_out=False
    with (attempt/'stdout.log').open('wb') as out,(attempt/'stderr.log').open('wb') as err:
        child=subprocess.Popen(command,cwd=source,env=env,stdout=out,stderr=err,preexec_fn=limits,start_new_session=True)
        manifest['owned_pid']=child.pid;(attempt/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(dict(attempt=str(attempt),pid=child.pid,source_commit=head)),flush=True)
        try:rc=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:rc=child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                rc=child.wait()
    receipt=dict(source_commit=head,exit_code=rc,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,
                 artifacts=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(attempt.rglob('*')) if p.is_file()])
    (attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for p in attempt.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'}),flush=True)
    sys.exit(rc if rc>=0 else 1)

if __name__=='__main__':main()
