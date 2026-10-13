"""New endpoint supervisor delta controls; synthetic statuses/tiny processes only."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import time
import pytest
R=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
L=load('endpoint_supervisor_controls',R/'codex/cusp-crosscheck/endpoint_launch.py')
P=load('unchanged_profile_supervisor',R/'codex/cusp-crosscheck/launch.py')
def args():return argparse.Namespace(attempt='endpoints-001',source='a'*40,plan='endpoints-001-plan.json',plan_sha256='b'*64)
def status(qualified=(True,True)):
 return dict(status='endpoint_check_completed',computational_completed=True,cusp_qualified=False,certificate_passed=False,
  all_endpoints_qualified=all(qualified),rows=[dict(index=i,numerical_qualified=q,status='numerical_endpoint_qualified' if q else 'numerical_endpoint_unqualified',cusp_qualified=False,certificate_passed=False)for i,q in zip([13,14],qualified)])
@pytest.mark.parametrize('qualified',[(True,True),(False,True),(True,False),(False,False)])
def test_scientific_failure_is_valid_computational_completion(qualified):L.validate_endpoint_completion(status(qualified))
@pytest.mark.parametrize('field,value',[('status','running'),('computational_completed',False),('certificate_passed',True),('cusp_qualified',True),('all_endpoints_qualified',False),('rows',[]),('rows',[dict(index=13)])])
def test_partial_or_promoted_status_refused(field,value):
 s=status();s[field]=value
 with pytest.raises(RuntimeError):L.validate_endpoint_completion(s)
@pytest.mark.parametrize('change',[dict(index=14),dict(index=13.0),dict(numerical_qualified=1),dict(status='numerical_endpoint_unqualified'),dict(certificate_passed=True)])
def test_every_endpoint_row_is_checked(change):
 s=status();s['rows'][0].update(change)
 with pytest.raises(RuntimeError):L.validate_endpoint_completion(s)
def test_fixed_plan_and_import_closure():
 spec=L.specification(args().plan)
 assert spec['caps']==dict(P.CAPS,timeout_seconds=6000) and len(spec['workers'])==7
 assert set(P.WORKER_FILES)<set(spec['workers']) and not spec['resource_probe_only']
 for field,value in [('attempt','other'),('plan','profile-001-plan.json')]:
  a=args();setattr(a,field,value)
  with pytest.raises(ValueError):L.validate_arguments(a)
def fixture(tmp_path,monkeypatch):
 a=args();repo=tmp_path/'repo';repo.mkdir();spec=L.specification(a.plan)
 blobs={n:b'# source\n'for n in spec['workers']};blobs[L.OWNERSHIP_FILE]=(R/L.OWNERSHIP_FILE).read_bytes()
 for n in ['endpoint_launch.py',spec['card']]:blobs['codex/cusp-crosscheck/'+n]=b'# exact own source/card\n'
 plan=dict(schema=spec['schema'],admission='reviewed',envelope=spec['caps'],source_sha256={n:L.hashlib.sha256(blobs[n]).hexdigest()for n in spec['workers']})
 rel='codex/cusp-crosscheck/'+a.plan;blobs[rel]=json.dumps(plan).encode();a.plan_sha256=L.hashlib.sha256(blobs[rel]).hexdigest()
 for n,b in blobs.items():p=repo/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 def git(cmd,**kw):
  if cmd[1]=='rev-parse':return (a.source+'\n').encode()
  if cmd[1]=='show':return blobs[cmd[2].split(':',1)[1]]
  if cmd[1]=='ls-remote':return (a.source+'\trefs/heads/main\n').encode()
  if cmd[1]=='merge-base':return b''
  raise AssertionError(cmd)
 monkeypatch.setattr(L.subprocess,'check_output',git);monkeypatch.setattr(L,'available',lambda:48*L.GIB)
 return a,repo,blobs,plan
def test_snapshot_captures_endpoint_self_and_card(tmp_path,monkeypatch):
 a,r,b,p=fixture(tmp_path,monkeypatch);got,captured,remote=L.snapshot(r,a,time.monotonic()+30,[])
 assert got==p and captured==b and remote==a.source
@pytest.mark.parametrize('admission',['prepared',None])
def test_unadmitted_plan_refuses_before_worker(tmp_path,monkeypatch,admission):
 a,r,b,p=fixture(tmp_path,monkeypatch);p['admission']=admission;f=r/'codex/cusp-crosscheck'/a.plan;f.write_text(json.dumps(p));a.plan_sha256=L.sha(f)
 with pytest.raises(ValueError,match='prepared'):L.snapshot(r,a,time.monotonic()+30,[])
@pytest.mark.parametrize('qualified,exit_code',[((True,True),0),((False,True),0),((True,True),3)])
def test_tiny_worker_closure_preserves_scientific_and_operational_outcomes(tmp_path,monkeypatch,qualified,exit_code):
 a=args();monkeypatch.setattr(L,'OWNED',tmp_path/'owned');monkeypatch.setattr(L,'available',lambda:48*L.GIB)
 parent=os.getpid();limits=L.establish_limits;monkeypatch.setattr(L,'establish_limits',lambda:limits() if os.getpid()!=parent else None)
 worker="import argparse,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');a=p.parse_args();o=Path(a.output);o.mkdir();time.sleep(.04)\n(o/'status.json').write_text("+repr(json.dumps(status(qualified)))+")\nraise SystemExit("+str(exit_code)+")\n"
 blobs={L.OWNERSHIP_FILE:(R/L.OWNERSHIP_FILE).read_bytes(),'codex/cusp-crosscheck/endpoint_runtime.py':worker.encode(),'codex/cusp-crosscheck/'+a.plan:b'{}'}
 monkeypatch.setattr(L,'snapshot',lambda *x:({},blobs,a.source))
 assert L.execute(a)==int(exit_code!=0)
 out=L.OWNED/a.attempt;receipt=json.loads((out/'receipt.json').read_text())
 assert receipt['exit_code']==exit_code and receipt['worker_termination_verified'] and not receipt['uncertain_ownership']
 assert receipt['scientific_endpoint_result_available']==(exit_code==0)
 assert not receipt['certificate_passed'] and not receipt['scientific_cusp_result'] and not receipt['resource_probe_only']
 assert receipt['status']==('endpoint_computation_completed' if exit_code==0 else 'endpoint_computation_failed')
 assert (out/'failure.json').exists()==(exit_code!=0)
 assert all((out/x['path']).stat().st_mode&0o222==0 for x in receipt['artifacts'])
