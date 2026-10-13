"""Task7 passage supervisor controls; tiny subprocesses only, no model."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('passage_existing_launcher_controls',ROOT/'tests/test_codex_hostile_fold_launch.py')
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
L=BASE.L


def status(censored=False):
    rows=[dict(id=f'e{i}-{entrance}',status='observed',primary_pass=False)
          for i in range(3) for entrance in ('full','midpoint')]
    if censored:rows[2]['status']='censored'
    return dict(status='passage_completed',computational_completed=True,
        passage_trajectories_evaluated=True,certificate_passed=False,
        resource_probe_only=False,natural_reinjection_measured=False,
        all_rows_observed=not censored,primary_all_pass=False,rows=rows)


def test_passage_exact_import_inventory(tmp_path,monkeypatch):
    original=BASE.args
    def arguments():
        args=original();args.plan='passage-001-plan.json';return args
    monkeypatch.setattr(BASE,'args',arguments)
    a,repo,blobs,plan=BASE.snapshot_fixture(tmp_path,monkeypatch)
    actual,captured,remote=L.snapshot(repo,a,time.monotonic()+10,[])
    assert actual==plan and captured==blobs and len(plan['source_sha256'])==6
    assert plan['schema']=='codex-hostile-passage-v1'
    assert plan['envelope']==dict(L.CAPS,timeout_seconds=1800)
    assert L.specification(a.plan)['worker']=='passage_runtime.py'
    assert L.sha(ROOT/L.OWNERSHIP_FILE)==L.OWNERSHIP_SHA


@pytest.mark.parametrize('censored',[False,True])
def test_successful_computation_preserves_scientific_nonpass(censored):
    result=status(censored);L.validate_passage_completion(result)
    assert not result['primary_all_pass']


@pytest.mark.parametrize('field,value',[
 ('status','fold_qualified'),('computational_completed',False),
 ('passage_trajectories_evaluated',False),('certificate_passed',True),
 ('natural_reinjection_measured',True),('resource_probe_only',True),
 ('primary_all_pass',True),('all_rows_observed',False),('rows',[])])
def test_incomplete_or_promoted_status_refused(field,value):
    result=status();result[field]=value
    with pytest.raises(RuntimeError):L.validate_passage_completion(result)


@pytest.mark.parametrize('change',[
 ('status','running'),('status','failed'),('id','duplicate'),('primary_pass',1)])
def test_every_row_must_be_complete(change):
    result=status();result['rows'][4][change[0]]=change[1]
    with pytest.raises(RuntimeError):L.validate_passage_completion(result)


def test_censoring_cannot_pass():
    result=status(True);result['rows'][2]['primary_pass']=True
    with pytest.raises(RuntimeError):L.validate_passage_completion(result)


@pytest.mark.parametrize('censored',[False,True])
def test_tiny_passage_receipt_and_owned_closure(tmp_path,monkeypatch,censored):
    a=BASE.execute_fixture(tmp_path,monkeypatch);a.plan='passage-001-plan.json'
    worker="""import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');a=p.parse_args()
out=Path(a.output);out.mkdir();time.sleep(.07)
(out/'status.json').write_text(STATUS)
""".replace('STATUS',repr(json.dumps(status(censored))))
    blobs={L.OWNERSHIP_FILE:(ROOT/L.OWNERSHIP_FILE).read_bytes(),
           'codex/hostile-fold-crosscheck/passage_runtime.py':worker.encode(),
           'codex/hostile-fold-crosscheck/'+a.plan:b'{}'}
    monkeypatch.setattr(L,'snapshot',lambda *x:({},blobs,'a'*40))
    assert L.execute(a)==0
    receipt=json.loads((L.OWNED/a.attempt/'receipt.json').read_text())
    assert receipt['status']=='passage_attempt_completed'
    assert receipt['controlled_passage_result_available']
    assert not receipt['scientific_fold_result'] and not receipt['certificate_passed']
    assert receipt['worker_termination_verified'] and receipt['uncertain_ownership']==[]
    assert receipt['resource_envelope']['timeout_seconds']==1800
