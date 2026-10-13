"""Small synthetic pilot tests only; no actual seed, large matrices or model."""
import copy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import gate_budget as driver
import derivative_evidence as evidence
import phase_domain as domain


def read(path):return json.loads(Path(path).read_text())
def write(path,data):driver.write_json(Path(path),data)


def seal(attempt):
    source=[dict(path=str(p.relative_to(attempt/'source')),bytes=p.stat().st_size,sha256=driver.sha(p))
        for p in sorted((attempt/'source').rglob('*'))if p.is_file()]
    write(attempt/'manifest.json',dict(source_commit='a'*40,source=source))
    write(attempt/'receipt.json',dict(source_commit='a'*40,exit_code=0,timed_out=False,
        artifacts=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=driver.sha(p))
        for p in sorted(attempt.rglob('*'))if p.is_file()and p.name!='receipt.json']))
    return driver.sha(attempt/'receipt.json')


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'PERIOD',2);monkeypatch.setattr(driver,'DIMENSION',2)
    owned=tmp_path/'owned';owned.mkdir();monkeypatch.setattr(driver,'OWNED',owned);monkeypatch.setattr(evidence,'OWNED',owned)
    attempt=owned/'seed';out=attempt/'run';out.mkdir(parents=True)
    for name in ('codex/certify-llm/center_seed.py','codex/crosscheck/qwen_map.py'):
        path=attempt/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    A=np.array([[[.5,0.],[0.,.25]],[[.25,0.],[0.,.5]]],dtype=np.float64);B=np.eye(2,dtype=np.float64)
    # These unneeded object members cannot be decoded with allow_pickle=False.
    # A correct proposal-only reader never touches them.
    unused={name:np.array([None],dtype=object)for name in ('centers','parameters','product','left_inverse_defect','right_inverse_defect')}
    np.savez(out/'preconditioner.npz',A=A,B=B,**unused)
    write(out/'status.json',dict(status='proposal_completed',certificate_passed=False,preconditioner_sha256=driver.sha(out/'preconditioner.npz')))
    receipt=seal(attempt)
    seed=dict(attempt=str(attempt),receipt_sha256=receipt,source_commit='a'*40,preconditioner_sha256=driver.sha(out/'preconditioner.npz'),
        driver_sha256=driver.sha(ROOT/'codex/certify-llm/center_seed.py'),map_source_sha256=driver.sha(ROOT/'codex/crosscheck/qwen_map.py'))
    plan=dict(schema=driver.SCHEMA,period=2,dimension=2,precision=1024,synthetic_only=True,seed=seed,
        recipe=copy.deepcopy(driver.RECIPE),source_sha256={name:driver.sha(ROOT/name)for name in driver.SOURCE_PATHS},envelope=dict(driver.ENVELOPE))
    previous=ctx.threads;ctx.threads=1
    with ctx.workprec(1024):yield plan,owned
    ctx.threads=previous


def test_only_A_B_are_decoded_and_original_bytes_unchanged(fixture):
    plan,_=fixture;before=driver.sha(Path(plan['seed']['attempt'])/'run/preconditioner.npz')
    A,B,report=driver.load_proposals(plan)
    assert A.shape==(2,2,2)and B.shape==(2,2)
    assert not A.flags.writeable and not B.flags.writeable
    assert report['decoded_npz_members']==['A','B']and not report['target_centers_or_images_decoded']
    assert driver.sha(Path(plan['seed']['attempt'])/'run/preconditioner.npz')==before


@pytest.mark.parametrize('value,sign',[(.5,1),(-.25,-1),(0.,1),(-0.,-1),(float.fromhex('0x1.0000000000001p-3'),1)])
def test_full_mantissa_perturbation_is_exact(value,sign):
    with ctx.workprec(1024):
        x=driver.perturbed_point(value,sign);mid,rad=domain._record_parts(domain.pack_ball(x))
        assert x.is_exact()and not rad
        assert mid==Fraction(*value.as_integer_ratio())+sign*Fraction(1,2**1000)
        assert abs(domain._pair(mid)[0]).bit_length()<=1024


@pytest.mark.parametrize('value,sign',[(float('nan'),1),(float('inf'),1),(float(2**30),1),(1.,True),(1.,0)])
def test_nonfinite_or_too_many_exact_bits_refused(value,sign):
    with pytest.raises(ValueError):driver.perturbed_point(value,sign)


def test_real_tiny_full_gate_and_serialization_are_unmistakably_synthetic(fixture,tmp_path):
    plan,_=fixture;out=tmp_path/'results';out.mkdir();rows=[]
    report=driver.run(plan,out,lambda name,**kw:rows.append(dict(event=name,**kw)))
    assert report['status']=='synthetic_gate_budget_completed'and report['synthetic_only']
    assert report['certificate_passed']is False and report['model_evaluations']==0
    assert not report['measured_upper_bound_for_target']and not report['target_centers_or_images_decoded']
    saved=read(out/'synthetic-gates.json')
    assert saved['synthetic_only']and saved['certificate_passed']is False
    assert saved['conditional_synthetic_result']['certificate_passed']is False
    assert len(saved['conditional_synthetic_result']['attraction']['powers'])==2
    assert report['report_sha256']==driver.sha(out/'synthetic-gates.json')
    assert report['report_bytes']==(out/'synthetic-gates.json').stat().st_size
    assert len(list(out.glob('synthetic-phase-*.json')))==2
    assert read(out/'base-proposals.json')['sha256']==driver.sha(out/'base-proposals.npz')
    assert [r['event']for r in rows][-2:]==['full_report_serialization_started','full_report_serialization_finished']
    inputs=read(out/'synthetic-inputs.json')
    assert inputs['centers']==[[domain.pack_ball(arb(0))]*2,[domain.pack_ball(arb(4))]*2]
    assert all(v['maximum_proposal_mantissa_bits']>=998 for v in inputs['phase_records'])


def test_arbitrary_failed_conditional_gate_is_retained_not_resource_failure(fixture,tmp_path,monkeypatch):
    plan,_=fixture;original=driver.load_proposals
    def unstable(p):
        A,B,report=original(p);return A*4,B,report
    monkeypatch.setattr(driver,'load_proposals',unstable)
    out=tmp_path/'results';out.mkdir();report=driver.run(plan,out,lambda *a,**k:None)
    assert report['status']=='synthetic_gate_budget_completed'
    assert report['synthetic_conditional_gates_passed']is False
    assert not read(out/'synthetic-gates.json')['conditional_synthetic_result']['attraction']['passed']


@pytest.mark.parametrize('fault',['period_bool','dimension','precision','synthetic','recipe','recipe_bool','envelope_bool',
    'envelope_float','timeout','source_missing','source_bad','source_bytes','seed_hash','seed_source','seed_npz',
    'seed_driver','seed_map','unknown','outside'])
def test_frozen_scope_and_trust_anchor_refusals(fixture,fault,tmp_path):
    plan,_=fixture
    if fault=='period_bool':plan['period']=True
    if fault=='dimension':plan['dimension']=3
    if fault=='precision':plan['precision']=2048
    if fault=='synthetic':plan['synthetic_only']=False
    if fault=='recipe':plan['recipe']['jacobian_radius_power']=-100
    if fault=='recipe_bool':plan['recipe']['powers']=[True,2]
    if fault=='envelope_bool':plan['envelope']['cpu_threads']=True
    if fault=='envelope_float':plan['envelope']['nice']=10.0
    if fault=='timeout':plan['envelope']['timeout_seconds']=3600
    if fault=='source_missing':plan['source_sha256'].pop(next(iter(plan['source_sha256'])))
    if fault=='source_bad':plan['source_sha256'][next(iter(plan['source_sha256']))]='x'*64
    if fault=='source_bytes':plan['source_sha256'][next(iter(plan['source_sha256']))]='0'*64
    if fault=='seed_hash':plan['seed']['receipt_sha256']='0'*64
    if fault=='seed_source':plan['seed']['source_commit']='0'*40
    if fault=='seed_npz':plan['seed']['preconditioner_sha256']='0'*64
    if fault=='seed_driver':plan['seed']['driver_sha256']='0'*64
    if fault=='seed_map':plan['seed']['map_source_sha256']='0'*64
    if fault=='unknown':plan['new_field']=True
    if fault=='outside':plan['seed']['attempt']=str(tmp_path)
    with pytest.raises(ValueError):driver.load_proposals(plan)


def test_gate_failure_keeps_completed_construction_evidence(fixture,tmp_path,monkeypatch):
    plan,_=fixture;out=tmp_path/'results';out.mkdir()
    def failed(*a,**k):raise MemoryError('synthetic failure')
    monkeypatch.setattr(driver,'measure_gate',failed)
    with pytest.raises(MemoryError):driver.run(plan,out,lambda *a,**k:None)
    assert(out/'synthetic-inputs.json').is_file()and(out/'base-proposals.npz').is_file()
    assert len(list(out.glob('synthetic-phase-*.json')))==2
    assert not(out/'synthetic-gates.json').exists()


def test_json_write_retains_partial_on_encoding_failure(tmp_path):
    path=tmp_path/'data.json'
    with pytest.raises(ValueError):driver.write_json(path,dict(first=[1,2],bad=float('nan')))
    assert not path.exists()and path.with_suffix('.json.partial').is_file()


def test_isolated_help_does_not_import_numerical_libraries():
    run=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/gate_budget.py'),'--help'],capture_output=True,text=True)
    assert run.returncode==0 and 'synthetic' in run.stdout.lower()


@pytest.mark.parametrize('args',[
    ['--driver','gate_budget','--seconds','1800'],
    ['--driver','gate_budget','--seconds','3600','--plan','example.json'],
    ['--driver','gate_budget','--seconds','1800','--plan','example.json','--threads','2'],
    ['--driver','gate_budget','--seconds','1800','--plan','example.json','--skip-box'],
    ['--driver','gate_budget','--seconds','1800','--plan','../example.json'],
    ['--driver','center_continue','--seconds','3600','--plan','example.json']])
def test_invalid_launcher_scope_refused_before_git(monkeypatch,args):
    spec=importlib.util.spec_from_file_location('gate_launch_test',ROOT/'codex/certify-llm/launch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(sys,'argv',['launch.py','--attempt','synthetic-test',*args])
    monkeypatch.setattr(mod.subprocess,'check_output',lambda *a,**k:pytest.fail('Git called before invalid scope refused'))
    with pytest.raises(ValueError):mod.main()


@pytest.mark.parametrize('name',['gate_budget','center_continue'])
def test_valid_new_and_existing_launcher_admission_stops_before_git(monkeypatch,name):
    spec=importlib.util.spec_from_file_location('gate_launch_test',ROOT/'codex/certify-llm/launch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(sys,'argv',['launch.py','--attempt','synthetic-test','--driver',name,'--seconds','1800','--plan','example.json'])
    class ReachedGit(Exception):pass
    def stop(*a,**k):raise ReachedGit
    monkeypatch.setattr(mod.subprocess,'check_output',stop)
    with pytest.raises(ReachedGit):mod.main()


@pytest.mark.parametrize('fault',[None,'failure','termination'])
def test_main_tiny_orchestration_complete_or_partial_status(fixture,tmp_path,monkeypatch,fault):
    import signal
    plan,owned=fixture;plan_path=tmp_path/'synthetic-plan.json';write(plan_path,plan)
    monkeypatch.setattr(driver,'HERE',tmp_path)
    monkeypatch.setattr(sys,'argv',['gate_budget.py','--output',str(owned/'new'/'run'),'--plan',str(plan_path)])
    monkeypatch.setenv('CERT_SOURCE_COMMIT','b'*40)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE'):
        monkeypatch.setenv(key,os.environ.get(key,''))
    monkeypatch.setattr(driver.os,'getpriority',lambda *args:10)
    monkeypatch.setattr(driver.resource,'getrlimit',lambda *args:(7*1024**3,9*1024**3))
    caps=[];monkeypatch.setattr(driver.resource,'setrlimit',lambda key,value:caps.append(value))
    if fault:
        def fail(*a,**kw):
            if fault=='termination':signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            raise MemoryError('synthetic memory failure')
        monkeypatch.setattr(driver,'measure_gate',fail)
    old=signal.getsignal(signal.SIGTERM)
    if fault:
        with pytest.raises((MemoryError,TimeoutError)):driver.main()
    else:driver.main()
    assert signal.getsignal(signal.SIGTERM)==old
    out=owned/'new'/'run';status=read(out/'status.json')
    assert status['synthetic_only']and status['certificate_passed']is False
    assert caps==[(7*1024**3,7*1024**3)]
    assert(out/'synthetic-inputs.json').exists()
    if fault:
        assert status['status']=='synthetic_gate_budget_failed'and status['phase']=='synthetic_inputs_finished'
        assert not(out/'synthetic-gates.json').exists()
    else:
        assert status['status']=='synthetic_gate_budget_completed'
        assert read(out/'inputs.json')['effective_address_space_cap']==7*1024**3
        rows=[json.loads(x)for x in (out/'events.jsonl').read_text().splitlines()]
        assert rows[-1]['event']=='completed'and all(r['synthetic_only']and not r['certificate_passed']for r in rows)


def test_B_is_exact_binary64_and_input_arrays_not_mutated(fixture,tmp_path):
    plan,_=fixture;A,B,_=driver.load_proposals(plan);beforeA=A.copy();beforeB=B.copy();out=tmp_path/'build';out.mkdir()
    args,_=driver.build_synthetic(A,B,out,lambda *a,**kw:None)
    for i in range(2):
        for j in range(2):
            mid,rad=domain._record_parts(domain.pack_ball(args[-1][i,j]))
            assert not rad and mid==Fraction(*float(B[i,j]).as_integer_ratio())
    assert np.array_equal(A,beforeA)and np.array_equal(B,beforeB)
