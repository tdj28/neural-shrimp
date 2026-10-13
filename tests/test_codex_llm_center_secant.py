"""One-candidate secant driver tests with tiny complete synthetic histories."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from flint import arb,arb_mat,ctx
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
sys.path.insert(0,str(ROOT/'tests'))
import center_secant as driver
import test_codex_llm_center_resume as fixtures
resume=driver.resume;base=driver.base


@pytest.fixture(autouse=True)
def tiny_scope(monkeypatch):
    monkeypatch.setattr(base,'PERIOD',2);monkeypatch.setattr(base,'DIMENSION',2)
    old=ctx.threads;ctx.threads=1
    with ctx.workprec(2048):yield
    ctx.threads=old


def plan_from_parent(parent_plan,count):
    return dict(schema=driver.SCHEMA,parent_admission_plan=copy.deepcopy(parent_plan),
        method_source_sha256={name:base.digest(ROOT/name) for name in driver.METHOD_SOURCES},
        precision=2048,history_tuple_count=count,max_candidates=1,
        caps=dict(driver.CAPS),envelope=dict(driver.ENVELOPE))


def fixture(tmp_path,monkeypatch,*,resumed=False):
    owned,pp,assets,binding,A,B=fixtures.prepare_base(tmp_path,monkeypatch)
    monkeypatch.setattr(driver,'OWNED',owned)
    if resumed:
        pp['max_corrections']=2
        inherited=resume.load_completed_parent(pp,assets,(1,2),binding=binding)
        attempt=owned/'resume-01';out=attempt/'run';out.mkdir(parents=True)
        iterations=[]
        resume.run_corrections(fixtures.TinyMap(),inherited['centers'],inherited['images'],A,B,out,2,
                              lambda *a,**k:None,iterations)
        rows=[dict(correction_index=v['correction_index'],**r) for v in iterations for r in v['phase_outcomes']]
        fixtures.write(out/'status.json',dict(status='proposal_resume_completed',certificate_passed=False,
                       corrections=2,iterations=iterations,phase_outcomes=rows))
        fixtures.write(out/'reuse-manifest.json',dict(parent_attempt=pp['parent_attempt'],parent_receipt_sha256=pp['parent_receipt_sha256'],
            phases=inherited['reuse'],lineage=inherited['lineage'],exact_centers=resume.center_records(inherited['centers']),
            cumulative_prior_phase_evaluations=inherited['cumulative_phase_evaluations'],
            cumulative_prior_elapsed_seconds=inherited['cumulative_elapsed_seconds'],residual_recomputed=True,no_image_radius_discarded=True))
        hashes=fixtures.source_hashes(attempt,'center_resume')
        fixtures.shared_metadata(attempt,pp,hashes,'center_resume',assets,binding)
        pp=dict(schema=resume.SCHEMA,base_plan=pp['base_plan'],parent_attempt=str(attempt),parent_kind='center_resume',
            parent_source_sha256=hashes,parent_receipt_sha256=fixtures.seal(attempt),max_corrections=1,envelope=pp['envelope'])
    return owned,plan_from_parent(pp,4 if resumed else 2),assets,binding,A,B


@pytest.mark.parametrize('resumed',[False,True])
def test_full_history_admission_keeps_every_tuple_and_full_precision(tmp_path,monkeypatch,resumed):
    _,plan,assets,binding,_,_=fixture(tmp_path,monkeypatch,resumed=resumed)
    states,inherited=driver.load_history(plan,assets,(1,2),binding=binding)
    assert len(states)==(4 if resumed else 2)
    assert inherited['cumulative_phase_evaluations']==2*len(states)
    assert [s['record']['sequence_index'] for s in states]==list(range(len(states)))
    assert [s['record']['evaluation_pass'] for s in states]==['before']+['after']*(len(states)-1)
    assert all(v.is_exact() and v==arb(1)+arb((1,-1700)) for s in states[1:] for row in s['centers'] for v in row)
    assert all(s['record']['residual_recomputed'] and s['record']['no_image_radius_discarded'] for s in states)
    for state in states:
        for i in range(2):
            for j in range(2):
                assert state['residuals'][i][j].overlaps(state['images'][i][j]-state['centers'][1-i][j])
                assert len(state['record']['phase_artifacts'][i]['stored_image'])==2


@pytest.mark.parametrize('fault',['history_count','wrong_kernel','wrong_importer','partial_status','failed_receipt','wrong_image_input'])
def test_history_admission_refuses_tamper_and_incomplete_parents(tmp_path,monkeypatch,fault):
    _,plan,assets,binding,_,_=fixture(tmp_path,monkeypatch)
    pp=plan['parent_admission_plan'];attempt=Path(pp['parent_attempt']);out=attempt/'run'
    if fault=='history_count':plan['history_tuple_count']=3
    if fault=='wrong_kernel':plan['method_source_sha256'][driver.METHOD_SOURCES[1]]='a'*64
    if fault=='wrong_importer':plan['method_source_sha256'][driver.METHOD_SOURCES[0]]='a'*64
    if fault=='partial_status':
        status=resume.read_json(out/'status.json');status['status']='proposal_setup_failed';fixtures.write(out/'status.json',status)
        pp['parent_receipt_sha256']=fixtures.seal(attempt)
    if fault=='failed_receipt':
        receipt=resume.read_json(attempt/'receipt.json');receipt['exit_code']=1;fixtures.write(attempt/'receipt.json',receipt)
        pp['parent_receipt_sha256']=base.digest(attempt/'receipt.json')
    if fault=='wrong_image_input':
        path=out/'before-phase-00-input.json';value=resume.read_json(path);value['center'][0]=base.packed(arb(99));fixtures.write(path,value)
        rows=resume.read_json(out/'phase-outcomes.json');rows[0]['input_sha256']=base.digest(path);fixtures.write(out/'phase-outcomes.json',rows)
        pp['parent_receipt_sha256']=fixtures.seal(attempt)
    with pytest.raises(ValueError):driver.load_history(plan,assets,(1,2),binding=binding)


@pytest.mark.parametrize('fault',['schema','extra','missing','precision','candidates_bool','two_candidates','count_bool','count_large',
                                 'threads_bool','seconds','caps_bool','caps_value','secant_parent','missing_pin'])
def test_invalid_plan_refused(tmp_path,monkeypatch,fault):
    _,plan,_,_,_,_=fixture(tmp_path,monkeypatch)
    if fault=='schema':plan['schema']='other'
    if fault=='extra':plan['retry']=True
    if fault=='missing':plan.pop('caps')
    if fault=='precision':plan['precision']=1024
    if fault=='candidates_bool':plan['max_candidates']=True
    if fault=='two_candidates':plan['max_candidates']=2
    if fault=='count_bool':plan['history_tuple_count']=True
    if fault=='count_large':plan['history_tuple_count']=17
    if fault=='threads_bool':plan['envelope']['cpu_threads']=True
    if fault=='seconds':plan['envelope']['timeout_seconds']=1801
    if fault=='caps_bool':plan['caps']['step_ratio_limit']=True
    if fault=='caps_value':plan['caps']['signal_ratio']=64
    if fault=='secant_parent':plan['parent_admission_plan']['parent_kind']='center_secant'
    if fault=='missing_pin':plan['method_source_sha256'].pop(driver.METHOD_SOURCES[0])
    with pytest.raises(ValueError):driver.validate_plan(plan)


def test_every_replay_decision_saved_and_one_proposal_only(tmp_path,monkeypatch):
    _,plan,assets,binding,A,B=fixture(tmp_path,monkeypatch,resumed=True)
    states,_=driver.load_history(plan,assets,(1,2),binding=binding)
    out=tmp_path/'new';out.mkdir();events=[]
    proposal,attempts=driver.prepare_candidate(states,A,B,out,lambda *a,**k:events.append((a,k)))
    assert len(attempts)==3 and all(r['status']=='completed' for r in attempts)
    assert [r['reason'] for r in attempts][-2:]==['zero_center_step','zero_center_step']
    assert len(list(out.glob('secant-*.json')))==4  # Three decisions plus outcomes.
    assert (out/'proposal.json').is_file() and (out/'centers-after.json').is_file()
    assert resume.read_json(out/'preconditioner-guard.json')['original_exact_matrices_unchanged']
    assert proposal['record']['reason']=='baseline_no_history'
    assert len([e for e in events if e[0][0]=='single_candidate_saved'])==1
    assert A[0]==arb_mat(2,2) and B==arb_mat([[1,0],[0,1]])


def test_mutating_kernel_is_detected_and_cannot_change_original_AB(tmp_path,monkeypatch):
    _,plan,assets,binding,A,B=fixture(tmp_path,monkeypatch)
    states,_=driver.load_history(plan,assets,(1,2),binding=binding)
    real=driver.InverseSecant
    class BadKernel(real):
        def update(self,*args,**kwargs):
            value=super().update(*args,**kwargs);self.A[0][0,0]=99;return value
    monkeypatch.setattr(driver,'InverseSecant',BadKernel)
    out=tmp_path/'new';out.mkdir()
    with pytest.raises(ValueError,match='preconditioner changed'):driver.prepare_candidate(states,A,B,out,lambda *a,**k:None)
    assert A[0][0,0]==0 and B==arb_mat([[1,0],[0,1]])
    assert (out/'secant-01.json').is_file() and not (out/'centers-after.json').exists()


def test_interrupted_phase_pass_preserves_candidate_and_prior_images(tmp_path,monkeypatch):
    _,plan,assets,binding,A,B=fixture(tmp_path,monkeypatch)
    states,_=driver.load_history(plan,assets,(1,2),binding=binding)
    out=tmp_path/'new';out.mkdir();proposal,_=driver.prepare_candidate(states,A,B,out,lambda *a,**k:None)
    outcomes=[]
    with pytest.raises(ValueError,match='interrupted'):
        driver.evaluate_candidate(fixtures.TinyMap(fail_at=2),proposal,states[-1],out,lambda *a,**k:None,outcomes)
    assert (out/'proposal.json').is_file() and (out/'after-phase-00-point.json').is_file()
    assert (out/'after-phase-01-input.json').is_file() and not (out/'after-phase-01-point.json').exists()
    assert len(outcomes)==2 and outcomes[0]['status']=='completed'


def test_nonimprovement_is_retained_without_a_retry(tmp_path,monkeypatch):
    _,plan,assets,binding,A,B=fixture(tmp_path,monkeypatch,resumed=True)
    states,_=driver.load_history(plan,assets,(1,2),binding=binding)
    out=tmp_path/'new';out.mkdir();proposal,_=driver.prepare_candidate(states,A,B,out,lambda *a,**k:None)
    class WorseMap:
        calls=0
        def forward(self,center):
            self.calls+=1;return [arb(100)]*2,[arb(0)]*2
    model=WorseMap();outcomes=[]
    comparison=driver.evaluate_candidate(model,proposal,states[-1],out,lambda *a,**k:None,outcomes)
    assert model.calls==2 and len(outcomes)==2
    assert comparison['upper_bound_decreased'] is False and comparison['strict_residual_norm_decrease_established'] is False
    assert (out/'residual-after.json').is_file() and (out/'residual-comparison.json').is_file()


@pytest.mark.parametrize('fails',[False,True])
def test_complete_fake_driver_runs_one_pass_and_preserves_parent(tmp_path,monkeypatch,fails):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'));import qwen_map
    owned,plan,assets,binding,_,_=fixture(tmp_path,monkeypatch,resumed=True)
    parent=Path(plan['parent_admission_plan']['parent_attempt'])
    before={str(p):base.digest(p) for p in owned.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();path=source/'plan.json';fixtures.write(path,plan)
    out=owned/'new-secant'/'run';calls=[]
    model=SimpleNamespace(config={'synthetic':True},weights={'w':torch.zeros(1,dtype=torch.float64)},
        vector=torch.zeros(2,dtype=torch.float64),steering_layer=12)
    class FakeBall(fixtures.TinyMap):
        def __init__(self,*a,**k):super().__init__(fail_at=2 if fails else None)
        def forward(self,center):calls.append(list(center));return super().forward(center)
    monkeypatch.setattr(driver,'HERE',source)
    monkeypatch.setattr(base,'load_bound_inputs',lambda *a:({'A':np.zeros((2,2,2)),'B':np.eye(2)},binding))
    monkeypatch.setattr(base,'BallQwen',FakeBall)
    monkeypatch.setattr(base,'load_prefix',lambda *a,**k:resume.read_json(parent/'run/prefix-binding.json'))
    monkeypatch.setattr(qwen_map,'PINNED_SHA256',assets)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda:model)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(os,'getpriority',lambda *a:10)
    monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['center_secant','--output',str(out),'--plan',str(path)])
    if fails:
        with pytest.raises(ValueError,match='interrupted'):driver.main()
    else:driver.main()
    assert {name:base.digest(name) for name in before}==before
    status=resume.read_json(out/'status.json')
    assert status['certificate_passed'] is False and len(calls)==2
    assert all(v.is_exact() and v==arb(1)+arb((1,-1700)) for row in calls for v in row)
    assert (out/'proposal.json').is_file() and len(resume.read_json(out/'secant-outcomes.json'))==3
    if fails:
        assert status['status']=='proposal_secant_failed' and status['phase_outcomes'][-1]['status']=='failed'
    else:
        assert status['status']=='proposal_secant_completed' and status['candidates']==1
        assert status['new_phase_evaluations']==2 and status['cumulative_phase_evaluations']==10


def test_parent_path_refused_before_mutation(tmp_path,monkeypatch):
    _,plan,_,_,_,_=fixture(tmp_path,monkeypatch)
    parent=Path(plan['parent_admission_plan']['parent_attempt'])
    before={str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}
    source=tmp_path/'source';source.mkdir();path=source/'plan.json';fixtures.write(path,plan)
    monkeypatch.setattr(driver,'HERE',source)
    monkeypatch.setattr(sys,'argv',['center_secant','--output',str(parent/'run'),'--plan',str(path)])
    with pytest.raises(ValueError,match='immutable'):driver.main()
    assert {str(p):base.digest(p) for p in parent.rglob('*') if p.is_file()}==before


def test_isolated_help():
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/center_secant.py'),'--help'],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert '--plan' in result.stdout
