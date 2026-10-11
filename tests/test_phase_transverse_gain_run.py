"""Synthetic lifecycle/source/denominator tests; no target gain calculation."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'tools'), str(ROOT/'tools')]
import run_phase_transverse_gain as r


def fake_result(cell):
    ids = [f'{cell["setting"]}--p{cell["phase_index"]}--d{dose}--{cell["direction"]}--s{sign:+d}'
           for dose in range(3) for sign in (-1,1)]
    return dict(cell=cell, status='complete', complete=True, qualified=True,
                qualified_algebraic=True, qualified_ideal_neutral=True,
                raw_array=np.arange(6,dtype=np.float64).reshape(2,3),
                compatibility=[dict(id=name,status='complete') for name in ids])


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    plan = dict(**r.CONTRACT, scope=r.SCOPE, source_files=[], cells_census=r.gain.cell_census())
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    review_path = tmp_path/'review.json'
    review_path.write_text('{}')
    review = dict(path='review.json',bytes=2,sha256=r.legacy.sha(review_path))
    head = 'a'*40
    bundle = dict(fixture=np.arange(12,dtype=np.float64).reshape(4,3))
    monkeypatch.setattr(r,'validate',lambda p:None)
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test',review=review))
    monkeypatch.setattr(r,'resource_check',lambda *a,**k:None)
    monkeypatch.setattr(r.inputs,'load_bound_inputs',lambda path:copy.deepcopy(bundle))
    monkeypatch.setattr(r.inputs,'cell_inputs',lambda b,c:{})
    monkeypatch.setattr(r.gain,'evaluate_cell',lambda cell,**k:fake_result(cell))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *a:head if a==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.runtime,'identity',lambda pid:None)
    monkeypatch.setattr(r.platform,'platform',lambda:'synthetic-host')
    monkeypatch.setattr(r.subprocess,'check_output',lambda args:review_path.read_bytes() if args[-1].endswith(':review.json') else plan_path.read_bytes())
    return plan,plan_path,head,tmp_path


def reseal(directory):
    receipt=r.read(directory/'receipt.json')
    receipt['artifacts']=r.inventory(directory)
    receipt['artifact_bytes']=sum(row['bytes'] for row in receipt['artifacts'])
    (directory/'receipt.json').write_text(json.dumps(receipt))


def test_complete_saved_packet_roundtrip_retains_all192_and1152(lifecycle):
    plan,path,head,tmp=lifecycle
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert result['complete'] and result['all_gates_pass_cells']==192
    assert result['realized_pulse_rows']==dict(required=1152,complete=1152,failed=0,unavailable=0)
    checked=r.verify(result['directory'])
    assert checked['complete'] and checked['replayed_cells']==192
    assert checked['producer_original_identity_absent'] and checked['new_integrations']==0


def test_failed_ideal_gate_does_not_erase_valid_algebra_or_calculation(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    def one_gate(cell,**kwargs):
        result=fake_result(cell)
        if cell==plan['cells_census'][0]:
            result.update(qualified=False,qualified_ideal_neutral=False)
        return result
    monkeypatch.setattr(r.gain,'evaluate_cell',one_gate)
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert result['complete'] and result['algebraically_qualified_cells']==192
    assert result['all_gates_pass_cells']==result['ideal_neutral_qualified_cells']==191
    assert r.verify(result['directory'])['complete']


def test_helper_dependency_failure_preserves_six_rows_and_census(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    def failed(cell,**kwargs):
        result=fake_result(cell)
        if cell==plan['cells_census'][0]:
            result.update(status='failed',complete=False,qualified=False,qualified_algebraic=False,qualified_ideal_neutral=False)
            for row in result['compatibility']:row['status']='failed'
        return result
    monkeypatch.setattr(r.gain,'evaluate_cell',failed)
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert not result['complete'] and result['calculated_cells']==191
    assert result['realized_pulse_rows']==dict(required=1152,complete=1146,failed=6,unavailable=0)
    assert not r.verify(result['directory'])['complete']


@pytest.mark.parametrize('failure',['input','seal_event','whole_in_cell','per_cell'])
def test_interruptions_preserve_denominators_and_verify(lifecycle,monkeypatch,failure):
    plan,path,head,tmp=lifecycle
    if failure=='input':
        def missing(*a):raise ValueError('synthetic unavailable input')
        monkeypatch.setattr(r.inputs,'load_bound_inputs',missing)
    elif failure=='seal_event':
        original=r.runtime.append
        def unavailable(path,value):
            if Path(path).name=='events.jsonl':raise ValueError('synthetic event failure')
            return original(path,value)
        monkeypatch.setattr(r.runtime,'append',unavailable)
    else:
        def stopped(cell,**kwargs):
            if cell==plan['cells_census'][1]:
                if failure=='whole_in_cell':raise r.ProducerStop('synthetic global deadline')
                raise r.CellStop('synthetic per-cell deadline')
            return fake_result(cell)
        monkeypatch.setattr(r.gain,'evaluate_cell',stopped)
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert not result['complete']
    rows=r.read(Path(result['directory'])/'terminal-records.json')
    assert len(rows)==192 and result['realized_pulse_rows']['required']==1152
    checked=r.verify(result['directory'])
    assert not checked['complete']
    if failure=='whole_in_cell':
        assert rows[1]['status']=='interrupted' and rows[1]['error']
        assert sum(row['status']=='not_started' for row in rows)==190
    elif failure=='per_cell':assert result['calculated_cells']==191


def test_missing_compatibility_identity_is_rejected_before_completion(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    def omitted(cell,**kwargs):
        result=fake_result(cell)
        if cell==plan['cells_census'][0]:result['compatibility'].pop()
        return result
    monkeypatch.setattr(r.gain,'evaluate_cell',omitted)
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert not result['complete'] and result['realized_pulse_rows']['unavailable']==6
    assert not r.verify(result['directory'])['complete']


def test_resealed_qualification_and_duplicate_terminal_tampering_rejected(lifecycle):
    plan,path,head,tmp=lifecycle
    result=r.run(path,tmp/'raw','attempt-001',head);directory=Path(result['directory'])
    rows=r.read(directory/'terminal-records.json');rows[0]['qualified']=False
    (directory/'terminal-records.json').write_text(json.dumps(rows));reseal(directory)
    with pytest.raises(ValueError):r.verify(directory)
    rows.append(rows[0]);(directory/'terminal-records.json').write_text(json.dumps(rows));reseal(directory)
    with pytest.raises(ValueError,match='denominator'):r.verify(directory)


def test_exclusive_external_output_and_alias_guard(tmp_path,monkeypatch):
    root=tmp_path/'repo';old=tmp_path/'old';root.mkdir();old.mkdir()
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'OLD',old)
    alias=tmp_path/'alias';alias.symlink_to(old,target_is_directory=True)
    for base in (root,old,alias):
        with pytest.raises(ValueError):r.attempt_directory(base,'attempt-001')
    r.attempt_directory(tmp_path/'safe','attempt-001')
    with pytest.raises(FileExistsError):r.attempt_directory(tmp_path/'safe','attempt-001')
    with pytest.raises(ValueError):r.attempt_directory(tmp_path/'safe','../escape')


def test_source_guard_requires_actual_declared_commit_review_and_remote(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();monkeypatch.setattr(r,'ROOT',root);head='a'*40
    path=root/r.PLAN_REL;path.parent.mkdir(parents=True);path.write_text('{}')
    review=root/r.REVIEW_REL;review.parent.mkdir();review.write_text(json.dumps(dict(status='ready_for_exploratory_execution',plan_sha256=r.legacy.sha(path))))
    def git(*args):
        if args[0]=='status':return ''
        if args==('rev-parse','HEAD'):return head
        if args==('branch','--show-current'):return 'codex/test'
        return head+'\trefs/heads/codex/test'
    monkeypatch.setattr(r.legacy,'git',git)
    monkeypatch.setattr(r.subprocess,'check_output',lambda args:(root/args[-1].split(':',1)[1]).read_bytes())
    assert r.source_guard({'source_files':[]},path,head)['source_commit']==head
    with pytest.raises(ValueError):r.source_guard({'source_files':[]},path,'b'*40)
    monkeypatch.setattr(r.legacy,'git',lambda *args:'dirty' if args[0]=='status' else git(*args))
    with pytest.raises(ValueError):r.source_guard({'source_files':[]},path,head)


def test_cancellation_is_not_swallowed_by_arithmetic_handlers():
    assert not issubclass(r.ProducerStop,Exception)
    assert not issubclass(r.CellStop,Exception)


def test_truncated_input_seal_is_preserved_as_failed_attempt(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    def partial(directory,name,value):
        (Path(directory)/(name+'.json')).write_text('{"partial":')
        raise r.ProducerStop('synthetic input-seal interruption')
    monkeypatch.setattr(r.codec,'save_packet',partial)
    result=r.run(path,tmp/'raw','attempt-001',head)
    checked=r.verify(result['directory'])
    assert not checked['complete'] and checked['replayed_cells']==0
    assert checked['input_seal_status']=='invalid_partial_bytes_preserved'
    assert checked['strict_saved_array_replay'] is False
    assert (Path(result['directory'])/'inputs-seal.json').read_text()=='{"partial":'


def test_signal_during_journal_commit_preserves_completed_cell(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    original=r.runtime.append
    sent=False
    def interrupt(path,value):
        nonlocal sent
        if Path(path).name=='cell-journal.jsonl' and not sent:
            sent=True
            os.kill(os.getpid(),signal.SIGTERM)
        return original(path,value)
    monkeypatch.setattr(r.runtime,'append',interrupt)
    result=r.run(path,tmp/'raw','attempt-001',head)
    assert not result['complete'] and result['calculated_cells']==1
    assert result['status_counts']=={'complete':1,'not_started':191}
    checked=r.verify(result['directory'])
    assert not checked['complete'] and checked['replayed_cells']==1


def test_resealed_unavailable_counter_corruption_is_rejected(lifecycle,monkeypatch):
    plan,path,head,tmp=lifecycle
    def unavailable(*a):raise ValueError('synthetic input failure')
    monkeypatch.setattr(r.inputs,'load_bound_inputs',unavailable)
    result=r.run(path,tmp/'raw','attempt-001',head);directory=Path(result['directory'])
    rows=r.read(directory/'terminal-records.json')
    rows[0]['compatibility_counts'].update(complete=6,unavailable=0)
    (directory/'terminal-records.json').write_text(json.dumps(rows))
    summary=r.read(directory/'summary.json');summary.update(r.summary_from_rows(rows))
    (directory/'summary.json').write_text(json.dumps(summary));reseal(directory)
    with pytest.raises(ValueError,match='Unavailable cell'):r.verify(directory)


def test_resealed_not_started_row_with_completed_packet_is_rejected(lifecycle):
    plan,path,head,tmp=lifecycle
    result=r.run(path,tmp/'raw','attempt-001',head);directory=Path(result['directory'])
    rows=r.read(directory/'terminal-records.json');rows[0]['status']='not_started'
    (directory/'terminal-records.json').write_text(json.dumps(rows))
    journal=directory/'cell-journal.jsonl'
    journal.write_text('\n'.join(journal.read_text().splitlines()[1:])+'\n')
    summary=r.read(directory/'summary.json');summary.update(r.summary_from_rows(rows));summary['complete']=False
    (directory/'summary.json').write_text(json.dumps(summary));reseal(directory)
    with pytest.raises(ValueError,match='Not-started cell'):r.verify(directory)
