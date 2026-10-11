"""Real, bounded synthetic subprocesses; no neural integration or data access."""
from pathlib import Path
import json
import os
import signal
import sys
import time
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import two_write_runtime as runtime


def specs(n=2):
    return [dict(id=f'task-{i}',kind='synthetic',case=dict(id=f'case-{i}'),method=dict(id='analytic')) for i in range(n)]


def contexts(n=2):return {f'case-{i}':{} for i in range(n)}


def completed_worker(task,context,directory,libraries,parent_pid):
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],
             pid=os.getpid(),status='complete',rmse=None)
    runtime.write(Path(directory)/'records'/f'{task["id"]}.json',row)


def partial_worker(task,context,directory,libraries,parent_pid):
    runtime.write(Path(directory)/'payloads'/f'{task["id"]}-partial.json',{'step':1,'pid':os.getpid()})
    time.sleep(2.)


def failing_worker(task,context,directory,libraries,parent_pid):
    runtime.write(Path(directory)/'payloads'/f'{task["id"]}-partial.json',{'step':1})
    os._exit(7)


def records(directory):
    return [json.loads(s) for s in (directory/'journal.jsonl').read_text().splitlines()]


def assert_cleanup(directory):
    report=json.loads((directory/'cleanup.json').read_text())
    assert report['all_registered_workers_joined'] is True
    assert report['all_original_identities_absent'] is True
    for row in report['workers']:
        assert runtime.identity(row['pid'])!=row['identity']
    return report


def test_all_synthetic_workers_finish_with_bound_identity(tmp_path):
    directory=tmp_path/'run'
    result=runtime.execute(specs(),contexts(),directory,{},workers=2,deadline=time.monotonic()+3,per_task=1,worker_fn=completed_worker)
    assert len(result)==2 and all(row['status']=='complete' for row in result)
    assert records(directory)==result==json.loads((directory/'terminal-records.json').read_text())
    assert len(assert_cleanup(directory)['workers'])==2


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM,signal.SIGALRM])
def test_signal_during_journal_commit_does_not_duplicate_completed_task(tmp_path,monkeypatch,sig):
    directory=tmp_path/'run'; original_append=runtime.append; fired=False
    class Interrupted(BaseException):pass
    def handler(signum,frame):raise Interrupted('synthetic handled interruption')
    def inject(path,value):
        nonlocal fired
        if Path(path).name=='journal.jsonl' and not fired:
            fired=True
            os.kill(os.getpid(),sig)
            # The handled signal is pending until durable+memory commit finishes.
            assert sig in signal.sigpending()
        original_append(path,value)
    monkeypatch.setattr(runtime,'append',inject)
    old=signal.signal(sig,handler)
    try:
        with pytest.raises(Interrupted):
            runtime.execute(specs(),contexts(),directory,{},workers=1,deadline=time.monotonic()+3,per_task=1,worker_fn=completed_worker)
    finally:signal.signal(sig,old)
    journal=records(directory); terminal=json.loads((directory/'terminal-records.json').read_text())
    assert fired and journal==terminal and len(journal)==2
    assert len({row['id'] for row in journal})==2
    assert journal[0]['status']=='complete'
    assert journal[1]['status']=='not_started_interruption'
    assert len(assert_cleanup(directory)['workers'])==1


def test_timeout_keeps_partial_payload_and_terminates_owned_process(tmp_path):
    directory=tmp_path/'run'
    result=runtime.execute(specs(1),contexts(1),directory,{},workers=1,deadline=time.monotonic()+3,per_task=.1,worker_fn=partial_worker)
    assert result[0]['status']=='task_timeout'
    assert json.loads((directory/'payloads/task-0-partial.json').read_text())['step']==1
    assert records(directory)==result
    assert len(assert_cleanup(directory)['workers'])==1


def test_failed_worker_keeps_evidence_and_remaining_task_finishes(tmp_path):
    directory=tmp_path/'run'
    def mixed(task,context,directory,libraries,parent_pid):
        if task['id']=='task-0':failing_worker(task,context,directory,libraries,parent_pid)
        else:completed_worker(task,context,directory,libraries,parent_pid)
    result=runtime.execute(specs(),contexts(),directory,{},workers=1,deadline=time.monotonic()+3,per_task=1,worker_fn=mixed)
    assert [row['status'] for row in result]==['worker_failed','complete']
    assert result[0]['exitcode']==7 and (directory/'payloads/task-0-partial.json').exists()
    assert records(directory)==result
    assert len(assert_cleanup(directory)['workers'])==2


def test_expired_deadline_preserves_all_not_started_rows(tmp_path):
    directory=tmp_path/'run'
    def forbidden(*args):raise AssertionError('must not launch')
    result=runtime.execute(specs(),contexts(),directory,{},workers=1,deadline=time.monotonic()-1,per_task=1,worker_fn=forbidden)
    assert len(result)==2 and all(row['status']=='not_started_deadline' for row in result)
    assert records(directory)==result
    assert assert_cleanup(directory)['workers']==[]


def test_start_failure_retains_both_planned_identities(tmp_path,monkeypatch):
    directory=tmp_path/'run'
    class StartFailureProcess:
        pid=None
        def __init__(self,**kwargs):pass
        def start(self):raise RuntimeError('synthetic fork failure')
        def is_alive(self):return False
    monkeypatch.setattr(runtime.mp,'get_context',lambda kind:SimpleNamespace(Process=StartFailureProcess))
    with pytest.raises(RuntimeError,match='fork failure'):
        runtime.execute(specs(),contexts(),directory,{},workers=1,deadline=time.monotonic()+3,per_task=1,worker_fn=completed_worker)
    result=records(directory)
    assert len(result)==2 and len({row['id'] for row in result})==2
    assert result[0]['status']=='interrupted' and result[1]['status']=='not_started_interruption'
    assert result==json.loads((directory/'terminal-records.json').read_text())
    assert assert_cleanup(directory)['workers']==[]


@pytest.mark.parametrize('workers',[0,5])
def test_existing_worker_cap_unchanged(tmp_path,workers):
    directory=tmp_path/'run'
    with pytest.raises(ValueError,match='Worker cap'):
        runtime.execute(specs(),contexts(),directory,{},workers=workers,deadline=time.monotonic()+1,per_task=1,worker_fn=completed_worker)
    assert not directory.exists()
