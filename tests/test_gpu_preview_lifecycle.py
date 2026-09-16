"""Mocked EXP-012 ownership/sealing guards; no CUDA, network or integration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_gpu_preview_qualification as driver


COMMIT = '1' * 40


def loaded(path):
    return json.loads(Path(path).read_text())


def harness(monkeypatch, tmp_path, *, start_failure=False, full_schedule=False):
    """Exercise real run/sealing code with fake process and system boundaries."""
    plan = dict(census=[dict(id=f'synthetic-{i:04d}') for i in range(3456 if full_schedule else 2)])
    plan_path = tmp_path/'synthetic-plan.json'
    plan_path.write_text(json.dumps(plan))
    old_plan = tmp_path/'synthetic-old-plan.json'
    old_plan.write_text('{}')
    monkeypatch.setattr(driver.preview, 'PLAN', old_plan)
    if not full_schedule:
        fake_jobs = [dict(id=f'cold-{arm}', arm=arm, dt=.0025, size=2,
                         kind='cold', rep=None, layout='canonical') for arm in ('cpu', 'gpu')]
        monkeypatch.setattr(driver, 'jobs', lambda: fake_jobs)
    validations, builds, processes, terminated = [], [], [], []
    monkeypatch.setattr(driver, 'validate', lambda *args, **kwargs: validations.append(kwargs))
    monkeypatch.setattr(driver, 'source_guard', lambda *_: dict(source_commit=COMMIT, pushed_branch='synthetic'))
    monkeypatch.setattr(driver.runtime, 'git', lambda *args: COMMIT)
    monkeypatch.setattr(driver.importlib.metadata, 'version', lambda name: driver.CONTRACT['required_packages'][name])
    snapshot = dict(nvidia_smi_status=0, available_ram_bytes=16*1024**3,
                    free_disk_bytes=16*1024**3, synthetic=True)
    monkeypatch.setattr(driver, 'resources', lambda: snapshot.copy())
    monkeypatch.setattr(driver, 'require_headroom', lambda _: snapshot.copy())
    monkeypatch.setattr(driver.runtime, 'build_native', lambda directory, deadline:
        builds.append(('cpu', directory)) or dict(path=str(directory/'fake.so')))
    monkeypatch.setattr(driver.cuda, 'build', lambda directory, **kwargs:
        builds.append(('gpu', directory)) or dict(library=str(directory/'fake.so'), wall_seconds=.001))
    monkeypatch.setattr(driver, 'audit', lambda *_: dict(complete=True, all_parity_pass=True,
                                                       gpu_deployment_qualified=True))

    # Do not touch pytest's actual handlers/timers, and still check restoration.
    handlers = {sig: object() for sig in (driver.signal.SIGINT, driver.signal.SIGTERM, driver.signal.SIGALRM)}
    original_handlers = handlers.copy()
    timers = []
    def signal_handler(sig, value):
        previous = handlers[sig]
        handlers[sig] = value
        return previous
    monkeypatch.setattr(driver.signal, 'signal', signal_handler)
    monkeypatch.setattr(driver.signal, 'setitimer', lambda *args: timers.append(args))
    monkeypatch.setattr(driver.signal, 'pthread_sigmask', lambda *args: set())

    class Connection:
        def __init__(self):
            self.messages, self.closed = [], False
        def send(self, message):
            self.messages.append(message)
        def close(self):
            self.closed = True

    class Process:
        def __init__(self, target, args):
            self.pid, self.exitcode, self.alive = None, None, False
            self.target, self.args = target, args
            processes.append(self)
        def start(self):
            self.pid, self.alive = 900000+len(processes), True
            if start_failure:
                raise RuntimeError('synthetic start failed after PID assignment')
        def is_alive(self):
            return self.alive
        def join(self, timeout=None):
            self.alive, self.exitcode = False, 0
    context = SimpleNamespace(Pipe=lambda: (Connection(), Connection()), Process=Process)
    monkeypatch.setattr(driver.mp, 'get_context', lambda name: context)
    def terminate(process):
        terminated.append(process)
        process.alive, process.exitcode = False, process.exitcode if process.exitcode is not None else -15
    monkeypatch.setattr(driver.runtime, 'terminate_owned', terminate)
    monkeypatch.setattr(driver, 'process_identity', lambda pid: next(
        (dict(pid=pid, start_ticks='synthetic', boot_id='synthetic')
         for process in processes if process.pid == pid and process.alive), None))
    def receive(process, connection, deadline):
        if not connection.messages:
            return dict(status='ready', pid=process.pid, startup_seconds=.001)
        job, directory = connection.messages[-1]
        return dict(**job, status='complete', pid=process.pid,
                    ids=[r['id'] for r in plan['census'][:job['size']]], end_to_end_seconds=.001)
    monkeypatch.setattr(driver, 'receive', receive)
    return SimpleNamespace(plan=plan, plan_path=plan_path, output=tmp_path/'external-attempt',
        validations=validations, builds=builds, processes=processes, terminated=terminated,
        handlers=handlers, original_handlers=original_handlers, timers=timers)


def assert_failed_sealed_attempt(state, expected_jobs):
    receipt = loaded(state.output/'receipt.json')
    records = loaded(state.output/'terminal-records.json')
    assert receipt['status'] == 'incomplete'
    assert receipt['summary']['complete'] is False
    assert receipt['summary']['all_parity_pass'] is False
    assert receipt['summary']['gpu_deployment_qualified'] is False
    assert [r['id'] for r in records] == [r['id'] for r in expected_jobs]
    assert len({r['id'] for r in records}) == len(expected_jobs)
    assert state.handlers == state.original_handlers
    assert state.timers[-1] == (driver.signal.ITIMER_REAL, 0)
    expected_files = {str(p.relative_to(state.output)) for p in state.output.rglob('*')
                      if p.is_file() and p.name != 'receipt.json'}
    assert {item['path'] for item in receipt['artifacts']} == expected_files
    for item in receipt['artifacts']:
        assert driver.runtime.binding(state.output/item['path'], root=state.output) == item
    return receipt, records


def test_initial_headroom_failure_seals_all_54_batches_and_64188_case_statuses(monkeypatch, tmp_path):
    state = harness(monkeypatch, tmp_path, full_schedule=True)
    schedule = driver.jobs()
    def refuse(_):
        raise RuntimeError('synthetic shared RAM reserve unavailable')
    monkeypatch.setattr(driver, 'require_headroom', refuse)
    driver.run(state.plan, state.output, state.plan_path)
    receipt, records = assert_failed_sealed_attempt(state, schedule)
    assert len(records) == 54 and sum(len(r['ids']) for r in records) == 64188
    assert all(r['status'] == 'not_started' and 'RAM reserve unavailable' in r['error'] for r in records)
    assert not state.builds and not state.processes
    assert 'RAM reserve unavailable' in receipt['summary']['error']


def test_audit_exception_preserves_complete_raw_statuses_but_disables_qualification(monkeypatch, tmp_path):
    state = harness(monkeypatch, tmp_path)
    def fail_audit(*_):
        raise ValueError('synthetic required replay mismatch')
    monkeypatch.setattr(driver, 'audit', fail_audit)
    driver.run(state.plan, state.output, state.plan_path)
    receipt, records = assert_failed_sealed_attempt(state, driver.jobs())
    assert all(r['status'] == 'complete' for r in records)
    assert 'required replay mismatch' in receipt['summary']['error']
    assert len(state.terminated) == len(state.processes) == 2
    assert not any(process.alive for process in state.processes)


def test_process_start_failure_after_pid_assignment_remains_owned_and_is_terminated(monkeypatch, tmp_path):
    state = harness(monkeypatch, tmp_path, start_failure=True)
    driver.run(state.plan, state.output, state.plan_path)
    receipt, records = assert_failed_sealed_attempt(state, driver.jobs())
    assert len(state.processes) == 1 and state.processes[0].pid == 900001
    assert state.terminated == state.processes and not state.processes[0].alive
    assert state.processes[0].args[0].closed
    cleanup = loaded(state.output/'cleanup.json')
    assert cleanup['all_owned_workers_stopped']
    assert cleanup['workers'][0]['pid'] == 900001
    assert all(r['status'] == 'not_started' for r in records)
    assert 'start failed after PID assignment' in receipt['summary']['error']


@pytest.mark.parametrize('failure', ['changed_head', 'changed_input'])
def test_postflight_source_or_input_change_cannot_leave_qualification_true(monkeypatch, tmp_path, failure):
    state = harness(monkeypatch, tmp_path)
    if failure == 'changed_head':
        monkeypatch.setattr(driver.runtime, 'git', lambda *args: '2'*40)
    else:
        def validate(*args, **kwargs):
            state.validations.append(kwargs)
            if len(state.validations) > 1:
                raise ValueError('synthetic frozen input hash changed')
        monkeypatch.setattr(driver, 'validate', validate)
    driver.run(state.plan, state.output, state.plan_path)
    receipt, records = assert_failed_sealed_attempt(state, driver.jobs())
    assert all(r['status'] == 'complete' for r in records)
    assert 'changed' in receipt['summary']['error']


def source_fixture(monkeypatch, tmp_path):
    root = tmp_path/'synthetic-repo'
    root.mkdir()
    names = ['tool.py', 'native/backend.cu', 'docs/protocol.md']
    committed = {}
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f'synthetic committed {name}\n'.encode()
        path.write_bytes(content)
        committed[name] = content
    monkeypatch.setattr(driver, 'ROOT', root)
    monkeypatch.setattr(driver, 'SOURCE_PATHS', names)
    delegated, shown = [], []
    guard = dict(source_commit=COMMIT, pushed_branch='synthetic', remote_head_verified_at='synthetic')
    monkeypatch.setattr(driver.runtime, 'source_guard', lambda *args: delegated.append(args) or guard)
    def show(command):
        assert command[:4] == ['git', '-C', str(root), 'show']
        source, name = command[4].split(':', 1)
        assert source == COMMIT
        shown.append(name)
        return committed[name]
    monkeypatch.setattr(driver.subprocess, 'check_output', show)
    return root, names, committed, delegated, shown, guard


def test_source_guard_checks_each_committed_dependency_after_delegated_clean_push_gate(monkeypatch, tmp_path):
    root, names, committed, delegated, shown, guard = source_fixture(monkeypatch, tmp_path)
    plan, path = dict(synthetic=True), root/'plan.json'
    assert driver.source_guard(plan, path) == guard
    assert delegated == [(plan, path)] and shown == names


@pytest.mark.parametrize('changed_index', [0, 1, 2])
def test_source_guard_rejects_any_uncommitted_dependency(monkeypatch, tmp_path, changed_index):
    root, names, committed, delegated, shown, guard = source_fixture(monkeypatch, tmp_path)
    (root/names[changed_index]).write_bytes(b'changed uncommitted dependency\n')
    with pytest.raises(ValueError, match='Uncommitted dependency'):
        driver.source_guard({}, root/'plan.json')
    assert shown == names[:changed_index+1]


@pytest.mark.parametrize('reason', ['dirty source', 'source is not pushed', 'wrong expected source'])
def test_source_guard_never_bypasses_delegated_rejection(monkeypatch, tmp_path, reason):
    root, names, committed, delegated, shown, guard = source_fixture(monkeypatch, tmp_path)
    def reject(*_):
        raise ValueError(reason)
    monkeypatch.setattr(driver.runtime, 'source_guard', reject)
    with pytest.raises(ValueError, match=reason):
        driver.source_guard({}, root/'plan.json')
    assert not shown


def test_attempt_must_be_new_external_and_refuses_existing_directory(monkeypatch, tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(driver, 'ROOT', root)
    destination = tmp_path/'external'/'attempt'
    assert driver.create_attempt(destination) == destination.resolve()
    assert (destination/'batches').is_dir()
    sentinel = destination/'existing.txt'
    sentinel.write_text('retain this attempt')
    with pytest.raises(FileExistsError):
        driver.create_attempt(destination)
    assert sentinel.read_text() == 'retain this attempt'


@pytest.mark.parametrize('inside', ['', 'raw', 'nested/new-attempt'])
def test_attempt_rejects_whole_repository_subtree(monkeypatch, tmp_path, inside):
    root = tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(driver, 'ROOT', root)
    with pytest.raises(ValueError, match='external'):
        driver.create_attempt(root/inside)
    assert not list(root.iterdir())


def test_external_symlink_into_repository_cannot_bypass_output_guard(monkeypatch, tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    monkeypatch.setattr(driver, 'ROOT', root)
    alias = tmp_path/'apparently-external'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='external'):
        driver.create_attempt(alias/'new-attempt')
    assert not list(root.iterdir())


@pytest.mark.parametrize('target', ['source', 'raw', 'raw_alias'])
def test_verifier_output_cannot_modify_source_or_raw_even_via_symlink(monkeypatch, tmp_path, target):
    root, raw = tmp_path/'repo', tmp_path/'raw'
    root.mkdir()
    raw.mkdir()
    monkeypatch.setattr(driver, 'ROOT', root)
    alias = tmp_path/'alias'
    alias.symlink_to(raw, target_is_directory=True)
    output = {'source': root/'audit', 'raw': raw/'audit', 'raw_alias': alias/'audit'}[target]
    with pytest.raises(ValueError, match='outside raw evidence/source'):
        driver.verify_attempt({}, raw, output)
    assert not list(root.iterdir()) and not list(raw.iterdir())
