"""Offline lifecycle faults; no credentials, rental, network, or model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import huginn_retrieval_guard as g


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_fixture(tmp_path):
    identity = dict(pod_id='owned', name='neural-shrimp-test', source='a'*40,
                    approval='approved', hard_deadline_epoch=2000., plan='docs/huginn-chaos-map-plan.json')
    out = tmp_path/'incremental'
    put(out/'data/plan.json',dict(resolution=2,batch=2))
    plan_sha = g.digest_file(out/'data/plan.json')
    put(out/'identity.json', dict(identity, source_archive_sha256='b'*64,
                                 source_files={identity['plan']:{'sha256':plan_sha}}))
    put(out/'data/runtime.json',dict(source='a'*40,plan_sha256=plan_sha,dtype='float32',tf32=False))
    rows = [dict(phase='native_gate',error=0.,repeat=0.),
            dict(phase='throughput_gate',seconds_per_cell_step=.01,projected_seconds_with_margin=100.,remaining_seconds=900.),
            dict(phase='chunk_complete',chunk_start=0,completed=2,total=4),
            dict(phase='chunk_complete',chunk_start=2,completed=4,total=4),dict(phase='complete')]
    (out/'data/progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'data/tail-0000-0.npz').write_bytes(b'full-state synthetic evidence')
    refresh_manifest(out)
    return out, identity


def refresh_manifest(out):
    files = sorted(p for p in (out/'data').iterdir() if p.name != 'manifest.json')
    manifest = {p.name:g.digest_file(p) for p in files}
    put(out/'data/manifest.json',manifest)
    put(out/'status.json', dict(state='complete',complete=True,pod_id='owned',source_archive_sha256='b'*64,
                               manifest_sha256=g.digest_file(out/'data/manifest.json')))
    (out/'receipts.jsonl').write_text(''.join(json.dumps(dict(event='verified',file='data/'+p.name,
                    sha256=manifest[p.name],bytes=p.stat().st_size))+'\n' for p in files))


def test_success_requires_verified_manifest_and_collector_exit(tmp_path):
    out, identity = complete_fixture(tmp_path)
    result = g.verified_completion(out, identity, 'b'*64, 0)
    assert result['retrieval_complete'] and result['manifest_files'] == 4
    for rc in (None, 1, -9):
        with pytest.raises(ValueError, match='exit successfully'):
            g.verified_completion(out, identity, 'b'*64, rc)


@pytest.mark.parametrize('fault', ['missing_manifest','wrong_manifest_hash','missing_tail','missing_receipt','wrong_identity'])
def test_no_false_completion_after_corruption_or_missing_data(tmp_path, fault):
    out, identity = complete_fixture(tmp_path)
    if fault == 'missing_manifest':
        (out/'data/manifest.json').unlink()
    elif fault == 'wrong_manifest_hash':
        (out/'data/manifest.json').write_text('{}')
    elif fault == 'missing_tail':
        (out/'data/tail-0000-0.npz').unlink()
    elif fault == 'missing_receipt':
        (out/'receipts.jsonl').write_text('')
    else:
        identity['pod_id'] = 'other'
    with pytest.raises((ValueError, FileNotFoundError)):
        g.verified_completion(out, identity, 'b'*64, 0)


def fake_guard(tmp_path):
    out, identity = complete_fixture(tmp_path)
    guard = g.ProtectedRun.__new__(g.ProtectedRun)
    guard.out, guard.identity = tmp_path, identity
    guard.source_archive_sha = 'b'*64
    guard.compute_deadline, guard.retrieval_deadline, guard.hard_deadline = 100., 200., 260.
    guard.fields = identity.copy()
    guard.collector = Mock()
    guard.write = Mock()
    return guard


def test_slow_active_retrieval_after_compute_completion_survives(tmp_path):
    guard = fake_guard(tmp_path)
    # Numerical computation has ended; retrieval outlasts its old no-progress
    # timer and still succeeds before the original hard deadline.
    guard.collector.poll.side_effect = [None, None, None, 0]
    remote = Mock(); remote.poll.return_value = 0
    ticks = iter([50., 110., 150., 190.])
    result = guard.supervise(remote, clock=lambda:next(ticks), sleep=lambda _:None)
    assert result['success'] and result['retrieval_complete']
    assert guard.write.call_args.kwargs['retrieval_complete']


def test_collector_death_never_looks_like_success(tmp_path):
    guard = fake_guard(tmp_path)
    guard.collector.poll.return_value = -9
    remote = Mock(); remote.poll.return_value = None
    with pytest.raises(RuntimeError, match='collector failed'):
        guard.supervise(remote, clock=lambda:50., sleep=lambda _:None)
    assert guard.write.call_args.args[0] != 'complete'


def test_missing_manifest_stops_at_original_retrieval_deadline(tmp_path):
    guard = fake_guard(tmp_path)
    guard.collector.poll.return_value = None
    remote = Mock(); remote.poll.return_value = 0
    (tmp_path/'incremental/data/manifest.json').unlink()
    with pytest.raises(TimeoutError, match='termination reserve'):
        guard.supervise(remote, clock=lambda:200., sleep=lambda _:None)


def test_compute_cannot_continue_into_retrieval_reserve(tmp_path):
    guard = fake_guard(tmp_path)
    guard.collector.poll.return_value = None
    remote = Mock(); remote.poll.return_value = None
    with pytest.raises(TimeoutError, match='compute timeout'):
        guard.supervise(remote, clock=lambda:116., sleep=lambda _:None)


def test_nonzero_numerical_exit_keeps_verified_evidence_but_fails_run(tmp_path):
    guard = fake_guard(tmp_path)
    guard.collector.poll.return_value = 0
    remote = Mock(); remote.poll.return_value = 124
    result = guard.supervise(remote, clock=lambda:100., sleep=lambda _:None)
    assert result['retrieval_complete'] and not result['success']


def test_racing_watchdog_termination_still_writes_result(tmp_path):
    provider = Mock()
    provider.pod_status.side_effect = [dict(id='owned',name='neural-shrimp-test'),None]
    provider.cmd_terminate.side_effect = SystemExit('already deleted by watchdog')
    provider.inventory.return_value = {'pods':[]}
    result = {'pod_id':'owned','name':'neural-shrimp-test'}
    assert g.terminate_and_record('owned', result, tmp_path, 1., provider)
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['termination_verified']
    assert saved['termination_error'] == 'already deleted by watchdog'


def test_termination_api_failure_preserves_unconfirmed_receipt(tmp_path):
    provider = Mock()
    provider.pod_status.side_effect = RuntimeError('offline')
    result = {'pod_id':'owned','name':'neural-shrimp-test'}
    assert not g.terminate_and_record('owned', result, tmp_path, 1., provider)
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['termination_verified'] is False
    assert saved['termination_error'] == 'offline'


def test_exact_identity_mismatch_never_terminates(tmp_path):
    provider = Mock()
    provider.pod_status.return_value = dict(id='other',name='neural-shrimp-test')
    provider.inventory.return_value = {'pods':[{'id':'owned'}]}
    assert not g.terminate_and_record('owned', {'name':'neural-shrimp-test'}, tmp_path, 1., provider)
    provider.cmd_terminate.assert_not_called()


def test_direct_launch_without_controller_refused_before_creation():
    with pytest.raises(ValueError, match='controller-root'):
        g.validate_request(argparse.Namespace(controller_root=None), 'a'*40)


def test_remote_command_has_absolute_bound_and_no_archive():
    import shlex
    command = g.remote_command('toolseq', 'scan 500 huginn_chaos_map.py --out llm-soft-loop/pod64/scan', 2000.)
    words = shlex.split(command)
    assert words[3] == '2000.0'
    assert 'time.time()-10' in words[2]
    assert '--kill-after=10s' in words[2]
    assert 'tar ' not in command
    runner = (g.ROOT/'tools/runpod_huginn_chaos_runner.sh').read_text()
    assert g.PROTOCOL in runner and 'trap pack' not in runner and 'tar czf' not in runner


def test_huginn_detection_uses_parsed_tool_position():
    assert g.is_huginn('toolseq', 'scan 500 "huginn_chaos_map.py" --mode scan')
    assert not g.is_huginn('toolseq', 'scan 500 other.py --note huginn_chaos_map.py')


def request_fixture(tmp_path, monkeypatch):
    source = 'a'*40
    original_exists = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda path: False if path == Path('/tmp/.git') else original_exists(path))
    root = tmp_path/'controller'
    args = argparse.Namespace(controller_root=root, name='neural-shrimp-test', approval='approved',
                              jobset='toolseq', jobset_args='hcm 500 huginn_chaos_map.py --mode scan',
                              run_minutes=550., ready_minutes=15., max_rate=1., max_cost=11.,
                              out=root/'workers/neural-shrimp-test')
    flags = ['--name','--commit','--approval','--jobset','--jobset-args',
             '--run-minutes','--ready-minutes','--max-rate','--max-cost']
    values = [args.name,source,args.approval,args.jobset,args.jobset_args,'550','15','1','11']
    cmd = ['python','tools/runpod_soft_float64.py']
    for flag,value in zip(flags,values):
        cmd.extend([flag,value])
    config = dict(source_commit=source, approval='approved', names=[args.name],commands=[cmd],
                  hard_minutes=600., retrieval_minutes=20., max_rate=1.,ceiling=11.)
    put(root/'config.json',config)
    put(root/'state.json',dict(finished=False,controller_pid=10,controller_ticks=111))
    put(root/'watchdog-ready.json',dict(pid=20,ticks=111))
    put(root/'watchdog-heartbeat.json',dict(pid=20,time=1000.))
    monkeypatch.setattr(g.subprocess,'run',lambda *a,**k:argparse.Namespace(stdout=g.PROTOCOL.encode()))
    return args,config


def test_valid_controller_request_and_exact_output_binding(tmp_path, monkeypatch):
    args,config = request_fixture(tmp_path,monkeypatch)
    assert g.validate_request(args,'a'*40,now=1010.,identity=lambda _:111) == config
    args.out = tmp_path/'invisible-to-watchdog'
    with pytest.raises(ValueError,match='watchdog visibility'):
        g.validate_request(args,'a'*40,now=1010.,identity=lambda _:111)


@pytest.mark.parametrize('fault', ['stale_watchdog','dead_controller','source','job','old_runner'])
def test_unsafe_controller_refused_before_paid_launch(tmp_path, monkeypatch, fault):
    args,config = request_fixture(tmp_path,monkeypatch)
    now,identity = 1010.,lambda _:111
    if fault == 'stale_watchdog':
        now = 1500.
    elif fault == 'dead_controller':
        identity = lambda _:None
    elif fault == 'source':
        config['source_commit'] = 'c'*40
        put(args.controller_root/'config.json',config)
    elif fault == 'job':
        args.jobset_args += ' extra'
    elif fault == 'old_runner':
        monkeypatch.setattr(g.subprocess,'run',lambda *a,**k:argparse.Namespace(stdout=b'trap pack EXIT'))
    with pytest.raises(ValueError):
        g.validate_request(args,'a'*40,now=now,identity=identity)


def test_failed_numerics_preserve_slow_but_advancing_tail_drain(tmp_path):
    guard = fake_guard(tmp_path)
    guard.compute_deadline,guard.retrieval_deadline,guard.hard_deadline = 100.,1000.,1060.
    guard.collector.pid = 123
    guard.collector.poll.side_effect = [None,None,None,0]
    remote = Mock(); remote.poll.return_value = 124
    ticks = iter([50.,190.,330.,470.])
    counter = 0
    def clock():
        nonlocal counter
        counter += 1
        path = tmp_path/'incremental/status.json'
        status = json.loads(path.read_text())
        status.update(pid=123,progress_bytes=counter*1000)
        put(path,status)
        return next(ticks)
    result = guard.supervise(remote,clock=clock,sleep=lambda _:None)
    assert result['retrieval_complete'] and result['success'] is False


def test_failed_numerics_with_no_new_bytes_stop_without_waiting_full_budget(tmp_path):
    guard = fake_guard(tmp_path)
    guard.collector.poll.return_value = None
    remote = Mock(); remote.poll.return_value = 1
    ticks = iter([50.,171.])
    with pytest.raises(RuntimeError,match='no new progress'):
        guard.supervise(remote,clock=lambda:next(ticks),sleep=lambda _:None)


def test_diagnostic_snapshots_preserve_explicit_truncation_and_missing_logs(tmp_path, monkeypatch):
    import base64
    data = b'last error lines'
    names = ['setup-uv.log','setup-torch.log','setup-packages.log','setup-model.log',
             'hardware.txt','libraries.txt','scan.log']
    rows = [dict(name=name,missing=True) for name in names[:-1]]
    rows.append(dict(name='scan.log',remote_bytes=200000,captured_bytes=len(data),tail_only=True,
                     captured_sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode()))
    monkeypatch.setattr(g.subprocess,'run',lambda *a,**k:argparse.Namespace(stdout=json.dumps(rows).encode()))
    result = g.retrieve_diagnostics(['ssh','owned'],tmp_path,'scan',g.time.time()+20)
    assert result['state'] == 'retrieved'
    assert (tmp_path/'diagnostics/scan.log').read_bytes() == data
    receipt = json.loads((tmp_path/'diagnostics/receipt.json').read_text())
    assert receipt['files'][-1]['tail_only'] is True
    assert 'base64' not in receipt['files'][-1]


@pytest.mark.parametrize('fault',['native','throughput','chunk_missing','runtime'])
def test_scientific_gate_failure_cannot_be_called_normal_completion(tmp_path,fault):
    out,identity = complete_fixture(tmp_path)
    if fault == 'runtime':
        runtime = json.loads((out/'data/runtime.json').read_text())
        runtime['source'] = 'c'*40
        put(out/'data/runtime.json',runtime)
    else:
        path = out/'data/progress.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == 'native':
            rows[0]['error'] = .01
        elif fault == 'throughput':
            rows[1]['projected_seconds_with_margin'] = 400.
        else:
            rows.pop(2)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    refresh_manifest(out)
    with pytest.raises(ValueError):
        g.verified_completion(out,identity,'b'*64,0)


def test_worker_creation_lookup_filters_exact_source_and_approval(tmp_path, monkeypatch):
    import runpod_soft_float64 as worker
    rows = [dict(event='created',name='neural-shrimp-test',pod_id='mine',time='1970-01-01T00:16:41+00:00',commit='a'*40,approval='approved'),
            dict(event='created',name='neural-shrimp-test',pod_id='other',time='1970-01-01T00:16:42+00:00',commit='b'*40,approval='other')]
    ledger = tmp_path/'pods.jsonl'
    ledger.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(worker.rp,'LEDGER',ledger)
    assert worker.own_creation('neural-shrimp-test',1000,'a'*40,'approved')['pod_id'] == 'mine'
    with pytest.raises(SystemExit,match='ambiguous'):
        worker.own_creation('neural-shrimp-test',1000)
