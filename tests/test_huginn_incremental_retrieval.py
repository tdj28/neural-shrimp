"""Synthetic transport tests only: no network, GPU, or process lifecycle calls."""
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import huginn_incremental_retrieval as h


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ownership_fixture():
    source = 'a'*40
    job = f'hcm3 540 huginn_chaos_map.py --plan docs/huginn-chaos-tight-zoom-plan.json --source-commit {source} --mode scan --out llm-soft-loop/pod64/hcm3/map-01'
    config = dict(source_commit=source, approval='approved', names=['neural-shrimp-test'],
                  preexisting_ids=['other'], started_epoch=1000, hard_minutes=600,
                  commands=[['runner', '--name', 'neural-shrimp-test', '--commit', source,
                             '--approval', 'approved', '--jobset', 'toolseq', '--jobset-args', job]])
    entry = dict(event='created', pod_id='owned', name='neural-shrimp-test', commit=source,
                 approval='approved', time='2026-10-23T07:00:00+00:00')
    return config, entry


def test_exact_ownership_and_existing_lifetime():
    config, entry = ownership_fixture()
    result = h.verify_ownership(config, [entry], 'owned', entry['name'], entry['commit'], entry['approval'])
    assert result['hard_deadline_epoch']-result['created_epoch'] == 36000
    for field in ('name', 'commit', 'approval'):
        wrong = dict(entry, **{field:'wrong'})
        with pytest.raises(ValueError):
            h.verify_ownership(config, [wrong], 'owned', entry['name'], entry['commit'], entry['approval'])
    config['preexisting_ids'].append('owned')
    with pytest.raises(ValueError, match='preexisting'):
        h.verify_ownership(config, [entry], 'owned', entry['name'], entry['commit'], entry['approval'])


class FakeReader:
    def __init__(self, files, snapshot=None):
        self.files, self.snapshot_value = files, snapshot
        self.transferred = []
        self.metadata_calls = []
        self.bad = None
        self.present = True

    def refresh(self):
        return self.present

    def snapshot(self, root):
        return self.snapshot_value

    def metadata(self, remote):
        self.metadata_calls.append(remote)
        data = self.files[remote]
        return dict(sha256=sha(data), bytes=len(data))

    def stream(self, remote, target, timeout, pulse):
        self.transferred.append(remote)
        data = self.files[remote]
        if self.bad == 'truncated':
            data = data[:-1]
        if self.bad == 'hash':
            data = b'X'*len(data)
        target.write(data)
        pulse(len(data))
        return dict(sha256=sha(data), bytes=len(data))


def setup_collector(tmp_path, snapshot=None):
    config, entry = ownership_fixture()
    identity = h.verify_ownership(config, [entry], 'owned', entry['name'], entry['commit'], entry['approval'])
    identity['hard_deadline_epoch'] = time.time()+3600
    plan = dict(resolution=2, batch=2, eps=[.001,.0003], retain_full_tails=True)
    plan_bytes = json.dumps(plan).encode()
    files = {identity['remote_data']+'/plan.json':plan_bytes,
             identity['remote_data']+'/runtime.json':json.dumps(dict(source=entry['commit'],plan_sha256=sha(plan_bytes))).encode(),
             identity['remote_data']+'/steering-direction.npy':b'direction',
             '/root/neural-shrimp/'+identity['plan']:plan_bytes,
             '/root/neural-shrimp/tools/example.py':b'source',
             '/root/neural-shrimp-data/explore/llm-soft-loop/pod64/logs/hardware.txt':b'GPU, driver, memory\n',
             '/root/neural-shrimp-data/explore/llm-soft-loop/pod64/logs/libraries.txt':b'library==version\n'}
    sources = {identity['plan']:dict(sha256=sha(plan_bytes), bytes=len(plan_bytes)),
               'tools/example.py':dict(sha256=sha(b'source'), bytes=6)}
    for start in (0,2):
        files[identity['remote_data']+f'/chunk-{start:04d}.npz'] = f'chunk{start}'.encode()
        for offset in (0,1):
            files[identity['remote_data']+f'/tail-{start:04d}-{offset}.npz'] = f'tail{start}{offset}'.encode()
    reader = FakeReader(files, snapshot or dict(ready=True, chunks=[0,2], manifest=None))
    collector = h.Collector(tmp_path/'out', identity, sources, 'b'*64, plan, reader)
    return collector, reader


@pytest.mark.parametrize('failure', ['truncated','hash'])
def test_failed_partial_preserved_and_retry_new_path(tmp_path, failure):
    collector, reader = setup_collector(tmp_path)
    reader.bad = failure
    remote = collector.identity['remote_data']+'/chunk-0000.npz'
    assert not collector.transfer(remote, 'data/chunk-0000.npz')
    partial = next((collector.out/'data').glob('*.partial'))
    old = partial.read_bytes()
    assert not (collector.out/'data/chunk-0000.npz').exists()
    reader.bad = None
    assert collector.transfer(remote, 'data/chunk-0000.npz')
    assert partial.read_bytes() == old
    assert (collector.out/'data/chunk-0000.npz').read_bytes() == b'chunk0'
    collector.close()


def test_skip_only_after_local_and_remote_hash_match(tmp_path):
    collector, reader = setup_collector(tmp_path)
    remote = collector.identity['remote_data']+'/chunk-0000.npz'
    assert collector.transfer(remote, 'data/chunk-0000.npz')
    assert collector.transfer(remote, 'data/chunk-0000.npz')
    assert reader.transferred == [remote]
    assert reader.metadata_calls == [remote]  # cached immutable file: no new SSH
    target = collector.out/'data/chunk-0000.npz'
    target.write_bytes(b'badexisting')
    with pytest.raises(ValueError, match='preserved'):
        collector.transfer(remote, 'data/chunk-0000.npz')
    assert target.read_bytes() == b'badexisting'
    collector.close()


def test_compact_priority_and_no_incomplete_chunk_or_early_final_files(tmp_path):
    collector, reader = setup_collector(tmp_path, dict(ready=True, chunks=[0], manifest=None))
    assert collector.step() == 'partial'
    names = [Path(p).name for p in reader.transferred]
    assert names.index('chunk-0000.npz') < names.index('tail-0000-0.npz')
    assert not any('0002' in p for p in names)
    assert not {'manifest.json', 'map.npz', 'summary.json', 'progress.jsonl'}.intersection(names)
    reader.snapshot_value['chunks'] = [0,2]
    reader.transferred.clear()
    collector.step()
    names = [Path(p).name for p in reader.transferred]
    assert names.index('chunk-0002.npz') < names.index('tail-0000-1.npz')
    collector.close()


def test_attempts_bounded_across_restart(tmp_path):
    collector, reader = setup_collector(tmp_path)
    remote = collector.identity['remote_data']+'/chunk-0000.npz'
    reader.bad = 'hash'
    for _ in range(4):
        assert not collector.transfer(remote, 'data/chunk-0000.npz')
    assert len(reader.transferred) == 3
    collector.close()
    restarted = h.Collector(collector.out, collector.identity, collector.sources,
                            'b'*64, collector.plan, reader)
    assert not restarted.transfer(remote, 'data/chunk-0000.npz')
    assert len(reader.transferred) == 3
    assert len(list((collector.out/'data').glob('*.partial'))) == 3
    restarted.close()


def test_absent_pod_exits_before_ssh_and_final_success_requires_all_hashes(tmp_path):
    collector, reader = setup_collector(tmp_path)
    reader.present = False
    assert collector.step() == 'absent'
    assert reader.transferred == []
    reader.present = True
    root = collector.identity['remote_data']
    for name in ['map.npz','summary.json','progress.jsonl']:
        reader.files[root+'/'+name] = name.encode()
    manifest = {key.removeprefix(root+'/'):sha(data) for key,data in reader.files.items() if key.startswith(root+'/')}
    reader.files[root+'/manifest.json'] = json.dumps(manifest).encode()
    reader.snapshot_value['manifest'] = manifest
    assert collector.step() == 'partial'
    assert json.loads((collector.out/'status.json').read_text())['complete'] is False
    for _ in range(3):
        result = collector.step()
    assert result == 'complete'
    assert json.loads((collector.out/'status.json').read_text())['complete'] is True
    for name,digest in manifest.items():
        assert h.digest_file(collector.out/'data'/name) == digest
    collector.close()


def test_live_identity_mismatch_refuses_endpoint():
    identity = dict(pod_id='owned', name='expected')
    reader = h.SSHReader(identity, time.time()+10, lambda _:dict(id='other',name='expected'))
    with pytest.raises(ValueError, match='identity'):
        reader.refresh()


def test_cached_chunk_rechecked_against_final_manifest(tmp_path):
    collector, reader = setup_collector(tmp_path)
    remote = collector.identity['remote_data']+'/chunk-0000.npz'
    assert collector.transfer(remote, 'data/chunk-0000.npz')
    with pytest.raises(ValueError, match='final manifest'):
        collector.transfer(remote, 'data/chunk-0000.npz', 'c'*64)
    collector.close()


def test_large_tails_hashed_only_after_all_final_files_present(tmp_path, monkeypatch):
    collector, reader = setup_collector(tmp_path)
    root = collector.identity['remote_data']
    for name in ['map.npz','summary.json','progress.jsonl']:
        reader.files[root+'/'+name] = name.encode()
    manifest = {key.removeprefix(root+'/'):sha(data) for key,data in reader.files.items() if key.startswith(root+'/')}
    reader.files[root+'/manifest.json'] = json.dumps(manifest).encode()
    reader.snapshot_value['manifest'] = manifest
    original = h.digest_file
    hashed = []
    def recording(path):
        hashed.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(h, 'digest_file', recording)
    for _ in range(3):
        assert collector.step() == 'partial'
    assert not any(n.startswith('tail-') for n in hashed)
    assert collector.step() == 'complete'
    assert len([n for n in hashed if n.startswith('tail-')]) == 4
    collector.close()


def test_stable_environment_preserved_once_with_hash_receipts(tmp_path):
    collector, reader = setup_collector(tmp_path)
    collector.step()
    collector.step()
    for name in ('hardware.txt', 'libraries.txt'):
        remote = '/root/neural-shrimp-data/explore/llm-soft-loop/pod64/logs/'+name
        target = collector.out/'environment'/name
        assert target.read_bytes() == reader.files[remote]
        assert reader.transferred.count(remote) == 1
        assert reader.metadata_calls.count(remote) == 1
        rows = [json.loads(line) for line in (collector.out/'receipts.jsonl').read_text().splitlines()]
        receipt = next(row for row in rows if row['event']=='verified' and row['file']=='environment/'+name)
        assert receipt['sha256'] == sha(reader.files[remote])
        assert receipt['bytes'] == len(reader.files[remote])
    collector.close()


def test_progress_counts_unique_bytes_not_retries_or_heartbeat(tmp_path):
    collector, reader = setup_collector(tmp_path)
    initial = json.loads((collector.out/'status.json').read_text())
    assert initial['state'] == 'ready'
    assert initial['progress_bytes'] == 0
    collector.pulse('data/a.npz', 10)
    collector.pulse('data/a.npz', 10)  # repeated chatter
    collector.pulse('data/a.npz', 2)   # restarted transfer
    collector.status('waiting')
    assert json.loads((collector.out/'status.json').read_text())['progress_bytes'] == 10
    collector.pulse('data/a.npz', 12)
    collector.pulse('data/b.npz', 5)
    assert json.loads((collector.out/'status.json').read_text())['progress_bytes'] == 17
    collector.close()


def test_collector_restart_preserves_verified_file_and_incomplete_transfer(tmp_path):
    collector, reader = setup_collector(tmp_path)
    root = collector.identity['remote_data']
    assert collector.transfer(root+'/chunk-0000.npz', 'data/chunk-0000.npz')
    reader.bad = 'truncated'
    assert not collector.transfer(root+'/tail-0000-0.npz', 'data/tail-0000-0.npz')
    partials = {p.name:p.read_bytes() for p in (collector.out/'data').glob('*.partial')}
    collector.close()
    reader.bad = None
    reader.transferred.clear()
    restarted = h.Collector(collector.out, collector.identity, collector.sources,
                            'b'*64, collector.plan, reader)
    assert restarted.transfer(root+'/chunk-0000.npz', 'data/chunk-0000.npz')
    assert restarted.transfer(root+'/tail-0000-0.npz', 'data/tail-0000-0.npz')
    assert root+'/chunk-0000.npz' not in reader.transferred
    assert {p.name:p.read_bytes() for p in (collector.out/'data').glob('*.partial')} == partials
    restarted.close()


def test_directory_fsync_after_atomic_publication(tmp_path, monkeypatch):
    collector, reader = setup_collector(tmp_path)
    synced = []
    original = h.sync_directory
    def record(path):
        synced.append(Path(path))
        original(path)
    monkeypatch.setattr(h, 'sync_directory', record)
    root = collector.identity['remote_data']
    assert collector.transfer(root+'/chunk-0000.npz', 'data/chunk-0000.npz')
    assert collector.out/'data' in synced
    collector.status('waiting')
    assert collector.out in synced
    collector.close()


def test_forced_kill_preserves_completed_shard_and_partial_for_restart(tmp_path):
    """Kill a real local collector process mid-transfer, not a mocked exception."""
    import multiprocessing
    import os
    ctx = multiprocessing.get_context('fork')
    partial_written = ctx.Event()

    def interrupted_collection():
        collector, reader = setup_collector(tmp_path)
        root = collector.identity['remote_data']
        assert collector.transfer(root+'/chunk-0000.npz', 'data/chunk-0000.npz')
        def interrupted_stream(remote, target, timeout, pulse):
            target.write(b'partial'); target.flush(); os.fsync(target.fileno())
            pulse(7)
            partial_written.set()
            time.sleep(30)
        reader.stream = interrupted_stream
        collector.transfer(root+'/tail-0000-0.npz', 'data/tail-0000-0.npz')

    process = ctx.Process(target=interrupted_collection)
    process.start()
    try:
        assert partial_written.wait(5), 'child never reached interrupted transfer'
        process.kill()  # this test's own local process, SIGKILL
        process.join(5)
        assert process.exitcode == -9
    finally:
        if process.is_alive():
            process.kill(); process.join(5)
    out = tmp_path/'out'
    partials = {p.name:p.read_bytes() for p in (out/'data').glob('*.partial')}
    assert list(partials.values()) == [b'partial']
    assert (out/'data/chunk-0000.npz').read_bytes() == b'chunk0'
    assert not (out/'data/tail-0000-0.npz').exists()
    # Retain exact immutable identity/deadline. New fake transport serves the
    # same pinned bytes; verified chunks are revalidated but not recopied.
    binding = json.loads((out/'identity.json').read_text())
    fixture, reader = setup_collector(tmp_path/'fresh-fixture')
    fixture.close()
    identity = {k:v for k,v in binding.items()
                if k not in ('source_archive_sha256','source_files','deadline_epoch')}
    restarted = h.Collector(out, identity, binding['source_files'],
                            binding['source_archive_sha256'], fixture.plan, reader,
                            deadline=binding['deadline_epoch'])
    root = identity['remote_data']
    assert restarted.transfer(root+'/chunk-0000.npz', 'data/chunk-0000.npz')
    assert restarted.transfer(root+'/tail-0000-0.npz', 'data/tail-0000-0.npz')
    assert root+'/chunk-0000.npz' not in reader.transferred
    assert {p.name:p.read_bytes() for p in (out/'data').glob('*.partial')} == partials
    restarted.close()
