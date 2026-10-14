"""Focused new Task18 schema adapter tests; shared lifecycle is unchanged."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task18_launch',ROOT/'codex/loop-gain-crosscheck/launch.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

def args():return argparse.Namespace(attempt='run-002',source='a'*40,plan='run-002-plan.json',plan_sha256='b'*64)
def completed():return dict(status='loop_gain_completed',computational_completed=True,certificate_passed=False,
    exploratory=True,producer_arrays_read=False,phases=[dict(index=i) for i in range(8)],
    cyclic_monodromies=[dict(start_phase=i) for i in range(8)],composed_controls=dict(passed=True,rows=[{}, {}, {}]))

def test_complete_adapter_accepts_own_result_without_producer_agreement():L.validate_completion(completed())

@pytest.mark.parametrize('key,value',[('certificate_passed',True),('producer_arrays_read',True),('computational_completed',False),('phases',[dict(index=i) for i in range(7)]),('cyclic_monodromies',[dict(start_phase=0) for i in range(8)]),('composed_controls',dict(passed=False,rows=[{}, {}, {}]))])
def test_incomplete_or_promoted_result_refused(key,value):
    status=completed();status[key]=value
    with pytest.raises(RuntimeError):L.validate_completion(status)


def test_profile_identity_not_reused():
    a=args();a.attempt='profile-001'
    with pytest.raises(ValueError):L.validate_arguments(a)
    assert L.specification('run-002-plan.json')['caps']==dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=3600,minimum_start_available_gib=28,minimum_runtime_available_gib=16,maximum_output_mib=512)


def fixture(tmp_path,monkeypatch):
    a=args();spec=L.specification(a.plan);blobs={k:b'scientific byte fixture' for k in spec['workers']}
    # Only hash equality is exercised here; unchanged lifecycle's real source is reviewed separately.
    blobs[L.OWNERSHIP_FILE]=b'owner fixture';monkeypatch.setattr(L,'OWNERSHIP_SHA',L.hashlib.sha256(b'owner fixture').hexdigest())
    blobs['codex/loop-gain-crosscheck/launch.py']=b'launcher fixture'
    blobs['codex/loop-gain-crosscheck/run-002-card.md']=b'card fixture'
    plan=dict(schema=spec['schema'],admission='reviewed',envelope=spec['caps'],source_sha256={n:L.hashlib.sha256(blobs[n]).hexdigest() for n in spec['workers']})
    rel='codex/loop-gain-crosscheck/run-002-plan.json';blobs[rel]=json.dumps(plan).encode();a.plan_sha256=L.hashlib.sha256(blobs[rel]).hexdigest()
    for name,data in blobs.items():p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    def git(command,**kwargs):
        assert kwargs['timeout']<=30
        if command[1]=='rev-parse':return (a.source+'\n').encode()
        if command[1]=='show':return blobs[command[2].split(':',1)[1]]
        if command[1]=='ls-remote':return (a.source+'\trefs/heads/main\n').encode()
        if command[1]=='merge-base':return b''
        raise AssertionError(command)
    monkeypatch.setattr(L.subprocess,'check_output',git);monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    return a,blobs,plan


def test_complete_original_snapshot_import_inventory(tmp_path,monkeypatch):
    a,blobs,plan=fixture(tmp_path,monkeypatch)
    actual,captured,remote=L.snapshot(tmp_path,a,time.monotonic()+10,[])
    assert actual==plan and captured==blobs and remote==a.source and len(captured)==7


def test_prepared_refused_before_worker(tmp_path,monkeypatch):
    a,blobs,plan=fixture(tmp_path,monkeypatch);plan['admission']='prepared'
    p=tmp_path/'codex/loop-gain-crosscheck/run-002-plan.json';p.write_text(json.dumps(plan));a.plan_sha256=L.sha(p)
    with pytest.raises(ValueError,match='prepared'):L.snapshot(tmp_path,a,time.monotonic()+10,[])


def test_changed_source_refused(tmp_path,monkeypatch):
    a,blobs,plan=fixture(tmp_path,monkeypatch);(tmp_path/L.WORKER_FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='committed source'):L.snapshot(tmp_path,a,time.monotonic()+10,[])
