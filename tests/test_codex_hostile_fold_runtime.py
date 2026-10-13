"""Synthetic Task7 runtime admission and retention; no pretrained model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'codex/hostile-fold-crosscheck'
spec=importlib.util.spec_from_file_location('task7_fold_runtime_test',TASK/'fold_runtime.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def plan():return json.loads((TASK/'fold-001-plan.json').read_text())


def seal(path,source='a'*40,**extra):
    receipt=dict(exit_code=0,timed_out=False,source_commit=source,artifacts=[],**extra)
    for file in sorted(path.rglob('*')):
        if file.is_file() and file!=path/'receipt.json':
            receipt['artifacts'].append(dict(path=str(file.relative_to(path)),bytes=file.stat().st_size,sha256=R.sha(file)))
    (path/'receipt.json').write_text(json.dumps(receipt))
    return R.sha(path/'receipt.json')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def resource_fixture(tmp_path,monkeypatch):
    profile=tmp_path/'profile';historical=tmp_path/'historical';profile.mkdir();historical.mkdir()
    p=plan()
    sources=[name for name in R.SOURCES if not name.endswith(('fold_runtime.py','fold_worker.py'))]
    for name in sources:
        path=profile/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        p['source_sha256'][name]=R.sha(path)
    write(profile/'run/original-plan.json',dict(source_sha256={s:p['source_sha256'][s]for s in sources},
        seed_sha256=p['seed_sha256'],vector_sha256=p['vector_sha256']))
    rows=[]
    for name in R.OPERATIONS:
        shape=(16,896)if name=='batched-composed-jvp' else(896,)
        filename=name+'.npz';key='output/0'if name=='critical-gradient-ad'else'output'
        np.savez(profile/'run'/filename,**{key:np.zeros(shape,dtype=np.float64)})
        desc=dict(file=filename,key=key,shape=list(shape),dtype='float64')
        output=[desc,0.]if name=='critical-gradient-ad' else desc
        write(profile/'run'/f'{name}.json',dict(output=output,array_file_sha256=R.sha(profile/'run'/filename)))
        rows.append(dict(operation=name,seconds=1.,max_rss_kib=1000))
    write(profile/'run/status.json',dict(status='resource_profile_completed',fold_qualified=False,
        certificate_passed=False,normal_form_coefficients_computed=False,resource_probe_only=True,rows=rows))
    write(profile/'run/model-loaded.json',dict(dimension=896,layers=24,vocabulary=151936,dtype='torch.float64',steering_layer=12))
    phash=seal(profile,R.PROFILE_SOURCE,status='resource_pilot_completed',worker_termination_verified=True,
        uncertain_ownership=[],scientific_fold_result=False,resource_probe_only=True)
    path=historical/'source/codex/crosscheck/qwen_map.py';path.parent.mkdir(parents=True);path.write_text('codex/crosscheck/qwen_map.py')
    write(historical/'run/throughput.json',[dict(batch=128,peak_rss_kib=6274700,seconds=2.)])
    hhash=seal(historical)
    monkeypatch.setattr(R,'PROFILE',profile);monkeypatch.setattr(R,'PROFILE_SHA',phash)
    monkeypatch.setattr(R,'HISTORICAL',historical);monkeypatch.setattr(R,'HISTORICAL_SHA',hhash)
    return p,profile,historical


def test_concrete_plan_and_complete_source_set():
    p=R.validate_plan(plan());assert p['numerics']==dict(chunk=128,max_newton=4,max_backtracks=2)
    assert p['envelope']['timeout_seconds']==1800
    for name,pin in p['source_sha256'].items():assert R.sha(ROOT/name)==pin


@pytest.mark.parametrize('mutation',[
    lambda p:p['envelope'].update(cpu_threads=True),
    lambda p:p['envelope'].update(timeout_seconds=1801),
    lambda p:p['numerics'].update(chunk=16),
    lambda p:p['numerics'].update(max_newton=True),
    lambda p:p.update(scope='passages'),
    lambda p:p['source_sha256'].pop('codex/hostile-fold-crosscheck/fold_worker.py'),
    lambda p:p['profile_authority'].update(receipt_sha256='f'*64),
    lambda p:p['historical_memory_authority'].update(batch=64),
])
def test_plan_scope_resources_and_external_pins_refused(mutation):
    p=plan();mutation(p)
    with pytest.raises(ValueError):R.validate_plan(p)


def test_actual_inventory_and_full_synthetic_profile_array_admission(tmp_path,monkeypatch):
    p,_,_=resource_fixture(tmp_path,monkeypatch)
    result=R.admit_resource_evidence(p)
    assert result['profile_artifact_count']>15
    assert result['current_batch16_linear_dense_seconds']==56
    assert result['historical_batch128']['batch']==128
    assert not result['certificate_passed'] and not result['git_commit_blob_verification_performed']


@pytest.mark.parametrize('kind',['failed','timeout','incomplete','unresolved','extra','changed','nonfinite','source_mismatch','badshape'])
def test_corrupt_or_incomplete_resource_authority_refused(tmp_path,monkeypatch,kind):
    p,profile,_=resource_fixture(tmp_path,monkeypatch)
    receipt=json.loads((profile/'receipt.json').read_text())
    if kind=='failed':receipt['exit_code']=1
    if kind=='timeout':receipt['timed_out']=True
    if kind=='unresolved':receipt['uncertain_ownership']=['unreaped']
    if kind in ('failed','timeout','unresolved'):
        write(profile/'receipt.json',receipt);monkeypatch.setattr(R,'PROFILE_SHA',R.sha(profile/'receipt.json'))
    if kind=='extra':(profile/'extra.txt').write_text('unlisted')
    if kind=='changed':(profile/'run/status.json').write_text('{}')
    if kind in ('incomplete','source_mismatch','badshape','nonfinite'):
        if kind=='incomplete':
            path=profile/'run/status.json';value=json.loads(path.read_text());value['rows'].pop();write(path,value)
        if kind=='source_mismatch':p['source_sha256']['codex/crosscheck/qwen_map.py']='e'*64
        if kind in ('badshape','nonfinite'):
            data=np.zeros((895 if kind=='badshape' else 896,),dtype=np.float64)
            if kind=='nonfinite':data[0]=np.nan
            np.savez(profile/'run/period-map.npz',output=data)
            path=profile/'run/period-map.json';value=json.loads(path.read_text())
            value['array_file_sha256']=R.sha(profile/'run/period-map.npz');write(path,value)
        phash=seal(profile,R.PROFILE_SOURCE,status='resource_pilot_completed',worker_termination_verified=True,
            uncertain_ownership=[],scientific_fold_result=False,resource_probe_only=True)
        monkeypatch.setattr(R,'PROFILE_SHA',phash)
    with pytest.raises(ValueError):R.admit_resource_evidence(p)


def test_isolated_help_constructs_no_model():
    result=subprocess.run([sys.executable,'-I','-B',str(TASK/'fold_runtime.py'),'--help'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and '--plan' in result.stdout and '--output' in result.stdout
