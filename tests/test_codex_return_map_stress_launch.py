"""Narrow changed-mode controls. No model, target data, Git or subprocess run."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import pytest

HERE=Path(__file__).resolve().parent
FILE=HERE/'stress_launch.py' if (HERE/'stress_launch.py').exists() else HERE.parent/'codex/return-map-crosscheck/stress_launch.py'
spec=importlib.util.spec_from_file_location('stress_launch_test',FILE)
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
REPO=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
BASE=REPO/'codex/return-map-crosscheck/launch.py'


def args(index):
    name=f'stress-cell{index:02d}-001'
    return argparse.Namespace(attempt=name,plan=name+'-plan.json',source='a'*40,plan_sha256='b'*64)


def complete(index):
    log_t,q=L.CELLS[index]
    status=dict(status='completed',computational_completed=True,certificate_passed=False,
        experimental_claims=False,mode='stress',cell_index=index,source_commit='a'*40)
    result=dict(transient_steps=1000,recorded_steps=2000,native_step_calls=3000,row_forwards=7000,seed=0,
        optional_cell_index=index,expected_producer_lead_period=q,
        optional_identity=dict(index=index,alpha=.0655,log10_temperature=log_t,expected_producer_lead_period=q,
            attempt=f'stress-cell{index:02d}-001',expected_period_is_comparison_only=True),
        offsets=[1e-6,1e-9],post_asset_hashes_equal=True,
        disjoint_windows=[dict(start=i,stop=i+500) for i in range(0,2000,500)],
        running_means=[dict(stop=i) for i in range(100,2001,100)],
        expected_producer_clock_matches=False,lead_period=0,numerical_repeating_text_chaos=False)
    return status,result


@pytest.mark.parametrize('index',[4,13])
def test_fixed_specification_and_negative_science(index):
    a=args(index);L.validate_arguments(a);s=L.specification(a.plan)
    assert s['attempt']==a.attempt and s['cell_index']==index and s['worker']=='stress_runtime.py'
    assert s['schema']=='codex-return-map-stress-v1' and len(s['workers'])==6
    assert s['caps']==dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=4200,
        minimum_start_available_gib=36,minimum_runtime_available_gib=16,maximum_output_mib=512)
    status,result=complete(index)
    L.validate_completion(status,result,'stress','a'*40,index)
    for key in ('cell_index','mode','computational_completed'):
        bad=copy.deepcopy(status);bad[key]=None
        with pytest.raises(RuntimeError):L.validate_completion(bad,result,'stress','a'*40,index)
    bad=copy.deepcopy(result);bad['running_means'].pop()
    with pytest.raises(RuntimeError):L.validate_completion(status,bad,'stress','a'*40,index)
    bad=copy.deepcopy(result);bad['optional_identity']['log10_temperature']=.2
    with pytest.raises(RuntimeError):L.validate_completion(status,bad,'stress','a'*40,index)
    with pytest.raises(RuntimeError):L.validate_completion(status,result,'stress','a'*40,13 if index==4 else 4)


def test_wrong_plan_attempt_or_arbitrary_retry_refuses():
    for change in [dict(plan='trajectory-001-plan.json'),dict(plan='stress-cell05-001-plan.json'),
        dict(plan='stress-cell04-002-plan.json'),dict(attempt='stress-cell13-001'),dict(attempt='../outside')]:
        a=args(4);vars(a).update(change)
        with pytest.raises(ValueError):L.validate_arguments(a)


def fixture(tmp_path,monkeypatch,index):
    a=args(index);s=L.specification(a.plan);blobs={}
    for path,pin in L.WORKER_PINS.items():
        source=REPO/path
        if path.endswith('/stress_runtime.py') and not source.exists():source=Path('/tmp/task19-optional-adapter/stress_runtime.py')
        blobs[path]=source.read_bytes();assert hashlib.sha256(blobs[path]).hexdigest()==pin
    blobs[L.OWNERSHIP_FILE]=(REPO/L.OWNERSHIP_FILE).read_bytes()
    blobs['codex/return-map-crosscheck/stress_launch.py']=FILE.read_bytes()
    for name in [s['card'],s['dispatch']]:blobs['codex/return-map-crosscheck/'+name]=b'synthetic operational bytes'
    plan=dict(schema=s['schema'],admission='reviewed',mode='stress',cell_index=index,
        envelope=s['caps'],source_sha256=dict(L.WORKER_PINS))
    rel='codex/return-map-crosscheck/'+a.plan;blobs[rel]=json.dumps(plan).encode();a.plan_sha256=hashlib.sha256(blobs[rel]).hexdigest()
    for name,raw in blobs.items():p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    def git(command,**kw):
        assert kw['timeout']<=30
        if command[1]=='rev-parse':return (a.source+'\n').encode()
        if command[1]=='show':return blobs[command[2].split(':',1)[1]]
        if command[1]=='ls-remote':return (a.source+'\trefs/heads/main\n').encode()
        if command[1]=='merge-base':return b''
        raise AssertionError(command)
    monkeypatch.setattr(L.subprocess,'check_output',git);monkeypatch.setattr(L,'available',lambda:40*L.GIB)
    return a,blobs,plan


@pytest.mark.parametrize('index',[4,13])
def test_exact_eleven_file_snapshot_and_wrong_cell_prepared_pin_refusal(tmp_path,monkeypatch,index):
    a,blobs,plan=fixture(tmp_path,monkeypatch,index)
    got,captured,remote=L.snapshot(tmp_path,a,time.monotonic()+30,[])
    assert got==plan and captured==blobs and len(captured)==11 and remote==a.source
    def changed(change):
        bad=copy.deepcopy(plan);change(bad)
        path=tmp_path/'codex/return-map-crosscheck'/a.plan;path.write_text(json.dumps(bad));a.plan_sha256=L.sha(path)
        with pytest.raises(ValueError):L.snapshot(tmp_path,a,time.monotonic()+30,[])
    changed(lambda p:p.update(admission='prepared'))
    changed(lambda p:p.update(cell_index=13 if index==4 else 4))
    changed(lambda p:p['source_sha256'].__setitem__('codex/return-map-crosscheck/stress_runtime.py','b'*64))


def test_lifecycle_and_resource_functions_ast_identical():
    old=ast.parse(BASE.read_text());new=ast.parse(FILE.read_text())
    def functions(tree):return {f.name:ast.dump(f,include_attributes=False) for f in tree.body if isinstance(f,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    for name in ('sha','write','module','establish_limits','available','validate_arguments','guard','close_worker','monitor','main'):
        assert a[name]==b[name],name
    # Execution changes only the completion index argument and recorded receipt index.
    old_execute=next(f for f in old.body if isinstance(f,ast.FunctionDef) and f.name=='execute')
    new_execute=next(f for f in new.body if isinstance(f,ast.FunctionDef) and f.name=='execute')
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='validate_completion':node.args=node.args[:4]
            if isinstance(node.func,ast.Name) and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg!='cell_index']
            return node
    assert ast.dump(Normalize().visit(old_execute),include_attributes=False)==ast.dump(Normalize().visit(new_execute),include_attributes=False)


@pytest.mark.parametrize('index',[4,13])
def test_dispatch_changes_only_fixed_paths(index):
    base=(REPO/'codex/return-map-crosscheck/trajectory-001-dispatch.sh').read_text()
    path=HERE/f'stress-cell{index:02d}-001-dispatch.sh'
    if not path.exists():path=FILE.parent/path.name
    got=path.read_text()
    assert got==base.replace('trajectory-001',f'stress-cell{index:02d}-001').replace('crosscheck/launch.py','crosscheck/stress_launch.py')
    assert 'setsid --wait' in got and 'nice -n' not in got and 'noclobber' in got
