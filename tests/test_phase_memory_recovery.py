"""Resource-only EXP014 recovery regression; no target solves, fits or network."""
import ast
import copy
import json
import os
from pathlib import Path
import sys

import pytest

TEST_ROOT=Path(__file__).resolve().parents[1]
REPO=TEST_ROOT if (TEST_ROOT/'tools/run_phase_memory.py').exists() else Path.cwd()
sys.path[:0]=[str(TEST_ROOT/'tools'),str(REPO/'tools'),str(REPO/'tests')]
import run_phase_memory as original
import run_phase_memory_recovery as recovery
import test_phase_memory_runner as fixtures

NEW_SOURCES=['tools/run_phase_memory_recovery.py','tests/test_phase_memory_recovery.py','docs/EXP-014-operational-amendment.md']
CAPS={'launch_available_ram_bytes':16,'reserve_ram_bytes':8,'owned_rss_bytes':12}
OLD_CAPS={'launch_available_ram_bytes':8,'reserve_ram_bytes':4,'owned_rss_bytes':2}


def test_full_module_ast_identical_except_declared_memory_and_freeze_administration():
    before=ast.parse(Path(original.__file__).read_text());after=ast.parse(Path(recovery.__file__).read_text())
    changed=[]
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                node.value=node.value.replace('EXP-014-recovery-plan.json','EXP-014-plan.json')
                node.value=node.value.replace('EXP-014-recovery-review.json','EXP-014-local-review.json')
            return node
        def visit_keyword(self,node):
            if node.arg in CAPS and isinstance(node.value,ast.BinOp) and isinstance(node.value.right,ast.Name) and node.value.right.id=='GIB':
                assert ast.unparse(node.value)==f'{CAPS[node.arg]} * GIB'
                node.value=ast.parse(f'{OLD_CAPS[node.arg]} * GIB',mode='eval').body;changed.append(node.arg)
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='SOURCE_PATHS' for t in node.targets):
                entries=node.value.args[0].args[0].elts
                assert [v.value for v in entries[:3]]==NEW_SOURCES
                del entries[:3]
            return self.generic_visit(node)
    # Only the descriptive module docstring is exempt; every import,
    # assignment, class, function, nested function and CLI statement is bound.
    assert isinstance(before.body[0],ast.Expr) and isinstance(after.body[0],ast.Expr)
    before.body.pop(0);after.body.pop(0)
    normalized=Normalize().visit(after)
    assert ast.dump(before,include_attributes=False)==ast.dump(normalized,include_attributes=False)
    assert sorted(changed)==sorted(CAPS)
    assert recovery.SOURCE_PATHS==NEW_SOURCES+original.SOURCE_PATHS


def test_science_and_fresh_complete_census_are_unchanged():
    expected=copy.deepcopy(original.CONTRACT)
    expected.update({key:value*recovery.GIB for key,value in CAPS.items()})
    assert recovery.CONTRACT==expected
    assert recovery.CONTRACT['producer_seconds']==1800
    assert recovery.CONTRACT['reference_task_seconds']==120 and recovery.CONTRACT['trace_task_seconds']==60
    assert recovery.CONTRACT['raw_budget_bytes']==2*recovery.GIB
    assert recovery.CONTRACT['workers']==4 and recovery.CONTRACT['thread_limit']==1
    plan={'tasks':recovery.acquire.build_tasks()}
    assert original.stages(plan)==recovery.stages(plan)
    assert {k:len(v) for k,v in recovery.stages(plan).items()}==dict(references=2,propagators=16,traces=912)
    assert recovery.PLAN.name=='EXP-014-recovery-plan.json' and recovery.REVIEW.name=='EXP-014-recovery-review.json'


def mocked_rss(monkeypatch,bytes_used,available):
    state=dict(available_ram_bytes=available,total_ram_bytes=96*recovery.GIB,disk_free_bytes=100*recovery.GIB)
    monkeypatch.setattr(recovery,'resources',lambda *a:dict(state))
    monkeypatch.setattr(original,'resources',lambda *a:dict(state))
    pid=os.getpid();identity=dict(pid=pid,start_ticks='synthetic',boot_id='synthetic')
    monkeypatch.setattr(recovery.work,'identity',lambda value:identity if value==pid else None)
    read_text=Path.read_text
    pages=bytes_used//os.sysconf('SC_PAGE_SIZE')
    def fake(self,*a,**kw):
        if str(self)==f'/proc/{pid}/task/{pid}/children':return ''
        if str(self)==f'/proc/{pid}/statm':return f'0 {pages} 0 0 0 0 0'
        return read_text(self,*a,**kw)
    monkeypatch.setattr(Path,'read_text',fake)


def test_measured_sumrss_fails_original_and_passes_new_bound_without_allocating(tmp_path,monkeypatch):
    mocked_rss(monkeypatch,6062612480,24*recovery.GIB)
    result=recovery.resource_check(tmp_path,launch=True)
    assert result['owned_rss_bytes']==6062612480 and result['shared_pages_counted_per_process'] is True
    with pytest.raises(RuntimeError,match='hard cap'):original.resource_check(tmp_path,launch=True)


@pytest.mark.parametrize('used,available,launch,message',[
    (13*1024**3,24*1024**3,False,'hard cap'),
    (1024**3,15*1024**3,True,'headroom'),
    (1024**3,7*1024**3,False,'headroom')])
def test_revised_resource_limits_remain_hard_caps(tmp_path,monkeypatch,used,available,launch,message):
    mocked_rss(monkeypatch,used,available)
    with pytest.raises(RuntimeError,match=message):recovery.resource_check(tmp_path,launch=launch)


def test_recovery_preflight_failure_retains930_rows_and_replays_its_own_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'run',recovery)
    path=fixtures.setup_run(tmp_path,monkeypatch)
    # The fixture supplies only fabricated metadata; bind recovery's distinct
    # committed-plan lookup explicitly and reject every other subprocess.
    def committed(command):
        if command[-1]=='1'*40+':experiments/hopfield/EXP-014-recovery-plan.json':return path.read_bytes()
        fixtures.forbidden()
    monkeypatch.setattr(recovery.subprocess,'check_output',committed)
    monkeypatch.setattr(recovery,'resource_check',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic recovery preflight stop')))
    monkeypatch.setattr(recovery.work,'execute',fixtures.forbidden)
    monkeypatch.setattr(recovery.work,'build_libraries',fixtures.forbidden)
    result=recovery.run(path,tmp_path/'data','attempt-002')
    directory=tmp_path/'data/EXP-014/attempt-002'
    assert result['complete'] is False and result['qualified'] is False
    assert 'recovery preflight stop' in result['error']
    rows=recovery.read(directory/'all-terminal-records.json')
    assert sum(map(len,rows.values()))==930
    assert all(r['status']=='not_started_dependency_failure' for group in rows.values() for r in group)
    assert recovery.verify(directory)['tasks']==930
    assert len(result['scientific']['paired_readouts'])==5184
    assert len(result['scientific']['signed_response_readouts'])==1728
    with pytest.raises(FileExistsError):recovery.external_attempt(tmp_path/'data','attempt-002')
