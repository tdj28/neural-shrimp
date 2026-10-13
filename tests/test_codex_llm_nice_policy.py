"""Prospective process-priority policy only; no Git, model or target process."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

L=module('nice_policy_launcher',ROOT/'codex/certify-llm/launch.py')
C=module('nice_policy_campaign',ROOT/'codex/certify-llm/derivative_campaign.py')

@pytest.mark.parametrize('inherited',[0,10,12])
def test_launcher_and_worker_priority_is_idempotent_or_refused(monkeypatch,inherited):
    current=[inherited];changes=[]
    monkeypatch.setattr(L.os,'getpriority',lambda *args:current[0])
    def nice(delta):
        changes.append(delta);current[0]+=delta;return current[0]
    monkeypatch.setattr(L.os,'nice',nice)
    if inherited>10:
        with pytest.raises(RuntimeError,match='inherited'):L.ensure_nice10()
        assert current[0]==inherited and changes==[]
    else:
        L.ensure_nice10()  # launcher/Git parent
        L.ensure_nice10()  # worker inherited the launcher priority
        assert current[0]==10 and changes==([10]if inherited==0 else [])

@pytest.mark.parametrize('inherited',[0,10,12])
def test_controller_requires_exact_ten_without_changing_priority(monkeypatch,inherited):
    monkeypatch.setattr(C.os,'getpriority',lambda *args:inherited)
    monkeypatch.setattr(C.os,'nice',lambda *args:pytest.fail('controller must require its launch envelope'))
    if inherited==10:C.require_controller_nice10()
    else:
        with pytest.raises(RuntimeError,match='controller'):C.require_controller_nice10()


def test_launcher_refuses_failed_priority_establishment(monkeypatch):
    monkeypatch.setattr(L.os,'getpriority',lambda *args:0)
    monkeypatch.setattr(L.os,'nice',lambda *args:0)
    with pytest.raises(RuntimeError,match='establish'):L.ensure_nice10()


def test_real_launcher_paths_normalize_before_git_and_do_not_add_again(tmp_path,monkeypatch):
    old=module('nice_launcher_existing_controls',ROOT/'tests/test_codex_llm_parallel_launch.py')
    owned,calls,signals=old.case(tmp_path,monkeypatch)
    current=[0];changes=[];limits=[]
    monkeypatch.setattr(old.launch.os,'getpriority',lambda *args:current[0])
    def nice(delta):changes.append(delta);current[0]+=delta;return current[0]
    monkeypatch.setattr(old.launch.os,'nice',nice)
    actual_output=old.launch.subprocess.check_output
    def git_at_ten(*args,**kwargs):
        assert current[0]==10
        return actual_output(*args,**kwargs)
    monkeypatch.setattr(old.launch.subprocess,'check_output',git_at_ten)
    monkeypatch.setattr(old.launch.resource,'setrlimit',lambda kind,value:limits.append((kind,value)))
    with pytest.raises(SystemExit)as outcome:old.launch.main()
    assert outcome.value.code==0
    child=next(c for c in calls if hasattr(c,'kwargs'))
    child.kwargs['preexec_fn']()
    assert current[0]==10 and changes==[10]
    assert limits==[(old.launch.resource.RLIMIT_AS,(18*1024**3,)*2)]*2  # launcher then worker
    assert signals==[]


def test_controller_refuses_before_output_or_other_admission(monkeypatch):
    monkeypatch.setattr(C.os,'getpriority',lambda *args:0)
    monkeypatch.setattr(C.time,'monotonic',lambda:pytest.fail('controller performed work before priority refusal'))
    with pytest.raises(RuntimeError,match='controller'):C.main()
