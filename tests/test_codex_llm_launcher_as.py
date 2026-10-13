"""Early launcher AS controls: mocked limits/Git/processes, no model or target."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

L=module('launcher_as_policy',ROOT/'codex/certify-llm/launch.py')
GiB=1024**3

@pytest.mark.parametrize('soft,hard',[(GiB,18*GiB),(L.resource.RLIM_INFINITY,L.resource.RLIM_INFINITY),(GiB,L.resource.RLIM_INFINITY)])
def test_inherited_soft_or_unlimited_is_bounded_before_git(monkeypatch,soft,hard):
    current=[soft,hard];calls=[]
    monkeypatch.setattr(L.resource,'getrlimit',lambda kind:tuple(current))
    def setlimit(kind,value):calls.append((kind,value));current[:]=value
    monkeypatch.setattr(L.resource,'setrlimit',setlimit)
    L.ensure_launcher_address_space()
    assert current==[18*GiB,18*GiB]
    assert calls==[(L.resource.RLIMIT_AS,(18*GiB,18*GiB))]


def test_too_small_hard_limit_refuses_without_setrlimit(monkeypatch):
    monkeypatch.setattr(L.resource,'getrlimit',lambda kind:(GiB,17*GiB))
    monkeypatch.setattr(L.resource,'setrlimit',lambda *args:pytest.fail('cannot raise inherited hard cap'))
    with pytest.raises(RuntimeError,match='inherited hard'):L.ensure_launcher_address_space()


def test_main_refuses_small_hard_before_git_or_attempt(tmp_path,monkeypatch):
    T=module('launcher_as_old_controls_refusal',ROOT/'tests/test_codex_llm_parallel_launch.py')
    owned,calls,signals=T.case(tmp_path,monkeypatch)
    monkeypatch.setattr(T.launch.resource,'getrlimit',lambda kind:(GiB,17*GiB))
    monkeypatch.setattr(T.launch.resource,'setrlimit',lambda *args:pytest.fail('hard-cap refusal must not set limits'))
    with pytest.raises(RuntimeError,match='inherited hard'):T.launch.main()
    assert calls==[] and signals==[] and not owned.exists()


def test_git_inherits_18gib_and_worker_keeps_exact_envelope(tmp_path,monkeypatch):
    T=module('launcher_as_old_controls_admission',ROOT/'tests/test_codex_llm_parallel_launch.py')
    owned,calls,signals=T.case(tmp_path,monkeypatch)
    current=[GiB,18*GiB];limits=[]
    monkeypatch.setattr(T.launch.resource,'getrlimit',lambda kind:tuple(current))
    def setlimit(kind,value):limits.append((kind,value));current[:]=value
    monkeypatch.setattr(T.launch.resource,'setrlimit',setlimit)
    output=T.launch.subprocess.check_output;run=T.launch.subprocess.run
    def check_output(*args,**kwargs):
        assert current==[18*GiB,18*GiB]
        assert T.launch.os.getpriority(T.launch.os.PRIO_PROCESS,0)==10
        return output(*args,**kwargs)
    def git_run(*args,**kwargs):
        assert current==[18*GiB,18*GiB]
        return run(*args,**kwargs)
    monkeypatch.setattr(T.launch.subprocess,'check_output',check_output)
    monkeypatch.setattr(T.launch.subprocess,'run',git_run)
    with pytest.raises(SystemExit)as result:T.launch.main()
    assert result.value.code==0
    child=next(c for c in calls if hasattr(c,'kwargs'));child.kwargs['preexec_fn']()
    assert limits==[(T.launch.resource.RLIMIT_AS,(18*GiB,18*GiB))]*2
    assert T.launch.os.getpriority(T.launch.os.PRIO_PROCESS,0)==10 and not signals
