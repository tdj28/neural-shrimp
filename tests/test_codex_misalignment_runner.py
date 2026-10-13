import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

SRC=Path(__file__).resolve().parents[1]/'codex/misalignment-audit'
def module(name):
    sp=importlib.util.spec_from_file_location('audit_'+name,SRC/(name+'.py'))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def test_prepared_card_rejected_before_limits(monkeypatch):
    r=module('run')
    monkeypatch.setattr(r.resource,'setrlimit',lambda *a:pytest.fail('unaccepted card changed limits'))
    with pytest.raises(ValueError,match='acceptance'):
        r.configure({'status':'prepared'})


def test_producer_tools_are_not_imported():
    for p in SRC.glob('*.py'):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
            assert not any(x.startswith(('tools','p1_','explore_llm','torch','transformers')) for x in names)


def test_real_tiny_owned_process_shutdown():
    launch=module('launch')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    try:
        ident=launch.process(child.pid)
        assert ident['pgrp']==child.pid and ident['session']==child.pid
        launch.stop_owned(child,ident)
        assert child.poll() is not None
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_changed_birth_identity_never_signalled():
    launch=module('launch')
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    try:
        ident=launch.process(child.pid);ident['starttime']-=1
        with pytest.raises(RuntimeError,match='ownership uncertain'):launch.stop_owned(child,ident)
        assert child.poll() is None
    finally:
        child.terminate();child.wait()


def test_protective_memory_floors_precede_certificate_floor():
    launch=module('launch')
    gib=1024**3
    assert launch.START_AVAILABLE==20*gib and launch.RUN_AVAILABLE==16*gib
    launch.available_gate(20*gib)
    launch.available_gate(16*gib,running=True)
    for n,running in [(20*gib-1,False),(16*gib-1,True)]:
        with pytest.raises(MemoryError,match='protective'):launch.available_gate(n,running)
