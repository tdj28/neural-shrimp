import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("runpod_pod", Path(__file__).resolve().parents[1] / "tools" / "runpod_pod.py")
runpod_pod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runpod_pod)


def test_terminate_refuses_pods_not_created_by_the_tool(monkeypatch, tmp_path):
    ledger = tmp_path / "pods.jsonl"
    ledger.write_text(json.dumps({"event": "created", "pod_id": "ours123"}) + "\n")
    monkeypatch.setattr(runpod_pod, "LEDGER", ledger)
    calls = []
    monkeypatch.setattr(runpod_pod, "graphql", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(SystemExit, match="refusing"):
        runpod_pod.cmd_terminate(type("Args", (), {"pod": "someone-elses-pod"})())
    assert calls == []
    assert runpod_pod.owned_pods() == {"ours123"}


def test_owned_pods_empty_without_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(runpod_pod, "LEDGER", tmp_path / "missing.jsonl")
    assert runpod_pod.owned_pods() == set()


def test_create_requires_the_project_prefix():
    args = type("Args", (), {"name": "phase0", "pubkey_file": "unused"})()
    with pytest.raises(SystemExit, match="must start with"):
        runpod_pod.cmd_create(args)


def test_terminate_rechecks_the_name_prefix(monkeypatch, tmp_path):
    ledger = tmp_path / "pods.jsonl"
    ledger.write_text(json.dumps({"event": "created", "pod_id": "abc"}) + "\n")
    monkeypatch.setattr(runpod_pod, "LEDGER", ledger)
    monkeypatch.setattr(runpod_pod, "pod_status", lambda pod: {"id": pod, "name": "someone-else"})
    calls = []
    monkeypatch.setattr(runpod_pod, "graphql", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(SystemExit, match="not named"):
        runpod_pod.cmd_terminate(type("Args", (), {"pod": "abc"})())
    assert calls == []
