"""Queue ownership, resource and failure retention checks without a model."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import batch_fate_finish as finish  # noqa: E402


def test_process_identity_uses_birth_ticks_and_full_command(tmp_path):
    p = tmp_path / "17"
    p.mkdir()
    fields = ["S"] + ["0"] * 18 + ["123456"] + ["0"] * 4
    (p / "stat").write_text("17 (python (worker)) " + " ".join(fields))
    expected = [str(finish.PYTHON), "tools/batch_fate.py", *finish.MAIN_OPTIONS]
    (p / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in expected) + b"\0")
    ident = finish.process_identity(17, tmp_path)
    assert finish.check_identity(ident, 123456)
    with pytest.raises(ValueError, match="reused"):
        finish.check_identity(ident, 999)
    with pytest.raises(ValueError, match="not the owned"):
        finish.check_identity((123456, ["another-job"]), 123456)
    assert finish.process_identity(18, tmp_path) is None
    assert not finish.check_identity(None, 123456)


def test_memory_uses_available_not_free(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal: 96000000 kB\nMemFree: 1000 kB\nMemAvailable: 12582912 kB\n")
    assert finish.available_gib(p) == 12


def test_gpu_occupancy_fails_before_other_queries(monkeypatch):
    monkeypatch.setattr(finish.subprocess, "check_output", lambda *a, **k: "17, other-job, 500 MiB\n")
    with pytest.raises(ValueError, match="already present"):
        finish.resource_snapshot()


def test_bounded_child_failure_retains_log(tmp_path):
    events = []
    log = tmp_path / "failed.log"
    with pytest.raises(ValueError, match="exited 3"):
        finish.run_bounded([sys.executable, "-c", "print('failure evidence'); raise SystemExit(3)"], log,
                           5, lambda state, **data: events.append((state, data)))
    assert "failure evidence" in log.read_text()
    assert events[0][0] == "child_started"
    with pytest.raises(FileExistsError):
        finish.run_bounded([sys.executable, "-c", "pass"], log, 5, lambda *a, **k: None)


def test_child_timeout_terminates_only_owned_child(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        finish.run_bounded([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "timeout.log",
                           .02, lambda *a, **k: None)


def test_failed_preflight_persists_status_and_refuses_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(finish.summary, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", ["finish", "--wait-pid", "17", "--wait-start-ticks", "123",
                                     "--source-commit", "invalid", "--out-run", "followup"])
    with pytest.raises(ValueError, match="full lowercase"):
        finish.main()
    status = json.loads((tmp_path / "explore/followup/status.json").read_text())
    assert status["status"] == "failed"
    assert "source commit" in status["latest"]["error"]
    assert (tmp_path / "explore/followup/events.jsonl").exists()
    with pytest.raises(FileExistsError):
        finish.main()


def mock_resources(monkeypatch, ram_gib, gpu_mib):
    monkeypatch.setattr(finish, "available_gib", lambda: ram_gib)
    def query(command, **kwargs):
        if any("query-compute-apps" in arg for arg in command):
            return ""
        if "--query-gpu=memory.free" in command:
            return str(gpu_mib) + "\n"
        return "RTX 2000 Ada, test-uuid, test-driver, 9000 MiB, 0 %\n"
    monkeypatch.setattr(finish.subprocess, "check_output", query)
    monkeypatch.setattr(finish, "version", lambda name: "test")


@pytest.mark.parametrize("ram,gpu,reason", [(23.9, 9000, "24 GiB"), (24, 8191, "8 GiB")])
def test_startup_reserves_headroom_for_model(monkeypatch, ram, gpu, reason):
    mock_resources(monkeypatch, ram, gpu)
    with pytest.raises(ValueError, match=reason):
        finish.resource_snapshot()


def test_startup_at_reserved_threshold_passes(monkeypatch):
    mock_resources(monkeypatch, 24, 8192)
    result = finish.resource_snapshot()
    assert result["available_ram_gib"] == 24
    assert result["free_gpu_mib"] == 8192


def test_memory_guard_stops_owned_model_child_and_preserves_evidence(tmp_path, monkeypatch):
    measurements = iter([24, 11.5])
    monkeypatch.setattr(finish, "available_gib", lambda: next(measurements))
    events = []
    log = tmp_path / "memory-failure.log"
    with pytest.raises(RuntimeError, match="below 12 GiB"):
        finish.run_bounded([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
                           log, 5, lambda state, **info: events.append((state, info)),
                           monitor_memory=True, poll_seconds=.05)
    assert "started" in log.read_text()
    assert events[-1][0] == "resource_guard_failed"
    assert events[-1][1]["available_ram_gib"] == 11.5
    assert events[0][1]["pid"] == events[-1][1]["pid"]
    assert finish.process_identity(events[0][1]["pid"]) is None


def test_memory_guard_allows_exact_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "available_gib", lambda: 12)
    finish.run_bounded([sys.executable, "-c", "print('ok')"], tmp_path / "floor.log", 5,
                       lambda *a, **k: None, monitor_memory=True, poll_seconds=.01)



@pytest.mark.parametrize("mode", [[], ["--main-complete", "--wait-pid", "17"],
                                  ["--main-complete", "--wait-start-ticks", "123"],
                                  ["--wait-pid", "17"], ["--wait-pid", "0", "--wait-start-ticks", "123"]])
def test_completion_and_wait_modes_cannot_be_ambiguous(mode):
    with pytest.raises(SystemExit) as exc:
        finish.parse_args(["--source-commit", "a" * 40, "--out-run", "followup", *mode])
    assert exc.value.code == 2


def test_completed_mode_does_not_invent_process_identity():
    args = finish.parse_args(["--source-commit", "a" * 40, "--out-run", "followup", "--main-complete"])
    assert args.main_complete and args.wait_pid is None and args.wait_start_ticks is None


def test_completed_mode_still_requires_full_main_gate_and_preserves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(finish.summary, "DATA", tmp_path)
    monkeypatch.setattr(sys, "argv", ["finish", "--main-complete", "--source-commit", "a" * 40,
                                     "--out-run", "followup"])
    monkeypatch.setattr(finish, "source_snapshot", lambda commit: {"commit": commit})
    def forbidden(*args, **kwargs):
        pytest.fail("completed mode must not inspect a PID or launch a child before main gates")
    monkeypatch.setattr(finish, "process_identity", forbidden)
    monkeypatch.setattr(finish, "run_bounded", forbidden)
    def incomplete(run):
        raise ValueError("incomplete main fixture")
    monkeypatch.setattr(finish.summary, "load", incomplete)
    with pytest.raises(ValueError, match="incomplete main"):
        finish.main()
    output = tmp_path / "explore/followup"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert events[0]["state"] == "validating_completed_main"
    assert events[0]["process_identity"] == "not_claimed"
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["args"]["main_complete"]
    assert status["args"]["wait_pid"] is None
