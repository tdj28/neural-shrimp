"""Offline guards: no provider account or network is used by these tests."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_runpod", Path(__file__).parents[1] / "tools" / "prepare_runpod.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "offline-test@example.invalid")
    git(repo, "config", "user.name", "Offline fixture")
    for name, data in {
        "tools/prepare_runpod.py": "# fixture\n",
        "tools/runpod_entrypoint.sh": "#!/bin/sh\n",
        "experiments/manifests/test.json": "{}\n",
        "src/neural_shrimp/__init__.py": "",
        "src/neural_shrimp/cli.py": "import pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); p.mkdir(); (p/'result.json').write_text('{}')\n",
        ".gitignore": ".env\n",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "offline fixture")
    return repo


def export(tmp_path, repo):
    archive = tmp_path / "bundle.tar"
    info = compute.bundle(repo, "experiments/manifests/test.json", [], archive)
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(archive) as handle:
        # This archive is produced in this same test, not downloaded input.
        for member in handle:
            assert member.isfile() and not member.name.startswith("/") and ".." not in Path(member.name).parts
            target = unpacked / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(handle.extractfile(member).read())
    return info, unpacked / "payload"


def quote(**updates):
    values = dict(unit_seconds=2.0, units=100, hourly_usd=1.0,
                  setup_seconds=30.0, retrieval_seconds=30.0, storage_transfer_usd=0.1,
                  slowdown_factor=2.0, hard_wall_seconds=600.0, hard_budget_usd=0.3,
                  no_progress_seconds=20.0, quote_source="https://example.invalid/test-quote",
                  quote_observed_at="2026-09-08T00:00:00Z", measurement_ref="fixture only",
                  measurement_host="offline test fixture")
    values.update(updates)
    return argparse.Namespace(**values)


def test_cost_arithmetic_and_no_invented_quote():
    result = compute.estimate(quote())
    assert result["base_seconds"] == 260
    assert result["slow_seconds"] == 460
    assert result["base_cost_usd"] == pytest.approx(260 / 3600 + 0.1)
    assert result["approval"].startswith("NOT APPROVED")
    with pytest.raises(ValueError, match="Missing quote_source"):
        compute.estimate(quote(quote_source=""))


@pytest.mark.parametrize("updates", [
    {"unit_seconds": float("nan")}, {"unit_seconds": float("inf")},
    {"units": -1}, {"units": 1.2}, {"slowdown_factor": 0.5},
    {"hard_budget_usd": 0.15}, {"hard_wall_seconds": 100},
    {"no_progress_seconds": 1}, {"retrieval_seconds": 0},
])
def test_invalid_or_underbudget_plan_refused(updates):
    with pytest.raises(ValueError):
        compute.estimate(quote(**updates))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "src/../x", "src\\x", "src/.env", "src/key.pem", "src/.git/config", "src//x"])
def test_unsafe_paths_refused(name):
    with pytest.raises(ValueError):
        compute.checked_path(name)


def test_bundle_binds_clean_commit_and_excludes_ignored_credentials(tmp_path):
    repo = repository(tmp_path)
    (repo / ".env").write_text("PRIVATE_TEST_VALUE=not-a-real-key\n")
    info, root = export(tmp_path, repo)
    receipt = compute.verify_bundle(root, info["source_commit"], info["receipt_sha256"])
    assert receipt["source_commit"] == git(repo, "rev-parse", "HEAD").decode().strip()
    assert ".env" not in receipt["files"]
    second = tmp_path / "second.tar"
    info2 = compute.bundle(repo, "experiments/manifests/test.json", [], second)
    assert info2["archive_sha256"] == info["archive_sha256"]
    with pytest.raises(FileExistsError):
        compute.bundle(repo, "experiments/manifests/test.json", [], second)
    (root / receipt["manifest"]).write_text('{"modified": true}')
    with pytest.raises(ValueError, match="integrity"):
        compute.verify_bundle(root, info["source_commit"], info["receipt_sha256"])


def test_dirty_worktree_refused(tmp_path):
    repo = repository(tmp_path)
    (repo / "untracked.txt").write_text("uncommitted")
    with pytest.raises(ValueError, match="clean"):
        compute.bundle(repo, "experiments/manifests/test.json", [], tmp_path / "bundle.tar")


def test_symlink_and_embedded_secret_refused(tmp_path):
    repo = repository(tmp_path)
    (repo / "src/link").symlink_to("/etc/passwd")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "symlink fixture")
    with pytest.raises(ValueError, match="regular"):
        compute.bundle(repo, "experiments/manifests/test.json", [], tmp_path / "bundle.tar")
    (repo / "src/link").unlink()
    (repo / "src/source.py").write_text("value='ghp_" + "x" * 40 + "'\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "dummy secret fixture")
    with pytest.raises(ValueError, match="credential"):
        compute.bundle(repo, "experiments/manifests/test.json", [], tmp_path / "bundle.tar")


def test_extra_file_commit_receipt_or_symlink_tampering_refused(tmp_path):
    info, root = export(tmp_path, repository(tmp_path))
    with pytest.raises(ValueError, match="commit"):
        compute.verify_bundle(root, "0" * 40, info["receipt_sha256"])
    with pytest.raises(ValueError, match="receipt hash"):
        compute.verify_bundle(root, info["source_commit"], "0" * 64)
    extra = root / "unexpected"
    extra.write_text("extra startup surface")
    with pytest.raises(ValueError, match="extra"):
        compute.verify_bundle(root, info["source_commit"], info["receipt_sha256"])
    extra.unlink()
    extra.symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="Symlink"):
        compute.verify_bundle(root, info["source_commit"], info["receipt_sha256"])


def ownership():
    return dict(task_id="fixture-task", pod_id="fixture-pod", preexisting_resource_ids=["someone-elses-pod"],
                create_response_sha256="a" * 64, billed_start_epoch=time.time(),
                controller_watchdog_ref="fixture only; no provider calls",
                runtime_image_digest="example.invalid/image@sha256:" + "b" * 64)


def test_resource_identity_ownership():
    record = ownership()
    compute.validate_ownership(record, "fixture-task", "fixture-pod")
    with pytest.raises(ValueError, match="RUNPOD_POD_ID"):
        compute.validate_ownership(record, "fixture-task", "other-pod")
    with pytest.raises(ValueError, match="different task"):
        compute.validate_ownership(record, "other-task", "fixture-pod")
    record["preexisting_resource_ids"].append("fixture-pod")
    with pytest.raises(ValueError, match="pre-existing"):
        compute.validate_ownership(record, "fixture-task", "fixture-pod")


def run_arguments(tmp_path, info, root):
    cost = compute.estimate(quote(unit_seconds=0.01, units=1, hourly_usd=1,
                                 setup_seconds=0, retrieval_seconds=1, slowdown_factor=1,
                                 hard_wall_seconds=20, no_progress_seconds=0.5))
    cost_path = tmp_path / "cost.json"
    owner_path = tmp_path / "ownership.json"
    cost_path.write_bytes(compute.json_bytes(cost))
    owner_path.write_bytes(compute.json_bytes(ownership()))
    return argparse.Namespace(bundle_root=root, expected_commit=info["source_commit"],
                              expected_receipt_sha256=info["receipt_sha256"], cost_report=cost_path,
                              ownership_receipt=owner_path, expected_cost_sha256=compute.sha256(cost_path.read_bytes()),
                              expected_ownership_sha256=compute.sha256(owner_path.read_bytes()), task_id="fixture-task",
                              approval_ref="offline fixture only; no paid launch", attempt_id="attempt-1",
                              output_base=tmp_path / "attempts")


def test_bounded_execution_preserves_attempt_and_hashes(tmp_path, monkeypatch):
    info, root = export(tmp_path, repository(tmp_path))
    args = run_arguments(tmp_path, info, root)
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture-pod")
    result = compute.run_workload(args)
    assert result["status"] == "process_exit_zero"
    assert result["pod_terminated"] is False
    ledger = json.loads((args.output_base / args.attempt_id / "SHA256.json").read_text())
    assert "results/result.json" in ledger["files"]
    with pytest.raises(FileExistsError):
        compute.run_workload(args)
    compute.verify_bundle(root, info["source_commit"], info["receipt_sha256"])


def test_no_progress_timeout_preserves_failure(tmp_path, monkeypatch):
    repo = repository(tmp_path)
    (repo / "src/neural_shrimp/cli.py").write_text("import time\ntime.sleep(30)\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "stall fixture")
    info, root = export(tmp_path, repo)
    args = run_arguments(tmp_path, info, root)
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture-pod")
    before = time.monotonic()
    result = compute.run_workload(args)
    assert time.monotonic() - before < 10
    assert result["status"] == "failed_or_incomplete"
    assert result["reason"] == "no_progress_timeout"
    assert result["exit_code"] != 0
    assert (args.output_base / args.attempt_id / "SHA256.json").exists()


def test_hard_compute_deadline_preserves_failure(tmp_path, monkeypatch):
    repo = repository(tmp_path)
    (repo / "src/neural_shrimp/cli.py").write_text("import time\ntime.sleep(30)\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "deadline fixture")
    info, root = export(tmp_path, repo)
    args = run_arguments(tmp_path, info, root)
    record = ownership()
    # Lifetime 20s - 1s retrieval - 5s termination - already billed 13.7s.
    record["billed_start_epoch"] = time.time() - 13.7
    args.ownership_receipt.write_bytes(compute.json_bytes(record))
    args.expected_ownership_sha256 = compute.sha256(args.ownership_receipt.read_bytes())
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture-pod")
    result = compute.run_workload(args)
    assert result["reason"] == "hard_compute_deadline"
    assert result["status"] == "failed_or_incomplete"


def test_blank_approval_refused_before_attempt_creation(tmp_path, monkeypatch):
    info, root = export(tmp_path, repository(tmp_path))
    args = run_arguments(tmp_path, info, root)
    args.approval_ref = " "
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture-pod")
    with pytest.raises(ValueError, match="approval"):
        compute.run_workload(args)
    assert not args.output_base.exists()


@pytest.mark.parametrize("kind", ["henon_scan", "training_scan"])
def test_real_cli_reads_source_archive_without_git(tmp_path, monkeypatch, kind):
    """Synthetic one-point instrument fixture, not an experimental attempt."""
    repo = repository(tmp_path)
    actual_sources = Path(__file__).parents[1] / "src" / "neural_shrimp"
    for source in actual_sources.glob("*.py"):
        (repo / "src/neural_shrimp" / source.name).write_bytes(source.read_bytes())
    manifest = dict(experiment_id="TEST-ARCHIVE", status="exploratory_pilot", kind="henon_scan",
                    question="Archive compatibility fixture, not a research result", seeds=[101],
                    a_values=[1.2], b_values=[0.3], steps=256, tail=64, max_period=8,
                    atol=1e-8, rtol=1e-6, divergence_threshold=1e8)
    manifest_path = repo / "experiments/manifests/test.json"
    if kind == "training_scan":
        manifest.update(kind=kind, d=1, width=2, ridge=0.01, n_train=8, n_validation=8, n_test=8,
                        lr_hidden_values=[0.01], lr_output_values=[0.01])
    manifest_path.write_text(json.dumps(manifest))
    if kind == "training_scan":
        import sys
        subprocess.check_call([sys.executable, "-B", "-m", "neural_shrimp.cli", "prepare-inputs", str(manifest_path)],
                              cwd=repo, env={**os.environ, "PYTHONPATH": str(repo / "src")})
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "actual CLI compatibility fixture")
    info, root = export(tmp_path, repo)
    args = run_arguments(tmp_path, info, root)
    # Real SciPy import is not a progress artifact; allow its startup time.
    cost = json.loads(args.cost_report.read_text())
    cost["inputs"]["no_progress_seconds"] = 10
    cost["inputs"]["hard_wall_seconds"] = 60
    args.cost_report.write_bytes(compute.json_bytes(compute.estimate(argparse.Namespace(**cost["inputs"]))))
    args.expected_cost_sha256 = compute.sha256(args.cost_report.read_bytes())
    monkeypatch.setenv("RUNPOD_POD_ID", "fixture-pod")
    result = compute.run_workload(args)
    if result["status"] != "process_exit_zero":
        pytest.fail((args.output_base / args.attempt_id / "stderr.log").read_text())
    started = json.loads((args.output_base / args.attempt_id / "results/started.json").read_text())
    assert started["source_commit"] == info["source_commit"]
    assert started["source_mode"] == "verified_archive"
    assert started["expected_trials"] == 1
    assert not (root / ".git").exists()
    if kind == "training_scan":
        expected_input = root / "experiments/inputs/TEST-ARCHIVE/seed-101.npz"
        assert expected_input.exists()
        copied_input = args.output_base / args.attempt_id / "results/inputs/seed-101.npz"
        assert compute.sha256(expected_input.read_bytes()) == compute.sha256(copied_input.read_bytes())


def test_runner_sigterm_kills_workload_and_preserves_receipt(tmp_path, monkeypatch):
    import signal
    import sys

    repo = repository(tmp_path)
    (repo / "src/neural_shrimp/cli.py").write_text("import time\ntime.sleep(30)\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "interrupt fixture")
    info, root = export(tmp_path, repo)
    args = run_arguments(tmp_path, info, root)
    command = [sys.executable, "-B", str(Path(__file__).parents[1] / "tools/prepare_runpod.py"), "run"]
    for name, value in vars(args).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    environment = {**os.environ, "RUNPOD_POD_ID": "fixture-pod"}
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        launch = args.output_base / args.attempt_id / "launch.json"
        deadline = time.monotonic() + 5
        while not launch.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert launch.exists()
        # Wait briefly for Popen and handler installation after launch receipt.
        time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, stderr.decode()
        result = json.loads(stdout)
        assert result["reason"] == "runner_interrupted"
        assert result["status"] == "failed_or_incomplete"
        assert result["exit_code"] is not None
        assert (args.output_base / args.attempt_id / "SHA256.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
