"""Bounded synthetic tests of EXP016 metadata, guards and failure retention.

The controller is loaded against explicit synthetic scientific adapters. No
historical packet is decoded and no compiler, integrator or scientific fit runs.
Adapter mathematics and the real worker scheduler have separate test suites.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
HEAD = "a" * 40


def histories(phases):
    # Independent Cartesian census, deliberately not the evaluator's code.
    result = []
    for setting in ("core", "failed-near-edge"):
        for phase in phases:
            cid = f'{setting}--p{phase["phase_index"]:02d}'
            for first in (-1, 0, 1):
                for second in (-1, 0, 1):
                    for arm in ("identity", "timing", "transverse"):
                        result.append(dict(id=f"{cid}--f{first}--s{second}--{arm}",
                            context_id=cid, setting=setting, **phase, first_sign=first,
                            second_sign=second, arm=arm, epsilon=1e-4))
    return result


@pytest.fixture
def runner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scientific acquisition is forbidden in these runner tests")
    fake_acq = SimpleNamespace(TANGENT_METHODS={"dop853": {}, "rk45": {}},
        METHODS={"rk4-coarse": {}, "rk4-fine": {}, "dop853-independent": {}},
        prepare_context=forbidden, acquire_tangent=forbidden, acquire_trace=forbidden)
    fake_eval = SimpleNamespace(history_census=histories, contract=lambda: {"synthetic": True})
    monkeypatch.setitem(sys.modules, "two_write_acquisition", fake_acq)
    monkeypatch.setitem(sys.modules, "two_write_evaluation", fake_eval)
    spec = importlib.util.spec_from_file_location("synthetic_two_write_runner", TOOLS / "run_two_write_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / "checkout"
    module.ROOT.mkdir()
    module.OLD = tmp_path / "historical"
    module.OLD.mkdir()
    # binding's real default is captured at function definition, not lookup.
    module.binding.__defaults__ = (module.ROOT,)
    module.SOURCE_PATHS = ["synthetic-source.py"]
    (module.ROOT / "synthetic-source.py").write_text("# synthetic committed source\n")
    old_artifacts = []
    for setting in ("core", "failed-near-edge"):
        for suffix in ("json", "npz"):
            path = module.OLD / f"references/payloads/{setting}/packet.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a decodable scientific packet")
            old_artifacts.append(module.binding(path, module.OLD))
    receipt = module.OLD / "receipt.json"
    receipt.write_text(json.dumps(dict(source_commit=module.OLD_SOURCE, artifacts=old_artifacts)))
    module.OLD_RECEIPT_SHA = module.legacy.sha(receipt)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def source_fixture(runner, monkeypatch):
    r = runner
    path = r.ROOT / r.PLAN_REL
    write_json(path, {"synthetic": True})
    review = r.ROOT / r.REVIEW_REL
    write_json(review, dict(status="ready_for_exploratory_execution", plan_sha256=r.legacy.sha(path)))
    state = {"dirty": "", "head": HEAD, "branch": "codex/synthetic", "remote": HEAD}
    committed = {name: (r.ROOT / name).read_bytes() for name in [*r.SOURCE_PATHS, r.PLAN_REL, r.REVIEW_REL]}
    def git(*args):
        if args[0] == "status": return state["dirty"]
        if args[0] == "rev-parse": return state["head"]
        if args[0] == "branch": return state["branch"]
        if args[0] == "ls-remote": return state["remote"] + "\trefs/heads/" + state["branch"]
        pytest.fail("Unexpected Git call")
    monkeypatch.setattr(r.legacy, "git", git)
    monkeypatch.setattr(r.subprocess, "check_output", lambda command: committed[command[-1].split(":", 1)[1]])
    return path, state, committed


def test_phase_census_is_reproducible_disjoint_and_not_old_grid(runner):
    a, b = runner.phase_rows(), runner.phase_rows()
    assert a == b and len(a) == 24
    assert [x["phase_index"] for x in a] == list(range(24))
    dev = {x["phase_fraction"] for x in a if x["split"] == "development"}
    held = {x["phase_fraction"] for x in a if x["split"] == "heldout"}
    assert len(dev) == 8 and len(held) == 16 and dev.isdisjoint(held)
    assert all(0 <= p < 1 and p not in {k / 8 for k in range(8)} for p in dev | held)


def test_exact_job_census_and_independent_panel(runner):
    stages = runner.stage_tasks(runner.phase_rows())
    assert {name: len(rows) for name, rows in stages.items()} == {"references": 96, "development": 918, "heldout": 1782}
    rows = [r for stage in stages.values() for r in stage]
    assert len(rows) == len({r["id"] for r in rows}) == 2796
    traces = stages["development"] + stages["heldout"]
    assert len({r["case"]["id"] for r in traces}) == 1296
    panel = [r for r in traces if r["method"]["id"] == "dop853-independent"]
    assert len(panel) == 108
    assert {r["case"]["phase_index"] for r in panel} == {0, 8}
    for setting in ("core", "failed-near-edge"):
        for phase in (0, 8):
            assert len([r for r in panel if r["case"]["setting"] == setting and r["case"]["phase_index"] == phase]) == 27


def test_metadata_build_and_validation_never_decode_or_integrate(runner, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail("Metadata preparation decoded numerical inputs")
    monkeypatch.setattr(runner.codec, "load_packet", forbidden)
    monkeypatch.setattr(runner.np, "load", forbidden)
    plan = runner.build_plan()
    assert len(plan["historical_inputs"]) == 5
    assert runner.validate(plan)["total_jobs"] == 2796
    changed = copy.deepcopy(plan)
    changed["phases"][8]["split"] = "development"
    with pytest.raises(ValueError): runner.validate(changed)


def test_metadata_rejects_changed_prior_bytes(runner):
    path = runner.OLD / "references/payloads/core/packet.npz"
    path.write_bytes(b"changed opaque bytes")
    with pytest.raises(ValueError, match="Historical reference changed"):
        runner.old_inputs()


def test_source_guard_accepts_exact_clean_reviewed_pushed_head(runner, monkeypatch):
    path, _, _ = source_fixture(runner, monkeypatch)
    result = runner.source_guard({}, path, HEAD)
    assert result["source_commit"] == HEAD


@pytest.mark.parametrize("field,value", [("dirty", "?? unexpected.py"), ("head", "b" * 40), ("branch", "main"), ("remote", "c" * 40)])
def test_source_guard_rejects_dirty_wrong_head_branch_or_unpushed(runner, monkeypatch, field, value):
    path, state, _ = source_fixture(runner, monkeypatch)
    state[field] = value
    with pytest.raises(ValueError): runner.source_guard({}, path, HEAD)


@pytest.mark.parametrize("kind", ["status", "plan_hash", "committed_bytes"])
def test_source_guard_rejects_inexact_review_or_uncommitted_bytes(runner, monkeypatch, kind):
    path, _, committed = source_fixture(runner, monkeypatch)
    review = runner.ROOT / runner.REVIEW_REL
    if kind == "committed_bytes":
        committed[runner.SOURCE_PATHS[0]] = b"different committed source"
    else:
        value = runner.read(review)
        value["status" if kind == "status" else "plan_sha256"] = "wrong"
        write_json(review, value)
        committed[runner.REVIEW_REL] = review.read_bytes()
    with pytest.raises(ValueError): runner.source_guard({}, path, HEAD)


def test_external_attempt_is_exclusive_and_cannot_write_checkout(runner, tmp_path):
    directory = runner.external_attempt(tmp_path / "raw", "attempt-001")
    (directory / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError): runner.external_attempt(tmp_path / "raw", "attempt-001")
    assert (directory / "sentinel").read_text() == "preserve"
    with pytest.raises(ValueError): runner.external_attempt(runner.ROOT / "raw", "attempt-002")
    with pytest.raises(ValueError): runner.external_attempt(tmp_path / "raw", "../escape")


def test_external_attempt_rejects_experiment_symlink(runner, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "EXP-016").symlink_to(runner.ROOT, target_is_directory=True)
    with pytest.raises(ValueError): runner.external_attempt(raw, "attempt-001")


@pytest.mark.parametrize("suffix", ["", "nested"])
def test_output_cannot_extend_immutable_historical_evidence(runner, suffix):
    with pytest.raises(ValueError): runner.external_attempt(runner.OLD / suffix, "attempt-001")


def mock_run(r, monkeypatch, tmp_path, *, fail_setup=False, analysis_complete=True):
    phases = r.phase_rows()
    plan = {"phases": phases}
    plan_path = tmp_path / "synthetic-plan.json"
    write_json(plan_path, plan)
    monkeypatch.setattr(r, "validate", lambda *a, **k: {})
    monkeypatch.setattr(r, "source_guard", lambda *a: dict(source_commit=HEAD, pushed_branch="codex/synthetic"))
    monkeypatch.setattr(r.importlib.metadata, "version", lambda k: r.CONTRACT["required_packages"][k])
    for key in r.legacy.THREAD_ENV: monkeypatch.setenv(key, "1")
    monkeypatch.setattr(r.legacy, "git", lambda *a: HEAD if a[0] == "rev-parse" else "")
    monkeypatch.setattr(r, "resource_check", lambda *a, **k: {})
    monkeypatch.setattr(r.signal, "setitimer", lambda *a: None)
    if fail_setup:
        def fail(*a, **k): raise RuntimeError("synthetic pre-input failure")
        monkeypatch.setattr(r, "resource_check", fail)
        return r.run(plan_path, tmp_path / "raw", "attempt-001", HEAD)
    original = r.stage_tasks(phases)
    # The separate census test covers 2796; lifecycle uses six synthetic tasks.
    tiny = {key: rows[:2] for key, rows in original.items()}
    monkeypatch.setattr(r, "stage_tasks", lambda p: tiny)
    contexts = {task["case"]["context_id"]: {"id": task["case"]["context_id"]}
                for rows in tiny.values() for task in rows}
    monkeypatch.setattr(r, "initial_contexts", lambda p: contexts)
    monkeypatch.setattr(r.work, "build_libraries", lambda *a: {})
    def seals(directory, stages, rows, inputs, *, persist):
        saved = {cid: dict(context_id=cid, status="complete", qualified=True,
                    acquisition=context, evaluation={}) for cid, context in contexts.items()}
        for cid, packet in saved.items(): r.codec.save_packet(Path(directory) / "seals", cid, packet)
        # Deliberately corrupt only the in-memory return; saved seals must govern.
        return {cid: dict(packet, qualified=False) for cid, packet in saved.items()}
    monkeypatch.setattr(r, "make_reference_seals", seals)
    observed_stages = []
    def execute(specs, inputs, directory, libraries, **kwargs):
        directory = Path(directory)
        if directory.name != "references":
            assert (directory.parent / "reference-seal-index.json").is_file()
            events = [json.loads(x) for x in (directory.parent / "events.jsonl").read_text().splitlines()]
            assert any(e["event"] == "reference_seals_complete" for e in events)
            assert len(specs) == 2  # qualified disk seals, not false in-memory copies
        observed_stages.append(directory.name)
        return [dict(id=t["id"], kind=t["kind"], case_id=t["case"]["id"], method=t["method"]["id"], status="complete", qualified=True)
                for t in specs]
    monkeypatch.setattr(r.runtime, "execute", execute)
    monkeypatch.setattr(r, "analyze", lambda *a: dict(packet_errors=[], summary=dict(complete=analysis_complete)))
    result = r.run(plan_path, tmp_path / "raw", "attempt-001", HEAD)
    assert observed_stages == ["references", "development", "heldout"]
    return result


def test_failure_before_inputs_preserves_all_2796_denominators(runner, monkeypatch, tmp_path):
    result = mock_run(runner, monkeypatch, tmp_path, fail_setup=True)
    directory = Path(result["directory"])
    ledger = runner.read(directory / "all-terminal-records.json")
    assert {k: len(v) for k, v in ledger.items()} == {"references": 96, "development": 918, "heldout": 1782}
    assert sum(len(v) for v in ledger.values()) == 2796
    assert result["summary"]["complete"] is False
    assert "synthetic pre-input failure" in result["summary"]["error"]
    assert runner.read(directory / "receipt.json")["status"] == "incomplete_or_invalid"
    assert not (directory / "input-seal.json").exists()


def test_every_reference_seal_is_reloaded_before_any_pulse(runner, monkeypatch, tmp_path):
    result = mock_run(runner, monkeypatch, tmp_path)
    assert result["summary"]["complete"] is True
    assert Path(result["directory"], "receipt.json").is_file()


def test_complete_raw_jobs_do_not_mask_incomplete_analysis(runner, monkeypatch, tmp_path):
    result = mock_run(runner, monkeypatch, tmp_path, analysis_complete=False)
    assert result["summary"]["complete"] is False
    assert runner.read(Path(result["directory"]) / "receipt.json")["status"] == "incomplete_or_invalid"


def test_metadata_prepare_cannot_overwrite_an_existing_plan(runner, monkeypatch, tmp_path):
    path = tmp_path / "existing-plan.json"
    original = b"immutable original plan\n"
    path.write_bytes(original)
    monkeypatch.setattr(runner, "build_plan", lambda: {"new": True})
    monkeypatch.setattr(runner, "validate", lambda *a: {})
    monkeypatch.setattr(sys, "argv", ["runner", "prepare", "--output", str(path)])
    with pytest.raises(FileExistsError): runner.main()
    assert path.read_bytes() == original


def partial_fixture(r, monkeypatch, tmp_path, *, input_seal=False, stage_journal=False):
    """Tiny receipt-bound interrupted attempt, with no scientific arrays."""
    directory = tmp_path / "partial"
    directory.mkdir()
    case = dict(id="synthetic-context", context_id="synthetic-context")
    stages = {
        "references": [dict(id="reference-0", kind="reference", case=case, method=dict(id="dop853"))],
        "development": [dict(id="development-0", kind="trace", case=case, method=dict(id="rk4-fine"))],
        "heldout": [dict(id="heldout-0", kind="trace", case=case, method=dict(id="rk4-fine"))],
    }
    plan = dict(phases=[], source_files=[r.binding(r.ROOT / name) for name in r.SOURCE_PATHS])
    committed_plan = json.dumps(plan, sort_keys=True).encode()
    plan_hash = hashlib.sha256(committed_plan).hexdigest()
    review_path = r.ROOT / r.REVIEW_REL
    write_json(review_path, {"synthetic_review": True})
    review = r.binding(review_path)
    committed = {r.PLAN_REL: committed_plan, r.REVIEW_REL: review_path.read_bytes(),
                 **{name: (r.ROOT / name).read_bytes() for name in r.SOURCE_PATHS}}
    monkeypatch.setattr(r.subprocess, "check_output", lambda command: committed[command[-1].split(":", 1)[1]])
    monkeypatch.setattr(r, "validate", lambda *a, **k: {})
    monkeypatch.setattr(r, "stage_tasks", lambda p: stages)
    contexts = {"synthetic-context": {"metadata_only": 1}}
    monkeypatch.setattr(r, "initial_contexts", lambda p: contexts)
    monkeypatch.setattr(r.work, "identity", lambda pid: None)
    manifest = dict(plan=plan, source_commit=HEAD, plan_sha256=plan_hash, review=review,
        producer_pid=99999999, producer_identity={"pid": 99999999, "start_ticks": "1", "boot_id": "synthetic"})
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "planned-ledger.json", stages)
    rows = {name: r.fill_rows(specs, [], "not_started_interruption") for name, specs in stages.items()}
    write_json(directory / "all-terminal-records.json", rows)
    error = "synthetic interruption"
    write_json(directory / "producer-error.json", dict(stage="input_seal", error=error))
    r.codec.save_packet(directory, "analysis", dict(status="unavailable", reason=error, packet_errors=[], summary=None))
    write_json(directory / "postflight.json", dict(source_bindings_unchanged=True, error=None))
    write_json(directory / "summary.json", dict(complete=False, source_valid=True, error=error,
        postflight_error=None, analysis=None,
        status_counts={name: dict(r.Counter(x["status"] for x in values)) for name, values in rows.items()}))
    if input_seal:
        r.codec.save_packet(directory, "input-seal", contexts)
    if stage_journal:
        stage = directory / "references"
        write_json(stage / "terminal-records.json", rows["references"])
        (stage / "journal.jsonl").write_text(json.dumps(rows["references"][0]) + "\n")
        write_json(stage / "cleanup.json", dict(workers=[], all_registered_workers_joined=True, all_original_identities_absent=True))
    def reseal():
        artifacts = r.inventory(directory)
        write_json(directory / "receipt.json", dict(source_commit=HEAD, plan_sha256=plan_hash,
            experiment_id="EXP-016", artifacts=artifacts, artifact_bytes=sum(x["bytes"] for x in artifacts),
            status="incomplete_or_invalid"))
    reseal()
    return directory, rows, reseal


def test_partial_verifier_accepts_preserved_pre_input_failure(runner, monkeypatch, tmp_path):
    directory, _, _ = partial_fixture(runner, monkeypatch, tmp_path)
    result = runner.verify(directory)
    assert result["complete"] is False
    assert result["saved_array_replay"] is False


def test_partial_input_seal_is_checked_even_without_analysis(runner, monkeypatch, tmp_path):
    directory, _, reseal = partial_fixture(runner, monkeypatch, tmp_path, input_seal=True)
    assert runner.verify(directory)["complete"] is False
    # Change a decoded input value, then truthfully regenerate the byte receipt.
    packet = runner.read(directory / "input-seal.json")
    def replace(value):
        if isinstance(value, dict):
            if value.get("metadata_only") == 1: value["metadata_only"] = 2
            for child in value.values(): replace(child)
        elif isinstance(value, list):
            for child in value: replace(child)
    replace(packet)
    write_json(directory / "input-seal.json", packet)
    reseal()
    with pytest.raises(ValueError): runner.verify(directory)


def test_duplicate_partial_stage_journal_is_not_a_complete_census(runner, monkeypatch, tmp_path):
    directory, rows, reseal = partial_fixture(runner, monkeypatch, tmp_path, stage_journal=True)
    assert runner.verify(directory)["complete"] is False
    duplicate = rows["references"] * 2
    write_json(directory / "references/terminal-records.json", duplicate)
    (directory / "references/journal.jsonl").write_text("".join(json.dumps(row) + "\n" for row in duplicate))
    reseal()
    with pytest.raises(ValueError): runner.verify(directory)


def test_partial_cleanup_failure_is_not_hidden_by_missing_analysis(runner, monkeypatch, tmp_path):
    directory, _, reseal = partial_fixture(runner, monkeypatch, tmp_path, stage_journal=True)
    path = directory / "references/cleanup.json"
    value = runner.read(path)
    value["all_original_identities_absent"] = False
    write_json(path, value)
    reseal()
    with pytest.raises(ValueError): runner.verify(directory)


def test_partial_terminal_task_identity_cannot_be_rehashed_away(runner, monkeypatch, tmp_path):
    directory, rows, reseal = partial_fixture(runner, monkeypatch, tmp_path)
    rows["development"][0]["case_id"] = "wrong-context"
    write_json(directory / "all-terminal-records.json", rows)
    reseal()
    with pytest.raises(ValueError): runner.verify(directory)


def test_partial_postflight_boolean_and_error_must_agree(runner, monkeypatch, tmp_path):
    directory, _, reseal = partial_fixture(runner, monkeypatch, tmp_path)
    write_json(directory / "postflight.json", dict(source_bindings_unchanged=False, error=None))
    reseal()
    with pytest.raises(ValueError): runner.verify(directory)
