"""Resume a synthetic saved orbit; no model, GPU or provider access."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_finish as audit
import baby_shrimp_memory_resume as resume
from trace_saddle_continuation import TraceSaddleContinuation
from test_baby_shrimp_memory_finish import synthetic_point, write_json


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    run, out, anchor = [tmp_path / name for name in ("run", "audit", "anchor")]
    for directory in (run, out, anchor):
        directory.mkdir()
    plan = dict(start=[.1, -.2], coordinate_scales=[1e-5, 2e-5], maximum_abs_beta=.01,
                maximum_scaled_coordinate=1000, maximum_evaluations=450, period=18,
                model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
                anchor_directory="anchor", anchor_hashes={})
    rows = []

    def point(center, beta, cyclic=False):
        index = len(rows)
        holder = run / f"holder-{index}"
        holder.mkdir()
        directory, _ = synthetic_point(holder, dimension=896)
        final = run / f"point-{index:04d}"
        directory.rename(final)
        holder.rmdir()
        row = audit.read(final / "point.json")
        center = np.asarray(center)
        x, y = center - np.array([1000 * beta, -500 * beta])
        tau = float(.0009 + beta - .02*x*x + .03*y*y)
        with np.load(final / "phase-0.npz") as z:
            arrays = {k: z[k] for k in z.files}
        values = arrays["values"]
        values[0] = tau - values[1]
        arrays["w"] = arrays["q"] * values.real
        arrays["direct"] = arrays["vectors"] * values[:2].real
        phase = row["phases"][0]
        multipliers = np.column_stack([values.real, values.imag]).tolist()
        phase.update(multipliers=multipliers, tau=tau, delta=float(np.prod(values[:2]).real),
                     radius=float(max(abs(values))))
        for history in phase["history"]:
            history["multipliers"] = multipliers
        row.update(index=index, coordinate=center.tolist(), beta=beta,
                   alpha=float(plan["start"][0] + plan["coordinate_scales"][0]*center[0]),
                   log10_temperature=float(plan["start"][1] + plan["coordinate_scales"][1]*center[1]),
                   tau=tau, delta=phase["delta"], radius=phase["radius"])
        row["phases"] = [{**copy.deepcopy(phase), "phase": shift}
                         for shift in ((0, 6, 12) if cyclic else (0,))]
        if cyclic:
            row["cyclic_pair_absolute_discrepancy"] = 0.
        for phase in row["phases"]:
            np.savez(final / f"phase-{phase['phase']}.npz", **arrays)
        write_json(final / "point.json", row)
        rows.append(row)
        return row

    baseline = point([0., 0.], 0., cyclic=True)
    with np.load(run / "point-0000/phase-0.npz") as z:
        values = z["values"]
    matrix = np.diag(np.r_[values.real, np.zeros(890)])
    for shift in (0, 6, 12):
        np.savez(anchor / f"phase-{shift}.npz", matrix=matrix, multipliers=values)
    write_json(anchor / "point.json", dict(tau=baseline["tau"], spectral_radius=baseline["radius"]))
    plan["anchor_hashes"] = {p.name: audit.digest(p) for p in anchor.iterdir()}
    directions = np.eye(896, 3)
    np.savez(run / "baseline-operator-check.npz", directions=directions, direct=matrix@directions,
             dense=matrix@directions, errors=np.zeros(3))
    write_json(run / "baseline.json", dict(point=baseline, passed=True, composed_jvp_relative_errors=[0.]*3,
        tau_error_relative_to_radius=0., cyclic_dense_comparisons=[dict(phase=s, pair_relative_error=0.,
        determinant_relative_error=0.) for s in (0, 6, 12)]))
    events, accepted = [], []

    def event(row):
        if row["stage"] == "saddle_complete":
            row["cyclic_endpoint"] = point(row["center"], row["parameter"], cyclic=True)
            write_json(run / f"saddle-{len(accepted):03d}.json", row)
        events.append({"stage": "continuation", "event": row})

    engine = TraceSaddleContinuation(point, stencils=(.01, .003), correction_tolerance=3e-5, record=event)
    for beta in (1e-6, 2e-6):
        accepted.append(engine.correct(beta, [1000*beta, -500*beta]))
    (run / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in events) + "\n")
    write_json(run / "result.json", dict(complete=True, points=accepted))
    write_json(run / "runtime.json", dict(norm_sensitivity=1e-12, rotary_float64=True, threads=8))
    blobs = {p: p.encode() for p in (*audit.BASE_SOURCES, "tools/trace_saddle_continuation.py",
                                   "tools/baby_shrimp_memory_finish.py", "tools/baby_shrimp_memory_resume.py",
                                   "tools/baby_shrimp_orbits.py", "tools/baby_shrimp_geometry.py",
                                   "tools/doubly_superstable.py")}
    blobs["plan.json"] = json.dumps(plan).encode()
    protocol = dict(commit="a"*40, plan=plan, plan_sha256=hashlib.sha256(blobs["plan.json"]).hexdigest(),
                    source_sha256={p: hashlib.sha256(v).hexdigest() for p, v in blobs.items() if p != "plan.json"},
                    args=dict(mode="continue", betas=[1e-6, 2e-6], stencils=[.01, .003],
                              correction_tolerance=3e-5, plan="plan.json"))
    write_json(run / "protocol.json", protocol)
    monkeypatch.setattr(audit, "git_bytes", lambda commit, path, root: blobs[path])
    monkeypatch.setattr(audit, "plot_centers", lambda centers, out: [])
    receipt = audit.audit_run(run, out, root=tmp_path, data=tmp_path)
    receipt.update(analysis_commit="b"*40,
                   analysis_source_sha256=hashlib.sha256(blobs["tools/baby_shrimp_memory_finish.py"]).hexdigest())
    path = out / "receipt.json"
    write_json(path, receipt)
    return SimpleNamespace(run=run, path=path, plan=plan, receipt=receipt, root=tmp_path,
                           endpoint=run / f"point-{receipt['accepted_centers'][-1]['cyclic_point_index']:04d}")


def load(evidence):
    return resume.load_resume(evidence.path, audit.digest(evidence.path), evidence.plan,
                              root=evidence.root, data=evidence.root)


def rebind_manifest(evidence):
    receipt = audit.read(evidence.path)
    receipt["manifest"] = [{"path": str(p.relative_to(evidence.run)), "sha256": audit.digest(p),
                            "bytes": p.stat().st_size} for p in sorted(evidence.run.rglob("*")) if p.is_file()]
    write_json(evidence.path, receipt)


def test_bound_endpoints_replay_and_seed_without_inherited_results(evidence):
    bundle = load(evidence)
    assert [p["parameter"] for p in bundle["history"]] == [1e-6, 2e-6]
    assert all(set(p) == {"parameter", "center", "tau"} for p in bundle["history"])
    checked = resume.verify_resume(bundle["provenance"], evidence.plan, root=evidence.root, data=evidence.root)
    assert checked["provenance"] == bundle["provenance"]
    evaluator = SimpleNamespace(warm=["old"], qs={})
    resume.seed_evaluator(evaluator, bundle)
    assert len(evaluator.warm) == 2 and set(evaluator.qs) == {0, 6, 12}
    np.testing.assert_array_equal(evaluator.warm[-1][2], np.load(evidence.endpoint / "points.npy")[0])
    np.testing.assert_allclose(resume.predictor(bundle["history"], 3e-6), [.003, -.0015], atol=1e-12)
    assert not hasattr(evaluator, "cache")


@pytest.mark.parametrize("mutation,match", [
    ("receipt", "receipt hash"), ("array", "raw manifest hash"),
    ("extra", "exact attempt"), ("duplicate", "duplicate raw manifest"),
    ("source", "source hash"), ("plan", "plan hash or contents"),
    ("analysis", "analysis source hash"),
])
def test_rejects_tampering(evidence, mutation, match):
    old_digest = audit.digest(evidence.path)
    if mutation == "receipt":
        evidence.path.write_text(evidence.path.read_text() + " ")
        with pytest.raises(ValueError, match=match):
            resume.load_resume(evidence.path, old_digest, evidence.plan)
        return
    if mutation == "array":
        with (evidence.endpoint / "points.npy").open("ab") as stream:
            stream.write(b"tampered")
    elif mutation == "extra":
        (evidence.run / "unrecorded.txt").write_text("new")
    elif mutation in ("source", "plan"):
        protocol = audit.read(evidence.run / "protocol.json")
        if mutation == "source":
            protocol["source_sha256"][audit.BASE_SOURCES[0]] = "0"*64
        else:
            protocol["plan"]["period"] = 36
        write_json(evidence.run / "protocol.json", protocol)
        rebind_manifest(evidence)
    else:
        receipt = audit.read(evidence.path)
        if mutation == "duplicate":
            receipt["manifest"].append(receipt["manifest"][0])
        else:
            receipt["analysis_source_sha256"] = "0"*64
        write_json(evidence.path, receipt)
    with pytest.raises(ValueError, match=match):
        load(evidence)


@pytest.mark.parametrize("mutation,match", [("phase", "bound cyclic endpoint"),
                                           ("spectrum", "Ritz basis is not orthonormal")])
def test_reaudits_arrays_and_cyclic_endpoint_even_with_updated_hashes(evidence, mutation, match):
    if mutation == "spectrum":
        path = evidence.endpoint / "phase-6.npz"
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["q"] *= 2
        np.savez(path, **arrays)
    else:
        row = audit.read(evidence.endpoint / "point.json")
        row["phases"] = row["phases"][:1]
        write_json(evidence.endpoint / "point.json", row)
    rebind_manifest(evidence)
    with pytest.raises(ValueError, match=match):
        load(evidence)


def test_missing_old_return_state_is_explicit_and_only_provisional(evidence):
    (evidence.endpoint / "return-state.npy").unlink()
    rebind_manifest(evidence)
    bundle = load(evidence)
    assert not bundle["provenance"]["endpoints"][-1]["closure_recomputed_from_saved_endpoint"]
    assert not resume.replay_template(bundle)["passed"]
    run, record, raw = fresh_replay(evidence, bundle)
    assert resume.verify_replay(run, bundle["provenance"], bundle, raw)["passed"]
    assert not (evidence.endpoint / "return-state.npy").exists()
    assert record["endpoints"][-1]["closure_recomputed_from_saved_endpoint"]


def fresh_replay(evidence, bundle):
    run = evidence.root / "fresh-replay"
    run.mkdir()
    record, raw = resume.replay_template(bundle), {}
    for i, endpoint in enumerate(bundle["provenance"]["endpoints"]):
        directory = run / f"point-{i:04d}"
        shutil.copytree(endpoint["point_directory"], directory)
        row = audit.read(directory / "point.json")
        row["index"] = i
        write_json(directory / "point.json", row)
        # This synthetic closed circle has an exact return. New replay evidence
        # is written only in the fresh fixture; a legacy source stays untouched.
        np.save(directory / "return-state.npy", np.load(directory / "points.npy")[0])
        raw[i] = row
        record["endpoints"].append(resume.audit_replay_endpoint(run, bundle, i, i))
    record["passed"] = True
    write_json(run / "resume-replay.json", record)
    return run, record, raw


def test_fresh_replay_is_recomputed_and_supplies_new_warm_state(evidence):
    bundle = load(evidence)
    run, record, raw = fresh_replay(evidence, bundle)
    checked = resume.verify_replay(run, bundle["provenance"], bundle, raw)
    assert checked["passed"]
    evaluator = SimpleNamespace(warm=[], qs={})
    resume.seed_replayed_evaluator(evaluator, run, bundle, record)
    np.testing.assert_array_equal(evaluator.warm[-1][2], np.load(run / "point-0001/points.npy")[0])
    record["passed"] = False
    record["endpoints"] = record["endpoints"][:1]
    write_json(run / "resume-replay.json", record)
    assert not resume.verify_replay(run, bundle["provenance"], bundle, raw)["passed"]
    with pytest.raises(ValueError, match="fresh endpoint replay required"):
        resume.seed_replayed_evaluator(evaluator, run, bundle, record)


@pytest.mark.parametrize("change,match", [("source", "comparison or source"), ("orbit", "orbit sheet"),
                                         ("return", "independent closure"), ("incomplete", "incomplete")])
def test_fresh_replay_rejects_mismatch_and_corrupt_state(evidence, change, match):
    bundle = load(evidence)
    run, record, raw = fresh_replay(evidence, bundle)
    if change == "source":
        record["endpoints"][0]["source"] = copy.deepcopy(record["endpoints"][0]["source"])
        record["endpoints"][0]["source"]["cyclic_point_index"] += 1
    elif change == "orbit":
        for name in ("points.npy", "return-state.npy"):
            path = run / "point-0000" / name
            np.save(path, 2*np.load(path))
    elif change == "return":
        (run / "point-0000/return-state.npy").unlink()
    else:
        record["endpoints"] = record["endpoints"][:1]
    write_json(run / "resume-replay.json", record)
    with pytest.raises(ValueError, match=match):
        resume.verify_replay(run, bundle["provenance"], bundle, raw)


def test_partial_parent_can_supply_only_qualified_endpoints(evidence):
    directory = evidence.run / "point-9999"
    directory.mkdir()
    write_json(directory / "point.json", dict(index=9999, qualified=False))
    write_json(evidence.run / "failure.json", dict(error="budget expired"))
    rebind_manifest(evidence)
    bundle = load(evidence)
    assert [p["parameter"] for p in bundle["history"]] == [1e-6, 2e-6]
    assert all(p["cyclic_point_index"] != 9999 for p in bundle["provenance"]["endpoints"])


def test_plan_amendments_are_explicit_and_scientific_coordinates_cannot_change(evidence):
    evidence.plan = copy.deepcopy(evidence.plan)
    evidence.plan["maximum_abs_beta"] = .02
    bundle = load(evidence)
    assert bundle["provenance"]["plan_changes"] == {"maximum_abs_beta": {"previous": .01, "current": .02}}
    evidence.plan["coordinate_scales"][0] *= 2
    with pytest.raises(ValueError, match="scientific plan fields"):
        load(evidence)


def test_rejects_changed_selected_endpoint_and_ancestry_cycles(evidence):
    bundle = load(evidence)
    changed = copy.deepcopy(bundle["provenance"])
    changed["endpoints"][0]["center"][0] += 1
    with pytest.raises(ValueError, match="provenance differs"):
        resume.verify_resume(changed, evidence.plan, root=evidence.root, data=evidence.root)
    for seen in ((str(evidence.run),), (str(evidence.path),), tuple(str(i) for i in range(16))):
        with pytest.raises(ValueError, match="ancestry cycle or depth|ancestry cycle or receipt"):
            resume.load_resume(evidence.path, audit.digest(evidence.path), evidence.plan, seen=seen)
    with pytest.raises(ValueError, match="duplicate consecutive"):
        resume.predictor(bundle["history"], 2e-6)


@pytest.mark.parametrize("state", ["passed", "missing", "failed", "wrong-source"])
def test_nested_resume_requires_parent_fresh_replay(evidence, state):
    parent = load(evidence)
    run = evidence.root / "resumed-parent"
    shutil.copytree(evidence.run, run)
    protocol = audit.read(run / "protocol.json")
    protocol["resume"] = parent["provenance"]
    write_json(run / "protocol.json", protocol)
    replay = resume.replay_template(parent)
    for index, source in enumerate(parent["provenance"]["endpoints"]):
        replay["endpoints"].append(resume.audit_replay_endpoint(run, parent, index, source["cyclic_point_index"]))
    replay["passed"] = state != "failed"
    if state == "wrong-source":
        replay["endpoints"][0]["source"] = copy.deepcopy(replay["endpoints"][0]["source"])
        replay["endpoints"][0]["source"]["center"][0] += 1
    if state != "missing":
        write_json(run / "resume-replay.json", replay)
    receipt = copy.deepcopy(evidence.receipt)
    receipt["raw_directory"] = str(run)
    path = evidence.root / "resumed-parent-receipt.json"
    write_json(path, receipt)
    rebind_manifest(SimpleNamespace(path=path, run=run))
    if state == "passed":
        bundle = resume.load_resume(path, audit.digest(path), evidence.plan,
                                    root=evidence.root, data=evidence.root)
        assert bundle["provenance"]["raw_directory"] == str(run)
    else:
        with pytest.raises((ValueError, FileNotFoundError),
                           match="resume-replay|parent resume replay|comparison or source"):
            resume.load_resume(path, audit.digest(path), evidence.plan,
                               root=evidence.root, data=evidence.root)


@pytest.mark.parametrize("stencils", [[.003], [.01], [.02, .003], [.01, .004], [.01, .003, .004]])
def test_resume_cannot_drop_or_coarsen_any_prior_stencil(stencils):
    with pytest.raises(ValueError, match="scales/count"):
        resume.check_resume_settings(stencils, 3e-5, dict(stencils=[.01, .003], correction_tolerance=3e-5))


@pytest.mark.parametrize("stencils", [[.01, .003], [.003, .001], [.01, .003, .001]])
def test_resume_preserves_two_scales_and_allows_finer_extra_scales(stencils):
    resume.check_resume_settings(stencils, 3e-5, dict(stencils=[.01, .003], correction_tolerance=3e-5))


@pytest.mark.parametrize("failure_stage", [None, "replay", "provenance"])
def test_driver_replays_baseline_and_endpoints_before_any_fresh_target(tmp_path, monkeypatch, failure_stage):
    import baby_shrimp_saddle_memory as driver
    import trace_saddle_continuation
    calls = []
    bundle = dict(provenance={"schema": "synthetic", "receipt_sha256": "a"*64, "raw_manifest_sha256": "b"*64},
                  previous_args=dict(stencils=[.01, .003], correction_tolerance=3e-5),
                  history=[dict(parameter=1e-6, center=[1., -2.], tau=.1),
                           dict(parameter=2e-6, center=[2., -4.], tau=.1)],
                  warm=[(np.array([1., -2.]), 1e-6, np.ones(896), {i: np.eye(896, 6) for i in (0, 6, 12)}),
                        (np.array([2., -4.]), 2e-6, np.ones(896)*2, {i: np.eye(896, 6) for i in (0, 6, 12)})])
    def load_resume(*args, **kwargs):
        if failure_stage == "provenance":
            raise ValueError("synthetic provenance failure")
        return bundle
    monkeypatch.setattr(resume, "load_resume", load_resume)
    def audit_replay(*args):
        if args[-1] == 1 and failure_stage == "replay":
            raise ValueError("synthetic replay failure")
        return {"point_index": args[-1]}
    monkeypatch.setattr(resume, "audit_replay_endpoint", audit_replay)
    monkeypatch.setattr(resume, "seed_replayed_evaluator", lambda *args: calls.append("fresh seeds"))
    class Evaluator:
        def __init__(self, *args):
            self.warm = []
            self.qs = {}
            self.attempts = 0
        def baseline(self):
            assert not self.warm
            calls.append("baseline")
            return {"passed": True}
        def __call__(self, center, beta, cyclic):
            calls.append(("replay", beta))
            row = {"index": self.attempts, "qualified": True, "coordinate": list(center), "beta": beta}
            self.attempts += 1
            return row
    class Engine:
        def __init__(self, ev, **kwargs):
            assert calls == ["baseline", ("replay", 1e-6), ("replay", 2e-6), "fresh seeds"] and len(ev.warm) == 2
            self.record = kwargs["record"]
        def correct(self, beta, center):
            calls.append((beta, center.tolist()))
            row = {"parameter": beta, "center": center.tolist(), "tau": .1, "converged": True}
            self.record({"stage": "saddle_complete", **row})
            return row
    monkeypatch.setattr(driver, "Evaluator", Evaluator)
    monkeypatch.setattr(trace_saddle_continuation, "TraceSaddleContinuation", Engine)
    plan = tmp_path / "plan.json"
    write_json(plan, {"maximum_evaluations": 10})
    out = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", ["driver", "--plan", str(plan), "--out", str(out), "--mode", "continue",
        "--source-commit", "a"*40, "--resume-receipt", str(tmp_path / "parent.json"), "--resume-sha256", "b"*64,
        "--stencils", ".01", ".003", "--correction-tolerance", "3e-5", "--betas", "3e-6", "4e-6"])
    if failure_stage == "provenance":
        with pytest.raises(ValueError, match="synthetic provenance failure"):
            driver.main()
        assert calls == []
        assert not (out / "result.json").exists() and (out / "failure.json").exists()
        assert audit.read(out / "protocol.json")["args"]["resume_sha256"] == "b"*64
        return
    if failure_stage == "replay":
        with pytest.raises(ValueError, match="synthetic replay failure"):
            driver.main()
        assert calls == ["baseline", ("replay", 1e-6), ("replay", 2e-6)]
        assert not (out / "result.json").exists() and (out / "failure.json").exists()
        partial = audit.read(out / "resume-replay.json")
        assert not partial["passed"] and len(partial["endpoints"]) == 1
        return
    driver.main()
    assert calls[4][0] == 3e-6
    np.testing.assert_allclose(calls[4][1], [3., -6.])
    result = audit.read(out / "result.json")
    assert [p["parameter"] for p in result["points"]] == [3e-6, 4e-6]
    assert len(list(out.glob("saddle-*.json"))) == 2
    protocol = audit.read(out / "protocol.json")
    assert protocol["resume"] == bundle["provenance"]
    assert {"tools/baby_shrimp_memory_resume.py", "tools/baby_shrimp_memory_finish.py"} <= set(protocol["source_sha256"])
