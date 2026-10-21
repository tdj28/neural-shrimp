"""Independent saved-evidence audits, with synthetic orbits/operators only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_finish as finish
from trace_saddle_continuation import TraceSaddleContinuation


def write_json(path, row):
    path.write_text(json.dumps(row, allow_nan=False))


def synthetic_point(tmp_path, dimension=8):
    directory = tmp_path / "point-0000"
    directory.mkdir()
    period = 18
    angle = np.arange(period) * 2 * np.pi / period
    points = np.zeros((period, dimension))
    points[:, 0], points[:, 1] = np.cos(angle), np.sin(angle)
    np.save(directory / "points.npy", points)
    np.save(directory / "return-state.npy", points[0])
    lower = np.linalg.norm(points[1:] - points[0], axis=1)
    values = np.array([1e-3, -1e-4, 1e-12, -1e-14, 1e-16, -1e-18], dtype=np.complex128)
    q = np.eye(dimension, 6)
    w = q * values.real
    vectors = q[:, :2]
    direct = vectors * values[:2].real
    phase = dict(phase=0, multipliers=np.column_stack([values.real, values.imag]).tolist(),
                 relative_residuals=[0., 0.], tau=float(sum(values[:2]).real),
                 delta=float(np.prod(values[:2]).real), radius=float(max(abs(values))),
                 history=[dict(iteration=i, multipliers=np.column_stack([values.real, values.imag]).tolist(),
                               relative_residuals=[0.] * 6) for i in (0, 1)])
    row = dict(index=0, qualified=True, coordinate=[0., 0.], beta=0., alpha=.1, log10_temperature=-.2,
               closure_history=[.1, 0.], tokens=[271, 785, 3405, 271, 785, 13885] * 3,
               primitive=dict(closed=True, no_lower_period=True, all_returns=[*lower, 0.],
                              minimum_lower_return=float(min(lower)), primitive_separation_tolerance=1e-8),
               phases=[phase], tau=phase["tau"], delta=phase["delta"], radius=phase["radius"])
    write_json(directory / "point.json", row)
    np.savez(directory / "phase-0.npz", values=values, q=q, w=w, vectors=vectors,
             direct=direct, relative_residuals=np.zeros(2))
    plan = dict(start=[.1, -.2], coordinate_scales=[1e-5, 2e-5], maximum_abs_beta=.01,
                maximum_scaled_coordinate=1000, period=period)
    return directory, plan


def test_recomputes_orbit_and_spectral_gates_and_marks_missing_endpoint(tmp_path):
    directory, plan = synthetic_point(tmp_path)
    _, summary = finish.audit_point(directory, plan, dimension=8)
    assert summary["qualified"] and summary["closure_recomputed_from_saved_endpoint"]
    assert summary["maximum_relative_residual"] == 0
    (directory / "return-state.npy").unlink()
    _, summary = finish.audit_point(directory, plan, dimension=8)
    assert not summary["closure_recomputed_from_saved_endpoint"]


@pytest.mark.parametrize("array,change,match", [
    ("q", lambda a: a * 2, "orthonormal"),
    ("w", lambda a: a * 1.1, "Ritz values"),
    ("direct", lambda a: a + 1e-7, "residual gate"),
])
def test_rejects_corrupted_spectral_evidence(tmp_path, array, change, match):
    directory, plan = synthetic_point(tmp_path)
    with np.load(directory / "phase-0.npz") as archive:
        values = {key: archive[key] for key in archive.files}
    values[array] = change(values[array])
    np.savez(directory / "phase-0.npz", **values)
    with pytest.raises(ValueError, match=match):
        finish.audit_point(directory, plan, dimension=8)


def test_rejects_changed_saved_orbit_and_return_state(tmp_path):
    directory, plan = synthetic_point(tmp_path)
    points = np.load(directory / "points.npy")
    points[3] = points[0]
    np.save(directory / "points.npy", points)
    with pytest.raises(ValueError, match="lower-lag returns"):
        finish.audit_point(directory, plan, dimension=8)
    other = tmp_path / "other"
    other.mkdir()
    directory, plan = synthetic_point(other)
    np.save(directory / "return-state.npy", np.ones(8))
    with pytest.raises(ValueError, match="return-state closure"):
        finish.audit_point(directory, plan, dimension=8)


def saddle_evidence():
    points, events = {}, []
    def evaluate(center, beta):
        x, y = center - [0.4 * beta, -0.2 * beta]
        row = {"index": 100 + len(points), "coordinate": center.tolist(), "beta": beta,
               "qualified": True, "tau": float(beta - .3 - 2*x*x + .5*x*y + 3*y*y),
               "delta": -1e-8, "radius": .5, "phases": [{"phase": 0}]}
        points[row["index"]] = row
        return row
    engine = TraceSaddleContinuation(evaluate, record=events.append, correction_tolerance=1e-10)
    saddle = engine.correct(.2, [0, 0])
    mapping = {e["index"]: points[e["callback"]["index"]]
               for e in events if e["stage"] == "evaluation_complete"}
    cyclic = copy.deepcopy(mapping[saddle["point_index"]])
    cyclic.update(index=1000, phases=[{"phase": shift} for shift in (0, 6, 12)])
    points[1000] = cyclic
    saddle["cyclic_endpoint"] = cyclic
    return saddle, mapping, points


def test_recomputes_nine_point_saddle_and_cyclic_center_link():
    saddle, mapping, points = saddle_evidence()
    summary = finish.audit_saddle(saddle, mapping, points, [.1, .03], 1e-10)
    assert summary["kind"] == "continued saddle"
    np.testing.assert_allclose(summary["hessian"], [[-4, .5], [.5, 6]], atol=1e-10)
    assert summary["tau"] == pytest.approx(-.1)
    altered = copy.deepcopy(saddle)
    altered["history"][-1]["hessian"][0][0] += 1e-3
    with pytest.raises(ValueError, match="stencil differs: hessian"):
        finish.audit_saddle(altered, mapping, points, [.1, .03], 1e-10)
    altered = copy.deepcopy(saddle)
    altered["cyclic_endpoint"]["coordinate"][0] += .1
    with pytest.raises(ValueError, match="bound cyclic endpoint"):
        finish.audit_saddle(altered, mapping, points, [.1, .03], 1e-10)


def test_stencil_cannot_use_a_wrong_parameter_point():
    saddle, mapping, points = saddle_evidence()
    wrong = copy.deepcopy(mapping)
    index = saddle["history"][0]["point_indices"][0][0]
    wrong[index]["beta"] += .01
    with pytest.raises(ValueError, match="wrong parameter"):
        finish.audit_saddle(saddle, wrong, points, [.1, .03], 1e-10)


def test_execution_sources_and_plan_are_bound_to_git_bytes(tmp_path, monkeypatch):
    plan = {"period": 18, "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "anchor_directory": "anchor", "anchor_hashes": {}}
    (tmp_path / "anchor").mkdir()
    blobs = {p: p.encode() for p in finish.BASE_SOURCES}
    blobs["plan.json"] = json.dumps(plan).encode()
    protocol = dict(commit="a" * 40, plan=plan, args=dict(mode="pilot", plan="plan.json"),
                    source_sha256={p: hashlib.sha256(blobs[p]).hexdigest() for p in finish.BASE_SOURCES},
                    plan_sha256=hashlib.sha256(blobs["plan.json"]).hexdigest())
    monkeypatch.setattr(finish, "git_bytes", lambda commit, path, root: blobs[path])
    assert finish.bind_protocol(protocol, data=tmp_path) == tmp_path / "anchor"
    changed = copy.deepcopy(protocol)
    changed["source_sha256"][finish.BASE_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        finish.bind_protocol(changed, data=tmp_path)
    changed = copy.deepcopy(protocol)
    changed["plan"]["period"] = 36
    with pytest.raises(ValueError, match="plan hash or contents"):
        finish.bind_protocol(changed, data=tmp_path)


def test_incomplete_point_remains_manifested_and_prevents_complete_label(tmp_path, monkeypatch):
    run, out, anchor = (tmp_path / name for name in ("run", "audit", "anchor"))
    for directory in (run, out, anchor):
        directory.mkdir()
    directory, plan = synthetic_point(run, dimension=896)
    row = finish.read(directory / "point.json")
    with np.load(directory / "phase-0.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    row["phases"] = [{**copy.deepcopy(row["phases"][0]), "phase": shift} for shift in (0, 6, 12)]
    row["cyclic_pair_absolute_discrepancy"] = 0.
    for shift in (6, 12):
        np.savez(directory / f"phase-{shift}.npz", **arrays)
    write_json(directory / "point.json", row)
    matrix = np.zeros((896, 896))
    np.fill_diagonal(matrix, np.r_[arrays["values"].real, np.zeros(890)])
    for shift in (0, 6, 12):
        np.savez(anchor / f"phase-{shift}.npz", matrix=matrix, multipliers=arrays["values"])
    write_json(anchor / "point.json", dict(tau=row["tau"], spectral_radius=row["radius"]))
    directions = np.eye(896, 3)
    action = matrix @ directions
    np.savez(run / "baseline-operator-check.npz", directions=directions, direct=action, dense=action, errors=np.zeros(3))
    baseline = dict(point=row, passed=True, composed_jvp_relative_errors=[0.] * 3,
                    tau_error_relative_to_radius=0., cyclic_dense_comparisons=[
                        dict(phase=shift, pair_relative_error=0., determinant_relative_error=0.) for shift in (0, 6, 12)])
    write_json(run / "baseline.json", baseline)
    write_json(run / "protocol.json", dict(commit="a" * 40, plan=plan, plan_sha256="b" * 64,
                                            args=dict(mode="pilot", betas=[0.])))
    write_json(run / "runtime.json", dict(norm_sensitivity=1e-12, rotary_float64=True, threads=8))
    write_json(run / "result.json", dict(complete=True, rows=[row]))
    monkeypatch.setattr(finish, "bind_protocol", lambda protocol, root, data: anchor)
    complete = finish.audit_run(run, out)
    assert complete["complete"] and complete["figures"] == []
    incomplete = run / "point-0001"
    incomplete.mkdir()
    write_json(incomplete / "point.json", dict(index=1, qualified=False))
    partial = finish.audit_run(run, out)
    assert not partial["complete"]
    assert len(partial["qualified_points"]) == 1
    assert partial["unqualified_or_incomplete_points"][0]["index"] == 1
    assert "point-0001/point.json" in {row["path"] for row in partial["manifest"]}
