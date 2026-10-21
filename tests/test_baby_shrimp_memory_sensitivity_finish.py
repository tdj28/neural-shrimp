"""Offline sensitivity-audit corruption gates, using tiny synthetic evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_sensitivity_finish as finish


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def pair_evidence():
    matrix = np.diag([.02, -.01, 1e-8, -1e-9])
    vectors = np.eye(4, 2)
    values = np.array([.02, -.01])
    arrays = dict(left=vectors.copy(), right=vectors.copy(), multipliers=values)
    row = dict(gates=finish.PAIR_GATES.copy(), multipliers=values.tolist(), trace=.01,
               determinant=-.0002, right_relative_residuals=[0., 0.], left_relative_residuals=[0., 0.],
               eigenvalue_condition_estimates=[1., 1.], biorthogonality_error=0., third_modulus=1e-8, passed=True)
    return arrays, matrix, row


def test_dense_pair_replay_rejects_corrupt_vectors_spectrum_and_record():
    arrays, matrix, row = pair_evidence()
    assert finish.audit_pair(arrays, matrix, row)["maximum_right_relative_residual"] == 0
    corrupt = copy.deepcopy(arrays)
    corrupt["left"][3, 0] = .01
    with pytest.raises(ValueError, match="left/right pair gate"):
        finish.audit_pair(corrupt, matrix, row)
    corrupt = copy.deepcopy(arrays)
    corrupt["multipliers"][0] += .001
    with pytest.raises(ValueError, match="dense spectrum"):
        finish.audit_pair(corrupt, matrix, row)
    corrupt = {**row, "trace": .011}
    with pytest.raises(ValueError, match="summary differs: trace"):
        finish.audit_pair(arrays, matrix, corrupt)


def test_saved_contractions_must_match_independently():
    left = np.eye(4, 2)
    columns = np.arange(8.).reshape(4, 2)
    row = dict(multiplier_derivatives=[0., 3.], trace_derivative=3.)
    assert finish.contract(left, columns, row) == ([0., 3.], 3.)
    columns[1, 1] += .01
    with pytest.raises(ValueError, match="multiplier derivative differs"):
        finish.contract(left, columns, row)


def tangent_evidence():
    matrix = np.diag([.02, -.01, 1e-8, -1e-9])
    angles = np.arange(19) * 2 * np.pi / 18
    states = np.column_stack([np.cos(angles), np.sin(angles), np.ones(19), np.zeros(19)])
    states[-1] = states[0]
    tangents = np.tile(np.array([1., 2., 3., 4.]), (19, 1))
    arrays = dict(tangents=tangents, forcing=states[:-1] - states[1:],
                  zero_start_return=(np.eye(4) - matrix) @ tangents[0])
    returns = np.linalg.norm(states[1:] - states[0], axis=1) / np.linalg.norm(states[0])
    qualification = dict(maximum_replayed_state_relative_difference=0., same_cyclic_text=True,
                         tokens=[271, 785, 3405, 271, 785, 13885] * 3,
                         primitive=dict(all_returns=returns.tolist(), closed=True, no_lower_period=True,
                                        minimum_lower_return=float(min(returns[:-1])), primitive_separation_tolerance=1e-8))
    row = dict(gates=finish.TANGENT_GATES.copy(), orbit_relative_closure=0., solve_relative_residual=0.,
               tangent_relative_closure=0., tangent_absolute_closure=0.,
               maximum_phase_tangent_norm=float(np.linalg.norm(tangents[0])),
               resolvent_condition=float(np.linalg.cond(np.eye(4) - matrix)), passed=True)
    return states, arrays, matrix, qualification, row


@pytest.mark.parametrize("name", ["forcing", "zero_start_return", "tangents"])
def test_periodic_tangent_checks_available_raw_constraints(name):
    states, arrays, matrix, qualification, row = tangent_evidence()
    assert finish.audit_orbit_tangent(states, states.copy(), arrays, matrix, qualification, row)["solve_relative_residual"] == 0
    corrupt = copy.deepcopy(arrays)
    corrupt[name].flat[-1] += .01
    with pytest.raises(ValueError, match="forcing differs|tangent summary differs"):
        finish.audit_orbit_tangent(states, states, corrupt, matrix, qualification, row)


def derivative_summary(arrays, left):
    values = {key: np.einsum("ij,ij->j", left, arrays[key]) for key in ("explicit", "orbit_mediated", "total")}
    return dict(multiplier_derivatives={key: value.tolist() for key, value in values.items()},
                trace_derivatives={key: float(sum(value)) for key, value in values.items()})


def fd_evidence(power=2):
    states, tangent, _, _, _ = tangent_evidence()
    left = np.eye(4, 2)
    ad = dict(left=left, tangents=tangent["tangents"], propagated=left * .02,
              explicit=np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]) * .1,
              orbit_mediated=np.ones((4, 2)))
    ad["total"] = ad["explicit"] + ad["orbit_mediated"]
    ad_row = derivative_summary(ad, left)
    steps, rows, all_arrays = [1e-3, 3e-4, 1e-4], [], []
    for h in steps:
        change = h**power * np.array([[1., 2.], [3., 5.], [7., 11.], [13., 17.]])
        arrays = dict(propagated=ad["propagated"].copy(), explicit=ad["explicit"].copy(),
                      orbit_mediated=ad["orbit_mediated"] + change, total=ad["total"] + change,
                      actual_parameter_steps=h * np.linalg.norm(states[:-1], axis=1) / np.linalg.norm(ad["tangents"][:-1], axis=1))
        row = derivative_summary(arrays, left)
        absolute = abs(row["trace_derivatives"]["total"] - ad_row["trace_derivatives"]["total"])
        row.update(relative_step=h, actual_parameter_steps=arrays["actual_parameter_steps"].tolist(),
                   total_column_relative_errors=(np.linalg.norm(change, axis=0) / np.linalg.norm(ad["total"], axis=0)).tolist(),
                   total_trace_absolute_error=absolute,
                   total_trace_relative_error=absolute / abs(ad_row["trace_derivatives"]["total"]))
        rows.append(row)
        all_arrays.append(arrays)
    return ad, ad_row, states, rows, all_arrays, steps


def test_recomputes_fd_columns_errors_and_quadratic_orders():
    evidence = fd_evidence()
    result = finish.audit_fd(*evidence)
    np.testing.assert_allclose(result["column_error_orders"], [2., 2.], atol=1e-8)
    assert result["passed"]
    with pytest.raises(ValueError, match="precision gate failed"):
        finish.audit_fd(*fd_evidence(power=1))


@pytest.mark.parametrize("corruption", ["column", "step", "reported_error"])
def test_fd_rejects_corrupted_saved_evidence(corruption):
    ad, ad_row, states, rows, arrays, steps = fd_evidence()
    if corruption == "column":
        arrays[-1]["total"][3, 0] += .01
    elif corruption == "step":
        arrays[-1]["actual_parameter_steps"][0] *= 1.01
    else:
        rows[-1]["total_column_relative_errors"][0] += .01
    with pytest.raises(ValueError, match="decomposition differs|steps differ|column errors differ"):
        finish.audit_fd(ad, ad_row, states, rows, arrays, steps)


def test_source_plan_and_dense_anchor_bound_to_exact_commit(tmp_path, monkeypatch):
    directory, anchor = tmp_path / "run", tmp_path / "anchor"
    directory.mkdir()
    anchor.mkdir()
    for name in ("points.npy", "phase-0.npz", "phase-6.npz", "phase-12.npz"):
        (anchor / name).write_bytes(name.encode())
    write_json(anchor / "point.json", dict(alpha=.1, log10_temperature=-.2))
    sources = (set(finish.audit.BASE_SOURCES) - {"tools/llm_feigenbaum.py"}) | {
        "tools/trace_pair_sensitivity.py", "tools/baby_shrimp_memory_mechanism.py"}
    blobs = {path: path.encode() for path in sources}
    plan = dict(period=18, model_revision="7ae557604adf67be50417f59c2c2f167def9a775", start=[.1, -.2],
                anchor_directory="anchor", anchor_hashes={p.name: finish.digest(p) for p in anchor.iterdir()})
    blobs["plan.json"] = json.dumps(plan).encode()
    protocol = dict(source_commit="a" * 40, args=dict(plan="plan.json"),
                    source_sha256={path: hashlib.sha256(blobs[path]).hexdigest() for path in sources},
                    plan_sha256=hashlib.sha256(blobs["plan.json"]).hexdigest())
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "runtime.json", dict(threads=8, rotary_float64=True, norm_sensitivity=1e-12))
    def git_bytes(commit, path, root):
        assert commit == "a" * 40
        return blobs[path]
    monkeypatch.setattr(finish.audit, "git_bytes", git_bytes)
    assert finish.bind_attempt(directory, "explicit", data=tmp_path)[2] == anchor
    changed = copy.deepcopy(protocol)
    changed["source_sha256"][next(iter(sources))] = "0" * 64
    write_json(directory / "protocol.json", changed)
    with pytest.raises(ValueError, match="source differs from Git"):
        finish.bind_attempt(directory, "explicit", data=tmp_path)
    changed = {**protocol, "plan_sha256": "0" * 64}
    write_json(directory / "protocol.json", changed)
    with pytest.raises(ValueError, match="plan hash differs"):
        finish.bind_attempt(directory, "explicit", data=tmp_path)
    write_json(directory / "protocol.json", protocol)
    (anchor / "phase-6.npz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="dense-anchor hash differs"):
        finish.bind_attempt(directory, "explicit", data=tmp_path)


def test_optional_pilot_comparison_combines_only_matching_locations_and_gates_finest():
    anchor = {"points.npy": "a" * 64}
    total = dict(trace_derivatives=dict(total=-.14))
    def pilot(steps):
        return dict(anchor_hashes=anchor, start=[.1, -.2], zero_beta_trace=1e-8,
                    differences=[dict(beta_step=h, central_slope=-.14 + 1e12*h*h, central_curvature=1.,
                                      phase_slopes={str(p): -.14 + 1e12*h*h for p in (0, 6, 12)}) for h in steps])
    pilots = [pilot([1e-7]), pilot([1e-8, 3e-9])]
    result = finish.compare_pilots(pilots, total, anchor)
    np.testing.assert_allclose(result["error_orders"], [2., 2.], atol=1e-10)
    assert result["passed"] and len(result["rows"]) == 3
    with pytest.raises(ValueError, match="finest central-slope"):
        finish.compare_pilots([pilots[0]], total, anchor)
    wrong = copy.deepcopy(pilots)
    wrong[-1]["start"][0] += .1
    with pytest.raises(ValueError, match="moved parameter location"):
        finish.compare_pilots(wrong, total, anchor)
    wrong = copy.deepcopy(pilots)
    wrong[-1]["anchor_hashes"]["points.npy"] = "b" * 64
    with pytest.raises(ValueError, match="different dense anchor"):
        finish.compare_pilots(wrong, total, anchor)
    wrong = copy.deepcopy(pilots)
    wrong[-1]["differences"][-1]["phase_slopes"]["12"] = -.15
    with pytest.raises(ValueError, match="cyclic phase fails"):
        finish.compare_pilots(wrong, total, anchor)
