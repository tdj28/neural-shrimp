"""Audit saved BSC3 memory pilots/continuation without loading a model.

All qualified rows are checked. Failed or unfinished rows remain in the manifest
and prevent a complete label. Optional --worker also performs the existing
read-only provider/termination check; ordinary local audits make no API calls.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
BASE_SOURCES = ("tools/baby_shrimp_saddle_memory.py", "tools/baby_shrimp_memory.py",
                "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
                "tools/explore_llm_phase_map.py", "tools/llm_feigenbaum.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_bytes(commit, path, root=ROOT):
    require(not Path(path).is_absolute() and ".." not in Path(path).parts, "invalid source path")
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)


def close(actual, expected, message, rtol=1e-10, atol=1e-14):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=rtol, atol=atol), message)


def complex_values(value):
    value = np.asarray(value, float)
    require(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all(), "invalid multiplier record")
    return value[:, 0] + 1j * value[:, 1]


def pair_gap(a, b):
    return float(min(np.max(abs(a - b)), np.max(abs(a - b[::-1]))))


def canonical(values):
    values = tuple(values)
    return min(values[i:] + values[:i] for i in range(len(values)))


def bind_protocol(protocol, root=ROOT, data=DATA):
    commit = protocol["commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "exact source commit missing")
    required = set(BASE_SOURCES)
    if protocol["args"]["mode"] == "continue":
        required.add("tools/trace_saddle_continuation.py")
    if "resume" in protocol:
        required.update(("tools/baby_shrimp_memory_resume.py", "tools/baby_shrimp_memory_finish.py",
                         "tools/baby_shrimp_orbits.py", "tools/baby_shrimp_geometry.py",
                         "tools/doubly_superstable.py"))
    require(required <= set(protocol["source_sha256"]), "incomplete execution source manifest")
    for path, expected in protocol["source_sha256"].items():
        require(hashlib.sha256(git_bytes(commit, path, root)).hexdigest() == expected,
                "execution source hash differs from Git: " + path)
    plan_blob = git_bytes(commit, protocol["args"]["plan"], root)
    require(hashlib.sha256(plan_blob).hexdigest() == protocol["plan_sha256"]
            and json.loads(plan_blob) == protocol["plan"], "plan hash or contents differ from Git")
    plan = protocol["plan"]
    require(plan["period"] == 18 and plan["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775",
            "unexpected orbit family or model")
    anchor = Path(data) / plan["anchor_directory"]
    for path, expected in plan["anchor_hashes"].items():
        require(digest(anchor / path) == expected, "dense anchor hash differs: " + path)
    return anchor


def audit_phase(path, row, dimension=896):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"values", "vectors", "direct", "relative_residuals", "q", "w"}
    require(required <= set(arrays), "missing spectral arrays")
    require(all(np.isfinite(a).all() for a in arrays.values()), "nonfinite spectral array")
    values, vectors, direct, q, w = (arrays[k] for k in ("values", "vectors", "direct", "q", "w"))
    rank = len(values)
    require(rank >= 4 and q.shape == w.shape == (dimension, rank)
            and vectors.shape == direct.shape == (dimension, 2), "spectral array dimensions differ")
    require(all(arrays[k].dtype == np.float64 for k in ("q", "w", "vectors", "direct", "relative_residuals"))
            and values.dtype in (np.dtype("float64"), np.dtype("complex128")), "spectral precision differs")
    close(q.T @ q, np.eye(rank), "Ritz basis is not orthonormal", rtol=0, atol=1e-10)
    recomputed = np.linalg.eigvals(q.T @ w)
    first, second = linear_sum_assignment(abs(recomputed[:, None] - values[None, :]))
    spectrum_error = float(np.max(abs(recomputed[first] - values[second])))
    require(spectrum_error < 1e-11 * max(abs(values)) + 1e-18, "stored Ritz values differ from q.T@w")
    require(np.all(np.diff(abs(values)) <= 1e-15 * max(abs(values))), "multipliers not ordered by modulus")
    require(min(abs(values[:2])) > 0 and max(abs(values[:2].imag)) <= 1e-10 * max(abs(values[:2])),
            "tracked pair is not finite nonzero real")
    require(abs(values[1]) > 100 * abs(values[2]), "leading pair lost Ritz separation")
    require(max(abs(values)) < 1, "recorded pair lost attraction")
    close(values, complex_values(row["multipliers"]), "JSON multiplier record differs", rtol=1e-12, atol=1e-18)
    close(q @ (q.T @ vectors), vectors, "leading vectors left recorded subspace", rtol=0, atol=1e-10)
    denominator = abs(values[:2]) * np.linalg.norm(vectors, axis=0)
    require(np.all(denominator > 0), "zero spectral residual denominator")
    residual = np.linalg.norm(direct - vectors * values[:2].real, axis=0) / denominator
    projected = np.linalg.norm(w @ (q.T @ vectors) - vectors * values[:2].real, axis=0) / denominator
    require(max(residual) < 1e-6 and max(projected) < 1e-6, "leading vector residual gate failed")
    close(residual, arrays["relative_residuals"], "saved direct residual differs", rtol=1e-5, atol=1e-12)
    close(residual, row["relative_residuals"], "JSON direct residual differs", rtol=1e-5, atol=1e-12)
    require(len(row["history"]) >= 2, "insufficient recorded subspace passes")
    close(values, complex_values(row["history"][-1]["multipliers"]), "last subspace spectrum differs",
          rtol=1e-12, atol=1e-18)
    require(max(row["history"][-1]["relative_residuals"][:2]) < 1e-6, "recorded projected residual failed")
    tau, delta, radius = float(sum(values[:2]).real), float(np.prod(values[:2]).real), float(max(abs(values)))
    for key, value in (("tau", tau), ("delta", delta), ("radius", radius)):
        close(row[key], value, "spectral summary differs: " + key, atol=1e-18)
    require(delta < 0, "opposite-sign pair regime changed")
    return {"phase": int(row["phase"]), "tau": tau, "delta": delta, "radius": radius,
            "pair": values[:2], "maximum_relative_residual": float(max(residual)),
            "maximum_projected_relative_residual": float(max(projected)),
            "ritz_spectrum_absolute_replay_error": spectrum_error}


def audit_point(directory, plan, dimension=896):
    row = read(directory / "point.json")
    require(row.get("qualified") is True, "unqualified point cannot supply a scientific result")
    require(directory.name == f"point-{row['index']:04d}", "point index does not match directory")
    coordinate = np.asarray(row["coordinate"], float)
    require(coordinate.shape == (2,) and np.isfinite(coordinate).all() and np.isfinite(row["beta"]),
            "invalid point coordinates")
    require(abs(row["beta"]) <= plan["maximum_abs_beta"]
            and max(abs(coordinate)) <= plan["maximum_scaled_coordinate"], "point outside declared domain")
    physical = np.asarray(plan["start"]) + np.asarray(plan["coordinate_scales"]) * coordinate
    close([row["alpha"], row["log10_temperature"]], physical, "physical coordinates differ", rtol=0, atol=1e-15)
    points = np.load(directory / "points.npy", allow_pickle=False)
    period = plan["period"]
    require(points.shape == (period, dimension) and points.dtype == np.float64 and np.isfinite(points).all(),
            "invalid saved orbit points")
    norm = np.linalg.norm(points[0])
    require(norm > 0, "zero orbit norm")
    lower = np.linalg.norm(points[1:] - points[0], axis=1) / norm
    primitive = row["primitive"]
    returns = np.asarray(primitive["all_returns"], float)
    history = np.asarray(row["closure_history"], float)
    require(returns.shape == (period,) and np.isfinite(returns).all() and np.all(returns >= 0)
            and history.ndim == 1 and len(history) > 0 and np.isfinite(history).all()
            and np.all(history >= 0), "invalid closure or primitive distances")
    close(lower, returns[:-1], "lower-lag returns differ from saved orbit", atol=1e-13)
    close(history[-1], returns[-1], "closure records disagree", rtol=1e-5, atol=1e-16)
    require(history[-1] < 1e-12 and returns[-1] < 1e-11 and min(lower) > 1e-8,
            "closure or primitive period gate failed")
    require(primitive["closed"] and primitive["no_lower_period"]
            and primitive["primitive_separation_tolerance"] == 1e-8, "stored primitive gate differs")
    close(primitive["minimum_lower_return"], min(lower), "minimum lower return differs", atol=1e-13)
    endpoint = directory / "return-state.npy"
    independently_closed = endpoint.exists()
    if independently_closed:
        final = np.load(endpoint, allow_pickle=False)
        require(final.size == dimension and final.dtype == np.float64 and np.isfinite(final).all(),
                "invalid saved return state")
        closure = float(np.linalg.norm(final.reshape(-1) - points[0]) / norm)
        close(closure, returns[-1], "return-state closure differs", rtol=1e-5, atol=1e-16)
        require(closure < 1e-12, "saved return state failed closure")
    require(len(row["tokens"]) == period and canonical(row["tokens"]) ==
            canonical([271, 785, 3405, 271, 785, 13885] * 3), "cyclic token family differs")
    shifts = [r["phase"] for r in row["phases"]]
    require(shifts in ([0], [0, 6, 12]), "missing or unexpected spectral phases")
    phases = [audit_phase(directory / f"phase-{p['phase']}.npz", p, dimension) for p in row["phases"]]
    for key in ("tau", "delta", "radius"):
        close(row[key], phases[0][key], "point spectral summary differs: " + key, atol=1e-18)
    discrepancy = None
    if len(phases) == 3:
        discrepancy = max(pair_gap(phases[0]["pair"], p["pair"]) for p in phases)
        require(discrepancy < 1e-5 * max(abs(phases[0]["pair"])), "cyclic spectra disagree")
        close(row["cyclic_pair_absolute_discrepancy"], discrepancy, "cyclic discrepancy differs", atol=1e-17)
    summary = {k: row[k] for k in ("index", "coordinate", "beta", "alpha", "log10_temperature", "tau", "delta", "radius")}
    summary.update(qualified=True, closure=float(returns[-1]), minimum_lower_return=float(min(lower)),
                   closure_recomputed_from_saved_endpoint=independently_closed, phases=shifts,
                   maximum_relative_residual=max(p["maximum_relative_residual"] for p in phases),
                   cyclic_pair_absolute_discrepancy=discrepancy)
    return row, summary


def audit_baseline(run, raw, anchor):
    baseline = read(run / "baseline.json")
    point = raw[baseline["point"]["index"]]
    require(point == baseline["point"] and point["beta"] == 0 and point["coordinate"] == [0, 0]
            and [p["phase"] for p in point["phases"]] == [0, 6, 12], "baseline point binding differs")
    with np.load(run / "baseline-operator-check.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require(all(np.isfinite(v).all() for v in arrays.values()), "nonfinite baseline arrays")
    with np.load(anchor / "phase-6.npz", allow_pickle=False) as z:
        matrix = z["matrix"]
    close(matrix @ arrays["directions"], arrays["dense"], "baseline dense action differs", atol=1e-14)
    denominator = np.maximum(np.maximum(np.linalg.norm(arrays["direct"], axis=0),
                                        np.linalg.norm(arrays["dense"], axis=0)), 1e-300)
    errors = np.linalg.norm(arrays["direct"] - arrays["dense"], axis=0) / denominator
    close(errors, arrays["errors"], "baseline operator errors differ", rtol=1e-5, atol=1e-12)
    close(errors, baseline["composed_jvp_relative_errors"], "baseline JSON errors differ", rtol=1e-5, atol=1e-12)
    expected = read(anchor / "point.json")
    tau_error = abs(point["tau"] - expected["tau"]) / expected["spectral_radius"]
    close(tau_error, baseline["tau_error_relative_to_radius"], "baseline trace error differs", atol=1e-15)
    require(max(errors) < 1e-7 and tau_error < 1e-5 and baseline["passed"], "baseline operator gate failed")
    for phase, stored in zip(point["phases"], baseline["cyclic_dense_comparisons"], strict=True):
        require(phase["phase"] == stored["phase"], "baseline phase ordering differs")
        with np.load(anchor / f"phase-{(phase['phase'] + 6) % 18}.npz", allow_pickle=False) as z:
            reference = z["multipliers"][:2]
        pair = complex_values(phase["multipliers"])[:2]
        err = pair_gap(pair, reference) / max(abs(reference))
        det_err = abs(np.prod(pair) - np.prod(reference)) / abs(np.prod(reference))
        close(err, stored["pair_relative_error"], "dense pair comparison differs", atol=1e-15)
        close(det_err, stored["determinant_relative_error"], "dense determinant comparison differs", atol=1e-15)
        require(err < 1e-5 and det_err < 1e-5, "dense pair comparison gate failed")
    return {"passed": True, "point_index": point["index"], "maximum_operator_relative_error": float(max(errors)),
            "trace_error_relative_to_radius": float(tau_error)}


def continuation_events(run, raw):
    events = []
    for line in (run / "progress.jsonl").read_text().splitlines():
        outer = json.loads(line)
        if outer.get("stage") == "continuation":
            events.append(outer["event"])
    mapping = {}
    for event in events:
        if event["stage"] != "evaluation_complete":
            continue
        point = raw[event["callback"]["index"]]
        require(event["callback"] == point, "continuation callback differs from raw point")
        close(event["center"], point["coordinate"], "callback coordinates differ", rtol=0, atol=0)
        require(event["parameter"] == point["beta"] and event["tau"] == point["tau"], "callback parameter/trace differs")
        require(event["index"] not in mapping, "duplicate continuation evaluation index")
        mapping[event["index"]] = point
    return events, mapping


def stationary_height_diagnostics(tau, gradient, hessian):
    """Local quadratic sensitivity estimates, never numerical error bounds."""
    gradient, hessian = np.asarray(gradient, float), np.asarray(hessian, float)
    values, basis = np.linalg.eigh(hessian)
    require(np.isfinite(values).all() and values[0] < 0 < values[1],
            "height diagnostic requires a nonsingular saddle Hessian")
    correction = -np.linalg.solve(hessian, gradient)
    signed_shift = float(.5 * gradient @ correction)
    # At a saddle, positive and negative curvature terms can cancel. Keep
    # their absolute sum as well as the signed quadratic correction.
    absolute_scale = float(.5 * np.sum(abs(values) * (basis.T @ correction)**2))
    return {"gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
            "hessian": hessian.tolist(), "correction": correction.tolist(),
            "correction_norm": float(np.linalg.norm(correction)),
            "quadratic_height_shift": signed_shift,
            "quadratic_corrected_height": float(tau + signed_shift),
            "absolute_curvature_height_scale": absolute_scale}


def compare_stencil_scales(coarse, fine):
    """Compare local models at a common center; extrapolate only if it is saved."""
    require(coarse["parameter"] == fine["parameter"]
            and coarse["stencil_step"] > fine["stencil_step"], "incompatible stencil scales")
    displacement = np.asarray(fine["center"]) - np.asarray(coarse["center"])
    gradient, hessian = np.asarray(coarse["gradient"]), np.asarray(coarse["hessian"])
    transported_trace = float(coarse["tau"] + gradient @ displacement
                              + .5 * displacement @ hessian @ displacement)
    transported_gradient = gradient + hessian @ displacement
    same_center = np.array_equal(np.asarray(coarse["center"]), np.asarray(fine["center"]))
    comparison = {"coarse_step": coarse["stencil_step"], "fine_step": fine["stencil_step"],
                  "centers_identical": same_center, "center_displacement": displacement.tolist(),
                  "center_displacement_norm": float(np.linalg.norm(displacement)),
                  "raw_height_difference": float(fine["tau"] - coarse["tau"]),
                  "quadratic_corrected_height_difference": float(
                      fine["quadratic_corrected_height"] - coarse["quadratic_corrected_height"]),
                  "coarse_quadratic_trace_at_fine_center": transported_trace,
                  "trace_transport_residual": float(fine["tau"] - transported_trace),
                  "gradient_transport_residual": (np.asarray(fine["gradient"]) - transported_gradient).tolist(),
                  "hessian_difference_norm": float(np.linalg.norm(np.asarray(fine["hessian"]) - hessian, 2)),
                  "scope": "Quadratic transport between saved centers; omitted higher orders are not bounded."}
    if not same_center:
        comparison["richardson"] = {"applied": False, "reason": "Stencil centers differ; direct Richardson extrapolation is invalid."}
        return comparison
    big, small = coarse["stencil_step"]**2, fine["stencil_step"]**2
    extrapolated_gradient = (big * np.asarray(fine["gradient"]) - small * gradient) / (big - small)
    extrapolated_hessian = (big * np.asarray(fine["hessian"]) - small * hessian) / (big - small)
    eigenvalues = np.linalg.eigvalsh(extrapolated_hessian)
    extrapolation = {"applied": True, "gradient": extrapolated_gradient.tolist(),
                     "hessian": extrapolated_hessian.tolist(), "hessian_eigenvalues": eigenvalues.tolist(),
                     "scope": "Assumes leading O(step^2) truncation; two scales do not establish convergence order or an error bound."}
    if eigenvalues[0] < 0 < eigenvalues[1] and np.linalg.cond(extrapolated_hessian) < 1e8:
        extrapolation.update(stationary_height_diagnostics(fine["tau"], extrapolated_gradient, extrapolated_hessian))
        extrapolation["saddle_hessian_retained"] = True
    else:
        extrapolation["saddle_hessian_retained"] = False
    comparison["richardson"] = extrapolation
    return comparison


def height_sensitivity(scales, phase_trace_spread):
    """Summarize empirical consistency without promoting any height sign."""
    comparisons = [compare_stencil_scales(a, b) for a, b in zip(scales, scales[1:])]
    heights = [p[key] for p in scales for key in ("tau", "quadratic_corrected_height")]
    terms = [float(phase_trace_spread)] + [p["absolute_curvature_height_scale"] for p in scales]
    for row, fine in zip(comparisons, scales[1:]):
        terms.extend([abs(row["quadratic_corrected_height_difference"]), abs(row["trace_transport_residual"])])
        richardson = row["richardson"]
        if richardson.get("saddle_hessian_retained"):
            heights.append(richardson["quadratic_corrected_height"])
            terms.extend([richardson["absolute_curvature_height_scale"],
                          abs(richardson["quadratic_corrected_height"] - fine["quadratic_corrected_height"])])
    sensitivity = float(max(terms))
    nominal = scales[-1]["quadratic_corrected_height"]
    signs = np.sign(heights)
    return {"status": "numerical sensitivity reported; sign not certified",
            "nominal_quadratic_corrected_height": nominal,
            "height_estimate_range": [float(min(heights)), float(max(heights))],
            "height_sign_consistent_across_estimates": bool(np.all(signs == signs[0]) and signs[0] != 0),
            "cyclic_trace_spread": float(phase_trace_spread),
            "empirical_height_sensitivity_scale": sensitivity,
            "height_to_sensitivity_ratio": float(abs(nominal) / sensitivity) if sensitivity > 0 else None,
            "scale_comparisons": comparisons,
            "scope": "Maximum observed quadratic correction scale, scale/transport discrepancy and cyclic trace spread; not an uncertainty bound. Unmeasured higher derivatives and operator errors may remain."}


def assess_height_changes(centers):
    """Compact point and secant diagnostics; never infer a root or interval."""
    rows = []
    for center in sorted(centers, key=lambda row: row["parameter"]):
        sensitivity = center.get("height_sensitivity")
        row = {"parameter": center["parameter"], "center": center["center"], "tau": center["tau"],
               "kind": center["kind"], "sensitivity_available": sensitivity is not None}
        if sensitivity is not None:
            for key in ("nominal_quadratic_corrected_height", "empirical_height_sensitivity_scale",
                        "height_sign_consistent_across_estimates", "height_to_sensitivity_ratio"):
                row[key] = sensitivity[key]
            if "gradient" in center and "correction" in center:
                row["gradient"] = center["gradient"]
                row["quadratic_stationary_center"] = (np.asarray(center["center"]) + center["correction"]).tolist()
        rows.append(row)
    changes = []
    for left, right in zip(rows, rows[1:]):
        step = right["parameter"] - left["parameter"]
        if step == 0 or not all(p["sensitivity_available"] for p in (left, right)):
            continue
        change = right["nominal_quadratic_corrected_height"] - left["nominal_quadratic_corrected_height"]
        scale = sum(p["empirical_height_sensitivity_scale"] for p in (left, right))
        velocity = (np.asarray(right["center"]) - left["center"]) / step
        comparison = {"parameters": [left["parameter"], right["parameter"]],
                        "quadratic_height_change": float(change), "secant_height_slope": float(change / step),
                        "sum_endpoint_sensitivity_scales": float(scale),
                        "secant_sensitivity_scale": float(scale / abs(step)),
                        "change_to_sensitivity_ratio": float(abs(change) / scale) if scale > 0 else None,
                        "center_secant_velocity": velocity.tolist()}
        if all("gradient" in p for p in (left, right)):
            comparison["residual_gradient_dot_center_secant_velocity"] = [float(np.asarray(p["gradient"]) @ velocity)
                                                                         for p in (left, right)]
            comparison["quadratic_center_secant_velocity"] = ((np.asarray(right["quadratic_stationary_center"])
                                                               - left["quadratic_stationary_center"]) / step).tolist()
        changes.append(comparison)
    return {"centers": rows, "adjacent_changes": changes,
            "scope": "Pointwise numerical sensitivities and finite secants only. A fixed-location AD slope equals the moving-saddle height derivative only at a stationary center; residual gradient times center velocity and changes of evaluation location require separate checks. No root, derivative sign, noncrossing interval or rigorous enclosure is certified."}


def audit_stencil(row, mapping, tolerance):
    h, center, beta = float(row["stencil_step"]), np.asarray(row["center"]), row["parameter"]
    require(h > 0 and np.isfinite(h), "invalid stencil scale")
    indices = np.asarray(row["point_indices"])
    require(indices.shape == (3, 3), "nine-point stencil required")
    f = np.empty((3, 3))
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            point = mapping[int(indices[iy, ix])]
            require(point["beta"] == beta and point["qualified"], "stencil uses wrong parameter or unqualified point")
            close(point["coordinate"], center + h * np.array([dx, dy]), "stencil point is misplaced", rtol=0, atol=1e-13)
            f[iy, ix] = point["tau"]
    gradient = np.array([(f[1, 2] - f[1, 0]) / (2 * h), (f[2, 1] - f[0, 1]) / (2 * h)])
    cross = (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0]) / (4 * h * h)
    hessian = np.array([[(f[1, 2] - 2 * f[1, 1] + f[1, 0]) / h**2, cross],
                        [cross, (f[2, 1] - 2 * f[1, 1] + f[0, 1]) / h**2]])
    values = np.linalg.eigvalsh(hessian)
    condition = np.linalg.cond(hessian)
    require(np.isfinite(condition) and condition < 1e8 and values[0] < 0 < values[1], "stencil is not a qualified saddle")
    correction = -np.linalg.solve(hessian, gradient)
    for key, value in (("tau", f[1, 1]), ("gradient", gradient), ("hessian", hessian),
                       ("hessian_eigenvalues", values), ("correction", correction),
                       ("correction_norm", np.linalg.norm(correction)), ("hessian_condition", condition)):
        close(row[key], value, "recomputed stencil differs: " + key, atol=1e-12)
    require(row["converged"] == (np.linalg.norm(correction) <= tolerance), "stencil convergence label differs")
    require(np.linalg.norm(correction) <= 2, "stencil exceeded trust radius")
    diagnostics = stationary_height_diagnostics(f[1, 1], gradient, hessian)
    residuals = [f[iy, ix] - (f[1, 1] + gradient @ (h * np.array([dx, dy]))
                 + .5 * (h * np.array([dx, dy])) @ hessian @ (h * np.array([dx, dy])))
                 for iy, dy in enumerate((-1, 0, 1)) for ix, dx in enumerate((-1, 0, 1))]
    return {"parameter": beta, "center": center.tolist(), "tau": float(f[1, 1]),
            **diagnostics, "hessian_eigenvalues": values.tolist(), "stencil_step": h,
            "quadratic_stencil_max_abs_residual": float(max(abs(np.asarray(residuals)))),
            "converged": row["converged"], "point_index": mapping[int(indices[1, 1])]["index"]}


def audit_saddle(row, mapping, raw, stencils, tolerance):
    require(row.get("converged") and row["history"], "accepted saddle lacks convergence history")
    checked = [audit_stencil(h, mapping, tolerance) for h in row["history"]]
    last_by_scale = {}
    for point in checked:
        last_by_scale[point["stencil_step"]] = point
    require(list(last_by_scale) == list(stencils) and all(p["converged"] for p in last_by_scale.values()),
            "accepted saddle did not converge at every declared stencil")
    final = checked[-1]
    require(row["parameter"] == final["parameter"], "accepted saddle parameter differs")
    for key in ("center", "tau", "hessian", "hessian_eigenvalues", "correction_norm"):
        close(row[key], final[key], "accepted saddle differs: " + key, atol=1e-12)
    primary = mapping[row["point_index"]]
    require(primary["index"] == final["point_index"], "accepted saddle linked to wrong center point")
    cyclic = row["cyclic_endpoint"]
    require(cyclic == raw[cyclic["index"]] and [p["phase"] for p in cyclic["phases"]] == [0, 6, 12],
            "accepted saddle lacks a bound cyclic endpoint")
    require(cyclic["coordinate"] == row["center"] and cyclic["beta"] == row["parameter"], "cyclic endpoint moved")
    require(abs(cyclic["tau"] - row["tau"]) < 1e-5 * cyclic["radius"], "cyclic endpoint trace disagrees")
    phase_spread = np.ptp([p["tau"] for p in cyclic["phases"]])
    sensitivity = height_sensitivity(list(last_by_scale.values()), phase_spread)
    return {**final, "kind": "continued saddle", "cyclic_point_index": cyclic["index"],
            "delta": cyclic["delta"], "radius": cyclic["radius"],
            "stencil_heights": [{"step": p["stencil_step"], "tau": p["tau"]} for p in last_by_scale.values()],
            "stencil_diagnostics": list(last_by_scale.values()), "height_sensitivity": sensitivity}


def accepted_centers(run, protocol, raw, result, root=ROOT):
    events, mapping = continuation_events(run, raw)
    args = protocol["args"]
    tolerance, stencils = args.get("correction_tolerance", .003), args.get("stencils", [.1, .03])
    for event in events:
        if event["stage"] == "saddle_iteration":
            audit_stencil(event, mapping, tolerance)
    accepted, saved = [], {}
    for path in sorted(run.glob("saddle-*.json")):
        row = read(path)
        key = (row["parameter"], *row["center"])
        require(key not in saved, "duplicate accepted saddle artifact")
        saved[key] = (row, audit_saddle(row, mapping, raw, stencils, tolerance))
    for point in result.get("points", []):
        if "receipt_sha256" in point:
            blob = git_bytes(protocol["commit"], "docs/baby-shrimp-zoom/crossing-receipt.json", root)
            require(hashlib.sha256(blob).hexdigest() == point["receipt_sha256"], "prior saddle receipt binding differs")
            receipt = json.loads(blob)
            require(receipt["summary"]["qualification"]["numerically_resolved_nonzero_saddle"], "prior saddle unqualified")
            prior = receipt["summary"]["qualification"]["converged_scales"][-1]
            require(point["parameter"] == 0 and point["center"] == [0, 0], "prior saddle parameter differs")
            for key in ("hessian", "hessian_eigenvalues"):
                close(point[key], prior[key], "prior saddle curvature differs", atol=1e-13)
            baseline = read(run / "baseline.json")["point"]
            close(point["tau"], baseline["tau"], "prior saddle replay height differs", atol=1e-15)
            accepted.append({"parameter": 0., "center": [0., 0.], "tau": point["tau"],
                             "kind": "previously audited BSC1 saddle estimate", "point_index": baseline["index"],
                             "delta": baseline["delta"], "radius": baseline["radius"],
                             "hessian_eigenvalues": point["hessian_eigenvalues"]})
        else:
            key = (point["parameter"], *point["center"])
            require(key in saved, "continuation result has no saved accepted saddle")
            stored, summary = saved.pop(key)
            require(all(stored.get(k) == v for k, v in point.items()), "accepted result differs from saddle artifact")
            accepted.append(summary)
    # A completed callback may have been saved just before an interruption of
    # the parent loop. Retain that qualified artifact, without completing a run.
    accepted.extend(pair[1] for pair in saved.values())
    return accepted, len([e for e in events if e["stage"] == "saddle_iteration"])


def plot_centers(centers, out):
    if not centers:
        return []
    centers = sorted(centers, key=lambda row: row["parameter"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    beta = np.array([p["parameter"] for p in centers])
    tau = np.array([p["tau"] for p in centers])
    coordinate = np.array([p["center"] for p in centers])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    fig.suptitle("Following the child's contraction saddle", fontsize=17, fontweight="bold")
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].plot(beta, tau, "o-", color="#175d8d", lw=1.3)
    axes[0].set(xlabel="Feedback memory parameter β", ylabel="Leading-pair trace at corrected saddle",
                title="Does the saddle height reach zero?")
    axes[1].plot(coordinate[:, 0], coordinate[:, 1], color="0.7", lw=1)
    scatter = axes[1].scatter(coordinate[:, 0], coordinate[:, 1], c=beta, cmap="viridis", s=55, edgecolors="white")
    axes[1].set(xlabel="Scaled steering displacement", ylabel="Scaled log-temperature displacement",
                title="The saddle moves as memory changes")
    fig.colorbar(scatter, ax=axes[1], label="β")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useOffset=False)
    fig.supxlabel("Only audited accepted centers are shown. Lines join samples; no exact crossing or superstability is certified.", fontsize=9)
    names = []
    for suffix in ("png", "pdf"):
        name = f"memory-saddle-continuation.{suffix}"
        fig.savefig(out / name, dpi=180)
        names.append(name)
    plt.close(fig)
    return names


def audit_run(run, out, *, root=ROOT, data=DATA, worker=None):
    run, out = Path(run).resolve(), Path(out).resolve()
    require(out != run and run not in out.parents, "audit output must not alter raw attempt directory")
    protocol = read(run / "protocol.json")
    anchor = bind_protocol(protocol, root, data)
    runtime = read(run / "runtime.json")
    require(0 < runtime["norm_sensitivity"] < 1e-9 and runtime["rotary_float64"] and runtime["threads"] == 8,
            "effective precision/runtime gate failed")
    raw, summaries, unqualified = {}, [], []
    for path in sorted(run.glob("point-*")):
        if not path.is_dir():
            continue
        record_path = path / "point.json"
        if not record_path.exists():
            unqualified.append({"directory": path.name, "reason": "missing point record"})
            continue
        row = read(record_path)
        if not row.get("qualified"):
            unqualified.append({"directory": path.name, "reason": "point incomplete or failed", "index": row.get("index")})
            continue
        row, summary = audit_point(path, protocol["plan"])
        require(row["index"] not in raw, "duplicate point index")
        raw[row["index"]] = row
        summaries.append(summary)
    require(raw, "no qualified point to audit")
    baseline = audit_baseline(run, raw, anchor)
    result_path = run / "result.json"
    result = read(result_path) if result_path.exists() else read(run / "continuation.json") if (run / "continuation.json").exists() else {}
    centers, iterations = [], 0
    if protocol["args"]["mode"] == "pilot":
        if result.get("complete"):
            require([p["beta"] for p in result["rows"]] == protocol["args"]["betas"], "pilot parameter rows differ")
            for row in result["rows"]:
                require(row == raw[row["index"]] and len(row["phases"]) == 3, "pilot row lacks qualified cyclic evidence")
    else:
        centers, iterations = accepted_centers(run, protocol, raw, result, root)
        if result.get("complete"):
            expected = protocol["args"]["betas"]
            if protocol["args"].get("anchor_saddle"):
                expected = [0.] + [b for b in expected if b != 0]
            require([p["parameter"] for p in result["points"]] == expected, "continuation parameter sequence incomplete")
    lifecycle = None
    if worker is not None:
        from baby_shrimp_transect_finish import verify_worker
        wr, folder, lifecycle, _ = verify_worker(Path(worker))
        require(folder.resolve() == run and wr["commit"] == protocol["commit"], "worker folder or source differs")
    resume = None
    if "resume" in protocol:
        from baby_shrimp_memory_resume import verify_resume, verify_replay
        bundle = verify_resume(protocol["resume"], protocol["plan"], root=root, data=data,
                               seen=(str(run),))
        replay_path = run / "resume-replay.json"
        replay = verify_replay(run, protocol["resume"], bundle, raw) if replay_path.exists() else {
            "passed": False, "endpoints": [], "status": "fresh endpoint replay not saved"}
        require(replay["passed"] or (not centers and not result.get("complete")),
                "continued target requires complete fresh endpoint replay")
        resume = {"provenance": bundle["provenance"], "fresh_replay": replay}
    complete = bool(result.get("complete") and not unqualified and not (run / "failure.json").exists())
    plots = plot_centers(centers, out)
    manifest = [{"path": str(p.relative_to(run)), "sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted(run.rglob("*")) if p.is_file()]
    receipt = {"id": "BSC3-memory-offline-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "execution_source_commit": protocol["commit"], "plan_sha256": protocol["plan_sha256"],
               "mode": protocol["args"]["mode"], "complete": complete,
               "status": "all saved-data gates passed" if complete else "partial attempt; qualified subset audited",
               "raw_directory": str(run), "runtime": runtime, "baseline": baseline,
               "qualified_points": summaries, "unqualified_or_incomplete_points": unqualified,
               "accepted_centers": centers, "stencil_iterations_recomputed": iterations,
               "height_change_diagnostics": assess_height_changes(centers),
               "worker": lifecycle, "resume": resume, "manifest": manifest, "figures": plots,
               "figure_visual_review": "pending" if plots else "not applicable",
               "limits": ["Saved-data replay, not a fresh model evaluation or interval certificate.",
                          "Ritz separation is conditional on the computed subspace; it is not a full-spectrum certificate.",
                          "Without return-state.npy, closure is checked against two recorded summaries; lower-lag distances are independently recomputed.",
                          "A trace-height sign change is distinct from exact superstability and does not identify native modal coupling."]}
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        source = "tools/baby_shrimp_memory_finish.py"
        require(hashlib.sha256(git_bytes(commit, source)).hexdigest() == digest(ROOT / source), "uncommitted analysis source")
        receipt = audit_run(args.run, args.out, worker=args.worker)
        receipt.update(analysis_commit=commit, analysis_source_sha256=digest(ROOT / source), threads=1)
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"out": str(args.out), "complete": receipt["complete"],
                          "points": len(receipt["qualified_points"]), "saddles": len(receipt["accepted_centers"])}))
    except BaseException as error:
        (args.out / "failed.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
