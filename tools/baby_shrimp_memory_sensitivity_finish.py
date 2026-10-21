"""Reproduce the saved explicit/total BSC3 sensitivity audits without a model.

Raw attempts are read-only. The optional earlier pilot is independently audited
only when supplied; its finite-offset slope is not substituted for a derivative.
No paid service, provider API, or model runtime is accessed.
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
import baby_shrimp_memory_finish as audit

ROOT, DATA = audit.ROOT, audit.DATA
read, require, close, digest = audit.read, audit.require, audit.close, audit.digest
PAIR_GATES = dict(relative_residual_tolerance=1e-6, separation_factor=100., condition_limit=1e8,
                  real_tolerance=1e-10, pair_gap_tolerance=1e-10)
TANGENT_GATES = dict(orbit_closure_tolerance=1e-11, solve_tolerance=1e-10,
                     tangent_closure_tolerance=1e-7, resolvent_condition_limit=1e10)


def load_arrays(path, allow_nan=()):
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    for name, value in result.items():
        require(value.dtype in (np.dtype("float64"), np.dtype("complex128")), "unexpected array precision: " + name)
        valid = np.isfinite(value) | (np.isnan(value) if name in allow_nan else False)
        require(valid.all(), "nonfinite sensitivity array: " + name)
    return result


def bind_attempt(directory, kind, root=ROOT, data=DATA):
    require(kind in ("explicit", "total"), "unknown sensitivity attempt kind")
    protocol = read(directory / "protocol.json")
    commit = protocol["source_commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "exact source commit missing")
    sources = set(audit.BASE_SOURCES) - {"tools/llm_feigenbaum.py"}
    sources.add("tools/trace_pair_sensitivity.py")
    sources.add("tools/baby_shrimp_memory_mechanism.py" if kind == "explicit" else "tools/baby_shrimp_memory_total.py")
    if kind == "total":
        sources.update(("tools/periodic_memory_sensitivity.py", "tools/baby_shrimp_orbits.py",
                        "tools/baby_shrimp_geometry.py", "tools/llm_feigenbaum.py"))
    require(sources <= set(protocol["source_sha256"]), "incomplete sensitivity source manifest")
    for path, expected in protocol["source_sha256"].items():
        require(hashlib.sha256(audit.git_bytes(commit, path, root)).hexdigest() == expected,
                "sensitivity source differs from Git: " + path)
    plan_path = protocol.get("args", {}).get("plan", "docs/baby-shrimp-zoom/saddle-memory-plan.json")
    blob = audit.git_bytes(commit, plan_path, root)
    require(hashlib.sha256(blob).hexdigest() == protocol["plan_sha256"], "sensitivity plan hash differs from Git")
    plan = json.loads(blob)
    require(plan["period"] == 18 and plan["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775",
            "unexpected model or orbit family")
    required = {"point.json", "points.npy", "phase-0.npz", "phase-6.npz", "phase-12.npz"}
    require(required <= set(plan["anchor_hashes"]), "incomplete dense-anchor bindings")
    anchor = Path(data) / plan["anchor_directory"]
    for path, expected in plan["anchor_hashes"].items():
        require(digest(anchor / path) == expected, "sensitivity dense-anchor hash differs: " + path)
    point = read(anchor / "point.json")
    close(plan["start"], [point["alpha"], point["log10_temperature"]], "sensitivity anchor moved", rtol=0, atol=0)
    runtime = read(directory / "runtime.json")
    require(runtime["threads"] == 8 and runtime["rotary_float64"] and 0 < runtime["norm_sensitivity"] < 1e-9,
            "sensitivity effective-precision gate failed")
    return protocol, plan, anchor, runtime


def audit_pair(arrays, matrix, row):
    left, right, values = (arrays[k] for k in ("left", "right", "multipliers"))
    n = len(matrix)
    require(matrix.shape == (n, n) and left.shape == right.shape == (n, 2) and values.shape == (2,),
            "leading-pair dimensions differ")
    require(row["gates"] == PAIR_GATES, "leading-pair qualification gates changed")
    require(np.isrealobj(left) and np.isrealobj(right) and np.isrealobj(values)
            and min(abs(values)) > 0 and abs(values[0] - values[1]) > 1e-10 * max(abs(values)),
            "finite distinct real pair required")
    spectrum = np.linalg.eigvals(matrix)
    spectrum = spectrum[np.argsort(-abs(spectrum))]
    require(audit.pair_gap(values, spectrum[:2]) < 1e-10 * max(abs(values)) + 1e-18,
            "saved pair differs from independently recomputed dense spectrum")
    third = float(abs(spectrum[2])) if len(spectrum) > 2 else 0.
    require(min(abs(values)) > 100 * third, "pair lost dense spectral separation")
    rres = np.linalg.norm(matrix @ right - right * values, axis=0) / (abs(values) * np.linalg.norm(right, axis=0))
    lres = np.linalg.norm(matrix.T @ left - left * values, axis=0) / (abs(values) * np.linalg.norm(left, axis=0))
    gram_error = float(np.linalg.norm(left.T @ right - np.eye(2), ord=np.inf))
    condition = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
    require(max(max(rres), max(lres)) < 1e-6 and max(condition) <= 1e8 and gram_error < 1e-5,
            "saved left/right pair gate failed")
    for key, value in (("multipliers", values), ("trace", values.sum()), ("determinant", values.prod()),
                       ("right_relative_residuals", rres), ("left_relative_residuals", lres),
                       ("eigenvalue_condition_estimates", condition), ("biorthogonality_error", gram_error)):
        close(row[key], value, "saved leading-pair summary differs: " + key, atol=1e-12)
    # The tiny third eigenvalue is near its numerical floor. Require the same
    # resolved isolation scale, without giving it spurious relative precision.
    close(row["third_modulus"], third, "third-modulus record differs", rtol=1e-3, atol=1e-15)
    require(row["passed"], "pair was not qualified by the producer")
    return {"multipliers": values.tolist(), "maximum_right_relative_residual": float(max(rres)),
            "maximum_left_relative_residual": float(max(lres)), "biorthogonality_error": gram_error,
            "eigenvalue_condition_estimates": condition.tolist(), "third_modulus": third}


def contract(left, columns, stored, key=None):
    require(left.shape == columns.shape and np.isfinite(columns).all(), "invalid derivative columns")
    values = np.einsum("ij,ij->j", left, columns)
    trace = float(values.sum())
    expected_values = stored["multiplier_derivatives"] if key is None else stored["multiplier_derivatives"][key]
    expected_trace = stored["trace_derivative"] if key is None else stored["trace_derivatives"][key]
    close(values, expected_values, "saved multiplier derivative differs", atol=1e-12)
    close(trace, expected_trace, "saved trace derivative differs", atol=1e-12)
    return values.tolist(), trace


def audit_explicit(directory, root=ROOT, data=DATA):
    protocol, plan, anchor, runtime = bind_attempt(directory, "explicit", root, data)
    result = read(directory / "result.json")
    require(result["complete"] and not (directory / "failure.json").exists(), "explicit sensitivity did not complete")
    require([r["phase"] for r in result["rows"]] == [6, 0, 12], "all three declared explicit phases required")
    rows = []
    for row in result["rows"]:
        phase = row["phase"]
        arrays = load_arrays(directory / f"phase-{phase}.npz")
        with np.load(anchor / f"phase-{phase}.npz", allow_pickle=False) as z:
            matrix = z["matrix"]
        pair = audit_pair(arrays, matrix, row)
        derivatives, trace = contract(arrays["left"], arrays["applied_derivative"], row)
        direct, dense = arrays["baseline_jvp"], matrix @ arrays["right"]
        errors = np.linalg.norm(direct - dense, axis=0) / np.maximum(np.linalg.norm(direct, axis=0), 1e-300)
        require(max(errors) < 1e-7, "explicit composed-operator gate failed")
        close(errors, row["composed_jvp_relative_errors"], "explicit operator errors differ", rtol=1e-5, atol=1e-12)
        rows.append({"phase": phase, "trace_derivative": trace, "multiplier_derivatives": derivatives,
                     "operator_relative_errors": errors.tolist(), "pair": pair})
    traces = np.array([r["trace_derivative"] for r in rows])
    spread = float(np.ptp(traces))
    close(result["cyclic_derivative_spread"], spread, "explicit cyclic spread differs", atol=1e-15)
    require(spread <= 1e-5 * max(abs(traces)), "explicit cyclic derivatives disagree")
    if (directory / "partial.json").exists():
        require(read(directory / "partial.json") == result["rows"], "explicit partial/final records disagree")
    return {"source_commit": protocol["source_commit"], "plan_sha256": protocol["plan_sha256"],
            "anchor_directory": str(anchor), "anchor_hashes": plan["anchor_hashes"], "runtime": runtime,
            "passed": True, "rows": rows, "cyclic_derivative_spread": spread,
            "cyclic_derivative_relative_spread": float(spread / max(abs(traces))),
            "scope": "Explicit identity-feedback derivative at the frozen original orbit; three cyclic phases."}


def audit_orbit_tangent(states, replayed, arrays, matrix, qualification, row):
    n = len(matrix)
    require(states.shape == replayed.shape == (19, n) and arrays["tangents"].shape == (19, n),
            "orbit/tangent dimensions differ")
    require(row["gates"] == TANGENT_GATES, "periodic-tangent gates changed")
    replay_error = np.linalg.norm(replayed - states, axis=1) / np.maximum(np.linalg.norm(states, axis=1), 1e-300)
    close(max(replay_error), qualification["maximum_replayed_state_relative_difference"], "orbit replay summary differs", atol=1e-16)
    require(max(replay_error) < 1e-11, "orbit replay failed")
    returns = np.linalg.norm(replayed[1:] - replayed[0], axis=1) / np.linalg.norm(replayed[0])
    primitive = qualification["primitive"]
    close(returns, primitive["all_returns"], "replayed primitive returns differ", atol=1e-16)
    require(returns[-1] < 1e-11 and min(returns[:-1]) > 1e-8 and primitive["closed"] and primitive["no_lower_period"],
            "replayed orbit lost primitive period18")
    close(min(returns[:-1]), primitive["minimum_lower_return"], "primitive separation summary differs", atol=1e-14)
    require(primitive["primitive_separation_tolerance"] == 1e-8 and qualification["same_cyclic_text"] and
            audit.canonical(qualification["tokens"]) == audit.canonical([271, 785, 3405, 271, 785, 13885] * 3),
            "primitive tolerance or text family differs")
    close(arrays["forcing"], states[:-1] - states[1:], "periodic forcing differs", rtol=0, atol=0)
    orbit_closure = float(np.linalg.norm(states[-1] - states[0]) / np.linalg.norm(states[0]))
    tangents = arrays["tangents"]
    resolvent = np.eye(n) - matrix
    condition = float(np.linalg.cond(resolvent))
    solved = resolvent @ tangents[0]
    forcing_return = arrays["zero_start_return"]
    solve_error = float(np.linalg.norm(solved - forcing_return) / max(np.linalg.norm(solved), np.linalg.norm(forcing_return), 1e-300))
    tangent_scale = float(np.linalg.norm(tangents, axis=1).max())
    absolute_closure = float(np.linalg.norm(tangents[-1] - tangents[0]))
    tangent_closure = absolute_closure / max(tangent_scale, 1e-300)
    values = dict(orbit_relative_closure=orbit_closure, solve_relative_residual=solve_error,
                  tangent_relative_closure=tangent_closure, tangent_absolute_closure=absolute_closure,
                  maximum_phase_tangent_norm=tangent_scale, resolvent_condition=condition)
    for key, value in values.items():
        close(row[key], value, "periodic-tangent summary differs: " + key, atol=1e-12)
    require(orbit_closure < 1e-11 and solve_error < 1e-10 and tangent_closure < 1e-7 and condition <= 1e10
            and row["passed"], "periodic-tangent gate failed")
    return {**values, "replayed_orbit_closure": float(returns[-1]),
            "minimum_lower_return": float(min(returns[:-1])), "maximum_replayed_state_relative_difference": float(max(replay_error))}


def audit_fd(ad_arrays, ad_row, states, fd_rows, fd_arrays, steps):
    require(len(fd_rows) == len(fd_arrays) == len(steps) == 3, "three Hessian difference scales required")
    traces, column_errors, trace_errors = [], [], []
    for row, arrays, relative in zip(fd_rows, fd_arrays, steps, strict=True):
        require(row["relative_step"] == relative, "Hessian difference scale differs")
        norms = np.linalg.norm(ad_arrays["tangents"][:-1], axis=1)
        nonzero = norms > 0
        actual = np.full(len(norms), np.nan)
        actual[nonzero] = relative * np.linalg.norm(states[:-1], axis=1)[nonzero] / norms[nonzero]
        require(np.array_equal(np.isnan(arrays["actual_parameter_steps"]), ~nonzero), "zero-tangent FD mask differs")
        close(arrays["actual_parameter_steps"][nonzero], actual[nonzero], "normalized Hessian difference steps differ", rtol=1e-13, atol=0)
        declared = np.array([np.nan if value is None else value for value in row["actual_parameter_steps"]])
        require(np.array_equal(np.isnan(declared), ~nonzero), "JSON zero-tangent mask differs")
        close(declared[nonzero], actual[nonzero], "JSON normalized Hessian steps differ", rtol=1e-13, atol=0)
        close(arrays["total"], arrays["explicit"] + arrays["orbit_mediated"], "finite-difference decomposition differs", atol=1e-12)
        close(arrays["explicit"], ad_arrays["explicit"], "FD check changed explicit derivative", atol=1e-12)
        close(arrays["propagated"], ad_arrays["propagated"], "FD check changed base operator", atol=1e-14)
        derivatives = {key: contract(ad_arrays["left"], arrays[key], row, key)[1]
                       for key in ("explicit", "orbit_mediated", "total")}
        ce = np.linalg.norm(arrays["total"] - ad_arrays["total"], axis=0) / np.maximum(np.linalg.norm(ad_arrays["total"], axis=0), 1e-300)
        absolute = abs(derivatives["total"] - ad_row["trace_derivatives"]["total"])
        te = absolute / max(abs(ad_row["trace_derivatives"]["total"]), 1e-300)
        close(ce, row["total_column_relative_errors"], "FD column errors differ", rtol=1e-8, atol=1e-15)
        close(absolute, row["total_trace_absolute_error"], "FD absolute trace error differs", atol=1e-15)
        close(te, row["total_trace_relative_error"], "FD relative trace error differs", atol=1e-15)
        column_errors.append(float(max(ce)))
        trace_errors.append(float(te))
        traces.append({"relative_step": relative, "trace_derivatives": derivatives,
                       "maximum_column_relative_error": float(max(ce)), "trace_relative_error": float(te)})
    require(min(column_errors) > 0 and min(trace_errors) > 0, "unresolved finite-difference convergence order")
    column_orders = np.log(np.array(column_errors[:-1]) / column_errors[1:]) / np.log(np.array(steps[:-1]) / steps[1:])
    trace_orders = np.log(np.array(trace_errors[:-1]) / trace_errors[1:]) / np.log(np.array(steps[:-1]) / steps[1:])
    quadratic = bool(np.all((column_orders >= 1.5) & (column_orders <= 2.5)))
    joint = min(max(a, b) for a, b in zip(column_errors, trace_errors))
    require(quadratic and joint < 1e-5, "Hessian finite-difference precision gate failed")
    return {"rows": traces, "column_error_orders": column_orders.tolist(), "trace_error_orders": trace_orders.tolist(),
            "best_joint_trace_and_column_relative_error": joint, "passed": True}


def audit_total(directory, root=ROOT, data=DATA):
    protocol, plan, anchor, runtime = bind_attempt(directory, "total", root, data)
    result = read(directory / "result.json")
    require(result["complete"] and result["passed"] and not (directory / "failure.json").exists(), "total sensitivity did not pass")
    require(result["phase"] == protocol["phase"] == 6 and protocol["period"] == 18, "unexpected total-sensitivity phase or period")
    require(protocol["hessian_difference_steps"] == [1e-3, 3e-4, 1e-4]
            and protocol["gates"]["best_trace_and_column_relative_error"] == 1e-5
            and protocol["gates"]["orbit_connection_tolerance"] == 1e-11, "total-sensitivity precision gates changed")
    ad = read(directory / "ad-result.json")
    require(ad == result["ad"] and ad["passed"], "AD partial/final records disagree")
    arrays = load_arrays(directory / "ad-columns.npz")
    orbit = load_arrays(directory / "orbit-inputs.npz")
    states = orbit["states"]
    with np.load(anchor / "phase-6.npz", allow_pickle=False) as z:
        matrix = z["matrix"]
    original = np.load(anchor / "points.npy", allow_pickle=False)
    close(states[:-1], np.roll(original, -6, axis=0), "total sensitivity used a different original orbit", rtol=0, atol=0)
    pair = audit_pair(arrays, matrix, ad["pair"])
    replayed = orbit["replayed_states"]
    replay_errors = np.linalg.norm(replayed - states, axis=1) / np.maximum(np.linalg.norm(states, axis=1), 1e-300)
    close(replay_errors, orbit["replayed_relative_differences"], "saved replay errors differ", atol=1e-16)
    tangent = audit_orbit_tangent(states, replayed, arrays, matrix, read(directory / "orbit-qualification.json"), ad["orbit_tangent"])
    close(arrays["total"], arrays["explicit"] + arrays["orbit_mediated"], "AD derivative decomposition differs", atol=1e-12)
    values = {key: contract(arrays["left"], arrays[key], ad, key) for key in ("explicit", "orbit_mediated", "total")}
    dense = matrix @ arrays["right"]
    errors = np.linalg.norm(arrays["propagated"] - dense, axis=0) / np.maximum(np.maximum(
        np.linalg.norm(arrays["propagated"], axis=0), np.linalg.norm(dense, axis=0)), 1e-300)
    close(errors, ad["operator_relative_errors"], "total operator errors differ", rtol=1e-5, atol=1e-12)
    require(ad["operator_tolerance"] == 1e-7 and max(errors) < 1e-7, "total operator gate failed")
    steps = protocol["hessian_difference_steps"]
    fd_arrays = [load_arrays(directory / f"fd-columns-{h:.0e}.npz", allow_nan=("actual_parameter_steps",)) for h in steps]
    fd = audit_fd(arrays, ad, states, result["finite_differences"], fd_arrays, steps)
    close(fd["column_error_orders"], result["total_column_error_orders"], "reported column orders differ", atol=1e-12)
    close(fd["trace_error_orders"], result["total_trace_error_orders"], "reported trace orders differ", atol=1e-12)
    require(result["quadratic_regime_found"] and result["joint_trace_and_column_accuracy_passed"], "stored FD qualification differs")
    require(read(directory / "fd-partial.json") == result["finite_differences"], "FD partial/final records disagree")
    return {"source_commit": protocol["source_commit"], "plan_sha256": protocol["plan_sha256"],
            "anchor_directory": str(anchor), "anchor_hashes": plan["anchor_hashes"], "runtime": runtime,
            "passed": True, "phase": 6, "pair": pair, "orbit_tangent": tangent,
            "operator_relative_errors": errors.tolist(), "trace_derivatives": {k: v[1] for k, v in values.items()},
            "multiplier_derivatives": {k: v[0] for k, v in values.items()}, "finite_difference_audit": fd,
            "scope": "One cyclic phase; total derivative at fixed alpha and temperature, including periodic-orbit motion."}


def audit_optional_pilot(directory, output):
    receipt = audit.audit_run(directory, output)
    require(receipt["mode"] == "pilot" and receipt["complete"], "optional central-slope pilot did not qualify")
    rows = read(directory / "result.json")["rows"]
    require(all(row["coordinate"] == [0, 0] for row in rows), "optional pilot is not at one fixed parameter location")
    by_beta = {row["beta"]: row for row in rows}
    require(0 in by_beta and len(by_beta) == len(rows), "optional pilot lacks a unique zero-offset baseline")
    differences = []
    for h in sorted(b for b in by_beta if b > 0 and -b in by_beta):
        plus, minus, zero = (by_beta[b]["tau"] for b in (h, -h, 0))
        phase_maps = [{str(row["phase"]): row["tau"] for row in by_beta[b]["phases"]} for b in (h, -h)]
        require(all(set(rows) == {"0", "6", "12"} for rows in phase_maps), "optional pilot lacks the three cyclic phases")
        differences.append({"beta_step": h, "central_slope": (plus - minus) / (2 * h),
                            "central_curvature": (plus + minus - 2 * zero) / h**2,
                            "phase_slopes": {phase: (phase_maps[0][phase] - phase_maps[1][phase]) / (2 * h)
                                             for phase in ("0", "6", "12")}})
    require(differences, "optional pilot lacks a symmetric offset pair")
    plan = read(directory / "protocol.json")["plan"]
    anchor = read(DATA / plan["anchor_directory"] / "point.json")
    close(plan["start"], [anchor["alpha"], anchor["log10_temperature"]], "optional pilot moved from its dense anchor", rtol=0, atol=0)
    return {"source_commit": receipt["execution_source_commit"], "plan_sha256": receipt["plan_sha256"],
            "directory": str(directory.resolve()), "anchor_hashes": plan["anchor_hashes"],
            "start": plan["start"], "zero_beta_trace": by_beta[0]["tau"],
            "differences": differences, "scope": "Finite-offset fixed-location estimates; no derivative-convergence claim."}


def compare_pilots(pilots, total, anchor_hashes):
    require(pilots and total is not None, "central-slope validation requires supplied pilots and a total AD audit")
    combined = {}
    for pilot in pilots:
        require(pilot["anchor_hashes"] == anchor_hashes, "central-slope pilot used a different dense anchor")
        close(pilot["start"], pilots[0]["start"], "central-slope pilot moved parameter location", rtol=0, atol=0)
        close(pilot["zero_beta_trace"], pilots[0]["zero_beta_trace"], "central-slope pilot changed zero-offset baseline", rtol=1e-8, atol=1e-15)
        for row in pilot["differences"]:
            h = row["beta_step"]
            require(np.isfinite(h) and h > 0, "invalid central-slope offset")
            if h in combined:
                close(row["central_slope"], combined[h]["central_slope"], "duplicate central-slope estimates disagree", atol=1e-12)
            combined[h] = dict(row)
    derivative = total["trace_derivatives"]["total"]
    require(np.isfinite(derivative) and derivative != 0, "relative central-slope comparison needs a nonzero AD derivative")
    rows = [combined[h] for h in sorted(combined, reverse=True)]
    for row in rows:
        row["relative_error_against_ad"] = abs(row["central_slope"] - derivative) / abs(derivative)
        require(set(row["phase_slopes"]) == {"0", "6", "12"}, "central-slope pilot lacks phase comparisons")
        row["phase_relative_errors_against_ad"] = {phase: abs(slope - derivative) / abs(derivative)
                                                    for phase, slope in row["phase_slopes"].items()}
    errors = np.array([r["relative_error_against_ad"] for r in rows])
    steps = np.array([r["beta_step"] for r in rows])
    require(errors[-1] < 1e-3, "finest central-slope estimate fails AD agreement gate")
    require(max(rows[-1]["phase_relative_errors_against_ad"].values()) < 1e-3,
            "a finest-offset cyclic phase fails AD agreement gate")
    orders = (np.log(errors[:-1] / errors[1:]) / np.log(steps[:-1] / steps[1:])).tolist() if len(rows) > 1 and min(errors) > 0 else []
    require(not orders or all(1.5 <= value <= 2.5 for value in orders), "central-slope convergence is not quadratic")
    return {"rows": rows, "error_orders": orders, "finest_relative_error_tolerance": 1e-3,
            "passed": True, "scope": "Direct symmetric finite-offset slopes at fixed alpha/temperature; no moving-saddle-height claim."}


def manifest(directory):
    return {"directory": str(directory.resolve()), "files": [
        {"path": str(p.relative_to(directory)), "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*")) if p.is_file()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit", type=Path, required=True)
    parser.add_argument("--total", type=Path)
    parser.add_argument("--pilot", type=Path, action="append", default=[], help="Explicitly include an audited symmetric pilot; repeat for several attempts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.pilot or args.total is not None, "--pilot requires --total for derivative validation")
    runs = [p.resolve() for p in (args.explicit, args.total, *args.pilot) if p is not None]
    output = args.out.resolve()
    require(all(output != p and p not in output.parents for p in runs), "audit output must not alter raw attempts")
    require(output.is_relative_to(DATA.resolve()), "audit output must remain in external data storage")
    output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        sources = ["tools/baby_shrimp_memory_sensitivity_finish.py", "tools/baby_shrimp_memory_finish.py"]
        for path in sources:
            require(hashlib.sha256(audit.git_bytes(commit, path)).hexdigest() == digest(ROOT / path), "uncommitted analysis source: " + path)
        explicit = audit_explicit(args.explicit)
        total = audit_total(args.total) if args.total else None
        pilots = [audit_optional_pilot(path, output) for path in args.pilot]
        pilot_comparison = compare_pilots(pilots, total, explicit["anchor_hashes"]) if pilots else None
        comparison = None
        if total:
            require(explicit["anchor_hashes"] == total["anchor_hashes"], "explicit and total responses used different anchors")
            original = next(row for row in explicit["rows"] if row["phase"] == 6)["trace_derivative"]
            comparison = {"phase": 6, "explicit_cross_run_absolute_difference": abs(original - total["trace_derivatives"]["explicit"])}
        receipt = {"id": "BSC3-memory-sensitivity-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "analysis_commit": commit, "analysis_source_sha256": {p: digest(ROOT / p) for p in sources},
                   "analysis_threads": 1, "passed": True, "explicit": explicit, "total": total,
                   "optional_pilots": pilots, "central_slope_validation": pilot_comparison, "comparison": comparison,
                   "manifests": [manifest(p) for p in runs],
                   "limits": ["Offline replay of saved evidence, not a fresh model evaluation or interval certificate.",
                              "The original runtime python field came from a PATH python --version probe, not the executing interpreter; it is preserved as metadata and does not identify the executor.",
                              "The total response is checked at one cyclic phase and fixed original alpha/temperature.",
                              "A moving saddle-height derivative additionally needs stationarity and saddle-motion evidence.",
                              "Neither injected-memory sensitivity identifies native modal coupling nor establishes exact superstability.",
                              "Individual Jacobian/Hessian actions are not retained, so their propagation cannot be reconstructed from these saved arrays alone."]}
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"out": str(output), "passed": True, "total_included": total is not None}))
    except BaseException as error:
        (output / "failed.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
