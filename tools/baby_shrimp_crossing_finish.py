"""Audit cyclic monodromies and decreasing-stencil trace-saddle evidence.

The local quadratic portrait is labelled as a model. Numerical separation is
not an interval certificate, nor does a dominant-sign exchange imply a zero.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
import baby_shrimp_finish as audit
from baby_shrimp_crossing import pair_discrepancy, saddle_step
from baby_shrimp_geometry import canonical, qualified_spectral_count
from baby_shrimp_probe_audit import precision_gate
from baby_shrimp_transect_finish import verify_worker


def primitive(points, record):
    returns = np.asarray(record["all_returns"])
    audit.require(points.shape == (18, 896) and returns.shape == (18,)
                  and np.isfinite(points).all() and np.isfinite(returns).all(), "incomplete primitive record")
    state_norm = float(np.linalg.norm(points[0]))
    audit.require(state_norm > 0, "nonzero orbit state required for relative returns")
    # SoftOrbit.return_distances divides by |X|, unlike the trajectory helper.
    distances = np.linalg.norm(points[1:] - points[0], axis=1) / state_norm
    audit.require(np.allclose(distances, returns[:-1], rtol=1e-7, atol=1e-14), "lower-return distances differ")
    passed = bool(returns[-1] < 1e-11 and (returns[:-1] > 1e-8).all())
    audit.require(passed == bool(record["closed"] and record["no_lower_period"]), "primitive label differs")
    return passed


def phase_gate(arrays, row, points):
    M, values, V = (arrays[k] for k in ("matrix", "multipliers", "vectors"))
    audit.require(M.shape == (896, 896) and values.shape == (896,) and V.shape == (896, 6), "matrix/spectrum shape differs")
    audit.require(all(np.isfinite(a).all() for a in arrays.values()), "nonfinite phase arrays")
    audit.require(np.all(np.diff(abs(values)) <= 1e-15), "multipliers not ordered")
    watched = qualified_spectral_count(values)
    audit.require(watched == row["watched"] and abs(values[1]) > 100 * abs(values[2]), "leading-pair separation differs")
    dense = M @ arrays["directions"]
    audit.require(np.allclose(dense, arrays["dense_jvp"], rtol=1e-10, atol=1e-12), "saved matrix-vector product differs")
    # Use the original saved product for the original gate; report replay separately.
    D, J = arrays["dense_jvp"], arrays["direct_jvp"]
    error = np.linalg.norm(D - J, axis=0) / np.maximum(np.maximum(np.linalg.norm(D, axis=0), np.linalg.norm(J, axis=0)), 1e-300)
    residual = np.linalg.norm(M @ V - V * values[:6], axis=0) / np.maximum(abs(values[:6]) * np.linalg.norm(V, axis=0), 1e-300)
    audit.require(np.allclose(error, arrays["jvp_relative_error"], rtol=1e-12, atol=1e-16)
                  and np.allclose(error, row["jvp_relative_error"], rtol=1e-12, atol=1e-16), "JVP error differs")
    recorded_residual = arrays["eigen_relative_residual"]
    audit.require(np.array_equal(recorded_residual, row["eigen_relative_residual"]), "eigen-residual record differs")
    good_primitive = primitive(points, row["primitive"])
    passed = bool(error.max() < 1e-7 and recorded_residual[:watched].max() < 1e-6 and good_primitive)
    audit.require(passed == row["passed"], "phase gate label differs")
    if passed:
        audit.require(residual[:watched].max() < 1e-6, "eigen-residual fails independent replay")
    audit.require(float((values[0] + values[1]).real) == row["tau"]
                  and float((values[0] * values[1]).real) == row["delta"], "pair invariant differs")
    stored = np.asarray(row["multipliers"])
    audit.require(np.array_equal(stored[:, 0] + 1j * stored[:, 1], values[:6]), "multiplier summary differs")
    return {"phase": row["phase"], "passed": passed, "maximum_jvp_error": float(error.max()),
            "maximum_watched_eigen_residual_replayed": float(residual[:watched].max()),
            "matrix_product_replay_absolute_error": float(np.linalg.norm(dense - D)),
            "tau": row["tau"], "delta": row["delta"]}


def raw_precision(folder, record, matrix=None):
    with np.load(folder / "derivative-check.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    exact, dense = a["direct_jvp"], a["dense_jvp"]
    error = float(np.linalg.norm(exact - dense) / max(np.linalg.norm(exact), np.linalg.norm(dense), 1e-300))
    fd = np.linalg.norm(a["finite_differences"] - exact, axis=1) / max(np.linalg.norm(exact), 1e-300)
    audit.require(np.allclose(fd, a["relative_errors"], rtol=1e-12, atol=1e-16)
                  and np.allclose(fd, record["central_difference_relative_errors"], rtol=1e-12, atol=1e-16)
                  and np.isclose(error, record["dense_vs_composed_jvp_relative_error"], rtol=1e-12, atol=1e-16)
                  and np.array_equal(a["relative_steps"], record["central_difference_relative_steps"]), "raw precision record differs")
    if matrix is not None:
        audit.require(np.allclose(matrix @ a["direction"], dense, rtol=1e-10, atol=1e-12), "precision matrix replay differs")
        values = np.linalg.eigvals(matrix)
        rho = float(abs(values).max())
        audit.require(abs(rho - record["dense_spectral_radius"]) < 1e-7 * rho,
                      "independent endpoint eigenspectrum differs")
    return precision_gate({"derivative_precision": record})


def sensitivity(endpoint, history):
    final_by_scale = []
    for row in history:
        if not final_by_scale or final_by_scale[-1]["stencil_step"] != row["stencil_step"]:
            final_by_scale.append(row)
        else:
            final_by_scale[-1] = row
    if len(final_by_scale) < 2:
        raise ValueError("at least two converged parameter stencil scales required")
    if any(np.linalg.norm(r["newton_step"]) >= .003 for r in final_by_scale):
        raise ValueError("an unconverged stencil cannot qualify a saddle")
    tau_change = abs(final_by_scale[-1]["tau"] - final_by_scale[-2]["tau"])
    endpoint_change = abs(endpoint["tau"] - final_by_scale[-1]["tau"])
    phase_spread = endpoint["phase_trace_spread"]
    scale = max(tau_change, endpoint_change, phase_spread, 1e-30)
    signs = [np.linalg.eigvalsh(r["hessian"]).tolist() for r in final_by_scale]
    saddle = all(v[0] < 0 < v[1] for v in signs)
    return {"converged_scales": final_by_scale, "last_two_saddle_value_difference": tau_change,
            "last_center_to_endpoint_value_difference": endpoint_change, "endpoint_phase_trace_spread": phase_spread,
            "trace_to_sensitivity_ratio": abs(endpoint["tau"]) / scale,
            "opposite_hessian_signs_at_every_scale": saddle,
            "numerically_resolved_nonzero_saddle": bool(saddle and abs(endpoint["tau"]) > 10 * scale),
            "scope": "Decreasing-stencil and cyclic-phase sensitivity, not a rigorous error bound or confidence interval."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result, folder, worker, manifest = verify_worker(args.worker)
    protocol = json.loads((folder / "protocol.json").read_text())
    plan = protocol["plan"]
    blob = subprocess.check_output(["git", "show", result["commit"] + ":" + protocol["args"]["plan"]], cwd=audit.ROOT)
    audit.require(hashlib.sha256(blob).hexdigest() == protocol["plan_sha256"] and json.loads(blob) == plan,
                  "crossing plan/source binding differs")
    seed = json.loads((folder / "seed/qualification.json").read_text())
    audit.require(seed["newton"][-1] < 1e-12 and seed["primitive"]["closed"] and seed["primitive"]["no_lower_period"]
                  and raw_precision(folder / "seed", seed["precision"]), "seed qualification fails")
    points, rows, matrices = {}, [], {}
    for path in sorted(folder.glob("point-*/point.json")):
        row = json.loads(path.read_text())
        audit.require(row["qualified"] and np.isfinite(row["newton"]).all() and row["newton"][-1] < 1e-12,
                      "unqualified crossing point retained; no complete conclusion")
        P = np.load(path.parent / "points.npy", allow_pickle=False)
        audit.require(primitive(P, row["primitive"])
                      and np.array_equal(P[0], np.load(path.parent / "state.npy", allow_pickle=False))
                      and canonical(row["tokens"]) == canonical([271, 785, 3405, 271, 785, 13885] * 3),
                      "point primitive/state/text differs")
        phases, spectra = [], []
        audit.require([p["phase"] for p in row["phases"]] == [0, 6, 12], "missing cyclic phase")
        selected_matrix = None
        for p in row["phases"]:
            shift = p["phase"]
            with np.load(path.parent / f"phase-{shift}.npz", allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            phases.append(phase_gate(a, p, np.roll(P, -shift, axis=0)))
            spectra.append(a["multipliers"][:2])
            if shift == row["selected_phase"]:
                selected_matrix = a["matrix"]
        passed = [p for p in phases if p["passed"]]
        audit.require(passed and passed[0]["phase"] == row["selected_phase"], "selected phase rule differs")
        selected = spectra[row["selected_phase"] // 6]
        discrepancy = max(pair_discrepancy(selected, v) for v in spectra)
        audit.require(discrepancy < 1e-5 * max(abs(selected))
                      and np.isclose(discrepancy, row["phase_pair_absolute_discrepancy"], rtol=1e-12, atol=1e-20),
                      "cyclic spectra disagree")
        audit.require(passed[0]["tau"] == row["tau"] and passed[0]["delta"] == row["delta"]
                      and float(abs(selected[0])) == row["spectral_radius"], "selected invariant differs")
        points[row["index"]] = row
        rows.append({k: row[k] for k in ("index", "alpha", "log10_temperature", "tau", "delta", "spectral_radius", "selected_phase")}
                    | {"audited_phases": phases})
        matrices[row["index"]] = selected_matrix
        print(json.dumps({"point": row["index"], "qualified_phases": len(passed)}), flush=True)
    final = json.loads((folder / "result.json").read_text())
    audit.require(final["complete"] and final["evaluations"] == len(points),
                  "crossing attempt incomplete")
    history = json.loads((folder / "saddle-history.json").read_text())
    audit.require(history == final["history"], "saddle history differs")
    u = np.zeros(2)
    for row in history:
        indices = row["point_indices"]
        values = np.array([[points[i]["tau"] for i in group] for group in indices])
        step, gradient, H = saddle_step(values, row["stencil_step"])
        audit.require(np.allclose(u, row["coordinate"], rtol=0, atol=1e-12)
                      and np.allclose(step, row["newton_step"], rtol=1e-12, atol=1e-12)
                      and np.array_equal(gradient, row["gradient"]) and np.array_equal(H, row["hessian"]),
                      "stationary-point update differs")
        for dy, group in zip((-1, 0, 1), indices):
            for dx, index in zip((-1, 0, 1), group):
                physical = np.asarray(plan["start"]) + np.asarray(plan["coordinate_scales"]) * (u + np.array([dx, dy]) * row["stencil_step"])
                audit.require(np.allclose(physical, [points[index]["alpha"], points[index]["log10_temperature"]], rtol=0, atol=1e-15),
                              "stencil parameter differs")
        u += step
    endpoint = final["endpoint"]
    audit.require(endpoint == points[endpoint["index"]], "endpoint record differs")
    original_precision_passed = raw_precision(folder, final["endpoint_precision"], matrices[endpoint["index"]])
    extension = None
    if args.precision:
        extension = json.loads((args.precision / "extension.json").read_text())
        ext_status = json.loads((args.precision / "status.json").read_text())
        audit.require(ext_status["status"] == "complete" and extension["complete"]
                      and extension["original_results_sha256"] == audit.digest(folder / "result.json"), "precision extension binding differs")
        for source, digest in ext_status["source_sha256"].items():
            saved = subprocess.check_output(["git", "show", ext_status["source_commit"] + ":" + source], cwd=audit.ROOT)
            audit.require(hashlib.sha256(saved).hexdigest() == digest, "precision source binding differs")
        from baby_shrimp_crossing_precision import select_phase
        audit.require(select_phase(extension["rows"]) == extension["selected_phase"], "precision phase selection differs")
        source = folder / f"point-{endpoint['index']:03d}"
        source_points = np.load(source / "points.npy", allow_pickle=False)
        audit.require(extension["source_points_sha256"] == audit.digest(source / "points.npy"), "precision point binding differs")
        for row in extension["rows"]:
            phase = row["phase"]
            path = source / f"phase-{phase}.npz"
            audit.require(audit.digest(path) == row["source_matrix_archive_sha256"], "precision matrix binding differs")
            with np.load(path, allow_pickle=False) as z:
                matrix = z["matrix"]
            raw_precision(args.precision / f"phase-{phase}", row["precision"], matrix)
            P = np.load(args.precision / f"phase-{phase}/points.npy", allow_pickle=False)
            audit.require(P.shape == (19, 896) and primitive(P[:-1], row["primitive"]), "local primitive replay fails")
            distances = np.linalg.norm(P[1:] - P[0], axis=1) / np.linalg.norm(P[0])
            audit.require(np.allclose(distances, row["primitive"]["all_returns"], rtol=1e-7, atol=1e-14), "local full return differs")
            drift = float(np.max(np.linalg.norm(P[:-1] - np.roll(source_points, -phase, axis=0), axis=1)
                                 / np.maximum(np.linalg.norm(P[:-1], axis=1), 1)))
            audit.require(drift < 1e-11 and drift == row["maximum_relative_orbit_replay_difference"], "local orbit replay differs")
        manifest.extend(audit.artifact(p) for p in sorted(args.precision.rglob("*")) if p.is_file())
    audit.require(original_precision_passed or extension is not None, "endpoint precision fails without a qualified extension")
    qualification = sensitivity(endpoint, history)
    values = np.array([r["tau"] for r in rows])
    determinant = np.array([r["delta"] for r in rows])
    summary = {"points": len(rows), "endpoint": endpoint,
               "trace_range": [float(values.min()), float(values.max())],
               "determinant_range": [float(determinant.min()), float(determinant.max())],
               "all_sampled_determinants_negative": bool((determinant < 0).all()),
               "qualification": qualification,
               "endpoint_precision": final["endpoint_precision"],
               "original_endpoint_precision_passed": original_precision_passed,
               "precision_extension": extension}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    scales = np.array(plan["coordinate_scales"])
    H = np.array(history[-1]["hessian"])
    x, y = np.meshgrid(np.linspace(-.5, .5, 251), np.linspace(-.5, .5, 251))
    quadratic = endpoint["tau"] + .5 * (H[0, 0] * x*x + 2 * H[0, 1] * x*y + H[1, 1] * y*y)
    extent = np.max(abs(quadratic))
    im = axes[0].pcolormesh(x * scales[0] * 1e6, y * scales[1] * 1e6, quadratic,
                            cmap="RdBu_r", vmin=-extent, vmax=extent, shading="auto", rasterized=True)
    axes[0].contour(x * scales[0] * 1e6, y * scales[1] * 1e6, quadratic, [0], colors="black", linestyles="dashed", linewidths=1.5)
    xy = np.array([[r["alpha"] - endpoint["alpha"], r["log10_temperature"] - endpoint["log10_temperature"]] for r in rows])
    visible = (abs(xy[:, 0]) <= .5 * scales[0]) & (abs(xy[:, 1]) <= .5 * scales[1])
    axes[0].scatter(xy[visible, 0] * 1e6, xy[visible, 1] * 1e6, c=values[visible],
                    cmap="RdBu_r", vmin=-extent, vmax=extent, s=23, edgecolors="white", linewidths=.6)
    axes[0].plot(0, 0, "*", color="#f2b134", markersize=16, markeredgecolor="#26334d")
    axes[0].set(title="Local quadratic model of the trace", xlabel="Steering offset from saddle (×10⁻⁶)",
                ylabel="log10 temperature offset (×10⁻⁶)")
    fig.colorbar(im, ax=axes[0], label="Trace of the leading multiplier pair")
    converged = qualification["converged_scales"]
    hs = [r["stencil_step"] for r in converged]
    ts = [r["tau"] for r in converged]
    axes[1].plot(hs, ts, "o-", color="#315bc4", label="Stationary trace at each stencil scale")
    axes[1].axhline(0, color="#8791a2", lw=.8)
    axes[1].set(xlabel="Normalized parameter-stencil spacing", ylabel="Trace at the stationary point",
                title="Does the saddle value converge to zero?")
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=8, loc="upper left", frameon=False)
    selected = next(p for p in endpoint["phases"] if p["phase"] == endpoint["selected_phase"])
    text = f"Final trace: {endpoint['tau']:.6g}\nDeterminant: {endpoint['delta']:.6g}\nLeading pair: "
    text += ", ".join(f"{v[0]:+.6g}" if abs(v[1]) < 1e-14 else f"{v[0]:+.4g}{v[1]:+.4g}i" for v in selected["multipliers"][:2])
    text += f"\nTrace / measured sensitivity: {qualification['trace_to_sensitivity_ratio']:.1f}"
    axes[1].text(.04, .12, text, transform=axes[1].transAxes, fontsize=10)
    fig.suptitle("Resolving the apparent crossing", fontsize=23, fontweight="bold")
    fig.supxlabel("Dots are full-network measurements; the colored field and dashed zero curves are a local quadratic approximation.\n"
                  "Three cyclic monodromies per point; all primitive and composed-derivative gates checked.\n"
                  "A nonzero trace saddle supports local avoidance. This numerical test is not an interval proof of the entire curves.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"baby-crossing-diagnostic.{ext}", dpi=170)
    plt.close(fig)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {"id": "BSC1-crossing-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip(),
               "source_sha256": audit.digest(__file__), "worker": worker, "plan_sha256": protocol["plan_sha256"],
               "summary": summary, "files": [audit.artifact(p) for p in args.out.iterdir() if p.is_file()],
               "limits": "Every retained eigenvector residual and stored matrix-vector product is replayed; the complete dense eigenspectrum is recomputed at the endpoint. Final closure uses the recorded full-state return; lower-lag separations are recomputed from all saved orbit points. Finite-difference sensitivity is not a rigorous enclosure. The quadratic field is an approximation and not a new high-resolution scan. No exact superstable intersection is certified.",
               "cost": "see private ledger"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("points", "trace_range", "determinant_range", "qualification")}))


if __name__ == "__main__":
    main()
