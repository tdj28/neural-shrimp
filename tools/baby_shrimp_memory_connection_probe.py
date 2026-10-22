"""Direct spectral contour probes on SHA-bound memory-perturbed P18 orbits.

This exploratory probe keeps full orbit/spectral evidence at every evaluated
point. It measures trace zeros and finite transverse radius comparisons, not
exact contour connectivity, exact superstability or a rigorous ridge theorem.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
SOURCES = ["tools/baby_shrimp_memory_connection_probe.py", "tools/baby_shrimp_saddle_memory.py",
           "tools/baby_shrimp_memory.py", "tools/baby_shrimp_crossing.py",
           "tools/baby_shrimp_memory_orbit_alignment.py", "tools/baby_shrimp_memory_finish.py",
           "tools/baby_shrimp_memory_resume.py", "tools/baby_shrimp_orbits.py",
           "tools/baby_shrimp_memory_determinant_gradient.py",
           "tools/baby_shrimp_geometry.py", "tools/doubly_superstable.py",
           "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
           "tools/explore_llm_phase_map.py", "tools/explore_latent_loop.py", "tools/llm_feigenbaum.py"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_probe(plan):
    c = plan["connection_probe"]
    require(plan["id"] == "BSC4-direct-connection" and plan["period"] == 18
            and 1 <= plan["maximum_evaluations"] <= 120, "wrong probe identity, period or point budget")
    require(c["before_beta"] == .006 and c["after_beta"] == .007, "probe requires the two audited beta cases")
    require(c["axis_offsets"] == [-.1, -.05, .05, .1], "declare all four signed axis offsets")
    steps = c["stencil_steps"]
    require(len(steps) == 2 and all(np.isfinite(x) and x > 0 for x in steps)
            and steps[0] > steps[1] and steps[0] <= .003, "two decreasing fine stencil steps required")
    require(c["slice_offsets"] == [0., -.08, .08], "central slice must precede the two outer slices")
    require(np.isfinite(c["root_bracket"]) and 0 < c["root_bracket"] <= .3,
            "invalid bounded root bracket")
    for name in ("maximum_correction_norm", "root_coordinate_tolerance", "root_trace_tolerance"):
        require(np.isfinite(c[name]) and c[name] > 0, "invalid " + name)
    require(c["maximum_correction_norm"] <= 3e-5 and c["root_coordinate_tolerance"] <= 1e-7
            and c["root_trace_tolerance"] <= 1e-10, "probe gate was loosened")
    require(isinstance(c["root_max_iterations"], int) and 1 <= c["root_max_iterations"] <= 20,
            "invalid root iteration budget")
    offsets = c["normal_offsets"]
    require(len(offsets) == 2 and np.isfinite(offsets).all()
            and offsets[0] < 0 < offsets[1] and offsets[0] == -offsets[1]
            and offsets[1] <= .003, "two symmetric bounded normal offsets required")
    require(c["candidate_tilt_offsets"] == [-.0003, -.0001, .0001, .0003],
            "declare all four candidate tilt offsets")
    return c


def checked(evaluate, center, beta, *, cyclic=False):
    center = np.asarray(center, float)
    p = evaluate(center, float(beta), cyclic=cyclic)
    require(p.get("qualified") is True and p["beta"] == beta
            and np.array_equal(p["coordinate"], center), "unqualified or misplaced evaluated point")
    require(np.isfinite([p["tau"], p["delta"], p["radius"]]).all()
            and p["delta"] < 0 and 0 < p["radius"] < 1, "invalid attracting opposite-sign pair")
    if cyclic:
        require([row["phase"] for row in p["phases"]] == [0, 6, 12], "root requires three cyclic phases")
    return p


def saddle_basis(hessian):
    H = np.asarray(hessian, float)
    require(H.shape == (2, 2) and np.isfinite(H).all() and np.allclose(H, H.T, rtol=0, atol=1e-13),
            "finite symmetric Hessian required")
    values, basis = np.linalg.eigh(H)
    require(values[0] < 0 < values[1] and np.linalg.cond(H) < 1e8, "nondegenerate saddle Hessian required")
    # Fix signs for reproducible orientations, without changing axis identity.
    for col in range(2):
        i = int(np.argmax(abs(basis[:, col])))
        if basis[i, col] < 0:
            basis[:, col] *= -1
    return values, basis


def axis_diagnostics(evaluate, center, beta, hessian, offsets):
    center = np.asarray(center, float)
    values, basis = saddle_basis(hessian)
    central = checked(evaluate, center, beta)
    axes = []
    for col, label in enumerate(("negative curvature", "positive curvature")):
        samples = [{"offset": 0., "point": central}]
        for offset in offsets:
            samples.append({"offset": offset, "point": checked(evaluate, center + offset*basis[:, col], beta)})
        samples.sort(key=lambda row: row["offset"])
        brackets = [{"offsets": [a["offset"], b["offset"]],
                     "point_indices": [a["point"]["index"], b["point"]["index"]],
                     "heights": [a["point"]["tau"], b["point"]["tau"]]}
                    for a, b in zip(samples, samples[1:]) if a["point"]["tau"]*b["point"]["tau"] < 0]
        axes.append({"axis": label, "direction": basis[:, col].tolist(),
                     "samples": samples, "strict_sign_brackets": brackets})
    return {"beta": beta, "center": center.tolist(), "hessian_eigenvalues": values.tolist(),
            "basis_source": "Previously audited saddle Hessian; direction predictor only.", "axes": axes}


def radius_tilt(evaluate, center, beta, direction, offsets, save_progress):
    center, direction = np.asarray(center, float), np.asarray(direction, float)
    require(direction.shape == (2,) and np.isfinite(direction).all()
            and np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12), "unit tilt direction required")
    result = {"beta": beta, "center": center.tolist(), "direction": direction.tolist(),
              "center_point": checked(evaluate, center, beta), "samples": [], "symmetric_slopes": [],
              "complete": False,
              "scope": "Measured finite offsets at the prior residual-qualified candidate, not an exact stationary trace zero. No monotonicity or minimum is assumed."}
    save_progress(result)
    for offset in offsets:
        result["samples"].append({"offset": offset, "point": checked(evaluate, center+offset*direction, beta)})
        save_progress(result)
    indexed = {row["offset"]: row["point"] for row in result["samples"]}
    for step in sorted(x for x in indexed if x > 0):
        require(-step in indexed, "tilt offsets must have both signs")
        result["symmetric_slopes"].append({"step": step, **{
            key: float((indexed[step][key]-indexed[-step][key])/(2*step)) for key in ("tau", "delta", "radius")}})
    result["complete"] = True
    save_progress(result)
    return result


def matched_stencils(evaluate, center, beta, steps, max_correction):
    center = np.asarray(center, float)
    rows = []
    for h in steps:
        points = [[checked(evaluate, center+h*np.array([dx, dy]), beta)
                   for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
        f = np.array([[p["tau"] for p in row] for row in points])
        g = np.array([(f[1, 2]-f[1, 0])/(2*h), (f[2, 1]-f[0, 1])/(2*h)])
        cross = (f[2, 2]-f[0, 2]-f[2, 0]+f[0, 0])/(4*h*h)
        H = np.array([[(f[1, 2]-2*f[1, 1]+f[1, 0])/h**2, cross],
                      [cross, (f[2, 1]-2*f[1, 1]+f[0, 1])/h**2]])
        values, _ = saddle_basis(H)
        correction = -np.linalg.solve(H, g)
        row = {"center": center.tolist(), "beta": beta, "step": h, "tau": float(f[1, 1]),
               "gradient": g.tolist(), "hessian": H.tolist(), "hessian_eigenvalues": values.tolist(),
               "correction": correction.tolist(), "correction_norm": float(np.linalg.norm(correction)),
               "point_indices": [[p["index"] for p in line] for line in points]}
        rows.append(row)
        require(row["correction_norm"] <= max_correction, "same-center saddle correction exceeds declared gate")
    return rows


def direct_slices(evaluate, center, beta, stencil, cfg, save_progress):
    from baby_shrimp_crossing import trace_zero_slices
    center = np.asarray(center, float)
    H, g = np.asarray(stencil["hessian"]), np.asarray(stencil["gradient"])
    _, basis = saddle_basis(H)
    # Negative height: solve along positive curvature, traverse negative curvature.
    basis = basis[:, [1, 0]]
    require(checked(evaluate, center, beta)["tau"] < 0, "higher-beta center is not negative")
    settings = {key: cfg[key] for key in ("slice_offsets", "root_bracket", "root_coordinate_tolerance",
                                          "root_trace_tolerance", "root_max_iterations")}
    settings["slice_basis"] = basis.tolist()
    roots = []
    def measured(u):
        return checked(evaluate, center + np.asarray(u), beta)
    def completed(rows):
        source = rows[-1]
        # Preserve the root-search result before cyclic/offset probes can fail.
        root = {**source, "beta": beta, "center": center.tolist(), "slice_basis": basis.tolist(),
                "complete": False, "normal_offsets": []}
        roots.append(root)
        save_progress(roots)
        q = checked(evaluate, source["point"]["coordinate"], beta, cyclic=True)
        root["cyclic_point"] = q
        require(abs(q["tau"]) < cfg["root_trace_tolerance"], "cyclic root trace residual exceeds declared gate")
        gradient = g + H @ (np.asarray(q["coordinate"])-center)
        require(np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0, "undefined predicted transverse direction")
        normal = gradient/np.linalg.norm(gradient)
        root["normal_direction"] = normal.tolist()
        root["normal_metric"] = "Euclidean distance in declared scaled alpha/log10T coordinates; unit normal from the saved local quadratic trace gradient, not an exactly computed contour normal."
        for offset in cfg["normal_offsets"]:
            p = checked(evaluate, np.asarray(q["coordinate"])+offset*normal, beta)
            root["normal_offsets"].append({"offset": offset, "point": p,
                                           "radius_difference_from_root": p["radius"]-q["radius"]})
            save_progress(roots)
        differences = [p["radius_difference_from_root"] for p in root["normal_offsets"]]
        root["sampled_radius_comparison"] = ("root below both sampled offsets" if min(differences) > 0 else
                                             "at least one sampled offset has lower or equal radius")
        root["scope"] = "Two finite offsets measure directional radius changes, not existence or uniqueness of a transverse minimum or a ridge-connectivity theorem."
        root["complete"] = True
        save_progress(roots)
    trace_zero_slices(measured, settings, completed)
    return roots


def seed_records(bundle, cfg):
    receipt = bundle["receipt"]
    root = receipt.get("height_root") or {}
    require(receipt["complete"] and root.get("complete"), "complete audited continuation and root candidate required")
    parameters = [cfg["before_beta"], cfg["after_beta"], root["result"]["saddle"]["parameter"]]
    seeds = []
    for beta in parameters:
        matches = [c for c in receipt["accepted_centers"] if c["parameter"] == beta and c["kind"] == "continued saddle"]
        require(len(matches) == 1, "unique audited continued saddle required")
        c = matches[0]
        if beta == parameters[-1]:
            require(c["center"] == root["result"]["saddle"]["center"], "root candidate center differs from accepted saddle")
        directory = bundle["run"]/f"point-{c['cyclic_point_index']:04d}"
        p = json.loads((directory/"point.json").read_text())
        require(p["coordinate"] == c["center"] and p["beta"] == beta and p["qualified"], "seed point differs from saddle")
        require([x["phase"] for x in p["phases"]] == [0, 6, 12], "cyclic seed required")
        bases = {}
        for phase in (0, 6, 12):
            with np.load(directory/f"phase-{phase}.npz", allow_pickle=False) as z:
                bases[phase] = z["q"].copy()
        seeds.append({"center": c, "point": p, "orbit": np.load(directory/"points.npy", allow_pickle=False),
                      "bases": bases, "directory": directory})
    return seeds


def replay_seed(ev, seed, plan):
    from baby_shrimp_memory_finish import audit_point, pair_gap, complex_values
    source = seed["point"]
    fresh = checked(ev, source["coordinate"], source["beta"], cyclic=True)
    directory = ev.out/f"point-{fresh['index']:04d}"
    _, summary = audit_point(directory, plan)
    require(summary["closure_recomputed_from_saved_endpoint"], "fresh endpoint closure array missing")
    orbit = np.load(directory/"points.npy", allow_pickle=False)
    error = float(np.max(np.linalg.norm(orbit-seed["orbit"], axis=1)/np.linalg.norm(seed["orbit"], axis=1)))
    require(np.isfinite(error) and error < 1e-8, "fresh seed replay left the bound orbit sheet")
    comparisons = []
    for old, new in zip(source["phases"], fresh["phases"], strict=True):
        a, b = complex_values(old["multipliers"])[:2], complex_values(new["multipliers"])[:2]
        row = {"phase": old["phase"], "pair_relative_error": float(pair_gap(a, b)/max(abs(a))),
               "trace_error_relative_to_radius": float(abs(old["tau"]-new["tau"])/max(abs(a))),
               "determinant_relative_error": float(abs(old["delta"]-new["delta"])/abs(old["delta"]))}
        require(all(np.isfinite(v) and v < 1e-5 for k, v in row.items() if k != "phase"), "fresh cyclic seed spectral mismatch")
        comparisons.append(row)
    return {"source_point_index": source["index"], "source_directory": str(seed["directory"].relative_to(DATA)),
            "fresh_point_index": fresh["index"], "beta": source["beta"], "maximum_relative_orbit_difference": error,
            "cyclic_comparisons": comparisons, "passed": True}


def load_tilt_direction(plan, bundle, seed):
    """Recompute the bound root determinant gradient before using its direction."""
    import baby_shrimp_memory_determinant_gradient as diagnostic
    record = plan["determinant_audit"]
    path = (DATA/record["path"]).resolve()
    require(path.is_relative_to(DATA.resolve()) and digest(path) == record["sha256"]
            and path.stat().st_size == record["bytes"], "determinant audit binding differs")
    receipt = json.loads(path.read_text())
    for source, sha in receipt["analysis_source_sha256"].items():
        require(hashlib.sha256(subprocess.check_output(["git", "show", receipt["analysis_commit"]+":"+source], cwd=ROOT)).hexdigest() == sha,
                "determinant analysis source differs")
    matches = [r for r in receipt["accepted_centers"] if r["completed_numerical_height_root_candidate"]
               and r["receipt_sha256"] == bundle["sha256"] and r["parameter"] == seed["point"]["beta"]
               and r["center"] == seed["point"]["coordinate"]]
    require(len(matches) == 1, "unique bound candidate determinant diagnostic required")
    previous = matches[0]
    artifact = (DATA/previous["saddle_artifact"]).resolve()
    require(artifact.is_relative_to(bundle["run"]) and digest(artifact) == previous["saddle_sha256"],
            "determinant source saddle differs")
    saddle = json.loads(artifact.read_text())
    recomputed = diagnostic.stencil_diagnostics(saddle["history"][-1], diagnostic.callback_mapping(bundle),
                                                np.asarray(plan["coordinate_scales"]))
    require(recomputed == previous["stencils"][-1], "saved determinant diagnostic does not replay")
    gradient = np.asarray(recomputed["delta"]["gradient"])
    require(np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0, "candidate determinant direction unresolved")
    return {"direction": (-gradient/np.linalg.norm(gradient)).tolist(), "determinant_gradient": gradient.tolist(),
            "step": recomputed["step"], "audit": record, "saddle_sha256": previous["saddle_sha256"],
            "scope": "Negative of the independently replayed saved determinant gradient in scaled coordinates; a declared probe direction, not a nonzero-gradient certificate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    require(90 < args.seconds <= 5400, "runtime must exceed the 90-second reserve and remain within 5400 seconds")
    started = time.monotonic()
    plan_path = ROOT/args.plan
    plan = json.loads(plan_path.read_text()); cfg = validate_probe(plan)
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "exact source commit required")
    hashes = {path: digest(ROOT/path) for path in SOURCES}
    for path, sha in {**hashes, args.plan: digest(plan_path)}.items():
        require(hashlib.sha256(subprocess.check_output(["git", "show", commit+":"+path], cwd=ROOT)).hexdigest() == sha,
                "commit execution inputs before running: "+path)
    out = args.out.resolve()
    require(out.is_relative_to((DATA/"explore").resolve()) and not out.exists(), "new external explore directory required")
    out.mkdir(parents=True)
    def emit(stage, **values):
        row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
        with (out/"progress.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    protocol = {"commit": commit, "source_sha256": hashes, "plan": plan, "plan_sha256": digest(plan_path),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "started_utc": datetime.now(timezone.utc).isoformat()}
    save(out/"protocol.json", protocol)
    result = {"complete": False, "seed_replays": [], "candidate_tilt": None,
              "axes": [], "matched_stencils": [], "roots": []}
    ev = None
    try:
        from baby_shrimp_memory_orbit_alignment import load_receipt, METADATA
        from baby_shrimp_saddle_memory import Evaluator
        from baby_shrimp_memory_resume import seed_evaluator
        bundle = load_receipt(args.audit, args.audit_sha256)
        bound = plan["prior_audit"]
        require(args.audit.resolve() == (DATA/bound["path"]).resolve() and args.audit_sha256 == bound["sha256"],
                "audit differs from the committed plan")
        require(args.audit.stat().st_size == bound["bytes"], "audit byte size differs from the committed plan")
        require(all(plan[key] == bundle["scientific_metadata"][key] for key in METADATA if key != "id"),
                "probe changed the audited map or coordinate system")
        seeds = seed_records(bundle, cfg)
        tilt_source = load_tilt_direction(plan, bundle, seeds[-1])
        protocol["candidate_tilt_direction"] = tilt_source
        protocol["prior_audit"] = {**bound, "execution_source_commit": bundle["receipt"]["execution_source_commit"],
                                  "analysis_commit": bundle["receipt"]["analysis_commit"]}
        save(out/"protocol.json", protocol)
        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.seconds = int(args.seconds-(time.monotonic()-started)-90)
        require(evaluation_args.seconds > 0, "preflight consumed the evaluation budget")
        protocol["internal_evaluation_seconds"] = evaluation_args.seconds
        protocol["wall_time_reserve_seconds"] = 90
        save(out/"protocol.json", protocol)
        ev = Evaluator(evaluation_args, plan, out, emit)
        result["baseline"] = ev.baseline()
        seed_evaluator(ev, {"warm": [(np.asarray(s["center"]["center"]), s["point"]["beta"],
                                       s["orbit"][0].copy(), s["bases"]) for s in seeds]})
        fresh_warm = []
        for seed in seeds:
            replay = replay_seed(ev, seed, plan)
            result["seed_replays"].append(replay)
            directory = out/f"point-{replay['fresh_point_index']:04d}"
            bases = {}
            for phase in (0, 6, 12):
                with np.load(directory/f"phase-{phase}.npz", allow_pickle=False) as archive:
                    bases[phase] = archive["q"].copy()
            fresh_warm.append((np.asarray(seed["point"]["coordinate"]), seed["point"]["beta"],
                               np.load(directory/"points.npy", allow_pickle=False)[0].copy(), bases))
            save(out/"probe-progress.json", result)
        seed_evaluator(ev, {"warm": fresh_warm})
        def retain_tilt(row):
            result["candidate_tilt"] = row
            save(out/"probe-progress.json", result)
        radius_tilt(ev, seeds[-1]["point"]["coordinate"], seeds[-1]["point"]["beta"],
                    tilt_source["direction"], cfg["candidate_tilt_offsets"], retain_tilt)
        emit("candidate_tilt_complete", evaluations=ev.attempts)
        # Axis samples are first, preserving the before/after milestone if a later budget expires.
        for seed in seeds[:2]:
            c = seed["center"]
            result["axes"].append(axis_diagnostics(ev, c["center"], c["parameter"], c["hessian"], cfg["axis_offsets"]))
            save(out/"probe-progress.json", result)
            emit("axis_case_complete", beta=c["parameter"], evaluations=ev.attempts)
        center = seeds[1]["center"]["center"]
        result["matched_stencils"] = matched_stencils(ev, center, cfg["after_beta"], cfg["stencil_steps"],
                                                      cfg["maximum_correction_norm"])
        save(out/"probe-progress.json", result)
        emit("same_center_stencils_complete", evaluations=ev.attempts)
        def retain(roots):
            result["roots"] = roots
            save(out/"probe-progress.json", result)
            emit("trace_roots_progress", completed=sum(r["complete"] for r in roots), evaluations=ev.attempts)
        result["roots"] = direct_slices(ev, center, cfg["after_beta"], result["matched_stencils"][-1], cfg, retain)
        result.update(complete=True, evaluations=ev.attempts, completed_utc=datetime.now(timezone.utc).isoformat(),
                      scope="Direct numerical spectral samples on the continued P18 orbit; no exact crossing, strongest-contraction ridge or fixed-opener basin certificate.")
        save(out/"result.json", result)
    except BaseException as error:
        save(out/"failure.json", {"type": type(error).__name__, "error": str(error),
                                 "evaluations": None if ev is None else ev.attempts,
                                 "time": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        save(out/"manifest.json", {"complete": result["complete"], "files": [
            {"path": str(p.relative_to(out)), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"]})


if __name__ == "__main__":
    main()
