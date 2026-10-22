"""Independent saved-evidence checks for the BSC5 contraction-minimum pilot.

No model evaluation or provider access. The measurement schema is supplied by
the frozen pilot; these building blocks preserve failed and interrupted points.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import argparse
import subprocess
from datetime import datetime, timezone
from collections import Counter

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import baby_shrimp_memory_finish as point_audit
import baby_shrimp_memory_orbit_alignment as binding

ROOT, DATA = point_audit.ROOT, point_audit.DATA
require = point_audit.require
close = point_audit.close
EXECUTION_SOURCES = {
    "tools/baby_shrimp_contraction_minima.py", "tools/contraction_section_minima.py",
    "tools/baby_shrimp_memory_connection_probe.py", "tools/baby_shrimp_saddle_memory.py",
    "tools/baby_shrimp_memory.py", "tools/baby_shrimp_crossing.py",
    "tools/baby_shrimp_memory_orbit_alignment.py", "tools/baby_shrimp_memory_finish.py",
    "tools/baby_shrimp_memory_resume.py", "tools/baby_shrimp_orbits.py",
    "tools/baby_shrimp_memory_determinant_gradient.py", "tools/baby_shrimp_geometry.py",
    "tools/doubly_superstable.py", "tools/explore_llm_soft_loop.py",
    "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py",
    "tools/explore_latent_loop.py", "tools/llm_feigenbaum.py"}
SOURCES = ["tools/baby_shrimp_contraction_minima_finish.py", "tools/baby_shrimp_memory_finish.py",
           "tools/baby_shrimp_memory_orbit_alignment.py"]


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path, base=None):
    path = Path(path).resolve()
    base = DATA if base is None else Path(base).resolve()
    return {"path": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": digest(path)}


def bind_execution(protocol, *, expected_id, required_sources):
    """Require an exact Git-bound source set, plan and original dense anchor."""
    commit = protocol.get("commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "exact execution commit required")
    sources = protocol["source_sha256"]
    require(set(required_sources) <= set(sources), "execution source manifest incomplete")
    for path, expected in sources.items():
        require(hashlib.sha256(point_audit.git_bytes(commit, path)).hexdigest() == expected,
                "execution source hash differs: "+path)
    blob = point_audit.git_bytes(commit, protocol["args"]["plan"])
    plan = protocol["plan"]
    require(hashlib.sha256(blob).hexdigest() == protocol["plan_sha256"] and json.loads(blob) == plan,
            "committed plan contents or hash differ")
    require(plan["id"] == expected_id and plan["period"] == 18, "unexpected pilot family or period")
    anchor = DATA/plan["anchor_directory"]
    require(anchor.resolve().is_relative_to(DATA.resolve()), "anchor outside data store")
    for path, expected in plan["anchor_hashes"].items():
        require(digest(anchor/path) == expected, "original dense anchor hash differs")
    return anchor


def verify_snapshot(run, entries, *, omit_python_manifest=False):
    """Check the exact immutable regular-file snapshot; never sanitize raw data."""
    run = Path(run).resolve()
    require(isinstance(entries, list) and entries, "empty raw manifest")
    require(not any(path.is_symlink() for path in run.rglob("*")), "raw snapshot contains a symlink")
    names = set()
    for entry in entries:
        name = Path(entry["path"])
        require(not name.is_absolute() and ".." not in name.parts and name.as_posix() not in names,
                "invalid or duplicate raw manifest path")
        path = run/name
        require(path.is_file() and path.resolve().is_relative_to(run)
                and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                "raw artifact path/hash/size differs")
        names.add(name.as_posix())
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    require(actual - ({"manifest.json"} if omit_python_manifest else set()) == names,
            "unmanifested or missing raw files")


def audit_saved_points(run, plan):
    """Replay every qualified point and retain each failed/incomplete attempt."""
    run = Path(run)
    directories = sorted(path for path in run.glob("point-*") if path.is_dir())
    require([path.name for path in directories] == [f"point-{i:04d}" for i in range(len(directories))],
            "point directory sequence has gaps")
    require(len(directories) <= plan["maximum_evaluations"], "point budget exceeded")
    runtime = read(run/"runtime.json") if (run/"runtime.json").is_file() else None
    require(runtime is not None or not directories, "numerical points require runtime metadata")
    if runtime is not None:
        require(0 < runtime["norm_sensitivity"] < 1e-9 and runtime["rotary_float64"] is True
                and runtime["threads"] == 8, "effective precision or model runtime failed")
    raw, summaries, pending = {}, [], []
    for directory in directories:
        path = directory/"point.json"
        if not path.is_file():
            pending.append({"directory": directory.name, "status": "incomplete", "reason": "point record not saved"})
            continue
        try:
            point = read(path)
        except (ValueError, UnicodeDecodeError) as error:
            pending.append({"directory": directory.name, "status": "incomplete; unreadable point record",
                            "reason": str(error), "record": artifact(path, run)})
            continue
        require(isinstance(point.get("index"), int) and not isinstance(point["index"], bool)
                and directory.name == f"point-{point['index']:04d}", "point directory/index differs")
        if point.get("qualified") is not True:
            pending.append({"directory": directory.name, "index": point["index"],
                            "status": "not qualified; failed or interrupted", "saved_fields": sorted(point),
                            "beta": point.get("beta"), "coordinate": point.get("coordinate")})
            continue
        point, summary = point_audit.audit_point(directory, plan)
        require(summary["closure_recomputed_from_saved_endpoint"], "qualified point lacks saved return-state closure")
        raw[point["index"]] = point
        summaries.append(summary)
    return {"raw": raw, "qualified_points": summaries,
            "unqualified_or_incomplete_points": pending, "runtime": runtime,
            "attempted_points": len(directories)}


def audit_baseline(run, raw, anchor):
    """Replay a failed or passed admission without relabeling its outcome."""
    baseline = read(run/"baseline.json")
    if baseline.get("passed") is True:
        return point_audit.audit_baseline(run, raw, anchor)
    require(baseline.get("passed") is False, "baseline admission flag missing")
    point = raw.get(baseline["point"]["index"])
    require(point == baseline["point"] and point["beta"] == 0 and point["coordinate"] == [0, 0]
            and [p["phase"] for p in point["phases"]] == [0, 6, 12], "failed baseline point binding differs")
    with np.load(run/"baseline-operator-check.npz", allow_pickle=False) as z:
        arrays = dict(z)
    require(all(np.isfinite(a).all() for a in arrays.values()), "nonfinite failed baseline arrays")
    with np.load(anchor/"phase-6.npz", allow_pickle=False) as z:
        close(z["matrix"]@arrays["directions"], arrays["dense"], "failed baseline dense action differs", atol=1e-14)
    den = np.maximum(np.maximum(np.linalg.norm(arrays["direct"], axis=0), np.linalg.norm(arrays["dense"], axis=0)), 1e-300)
    errors = np.linalg.norm(arrays["direct"]-arrays["dense"], axis=0)/den
    close(errors, arrays["errors"], "failed baseline operator arrays differ", rtol=1e-5, atol=1e-12)
    close(errors, baseline["composed_jvp_relative_errors"], "failed baseline operator JSON differs", rtol=1e-5, atol=1e-12)
    expected = read(anchor/"point.json")
    tau_error = abs(point["tau"]-expected["tau"])/expected["spectral_radius"]
    close(tau_error, baseline["tau_error_relative_to_radius"], "failed baseline trace differs", atol=1e-15)
    comparisons = []
    for phase, saved in zip(point["phases"], baseline["cyclic_dense_comparisons"], strict=True):
        require(saved["phase"] == phase["phase"], "failed baseline phase differs")
        with np.load(anchor/f"phase-{(phase['phase']+6)%18}.npz", allow_pickle=False) as z:
            reference = z["multipliers"][:2]
        pair = point_audit.complex_values(phase["multipliers"])[:2]
        gap = point_audit.pair_gap(pair, reference)/max(abs(reference))
        det = abs(np.prod(pair)-np.prod(reference))/abs(np.prod(reference))
        close(gap, saved["pair_relative_error"], "failed baseline pair gap differs", atol=1e-15)
        close(det, saved["determinant_relative_error"], "failed baseline determinant gap differs", atol=1e-15)
        comparisons.append(gap < 1e-5 and det < 1e-5)
    require(not (max(errors) < 1e-7 and tau_error < 1e-5 and all(comparisons)), "recorded failed baseline actually meets gates")
    return {"passed": False, "point_index": point["index"], "maximum_operator_relative_error": float(max(errors)),
            "trace_error_relative_to_radius": float(tau_error), "status": "saved arrays replayed; baseline admission failed"}


def checked_point(raw, index, coordinate, beta):
    require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)) and index in raw,
            "measurement has no independently qualified point")
    point = raw[int(index)]
    require(point.get("qualified") is True and point["beta"] == beta
            and np.isfinite([point["tau"], point["delta"], point["radius"]]).all()
            and point["delta"] < 0 and 0 < point["radius"] < 1,
            "invalid measured opposite-sign attracting pair")
    close(point["coordinate"], coordinate, "measurement point coordinate differs", rtol=0, atol=1e-13)
    return point


def recompute_jet(jet, raw):
    """Replay a matched nine-point trace/determinant jet from actual point IDs."""
    center = np.asarray(jet["center"], float)
    step, beta = jet["step"], jet["beta"]
    require(center.shape == (2,) and np.isfinite(center).all() and np.isfinite(beta)
            and np.isfinite(step) and step > 0, "invalid jet center or scale")
    indices = np.asarray(jet["point_indices"])
    require(indices.shape == (3, 3) and np.issubdtype(indices.dtype, np.integer), "integer-indexed nine-point jet required")
    values = {key: np.empty((3, 3)) for key in ("tau", "delta")}
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            point = checked_point(raw, indices[iy, ix], center+step*np.array([dx, dy]), beta)
            for key in values:
                values[key][iy, ix] = point[key]
    fields = {}
    for key, f in values.items():
        gradient = np.array([(f[1, 2]-f[1, 0])/(2*step), (f[2, 1]-f[0, 1])/(2*step)])
        cross = (f[2, 2]-f[0, 2]-f[2, 0]+f[0, 0])/(4*step*step)
        hessian = np.array([[(f[1, 2]-2*f[1, 1]+f[1, 0])/step**2, cross],
                            [cross, (f[2, 1]-2*f[1, 1]+f[0, 1])/step**2]])
        actual = {"value": float(f[1, 1]), "gradient": gradient.tolist(), "hessian": hessian.tolist()}
        for name, value in actual.items():
            close(jet["fields"][key][name], value, "jet field differs: "+key+"/"+name,
                  rtol=1e-10, atol=1e-18 if name == "value" else 1e-13)
        fields[key] = actual
    H = np.asarray(fields["tau"]["hessian"])
    eigenvalues = np.linalg.eigvalsh(H)
    condition = float(np.linalg.cond(H))
    require(eigenvalues[0] < 0 < eigenvalues[1] and condition < 1e8, "jet does not define a nondegenerate trace saddle frame")
    correction = -np.linalg.solve(H, fields["tau"]["gradient"])
    if "hessian_eigenvalues" in jet:
        close(jet["hessian_eigenvalues"], eigenvalues, "jet Hessian eigenvalues differ", rtol=1e-10, atol=1e-13)
    close(jet["correction"], correction, "jet correction differs", rtol=1e-10, atol=1e-13)
    close(jet["correction_norm"], np.linalg.norm(correction), "jet correction norm differs", rtol=1e-10, atol=1e-13)
    return {"center": center.tolist(), "beta": beta, "step": step, "point_indices": indices.tolist(),
            "fields": fields, "correction": correction.tolist(), "correction_norm": float(np.linalg.norm(correction)),
            "trace_hessian_eigenvalues": eigenvalues.tolist(), "trace_hessian_condition": condition,
            "scope": "Trace/determinant finite differences; the trace frame is a predictor, not a contraction-minimum certificate."}


def audit_section_search(search, raw, origin, normal, beta, *, expected_settings=None):
    """Recompute mesh brackets, sampled golden steps, and both finite-offset gates.

    This intentionally does not import the search implementation. It audits the
    sampled section evidence; unresolved or unsampled extrema remain unresolved.
    """
    origin, normal = np.asarray(origin, float), np.asarray(normal, float)
    require(origin.shape == normal.shape == (2,) and np.isfinite(origin).all() and np.isfinite(normal).all()
            and abs(np.linalg.norm(normal)-1) < 1e-12, "finite unit section geometry required")
    cfg = search["settings"]
    if expected_settings is not None:
        require(all(cfg[key] == value for key, value in expected_settings.items()), "section settings differ from plan")
    mesh, tolerance, offsets = cfg["mesh"], cfg["coordinate_tolerance"], cfg["validation_offsets"]
    require(len(mesh) >= 3 and np.isfinite(mesh).all() and all(a < b for a, b in zip(mesh, mesh[1:]))
            and cfg["support"] == [mesh[0], mesh[-1]] and tolerance > 0
            and cfg["radius_floor"] >= 0 and cfg["signal_multiple"] >= 1, "invalid section settings")
    threshold = cfg["radius_floor"]*cfg["signal_multiple"]
    require(np.isfinite(threshold) and cfg["signal_threshold"] == threshold
            and len(offsets) == 2 and offsets[0] > offsets[1] > tolerance, "invalid section signal or offset settings")
    evaluations = search["evaluations"]
    require(len(evaluations) <= cfg["max_evaluations"] and search["evaluation_count"] == len(evaluations),
            "section callback count differs")
    samples, attempted = {}, set()
    for row in evaluations:
        s = row["s"]
        require(np.isfinite(s) and mesh[0] <= s <= mesh[-1] and s not in attempted, "invalid or repeated section callback coordinate")
        attempted.add(s)
        require(row["status"] in ("complete", "failed", "started"), "unknown section callback status")
        if row["status"] == "complete":
            p = checked_point(raw, row["index"], origin+s*normal, beta)
            require(row["radius"] == p["radius"], "section objective differs from qualified pair radius")
            samples[s] = {"s": s, "index": row["index"], "radius": row["radius"]}
        else:
            require(row["radius"] is None and row["index"] is None, "unfinished callback invents a measured objective")
    def sample(row):
        require(row.get("s") in samples and row == samples[row["s"]], "sample differs from completed section callback")
        return row
    measured_mesh = search["mesh_samples"]
    require([row["s"] for row in measured_mesh] == mesh[:len(measured_mesh)]
            and search["mesh_complete"] == (len(measured_mesh) == len(mesh)), "section mesh completion/order differs")
    for row in measured_mesh:
        sample(row)
    brackets = []
    for i in range(1, len(measured_mesh)-1):
        left, center, right = measured_mesh[i-1:i+2]
        if center["radius"] < min(left["radius"], right["radius"]):
            brackets.append({"mesh_center_position": i, "left": left, "center": center, "right": right,
                             "minimum_mesh_contrast": min(left["radius"], right["radius"])-center["radius"]})
    require(search["mesh_brackets"] == brackets and len(search["candidates"]) == len(brackets),
            "strict competing mesh brackets were changed or omitted")
    minimum_rows = []
    for candidate, bracket in zip(search["candidates"], brackets, strict=True):
        require(candidate["bracket"] == bracket and isinstance(candidate["qualified"], bool), "candidate bracket/qualification differs")
        require(candidate["status"] in ("converged", "low_signal", "nonconverged", "budget_exhausted", "evaluation_failed")
                and candidate["qualified"] is (candidate["status"] == "converged"), "candidate qualification/status differs")
        best = sample(candidate["best_sample"])
        require(bracket["left"]["s"] < best["s"] < bracket["right"]["s"], "candidate is not interior to original bracket")
        refinement = candidate["refinement"]
        steps = refinement["steps"]
        require(isinstance(steps, int) and not isinstance(steps, bool) and 0 <= steps <= cfg["max_refinement_steps"],
                "invalid refinement count")
        if "left" in refinement:
            left, middle, right = bracket["left"], bracket["center"], bracket["right"]
            for _ in range(steps):
                require(right["s"]-left["s"] > tolerance, "refinement continued after declared width gate")
                boundary = right if right["s"]-middle["s"] >= middle["s"]-left["s"] else left
                s = middle["s"]+(3-math.sqrt(5))/2*(boundary["s"]-middle["s"])
                require(s in samples and left["s"] < s < right["s"] and s != middle["s"], "golden step lacks its measured objective")
                trial = samples[s]
                if s < middle["s"]:
                    if trial["radius"] < middle["radius"]:
                        right, middle = middle, trial
                    else:
                        left = trial
                elif trial["radius"] < middle["radius"]:
                    left, middle = middle, trial
                else:
                    right = trial
            require(refinement["left"] == left and refinement["center"] == middle and refinement["right"] == right
                    and best == middle and refinement["width"] == right["s"]-left["s"], "refined measured bracket differs")
        else:
            require(steps == 0 and best == bracket["center"], "unstarted refinement invents a corrected minimum")
        validation = candidate["validation"]
        require([v["offset"] for v in validation] == offsets[:len(validation)], "validation scale/order differs")
        for row in validation:
            require(row["status"] in ("started", "complete") and len(row["samples"]) <= 2,
                    "invalid finite-offset validation status")
            contrasts = []
            for sign, observed in zip((-1, 1), row["samples"]):
                sample(observed)
                require(observed["s"] == best["s"]+sign*row["offset"]
                        and mesh[0] <= observed["s"] <= mesh[-1], "validation offset is misplaced or outside declared support")
                contrasts.append(observed["radius"]-best["radius"])
            require(row["contrasts"] == contrasts, "finite-offset contrast differs")
            if row["status"] == "complete":
                require(len(contrasts) == 2 and row["resolved"] is (min(contrasts) > threshold), "finite-offset signal gate differs")
                require(row["lower_neighbor_resolved"] is (min(contrasts) < -threshold), "resolved lower-neighbor gate differs")
            else:
                require("resolved" not in row and "lower_neighbor_resolved" not in row, "unfinished offsets invent a resolved gate")
        if candidate["qualified"]:
            require(candidate["status"] == refinement["status"] == "converged"
                    and refinement["width"] <= tolerance and bracket["minimum_mesh_contrast"] > threshold
                    and len(validation) == 2 and all(v["status"] == "complete" and v["resolved"] for v in validation),
                    "minimum lacks actual narrow bracket and both resolved scales")
            point = raw[best["index"]]
            minimum_rows.append({"point_index": best["index"], "s": best["s"], "coordinate": point["coordinate"],
                "beta": beta, "tau": point["tau"], "delta": point["delta"], "radius": point["radius"],
                "refined_bracket": [refinement["left"]["s"], refinement["right"]["s"]],
                "bracket_width": refinement["width"], "mesh_bracket": [bracket["left"]["s"], bracket["right"]["s"]],
                "validation": validation})
        elif candidate["status"] == "low_signal":
            require(bracket["minimum_mesh_contrast"] <= threshold
                    or len(validation) == 2 and all(v["status"] == "complete" for v in validation)
                    and not all(v["resolved"] for v in validation)
                    and not any(v["lower_neighbor_resolved"] for v in validation),
                    "low-signal label lacks an unresolved contrast gate")
    require(search["minima"] == [c for c in search["candidates"] if c["qualified"]], "qualified competing minima were omitted or added")
    counts = dict(Counter(c["status"] for c in search["candidates"]))
    require(search["candidate_status_counts"] == counts, "candidate status counts differ")
    status = search["status"]
    if status == "evaluation_failed":
        require(any(row["status"] == "failed" for row in evaluations), "failed search has no failed callback")
    elif status == "budget_exhausted":
        require(len(evaluations) == cfg["max_evaluations"], "section budget label lacks an exhausted callback budget")
    else:
        expected = "no_bracket" if not brackets else "nonconverged" if counts.get("nonconverged") else "low_signal" if counts.get("low_signal") else "converged"
        require(status == expected and all(row["status"] == "complete" for row in evaluations), "section terminal status differs")
    require(search["complete"] is (search["mesh_complete"] and status in ("converged", "low_signal", "no_bracket")),
            "section completion was promoted or changed")
    return {"status": status, "complete": search["complete"], "settings": cfg, "evaluation_count": len(evaluations),
            "strict_mesh_brackets": len(brackets), "candidate_status_counts": counts, "minima": minimum_rows,
            "scope": "Audited finite-section minima and unresolved brackets; no unique ridge, exact minimum, or connection certificate."}


def terminal_evidence(run, protocol, controller_manifest=None, manifest_sha=None,
                      controller_status=None, status_sha=None):
    supplied = (controller_manifest, manifest_sha, controller_status, status_sha)
    require(not any(supplied) or all(supplied), "controller manifest/status require explicit paired hashes")
    controller = None
    result = read(run/"result.json") if (run/"result.json").is_file() else None
    if all(supplied):
        paths = [Path(p).resolve() for p in (controller_manifest, controller_status)]
        require(all(p.is_file() and p.is_relative_to(DATA.resolve()) for p in paths), "controller packet outside data store")
        require(digest(paths[0]) == manifest_sha and digest(paths[1]) == status_sha, "controller packet hash differs")
        status = read(paths[1])
        require(status["worker_reaped"] is True and status["source_commit"] == protocol["commit"]
                and Path(status["output"]).resolve() == run and type(status["exit_code"]) is int
                and type(status["timed_out"]) is bool and type(status["complete"]) is bool
                and "ended_utc" in status, "controller lacks a bound reaped terminal worker")
        require(status["complete"] == (status["exit_code"] == 0 and result is not None)
                and not (status["complete"] and status["timed_out"]), "controller execution completion differs")
        # Execution completion and resolution of every scientific search differ.
        require(not status["complete"] or result.get("execution_finished") is True,
                "successful controller lacks finished execution result")
        require(type(status.get("scientific_complete")) is bool
                and status["scientific_complete"] is bool(result and result.get("complete") is True),
                "controller scientific completion differs")
        require(status["hard_seconds"] == protocol["args"]["seconds"] <= 10800
                and 0 <= status["termination_grace_seconds"] <= 10, "controller time bound differs")
        command = status["command"]
        for flag, value in (("--source-commit", protocol["commit"]), ("--out", str(run)),
                            ("--plan", protocol["args"]["plan"]), ("--device", protocol["args"]["device"]),
                            ("--seconds", str(protocol["args"]["seconds"]))):
            require(command.count(flag) == 1 and command.index(flag)+1 < len(command)
                    and command[command.index(flag)+1] == value, "controller command differs: "+flag)
        require("tools/baby_shrimp_contraction_minima.py" in command, "controller ran another worker")
        verify_snapshot(run, read(paths[0]))
        controller = {"manifest": artifact(paths[0]), "status_artifact": artifact(paths[1]),
            **{k: status[k] for k in ("worker_pid", "controller_pid", "worker_reaped", "exit_code",
                                     "timed_out", "elapsed_seconds", "ended_utc")},
            "execution_complete": status["complete"]}
    manifest_path = run/"manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        require(type(manifest.get("complete")) is bool, "Python manifest completion flag missing")
        verify_snapshot(run, manifest["files"], omit_python_manifest=True)
        require(result is not None or (run/"failure.json").is_file(), "Python finalization lacks terminal outcome")
        require(manifest["complete"] == (result or {"complete": False})["complete"], "manifest/result completion differs")
        require(not manifest["complete"] or not (run/"failure.json").exists(), "completed manifest contains failure")
    else:
        require(controller is not None, "missing Python finalization requires terminal controller packet")
    return {"python_finalization": manifest is not None, "python_complete": bool(manifest and manifest["complete"]),
            "python_manifest": artifact(manifest_path) if manifest else None, "controller": controller}


def bind_prior(protocol):
    plan, prior = protocol["plan"], protocol["plan"]["prior_audit"]
    require(protocol["prior_audit"] == prior, "protocol prior audit differs from plan")
    path = DATA/prior["path"]
    require(path.resolve().is_relative_to(DATA.resolve()) and path.stat().st_size == prior["bytes"], "prior audit location/size differs")
    bundle = binding.load_receipt(path, prior["sha256"])
    receipt = bundle["receipt"]
    require(receipt["complete"] and receipt["height_root"]["complete"], "prior candidate continuation is incomplete")
    require(all(plan[k] == bundle["scientific_metadata"][k] for k in binding.METADATA if k != "id"),
            "pilot changed audited map or coordinates")
    root = receipt["height_root"]["result"]["saddle"]
    matches = [r for r in receipt["accepted_centers"] if r.get("kind") == "continued saddle"
               and r["parameter"] == root["parameter"] and r["center"] == root["center"]]
    require(len(matches) == 1, "prior candidate not unique")
    seed = matches[0]
    lower = [r for r in receipt["accepted_centers"] if r.get("kind") == "continued saddle" and r["parameter"] < seed["parameter"]]
    require(lower, "missing neighboring predictor saddle")
    previous = max(lower, key=lambda r: r["parameter"])
    db = seed["parameter"]-previous["parameter"]
    hp = (seed["tau"]-previous["tau"])/db
    cp = (np.asarray(seed["center"])-previous["center"])/db
    expected = {"lower": {k: previous[k] for k in ("parameter", "center", "tau")},
                "upper": {k: seed[k] for k in ("parameter", "center", "tau")},
                "height_secant": hp, "center_secant": cp.tolist()}
    require(hp < 0 and np.isfinite(hp) and np.isfinite(cp).all(), "invalid predictor secants")
    if "predictor_derivatives" in protocol:
        require(all(protocol["predictor_derivatives"][k] == v for k, v in expected.items()), "prior predictor secants differ")
    return bundle, seed, expected


def audit_seed(run, saved, raw, bundle, seed):
    if saved is None:
        return None
    old_dir = bundle["run"]/f"point-{seed['cyclic_point_index']:04d}"
    old = read(old_dir/"point.json")
    new = checked_point(raw, saved["fresh_point_index"], seed["center"], seed["parameter"])
    require([r["phase"] for r in new["phases"]] == [0, 6, 12], "fresh seed lacks three phases")
    a = np.load(old_dir/"points.npy", allow_pickle=False)
    b = np.load(run/f"point-{new['index']:04d}"/"points.npy", allow_pickle=False)
    error = float(np.max(np.linalg.norm(b-a, axis=1)/np.linalg.norm(a, axis=1)))
    require(error < 1e-8, "fresh seed orbit sheet differs")
    comparisons = []
    for x, y in zip(old["phases"], new["phases"], strict=True):
        av, bv = point_audit.complex_values(x["multipliers"])[:2], point_audit.complex_values(y["multipliers"])[:2]
        row = {"phase": x["phase"], "pair_relative_error": float(point_audit.pair_gap(av, bv)/max(abs(av))),
               "trace_error_relative_to_radius": float(abs(x["tau"]-y["tau"])/max(abs(av))),
               "determinant_relative_error": float(abs(x["delta"]-y["delta"])/abs(x["delta"]))}
        require(all(v < 1e-5 for k, v in row.items() if k != "phase"), "fresh seed spectral comparison failed")
        comparisons.append(row)
    expected = {"source_point_index": old["index"], "source_directory": str(old_dir.relative_to(DATA)),
        "fresh_point_index": new["index"], "beta": new["beta"], "maximum_relative_orbit_difference": error,
        "cyclic_comparisons": comparisons, "passed": True}
    require(saved == expected, "saved seed replay differs")
    return expected


def audit_expanded(run, saved, raw, columns):
    path = run/"expanded-spectrum/check.json"
    array_paths = sorted(path.parent.glob("*.npz"))
    require(all(p.name in {"phase-0.npz", "phase-6.npz", "phase-12.npz"} for p in array_paths),
            "unexpected expanded array identity")
    if not path.is_file():
        require(saved is None, "expanded result lacks raw record")
        return {"status": "interrupted before spectrum row saved" if array_paths else "untested",
                "complete": False, "rows": [], "unrecorded_array_attempts": [artifact(p) for p in array_paths],
                "unrecorded_array_scope": "Hash-bound preserved attempts; no saved row exists to verify or admit their spectrum."}
    check = read(path)
    require(saved is None or saved == check, "expanded result differs from raw check")
    require(type(check.get("complete")) is bool and len(check["rows"]) <= 3, "invalid expanded check status")
    center = raw.get(check["point_index"])
    require(center is not None and [p["phase"] for p in center["phases"]] == [0, 6, 12], "expanded spectrum lacks qualified cyclic center")
    rows = []
    for phase, row in zip((0, 6, 12), check["rows"]):
        require(row["phase"] == phase and row["columns"] == columns == 12 and row["random_seed"] == 5801+phase,
                "expanded phase/rank/seed differs")
        array_path = path.parent/f"phase-{phase}.npz"
        with np.load(array_path, allow_pickle=False) as arrays:
            require(arrays["q"].shape == (896, columns), "expanded saved rank differs")
        # A saved expanded attempt may have valid arrays but fail attraction or
        # the opposite-sign regime. Preserve that failed admission separately.
        replay = replay_expanded_phase(array_path, row)
        original = next(p for p in center["phases"] if p["phase"] == phase)
        gap = float(point_audit.pair_gap(replay["pair"], point_audit.complex_values(original["multipliers"])[:2])/original["radius"])
        close(row["pair_relative_difference"], gap, "expanded pair discrepancy differs", atol=1e-18)
        passed = replay["delta"] < 0 and replay["radius"] < 1 and gap < 1e-5
        require(row["passed"] is passed, "expanded admission label differs")
        rows.append({k: v for k, v in replay.items() if k != "pair"} | {"pair_relative_difference": gap, "passed": passed})
    variations = None
    if check["complete"]:
        require(len(rows) == 3 and all(row["passed"] for row in rows), "completed expanded check lacks three passed phases")
        variations = {k: float(np.ptp([p[k] for p in center["phases"]]+[p[k] for p in rows])) for k in ("tau", "delta", "radius")}
        require(check["variations"] == variations, "expanded cyclic variations differ")
    recorded_paths = {f"phase-{row['phase']}.npz" for row in rows}
    unrecorded = [artifact(p) for p in array_paths if p.name not in recorded_paths]
    require(not check["complete"] or not unrecorded, "completed expanded check leaves unrecorded arrays")
    return {"status": "complete" if check["complete"] else "partial expanded check", "complete": check["complete"],
            "point_index": center["index"], "rows": rows, "variations": variations,
            "check": artifact(path), "unrecorded_array_attempts": unrecorded,
            "unrecorded_array_scope": "Hash-bound preserved attempts without a matching saved row are not admitted.",
            "scope": "Independent saved 12-column projected spectra; no full-spectrum certificate."}


def replay_expanded_phase(path, row):
    values = point_audit.complex_values(row["multipliers"])
    if float(max(abs(values))) < 1 and float(np.prod(values[:2]).real) < 0:
        return point_audit.audit_phase(path, row)
    require(row["passed"] is False, "inadmissible expanded spectrum is labeled passed")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    require({"q", "w", "values", "vectors", "direct", "relative_residuals"} <= set(arrays)
            and all(np.isfinite(a).all() for a in arrays.values()), "invalid failed expanded arrays")
    q, w, vectors, direct, saved = (arrays[k] for k in ("q", "w", "vectors", "direct", "values"))
    rank = len(saved)
    require(q.shape == w.shape == (896, rank) and vectors.shape == direct.shape == (896, 2)
            and all(arrays[k].dtype == np.float64 for k in ("q", "w", "vectors", "direct", "relative_residuals"))
            and saved.dtype in (np.dtype("float64"), np.dtype("complex128")), "failed expanded precision/shape differs")
    close(q.T@q, np.eye(rank), "failed expanded basis is not orthonormal", rtol=0, atol=1e-10)
    replay = np.linalg.eigvals(q.T@w)
    i, j = point_audit.linear_sum_assignment(abs(replay[:, None]-saved[None, :]))
    error = float(max(abs(replay[i]-saved[j])))
    require(error < 1e-11*max(abs(saved))+1e-18 and np.all(np.diff(abs(saved)) <= 1e-15*max(abs(saved))),
            "failed expanded projected spectrum differs")
    require(min(abs(saved[:2])) > 0 and max(abs(saved[:2].imag)) <= 1e-10*max(abs(saved[:2]))
            and abs(saved[1]) > 100*abs(saved[2]), "failed expanded pair is not the recorded isolated real pair")
    close(saved, values, "failed expanded JSON spectrum differs", rtol=1e-12, atol=1e-18)
    close(q@(q.T@vectors), vectors, "failed expanded vectors leave subspace", rtol=0, atol=1e-10)
    denominator = abs(saved[:2])*np.linalg.norm(vectors, axis=0)
    require(np.all(denominator > 0), "zero failed expanded residual denominator")
    residual = np.linalg.norm(direct-vectors*saved[:2].real, axis=0)/denominator
    projected = np.linalg.norm(w@(q.T@vectors)-vectors*saved[:2].real, axis=0)/denominator
    require(max(residual) < 1e-6 and max(projected) < 1e-6, "failed expanded solver residual differs")
    close(residual, arrays["relative_residuals"], "failed expanded saved residual differs", rtol=1e-5, atol=1e-12)
    close(residual, row["relative_residuals"], "failed expanded JSON residual differs", rtol=1e-5, atol=1e-12)
    require(len(row["history"]) >= 2 and max(row["history"][-1]["relative_residuals"][:2]) < 1e-6,
            "failed expanded convergence history differs")
    close(saved, point_audit.complex_values(row["history"][-1]["multipliers"]), "failed expanded final history differs", rtol=1e-12, atol=1e-18)
    tau, delta, radius = float(sum(saved[:2]).real), float(np.prod(saved[:2]).real), float(max(abs(saved)))
    for key, value in (("tau", tau), ("delta", delta), ("radius", radius)):
        close(row[key], value, "failed expanded summary differs: "+key, atol=1e-18)
    return {"phase": row["phase"], "tau": tau, "delta": delta, "radius": radius, "pair": saved[:2],
            "maximum_relative_residual": float(max(residual)), "maximum_projected_relative_residual": float(max(projected)),
            "ritz_spectrum_absolute_replay_error": error}


def null_arms(hessian):
    H = np.asarray(hessian, float)
    values, Q = np.linalg.eigh(H)
    require(values[0] < 0 < values[1], "trace frame is not a saddle")
    for i in range(2):
        if Q[np.argmax(abs(Q[:, i])), i] < 0:
            Q[:, i] *= -1
    arms = []
    for family, sign in enumerate((1., -1.)):
        tangent = Q @ np.array([math.sqrt(values[1]), sign*math.sqrt(-values[0])])
        tangent /= np.linalg.norm(tangent)
        for ray in (1., -1.):
            t = ray*tangent
            arms.append({"arm_id": f"line-{family}-ray-{int(ray):+d}", "tangent": t.tolist(),
                         "normal": [-float(t[1]), float(t[0])]})
    return arms


def audit_refinements(result, raw, seed, derivatives, cfg):
    jets = [recompute_jet(jet, raw) for jet in result.get("local_jets", [])]
    refinements = result.get("refinement", [])
    require(len(refinements) <= cfg["maximum_refinements"] and len(jets) <= 2+len(refinements), "jet/refinement count differs")
    for jet, step in zip(jets[:2], cfg["jet_steps"]):
        require(jet["center"] == seed["center"] and jet["beta"] == seed["parameter"] and jet["step"] == step,
                "initial matched stencils differ from bound candidate")
    for i, row in enumerate(refinements):
        require(len(jets) >= i+2, "refinement lacks preceding jet")
        old = jets[i+1]
        H, g = np.asarray(old["fields"]["tau"]["hessian"]), np.asarray(old["fields"]["tau"]["gradient"])
        dp = -np.linalg.solve(H, g)
        height = float(old["fields"]["tau"]["value"]+.5*g@dp)
        db = -height/derivatives["height_secant"]
        center = np.asarray(old["center"])+dp+np.asarray(derivatives["center_secant"])*db
        require(abs(db) <= cfg["maximum_beta_correction"] and np.linalg.norm(dp) < .001, "refinement left local trust region")
        expected = {"old_center": old["center"], "old_beta": old["beta"], "stationary_height_predictor": height,
                    "step_center": dp.tolist(), "step_beta": db, "new_center": center.tolist(), "new_beta": old["beta"]+db}
        for key, value in expected.items():
            close(row[key], value, "refinement predictor differs: "+key, rtol=1e-11, atol=1e-15)
        checked_point(raw, row["point_index"], center, expected["new_beta"])
        if len(jets) > i+2:
            next_jet = jets[i+2]
            require(next_jet["center"] == row["new_center"] and next_jet["beta"] == row["new_beta"]
                    and next_jet["step"] == cfg["refinement_step"], "fresh correction stencil is misplaced")
    center = result.get("center_point")
    if center is not None:
        require(len(jets) == 2+len(refinements), "center lacks final fresh verification stencil")
        jet = jets[-1]
        require(center == checked_point(raw, center["index"], jet["center"], jet["beta"]), "embedded center differs")
        require(abs(jet["fields"]["tau"]["value"]) <= cfg["height_tolerance"]
                and jet["correction_norm"] <= cfg["center_correction_tolerance"]
                and [r["phase"] for r in center["phases"]] == [0, 6, 12]
                and max(abs(r["tau"]) for r in center["phases"]) <= cfg["height_tolerance"], "fresh center admission failed")
    comparisons = []
    if len(jets) >= 2:
        for name in ("tau", "delta"):
            coarse, fine = jets[0]["fields"][name], jets[1]["fields"][name]
            comparisons.append({"field": name, "gradient_difference": (np.asarray(fine["gradient"])-coarse["gradient"]).tolist(),
                "hessian_difference": (np.asarray(fine["hessian"])-coarse["hessian"]).tolist(),
                "scope": "Two-scale comparison only; not a convergence-order or error bound."})
    return {"local_jets": jets, "refinement": refinements, "matched_scale_comparisons": comparisons,
            "center_point_index": None if center is None else center["index"]}


def audit_sections(run, result, raw, jets, cfg):
    sections = result.get("sections", [])
    if not sections and "arms" not in result:
        return []
    require(result.get("center_qualified") is True and jets and result.get("center_point") is not None,
            "section geometry lacks admitted center")
    center_point, jet = result["center_point"], jets[-1]
    center, beta = np.asarray(center_point["coordinate"]), center_point["beta"]
    arms = null_arms(jet["fields"]["tau"]["hessian"])
    require(len(result["arms"]) == 4, "four trace-frame predictor arms required")
    for saved, expected in zip(result["arms"], arms, strict=True):
        require(saved["arm_id"] == expected["arm_id"], "arm identity differs")
        for key in ("tangent", "normal"):
            close(saved[key], expected[key], "trace predictor arm differs", rtol=0, atol=1e-13)
    order = [(l, a) for l in cfg["longitudinal_steps"] for a in arms]
    require(len(sections) <= len(order), "unexpected extra sections")
    center_orbit = np.load(run/f"point-{center_point['index']:04d}"/"points.npy", allow_pickle=False)
    summaries = []
    for section, (longitudinal, arm) in zip(sections, order):
        require(section["arm_id"] == arm["arm_id"] and section["longitudinal"] == longitudinal
                and section["center"] == center.tolist() and section["beta"] == beta
                and section["mesh"] == cfg["section_mesh"], "section order/center/mesh differs")
        normal, tangent = np.asarray(section["normal"]), np.asarray(section["tangent"])
        close(normal, arm["normal"], "section normal differs", rtol=0, atol=1e-13)
        close(tangent, arm["tangent"], "section tangent differs", rtol=0, atol=1e-13)
        origin = center+longitudinal*tangent
        close(section["origin"], origin, "section origin differs", rtol=0, atol=1e-13)
        measured = {}
        for row in section["predictor_checks"]:
            s = row["s"]
            require(cfg["section_mesh"][0] <= s <= cfg["section_mesh"][-1] and s not in measured,
                    "predictor sample outside support or repeated")
            point = checked_point(raw, row["point_index"], origin+s*normal, beta)
            q = np.asarray(point["coordinate"])-center
            predicted = {}
            for name in ("tau", "delta"):
                field = jet["fields"][name]
                predicted[name] = float(field["value"]+np.asarray(field["gradient"])@q+.5*q@np.asarray(field["hessian"])@q)
            if predicted["delta"] < 0:
                predicted["radius"] = float((math.sqrt(predicted["tau"]**2-4*predicted["delta"])+abs(predicted["tau"]))/2)
            require(set(row["predicted"]) == set(predicted), "predictor fields differ")
            for name, value in predicted.items():
                close(row["predicted"][name], value, "local predictor value differs", rtol=1e-11, atol=1e-18)
            require(row["measured"] == {k: point[k] for k in ("tau", "delta", "radius")}, "predictor measured values differ")
            orbit = np.load(run/f"point-{point['index']:04d}"/"points.npy", allow_pickle=False)
            distances = [float(np.max(np.linalg.norm(np.roll(orbit, shift, axis=0)-center_orbit, axis=1)
                                     /np.linalg.norm(center_orbit, axis=1))) for shift in range(18)]
            alignment = row["orbit_alignment"]
            require(alignment["reference_point_index"] == center_point["index"] and alignment["best_right_roll"] == int(np.argmin(distances)),
                    "section orbit alignment identity differs")
            close(alignment["all_alignment_maximum_relative_changes"], distances, "section orbit alignment distances differ", rtol=1e-10, atol=1e-15)
            passed = alignment["best_right_roll"] == 0 and distances[0] < 1e-4
            measured[s] = {"s": s, "point_index": point["index"], "coordinate": point["coordinate"],
                "tau": point["tau"], "delta": point["delta"], "radius": point["radius"], "predicted": predicted,
                "prediction_error": {k: point[k]-v for k, v in predicted.items()},
                "prediction_relative_error": {k: abs(point[k]-v)/max(abs(point[k]), 1e-300) for k, v in predicted.items()},
                "orbit_alignment": alignment, "orbit_alignment_passed": passed}
        # Replay the callback event stream even if termination precluded search return.
        attempts, active = [], None
        for event in section["events"]:
            kind = event["event"]
            if kind == "evaluation_started":
                require(active is None and event["attempt"] == len(attempts)+1, "section callback event sequence differs")
                require(event["status"] == "started" and event["radius"] is None and event["index"] is None,
                        "started callback invents objective")
                active = {k: event[k] for k in ("s", "status", "radius", "index")}
                attempts.append(active)
            elif kind in ("evaluation_complete", "evaluation_failed"):
                require(active is not None and event["s"] == active["s"], "callback completion lacks matching start")
                active.update({k: v for k, v in event.items() if k != "event"})
                if kind == "evaluation_complete":
                    observed = measured.get(event["s"])
                    require(observed is not None and observed["point_index"] == event["index"]
                            and observed["radius"] == event["radius"] and observed["orbit_alignment_passed"],
                            "completed callback lacks admitted actual objective")
                active = None
            else:
                require(kind in ("mesh_finished", "candidate_finished", "refinement_step", "section_finished"), "unknown section event")
        require(len(attempts) <= cfg["section_maximum_evaluations"], "callback events exceed section budget")
        search = section["search"]
        if search is not None:
            require(search["evaluations"] == attempts, "search callbacks differ from recorded event stream")
            expected = {"mesh": cfg["section_mesh"], "coordinate_tolerance": cfg["coordinate_tolerance"],
                "max_evaluations": cfg["section_maximum_evaluations"], "max_refinement_steps": 128,
                "radius_floor": result["radius_floor"], "signal_multiple": cfg["signal_multiple"],
                "validation_offsets": cfg["validation_offsets"]}
            audited = audit_section_search(search, raw, origin, normal, beta, expected_settings=expected)
            require(section["complete"] is search["complete"], "section completion differs from search")
        else:
            require(section["complete"] is False, "unreturned search promoted to complete")
            audited = {"status": "interrupted before search returned", "complete": False, "minima": [], "evaluation_count": len(attempts)}
        summaries.append({**{k: section[k] for k in ("arm_id", "tangent", "normal", "longitudinal", "center", "beta", "origin")},
            "search": audited, "predictor_checks": list(measured.values()), "callback_attempts": attempts,
            "maximum_absolute_prediction_error": {k: max((abs(p["prediction_error"].get(k, 0)) for p in measured.values()), default=None)
                                                   for k in ("tau", "delta", "radius")}})
    return summaries


def run_audit(run, *, allow_partial=False, controller_manifest=None, manifest_sha=None,
              controller_status=None, status_sha=None):
    run = Path(run).resolve()
    require(run.is_relative_to((DATA/"explore").resolve()), "raw pilot outside external explore store")
    protocol = read(run/"protocol.json")
    anchor = bind_execution(protocol, expected_id="BSC5-contraction-minima", required_sources=EXECUTION_SOURCES)
    plan, cfg = protocol["plan"], protocol["plan"]["contraction_minima"]
    require(type(plan["maximum_evaluations"]) is int and 1 <= plan["maximum_evaluations"] <= 300
            and 90 < protocol["args"]["seconds"] <= 10800 and protocol["args"]["device"] in ("cuda", "cpu"), "execution budget or device differs")
    require(Path(protocol["args"]["out"]).resolve() == run
            and protocol["args"].get("source_commit") in (None, protocol["commit"]), "execution output/source arguments differ")
    terminal_args = (controller_manifest, manifest_sha, controller_status, status_sha)
    lifecycle = terminal_evidence(run, protocol, *terminal_args)
    result_path = run/"result.json" if (run/"result.json").exists() else run/"probe-progress.json"
    result = read(result_path) if result_path.exists() else {"complete": False}
    require(type(result.get("complete")) is bool, "scientific completion flag missing")
    points = audit_saved_points(run, plan); raw = points.pop("raw")
    require(points["runtime"] is not None or not result.get("complete"), "complete output lacks runtime")
    baseline = audit_baseline(run, raw, anchor) if (run/"baseline.json").exists() else {"status": "untested"}
    if result.get("baseline") is not None:
        require(result["baseline"] == read(run/"baseline.json"), "embedded baseline differs")
    bundle, seed, derivatives = bind_prior(protocol)
    replay = audit_seed(run, result.get("seed_replay"), raw, bundle, seed)
    require(replay is None or baseline.get("passed") is True, "seed replay lacks dense baseline admission")
    require(not result.get("local_jets") or replay is not None and "predictor_derivatives" in protocol,
            "local measurements lack prior seed/derivative admission")
    local = audit_refinements(result, raw, seed, derivatives, cfg)
    expanded = audit_expanded(run, result.get("expanded_spectrum"), raw, cfg["expanded_columns"])
    if result.get("center_qualified"):
        require(local["center_point_index"] is not None and expanded["complete"]
                and expanded["point_index"] == local["center_point_index"], "qualified center lacks fresh/expanded admission")
        require(max(abs(row["tau"]) for row in expanded["rows"]) <= cfg["height_tolerance"],
                "qualified center fails expanded trace-height admission")
        floor = max(cfg["radius_floor_minimum"], expanded["variations"]["radius"])
        require(result["radius_floor"] == floor <= 1e-11, "empirical objective floor differs or exceeds admission")
    else:
        require(not result.get("sections"), "sections began before center admission")
    sections = audit_sections(run, result, raw, local["local_jets"], cfg)
    failure = read(run/"failure.json") if (run/"failure.json").exists() else None
    expected_count = 4*len(cfg["longitudinal_steps"])
    if result.get("execution_finished"):
        require(len(sections) == expected_count and all(s["search"]["status"] != "evaluation_failed" for s in sections)
                and result["evaluations"] == points["attempted_points"] and failure is None,
                "finished execution lacks all planned sections or contains failure")
        require(result["complete"] is all(s["search"]["complete"] for s in sections), "scientific completion differs from individual searches")
    if result["complete"]:
        require(result.get("execution_finished") is True and result.get("center_qualified") is True
                and not points["unqualified_or_incomplete_points"] and failure is None,
                "complete result has missing or failed evidence")
    complete = bool(result["complete"] and lifecycle["python_complete"] and lifecycle["python_finalization"])
    if lifecycle["controller"]:
        complete = complete and lifecycle["controller"]["execution_complete"] and not lifecycle["controller"]["timed_out"]
    require(complete or allow_partial, "partial/unresolved pilot requires explicit --allow-partial")
    require(terminal_evidence(run, protocol, *terminal_args) == lifecycle, "terminal evidence changed during audit")
    stages = []
    for stage, done in (("dense baseline", baseline.get("passed") is True), ("candidate seed replay", replay is not None),
                        ("local center admission", result.get("center_qualified") is True),
                        ("expanded cyclic spectra", expanded["complete"]), ("all section searches", len(sections) == expected_count)):
        if not done:
            stages.append(stage)
    return {"id": "BSC5-contraction-minima-offline-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution_source_commit": protocol["commit"], "execution_source_sha256": protocol["source_sha256"],
        "plan_path": protocol["args"]["plan"], "plan_sha256": protocol["plan_sha256"],
        "scientific_metadata": {k: plan[k] for k in binding.METADATA}, "raw_directory": str(run),
        "complete": complete, "audit_finished": True, "execution_finished": result.get("execution_finished", False),
        "status": "all saved-data gates passed" if complete else "partial or unresolved attempt; available measurements audited",
        "lifecycle": lifecycle, **points, "baseline": baseline, "prior_audit": plan["prior_audit"],
        "seed_replay": replay, "predictor_derivatives": derivatives, **local, "expanded_spectrum": expanded,
        "center_qualified": result.get("center_qualified", False), "radius_floor": result.get("radius_floor"),
        "sections": sections, "untested_or_incomplete_stages": stages, "terminal_failure": failure,
        "manifest": [artifact(p, run) for p in sorted(run.rglob("*")) if p.is_file()],
        "limits": ["Saved-array replay, not a fresh model run or interval certificate.",
                   "Radius is the largest computed Ritz modulus. Dense baseline and independent expanded subspaces support but do not certify full-spectrum dominance everywhere.",
                   "Empirical cyclic/subspace radius spread is a resolution diagnostic, not a validated error bound.",
                   "Trace and determinant jets predict a coordinate frame; objectives and finite-offset minimum gates use measured radii without a trace-zero constraint.",
                   "Qualified points are resolved finite-section minima at two declared offsets. Unseen mesh minima, intrinsic ridges, connectivity and exact superstability are not established.",
                   "This pilot traces inward at one refined beta. It does not continue minimum branches in beta or rule out a connection at nearby beta.",
                   "A missing stage is untested. A failed admission or unresolved section is retained and is not evidence against every possible minimum."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--controller-manifest", type=Path)
    parser.add_argument("--controller-manifest-sha256")
    parser.add_argument("--controller-status", type=Path)
    parser.add_argument("--controller-status-sha256")
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists() and not out.is_relative_to(run)
            and not run.is_relative_to(out), "exclusive external audit output disjoint from raw required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {source: digest(ROOT/source) for source in SOURCES}
    require(all(hashlib.sha256(point_audit.git_bytes(commit, path)).hexdigest() == sha for path, sha in hashes.items()),
            "commit auditor sources before execution")
    out.mkdir(parents=True)
    try:
        receipt = run_audit(run, allow_partial=args.allow_partial, controller_manifest=args.controller_manifest,
            manifest_sha=args.controller_manifest_sha256, controller_status=args.controller_status, status_sha=args.controller_status_sha256)
        receipt.update(analysis_commit=commit, analysis_source_sha256=hashes, threads=1)
        path = out/"receipt.json"
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"out": str(path), "sha256": digest(path), "complete": receipt["complete"],
                          "qualified_points": len(receipt["qualified_points"])}))
    except BaseException as error:
        (out/"failed.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
