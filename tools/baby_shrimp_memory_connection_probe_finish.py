"""Independent saved-data audit of BSC4, including explicitly partial attempts.

No model, provider calls or figures. Every saved qualified orbit is audited;
missing stages are untested, not scientific failures. Raw files are read only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.optimize import brentq
import baby_shrimp_memory_finish as point_audit
import baby_shrimp_memory_orbit_alignment as binding

ROOT, DATA = binding.ROOT, binding.DATA
SOURCES = ["tools/baby_shrimp_memory_connection_probe_finish.py", "tools/baby_shrimp_memory_finish.py",
           "tools/baby_shrimp_memory_orbit_alignment.py", "tools/baby_shrimp_memory_determinant_gradient.py"]
EXECUTION_SOURCES = {
    "tools/baby_shrimp_memory_connection_probe.py", "tools/baby_shrimp_saddle_memory.py",
    "tools/baby_shrimp_memory.py", "tools/baby_shrimp_crossing.py",
    "tools/baby_shrimp_memory_orbit_alignment.py", "tools/baby_shrimp_memory_finish.py",
    "tools/baby_shrimp_memory_resume.py", "tools/baby_shrimp_orbits.py",
    "tools/baby_shrimp_memory_determinant_gradient.py", "tools/baby_shrimp_geometry.py",
    "tools/doubly_superstable.py", "tools/explore_llm_soft_loop.py",
    "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py",
    "tools/explore_latent_loop.py", "tools/llm_feigenbaum.py"}
require, close = point_audit.require, point_audit.close


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(path):
    return binding.digest(path)


def artifact(path, base=None):
    path = Path(path).resolve()
    base = DATA if base is None else base
    return {"path": str(path.relative_to(base.resolve())), "sha256": digest(path), "bytes": path.stat().st_size}


def eigenframe(H):
    H = np.asarray(H, float)
    require(H.shape == (2, 2) and np.isfinite(H).all() and np.allclose(H, H.T, rtol=0, atol=1e-13),
            "invalid symmetric saddle Hessian")
    values, vectors = np.linalg.eigh(H)
    require(values[0] < 0 < values[1] and np.linalg.cond(H) < 1e8, "nondegenerate saddle Hessian required")
    for column in range(2):
        if vectors[np.argmax(abs(vectors[:, column])), column] < 0:
            vectors[:, column] *= -1
    return values, vectors


def recompute_measurements(result, protocol, point_rows):
    """Recompute embedded measurements; verified_seeds are supplied by run()."""
    plan, cfg = protocol["plan"], protocol["plan"]["connection_probe"]
    require(isinstance(result.get("complete"), bool), "result completion flag must be boolean")
    seeds = protocol["verified_seeds"]
    require(len(seeds) == 3 and [s["parameter"] for s in seeds[:2]] == [cfg["before_beta"], cfg["after_beta"]],
            "bound seed sequence differs")
    used = set()
    def point(row, center=None, beta=None, cyclic=False):
        require(isinstance(row, dict) and isinstance(row.get("index"), int) and not isinstance(row["index"], bool)
                and row.get("index") in point_rows
                and row == point_rows[row["index"]] and row.get("qualified") is True,
                "embedded point differs from independently qualified raw record")
        require(np.isfinite([row["tau"], row["delta"], row["radius"]]).all()
                and row["delta"] < 0 and 0 < row["radius"] < 1, "invalid attracting opposite-sign pair")
        if center is not None:
            close(row["coordinate"], center, "point coordinate differs", rtol=0, atol=1e-13)
        if beta is not None:
            require(row["beta"] == beta, "point beta differs")
        if cyclic:
            require([p["phase"] for p in row["phases"]] == [0, 6, 12], "three cyclic phases required")
        used.add(row["index"])
        return row
    def at(center, beta):
        # The evaluator caches the first single-phase point until cyclic replay;
        # the lowest raw index therefore reconstructs a root search's values.
        matches = [p for p in point_rows.values() if p["beta"] == beta
                   and np.array_equal(np.asarray(p["coordinate"]), np.asarray(center))]
        require(matches, "root replay lacks an evaluated coordinate")
        return point(min(matches, key=lambda p: p["index"]), center, beta)
    summary = {"candidate_tilt": {"status": "untested"}, "axes": [], "matched_stencils": [], "roots": []}
    plot = {"candidate_tilt": None, "axes": [], "roots": [],
            "coordinate_metric": "Euclidean in the plan's scaled coordinates"}
    tilt = result.get("candidate_tilt")
    if tilt is not None:
        require(isinstance(tilt.get("complete"), bool), "tilt completion flag must be boolean")
        seed = seeds[-1]; center = np.asarray(seed["center"]); beta = seed["parameter"]
        require(tilt["beta"] == beta and tilt["center"] == seed["center"], "candidate tilt seed differs")
        direction = np.asarray(protocol["candidate_tilt_direction"]["direction"])
        close(tilt["direction"], direction, "candidate tilt direction differs", rtol=0, atol=1e-13)
        require(abs(np.linalg.norm(direction)-1) < 1e-12, "candidate tilt direction is not unit length")
        central = point(tilt["center_point"], center, beta)
        offsets = [p["offset"] for p in tilt["samples"]]
        require(offsets == cfg["candidate_tilt_offsets"][:len(offsets)], "candidate tilt offset order differs")
        samples = {s["offset"]: point(s["point"], center+s["offset"]*direction, beta) for s in tilt["samples"]}
        slopes = [{"step": step, **{key: float((samples[step][key]-samples[-step][key])/(2*step))
                                   for key in ("tau", "delta", "radius")}}
                  for step in sorted(x for x in samples if x > 0) if -step in samples]
        recorded = tilt["symmetric_slopes"]
        require(recorded == slopes[:len(recorded)], "candidate symmetric slopes differ")
        require(not tilt["complete"] or offsets == cfg["candidate_tilt_offsets"] and recorded == slopes,
                "completed tilt lacks both signed pairs")
        summary["candidate_tilt"] = {"status": "complete" if tilt["complete"] else "partial; remaining offsets untested",
                                     "center_point_index": central["index"], "symmetric_slopes": slopes}
        plot["candidate_tilt"] = {"beta": beta, "center": center.tolist(), "direction": direction.tolist(),
            "center_point_index": central["index"], "samples": [
                {"offset": offset, "point_index": p["index"], "radius_difference": p["radius"]-central["radius"]}
                for offset, p in samples.items()]}
    axes = result.get("axes", [])
    require(len(axes) <= 2, "too many axis cases")
    for case, seed in zip(axes, seeds[:2]):
        beta, center = seed["parameter"], np.asarray(seed["center"])
        require(case["beta"] == beta and case["center"] == seed["center"], "axis seed differs")
        values, basis = eigenframe(seed["hessian"])
        close(case["hessian_eigenvalues"], values, "axis Hessian eigenvalues differ")
        require(len(case["axes"]) == 2, "axis case missing a direction")
        counts = []
        for column, axis in enumerate(case["axes"]):
            require(axis["axis"] == ("negative curvature", "positive curvature")[column], "axis curvature label differs")
            close(axis["direction"], basis[:, column], "axis basis differs", rtol=0, atol=1e-13)
            require([r["offset"] for r in axis["samples"]] == sorted([0., *cfg["axis_offsets"]]), "axis samples incomplete")
            for sample in axis["samples"]:
                point(sample["point"], center+sample["offset"]*basis[:, column], beta)
            brackets = [{"offsets": [a["offset"], b["offset"]],
                         "point_indices": [a["point"]["index"], b["point"]["index"]],
                         "heights": [a["point"]["tau"], b["point"]["tau"]]}
                        for a, b in zip(axis["samples"], axis["samples"][1:]) if a["point"]["tau"]*b["point"]["tau"] < 0]
            require(axis["strict_sign_brackets"] == brackets, "axis sign brackets differ")
            counts.append(len(brackets))
        summary["axes"].append({"beta": beta, "negative_axis_brackets": counts[0], "positive_axis_brackets": counts[1]})
        plot["axes"].append({"beta": beta, "center": center.tolist(), "hessian_eigenvalues": values.tolist(),
            "axes": [{"label": axis["axis"], "direction": axis["direction"], "samples": [
                {"offset": s["offset"], "point_index": s["point"]["index"]} for s in axis["samples"]]}
                for axis in case["axes"]]})
    stencils = result.get("matched_stencils", [])
    require(len(stencils) <= len(cfg["stencil_steps"]), "too many stencil scales")
    center = np.asarray(seeds[1]["center"]); beta = cfg["after_beta"]
    recomputed = []
    for row, step in zip(stencils, cfg["stencil_steps"]):
        require(row["step"] == step and row["beta"] == beta and row["center"] == center.tolist(), "matched stencil center/scale differs")
        indices = np.asarray(row["point_indices"])
        require(indices.shape == (3, 3) and np.issubdtype(indices.dtype, np.integer)
                and np.all(indices >= 0), "complete integer-indexed nine-point stencil required")
        f = np.empty((3, 3))
        for iy, dy in enumerate((-1, 0, 1)):
            for ix, dx in enumerate((-1, 0, 1)):
                index = int(indices[iy, ix]); require(index in point_rows, "stencil point missing")
                p = point(point_rows[index], center+step*np.array([dx, dy]), beta)
                f[iy, ix] = p["tau"]
        g = np.array([(f[1, 2]-f[1, 0])/(2*step), (f[2, 1]-f[0, 1])/(2*step)])
        cross = (f[2, 2]-f[0, 2]-f[2, 0]+f[0, 0])/(4*step*step)
        H = np.array([[(f[1, 2]-2*f[1, 1]+f[1, 0])/step**2, cross],
                      [cross, (f[2, 1]-2*f[1, 1]+f[0, 1])/step**2]])
        values, _ = eigenframe(H); correction = -np.linalg.solve(H, g)
        for key, value in (("tau", f[1, 1]), ("gradient", g), ("hessian", H),
                           ("hessian_eigenvalues", values), ("correction", correction),
                           ("correction_norm", np.linalg.norm(correction))):
            close(row[key], value, "matched stencil differs: "+key, rtol=1e-10, atol=1e-14)
        require(np.linalg.norm(correction) <= cfg["maximum_correction_norm"], "same-center saddle correction exceeds gate")
        recomputed.append((g, H))
        summary["matched_stencils"].append({"step": step, "tau": float(f[1, 1]), "gradient": g.tolist(),
                                             "hessian": H.tolist(), "correction_norm": float(np.linalg.norm(correction))})
    roots = result.get("roots", [])
    require(len(roots) <= 2*len(cfg["slice_offsets"]), "too many roots")
    if roots:
        require(len(recomputed) == len(cfg["stencil_steps"]), "roots lack both checked same-center stencils")
        g, H = recomputed[-1]; _, basis = eigenframe(H); basis = basis[:, [1, 0]]
        require(at(center, beta)["tau"] < 0, "higher-beta center is not negative")
        expected = [(offset, side) for offset in cfg["slice_offsets"] for side in ("negative", "positive")]
        for root, (offset, side) in zip(roots, expected):
            require(isinstance(root.get("complete"), bool), "root completion flag must be boolean")
            require(root["beta"] == beta and root["center"] == center.tolist()
                    and root["transverse"] == offset and root["side"] == side, "root order or slice differs")
            close(root["slice_basis"], basis, "root slice basis differs", rtol=0, atol=1e-13)
            bounds = [-cfg["root_bracket"], 0.] if side == "negative" else [0., cfg["root_bracket"]]
            require(root["bracket"] == bounds, "root bracket differs")
            def function(along):
                return at(center+basis@np.array([along, offset]), beta)["tau"]
            ends = [function(x) for x in bounds]
            require(ends[0]*ends[1] < 0 and root["bracket_values"] == ends, "root lacks the declared strict sign bracket")
            along, info = brentq(function, *bounds, xtol=cfg["root_coordinate_tolerance"], rtol=1e-12,
                                 maxiter=cfg["root_max_iterations"], full_output=True, disp=True)
            require(info.converged and root["iterations"] == info.iterations and root["function_calls"] == info.function_calls,
                    "root optimizer replay differs")
            close(root["along"], along, "root coordinate replay differs", rtol=0, atol=1e-13)
            location = center+basis@np.array([along, offset])
            primary = point(root["point"], location, beta)
            require(abs(primary["tau"]) < cfg["root_trace_tolerance"], "primary root trace residual exceeds gate")
            item = {"transverse": offset, "side": side, "along": float(along), "primary_point_index": primary["index"],
                    "status": "partial; remaining cyclic/offset checks untested", "radius_offsets": []}
            q = root.get("cyclic_point")
            offsets = root["normal_offsets"]
            if q is None:
                require(not root["complete"] and not offsets, "root invents offsets without cyclic endpoint")
                require("normal_direction" not in root and "sampled_radius_comparison" not in root,
                        "root invents normal/comparison before cyclic replay")
            else:
                q = point(q, location, beta, cyclic=True)
                require(abs(q["tau"]) < cfg["root_trace_tolerance"], "cyclic root trace residual exceeds gate")
                item.update(cyclic_point_index=q["index"], tau=q["tau"], delta=q["delta"], radius=q["radius"])
                gradient = g+H@(location-center); normal = gradient/np.linalg.norm(gradient)
                if "normal_direction" in root:
                    close(root["normal_direction"], normal, "root normal direction differs", rtol=0, atol=1e-13)
                require([s["offset"] for s in offsets] == cfg["normal_offsets"][:len(offsets)], "normal offsets reordered or invented")
                require(not offsets or "normal_direction" in root, "normal offsets lack a declared direction")
                for sample in offsets:
                    p = point(sample["point"], location+sample["offset"]*normal, beta)
                    difference = p["radius"]-q["radius"]
                    close(sample["radius_difference_from_root"], difference, "radius offset difference differs", rtol=0, atol=1e-18)
                    item["radius_offsets"].append({"offset": sample["offset"], "point_index": p["index"], "radius_difference": difference})
                if root["complete"] or "sampled_radius_comparison" in root:
                    require(len(offsets) == 2, "radius comparison lacks both offsets")
                    comparison = ("root below both sampled offsets" if min(s["radius_difference"] for s in item["radius_offsets"]) > 0
                                  else "at least one sampled offset has lower or equal radius")
                    require(root["sampled_radius_comparison"] == comparison, "sampled radius comparison differs")
                    item["sampled_radius_comparison"] = comparison
                if root["complete"]:
                    item["status"] = "complete"
            summary["roots"].append(item)
            plot["roots"].append({"beta": beta, "center": center.tolist(), "slice_basis": basis.tolist(),
                "along": float(along), "transverse": offset, "side": side,
                "primary_point_index": primary["index"], "cyclic_point_index": item.get("cyclic_point_index"),
                "status": item["status"]})
    complete = (tilt is not None and tilt["complete"] and len(axes) == 2 and len(stencils) == 2
                and len(roots) == 2*len(cfg["slice_offsets"]) and all(r["complete"] for r in roots))
    require(not result.get("complete") or complete, "completed result omits required stages")
    summary.update(all_measurement_stages_complete=complete, used_qualified_point_indices=sorted(used),
                   untested_stages=[label for label, ok in (("candidate tilt", tilt is not None and tilt["complete"]),
                       ("both axis cases", len(axes) == 2), ("both same-center stencils", len(stencils) == 2),
                       ("all cyclic roots and finite radius offsets", len(roots) == 6 and all(r["complete"] for r in roots))) if not ok])
    summary["plot_data"] = plot
    return summary


def verify_files(run, entries, *, omit_manifest=False):
    """Exact regular-file snapshot, rejecting links and path escapes."""
    require(isinstance(entries, list) and entries, "empty terminal manifest")
    require(not any(p.is_symlink() for p in run.rglob("*")), "raw snapshot contains a symlink")
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() not in seen,
                "invalid or duplicate manifest path")
        path = run/relative
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents) if p != run.parent)
                and path.resolve().is_relative_to(run), "manifest path missing, linked or outside raw attempt")
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "terminal file hash/size differs")
        seen.add(relative.as_posix())
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    require(actual - ({"manifest.json"} if omit_manifest else set()) == seen,
            "unmanifested or missing terminal files")


def terminal_evidence(run, protocol, controller_manifest=None, manifest_sha=None,
                      controller_status=None, status_sha=None):
    supplied = (controller_manifest, manifest_sha, controller_status, status_sha)
    require(not any(supplied) or all(supplied), "controller manifest and status require explicit paired hashes")
    controller = None
    if all(supplied):
        paths = (Path(controller_manifest).resolve(), Path(controller_status).resolve())
        require(all(p.is_relative_to(DATA.resolve()) and p.is_file() for p in paths), "controller packet must be in external data")
        require(digest(paths[0]) == manifest_sha and digest(paths[1]) == status_sha, "controller packet hash differs")
        entries, status = read(paths[0]), read(paths[1])
        require(status["worker_reaped"] is True and status["source_commit"] == protocol["commit"]
                and Path(status["output"]).resolve() == run and isinstance(status["exit_code"], int)
                and not isinstance(status["exit_code"], bool) and isinstance(status["timed_out"], bool)
                and isinstance(status.get("complete"), bool) and "ended_utc" in status,
                "controller has not bound a reaped terminal worker")
        require(status["complete"] == (status["exit_code"] == 0 and (run/"result.json").is_file())
                and not (status["complete"] and status["timed_out"]), "controller completion is inconsistent")
        require(not status["complete"] or read(run/"result.json").get("complete") is True,
                "successful controller contains an incomplete result")
        require(status["hard_seconds"] == protocol["args"]["seconds"] <= 5400
                and status["termination_grace_seconds"] <= 10, "controller lifetime differs")
        command = status["command"]
        for flag, value in (("--source-commit", protocol["commit"]), ("--out", str(run)),
                            ("--plan", protocol["args"]["plan"]), ("--audit-sha256", protocol["args"]["audit_sha256"]),
                            ("--device", protocol["args"]["device"]), ("--seconds", str(protocol["args"]["seconds"])),
                            ("--audit", protocol["args"]["audit"])):
            require(command.count(flag) == 1 and command.index(flag)+1 < len(command)
                    and command[command.index(flag)+1] == value,
                    "controller command differs: "+flag)
        require("tools/baby_shrimp_memory_connection_probe.py" in command, "controller ran another worker")
        verify_files(run, entries)
        controller = {"manifest": artifact(paths[0]), "status_artifact": artifact(paths[1]),
                      "worker_pid": status["worker_pid"], "controller_pid": status["controller_pid"],
                      "worker_reaped": True, "exit_code": status["exit_code"], "timed_out": status["timed_out"],
                      "elapsed_seconds": status["elapsed_seconds"], "ended_utc": status["ended_utc"],
                      "controller_complete": status["complete"]}
    path = run/"manifest.json"
    python_manifest = read(path) if path.exists() else None
    if python_manifest is not None:
        require(isinstance(python_manifest.get("complete"), bool), "Python manifest status missing")
        verify_files(run, python_manifest["files"], omit_manifest=True)
        require((run/"result.json").is_file() or (run/"failure.json").is_file(), "Python finalization lacks terminal outcome")
        result = read(run/"result.json") if (run/"result.json").is_file() else {"complete": False}
        require(isinstance(result.get("complete"), bool) and python_manifest["complete"] == result["complete"],
                "Python manifest and result completion differ")
        require(not python_manifest["complete"] or not (run/"failure.json").exists(),
                "completed Python manifest contains a failure")
        require(controller is None or not controller["controller_complete"] or python_manifest["complete"],
                "successful controller lacks complete Python result")
    else:
        require(controller is not None, "missing Python finalization requires a hash-bound controller terminal snapshot")
    return {"python_finalization": python_manifest is not None,
            "python_manifest": artifact(path) if path.exists() else None,
            "python_complete": bool(python_manifest and python_manifest["complete"]), "controller": controller}


def bind_execution(protocol):
    require(isinstance(protocol.get("commit"), str) and re.fullmatch(r"[0-9a-f]{40}", protocol["commit"]),
            "exact 40-character execution commit required")
    require(EXECUTION_SOURCES <= set(protocol["source_sha256"]), "execution source manifest incomplete")
    for path, sha in protocol["source_sha256"].items():
        require(hashlib.sha256(point_audit.git_bytes(protocol["commit"], path)).hexdigest() == sha,
                "execution source differs from exact commit: "+path)
    blob = point_audit.git_bytes(protocol["commit"], protocol["args"]["plan"])
    require(hashlib.sha256(blob).hexdigest() == protocol["plan_sha256"] and json.loads(blob) == protocol["plan"],
            "committed plan hash or contents differ")
    plan = protocol["plan"]
    require(plan["id"] == "BSC4-direct-connection" and plan["period"] == 18
            and 1 <= plan["maximum_evaluations"] <= 120 and protocol["args"]["seconds"] <= 5400,
            "wrong probe family or budget")
    anchor = DATA/plan["anchor_directory"]
    for path, sha in plan["anchor_hashes"].items():
        require(digest(anchor/path) == sha, "anchor hash differs")
    return anchor


def bind_seeds(protocol):
    import baby_shrimp_memory_determinant_gradient as determinant
    plan = protocol["plan"]; prior = plan["prior_audit"]
    path = DATA/prior["path"]
    require(path.stat().st_size == prior["bytes"] and protocol["args"]["audit_sha256"] == prior["sha256"]
            and Path(protocol["args"]["audit"]).resolve() == path.resolve(), "prior audit argument differs from plan")
    bundle = binding.load_receipt(path, prior["sha256"])
    require(bundle["receipt"]["complete"] and bundle["receipt"]["height_root"]["complete"], "prior continuation or root is incomplete")
    require(all(plan[k] == bundle["scientific_metadata"][k] for k in binding.METADATA if k != "id"),
            "probe changed the audited map or coordinates")
    beta_root = bundle["receipt"]["height_root"]["result"]["saddle"]["parameter"]
    betas = [plan["connection_probe"]["before_beta"], plan["connection_probe"]["after_beta"], beta_root]
    seeds = []
    for beta in betas:
        matches = [c for c in bundle["receipt"]["accepted_centers"] if c["parameter"] == beta and c["kind"] == "continued saddle"]
        require(len(matches) == 1, "prior saddle not unique")
        seeds.append(matches[0])
    require(seeds[-1]["center"] == bundle["receipt"]["height_root"]["result"]["saddle"]["center"], "root seed location differs")
    record = plan["determinant_audit"]; det_path = DATA/record["path"]
    require(det_path.stat().st_size == record["bytes"] and digest(det_path) == record["sha256"], "determinant receipt binding differs")
    det = read(det_path)
    for source, sha in det["analysis_source_sha256"].items():
        require(hashlib.sha256(point_audit.git_bytes(det["analysis_commit"], source)).hexdigest() == sha,
                "determinant source differs")
    rows = [r for r in det["accepted_centers"] if r["completed_numerical_height_root_candidate"]
            and r["receipt_sha256"] == prior["sha256"] and r["parameter"] == beta_root and r["center"] == seeds[-1]["center"]]
    require(len(rows) == 1, "candidate determinant diagnostic not unique")
    prior_row = rows[0]; saddle_path = DATA/prior_row["saddle_artifact"]
    require(saddle_path.resolve().is_relative_to(bundle["run"]) and digest(saddle_path) == prior_row["saddle_sha256"],
            "candidate saddle artifact differs")
    source = read(saddle_path)
    replay = determinant.stencil_diagnostics(source["history"][-1], determinant.callback_mapping(bundle),
                                              np.asarray(plan["coordinate_scales"]))
    require(replay == prior_row["stencils"][-1], "candidate determinant stencil replay differs")
    gradient = np.asarray(replay["delta"]["gradient"])
    require(np.linalg.norm(gradient) > 0, "candidate tilt direction unresolved")
    tilt = protocol.get("candidate_tilt_direction")
    if tilt is not None:
        close(tilt["direction"], -gradient/np.linalg.norm(gradient), "bound tilt direction differs", rtol=0, atol=1e-13)
        require(tilt["determinant_gradient"] == gradient.tolist() and tilt["step"] == replay["step"]
                and tilt["audit"] == record and tilt["saddle_sha256"] == prior_row["saddle_sha256"], "tilt provenance differs")
    return bundle, seeds


def audit_replays(run, result, raw, bundle, seeds):
    rows = result.get("seed_replays", [])
    require(len(rows) <= 3, "too many seed replays")
    checked = []
    for saved, seed in zip(rows, seeds):
        directory = bundle["run"]/f"point-{seed['cyclic_point_index']:04d}"
        old = read(directory/"point.json")
        require(saved["fresh_point_index"] in raw, "fresh seed point missing")
        new = raw[saved["fresh_point_index"]]
        require(new["coordinate"] == seed["center"] and new["beta"] == seed["parameter"]
                and [p["phase"] for p in new["phases"]] == [0, 6, 12], "fresh seed coordinates/phases differ")
        a = np.load(directory/"points.npy", allow_pickle=False)
        b = np.load(run/f"point-{new['index']:04d}/points.npy", allow_pickle=False)
        error = float(np.max(np.linalg.norm(b-a, axis=1)/np.linalg.norm(a, axis=1)))
        require(error < 1e-8, "fresh seed orbit sheet differs")
        comparisons = []
        for x, y in zip(old["phases"], new["phases"], strict=True):
            av, bv = point_audit.complex_values(x["multipliers"])[:2], point_audit.complex_values(y["multipliers"])[:2]
            row = {"phase": x["phase"], "pair_relative_error": float(point_audit.pair_gap(av, bv)/max(abs(av))),
                   "trace_error_relative_to_radius": float(abs(x["tau"]-y["tau"])/max(abs(av))),
                   "determinant_relative_error": float(abs(x["delta"]-y["delta"])/abs(x["delta"]))}
            require(all(v < 1e-5 for k, v in row.items() if k != "phase"), "fresh seed spectral gate failed")
            comparisons.append(row)
        expected = {"source_point_index": old["index"], "source_directory": str(directory.relative_to(DATA)),
                    "fresh_point_index": new["index"], "beta": new["beta"], "maximum_relative_orbit_difference": error,
                    "cyclic_comparisons": comparisons, "passed": True}
        require(saved == expected, "saved seed replay differs from independent computation")
        checked.append(expected)
    later = result.get("candidate_tilt") is not None or any(result.get(k) for k in ("axes", "matched_stencils", "roots"))
    require(not later or len(checked) == 3, "measurements began without all seed replay gates")
    return checked


def run_audit(run, *, allow_partial=False, controller_manifest=None, manifest_sha=None,
              controller_status=None, status_sha=None):
    run = Path(run).resolve()
    require(run.is_relative_to((DATA/"explore").resolve()), "raw probe must be outside Git in the explore store")
    protocol = read(run/"protocol.json"); anchor = bind_execution(protocol)
    terminal_args = (controller_manifest, manifest_sha, controller_status, status_sha)
    lifecycle = terminal_evidence(run, protocol, *terminal_args)
    bundle, seeds = bind_seeds(protocol)
    result_path = run/"result.json"
    result = read(result_path) if result_path.exists() else read(run/"probe-progress.json") if (run/"probe-progress.json").exists() else {"complete": False}
    if not lifecycle["python_finalization"]:
        require(allow_partial, "controller-only snapshot requires --allow-partial")
    raw, summaries, pending = {}, [], []
    directories = sorted(p for p in run.glob("point-*") if p.is_dir())
    require([p.name for p in directories] == [f"point-{i:04d}" for i in range(len(directories))], "point directory sequence has gaps")
    require(len(directories) <= protocol["plan"]["maximum_evaluations"], "point budget exceeded")
    for directory in directories:
        path = directory/"point.json"
        if not path.is_file():
            pending.append({"directory": directory.name, "status": "incomplete", "reason": "point record was not saved"})
            continue
        try:
            row = read(path)
        except (ValueError, UnicodeDecodeError) as error:
            pending.append({"directory": directory.name, "status": "incomplete; unreadable point record",
                            "reason": str(error), "record": artifact(path, run)})
            continue
        require(directory.name == f"point-{row['index']:04d}", "raw point directory index differs")
        if row.get("qualified") is not True:
            pending.append({"directory": directory.name, "index": row["index"], "status": "not qualified; failed or interrupted",
                            "saved_fields": sorted(row), "beta": row["beta"], "coordinate": row["coordinate"]})
            continue
        point, summary = point_audit.audit_point(directory, protocol["plan"])
        require(summary["closure_recomputed_from_saved_endpoint"], "BSC4 qualified point lacks saved final return state")
        raw[point["index"]] = point; summaries.append(summary)
    runtime = read(run/"runtime.json") if (run/"runtime.json").exists() else None
    require(runtime is not None or not directories and not result.get("complete"),
            "numerical points or complete output require runtime metadata")
    if runtime is not None:
        require(0 < runtime["norm_sensitivity"] < 1e-9 and runtime["rotary_float64"] is True and runtime["threads"] == 8,
                "effective precision/runtime failed")
    baseline = point_audit.audit_baseline(run, raw, anchor) if (run/"baseline.json").exists() else {"status": "untested"}
    if "baseline" in result:
        require(result["baseline"] == read(run/"baseline.json"), "embedded baseline differs")
    replays = audit_replays(run, result, raw, bundle, seeds)
    if replays:
        require(baseline.get("passed") is True, "seed replays lack a passed dense baseline")
    enriched = {**protocol, "verified_seeds": seeds}
    measurements = recompute_measurements(result, enriched, raw)
    plot_data = measurements.pop("plot_data")
    failure = read(run/"failure.json") if (run/"failure.json").exists() else None
    complete = bool(result.get("complete") and lifecycle["python_complete"] and not pending and failure is None)
    if lifecycle["controller"] is not None:
        complete = complete and lifecycle["controller"]["exit_code"] == 0 and not lifecycle["controller"]["timed_out"]
    require(not result.get("complete") or not pending and failure is None, "completed result contains failed/incomplete points")
    require(not result.get("complete") or result["evaluations"] == len(directories), "completed evaluation count differs")
    require(complete or allow_partial, "incomplete probe requires explicit --allow-partial")
    missing = measurements["untested_stages"]
    interrupted = bool((failure or {}).get("type") in ("TimeoutError", "KeyboardInterrupt")
                       or lifecycle["controller"] and lifecycle["controller"]["timed_out"])
    incomplete_context = None if complete else {
        "next_uncompleted_measurement_stage": missing[0] if missing else None,
        "terminal_classification": "interrupted or budget-limited" if interrupted else
            "reported execution/precondition failure" if failure is not None else "incomplete finalization",
        "reported_failure": failure,
        "scope": "The stage name is inferred from saved stage order. Missing later stages are untested; a reported execution failure is retained separately and is not evidence against contour existence."}
    # Recheck after all reads; never turn a moving snapshot into terminal evidence.
    require(terminal_evidence(run, protocol, *terminal_args) == lifecycle, "terminal evidence changed during audit")
    manifest = [artifact(p, run) for p in sorted(run.rglob("*")) if p.is_file()]
    return {"id": "BSC4-direct-connection-offline-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "execution_source_commit": protocol["commit"], "plan_sha256": protocol["plan_sha256"],
            "execution_source_sha256": protocol["source_sha256"], "plan_path": protocol["args"]["plan"],
            "scientific_metadata": {key: protocol["plan"][key] for key in binding.METADATA},
            "raw_directory": str(run), "complete": bool(complete),
            "status": "all saved-data gates passed" if complete else "partial attempt; available measurements audited, missing stages untested",
            "lifecycle": lifecycle, "runtime": runtime, "baseline": baseline, "prior_audit": protocol["plan"]["prior_audit"],
            "seed_replays": replays, "qualified_points": summaries, "unqualified_or_incomplete_points": pending,
            "measurements": measurements, "plot_data": plot_data, "terminal_failure": failure, "incomplete_stage_context": incomplete_context,
            "manifest": manifest,
            "limits": ["Saved-data numerical replay, not a fresh model evaluation or an interval certificate.",
                       "Ritz isolation is conditional on the computed subspace; the original dense baseline does not certify the entire spectrum at every new point.",
                       "Measured trace zeros balance nonzero multipliers; no exact superstability or native coupling mechanism is certified.",
                       "Two finite offsets along an approximate scaled-coordinate normal do not prove a local contraction minimum or ridge connectivity.",
                       "A controller-only terminal snapshot never qualifies as a complete scientific run. Missing stages are untested, not evidence against the hypothesis."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--controller-manifest", type=Path)
    parser.add_argument("--controller-manifest-sha256")
    parser.add_argument("--controller-status", type=Path)
    parser.add_argument("--controller-status-sha256")
    args = parser.parse_args()
    out, run = args.out.resolve(), args.run.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists()
            and not out.is_relative_to(run) and not run.is_relative_to(out), "new external audit directory disjoint from raw required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {source: digest(ROOT/source) for source in SOURCES}
    require(all(hashlib.sha256(point_audit.git_bytes(commit, path)).hexdigest() == sha for path, sha in hashes.items()),
            "commit auditor sources before execution")
    out.mkdir(parents=True)
    try:
        receipt = run_audit(run, allow_partial=args.allow_partial, controller_manifest=args.controller_manifest,
                            manifest_sha=args.controller_manifest_sha256, controller_status=args.controller_status,
                            status_sha=args.controller_status_sha256)
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
