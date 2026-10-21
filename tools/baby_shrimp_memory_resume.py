"""Read-only, hash-bound predictor seeds from audited BSC3 attempts.

This imports no model and writes nothing. Old endpoints supply predictor/warm
history only; they are never inserted into a new attempt's scientific results.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import baby_shrimp_memory_finish as audit

ROOT, DATA = audit.ROOT, audit.DATA
MAX_ANCESTRY = 8
AMENDABLE_PLAN_FIELDS = {"maximum_evaluations", "maximum_abs_beta",
                         "maximum_scaled_coordinate", "resources", "continuation_amendment"}
REPLAY_GATES = {"maximum_relative_orbit_difference": 1e-8,
                "pair_relative_error": 1e-5, "trace_error_relative_to_radius": 1e-5,
                "determinant_relative_error": 1e-5}


def _sha_text(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _manifest(run, entries):
    """Require an exact regular-file snapshot, including incomplete evidence."""
    audit.require(isinstance(entries, list) and entries, "missing raw manifest")
    names = set()
    for entry in entries:
        relative = Path(entry["path"])
        audit.require(not relative.is_absolute() and ".." not in relative.parts
                      and str(relative) == entry["path"] and entry["path"] not in names,
                      "invalid or duplicate raw manifest path")
        names.add(entry["path"])
        path = run / relative
        audit.require(path.is_file() and not path.is_symlink()
                      and all(not parent.is_symlink() for parent in path.parents if parent != run.parent),
                      "raw manifest requires regular files without symlinks")
        audit.require(path.stat().st_size == entry["bytes"] and audit.digest(path) == entry["sha256"],
                      "raw manifest hash/size differs: " + entry["path"])
    actual = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file() or path.is_symlink()}
    audit.require(actual == names, "raw manifest does not cover the exact attempt")


def _compatible_plan(previous, current):
    audit.require(set(previous) == set(current), "resume plan fields differ")
    changes = {key: {"previous": previous[key], "current": current[key]}
               for key in previous if previous[key] != current[key]}
    audit.require(set(changes) <= AMENDABLE_PLAN_FIELDS,
                  "resume changes scientific plan fields")
    for key in ("maximum_evaluations", "maximum_abs_beta", "maximum_scaled_coordinate"):
        audit.require(np.isfinite(current[key]) and current[key] > 0, "invalid amended plan limit")
    audit.require(current["maximum_abs_beta"] <= .05, "resume exceeds memory map domain")
    return changes


def load_resume(receipt_path, receipt_sha256, plan, *, root=ROOT, data=DATA, seen=()):
    """Verify a pinned offline receipt and return provenance plus in-memory seeds.

    ``seen`` contains ancestor raw-directory/receipt paths. The two last accepted
    centers must be fully continued saddles with all three cyclic phases.
    Older missing return states remain an explicit limitation: these seeds are
    provisional until both endpoints pass fresh model replay. A partial parent attempt is allowed; its
    incomplete rows remain bound by the manifest and are never used as seeds.
    """
    receipt_path = Path(receipt_path).resolve()
    audit.require(len(seen) < 2 * MAX_ANCESTRY and str(receipt_path) not in seen,
                  "resume ancestry cycle or depth limit")
    audit.require(isinstance(receipt_sha256, str) and len(receipt_sha256) == 64
                  and all(c in "0123456789abcdef" for c in receipt_sha256)
                  and audit.digest(receipt_path) == receipt_sha256, "resume receipt hash differs")
    receipt = audit.read(receipt_path)
    audit.require(receipt["id"] == "BSC3-memory-offline-audit" and receipt["mode"] == "continue",
                  "resume requires a continuation audit receipt")
    analysis_commit = receipt["analysis_commit"]
    audit.require(len(analysis_commit) == 40 and all(c in "0123456789abcdef" for c in analysis_commit),
                  "resume analysis commit missing")
    audit.require(hashlib.sha256(audit.git_bytes(analysis_commit, "tools/baby_shrimp_memory_finish.py", root)).hexdigest()
                  == receipt["analysis_source_sha256"], "resume analysis source hash differs")
    run = Path(receipt["raw_directory"]).resolve()
    audit.require(str(run) not in seen and run != receipt_path and run not in receipt_path.parents,
                  "resume ancestry cycle or receipt inside raw attempt")
    ancestry = (*seen, str(receipt_path), str(run))
    _manifest(run, receipt["manifest"])
    protocol = audit.read(run / "protocol.json")
    audit.require(protocol["args"]["mode"] == "continue"
                  and receipt["execution_source_commit"] == protocol["commit"]
                  and receipt["plan_sha256"] == protocol["plan_sha256"], "resume execution binding differs")
    anchor = audit.bind_protocol(protocol, root, data)
    changes = _compatible_plan(protocol["plan"], plan)
    parent_bundle = None
    if "resume" in protocol:
        parent_bundle = verify_resume(protocol["resume"], protocol["plan"], root=root, data=data, seen=ancestry)
    runtime = audit.read(run / "runtime.json")
    audit.require(0 < runtime["norm_sensitivity"] < 1e-9 and runtime["rotary_float64"]
                  and runtime["threads"] == 8, "resume effective precision gate failed")
    raw = {}
    for directory in sorted(run.glob("point-*")):
        record = directory / "point.json"
        if not record.is_file() or not audit.read(record).get("qualified"):
            continue
        row, _ = audit.audit_point(directory, protocol["plan"])
        audit.require(row["index"] not in raw, "duplicate resume point")
        raw[row["index"]] = row
    if parent_bundle is not None:
        parent_replay = verify_replay(run, protocol["resume"], parent_bundle, raw)
        audit.require(parent_replay["passed"], "parent resume replay must pass before reusing target centers")
    audit.audit_baseline(run, raw, anchor)
    result_path = run / "result.json"
    result = audit.read(result_path if result_path.exists() else run / "continuation.json")
    centers, _ = audit.accepted_centers(run, protocol, raw, result, root)
    audit.require(len(centers) >= 2 and len(receipt["accepted_centers"]) == len(centers),
                  "resume requires two audited accepted endpoints")
    for actual, recorded in zip(centers, receipt["accepted_centers"], strict=True):
        for key in ("parameter", "center", "tau", "kind", "cyclic_point_index"):
            audit.require(actual.get(key) == recorded.get(key), "resume accepted-center receipt differs: " + key)
    selected = centers[-2:]
    audit.require(all(p["kind"] == "continued saddle" for p in selected)
                  and selected[0]["parameter"] != selected[1]["parameter"],
                  "resume needs two distinct fully continued endpoints")
    history, warm, endpoints, orbits, point_records = [], [], [], [], []
    for center in selected:
        point = raw[center["cyclic_point_index"]]
        directory = run / f"point-{point['index']:04d}"
        audit.require([p["phase"] for p in point["phases"]] == [0, 6, 12],
                      "resume endpoint needs three phases")
        audit.require(abs(center["parameter"]) <= plan["maximum_abs_beta"]
                      and max(abs(np.asarray(center["center"]))) <= plan["maximum_scaled_coordinate"],
                      "resume endpoint outside current plan")
        points = np.load(directory / "points.npy", allow_pickle=False)
        bases = {}
        for phase in (0, 6, 12):
            with np.load(directory / f"phase-{phase}.npz", allow_pickle=False) as archive:
                bases[phase] = archive["q"].copy()
        history.append({key: center[key] for key in ("parameter", "center", "tau")})
        warm.append((np.asarray(center["center"], float), float(center["parameter"]), points[0].copy(), bases))
        orbits.append(points.copy())
        point_records.append(point)
        endpoints.append({**history[-1], "cyclic_point_index": point["index"],
                          "point_directory": str(directory),
                          "closure_recomputed_from_saved_endpoint": (directory / "return-state.npy").is_file()})
    # Prevent a concurrently changing attempt from becoming a resume snapshot.
    _manifest(run, receipt["manifest"])
    audit.require(audit.digest(receipt_path) == receipt_sha256, "resume receipt changed during verification")
    provenance = {"schema": "BSC3-memory-resume-v1", "receipt_path": str(receipt_path),
                  "receipt_sha256": receipt_sha256, "raw_directory": str(run),
                  "raw_manifest_sha256": _sha_text(receipt["manifest"]),
                  "execution_source_commit": protocol["commit"], "plan_sha256": protocol["plan_sha256"],
                  "analysis_commit": analysis_commit, "analysis_source_sha256": receipt["analysis_source_sha256"],
                  "plan_changes": changes, "endpoints": endpoints,
                  "scope": "External predictor and warm-state history only; no inherited result points."}
    return {"provenance": provenance, "history": history, "warm": warm,
            "orbits": orbits, "point_records": point_records, "plan": plan,
            "previous_args": protocol["args"]}


def verify_resume(provenance, plan, *, root=ROOT, data=DATA, seen=()):
    """Pure verifier for ``protocol.resume``; returns the same bundle as load_resume."""
    bundle = load_resume(provenance["receipt_path"], provenance["receipt_sha256"], plan,
                         root=root, data=data, seen=seen)
    audit.require(bundle["provenance"] == provenance, "resume provenance differs from verified endpoints")
    return bundle


def seed_evaluator(evaluator, bundle):
    """Call only after fresh baseline replay; retain external states, not cache rows."""
    evaluator.warm = [(u.copy(), beta, state.copy(), bases[0].copy())
                      for u, beta, state, bases in bundle["warm"]]
    evaluator.qs = {phase: q.copy() for phase, q in bundle["warm"][-1][3].items()}


def check_resume_settings(stencils, correction_tolerance, previous_args):
    prior = previous_args.get("stencils", [.1, .03])
    audit.require(len(stencils) >= len(prior)
                  and all(np.isfinite(h) and h > 0 for h in stencils)
                  and all(a > b for a, b in zip(stencils, stencils[1:]))
                  and all(h <= old for h, old in zip(stencils, prior))
                  and np.isfinite(correction_tolerance) and 0 < correction_tolerance
                  <= previous_args.get("correction_tolerance", .003),
                  "resume cannot loosen saddle correction tolerance or stencil scales/count")


def replay_template(bundle):
    provenance = bundle["provenance"]
    return {"schema": "BSC3-memory-resume-replay-v1",
            "receipt_sha256": provenance["receipt_sha256"],
            "raw_manifest_sha256": provenance["raw_manifest_sha256"],
            "gates": dict(REPLAY_GATES), "passed": False, "endpoints": [],
            "scope": "Fresh cyclic replay of both bound endpoints precedes target continuation; maximum state difference uses direct phase alignment at every orbit state."}


def audit_replay_endpoint(run, bundle, index, point_index):
    """Recompute one replay's closure, orbit sheet and cyclic spectral agreement."""
    audit.require(isinstance(point_index, int) and not isinstance(point_index, bool) and point_index >= 0,
                  "invalid fresh replay point index")
    directory = Path(run) / f"point-{point_index:04d}"
    fresh, summary = audit.audit_point(directory, bundle["plan"])
    old = bundle["point_records"][index]
    source = bundle["provenance"]["endpoints"][index]
    audit.require(fresh["coordinate"] == old["coordinate"] and fresh["beta"] == old["beta"]
                  and summary["phases"] == [0, 6, 12]
                  and summary["closure_recomputed_from_saved_endpoint"],
                  "fresh resume endpoint lacks bound coordinates, cyclic phases or independent closure")
    points = np.load(directory / "points.npy", allow_pickle=False)
    previous = bundle["orbits"][index]
    denominator = np.linalg.norm(previous, axis=1)
    audit.require(np.all(denominator > 0), "zero prior orbit state")
    state_error = float(np.max(np.linalg.norm(points - previous, axis=1) / denominator))
    audit.require(state_error < REPLAY_GATES["maximum_relative_orbit_difference"],
                  "fresh replay left the bound orbit sheet")
    comparisons = []
    for a, b in zip(old["phases"], fresh["phases"], strict=True):
        audit.require(a["phase"] == b["phase"], "replay cyclic phase differs")
        values = audit.complex_values(a["multipliers"])[:2]
        current = audit.complex_values(b["multipliers"])[:2]
        row = {"phase": a["phase"],
               "pair_relative_error": audit.pair_gap(values, current) / max(abs(values)),
               "trace_error_relative_to_radius": abs(a["tau"] - b["tau"]) / max(abs(values)),
               "determinant_relative_error": abs(a["delta"] - b["delta"]) / abs(a["delta"])}
        audit.require(all(np.isfinite(row[key]) and row[key] < REPLAY_GATES[key]
                          for key in REPLAY_GATES if key != "maximum_relative_orbit_difference"),
                      "fresh replay spectral agreement failed")
        comparisons.append(row)
    return {"source": source, "point_index": point_index,
            "maximum_relative_orbit_difference": state_error,
            "closure_recomputed_from_saved_endpoint": True, "cyclic_comparisons": comparisons}


def verify_replay(run, protocol_resume, bundle, raw):
    """Read-only verification, including partial replay before any target exists.

    The caller must require ``passed`` before accepting target saddles or a
    completed continuation. A failed/interrupted replay may retain fewer rows.
    """
    audit.require(protocol_resume == bundle["provenance"], "replay resume provenance differs")
    record = audit.read(Path(run) / "resume-replay.json")
    template = replay_template(bundle)
    audit.require(set(record) == set(template), "resume replay schema differs")
    for key in template.keys() - {"passed", "endpoints"}:
        audit.require(record[key] == template[key], "resume replay binding differs: " + key)
    audit.require(isinstance(record["passed"], bool) and isinstance(record["endpoints"], list)
                  and len(record["endpoints"]) <= 2, "invalid resume replay status")
    indices = set()
    for index, stored in enumerate(record["endpoints"]):
        point_index = stored["point_index"]
        audit.require(point_index in raw and point_index not in indices, "missing or duplicate fresh replay row")
        indices.add(point_index)
        audit.require(raw[point_index] == audit.read(Path(run) / f"point-{point_index:04d}" / "point.json"),
                      "fresh replay raw point binding differs")
        actual = audit_replay_endpoint(run, bundle, index, point_index)
        audit.require(actual == stored, "resume replay comparison or source differs")
    audit.require(not record["passed"] or len(record["endpoints"]) == 2, "resume replay is incomplete")
    return record


def seed_replayed_evaluator(evaluator, run, bundle, replay):
    """Replace provisional seeds with freshly replayed endpoint states/bases."""
    audit.require(replay["passed"] and len(replay["endpoints"]) == 2, "fresh endpoint replay required")
    warm = []
    for index, row in enumerate(replay["endpoints"]):
        directory = Path(run) / f"point-{row['point_index']:04d}"
        bases = {}
        for phase in (0, 6, 12):
            with np.load(directory / f"phase-{phase}.npz", allow_pickle=False) as z:
                bases[phase] = z["q"].copy()
        source = bundle["history"][index]
        warm.append((np.asarray(source["center"]), source["parameter"],
                     np.load(directory / "points.npy", allow_pickle=False)[0].copy(), bases))
    seed_evaluator(evaluator, {"warm": warm})


def predictor(history, beta):
    if not history:
        return np.zeros(2)
    point = np.asarray(history[-1]["center"], float).copy()
    audit.require(beta != history[-1]["parameter"], "duplicate consecutive continuation parameter")
    if len(history) >= 2:
        a, b = history[-2:]
        audit.require(a["parameter"] != b["parameter"], "duplicate predictor parameters")
        point += ((beta - b["parameter"]) / (b["parameter"] - a["parameter"])) * (
            np.asarray(b["center"]) - np.asarray(a["center"]))
    audit.require(np.isfinite(point).all(), "nonfinite continuation predictor")
    return point
