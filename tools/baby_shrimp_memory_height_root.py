"""Bounded numerical trace-height roots and read-only saved-result linkage.

No interval enclosure or native-coupling mechanism is certified by this search.

Analytic control in test_baby_shrimp_memory_height_root: at a 1e-10 residual,
h(beta)=beta+0.2*beta**2-0.6 took 18 corrections with fraction .1, and 24
with .001 alone. The explicit residual-progress option reduced this to 9;
the smaller fraction alone is not an efficiency guarantee. A stagnating
false-position control still fails at its unchanged iteration budget.
"""
from pathlib import Path

import numpy as np

import baby_shrimp_memory_finish as audit

SCOPE = "Numerical root of the corrected trace-saddle height; no interval enclosure, exact superstability or native-coupling claim."


def template(height_tolerance=1e-8, parameter_tolerance=1e-10, max_iterations=10,
             interior_fraction=.1, progress_safeguard=False):
    audit.require(np.isfinite(height_tolerance) and height_tolerance > 0
                  and np.isfinite(parameter_tolerance) and parameter_tolerance > 0
                  and np.isfinite(interior_fraction) and 0 < interior_fraction < .5
                  and isinstance(progress_safeguard, bool)
                  and isinstance(max_iterations, int) and 1 <= max_iterations <= 10,
                  "invalid bounded height-root settings")
    return {"schema": "BSC3-memory-height-root-v1", "complete": False,
            "height_tolerance": height_tolerance, "parameter_tolerance": parameter_tolerance,
            "max_iterations": max_iterations, "interior_fraction": interior_fraction,
            "progress_safeguard": progress_safeguard,
            "candidates": [], "initial_bracket": None,
            "saddle_artifacts": [], "iterations": [], "result": None, "failure": None, "scope": SCOPE}


def artifact_rows(run):
    return {path.name: audit.read(path) for path in sorted(Path(run).glob("saddle-*.json"))}


def saddle_result(row):
    return {key: value for key, value in row.items() if key not in ("stage", "cyclic_endpoint")}


def candidates(run, fresh, *, provenance=None, replay=None, raw=None):
    """Bind candidates to fresh saddle artifacts or freshly replayed old centers."""
    saved = artifact_rows(run)
    rows = []
    if provenance is not None:
        audit.require(replay is not None and replay["passed"] and len(replay["endpoints"]) == 2,
                      "height-root parent endpoints require complete fresh replay")
        for index, source in enumerate(provenance["endpoints"]):
            bound = replay["endpoints"][index]
            audit.require(bound["source"] == source, "height-root replay source differs")
            point = raw[bound["point_index"]] if raw is not None else audit.read(
                Path(run) / f"point-{bound['point_index']:04d}" / "point.json")
            audit.require(point["coordinate"] == source["center"] and point["beta"] == source["parameter"]
                          and point["qualified"] and [p["phase"] for p in point["phases"]] == [0, 6, 12],
                          "height-root replay point differs")
            rows.append({"parameter": source["parameter"], "center": source["center"],
                         "tau": point["tau"], "converged": True,
                         "source": {"kind": "replayed parent center", "endpoint_index": index,
                                    "receipt_sha256": provenance["receipt_sha256"],
                                    "raw_manifest_sha256": provenance["raw_manifest_sha256"],
                                    "replay_point_index": point["index"], "endpoint": source}})
    for point in fresh:
        audit.require(point.get("converged") and "receipt_sha256" not in point,
                      "height-root requires newly corrected target saddles")
        matches = [(name, stored) for name, stored in saved.items()
                   if stored["parameter"] == point["parameter"] and stored["center"] == point["center"]
                   and saddle_result(stored) == point]
        audit.require(len(matches) == 1, "fresh height-root candidate lacks a unique saddle artifact")
        name, stored = matches[0]
        rows.append({"parameter": point["parameter"], "center": point["center"], "tau": point["tau"],
                     "converged": True, "source": {"kind": "fresh target", "artifact": name,
                                                     "cyclic_point_index": stored["cyclic_endpoint"]["index"]}})
    rows.sort(key=lambda row: row["parameter"])
    audit.require(len({r["parameter"] for r in rows}) == len(rows), "duplicate height-root candidate parameters")
    audit.require(all(np.isfinite(r["parameter"]) and np.isfinite(r["tau"])
                      and np.asarray(r["center"]).shape == (2,) and np.isfinite(r["center"]).all() for r in rows),
                  "nonfinite height-root candidate")
    return rows


def select_bracket(rows):
    brackets = [(left, right) for i, left in enumerate(rows) for right in rows[i+1:]
                if left["parameter"] < right["parameter"] and left["tau"] * right["tau"] < 0]
    audit.require(brackets, "no strict sign-changing saddle-height bracket")
    return list(min(brackets, key=lambda pair: (pair[1]["parameter"]-pair[0]["parameter"], pair[0]["parameter"])))


def _audited_artifacts(run, centers):
    saved = artifact_rows(run)
    names = [f"saddle-{i:03d}.json" for i in range(len(saved))]
    audit.require(list(saved) == names, "height-root saddle artifact sequence is not unique and contiguous")
    for row in saved.values():
        matches = [c for c in centers if c["parameter"] == row["parameter"] and c["center"] == row["center"]
                   and c["tau"] == row["tau"] and c.get("cyclic_point_index") == row["cyclic_endpoint"]["index"]]
        audit.require(len(matches) == 1, "height-root saddle lacks an audited center")
    return saved


def verify_height_root(run, protocol, result, centers, raw, *, resume_bundle=None, replay=None):
    """Pure verifier after all points/saddles (and optional resume) were audited.

    Accepts a caught partial root failure, but never labels it complete. The
    caller must require ``complete`` when a root-requesting run claims success.
    ``result`` is result.json, or continuation.json for an incomplete attempt.
    """
    args = protocol["args"]
    audit.require(args.get("height_root") and args["mode"] == "continue"
                  and not args.get("anchor_saddle"), "height-root protocol differs")
    expected = template(args.get("height_tolerance", 1e-8), args.get("height_parameter_tolerance", 1e-10),
                        args.get("height_root_iterations", 10), args.get("height_root_interior_fraction", .1),
                        args.get("height_root_progress_safeguard", False))
    path = Path(run) / "height-root.json"
    if not path.exists():
        audit.require(not result.get("complete") and "height_root" not in result,
                      "completed root search lacks its saved root receipt")
        return {"schema": expected["schema"], "complete": False,
                "status": "not started; no saved root receipt", "scope": SCOPE,
                "requested_parameters": args["betas"],
                "recorded_target_parameters": [p["parameter"] for p in result.get("points", [])],
                "audited_center_count": len(centers)}
    record = audit.read(path)
    audit.require(set(record) == set(expected), "height-root receipt schema differs")
    audit.require(isinstance(record["complete"], bool) and isinstance(record["iterations"], list)
                  and isinstance(record["saddle_artifacts"], list), "invalid height-root receipt types")
    for key in ("schema", "height_tolerance", "parameter_tolerance", "max_iterations", "interior_fraction", "progress_safeguard", "scope"):
        audit.require(record[key] == expected[key], "height-root setting differs: " + key)
    audit.require([p["parameter"] for p in result["points"]] == args["betas"],
                  "height root started before all requested fresh targets completed")
    provenance = protocol.get("resume")
    if provenance is not None:
        audit.require(resume_bundle is not None and resume_bundle["provenance"] == provenance
                      and replay is not None and replay["passed"], "height-root resume ancestry not verified")
    saved = _audited_artifacts(run, centers)
    expected_candidates = candidates(run, result["points"], provenance=provenance, replay=replay, raw=raw)
    audit.require(record["candidates"] == expected_candidates, "height-root candidate linkage differs")
    fresh_names = [r["source"]["artifact"] for r in expected_candidates if r["source"]["kind"] == "fresh target"]
    try:
        bracket = select_bracket(expected_candidates)
    except ValueError:
        audit.require(record["initial_bracket"] is None and not record["complete"]
                      and not record["iterations"] and not record["saddle_artifacts"]
                      and record["result"] is None and record["failure"] is not None
                      and not result.get("complete") and set(saved) == set(fresh_names),
                      "missing bracket cannot produce a root result")
        return record
    audit.require(record["initial_bracket"] == bracket, "height-root initial bracket differs")
    artifacts = record["saddle_artifacts"]
    audit.require(artifacts == [name for name in saved if name not in fresh_names],
                  "root saddle artifact linkage or order differs")
    iterations = record["iterations"]
    audit.require(len(artifacts) <= record["max_iterations"]
                  and len(iterations) <= len(artifacts) <= len(iterations)+1,
                  "root iteration/saddle count differs")
    lo, hi = bracket
    converged = False
    previous_progress = False
    for index, name in enumerate(artifacts):
        saddle = saved[name]
        # A caught interruption may preserve a completed callback before its
        # iteration event. Still bind that saddle to the next valid candidate.
        row = iterations[index] if index < len(iterations) else {
            "stage": "height_root_iteration", "iteration": index, "artifact": name,
            **{key: saddle[key] for key in ("parameter", "center", "tau")},
            "bracket": [lo["parameter"], hi["parameter"]], "bracket_heights": [lo["tau"], hi["tau"]]}
        audit.require(not converged and row["stage"] == "height_root_iteration"
                      and row["iteration"] == index and row["artifact"] == artifacts[index],
                      "root iteration ordering differs")
        width = hi["parameter"]-lo["parameter"]
        audit.require(width > record["parameter_tolerance"] and lo["tau"]*hi["tau"] < 0,
                      "root iteration lacks a strict unresolved bracket")
        candidate = (lo["parameter"]*hi["tau"]-hi["parameter"]*lo["tau"])/(hi["tau"]-lo["tau"])
        fraction = record["interior_fraction"]
        eligible = bool(record["progress_safeguard"] and previous_progress and lo["parameter"] < candidate < hi["parameter"])
        if index == len(iterations):
            row["progress_secant_eligible"] = eligible
        if not eligible and not lo["parameter"]+fraction*width <= candidate <= hi["parameter"]-fraction*width:
            candidate = (lo["parameter"]+hi["parameter"])/2
        audit.require(lo["parameter"] < candidate < hi["parameter"] and row["progress_secant_eligible"] == eligible
                      and row["parameter"] == candidate and row["bracket"] == [lo["parameter"], hi["parameter"]]
                      and row["bracket_heights"] == [lo["tau"], hi["tau"]], "root iteration bracket or candidate differs")
        audit.require(all(row[key] == saddle[key] for key in ("parameter", "center", "tau")),
                      "root iteration refers to a different audited saddle")
        weight = (candidate-lo["parameter"])/width
        predictor = (1-weight)*np.asarray(lo["center"])+weight*np.asarray(hi["center"])
        audit.close(saddle["history"][0]["center"], predictor, "root predictor differs", rtol=0, atol=1e-13)
        converged = abs(row["tau"]) <= record["height_tolerance"]
        previous_progress = abs(row["tau"]) <= .5 * min(abs(lo["tau"]), abs(hi["tau"]))
        if not converged:
            if lo["tau"]*row["tau"] < 0:
                hi = saddle
            else:
                lo = saddle
    if record["complete"]:
        audit.require(converged and len(artifacts) == len(iterations) and record["failure"] is None,
                      "completed root has not met the height gate")
        numerical = record["result"]
        audit.require(numerical["complete"] and numerical["height_tolerance"] == record["height_tolerance"]
                      and numerical["parameter_tolerance"] == record["parameter_tolerance"]
                      and numerical["interior_fraction"] == record["interior_fraction"]
                      and numerical["progress_safeguard"] == record["progress_safeguard"]
                      and numerical["bracket"] == iterations[-1]["bracket"]
                      and numerical["bracket_width"] == numerical["bracket"][1]-numerical["bracket"][0],
                      "numerical root summary differs")
        history = [{k: v for k, v in row.items() if k not in ("stage", "artifact")} for row in iterations]
        audit.require(numerical["history"] == history, "numerical root iteration history differs")
        audit.require(saddle_result(saved[artifacts[-1]]) == numerical["saddle"],
                      "numerical root saddle linkage differs")
        _, evaluations = audit.continuation_events(Path(run), raw)
        audit.require(numerical["evaluations"] == len(evaluations), "root evaluation count differs")
        audit.require(result.get("complete") and result.get("height_root") == record,
                      "completed root differs from final result")
    else:
        audit.require(record["result"] is None and not result.get("complete"),
                      "partial root cannot produce a completed result")
    return record
