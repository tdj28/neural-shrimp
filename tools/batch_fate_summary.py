"""Summary of a batch_fate.py run (run BF): per layout the captures, the agreement between layouts start by start,
the candidate-detection step distributions, trajectory separation, and both required determinism controls.
Legacy capture/orbit field names denote recurrence candidates only, never verified periodic attractors.
Incomplete records, missing controls, or failed controls raise before a receipt is written.

Usage: python tools/batch_fate_summary.py --run llm-soft-loop/batch-fate/bf1 \
           --control llm-soft-loop/batch-fate/bf1-repeat64 --control llm-soft-loop/batch-fate/bf1-repeat1 \
           --out docs/batch-fate/bf1-receipt.json
"""
import argparse
import hashlib
import json
import math
import os
import sys
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analysis_source():
    path = "tools/batch_fate_summary.py"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", path], cwd=ROOT, text=True).strip())
    return {"path": path, "sha256": sha256(Path(__file__)), "git_head": head, "tool_dirty": dirty}


def artifact(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_layout(arrays, starts, args, label):
    """Structural gates for every saved row; does not re-evaluate the model."""
    n, d = starts.shape
    steps, every = args["steps"], args["record_every"]
    cap, per, rec = arrays["cap"], arrays["per"], arrays["rec"]
    require(cap.shape == per.shape == arrays["mind"].shape == (n,), f"{label}: row count")
    require(cap.dtype.kind in "iu" and per.dtype.kind in "iu", f"{label}: integer detector outputs")
    require(np.all((cap == -1) | ((cap >= 2) & (cap <= steps))), f"{label}: capture step")
    caught = cap >= 0
    require(np.all(per[~caught] == 0), f"{label}: period without recurrence")
    require(np.all((per[caught] >= 1) & (per[caught] <= args["max_period"]) &
                   (per[caught] < cap[caught])), f"{label}: detected period")
    require(rec.shape == (steps // every + 1, n, d) and rec.dtype == np.float64,
            f"{label}: recorded-state shape or dtype")
    require(np.array_equal(rec[0], starts), f"{label}: start identity")
    require(arrays["final"].shape == (n, d) and np.isfinite(arrays["final"]).all(),
            f"{label}: finite final states")
    require(set(arrays["cyc"]) == set(np.flatnonzero(caught)) == set(arrays["cyc_tok"]),
            f"{label}: candidate identity")
    for row in range(n):
        record = rec[:, row]
        finite = np.isfinite(record).all(axis=1)
        blank = np.isnan(record).all(axis=1)
        require(np.all(finite | blank) and not np.any(np.diff(finite.astype(int)) > 0),
                f"{label}/{row}: nonfinite or interrupted records")
        if not caught[row]:
            require(finite.all() and np.isfinite(arrays["mind"][row]) and arrays["mind"][row] >= 0,
                    f"{label}/{row}: incomplete horizon")
        else:
            q = int(per[row])
            pts = arrays["cyc"][row]
            require(pts.shape == (q, d) and pts.dtype == np.float64 and np.isfinite(pts).all() and
                    np.all(np.linalg.norm(pts, axis=1) > 0), f"{label}/{row}: candidate states")
            require(len(arrays["cyc_tok"][row]) == q, f"{label}/{row}: candidate token count")
            require(finite[:cap[row] // every + 1].all(), f"{label}/{row}: records before detection")
    return {"rows_checked": n, "structural_status": "pass",
            "detector_residuals": "not_saved", "primitive_period": "not_independently_verified",
            "stability": "not_evaluated", "crossing_type": "not_applicable"}


def load(run):
    d = DATA / "explore" / run
    rec = json.loads((d / "fate.json").read_text())
    with np.load(d / "fate.npz", allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    args = rec["args"]
    n = args["starts"]
    require(n > 0 and args["steps"] >= 2 and args["record_every"] > 0, "invalid run dimensions")
    require(z["starts"].ndim == 2 and z["starts"].shape[0] == n and
            z["starts"].dtype == np.float64 and np.isfinite(z["starts"]).all() and
            np.all(np.linalg.norm(z["starts"], axis=1) > 0), "invalid starts")
    require(hashlib.sha256(np.ascontiguousarray(z["starts"]).tobytes()).hexdigest() == rec["starts_sha256"],
            "starts SHA-256 mismatch")
    require(z["directions"].shape == z["starts"].shape and np.isfinite(z["directions"]).all(),
            "invalid directions")
    sizes = [int(v) for v in args["layouts"].split(",")]
    expected_labels, seen = [], {}
    for size in sizes:
        require(size > 0 and n % size == 0, "invalid layout size")
        seen[size] = seen.get(size, 0) + 1
        expected_labels.append(f"b{size}" if seen[size] == 1 else f"b{size}-run{seen[size]}")
    require(rec["labels"] == expected_labels, "layout census mismatch")
    expected_chunks = {f"{lab}-{start:05d}.npz" for lab, size in zip(expected_labels, sizes)
                       for start in range(0, n, size)}
    require({p.name for p in (d / "chunks").glob("*.npz")} == expected_chunks, "incomplete or extra chunk set")
    lays = {}
    for lab, size in zip(expected_labels, sizes):
        tokens = {int(k): v for k, v in json.loads(str(z[f"{lab}_cyc_tok"][0])).items()}
        states = {int(k.rsplit("_", 1)[1]): z[k] for k in z if k.startswith(f"{lab}_cyc_state_")}
        layout = {key: z[f"{lab}_{key}"] for key in ("cap", "per", "rec", "mind", "final")}
        layout.update(cyc_tok=tokens, cyc=states)
        layout["gates"] = validate_layout(layout, z["starts"], args, lab)
        caught = layout["cap"] >= 0
        summary = rec["summary"][lab]
        require(summary["captured"] == int(caught.sum()) and summary["not_captured"] == int((~caught).sum()),
                f"{lab}: JSON census disagrees with raw arrays")
        for start in range(0, n, size):
            with np.load(d / "chunks" / f"{lab}-{start:05d}.npz", allow_pickle=False) as chunk:
                end = int(chunk["t_end"])
                expected_end = int(layout["cap"][start:start + size].max())
                if np.any(layout["cap"][start:start + size] < 0):
                    expected_end = args["steps"]
                require(end == expected_end, f"{lab}/{start}: early or inconsistent termination")
                for key in ("cap", "per", "final", "mind", "rec"):
                    target = layout[key][:, start:start + size] if key == "rec" else layout[key][start:start + size]
                    require(np.array_equal(chunk[key], target, equal_nan=True), f"{lab}/{start}: chunk {key} mismatch")
                recorded = end // args["record_every"] + 1
                require(np.isfinite(chunk["rec"][:recorded]).all() and np.isnan(chunk["rec"][recorded:]).all(),
                        f"{lab}/{start}: records inconsistent with termination")
                keys = [int(k) for k in chunk["cyc_keys"]]
                require(len(keys) == len(set(keys)) and set(keys) == set(np.flatnonzero(caught[start:start + size]) + start),
                        f"{lab}/{start}: chunk candidate census")
                ctok = {int(k): v for k, v in json.loads(str(chunk["cyc_tok"][0])).items()}
                require(ctok == {k: tokens[k] for k in keys}, f"{lab}/{start}: chunk tokens")
                for key in keys:
                    require(np.array_equal(chunk[f"cyc_state_{key}"], states[key]),
                            f"{lab}/{start}: chunk candidate states")
        lays[lab] = layout
    return d, rec, z, lays


def validate_controls(rec, z, lays, loaded_controls):
    """Require both BF determinism controls with exact settings, starts and horizon."""
    n = len(z["starts"])
    require(n >= 2 and "b1" in lays and f"b{n}" in lays, "BF requires singleton and full-batch layouts")
    required = {f"b{n}": n, "b1": 2}
    horizon = min(500, rec["args"]["steps"])
    results, seen = [], set()
    settings = ("model", "layer", "vectors", "concept", "alpha", "log_t", "spread", "seed", "max_period",
                "tol", "record_every", "dtype", "device", "threads")
    for name, (_, crec, cz, clays) in loaded_controls:
        require(crec["args"]["steps"] == horizon, f"{name}: wrong control horizon")
        require(all(crec["args"][k] == rec["args"][k] for k in settings), f"{name}: control settings differ")
        require(len(clays) == 1, f"{name}: expected exactly one control layout")
        lab, layout = next(iter(clays.items()))
        require(lab in required and lab not in seen, f"{name}: unexpected or duplicate control")
        rows = required[lab]
        require(crec["args"]["starts"] == rows and np.array_equal(cz["starts"], z["starts"][:rows]),
                f"{name}: control start identity")
        require(np.array_equal(cz["directions"], z["directions"][:rows]), f"{name}: control direction identity")
        expected_hash = hashlib.sha256(np.ascontiguousarray(z["starts"][:rows]).tobytes()).hexdigest()
        require(crec["starts_sha256"] == expected_hash, f"{name}: full control start hash")
        expected = lays[lab]["rec"][:horizon // rec["args"]["record_every"] + 1, :rows]
        require(layout["rec"].shape == expected.shape and np.array_equal(layout["rec"], expected, equal_nan=True),
                f"{name}: recorded states are not bitwise reproducible")
        expected_cap = lays[lab]["cap"][:rows].copy()
        expected_cap[expected_cap > horizon] = -1
        expected_per = lays[lab]["per"][:rows].copy()
        expected_per[expected_cap < 0] = 0
        require(np.array_equal(layout["cap"], expected_cap) and np.array_equal(layout["per"], expected_per),
                f"{name}: recurrence detector is not reproducible")
        seen.add(lab)
        results.append({"run": name, "layout": lab, "steps": horizon, "rows": rows,
                        "settings_match": True, "starts_sha256_match": True,
                        "recorded_states_bitwise_equal": True, "detector_outputs_equal": True})
    require(seen == set(required), "missing BF determinism controls (full batch and first two singletons)")
    return results


def orbit_ids(lays, tol=1e-6):
    """Group recurrence candidates only when every phase-aligned full state matches."""
    reps, ids = [], {}
    for lab, L in lays.items():
        for s, pts in sorted(L["cyc"].items()):
            for k, (q, rp) in enumerate(reps):
                if q == len(pts):
                    dist = min(float(np.max(np.linalg.norm(rp - np.roll(pts, shift, axis=0), axis=1) /
                                                   np.minimum(np.linalg.norm(rp, axis=1),
                                                              np.linalg.norm(np.roll(pts, shift, axis=0), axis=1))))
                               for shift in range(q))
                    if dist < tol:
                        ids[(lab, s)] = k
                        break
            else:
                reps.append((len(pts), pts))
                ids[(lab, s)] = len(reps) - 1
    return reps, ids



def start_census(lays, n):
    detected = {lab: set(np.flatnonzero(L["cap"] >= 0).tolist()) for lab, L in lays.items()}
    union = set.union(*detected.values())
    intersection = set.intersection(*detected.values())
    steps = [int(L["cap"][row]) for lab, L in lays.items() for row in detected[lab]]
    return {"detected_start_ids": {lab: sorted(rows) for lab, rows in detected.items()},
            "detected_in_any_layout": len(union), "detected_in_all_layouts": len(intersection),
            "undetected_in_all_layouts": n - len(union),
            "union_start_ids": sorted(union), "intersection_start_ids": sorted(intersection),
            "detection_step_range": [min(steps), max(steps)] if steps else None}


def recorded_return_checks(lays, every):
    """Compare saved returns separated by whole candidate periods, not every step."""
    rows, maximum = {}, None
    for lab, L in lays.items():
        rows[lab] = []
        for row in np.flatnonzero(L["cap"] >= 0):
            q, cap = int(L["per"][row]), int(L["cap"][row])
            spacing = math.lcm(every, q) // every
            recorded = L["rec"][:, row]
            indices = {i for i in range(len(recorded)) if i * every >= cap and np.isfinite(recorded[i]).all()}
            pairs = [(i, i + spacing) for i in sorted(indices) if i + spacing in indices]
            distances = [float(np.linalg.norm(recorded[i] - recorded[j]) / np.linalg.norm(recorded[i]))
                         for i, j in pairs]
            bound = max(distances) if distances else None
            if bound is not None:
                require(np.isfinite(bound), f"{lab}/{row}: nonfinite saved-return distance")
                maximum = bound if maximum is None else max(maximum, bound)
            rows[lab].append({"start": int(row), "candidate_period": q, "detected_at": cap,
                              "post_detection_record_pairs": len(pairs),
                              "max_relative_distance_whole_periods_apart": bound})
    return {"record_every_steps": every, "per_layout": rows,
            "largest_checked_distance": maximum,
            "scope": "Saved same-phase records only; intervening steps, primitive period and stability are unverified. Singleton trajectories stop at detection."}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d, rec, z, lays = load(args.run)
    n = len(z["starts"])
    steps = rec["args"]["steps"]
    every = rec["args"]["record_every"]
    loaded_controls = [(c, load(c)) for c in args.control]
    controls = validate_controls(rec, z, lays, loaded_controls)
    reps, ids = orbit_ids(lays)
    orbits = [{"orbit": k, "period": int(q), "tokens_first_capture": next(
        (lays[lab]["cyc_tok"][s] for (lab, s), kk in ids.items() if kk == k), None)} for k, (q, _) in enumerate(reps)]
    per_layout = {}
    for lab, L in lays.items():
        cap = L["cap"]
        caught = cap >= 0
        surv = {str(t): int(np.sum(~caught | (cap > t))) for t in sorted({t for t in (100, 300, 500, 1000, 2000, 3000, 4000, steps) if t <= steps})}
        per_layout[lab] = {
            "captured": int(caught.sum()), "not_captured": int((~caught).sum()),
            "by_orbit": {str(k): int(sum(1 for s in range(n) if ids.get((lab, s)) == k)) for k in range(len(reps))},
            "capture_steps": {str(s): int(cap[s]) for s in np.nonzero(caught)[0]},
            "uncaptured_after_steps": surv,
            "uncaptured_min_recurrence_distance": (float(np.nanmin(L["mind"][~caught])) if (~caught).any() else None)}
    agreement = {}
    for a, b in combinations(lays, 2):
        ca, cb = lays[a]["cap"] >= 0, lays[b]["cap"] >= 0
        both = np.nonzero(ca & cb)[0]
        same_orbit = int(sum(1 for s in both if ids[(a, s)] == ids[(b, s)]))
        same_step = int(sum(1 for s in both if lays[a]["cap"][s] == lays[b]["cap"][s]))
        agreement[f"{a}~{b}"] = {"both": int(len(both)), "both_same_orbit": same_orbit, "both_same_step": same_step,
                                 f"only_{a}": int(np.sum(ca & ~cb)), f"only_{b}": int(np.sum(~ca & cb)),
                                 "neither": int(np.sum(~ca & ~cb)),
                                 "starts_only_one": [int(s) for s in np.nonzero(ca ^ cb)[0]]}
    # separation of the layouts' trajectories at each recorded step (rows still running in both layouts)
    separation = {}
    for a, b in combinations(lays, 2):
        ra, rb = lays[a]["rec"], lays[b]["rec"]
        m = min(len(ra), len(rb))
        out = {}
        for i in range(1, m):
            ok = ~np.isnan(ra[i, :, 0]) & ~np.isnan(rb[i, :, 0])
            if ok.sum() == 0:
                continue
            rel = np.linalg.norm(ra[i, ok] - rb[i, ok], axis=1) / np.linalg.norm(ra[i, ok], axis=1)
            out[str(i * every)] = {"rows": int(ok.sum()), "median": float(np.median(rel)), "max": float(rel.max()),
                                   "min": float(rel.min())}
        separation[f"{a}~{b}"] = out
    receipt = {"schema": "batch-fate-recurrence-candidates-v2", "status": "complete_structural_and_control_gates_passed",
               "scope": "exploratory finite-horizon recurrence detection at one cell and device",
               "candidate_semantics": "First single full-state recurrence below tol. No sustained closure, Newton refinement, primitive-period or stability verification. Uncaptured rows are unresolved at this horizon, not verified chaotic attractors.",
               "orbit_grouping": {"method": "all phase-aligned states", "relative_tolerance": 1e-6,
                                  "label": "candidate groups, not verified orbits"},
               "row_gates": {lab: L["gates"] for lab, L in lays.items()},
               "start_census": start_census(lays, n),
               "recorded_returns": recorded_return_checks(lays, every),
               "metadata_gaps": ["Original runner did not save execution commit, runtime versions, precision qualification or capture residuals; current versions cannot establish historical runtime metadata."],
               "data_root": str(DATA),
               "artifacts": [artifact(p, DATA) for rd in [d] + [r[0] for _, r in loaded_controls]
                             for p in [rd / "fate.json", rd / "fate.npz", *sorted((rd / "chunks").glob("*.npz"))]],
               "analysis_source": analysis_source(),
               "original_run_execution_commit": None,
               "run": args.run, "args": rec["args"], "labels": rec["labels"], "starts_sha256": rec["starts_sha256"],
               "fate_npz_sha256": sha256(d / "fate.npz"), "fate_json_sha256": sha256(d / "fate.json"),
               "orbits": orbits, "per_layout": per_layout, "agreement": agreement,
               "separation_between_layouts": separation, "controls": controls,
               "seconds_per_layout": {lab: rec["summary"][lab]["seconds"] for lab in rec["labels"]}}
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("x") as f:
        f.write(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("orbits", "per_layout", "agreement", "controls")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
