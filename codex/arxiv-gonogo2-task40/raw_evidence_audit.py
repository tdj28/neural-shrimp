"""Task40 independent saved-evidence reductions. No producer imports or model calls.

Prospective scope: A1 answer/reset records; A5 nearest fold/flip points; A6 six
saved cusp unfoldings; A7 all sync protocols; A8 saddle/exit/tip records.
Run only after Root checkpoints this source and admits its bounded CPU lane.
Inputs are original saved records, not report summaries (summaries are comparisons).
Output is exclusive; errors remain visible per section, never silently dropped.
Suggested envelope: one thread, nice10, 4GiB AS, 90CPU/120wall seconds.
No eigensolve except the six 2x2 SVDs; no scientific producer or backend import.
"""
import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path("/data2/neural-shrimp")
DATA = Path("/data2/neural-shrimp-data")
LOOP = DATA / "explore/llm-soft-loop"
POD = LOOP / "pod64"
OUT = ROOT / "codex/arxiv-gonogo2-task40"
ARTICLE = "bb42a82189da91c37f50804c077832df9e2237ae"
RESEARCH = "c567545ca93f90324dae75be2c333c8ac452dab5"
MAX_BYTES = 256 * 1024**2
MANIFEST = {}
np = None


def original(path):
    """Hash exactly the bytes decoded; refuse ordinary concurrent mutation."""
    path = Path(path)
    if not (path.is_relative_to(DATA) or path.is_relative_to(ROOT / "docs")
            or path.is_relative_to(ROOT / "tools")):
        raise ValueError(f"input outside declared roots: {path}")
    before = path.stat()
    if before.st_size > MAX_BYTES:
        raise ValueError(f"input exceeds disclosed cap: {path}")
    b = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while read: {path}")
    pin = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    if str(path) in MANIFEST and MANIFEST[str(path)] != pin:
        raise ValueError(f"input changed between reads: {path}")
    MANIFEST[str(path)] = pin
    return b


def unique(items):
    d = {}
    for k, v in items:
        if k in d:
            raise ValueError(f"duplicate JSON key {k}")
        d[k] = v
    return d


def decode(b):
    # Original records can contain nonfinite diagnostic placeholders. Preserve
    # their actual spelling as strings, then gate numeric fields explicitly.
    return json.loads(b, object_pairs_hook=unique,
                      parse_constant=lambda s: f"NONFINITE:{s}")


def js(path):
    return decode(original(path))


def jl(path):
    return [decode(line) for line in original(path).splitlines() if line.strip()]


def nz(path):
    b = original(path)
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def pair_records(path):
    by = {}
    for r in jl(path):
        key, lay = r["index"], r["layout"]
        if lay in by.setdefault(key, {}):
            raise ValueError(f"duplicate index/layout: {path}: {key}/{lay}")
        by[key][lay] = r
    bad = [i for i, row in by.items() if set(row) != {8, 16}]
    if bad:
        raise ValueError(f"incomplete pairs at {path}: {bad}")
    return by


def paired_stats(by):
    flips = [i for i, r in by.items() if r[8]["answer"] != r[16]["answer"]]
    n = len(by)
    return {"n": n, "changed": len(flips), "changed_indices": flips,
            "percent": 100 * len(flips) / n,
            "correct": {str(l): sum(r[l]["answer"] == r[l]["gold"] for r in by.values()) for l in (8, 16)},
            "correctness_changed": sum((r[8]["answer"] == r[8]["gold"]) !=
                                      (r[16]["answer"] == r[16]["gold"]) for r in by.values())}


def answers():
    out = {"arithmetic": {}, "reset": {}, "published": {}, "reasoning": {}, "rerun4096": {}}
    for model, folder in (("qwen0.5b", "t05"), ("qwen7b", "t7")):
        recs = [r for r in jl(POD / f"repro/{folder}/runs.jsonl") if r.get("set") == "heldout" and "mode" in r]
        groups = collections.defaultdict(list)
        for r in recs:
            groups[(r["dtype"], r["mode"], r["log10_temperature"])].append(r)
        for key, rows in sorted(groups.items()):
            ids = [r["i"] for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate arithmetic problem within configuration")
            changed = [r for r in rows if r["single"]["parsed"] != r["batch4"]["parsed"]]
            out["arithmetic"][f"{model}/{key}"] = {
                "n": len(rows), "changed": len(changed), "percent": 100 * len(changed) / len(rows),
                "number_vs_none": sum((r["single"]["parsed"] is None) != (r["batch4"]["parsed"] is None)
                                      for r in changed),
                "correct": {a: sum(r[a]["parsed"] == r["problem"]["answer"] for r in rows)
                            for a in ("single", "batch4")},
                "same_layout_failed": sum(not r.get("rerun_identical", False) for r in rows)}
    reset_dirs = {
        "local-bf16": LOOP / "rr/heldout-bf16", "local-float32": LOOP / "rr/heldout-float32",
        "h05-bf16": POD / "rr/h05-heldout-bf16", "h05-float32": POD / "rr/h05-heldout-f32",
        "q7-bf16": POD / "rr/q7-heldout-bf16", "q7-float32": POD / "rr/q7-heldout-f32"}
    for name, folder in reset_dirs.items():
        rows = jl(folder / "runs.jsonl")
        arms = ("free", "reset", "native", "reference_rerun")
        out["reset"][name] = {
            "n": len(rows), "failed_exactness_checks": sum(not r["batch1_checks_pass"] for r in rows),
            "integer_changes": {a: sum(r[a]["parsed"] != r["reference"]["parsed"] for r in rows) for a in arms},
            "text_changes": {a: sum(r[a]["text"] != r["reference"]["text"] for r in rows) for a in arms},
            "raw_compared_flags_not_used": True}
    for arm in ("top15-s64", "replay", "greedy", "top1-s64"):
        out["published"][arm] = paired_stats(pair_records(POD / f"pi/pm/{arm}/records.jsonl"))
    roots = {
        "r1qwen7b": DATA / "explore/pi/pm2-A/explore/llm-soft-loop/pod64/pi/pm2",
        "r1llama8b": POD / "pi/pm2fix", "qwq32b": POD / "pi/pm2"}
    totals = {a: collections.Counter() for a in ("soft", "plain")}
    for model, folder in roots.items():
        for arm in ("soft", "plain"):
            d = folder / f"{model}-gsm8k-{arm}"
            run = js(d / "run.json")
            args = run.get("args", run)
            by = pair_records(d / "records.jsonl")
            stats = paired_stats(by)
            counts = collections.Counter()
            for pair in by.values():
                capped = any(r["forced_switch"] or r["switch_step"] >= args["think_budget"]
                             or r["answer_tokens"] >= args["answer_budget"] - 1 for r in pair.values())
                bucket = "budget_hit" if capped else "finished"
                counts[bucket] += 1
                counts[bucket + "_changed"] += pair[8]["answer"] != pair[16]["answer"]
            stats.update(dict(counts), think_budget=args["think_budget"], answer_budget=args["answer_budget"])
            out["reasoning"][f"{model}/{arm}"] = stats
            totals[arm].update(counts)
    out["reasoning_totals"] = {a: dict(v) for a, v in totals.items()}
    for arm in ("soft", "plain"):
        d = POD / f"pi/pm2cs-q7g/r1qwen7b-gsm8k-{arm}"
        out["rerun4096"][arm] = paired_stats(pair_records(d / "records.jsonl"))
        out["rerun4096"][arm]["run"] = js(d / "run.json")
    out["limits"] = ["Parsed answers are audited as saved; tokenizer decoding is not rerun.",
                     "Batch membership reconstruction needs tokenizer/dataset authority and is not rerun.",
                     "Historical 448-cell reproducibility and behavioral 4000/1000-item studies outside A1 new correction scope."]
    return out


def gate_point(r, kind):
    target = 1 if kind == "fold" else -1
    mu = r.get("critical_multiplier_refined", r.get("critical_multiplier"))
    ok = mu is not None and all(finite(v) for v in mu)
    reasons = []
    if not finite(r.get("residual")) or r["residual"] > 1e-10:
        reasons.append("residual")
    if not finite(r.get("g")) or abs(r["g"]) > 1e-9:
        reasons.append("g")
    if not ok or abs(complex(*mu) - target) > 1e-7:
        reasons.append("critical_multiplier")
    return {"pass": not reasons, "failed": reasons,
            **{k: r.get(k) for k in ("index", "alpha", "log10_temperature", "theta", "residual", "g",
                                    "critical_multiplier", "critical_multiplier_refined", "other_spectral_radius",
                                    "newton", "jvps", "gmres_unconverged", "rejected", "refine_residual")}}


def crossings():
    curves = {"fold_plus": POD / "kite/kite-la-curve-plus/curve.json",
              "fold_minus": POD / "kite/kite-la-curve-minus/curve.json",
              "flip_A": POD / "p6s/p6s-td-curve-minus/curve.json",
              "flip_B": POD / "p6s/p6s-ad-curve-plus/curve.json"}
    selected = {"fold_plus": [0, 23], "fold_minus": [1], "flip_A": [1], "flip_B": [5]}
    out = {}
    for name, path in curves.items():
        rec = js(path)
        kind = rec["kind"]
        rows = []
        for index in selected[name]:
            r = rec["points"][index]
            row = gate_point(r, kind)
            mu = [complex(*v) for v in r["multipliers"]]
            critical = min(range(len(mu)), key=lambda i: abs(mu[i] - (1 if kind == "fold" else -1)))
            row["listed_other_modulus_max"] = max(abs(v) for i, v in enumerate(mu) if i != critical)
            row["record_spectrum_count"] = len(mu)
            rows.append(row)
        near = {}
        for corner, (a, t) in {"A": (0.072991, -0.153984), "B": (0.072900, -0.153707)}.items():
            # Declare metric rather than treating "nearest" as metric independent.
            idx = sorted(range(len(rec["points"])), key=lambda i:
                         (10 * (rec["points"][i]["alpha"] - a))**2 +
                         (rec["points"][i]["log10_temperature"] - t)**2)[:3]
            near[corner] = [{"index": i, **gate_point(rec["points"][i], kind)} for i in idx]
        out[name] = {"kind": kind, "period": rec["period"], "args": rec["args"], "selected": rows,
                     "nearest_in_scaled_theta": near, "all_row_gates": [gate_point(r, kind) for r in rec["points"]]}
    out["limit"] = "Nearby saved spectra support distinct sheets locally; no exact common-parameter A/B solve is replayed."
    return out


def cusps():
    paths = [f"kite/locate-la-plus-{i}" for i in (14, 94, 112)] + [
        "p6s/locate-wide2-a-64", "p6s/locate-wide2-b-82", "p6s/locate-tu-minus-34"]
    rows = []
    for rel in paths:
        r = js(POD / rel / "locate.json")
        f = r["final"]
        matrix = np.asarray(f["J"], dtype=float)
        sv = np.linalg.svd(matrix, compute_uv=False)
        det = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
        curve = js(DATA / "explore" / r["curve"] / "curve.json")
        left = r["left"]
        rows.append({"path": rel, "located": r["located"], "checks": r["checks"], "period": r["period"],
                     "parameter_scale": curve["args"]["alpha_scale"], "theta": f["theta"],
                     "alpha": f["alpha"], "log10_temperature": f["log10_temperature"],
                     "oriented_b_small_step": f["f"], "unfolding_b_other_step": f["b"], "c": f["c"],
                     "c_plateau_relative_spread": f["c_plateau_relative_spread"],
                     "initial_f": [p["f"] for p in r["points"][:2]],
                     "det_recomputed": det, "det_saved": f["det_J"], "sv_recomputed": sv.tolist(),
                     "sv_saved": f["singular_values"], "critical": f["critical_multiplier"],
                     "critical_error": abs(complex(*f["critical_multiplier"]) - 1),
                     "next_modulus": f["second_modulus"],
                     "bracket_point_gates": [gate_point(curve["points"][i], "fold") for i in (left, left + 1)],
                     "final_residual_saved": f.get("residual"), "solver_counts_saved": f.get("newton"),
                     "limits": "Final locate JSON omits final Newton residual/count; small f and coarse b use different step selections."})
    return {"rows": rows,
            "ranges": {key: [min(abs(r[key]) for r in rows), max(abs(r[key]) for r in rows)]
                       for key in ("det_recomputed", "oriented_b_small_step", "c", "critical_error", "next_modulus")},
            "smallest_sv_range": [min(r["sv_recomputed"][-1] for r in rows), max(r["sv_recomputed"][-1] for r in rows)]}


def threshold(rows, tol):
    ordered = sorted(rows, key=lambda r: r["c"])
    failures = [i for i, r in enumerate(ordered) if not r["tail_max"] < tol]
    last = max(failures, default=-1)
    return {"c_lo": ordered[last]["c"] if last >= 0 else None,
            "c_hi": ordered[last + 1]["c"] if last + 1 < len(ordered) else None,
            "non_monotone": any(r["tail_max"] < tol for r in ordered[:last + 1])}


def block_se(values):
    # Match the declared original ten contiguous equal blocks; discard only a
    # remainder, if present, and expose the convention in the result key.
    values = np.asarray(values)
    width = len(values) // 10
    means = values[:10 * width].reshape(10, width).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(len(means)))


def sync():
    base = DATA / "explore/sync"
    summary = js(ROOT / "docs/sync/sync-summary.json")
    meta = js(base / "transverse/transverse.json")
    arrays = nz(base / "transverse/transverse.npz")
    out = {"transverse_source": meta, "cells": {}, "sweeps": {}, "periodic": {},
           "transverse_npz_hash_matches": MANIFEST[str(base / "transverse/transverse.npz")]["sha256"] ==
           meta["transverse_npz_sha256"]}
    for i, c in enumerate(meta["cells"]):
        logs = arrays["jvp_logs"][i]
        if not np.isfinite(logs).all():
            raise ValueError("nonfinite single-loop exponent logs")
        lam = float(np.mean(logs))
        rows = []
        for k, pair in enumerate(meta["pairs"]):
            if pair["cell"] == c["index"]:
                values = arrays["logs"][k]
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite transverse exponent logs")
                measured = float(values.mean())
                pred = lam + math.log(abs(1 - 2 * pair["c"]))
                rows.append({"c": pair["c"], "measured": measured, "predicted": pred,
                             "difference": measured - pred, "se10blocks": block_se(values),
                             "n": len(values)})
        out["cells"][str(c["index"])] = {"lambda1": lam, "n": len(logs),
                                         "cstar": (1 - math.exp(-lam)) / 2, "transverse": rows}
    for name in ("sweep-near", "sweep-refine", "sweep-far"):
        m = js(base / name / "sweep.json")
        z = nz(base / name / "sweep.npz")
        rows = []
        for i, pair in enumerate(m["pairs"]):
            values = z["dist"][i, 1:]
            if len(values) != m["steps"] or not np.isfinite(values).all():
                raise ValueError(f"bad saved distance series: {name}/{i}")
            armed, bursts = False, 0
            for x in values:
                if x < 1e-8:
                    armed = True
                elif x > 1e-3 and armed:
                    bursts += 1
                    armed = False
            rows.append({**pair, "tail_max": float(values[-500:].max()),
                         "min": float(values.min()), "max": float(values.max()), "bursts": bursts})
        brackets = {}
        for ci in (0, 1, 2):
            these = [r for r in rows if r["cell"] == ci]
            brackets[str(ci)] = {str(t): threshold(these, t) for t in (1e-8, 1e-6)}
        out["sweeps"][name] = {"args": m["args"], "steps": m["steps"], "complete": m["complete"],
                               "commit": m["commit"], "npz_hash_matches":
                               MANIFEST[str(base / name / "sweep.npz")]["sha256"] == m["sweep_npz_sha256"],
                               "brackets": brackets, "rows": rows}
    out["summary_bracket_matches"] = {}
    for name, rec in out["sweeps"].items():
        for ci in ("0", "1", "2"):
            for tol, key in (("1e-08", "threshold"), ("1e-06", "threshold_tol_1e-6_secondary")):
                saved = summary["cells"][ci]["sweeps"][name][key]
                calc = rec["brackets"][ci][tol]
                out["summary_bracket_matches"][f"{name}/{ci}/{tol}"] = all(
                    calc[k] == saved.get(k, False if k == "non_monotone" else None) for k in calc)
    for name in ("periodic", "periodic-synced"):
        rec = js(base / name / "periodic.json")
        out["periodic"][name] = {"args": rec["args"], "commit": rec["commit"],
                                 "rows": [{**r, "saved_closure_gate": r["primitive_period"] is not None and
                                           finite(r["return_distance_at_P"]) and
                                           r["return_distance_at_P"] < rec["args"]["tol"]}
                                          for r in rec["pairs"]],
                                 "limit": "Original saves selected return diagnostics, not all 40 returns or state sequence; minimality cannot be independently reconstructed."}
    out["check"] = js(base / "check/check.json")
    out["max_abs_discrepancy"] = max(abs(r["difference"]) for c in out["cells"].values() for r in c["transverse"])
    out["prediction_range"] = [min(r["predicted"] for c in out["cells"].values() for r in c["transverse"]),
                               max(r["predicted"] for c in out["cells"].values() for r in c["transverse"])]
    return out


def hook():
    saddle = js(POD / "kite/hook-saddle/hook-saddle.json")
    s = saddle["saddle"]
    mods = sorted((abs(complex(*m)) for m in s["multipliers"]), reverse=True)
    exdir = LOOP / "kite/hook-saddle-exit-v2"
    exitrec = js(exdir / "hook-saddle-exit.json")
    z = nz(exdir / "distance-to-saddle.npz")
    rows = []
    for i, r in enumerate(exitrec["rows"]):
        d = z["distance"][:, i]
        if not np.isfinite(d).all():
            raise ValueError("nonfinite saddle distance")
        rr = {"side": r["side"], "delta": float(z["delta"][i]), "closest": float(d.min()),
              "closest_step": int(d.argmin()), "ends_with_saved": r["ends_with"],
              "end_return6_saved": r["end_return6"]}
        for rad in (1e-3, 3e-4):
            inside = np.flatnonzero(d < rad)
            rr[f"exit_{rad:g}"] = int(inside[-1]) if len(inside) else None
        rr["saved_min_matches"] = rr["closest"] == r["closest"]
        rows.append(rr)
    fits = {}
    for side in ("below", "above"):
        these = [r for r in rows if r["side"] == side]
        x = np.log([r["delta"] for r in these])
        y = np.log([r["closest"] for r in these])
        fits[side + "_closest"] = float(np.polyfit(x, y, 1)[0])
        for rad in (1e-3, 3e-4):
            use = [r for r in these if r[f"exit_{rad:g}"] is not None and r["closest"] < rad / 10
                   and r[f"exit_{rad:g}"] < z["distance"].shape[0] - 1]
            p = np.polyfit(np.log([1 / r["delta"] for r in use]), [r[f"exit_{rad:g}"] for r in use], 1)
            fits[f"{side}_exit_{rad:g}"] = {"n": len(use), "slope": float(p[0])}
    folds = js(POD / "kite/hook-tip-folds/hook-tip-folds.json")
    good = []
    gates = []
    for r in folds["rows"]:
        ok = r.get("located") and r["fold_residual"] <= 1e-10 and abs(r["fold_g"]) <= 1e-9 and (
            abs(complex(*r["critical_multiplier_refined"]) - 1) <= 1e-7)
        gates.append({"alpha": r["alpha"], "pass": bool(ok),
                      **{k: r.get(k) for k in ("fold_residual", "fold_g", "critical_multiplier_refined",
                                              "orbit_return", "u0", "mu_r", "mu_l")}})
        if ok:
            good.append(r)
    good.sort(key=lambda r: r["alpha"])
    zeros = [a["alpha"] - a["u0"] * (b["alpha"] - a["alpha"]) / (b["u0"] - a["u0"])
             for a, b in zip(good, good[1:]) if a["u0"] * b["u0"] < 0]
    strips = js(POD / "kite/hook-tip-strip2/hook-tip-strip.json")
    boundaries = []
    for offset in sorted({r["rows_below_fold"] for r in strips["cells"]}):
        rs = sorted((r for r in strips["cells"] if r["rows_below_fold"] == offset), key=lambda r: r["alpha"])
        labels = [r["lambda_long"] > -0.15 for r in rs]
        crossings = [i for i in range(1, len(rs)) if labels[i] != labels[i - 1]]
        boundaries.append({"offset": offset, "n": len(rs), "switches": crossings,
                           "brackets": [[rs[i - 1]["alpha"], rs[i]["alpha"]] for i in crossings]})
    offsets = np.array([r["offset"] for r in boundaries])
    mids = np.array([sum(r["brackets"][0]) / 2 for r in boundaries if len(r["brackets"]) == 1])
    if len(mids) != len(offsets):
        raise ValueError("tip extrapolation needs one bracket per offset")
    extrap = {str(power): float(np.polyfit(offsets**power, mids, 1)[1]) for power in (0.5, 0.4, 1 / 3)}
    return {"saddle": s, "bisection_count": len(saddle["bisection"]),
            "edge_log10T": saddle["edge_log10T"], "mu_u": mods[0], "next": mods[1],
            "predicted_exit_slope": 6 / math.log(mods[0]),
            "predicted_closest_exponent": -math.log(mods[1]) / (-math.log(mods[1]) + math.log(mods[0])),
            "saddle_fits_invalid_original_preserved": saddle["slope_steps_per_e_fold"],
            "exit_rows": rows, "exit_fits": fits, "fold_gates": gates, "u0_zeros_linear": zeros,
            "strip2_boundaries": boundaries, "tip_extrapolations": extrap,
            "basin_neighbor_original": js(LOOP / "kite/basin-check/basin-check.json"),
            "limits": ["No model/Newton replay; only saved distances and diagnostics are reduced.",
                       "Original invalid settling-time fit is not used. Valid v2 distance arrays are used.",
                       "Random-start edge absence is a finite-map visual comparison, not formal basin proof.",
                       "Positive folded coordinate is local-normal-form evidence, not exact stable-manifold certification."]}


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise SystemExit(f"required {key}=1")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("offline CPU-only environment required")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT / "raw-evidence.json")
    ap.add_argument("--sections", nargs="+", choices=("answers", "crossings", "cusps", "sync", "hook"),
                    default=["answers", "crossings", "cusps", "sync", "hook"])
    args = ap.parse_args()
    if args.output.parent.resolve() != OUT.resolve():
        raise SystemExit("output must be inside Task40")
    global np
    import numpy as np
    start = time.monotonic()
    result = {"article_commit": ARTICLE, "research_commit": RESEARCH, "sections": {},
              "numpy_version": np.__version__, "model_calls": 0, "producer_imports": False,
              "scope": "Saved-record arithmetic, not original execution/asset authentication or formal proof."}
    with args.output.open("x") as handle:
        for name in args.sections:
            try:
                result["sections"][name] = {"status": "completed", "data": globals()[name]()}
            except Exception as exc:
                result["sections"][name] = {"status": "failed_or_partial", "error": repr(exc),
                                            "traceback": traceback.format_exc()}
        result["manifest"] = MANIFEST
        result["elapsed_seconds"] = time.monotonic() - start
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "elapsed": result["elapsed_seconds"],
                      "sections": {k: v["status"] for k, v in result["sections"].items()},
                      "input_count": len(MANIFEST)}))
    return int(any(v["status"] != "completed" for v in result["sections"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
