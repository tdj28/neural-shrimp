"""Independent saved-scalar cusp audit. No producer imports or native replay.

The card must be reviewed before the command-line layer reads any target.
All scientific functions also accept manufactured records for controls.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

SCHEMA = "codex-task21e-saved-cusps-v1"
LIMITS = {"cpu_seconds": 20, "wall_seconds": 30, "as_bytes": 1024 ** 3,
          "input_bytes": 4 * 1024 ** 2, "output_bytes": 16 * 1024 ** 2,
          "startup_available_bytes": 20 * 1024 ** 3, "runtime_available_bytes": 16 * 1024 ** 3}
LOCATOR_SHA = "a8e711ad9bf4530633b6c41b341b43680326ff6d869f859634e6ec2fae3957de"
FOLD_SOURCE_SHA = "d7b665f874aa9183e5dc62be79a9e0ba75051a30d20b9b7e926781e5e541fcf6"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, path):
    require(type(value) in (int, float), f"{path}: expected a real scalar")
    return float(value)


def pair(value, path):
    require(isinstance(value, list) and len(value) == 2, f"{path}: expected two entries")
    return [number(x, f"{path}/{i}") for i, x in enumerate(value)]


def tagged(value):
    """Preserve nonfinite values in strict JSON without quietly deleting rows."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {k: tagged(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tagged(v) for v in value]
    return value


def nonfinite_paths(value, path=""):
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_paths(v, path + "/" + k)]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in nonfinite_paths(v, path + "/" + str(i))]
    return []


def compare(path, calculated, saved):
    result = {"path": path, "calculated": calculated, "saved": saved,
              "exact_equal": type(calculated) is type(saved) and calculated == saved}
    if type(calculated) in (int, float) and type(saved) in (int, float):
        result["exact_equal"] = calculated == saved
        if math.isfinite(calculated) and math.isfinite(saved):
            result.update(signed_difference=calculated - saved,
                          absolute_difference=abs(calculated - saved))
    return result


def sign(value):
    return None if not math.isfinite(value) else (value > 0) - (value < 0)


def matrix_scalars(matrix):
    """Closed-form 2x2 scalar identities, no eigensolver or SVD call."""
    require(isinstance(matrix, list) and len(matrix) == 2, "J: expected two rows")
    (a, b), (c, d) = [pair(row, f"J/{i}") for i, row in enumerate(matrix)]
    if not all(math.isfinite(x) for x in (a, b, c, d)):
        return {"available": False, "reason": "nonfinite saved J"}
    determinant = a * d - b * c
    smax = (math.hypot(a + d, c - b) + math.hypot(a - d, c + b)) / 2
    smin = abs(determinant) / smax if smax else 0.0
    return {"available": True, "det_J": determinant, "singular_values": [smax, smin],
            "condition": smax / smin if smin else float("inf"),
            "method": "direct determinant; closed-form hypot singular values"}


def spectral(mu, second, unit_tol, separation):
    re, im = pair(mu, "multiplier")
    second = number(second, "second_modulus")
    # These are the actual refusal predicates, including their NaN behavior.
    refusal_real = abs(im) > unit_tol or abs(re - 1) > unit_tol
    refusal_complement = second > 1 - separation * 1e-3 and second >= 1 - 1e-3
    return {"multiplier": mu, "modulus": math.hypot(re, im),
            "real_displacement_from_one": re - 1, "imaginary_part": im,
            "second_modulus": second, "unit_tol": unit_tol, "separation": separation,
            "source_real_refusal": refusal_real,
            "source_complement_refusal": refusal_complement,
            "source_accepts": not (refusal_real or refusal_complement),
            "all_saved_scalars_finite": all(math.isfinite(x) for x in (re, im, second)),
            "native_classification_recomputed": False}


def cubic_ladder(table):
    """Optional scalar reconstruction; the normal locator omits this table."""
    if table is None:
        return {"available": False, "reason": "c_steps not saved; coefficient and spread inherited"}
    require(isinstance(table, list) and len(table) > 0, "c_steps: empty or invalid")
    rows = []
    values = []
    for i, row in enumerate(table):
        direct = number(row["c_direct"], "c_direct")
        centre = number(row["c_centre_manifold"], "c_centre_manifold")
        saved = number(row["c"], "c")
        rows.append(compare(f"/c_steps/{i}/c", direct + centre, saved))
        values.append(saved)
    if not all(math.isfinite(x) for x in values):
        return {"available": False, "reason": "nonfinite c ladder", "rows": rows}
    diffs = [abs(b - a) for a, b in zip(values, values[1:])]
    k = min(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    coefficient = (values[k] + values[k + 1]) / 2 if diffs else values[0]
    return {"available": True, "rows": rows, "selected_indices": [k, k + 1] if diffs else [0],
            "c": coefficient,
            "spread": diffs[k] / max(abs(coefficient), 1e-300) if diffs else None,
            "selection": "first minimum absolute adjacent difference"}


def bracket_checks(record):
    points, brackets = record["points"], record["brackets"]
    require(isinstance(points, list) and len(points) >= 2, "points: initial pair missing")
    require(isinstance(brackets, list) and len(brackets) >= 2, "brackets: refinement missing")
    args = record["args"]
    rounds, per_round = args["rounds"], args["per_round"]
    require(type(rounds) is int and rounds >= 1, "rounds: invalid")
    require(type(per_round) is int and per_round >= 1, "per_round: invalid")
    comparisons = [compare("/brackets/count", rounds + 1, len(brackets)),
                   compare("/points/initial_rounds", [0, 0], [p["round"] for p in points[:2]])]
    initial = [number(p["f"], "initial f") for p in points[:2]]
    comparisons.append(compare("/initial_f_max", max(map(abs, initial)), record["initial_f_max"]))
    previous_left = points[0]
    scans = []
    last_arc_pair = None
    prior_round = 0
    for p in points[2:]:
        require(type(p["round"]) is int and 1 <= p["round"] <= rounds, "point round out of range")
        require(p["round"] >= prior_round, "point rounds unordered")
        pair(p["theta"], "point theta")
        number(p["f"], "point f")
        number(p["arclength"], "point arclength")
        prior_round = p["round"]
    for i, bracket in enumerate(brackets):
        comparisons.append(compare(f"/brackets/{i}/round", i, bracket["round"]))
        fs = pair(bracket["f"], "bracket f")
        gap = number(bracket["gap"], "bracket gap")
        scans.append({"round": i, "gap": gap, "f": fs,
                      "finite_sign_change": None if None in [sign(x) for x in fs]
                      else sign(fs[0]) != sign(fs[1]),
                      "gap_recomputed_from_states": False})
        if i == 0:
            comparisons.append(compare("/brackets/0/f", initial, bracket["f"]))
            continue
        group = [p for p in points[2:] if p["round"] == i]
        require(1 <= len(group) <= per_round, "per-round point coverage invalid")
        left = previous_left
        left_arc = 0.0  # trace_curve's first record always has arclength zero.
        found = None
        for pos, right in enumerate(group):
            ls, rs = sign(number(left["f"], "left f")), sign(number(right["f"], "right f"))
            if ls is not None and rs is not None and ls != rs:
                found = (left, right, left_arc, right["arclength"], pos)
                break
            left, left_arc = right, right["arclength"]
        scans[-1]["first_crossing_found"] = found is not None
        if found is not None:
            left, right, la, ra, pos = found
            comparisons.extend([
                compare(f"/brackets/{i}/f", [left["f"], right["f"]], bracket["f"]),
                compare(f"/brackets/{i}/theta/1", right["theta"], bracket["theta"][1]),
                compare(f"/round/{i}/stopped_at_first_crossing", pos + 1, len(group))])
            if pos > 0:
                comparisons.append(compare(f"/brackets/{i}/theta/0", left["theta"], bracket["theta"][0]))
            scans[-1]["left_theta_reconstruction"] = ("saved preceding point" if pos > 0 else
                "unavailable: trace_curve corrects its initial point; that initial record is omitted")
            previous_left = left
            last_arc_pair = (ra - la, left["f"], right["f"])
        else:
            last_arc_pair = None
    if last_arc_pair is not None:
        arc, fl, fr = last_arc_pair
        step = arc * fl / (fl - fr)
        comparisons.append(compare("/final/secant_step", step, record["final"]["secant_step"]))
    return {"rows": scans, "comparisons": comparisons,
            "secant_reconstruction_available": last_arc_pair is not None}


def analyze_record(record, identity, *, alpha_scale=10.0, unit_tol=1e-6, separation=0.5):
    require(record["test"] == "cusp" and record["kind"] == "fold", "not a fold cusp record")
    require(type(record["located"]) is bool, "located: expected boolean")
    require(set(record["checks"]) == {"b_small", "c_nonzero", "unfolding_regular"}, "check keys differ")
    require(all(type(x) is bool for x in record["checks"].values()), "check type differs")
    final = record["final"]
    f = number(final["f"], "final/f")
    b = number(final["b"], "final/b")
    c = number(final["c"], "final/c")
    spread = final["c_plateau_relative_spread"]
    if spread is not None:
        number(spread, "c_plateau_relative_spread")
    f0 = number(record["initial_f_max"], "initial_f_max")
    btol = number(record["btol"], "btol")
    determinant = number(final["det_J"], "det_J")
    condition = number(final["condition"], "condition")
    theta = pair(final["theta"], "final/theta")
    sv = pair(final["singular_values"], "singular_values")
    computed = {"b_small": abs(f) <= btol * f0,
                "c_nonzero": c != 0 and (spread is None or spread < 0.1),
                "unfolding_regular": abs(determinant) > 0 and math.isfinite(condition)}
    comparisons = [compare("/checks/" + k, v, record["checks"][k]) for k, v in computed.items()]
    comparisons.extend([compare("/located", all(computed.values()), record["located"]),
                        compare("/btol_vs_args", btol, record["args"]["btol"]),
                        compare("/final/log10_temperature", theta[0], final["log10_temperature"]),
                        compare("/final/alpha", theta[1] / alpha_scale, final["alpha"])])
    small_matrix = matrix_scalars(final["J"])
    if small_matrix["available"]:
        comparisons.append(compare("/final/det_J", small_matrix["det_J"], determinant))
        for i, value in enumerate(small_matrix["singular_values"]):
            comparisons.append(compare(f"/final/singular_values/{i}", value, sv[i]))
        comparisons.append(compare("/final/condition_from_J", small_matrix["condition"], condition))
    saved_ratio = sv[0] / sv[-1] if sv[-1] > 0 else float("inf")
    comparisons.append(compare("/final/condition_from_saved_singular_values", saved_ratio, condition))
    brackets = bracket_checks(record)
    comparisons.extend(brackets["comparisons"])
    ladder = cubic_ladder(final.get("c_steps"))
    if ladder["available"]:
        comparisons.extend([compare("/final/c", ladder["c"], c),
                            compare("/final/c_plateau_relative_spread", ladder["spread"], spread)])
    return tagged({"id": identity, "original": record, "source_checks_reconstructed": computed,
                   "source_located_reconstructed": all(computed.values()),
                   "nonfinite_paths": nonfinite_paths(record),
                   "relative_oriented_f": abs(f) / f0 if math.isfinite(f0) and f0 > 0 else None,
                   "small_step_oriented_f": f, "coarser_unfolding_b": b,
                   "b_estimates_are_distinct": True, "bracket_checks": brackets,
                   "spectral_source_checks": spectral(final["multiplier"], final["second_modulus"],
                                                       unit_tol, separation),
                   "matrix_scalar_reconstruction": small_matrix, "cubic_ladder": ladder,
                   "comparisons": comparisons,
                   "nonexact_comparisons": [x for x in comparisons if not x["exact_equal"]],
                   "native_evidence_recomputed": False, "rigorous_cusp_certificate": False})


def analyze_curve(curve, identity):
    """Every stored tangent reversal, within the declared original curve only."""
    require(curve["kind"] == "fold", "selected source curve is not a fold")
    require(type(curve["period"]) is int and curve["period"] > 0, "curve period invalid")
    points = curve["points"]
    require(isinstance(points, list) and points, "curve has no saved points")
    rows, computed_reversals, saved_reversals = [], [], []
    previous = None
    for i, point in enumerate(points):
        require(type(point["index"]) is int and point["index"] == i, "curve index coverage invalid")
        require(type(point["cusp_candidate"]) is bool, "cusp_candidate is not Boolean")
        tangent = pair(point["parameter_tangent"], "parameter_tangent")
        dot = None if previous is None else previous[0] * tangent[0] + previous[1] * tangent[1]
        computed = previous is not None and dot < 0
        if computed:
            computed_reversals.append({"left": i - 1, "right": i})
        if point["cusp_candidate"]:
            saved_reversals.append({"left": i - 1, "right": i})
        rows.append({"index": i, "original": point, "tangent_dot_previous": dot,
                     "candidate_comparison": compare(f"/points/{i}/cusp_candidate", computed, point["cusp_candidate"]),
                     "nonfinite_paths": nonfinite_paths(point)})
        previous = tangent
    return tagged({"id": identity, "kind": curve["kind"], "period": curve["period"],
                   "args": curve["args"], "point_count": len(points), "rows": rows,
                   "computed_reversals": computed_reversals, "saved_reversals": saved_reversals,
                   "candidate_mismatches": [x["candidate_comparison"] for x in rows
                                            if not x["candidate_comparison"]["exact_equal"]],
                   "scope": "complete saved points of this declared curve, not all possible continuations"})


def reversal_coverage(curves, selected):
    """Compare the frozen nine-row selection with every admitted curve reversal."""
    require(len({c["id"] for c in curves}) == len(curves), "duplicate source curve id")
    by_id = {c["id"]: c for c in curves}
    selected_pairs = []
    rows = []
    require(len({s["id"] for s in selected}) == len(selected), "duplicate selected cusp id")
    for item in selected:
        curve_id, left = item["curve_id"], item["left"]
        require(curve_id in by_id, "selected cusp source curve missing")
        require(type(left) is int and 0 <= left < by_id[curve_id]["point_count"] - 1,
                "selected bracket index outside source curve")
        key = (curve_id, left)
        require(key not in selected_pairs, "duplicate selected reversal bracket")
        selected_pairs.append(key)
        curve = by_id[curve_id]
        bracket = {"left": left, "right": left + 1}
        rows.append({"id": item["id"], "curve_id": curve_id, **bracket,
                     "matches_saved_candidate": bracket in curve["saved_reversals"],
                     "matches_reconstructed_reversal": bracket in curve["computed_reversals"]})
    computed = {(c["id"], p["left"]) for c in curves for p in c["computed_reversals"]}
    saved = {(c["id"], p["left"]) for c in curves for p in c["saved_reversals"]}
    chosen = set(selected_pairs)
    encode = lambda pairs: [{"curve_id": c, "left": i, "right": i + 1} for c, i in sorted(pairs)]
    return {"selected_count": len(rows), "rows": rows,
            "computed_reversal_count": len(computed), "saved_candidate_count": len(saved),
            "unselected_computed_reversals": encode(computed - chosen),
            "selected_without_computed_reversal": encode(chosen - computed),
            "unselected_saved_candidates": encode(saved - chosen),
            "selected_without_saved_candidate": encode(chosen - saved),
            "selection_exactly_covers_declared_curve_reversals": chosen == computed,
            "global_curve_universe_exhaustion_claimed": False}


def parse_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def admit_manifest(manifest, source_map):
    require(manifest["schema"] == "task21e-saved-inputs-v1", "wrong input schema")
    records, curves = manifest["records"], manifest["curves"]
    require((manifest["selected_cusp_count"], manifest["separate_other_count"], manifest["curve_count"]) == (9, 6, 7),
            "declared coverage differs")
    require(len(records) == 15 and len({r["id"] for r in records}) == 15, "missing/duplicate record")
    require(len(curves) == 7 and len({c["id"] for c in curves}) == 7, "missing/duplicate curve")
    cusps = [r for r in records if r["test"] == "cusp"]
    require(len(cusps) == 9, "nine cusp records required")
    require(all(r["test"] in ("cusp", "crossing", "reversal") for r in records), "unknown disposition")
    require(len({r["file"]["path"] for r in records}) == 15, "duplicate locate input path")
    require(len({c["file"]["path"] for c in curves}) == 7, "duplicate curve input path")
    by_curve = {c["id"]: c for c in curves}
    source = {(s["source_commit"], s["repository_path"]): s["sha256"] for s in source_map["source_files"]}
    for row in records:
        require(type(row["left"]) is int and row["right"] == row["left"] + 1, "bad declared bracket")
        require(source[(row["source_commit"], "tools/cusp_locate.py")] == LOCATOR_SHA and
                source[(row["source_commit"], "tools/bifurcation_curves.py")] == FOLD_SOURCE_SHA,
                "original source contract differs")
        if row["test"] == "cusp":
            require(row["curve_id"] in by_curve and row["declared_kind"] == "fold", "cusp curve missing/wrong kind")
            c = by_curve[row["curve_id"]]
            require(row["declared_period"] == c["declared_period"], "source period mismatch")
            require(row["curve_file"] == c["file"], "source curve byte binding differs")
            require(row["curve_metadata_scale"] == c["declared_args"]["alpha_scale"] and
                    math.isfinite(row["curve_metadata_scale"]) and row["curve_metadata_scale"] != 0,
                    "source alpha scale mismatch")
    for c in curves:
        actual = sorted(r["left"] for r in cusps if r["curve_id"] == c["id"])
        require(actual == sorted(c["selected_left_indices"]) and len(set(actual)) == len(actual),
                "selected curve indices missing/duplicated")
        require(c["declared_kind"] == "fold" and type(c["point_count"]) is int and c["point_count"] > 1,
                "invalid declared source curve")
    return cusps


def admit_record(record, spec):
    require(record["args"] == spec["declared_args"], "record args differ")
    for actual, expected in [(record["curve"], spec["curve_id"]), (record["left"], spec["left"]),
                             (record["period"], spec["declared_period"]), (record["kind"], spec["declared_kind"]),
                             (record["test"], spec["test"])]:
        require(actual == expected, "record identity differs")
    require(record["args"]["curve"] == record["curve"] and record["args"]["left"] == record["left"] and
            record["args"]["test"] == record["test"], "internal record identity differs")
    require(type(record["located"]) is bool, "record located flag is not Boolean")
    require(len(record["points"]) == spec["schema"]["points"]["length"] and
            len(record["brackets"]) == spec["schema"]["brackets"]["length"], "saved row coverage differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    card_raw = Path(args.card).read_bytes()
    card = parse_json(card_raw)
    require(card.get("schema") == SCHEMA and card.get("admission") == "reviewed" and card.get("limits") == LIMITS,
            "prepared or invalid card: refusing execution")
    out = Path(args.out)
    require(out.is_absolute() and str(out) == card["output"] and not out.exists(), "exact fresh output required")
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    require(priority <= 10, "exact nice10 required")
    os.nice(10 - priority)
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    require(hard == resource.RLIM_INFINITY or hard >= LIMITS["as_bytes"], "inherited hard AS too small")
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS["as_bytes"], LIMITS["as_bytes"]))
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["cpu_seconds"]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", "CPU1 environment required: " + name)
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    require(set(card["source_sha256"]) == {"analyze.py"} and
            sha(Path(__file__).read_bytes()) == card["source_sha256"]["analyze.py"], "own source pin differs")
    admitted, cache, records, curves, others = [], {}, [], [], []
    input_total = 0

    def load(binding, role, decode=True):
        nonlocal input_total
        path = Path(binding["path"])
        require(path.is_absolute(), "input path must be absolute")
        if str(path) in cache:
            raw = cache[str(path)]
        else:
            require(path.stat().st_size <= LIMITS["input_bytes"] - input_total, "input byte ceiling exceeded")
            raw = path.read_bytes()
            input_total += len(raw)
            cache[str(path)] = raw
            admitted.append({"path": str(path), "bytes": len(raw), "sha256": sha(raw), "first_role": role})
        require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"], "input size/hash mismatch: " + str(path))
        return parse_json(raw) if decode else None

    out.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    try:
        manifest = load(card["inputs_manifest"], "input inventory")
        authority = load(manifest["authority"], "inherited original authority")
        source_map = load(manifest["source_map"], "original source map")
        selected = admit_manifest(manifest, source_map)
        for source in source_map["source_files"] + source_map["claims"]:
            load(source, "original source or claim bytes", decode=False)
        for spec in manifest["curves"]:
            raw = load(spec["file"], "source curve")
            require(raw["args"] == spec["declared_args"] and raw["kind"] == spec["declared_kind"] and
                    raw["period"] == spec["declared_period"] and len(raw["points"]) == spec["point_count"],
                    "source curve args/coverage differs")
            row = {"id": spec["id"], "computational_completed": False, "original": raw}
            curves.append(row)
            row.update(analyze_curve(raw, spec["id"]))
            row.pop("original")  # Every point is retained in rows; avoid a second full copy.
            row.update(computational_completed=True, file=spec["file"], association_limit=spec["association_limit"])
        for spec in manifest["records"]:
            raw = load(spec["file"], "located disposition")
            row = {"id": spec["id"], "computational_completed": False, "original": tagged(raw)}
            (records if spec["test"] == "cusp" else others).append(row)
            admit_record(raw, spec)
            if spec["test"] == "cusp":
                row.update(analyze_record(raw, spec["id"], alpha_scale=spec["curve_metadata_scale"]))
            else:
                row.update(scope="separate original crossing/reversal disposition, not included in the cusp count",
                           native_typing_recomputed=False)
            row.update(computational_completed=True, file=spec["file"], original_path=spec["original_path"],
                       original_attempt=spec["attempt"], original_source_commit=spec["source_commit"])
        coverage = reversal_coverage(curves, selected)
        for binding in admitted:
            raw = Path(binding["path"]).read_bytes()
            require(len(raw) == binding["bytes"] and sha(raw) == binding["sha256"], "input changed during audit")
        result = {"schema": SCHEMA, "computational_completed": True, "scientific_claims_accepted": False,
                  "cusps": records, "curves": curves, "other_dispositions": others, "reversal_coverage": coverage,
                  "input_pins": admitted, "input_bytes": input_total,
                  "card_sha256": sha(card_raw), "source_sha256": card["source_sha256"],
                  "original_authority_inherited": authority, "source_association_inherited": source_map,
                  "calculation_input_io_and_posthash_seconds": time.perf_counter() - started,
                  "observed_peak_rss_kib_before_serialization": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "scope": "Independent unblinded saved-scalar arithmetic; original native calculations inherited."}
        payload = json.dumps(tagged(result), indent=2, allow_nan=False).encode() + b"\n"
        inventory = json.dumps({"files": [{"path": "report.json", "bytes": len(payload), "sha256": sha(payload)}]},
                               indent=2).encode() + b"\n"
        require(len(payload) + len(inventory) <= LIMITS["output_bytes"], "output byte ceiling exceeded")
        (out / "report.json").open("xb").write(payload)
        (out / "manifest.json").open("xb").write(inventory)
    except Exception as exc:
        failure = {"computational_completed": False, "error_type": type(exc).__name__, "error": str(exc),
                   "cusps": records, "curves": curves, "other_dispositions": others, "admitted": admitted}
        payload = json.dumps(tagged(failure), indent=2, allow_nan=False).encode() + b"\n"
        if len(payload) <= LIMITS["output_bytes"]:
            (out / "failure.json").open("xb").write(payload)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
