"""Independent saved-scalar cusp audit. No producer imports or native replay.

Source preparation only: the command-line admission layer awaits the exact
metadata packet. Functions below accept manufactured records for controls.
"""
import math


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
