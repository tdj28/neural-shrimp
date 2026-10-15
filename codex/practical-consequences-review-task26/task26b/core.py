"""Independent saved REPRO1 arithmetic. No producer/model imports or fitting."""
import math
from collections import Counter
from decimal import Decimal, ROUND_CEILING, localcontext

import numpy as np

COMPARISONS = ("batch64", "batch32", "precision", "bf16")
LABELS = ("chaotic", "periodic", "other")
OLD_FIELDS = ("first_step_dist_above_1e-2", "first_disagreement",
              "disagreeing_for_good_from", "tail_disagreement_fraction",
              "final_dist", "max_dist", "min_nonzero_dist")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def upward(value, digits=3):
    """Display upper bound; raw extrema remain the authoritative values."""
    if value == 0:
        return "0"
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal.from_float(float(value))
        return str(d.quantize(Decimal(1).scaleb(d.adjusted() - digits + 1),
                              rounding=ROUND_CEILING))


def distribution(values, ids):
    require(len(values) == len(ids), "distribution association")
    if not values:
        return {"n": 0, "min": None, "max": None, "median": None,
                "q05": None, "q95": None, "min_cells": [], "max_cells": [],
                "max_upward_3sig": None}
    a = np.asarray(values, dtype=np.float64)
    require(np.isfinite(a).all(), "nonfinite distribution")
    lo, hi = float(a.min()), float(a.max())
    return {"n": len(values), "min": lo, "max": hi,
            "median": float(np.median(a)), "q05": float(np.quantile(a, .05)),
            "q95": float(np.quantile(a, .95)),
            "min_cells": [i for i, x in zip(ids, a) if x == lo],
            "max_cells": [i for i, x in zip(ids, a) if x == hi],
            "max_upward_3sig": upward(hi)}


def cell_summary(distance, agreement, tail):
    require(distance.ndim == agreement.ndim == 1 and distance.shape == agreement.shape,
            "cell shape")
    require(distance.dtype == np.dtype("<f4") and agreement.dtype == np.dtype("bool"),
            "cell dtype")
    require(0 < tail <= len(distance), "tail")
    require(np.isfinite(distance).all() and (distance >= 0).all(), "invalid distance")
    over = np.flatnonzero(distance > .01)
    different = np.flatnonzero(~agreement)
    positive = np.flatnonzero(distance > 0)
    agreeing = np.flatnonzero(agreement)
    suffix = None if agreement[-1] else (int(agreeing[-1]) + 1 if len(agreeing) else 0)
    minimum = float(distance[positive].min()) if len(positive) else 0.
    tail_count = int(np.count_nonzero(~agreement[-tail:]))
    return {
        "first_step_dist_above_1e-2": int(over[0]) + 1 if len(over) else None,
        "first_disagreement": int(different[0]) + 1 if len(different) else None,
        "disagreeing_for_good_from": suffix,
        "terminal_disagreeing_suffix_start_step_1based": suffix + 1 if suffix is not None else None,
        "tail_disagreement_fraction": tail_count / tail,
        "tail_disagreement_count": tail_count, "any_tail_disagreement": tail_count > 0,
        "final_disagreement": bool(not agreement[-1]),
        "final_dist": float(distance[-1]), "max_dist": float(distance.max()),
        "max_dist_steps_1based": (np.flatnonzero(distance == distance.max()) + 1).tolist(),
        "min_nonzero_dist": minimum,
        "minimum_positive_steps_1based": (np.flatnonzero(distance == minimum) + 1).tolist() if minimum > 0 else [],
        "first_positive_dist": float(distance[positive[0]]) if len(positive) else None,
        "first_positive_step_1based": int(positive[0]) + 1 if len(positive) else None,
        "max_dist_le_1e-2": not len(over),
    }


def validate_spec(spec, saved, steps, tail, expected_counts=None):
    n = len(spec["cells"])
    require(n > 0 and all(len(spec[k]) == n for k in
            ("label", "source", "map_exponent", "map_period")), "spec lengths")
    cells = np.asarray(spec["cells"], dtype=np.float64)
    require(cells.shape == (n, 2) and np.isfinite(cells).all(), "cell coordinates")
    require(np.isfinite(np.asarray(spec["map_exponent"], dtype=float)).all(), "map exponents")
    require(all(x in LABELS for x in spec["label"]), "unknown label")
    require(len(set((str(s), *c) for s, c in zip(spec["source"], spec["cells"]))) == n,
            "duplicate source/coordinate identity")
    if expected_counts is not None:
        require(dict(Counter(spec["label"])) == expected_counts, "cohort count")
    require(saved["cells"] == spec["cells"] and saved["labels"] == spec["label"],
            "original cell/label order")
    require(set(saved["comparisons"]) == set(COMPARISONS), "comparison coverage")
    require(saved["args"]["steps"] == steps and saved["args"]["tail"] == tail
            and saved["args"]["compare"] == list(COMPARISONS), "step/tail/compare contract")
    for comp in COMPARISONS:
        require(len(saved["comparisons"][comp]) == n, "summary row coverage")
        require(all(set(r) == set(OLD_FIELDS) for r in saved["comparisons"][comp]),
                "summary field coverage")
    return n


def group_summary(rows):
    n = len(rows)
    crossed = [r for r in rows if r["first_step_dist_above_1e-2"] is not None]
    noncrossed = [r for r in rows if r["first_step_dist_above_1e-2"] is None]
    result = {"n": n, "cells": [r["cell"] for r in rows],
              "crossing_count": len(crossed), "noncrossing_cells": [r["cell"] for r in noncrossed],
              "crossing_share": len(crossed) / n if n else None,
              "conditional_crossing_steps_1based": distribution(
                  [r["first_step_dist_above_1e-2"] for r in crossed], [r["cell"] for r in crossed])}
    for metric in ("final_dist", "max_dist", "first_positive_dist", "min_nonzero_dist",
                   "tail_disagreement_fraction", "tail_disagreement_count"):
        defined = [r for r in rows if r[metric] is not None]
        result[metric] = distribution([r[metric] for r in defined], [r["cell"] for r in defined])
        result[metric]["undefined_cells"] = [r["cell"] for r in rows if r[metric] is None]
    for metric in ("any_tail_disagreement", "final_disagreement"):
        ids = [r["cell"] for r in rows if r[metric]]
        result[metric] = {"count": len(ids), "denominator": n, "share": len(ids) / n if n else None,
                          "cells": ids}
    result["noncrossers_max_dist"] = distribution([r["max_dist"] for r in noncrossed],
                                                  [r["cell"] for r in noncrossed])
    result["literal_final_dist_le_3e-12"] = {
        "count": sum(r["final_dist"] <= 3e-12 for r in rows), "denominator": n,
        "share": sum(r["final_dist"] <= 3e-12 for r in rows) / n if n else None,
        "q95_linear": result["final_dist"]["q95"]}
    return result


def forecast_row(row, seed_key, steps):
    lam, seed = row["map_exponent"], row[seed_key]
    observed = row["first_step_dist_above_1e-2"]
    eligible = observed is not None and lam > 0
    reasons = ([] if observed is not None else ["observed_noncrosser"]) + ([] if lam > 0 else ["nonpositive_map_exponent"])
    result = {"cell": row["cell"], "map_exponent": lam, "seed_raw": seed,
              "first_positive_step_1based": row["first_positive_step_1based"],
              "observed_crossing_step_1based": observed, "historical_cohort_eligible": eligible,
              "historical_exclusion_reasons": reasons, "prediction_duration": None,
              "ratio_observed_over_prediction": None, "prediction_exceeds_horizon": None,
              "disposition": "undefined_seed" if seed is None else "undefined_nonpositive_exponent"}
    if seed is None or lam <= 0:
        return result
    prediction = math.log(.01 / max(seed, 1e-17)) / lam
    if not math.isfinite(prediction):
        result["disposition"] = "nonfinite_prediction"
        return result
    result.update(prediction_duration=prediction, prediction_exceeds_horizon=prediction > steps,
                  disposition="defined_positive" if prediction > 0 else "nonpositive_prediction")
    if observed is not None and prediction != 0:
        result["ratio_observed_over_prediction"] = observed / prediction
    return result


def forecast_summary(rows):
    cohort = [r for r in rows if r["historical_cohort_eligible"]]
    problems = [r["cell"] for r in cohort if r["disposition"] != "defined_positive"]
    answer = {"all_chaotic_n": len(rows), "historical_cohort_n": len(cohort),
              "excluded_cells": [r["cell"] for r in rows if not r["historical_cohort_eligible"]],
              "invalid_prediction_cells": problems, "median_ratio": None, "loglog_r": None,
              "aggregate_disposition": "undefined_empty"}
    if not cohort or problems:
        if problems:
            answer["aggregate_disposition"] = "undefined_invalid_prediction_no_silent_filter"
        return answer
    obs = np.asarray([r["observed_crossing_step_1based"] for r in cohort], dtype=float)
    pred = np.asarray([r["prediction_duration"] for r in cohort], dtype=float)
    answer["median_ratio"] = float(np.median(obs / pred))
    answer["ratios"] = distribution((obs / pred).tolist(), [r["cell"] for r in cohort])
    a, b = np.log(obs), np.log(pred)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if len(cohort) < 2 or denom == 0:
        answer["aggregate_disposition"] = "undefined_constant_or_singleton_correlation"
    else:
        answer["loglog_r"] = float(np.dot(a, b) / denom)
        answer["aggregate_disposition"] = "defined"
    return answer


def calculate(arrays, saved, spec, steps=1500, tail=500, expected_counts=None):
    n = validate_spec(spec, saved, steps, tail, expected_counts)
    expected = {f"{c}_{k}" for c in COMPARISONS for k in ("dist", "agree")}
    require(set(arrays) == expected, "array coverage")
    cells, differences, groups, by_source, figure = [], [], {}, {}, {}
    for comp in COMPARISONS:
        d, a = arrays[f"{comp}_dist"], arrays[f"{comp}_agree"]
        require(d.shape == a.shape == (steps, n), "array shape")
        require(d.dtype == np.dtype("<f4") and a.dtype == np.dtype("bool"), "array dtype")
        require(np.isfinite(d).all() and (d >= 0).all(), "nonfinite/negative distance: " + comp)
        rows = []
        for i in range(n):
            row = cell_summary(d[:, i], a[:, i], tail)
            row.update(cell=i, comparison=comp, source=spec["source"][i], label=spec["label"][i],
                       coordinates=spec["cells"][i], map_exponent=spec["map_exponent"][i],
                       map_period=spec["map_period"][i])
            for key in OLD_FIELDS:
                old, own = saved["comparisons"][comp][i][key], row[key]
                if old != own:
                    differences.append({"comparison": comp, "cell": i, "field": key,
                                        "original": old, "own": own,
                                        "delta": own - old if isinstance(old, (float, int)) and isinstance(own, (float, int)) else None})
            rows.append(row)
        cells.extend(rows)
        groups[comp] = {label: group_summary([r for r in rows if r["label"] == label]) for label in LABELS}
        by_source[comp] = {str(source): {label: group_summary([r for r in rows if r["source"] == source and r["label"] == label])
                                       for label in LABELS} for source in sorted(set(spec["source"]), key=str)}
        figure[comp] = {}
        for label in LABELS:
            selected = d[:, np.asarray(spec["label"]) == label]
            if selected.shape[1]:
                # Preserve the original float32 clipping before float64 quantiles.
                figure[comp][label] = {"raw_q25_q50_q75": np.quantile(selected, [.25, .5, .75], axis=1).tolist(),
                    "display_floor_1e-17_q25_q50_q75": np.quantile(np.maximum(selected, 1e-17), [.25, .5, .75], axis=1).tolist(),
                    "display_floor_float32_median_used_in_panel_b": np.median(np.maximum(selected, 1e-17), axis=1).tolist()}
            else:
                figure[comp][label] = None
    chaotic = [r for r in cells if r["comparison"] == "batch64" and r["label"] == "chaotic"]
    forecasts = {}
    for name, key in (("literal_whole_run_minimum", "min_nonzero_dist"), ("first_positive_correction", "first_positive_dist")):
        rows = [forecast_row(r, key, steps) for r in chaotic]
        forecasts[name] = {"rows": rows, "summary": forecast_summary(rows)}
    return {"schema": "task26b-arithmetic-v1", "computational_completed": True,
            "original_summary_fields_checked": len(cells) * len(OLD_FIELDS),
            "original_summary_differences": differences, "all_original_summaries_equal": not differences,
            "steps": steps, "tail": tail, "cell_count": n, "cohort_counts": dict(Counter(spec["label"])),
            "groups": groups, "groups_by_source": by_source, "forecasts": forecasts,
            "seed_definition_differs_cells": [r["cell"] for r in chaotic if r["first_positive_dist"] != r["min_nonzero_dist"]],
            "cells": cells, "figure_summaries": figure,
            "scope": "Saved float32 distances and boolean pre-update readout agreement; no model/state/token-ID reconstruction. Source-map labels and original historical load order retained."}
