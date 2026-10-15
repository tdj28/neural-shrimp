"""Own saved-field MECH/XO statistics, with the declared producer RNG contract."""
import math
import numpy as np

FIELDS = ("mean_log_g_net", "mean_log_g_read", "mean_entropy", "mean_gap",
          "mean_log_r1_rel", "mean_stable_rank")
ARMS = ("arm00", "arm10", "arm01", "arm11")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def weighted(values, weights):
    return float(np.sum(np.asarray(values, dtype=float) * weights) / weights.sum())


def finite_summary(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return {"count": int(a.size), "finite_count": int(finite.size),
            "nan_count": int(np.isnan(a).sum()), "positive_infinity_count": int(np.isposinf(a).sum()),
            "negative_infinity_count": int(np.isneginf(a).sum()),
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "minimum": float(finite.min()) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None}


def row_contrast(cells, field, other_group, rng, draws=10000):
    """Literal field-finite masks, rounded/isclose rows, weighted resampling/shuffles."""
    require(other_group in ("near_zero", "periodic"), "row comparison group")
    require(type(draws) is int and draws > 0, "positive integer draw count")
    t = np.array([c["log10_temperature"] for c in cells], dtype=float)
    values = np.array([c[field] for c in cells], dtype=float)
    group_a = np.array([c["chaotic"] for c in cells], dtype=bool)
    group_b = (np.abs([c["alpha"] for c in cells]) <= .01 if other_group == "near_zero"
               else np.array([c["period"] > 0 for c in cells]))
    rows, included, means, counts, members = [], [], [], [], []
    before = rng.bit_generator.state
    for temperature in np.unique(np.round(t, 6)):
        row = np.isclose(t, temperature)
        finite = row & np.isfinite(values)
        a = np.flatnonzero(finite & group_a)
        b = np.flatnonzero(finite & group_b)
        overlap = np.intersect1d(a, b).tolist()
        admitted = len(a) >= 3 and len(b) >= 3
        row_record = {"rounded_temperature": float(temperature), "all_row_indices": np.flatnonzero(row).tolist(),
                      "nonfinite_field_indices": np.flatnonzero(row & ~np.isfinite(values)).tolist(),
                      "group_a_indices": a.tolist(), "group_b_indices": b.tolist(),
                      "overlap_indices": overlap, "group_a_count": len(a), "group_b_count": len(b),
                      "admitted": admitted, "weight": len(a) if admitted else None,
                      "exclusion": None if admitted else "fewer_than_three_in_at_least_one_finite_group"}
        if admitted:
            xa, xb = values[a], values[b]
            difference = float(xa.mean() - xb.mean())
            row_record.update(group_a_mean=float(xa.mean()), group_b_mean=float(xb.mean()),
                              difference=difference)
            included.append(len(rows))
            means.append(difference)
            counts.append(len(a))
            members.append((xa, xb))
        rows.append(row_record)
    if not members:
        return {"summary": None, "rows": rows, "rng_before": before, "rng_after": rng.bit_generator.state,
                "reason": "no_eligible_temperature_rows"}
    d, w = np.array(means), np.array(counts, dtype=float)
    estimate = weighted(d, w)
    bootstrap = []
    for _ in range(draws):
        indices = rng.integers(0, len(d), len(d))
        bootstrap.append(weighted(d[indices], w[indices]))
    permutations, exceedances = [], 0
    for _ in range(draws):
        shuffled_differences = []
        for xa, xb in members:
            combined = np.concatenate((xa, xb))
            rng.shuffle(combined)
            shuffled_differences.append(float(combined[:len(xa)].mean() - combined[len(xa):].mean()))
        statistic = weighted(shuffled_differences, w)
        permutations.append(statistic)
        exceedances += abs(statistic) >= abs(estimate)
    summary = {"estimate": estimate, "ci95": [float(v) for v in np.quantile(bootstrap, [.025, .975], method="linear")],
               "p_permutation": (int(exceedances) + 1) / (draws + 1), "rows": len(members),
               "rows_larger_than_zero": int(np.count_nonzero(d > 0)),
               "group_a_cells": sum(len(a) for a, _ in members), "group_b_cells": sum(len(b) for _, b in members)}
    return {"summary": summary, "rows": rows, "included_row_indices": included, "draws": draws,
            "permutation_exceedances": int(exceedances), "bootstrap_statistics": bootstrap,
            "permutation_statistics": permutations, "rng_before": before, "rng_after": rng.bit_generator.state,
            "overlap_in_an_included_row": any(rows[i]["overlap_indices"] for i in included),
            "inference_scope": "temperature-row bootstrap and literal within-row pooled shuffle; overlap duplicates shared observations"}


def xo_statistics(data, selection, rng, draws=10000):
    require(type(draws) is int and draws > 0, "positive integer draw count")
    cells, labels = data["cells"], data["labels"]
    require(len(cells) == len(labels) == len(selection["cells"]) == len(selection["labels"]), "XO coverage")
    require(labels == selection["labels"], "XO fixed original labels")
    expected_windows = data["args"]["record"] // data["args"]["window"]
    require(data["args"]["record"] % data["args"]["window"] == 0 and expected_windows > 0, "XO complete windows")
    records, errors, selected = [], [], []
    for i, (cell, expected, label) in enumerate(zip(cells, selection["cells"], labels)):
        require([cell["log10_temperature"], cell["alpha"]] == expected, f"XO coordinate {i}")
        windows = {}
        for arm in ARMS:
            values = cell[arm + "_windows"]
            require(isinstance(values, list) and len(values) == expected_windows, f"XO windows {i}/{arm}")
            mean = float(np.mean(values))
            windows[arm] = {"saved_windows": values, "saved_mean": cell[arm], "own_mean": mean,
                            "signed_mean_difference": mean - cell[arm],
                            "finite": bool(np.isfinite(values).all() and math.isfinite(cell[arm]))}
        error = cell["arm11"] - cell["twin_exponent"]
        ok = math.isfinite(error) and abs(error) <= .02
        held_chaotic = label == "chaotic" and i >= 2
        if held_chaotic:
            selected.append(i)
        records.append({"index": i, "coordinates": expected, "original_label": label,
                        "original_map_exponent": selection.get("map_exponent", [None] * len(cells))[i],
                        "development": label == "chaotic" and i < 2, "selected_for_contrasts": held_chaotic,
                        "arm_windows": windows, "twin_exponent": cell["twin_exponent"],
                        "arm11_minus_twin": error, "original_validation_within_0.02": ok,
                        "current_arm11_positive": cell["arm11"] > 0,
                        "current_twin_positive": cell["twin_exponent"] > 0})
        errors.append(error)
    X = np.array([[c[arm] for arm in ARMS] for c in cells], dtype=float)
    require(selected, "no fixed held-out XO chaotic cells")
    Xs = X[selected]
    vectors = (
        ("readout_effect (10-00)", Xs[:, 1] - Xs[:, 0]),
        ("network_effect (01-00)", Xs[:, 2] - Xs[:, 0]),
        ("interaction (11-10-01+00)", Xs[:, 3] - Xs[:, 1] - Xs[:, 2] + Xs[:, 0]),
        ("full_minus_readout_only (11-10)", Xs[:, 3] - Xs[:, 1]),
    )
    summary, details = {}, {}
    for name, v in vectors:
        before = rng.bit_generator.state
        boots = [float(rng.choice(v, len(v)).mean()) for _ in range(draws)]
        summary[name] = {"mean": float(v.mean()), "ci95": [float(x) for x in np.quantile(boots, [.025, .975], method="linear")],
                         "cells": len(v)}
        details[name] = {"cell_indices": selected, "values": v.tolist(), "bootstrap_statistics": boots,
                         "finite": bool(np.isfinite(v).all()), "rng_before": before, "rng_after": rng.bit_generator.state}
    summary["validation_max_abs(arm11 - twin)"] = float(max(abs(e) for e in errors))
    summary["validation_cells_within_0.02"] = sum(r["original_validation_within_0.02"] for r in records)
    summary["arm10_minus_arm11_mean"] = float((Xs[:, 1] - Xs[:, 3]).mean())
    for tag, sign in (("log10T<0", lambda t: t < 0), ("log10T>=0", lambda t: t >= 0)):
        indices = [i for i in selected if sign(cells[i]["log10_temperature"])]
        if indices:
            a = X[indices]
            summary[tag] = {"cells": len(indices), "arm_means": dict(zip(ARMS, a.mean(0).round(4).tolist())),
                            "readout_effect": float((a[:, 1] - a[:, 0]).mean()),
                            "network_effect": float((a[:, 2] - a[:, 0]).mean()),
                            "interaction": float((a[:, 3] - a[:, 1] - a[:, 2] + a[:, 0]).mean())}
            details[tag] = {"cell_indices": indices, "unrounded_arm_means": dict(zip(ARMS, a.mean(0).tolist()))}
    closeness = abs(summary["arm10_minus_arm11_mean"])
    interval = summary["readout_effect (10-00)"]["ci95"]
    separation = all(math.isfinite(x) for x in interval) and (interval[0] > 0 or interval[1] < 0)
    gates = {"all_24_validation_pass": all(r["original_validation_within_0.02"] for r in records),
             "within_0.1_on_average": math.isfinite(closeness) and closeness <= .1,
             "absolute_mean_arm10_minus_arm11": closeness,
             "mean_absolute_difference_descriptive_only": float(np.abs(Xs[:, 1] - Xs[:, 3]).mean()),
             "readout_contrast_interval_excludes_zero": separation,
             "original_sufficiency_joint_pass": math.isfinite(closeness) and closeness <= .1 and separation}
    return {"summary": summary, "cell_records": records, "contrasts": details, "gates": gates}


def combined_statistics(alpha_cells, bias_cells, xo, selection, draws=10000):
    rng = np.random.default_rng(0)
    literal, detail = {}, {}
    for name, fields, comparison in (("R1_chaotic_minus_near_zero", FIELDS[:2], "near_zero"),
                                      ("R2_chaotic_minus_periodic", FIELDS, "periodic")):
        literal[name], detail[name] = {}, {}
        for field in fields:
            result = row_contrast(alpha_cells, field, comparison, rng, draws)
            literal[name][field] = result["summary"]
            detail[name][field] = result
    xo_result = xo_statistics(xo, selection, rng, draws)
    literal["XO"], detail["XO"] = xo_result["summary"], xo_result
    literal["chaotic_share"], detail["chaotic_share"] = {}, {}
    for cells, label in ((alpha_cells, "residual steering, block 12 (alpha map)"),
                         (bias_cells, "logit bias only (beta map)")):
        flags = np.array([c["chaotic"] for c in cells], dtype=bool)
        t = np.array([c["log10_temperature"] for c in cells], dtype=float)
        literal["chaotic_share"][label] = {"share": float(flags.mean()), "cells": len(cells), "chaotic": int(flags.sum())}
        detail["chaotic_share"][label] = [
            {"rounded_temperature": float(row), "indices": np.flatnonzero(np.isclose(t, row)).tolist(),
             "chaotic": int(flags[np.isclose(t, row)].sum()), "cells": int(np.isclose(t, row).sum()),
             "share": float(flags[np.isclose(t, row)].mean())} for row in np.unique(np.round(t, 6))]
    b = literal["chaotic_share"]["logit bias only (beta map)"]
    return {"literal_producer_shape": literal, "details": detail,
            "R4": {"chaotic_count": b["chaotic"], "cells": b["cells"], "share": b["share"],
                   "count_at_least_20": b["chaotic"] >= 20, "share_at_least_half_percent": b["chaotic"] * 200 >= b["cells"],
                   "criterion_pass": b["chaotic"] >= 20 and b["chaotic"] * 200 >= b["cells"]},
            "rng_final": rng.bit_generator.state,
            "scope": "R1 map analysis differs from original line-design wording; original XO only; no later replication enters RNG stream"}
