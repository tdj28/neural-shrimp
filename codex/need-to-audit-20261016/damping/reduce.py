"""Independent saved arithmetic for NEED_TO_AUDIT items 3–5; no model imports."""
import math
import numpy as np


def finite_summary(a):
    a = np.asarray(a, dtype=float)
    v = a[np.isfinite(a)]
    return {"shape": list(a.shape), "count": int(a.size), "finite": int(v.size),
            "nan": int(np.isnan(a).sum()), "positive_infinity": int(np.isposinf(a).sum()),
            "negative_infinity": int(np.isneginf(a).sum()),
            "finite_min": float(v.min()) if v.size else None,
            "finite_max": float(v.max()) if v.size else None}


def median_row(values, indices):
    a = np.asarray(values, dtype=float)[indices]
    return {"indices": list(map(int, indices)), "count": len(a),
            "literal_median": float(np.median(a)) if len(a) else None,
            "finite_summary": finite_summary(a), "qualified": bool(len(a) and np.isfinite(a).all())}


def damping(doc, arrays, model_id):
    cells = doc["cells"]
    n, L = len(cells), int(doc["layers"])
    H = np.array([c["mean_entropy"] for c in cells], float)
    g = np.array([c["mean_log_g_net"] for c in cells], float)
    A, N = np.asarray(arrays["mean_A"]), np.asarray(arrays["mean_N"])
    if A.shape != (L, n) or N.shape != (L, n):
        raise ValueError("block-input mean array shape")
    if list(map(float, arrays["log10_temperature"])) != [float(c["log10_temperature"]) for c in cells]:
        raise ValueError("layer/cell temperature association")
    near = int(np.argmin(np.abs(H - 1.0))) if np.isfinite(H).all() and n else None
    bins = [median_row(g, np.flatnonzero((H >= a) & (H < b))) for a, b in ((0, 1), (1, 3), (3, 6))]
    main = median_row(g, np.flatnonzero((H >= .3) & (H < 6)))
    high = median_row(g, np.flatnonzero(H > 10))
    row_checks = []
    for i, c in enumerate(cells):
        split = c["mean_stack"] - c["mean_dilution"] + c["mean_final_norm"]
        le = c["mean_log_g_net"] + c["mean_log_g_read"] + c["boundary_term"]
        cl = c["period"] == 0 and c["lyapunov"] > 1e-3
        share = c["mean_c_x"] + float(np.sum(arrays["mean_c_attn"][:, i])) + float(np.sum(arrays["mean_c_mlp"][:, i]))
        row_checks.append({"index": i, "source_cell": c, "recomputed_chaotic": cl,
                           "classification_matches": cl == c["chaotic"],
                           "network_split_signed_difference": split - c["mean_log_g_net"],
                           "finite_twin_identity_signed_difference": le - c["lyapunov"],
                           "sublayer_share_sum_minus_one": share - 1,
                           "sublayer_quantities_qualified": not model_id.startswith("olmo2-"),
                           "true_final_residual_relative_change": c["mean_stack"] - c["mean_dilution"],
                           "last_saved_block_input_relative_change": float(A[-1, i] - N[-1, i])})
    max_closure = float(np.max([c["max_closure"] for c in cells]))
    profile = None if near is None else {
        "index": near, "entropy": float(H[near]), "A_block_inputs": A[:, near].tolist(),
        "N_block_inputs": N[:, near].tolist(), "first_block_gap": float(A[1, near] - N[1, near]),
        "literal_plot_depth_deficit": float((N[-1, near]-N[2, near])-(A[-1, near]-A[2, near])),
        "literal_plot_final_relative_change": float(A[-1, near]-N[-1, near]),
        "true_final_residual_relative_change": cells[near]["mean_stack"]-cells[near]["mean_dilution"],
        "last_saved_index_is_before_final_block": True}
    ko_rows = []
    if "knockouts" in doc:
        ko = doc["knockouts"]
        indices = ko["cells"]
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError("knockout indices")
        if int(ko["steps"]) <= 0:
            raise ValueError("knockout sample count")
        aa, mm = np.array(ko["delta_attn"], float), np.array(ko["delta_mlp"], float)
        if aa.shape != (L, len(indices)) or mm.shape != aa.shape or len(ko["plain_twin_check"]) != len(indices):
            raise ValueError("knockout shape")
        for j, i in enumerate(indices):
            both = np.concatenate([aa[:, j], mm[:, j]])
            valid = bool(np.isfinite(both).all())
            ko_rows.append({"index": i, "entropy": float(H[i]), "log10_temperature": cells[i]["log10_temperature"],
                            "delta_attn": aa[:, j].tolist(), "delta_mlp": mm[:, j].tolist(),
                            "plain_twin_check": ko["plain_twin_check"][j], "steps": ko["steps"],
                            "qualified": valid, "maximum_factor_rise_on_removal": float((-both).max()) if valid else None,
                            "maximum_factor_drop_on_removal": float(both.max()) if valid else None,
                            "source_sign": "delta = free log gain minus knockout log gain"})
    ko_near = min(range(len(ko_rows)), key=lambda j: abs(ko_rows[j]["entropy"]-1)) if ko_rows and all(math.isfinite(r["entropy"]) for r in ko_rows) else None
    routing = None
    if doc.get("routers"):
        med = np.array([c["margin_median"] for c in cells], float)
        mn = np.array([c["margin_min_over_steps"] for c in cells], float)
        dif = np.array([c["mean_log_g_net_frozen"]-c["mean_log_g_net"] for c in cells])
        switches = np.array([c["mean_switches"] for c in cells])
        routing = {"routers": doc["routers"], "median_gap_range": [float(med.min()), float(med.max())],
                   "min_gap_range": [float(mn.min()), float(mn.max())], "signed_frozen_minus_free": dif.tolist(),
                   "max_frozen_minus_free": float(np.max(np.abs(dif))), "total_mean_switches": float(switches.sum()),
                   "switch_means": switches.tolist(), "all_zero_means": bool(np.all(switches == 0)),
                   "all_means_nonnegative_finite": bool(np.isfinite(switches).all() and (switches >= 0).all()),
                   "margin_aggregates_not_full_step_router_records": True}
    projection = {"layers": L, "hidden": doc["hidden"], "median_g_net_0.3_6": main["literal_median"],
                  "bin_medians": {k: b["literal_median"] for k, b in zip(("0-1", "1-3", "3-6"), bins)}, "chaotic_temperatures": sum(c["chaotic"] for c in cells),
                  "profile_entropy": None if profile is None else profile["entropy"],
                  "first_block_gap": None if profile is None else profile["first_block_gap"],
                  "depth_deficit": None if profile is None else profile["literal_plot_depth_deficit"],
                  "final_relative_change": None if profile is None else profile["literal_plot_final_relative_change"],
                  "max_closure": max_closure, "decomposition_valid": bool(max_closure < 1e-5)}
    return {"args": doc["args"], "rows": row_checks, "main_bin": main, "bins": bins, "above_10": high,
            "profile": profile, "knockouts": ko_rows, "nearest_one_nat_knockout_row": ko_near,
            "nearest_grid_cell_has_knockout": near is not None and any(r["index"] == near for r in ko_rows),
            "routing": routing, "projection": projection,
            "array_coverage": {k: finite_summary(v) for k, v in arrays.items()},
            "scope": "saved full-record means and first keep_steps float32 layer samples; no native replay; OLMo2 sublayer shares/gains/parallel coefficients excluded"}


def scan(arrays, max_period):
    le, p = np.asarray(arrays["lyapunov"]), np.asarray(arrays["period"])
    if le.shape != p.shape:
        raise ValueError("scan paired shapes")
    finite = np.isfinite(le) & np.isfinite(p)
    valid_p = np.isfinite(p) & (p >= 0) & (p <= max_period) & (p == np.floor(p))
    valid = finite & valid_p
    chaotic = (p == 0) & (le > .001)
    rows = [{"flat_index": i, "lyapunov": float(x), "period": float(q),
             "literal_chaotic": bool(c), "qualified": bool(v),
             "period_positive_exponent": bool(q > 0 and x > .001),
             "no_period_not_above_threshold": bool(q == 0 and x <= .001)}
            for i, (x, q, c, v) in enumerate(zip(le.flat, p.flat, chaotic.flat, valid.flat))]
    return {"shape": list(le.shape), "rows": rows, "total": int(le.size),
            "literal_count": int(chaotic.sum()), "literal_max_exponent": float(le.max()),
            "qualified_count": int((chaotic & valid).sum()), "qualified_denominator": int(valid.sum()),
            "invalid_period": int((~valid_p).sum()), "nonfinite": int((~finite).sum()),
            "period_positive_exponent": sum(r["period_positive_exponent"] for r in rows),
            "no_period_not_above_threshold": sum(r["no_period_not_above_threshold"] for r in rows)}


def distance_interval(rounded_log):
    r = float(rounded_log)
    if not math.isfinite(r):
        return {"qualified": False, "rounded_log10": r}
    lo = 0.0 if r == -300 else math.nextafter(10.0 ** (r - .005), -math.inf)
    hi = math.nextafter(10.0 ** (r + .005), math.inf)
    return {"qualified": True, "rounded_log10": r, "display_value": 10.0 ** r,
            "possible_distance_lower": lo, "possible_distance_upper": hi,
            "rounding_scope": "source rounds log10(max(distance,1e-300)) to 2dp; closed conservative ±.005 interval"}


def text_cell(args, c):
    steps = c["steps"]
    if len(steps) != args["record"]:
        raise ValueError("text record length")
    rows = []
    for i, s in enumerate(steps):
        top = s["top"]
        if len(top) != 3 or any(type(t["token"]) is not int for t in top):
            raise ValueError("top-three schema")
        probs = [float(t["p"]) for t in top]
        q = (all(math.isfinite(p) and 0 <= p <= 1 for p in probs) and probs[0] >= probs[1] >= probs[2]
             and sum(probs) <= 1 and len({t["token"] for t in top}) == 3)
        rows.append({"step": i+1, "top": top, "twin_text": s["twin"], "probabilities_qualified": q,
                     "leading_probability": probs[0], "top_two_lead": probs[0]-probs[1],
                     "text_differs": top[0]["text"] != s["twin"],
                     "twin_matches_runner_up_text": top[1]["text"] == s["twin"],
                     "distance": distance_interval(s["log10_distance"])})
    first = next((r["step"] for r in rows if r["text_differs"]), None)
    split = c["split_step"]
    if split is not None and (type(split) is not int or not 1 <= split <= len(rows)):
        raise ValueError("split outside saved record")
    split_row = rows[split-1] if split is not None else None
    le = c["lyapunov"]
    if len(le) != 2:
        raise ValueError("two text twin exponents required")
    literal_chaotic = c["period"] == 0 and all(x > .001 for x in le)
    lam = sum(le)/len(le)
    first_above = next((r["step"] for r in rows if r["distance"]["rounded_log10"] > -2), None)
    return {"alpha": c["alpha"], "log10T": c["log10T"], "period": c["period"], "lyapunov": le,
            "mean_lyapunov": lam, "literal_chaotic": literal_chaotic, "saved_chaotic": c["chaotic"],
            "classification_matches": literal_chaotic == c["chaotic"],
            "native_label_qualified": all(math.isfinite(x) for x in le) and type(c["period"]) is int and 0 <= c["period"] <= 64,
            "saved_id_based_split": split, "first_saved_text_difference": first,
            "saved_split_equals_text_split": split == first,
            "split_row": split_row, "all_steps": rows,
            "base_concatenation_matches": "".join(s["top"][0]["text"] for s in steps) == c["text"],
            "twin_concatenation_matches": "".join(s["twin"] for s in steps) == c["twin_text"],
            "maximum_top_probability": max(r["leading_probability"] for r in rows),
            "first_rounded_distance_above_0.01": first_above,
            "steps_split_after_rounded_0.01_crossing": split-first_above if split is not None and first_above is not None else None,
            "native_twin_token_ids_saved": False,
            "scope": "decoded-text comparisons do not independently prove ID equality; no arbitrary-precision or infinite-time claim"}


def thread_comparison(left, right):
    a, b = left["steps"], right["steps"]
    n = min(len(a), len(b))
    rows = [{"step": i+1, "left_token": a[i]["top"][0]["token"], "right_token": b[i]["top"][0]["token"],
             "same_token": a[i]["top"][0]["token"] == b[i]["top"][0]["token"],
             "same_text": a[i]["top"][0]["text"] == b[i]["top"][0]["text"],
             "leading_probability_signed_difference": b[i]["top"][0]["p"] - a[i]["top"][0]["p"]} for i in range(n)]
    return {"common_steps": n, "rows": rows,
            "first_token_difference": next((r["step"] for r in rows if not r["same_token"]), None),
            "first_text_difference": next((r["step"] for r in rows if not r["same_text"]), None)}


def vocabulary(doc):
    claimed = [" I", "The", "!", " It", " infinite", " possibilities", " are", " you"]
    rows = []
    for i, c in enumerate(doc["cells"]):
        tokens, texts = c["lead_tokens"], c["lead_text"]
        if len(tokens) != len(texts) or len(tokens) > 8:
            raise ValueError("70B leading vocabulary schema")
        finite = all(math.isfinite(x) for x in c["lyapunov"])
        rows.append({"index": i, "saved_cell": c, "saved_first_eight_match_paper": texts == claimed,
                     "stored_token_ids_unique": len(tokens) == len(set(tokens)),
                     "stored_token_count_compatible_with_distinct_summary": len(tokens) == min(8, c["lead_distinct"]),
                     "three_offset_positive_no_period": len(c["lyapunov"]) == 3 and finite and c["period"] == 0 and all(x > .001 for x in c["lyapunov"]),
                     "tangent_positive": math.isfinite(c["lyapunov_tangent"]) and c["lyapunov_tangent"] > .001})
    return {"args": doc["args"], "rows": rows, "paper_first_eight": claimed,
            "full_leading_sequence_saved": False, "distinct_count_recomputed": False,
            "scope": "first eight saved IDs/texts and inherited distinct-count summary only; no tokenizer or 300-step vocabulary replay"}


def comparisons(own, published, path=""):
    """All published leaves retained; extra own evidence does not create false missing-source discrepancies."""
    if isinstance(published, dict):
        out = []
        for k, v in published.items():
            p = path + "/" + str(k)
            if not isinstance(own, dict) or k not in own:
                out.append({"path": p, "status": "missing_own", "published": v})
            else:
                out.extend(comparisons(own[k], v, p))
        return out
    if isinstance(published, list):
        if not isinstance(own, list) or len(own) != len(published):
            return [{"path": path, "status": "shape_difference", "own": own, "published": published}]
        return [r for i, (a, b) in enumerate(zip(own, published)) for r in comparisons(a, b, path+"/"+str(i))]
    row = {"path": path, "own": own, "published": published, "status": "exact" if own == published else "different"}
    if isinstance(own, (int, float)) and not isinstance(own, bool) and isinstance(published, (int, float)) and not isinstance(published, bool):
        row.update(signed_difference=own-published, absolute_difference=abs(own-published))
    return [row]
