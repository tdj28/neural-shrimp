"""Independent CP saved-scalar score and rounded-profile compatibility. No I/O."""
import math
import numpy as np

FROZEN = (
    ("qwen0.5b", "Qwen2.5", "0.5B", .49, .018),
    ("qwen1.5b", "Qwen2.5", "1.5B", 1.54, .231),
    ("qwen3b", "Qwen2.5", "3B", 3.09, .030),
    ("qwen7b", "Qwen2.5", "7B", 7.62, .165),
    ("qwen14b", "Qwen2.5", "14B", 14.77, .050),
    ("qwen32b", "Qwen2.5", "32B", 32.76, .010),
    ("qwen72b", "Qwen2.5", "72B", 72.71, .000),
    ("llama3.1-8b", "Llama-3.1", "8B", 8.03, .244),
    ("llama3.1-70b", "Llama-3.1", "70B", 70.55, .031),
    ("llama3.3-70b", "Llama-3.3", "70B", 70.55, .000),
    ("olmo2-1b", "OLMo-2", "1B", 1.48, .050),
    ("olmo2-7b", "OLMo-2", "7B", 7.30, .081),
    ("olmo2-13b", "OLMo-2", "13B", 13.72, .000),
    ("olmo2-32b", "OLMo-2", "32B", 32.23, .252),
)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def ranks(values):
    """Stable sorting with average ranks for exact represented-value ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranked = [None] * len(values)
    ties = []
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        rank = (left + 1 + right) / 2
        for i in order[left:right]:
            ranked[i] = rank
        if right - left > 1:
            ties.append(order[left:right])
        left = right
    return ranked, order, ties


def spearman(x, y):
    """Own average-rank correlation; source-compatible asymptotic t approximation.

    No row filtering, no significance gate, and no exact/randomization p claim.
    The library special function evaluates a t CDF, not producer analysis.
    """
    if len(x) != len(y):
        raise ValueError("paired vectors must have equal length")
    n = len(x)
    out = {"x": list(x), "y": list(y), "n": n, "rho": None, "p": None,
           "p_method": "two-sided Student-t approximation with n-2 df, same convention as source scipy.stats.spearmanr",
           "status": "undefined", "nonfinite_indices": [i for i in range(n) if not finite(x[i]) or not finite(y[i])]}
    if n < 2 or out["nonfinite_indices"]:
        out["reason"] = "fewer than two rows or nonfinite input; no rows omitted"
        return out
    rx, ox, tx = ranks(x)
    ry, oy, ty = ranks(y)
    centre = (n + 1) / 2
    dx, dy = [a-centre for a in rx], [b-centre for b in ry]
    xx, yy = math.fsum(a*a for a in dx), math.fsum(b*b for b in dy)
    out.update(rank_x=rx, rank_y=ry, permutation_x=ox, permutation_y=oy, ties_x=tx, ties_y=ty)
    if not xx or not yy:
        out["reason"] = "constant ranked input"
        return out
    rho = math.fsum(a*b for a, b in zip(dx, dy)) / math.sqrt(xx*yy)
    rho = min(1.0, max(-1.0, rho))
    out.update(rho=rho, status="computed", p_status="undefined for n=2")
    if n > 2:
        if abs(rho) == 1:
            p = 0.0
        else:
            from scipy.special import stdtr
            t = abs(rho) * math.sqrt((n-2) / ((1-rho)*(1+rho)))
            p = float(2 * stdtr(n-2, -t))
        out.update(p=p, p_status="asymptotic approximation")
    return out


def rounded_cells(values):
    """Conservative rounding compatibility, conditional on source binary64 round(5).

    ±0.5e-5 plus an eight-ulp-scale allowance covers small binary multiply/divide
    rounding in this routine; nextafter expands outward. These are saved-number
    compatibility cells, not enclosures of the neural map or its native error.
    """
    result = []
    for value in values:
        if not finite(value):
            result.append(None)
            continue
        allowance = 8 * np.finfo(float).eps * max(1.0, abs(value))
        lo = math.nextafter(value - .5e-5 - allowance, -math.inf)
        hi = math.nextafter(value + .5e-5 + allowance, math.inf)
        result.append([lo, hi])
    return result


def first_crossing_candidates(bounds, threshold):
    candidates = []
    can_still_be_first = True
    for i, (lo, hi) in enumerate(bounds):
        if can_still_be_first and hi > threshold:
            candidates.append(i)
        if lo > threshold:
            can_still_be_first = False
    return {"possible_indices": candidates, "no_crossing_possible": can_still_be_first}


def profile_compatibility(row, log_t, gain_log_t):
    """All 601/81 records retained; rounded reductions never replace source scalars."""
    out = {"prefix_index": row.get("prefix_index"), "status": "unavailable",
           "scope": "Conditional five-decimal saved-profile compatibility only; original full-precision scalar endpoints remain primary."}
    try:
        h, read = row["entropy_profile"], row["log_g_read"]
        t, gt = np.asarray(log_t, float), np.asarray(gain_log_t, float)
        if len(h) != len(t) or len(read) != len(gt) or len(t) < 2 or not len(gt):
            out["reason"] = "profile/axis shape mismatch"
            return out
        out["source_grid_shape"] = {"entropy": len(t), "gain": len(gt), "expected": [601, 81]}
        out["source_grid_exact"] = bool(len(t) == 601 and len(gt) == 81 and np.array_equal(t, np.linspace(-1.5, 1.5, 601)) and np.array_equal(gt, np.linspace(-1, 1, 81)))
        hb, rb = rounded_cells(h), rounded_cells(read)
        out.update(entropy_rounding_cells=hb, readout_rounding_cells=rb)
        if any(b is None for b in hb + rb) or not np.isfinite(t).all() or not np.isfinite(gt).all() or not np.all(np.diff(t) > 0):
            out["reason"] = "nonfinite profile/axis or nonincreasing entropy axis; no invalid rows omitted"
            return out
        weights = np.gradient(t)
        definite = [lo > 1 and hi < 10 for lo, hi in hb]
        possible = [hi > 1 and lo < 10 for lo, hi in hb]
        literal = [v > 1 and v < 10 for v in h]
        lower = math.nextafter(math.fsum(float(w) for w, keep in zip(weights, definite) if keep), -math.inf)
        upper = math.nextafter(math.fsum(float(w) for w, keep in zip(weights, possible) if keep), math.inf)
        saved_breadth = row["breadth_decades"]
        # Original reduction is np.sum; its sum-rounding allowance is reported.
        sum_allowance = float(np.finfo(float).eps * len(weights) * math.fsum(abs(float(w)) for w in weights))
        lower = math.nextafter(lower - sum_allowance, -math.inf)
        upper = math.nextafter(upper + sum_allowance, math.inf)
        direct_breadth = float(np.sum(weights * np.asarray(literal)))
        out["entropy_ledger"] = [{"ordinal": i, "log10T": float(t[i]), "saved_5dp": h[i], "weight": float(weights[i]), "rounding_cell": hb[i], "literal_mid": literal[i], "definite_mid": definite[i], "possible_mid": possible[i]} for i in range(len(t))]
        out["breadth"] = {"saved_scalar": saved_breadth, "rounded_profile_reduction": direct_breadth,
                          "rounded_minus_saved": direct_breadth-saved_breadth if finite(saved_breadth) else None,
                          "conditional_interval": [lower, upper], "sum_rounding_allowance": sum_allowance,
                          "compatible": lower <= saved_breadth <= upper if finite(saved_breadth) else None,
                          "ambiguous_indices": [i for i in range(len(h)) if definite[i] != possible[i]]}
        c1, c10 = first_crossing_candidates(hb, 1), first_crossing_candidates(hb, 10)
        width = row["crossing_width"]
        width_min, width_max, pair_count, matches = None, None, 0, []
        for i in c1["possible_indices"]:
            for j in c10["possible_indices"]:
                if j < i:
                    continue
                value = float(t[j]-t[i])
                pair_count += 1
                width_min = value if width_min is None else min(width_min, value)
                width_max = value if width_max is None else max(width_max, value)
                if finite(width) and value == width:
                    matches.append([i, j])
        undefined_possible = c1["no_crossing_possible"] or c10["no_crossing_possible"]
        out["crossing_width"] = {"saved_scalar": width, "threshold1": c1, "threshold10": c10,
                                 "candidate_pair_count": pair_count, "candidate_width_range": [width_min, width_max],
                                 "matching_saved_width_pairs": matches, "undefined_possible": undefined_possible,
                                 "compatible": bool(matches) if finite(width) else (isinstance(width, float) and math.isnan(width) and undefined_possible),
                                 "scope": "Conservative crossing candidates from rounded cells, no continuous-root interpolation."}
        zero_index = int(np.abs(t).argmin())
        h0 = row["entropy_at_T1"]
        out["entropy_at_T1"] = {"index": zero_index, "saved_scalar": h0, "rounded_value": h[zero_index],
                                "conditional_interval": hb[zero_index], "compatible": hb[zero_index][0] <= h0 <= hb[zero_index][1] if finite(h0) else None}
        net, maximum = row["log_g_net_median"], row["one_step_gain_max"]
        saved_range = row.get("log_g_net_range")
        out["network_range"] = {"saved_range": saved_range, "saved_log_median": net,
                                "ordered_and_contains_median": bool(isinstance(saved_range, list) and len(saved_range) == 2 and all(finite(x) for x in saved_range) and finite(net) and saved_range[0] <= net <= saved_range[1]),
                                "scope": "Saved scalar order only; individual direction gains are unavailable."}
        if finite(net):
            sums = [net + v for v in read]
            bounds = [[math.nextafter(net+lo, -math.inf), math.nextafter(net+hi, math.inf)] for lo, hi in rb]
            max_bounds = [max(b[0] for b in bounds), max(b[1] for b in bounds)]
            candidates = [i for i, b in enumerate(bounds) if b[1] >= max_bounds[0]]
            saved_arg = row["one_step_gain_argmax_log10T"]
            out["gain_ledger"] = [{"ordinal": i, "log10T": float(gt[i]), "saved_log_read_5dp": read[i], "net_plus_rounded_read": sums[i], "conditional_interval": bounds[i]} for i in range(len(gt))]
            out["one_step_max"] = {"saved_scalar": maximum, "rounded_profile_max": max(sums), "rounded_minus_saved": max(sums)-maximum if finite(maximum) else None,
                                   "conditional_interval": max_bounds, "compatible": max_bounds[0] <= maximum <= max_bounds[1] if finite(maximum) else None,
                                   "source_first_argmax_on_rounded_profile": int(np.argmax(sums)),
                                   "saved_argmax_log10T": saved_arg, "possible_argmax_indices": candidates,
                                   "argmax_compatible": any(float(gt[i]) == saved_arg for i in candidates),
                                   "proxy": "log(median directional g_net) + log(median directional g_read), separately; not median(log gain) or a median paired product."}
        else:
            out["one_step_max"] = {"status": "undefined", "reason": "nonfinite network log-median", "saved_scalar": maximum}
        out["status"] = "computed"
    except (KeyError, TypeError, ValueError) as e:
        out["reason"] = str(e)
    return out


def model_row(spec, document, current):
    key, family, model, params, decades = spec
    out = {"model": key, "family": family, "scale_model": model, "params_b": params,
           "decades_frozen": decades, "missing": document is None, "source_refusal": None,
           "raw_document": document, "current_scale_record": current, "decades_current": None}
    if document is None:
        return out
    try:
        pref = document["prefixes"]
        if len(pref) != 10 or pref[0]["prefix_index"] != 0:
            out["source_refusal"] = "source requires ten prefixes with prefix zero first"
            return out
        indices = [p.get("prefix_index") for p in pref]
        out["prefix_identity"] = {"source_order": indices, "exact_0_through_9": indices == list(range(10)),
                                  "unique_0_through_9": sorted(indices) == list(range(10)) if all(isinstance(i, int) for i in indices) else False}
        out.update(breadth_p0=pref[0]["breadth_decades"], breadth_median=float(np.median([p["breadth_decades"] for p in pref])),
                   log_g_net_p0=pref[0]["log_g_net_median"], one_step_max_p0=pref[0]["one_step_gain_max"], entropy_T1_p0=pref[0]["entropy_at_T1"])
        if current is not None and not current.get("missing") and not current.get("source_refusal"):
            out["decades_current"] = current["chaos_decades"]
        out["profiles"] = [profile_compatibility(p, document["log10T"], document["gain_log10T"]) for p in pref]
        out["scored_fields_finite"] = all(finite(out[k]) for k in ("breadth_p0", "breadth_median", "log_g_net_p0", "one_step_max_p0"))
    except (KeyError, TypeError, ValueError) as e:
        out["source_refusal"] = str(e)
    return out


def reduce_cp(documents, current=None):
    """documents keyed by fixed model; current keyed (family, scale_model).

    Current values are explicitly the sensitivity projection, never primary.
    A source-refusing present model prevents a literal global score; it is not
    silently treated as missing. Nonfinite scalars remain in each rank input.
    """
    current = {} if current is None else current
    rows = [model_row(s, documents.get(s[0]), current.get((s[1], s[2]))) for s in FROZEN]
    refusals = [r["model"] for r in rows if r["source_refusal"]]
    out = {"rows": rows, "frozen_outcome_precision": "literal three-decimal values from frozen design",
           "models_present": sum(not r["missing"] for r in rows), "models_missing": [r["model"] for r in rows if r["missing"]],
           "source_refusals": refusals, "score": None,
           "unexpected_model_keys": sorted(set(documents) - {s[0] for s in FROZEN}),
           "scope": "Descriptive cross-model associations; heterogeneous families/training and already-known outcomes. No causal size inference."}
    if refusals:
        return out
    available = [r for r in rows if not r["missing"]]
    def corr(key, subset, outcome):
        values = [math.log(r["params_b"]) if key == "log_params" else r[key] for r in subset]
        result = spearman(values, [r[outcome] for r in subset])
        result["models_in_order"] = [r["model"] for r in subset]
        return result
    score = {"CP1": corr("breadth_p0", available, "decades_frozen"), "CP2": corr("breadth_median", available, "decades_frozen"),
             "CP3": {k: corr(k, available, "decades_frozen") for k in ("log_g_net_p0", "one_step_max_p0", "log_params")}}
    for key in ("CP1", "CP2"):
        rho = score[key]["rho"]
        score[key]["holds"] = rho >= .5 if rho is not None else False
        score[key]["scientific_disposition"] = ("holds" if rho >= .5 else "fails") if rho is not None else "undefined"
    sensitive = [r for r in available if r["decades_current"] is not None]
    score["sensitivity_current_decades"] = {k: corr(k, sensitive, "decades_current") for k in ("breadth_p0", "breadth_median", "log_params")}
    out["score"] = score
    return out
