"""Independent saved-scalar DSS checks. No producer imports or native model calls.

This core is preparation only: admission, historical field availability and the
six-run identity map belong to the subsequently frozen driver/card.
"""
from itertools import combinations
import math
import numpy as np


SCALARS = ("mu", "mu_imag", "next_modulus", "product", "newton", "min_reference_cosine")
FLAGS = ("ok", "clean", "same_tokens")
TRANSPORT = ("factors_transported", "stretchings", "transport_cosine")


def plain(x):
    """JSON-safe, with nonfinite values visible rather than dropped/nullified."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return {"nonfinite": "nan" if math.isnan(x) else ("+inf" if x > 0 else "-inf")}
    return x


def delta(own, saved):
    a, b = np.asarray(own, dtype=np.float64), np.asarray(saved, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("comparison shapes differ")
    with np.errstate(all="ignore"):
        d = a - b
    return {"own": a, "saved": b, "signed": d, "absolute": np.abs(d),
            "both_finite": np.isfinite(a) & np.isfinite(b),
            "exact_equal_including_nan": (a == b) | (np.isnan(a) & np.isnan(b))}


def stats(values, mask=None):
    a = np.asarray(values, dtype=np.float64)
    m = np.ones(a.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if a.shape != m.shape:
        raise ValueError("statistics mask shape")
    f = m & np.isfinite(a)
    out = {"selected": int(m.sum()), "finite": int(f.sum()),
           "nan": int((m & np.isnan(a)).sum()), "positive_infinity": int((m & np.isposinf(a)).sum()),
           "negative_infinity": int((m & np.isneginf(a)).sum()), "exact_zero": int((m & (a == 0)).sum())}
    if f.any():
        v = a[f]
        lo, hi = float(v.min()), float(v.max())
        out.update(minimum=lo, maximum=hi, minimum_indices=np.argwhere(f & (a == lo)),
                   maximum_indices=np.argwhere(f & (a == hi)),
                   quantiles_0_10_50_90_100=np.quantile(v, [0, .1, .5, .9, 1], method="linear"))
    else:
        out.update(minimum=None, maximum=None, minimum_indices=[], maximum_indices=[],
                   quantiles_0_10_50_90_100=None)
    return out


def validate_grid(g, period):
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("positive integer period required")
    for key in ("alpha", "log10_temperature"):
        a = np.asarray(g[key])
        if a.dtype.kind != "f" or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not (np.diff(a) > 0).all():
            raise ValueError("invalid exact axis " + key)
    shape = (len(g["log10_temperature"]), len(g["alpha"]))
    for key in SCALARS + FLAGS + ("factors",):
        a = np.asarray(g[key])
        expected = shape + (period,) if key == "factors" else shape
        if a.shape != expected or (a.dtype.kind != "b" if key in FLAGS else a.dtype.kind != "f"):
            raise ValueError("field shape/dtype " + key)
    present = [key in g for key in TRANSPORT]
    if any(present) and not all(present):
        raise ValueError("incomplete transported-field bundle")
    if all(present):
        for key in TRANSPORT:
            a = np.asarray(g[key])
            expected = shape if key == "transport_cosine" else shape + (period,)
            if a.shape != expected or a.dtype.kind != "f":
                raise ValueError("transport shape/dtype " + key)
    if "multipliers" in g and (np.asarray(g["multipliers"]).shape != shape + (3, 2)
                              or np.asarray(g["multipliers"]).dtype.kind != "f"):
        raise ValueError("multiplier shape/dtype")
    return shape


def masks(g):
    """Existing thresholds only; named joint views are diagnostics, not new gates."""
    mu, im, nxt = (np.asarray(g[k]) for k in ("mu", "mu_imag", "next_modulus"))
    finite = np.isfinite(mu) & np.isfinite(im) & np.isfinite(nxt)
    with np.errstate(all="ignore"):
        mod = np.hypot(mu, im)
        clean = mod > 10 * nxt
    # The literal zoom's negated > test admits NaN imaginary parts. Preserve it.
    result = {"plot_coarse_ok": g["ok"].copy(), "plot_zoom_not_complex": ~(np.abs(im) > 1e-10),
              "finite_spectral": finite, "source_clean_saved": g["clean"].copy(),
              "source_clean_reconstructed_finite": finite & clean,
              "exact_real_finite": finite & (im == 0),
              "seed_real_threshold_finite": finite & (np.abs(im) <= 1e-8),
              "plot_real_threshold_finite": finite & (np.abs(im) <= 1e-10),
              "same_tokens_saved": g["same_tokens"].copy(), "modulus_below_one_finite": finite & (mod < 1)}
    for label in ("seed", "plot"):
        base = g["ok"] & g["clean"] & result[label + "_real_threshold_finite"]
        result["ok_clean_" + label + "_real"] = base
        result["ok_clean_" + label + "_real_same_tokens"] = base & g["same_tokens"]
        if all(k in g for k in TRANSPORT):
            carry = np.isfinite(g["transport_cosine"]) & (g["transport_cosine"] > .3)
            result["transport_" + label + "_real_carry_above_point3"] = base & carry
            result["transport_" + label + "_real_carry_above_point3_same_tokens"] = base & carry & g["same_tokens"]
    return result


def cell_rows(g, period):
    shape = validate_grid(g, period)
    views = masks(g)
    rows = []
    for i, j in np.ndindex(shape):
        saved = {key: np.asarray(g[key])[i, j] for key in SCALARS + FLAGS + ("factors",)}
        with np.errstate(all="ignore"):
            product = np.prod(saved["factors"], dtype=np.float64)
            modulus = np.hypot(saved["mu"], saved["mu_imag"])
        row = {"i": i, "j": j, "alpha": float(g["alpha"][j]),
               "log10_temperature": float(g["log10_temperature"][i]), "saved": saved,
               "masks": {name: bool(a[i, j]) for name, a in views.items()}, "leading_modulus": modulus,
               "fixed_product_vs_saved_product": delta(product, saved["product"]),
               "fixed_product_vs_mu_real": delta(product, saved["mu"]),
               "fixed_product_minus_complex_mu": [product - saved["mu"], -saved["mu_imag"]],
               "source_clean_mismatch_on_finite": bool(views["finite_spectral"][i, j]
                   and saved["clean"] != views["source_clean_reconstructed_finite"][i, j])}
        if all(k in g for k in TRANSPORT):
            ct, s, cos = (np.asarray(g[k])[i, j] for k in TRANSPORT)
            with np.errstate(all="ignore"):
                cp, sp = np.prod(ct, dtype=np.float64), np.prod(s, dtype=np.float64)
            saved.update(factors_transported=ct, stretchings=s, transport_cosine=cos)
            row.update(transport_product_vs_mu_real=delta(cp, saved["mu"]),
                       transport_product_minus_complex_mu=[cp-saved["mu"], -saved["mu_imag"]],
                       phase_absolute_factor_vs_stretching=delta(np.abs(ct), s),
                       absolute_product_vs_stretching_product=delta(abs(cp), sp),
                       negative_stretching_indices=np.flatnonzero(s < 0),
                       zero_stretching_indices=np.flatnonzero(s == 0))
        if "multipliers" in g:
            saved["multipliers"] = g["multipliers"][i, j]
        rows.append(row)
    return rows


def contours(field, mask, xs, ys):
    """Literal ordered marching-squares rule, plus every quad's disposition."""
    f, m = np.asarray(field, dtype=np.float64), np.asarray(mask, dtype=bool)
    if f.shape != (len(ys), len(xs)) or m.shape != f.shape:
        raise ValueError("contour dimensions")
    segments, quads = [], []
    for i in range(len(ys)-1):
        for j in range(len(xs)-1):
            ij = [(i,j), (i,j+1), (i+1,j+1), (i+1,j)]
            v = np.asarray([f[a,b] for a,b in ij])
            p = np.asarray([[xs[j],ys[i]], [xs[j+1],ys[i]], [xs[j+1],ys[i+1]], [xs[j],ys[i+1]]], dtype=float)
            eligible = all(m[a,b] for a,b in ij)
            q = {"i":i, "j":j, "eligible":eligible, "values":v, "finite":bool(np.isfinite(v).all()),
                 "zero_corner_indices":np.flatnonzero(v == 0), "cuts":[], "segment_indices":[]}
            if eligible:
                sign = np.sign(v)
                q["literal_checkerboard"] = bool(sign[0] == sign[2] and sign[1] == sign[3] and sign[0] != sign[1])
                q["finite_nonzero_checkerboard"] = bool(q["literal_checkerboard"] and q["finite"] and (v != 0).all())
                q["opposite_sign_underflow_edges"] = []
                with np.errstate(all="ignore"):
                    for edge in range(4):
                        a,b = v[edge],v[(edge+1)%4]
                        if a != 0 and b != 0 and np.isfinite(a) and np.isfinite(b) and np.signbit(a) != np.signbit(b) and a*b == 0:
                            q["opposite_sign_underflow_edges"].append(edge)
                        if a == 0 or a*b < 0:
                            t = a/(a-b) if a != b else 0.
                            q["cuts"].append({"edge":edge, "t":t, "point":p[edge]+t*(p[(edge+1)%4]-p[edge])})
                    criterion = np.mean(v)*v[0]
                q["mean_times_first"] = criterion
                n = len(q["cuts"])
                pairs = [(0,1)] if n == 2 else ([(0,1),(2,3)] if criterion > 0 else [(0,3),(1,2)]) if n == 4 else []
                q["cut_pairs"] = pairs
                for a,b in pairs:
                    q["segment_indices"].append(len(segments))
                    segments.append({"quad":[i,j], "cut_pair":[a,b],
                                     "points":[q["cuts"][a]["point"],q["cuts"][b]["point"]]})
            else:
                q.update(literal_checkerboard=False, finite_nonzero_checkerboard=False, cut_pairs=[])
            quads.append(q)
    return {"segments":segments,"quads":quads,
            "literal_checkerboards":sum(q["literal_checkerboard"] for q in quads),
            "finite_nonzero_checkerboards":sum(q["finite_nonzero_checkerboard"] for q in quads)}


def intersections(first, second):
    """Source-formula 1e-30 rule after geometric pruning; inclusive endpoints.

    Distinct geometrically disjoint bounding boxes cannot intersect. Parallel
    pairs are counted separately from descriptive EXACT collinear overlaps;
    those overlaps do not add to the own-method intersection count. The
    geometric pruning need not be bit-exact to an unfiltered floating-point
    producer loop. This is a finite interpolation diagnostic, not a zero proof.
    """
    a = np.asarray([x["points"] for x in first], dtype=float).reshape(-1,2,2)
    b = np.asarray([x["points"] for x in second], dtype=float).reshape(-1,2,2)
    af, bf = np.isfinite(a).all(axis=(1,2)), np.isfinite(b).all(axis=(1,2))
    out = {"method":"source-formula arithmetic with geometric bbox pruning; not a bit-exact replay of the unfiltered producer helper",
           "pairs":len(a)*len(b), "nonfinite_segment_ids_a":np.flatnonzero(~af),
           "nonfinite_segment_ids_b":np.flatnonzero(~bf), "parallel_cutoff_pairs":0,
           "bbox_disjoint_finite_pairs":0, "intersections":[], "collinear_overlaps":[],
           "near_parallel_candidate_pairs":[], "nonfinite_arithmetic_candidates":[]}
    for ia in np.flatnonzero(af):
        p1,p2 = a[ia]; d=p2-p1
        ids = np.flatnonzero(bf)
        if not len(ids):
            continue
        q=b[ids]; e=q[:,1]-q[:,0]; w=q[:,0]-p1
        with np.errstate(all="ignore"):
            den=d[0]*e[:,1]-d[1]*e[:,0]
            wn=w[:,0]*d[1]-w[:,1]*d[0]
        parallel=np.abs(den)<1e-30
        overlap=np.all(np.maximum(a[ia].min(axis=0),q.min(axis=1)) <= np.minimum(a[ia].max(axis=0),q.max(axis=1)),axis=1)
        out["parallel_cutoff_pairs"] += int(parallel.sum())
        out["bbox_disjoint_finite_pairs"] += int((~overlap).sum())
        for k in np.flatnonzero(overlap):
            ib=int(ids[k]); pair=[int(ia),ib]
            if parallel[k]:
                if den[k] == 0 and wn[k] == 0:
                    # Degenerate point segments need equality/on-line checks.
                    dz=bool((d == 0).all()); ez=bool((e[k] == 0).all())
                    if dz and ez:
                        collinear=bool((p1 == q[k,0]).all())
                    elif dz:
                        collinear=bool((p1[0]-q[k,0,0])*e[k,1]-(p1[1]-q[k,0,1])*e[k,0] == 0)
                    else:
                        collinear=True
                    if collinear:
                        out["collinear_overlaps"].append({"pair":pair,"a":a[ia],"b":q[k],"degenerate":dz or ez})
                else:
                    out["near_parallel_candidate_pairs"].append({"pair":pair,"denominator":den[k],"collinearity_cross":wn[k]})
                continue
            with np.errstate(all="ignore"):
                t=(w[k,0]*e[k,1]-w[k,1]*e[k,0])/den[k]
                u=wn[k]/den[k]
            if not np.isfinite([den[k],t,u]).all():
                out["nonfinite_arithmetic_candidates"].append({"pair":pair,"denominator":den[k],"t":t,"u":u})
            if 0 <= t <= 1 and 0 <= u <= 1:
                out["intersections"].append({"pair":pair,"denominator":den[k],"t":t,"u":u,"point":p1+t*d})
    return out


def factor_geometry(g, field, mask):
    """Within one original grid/gauge; all distinct phase pairs, no stitching."""
    if field not in g:
        return {"available":False,"field":field}
    phase=[contours(g[field][...,k],mask,g["alpha"],g["log10_temperature"])
           for k in range(g[field].shape[-1])]
    return {"available":True,"field":field,"phases":phase,
            "phase_pairs":[{"phases":[a,b],"crossings":intersections(phase[a]["segments"],phase[b]["segments"])}
                           for a,b in combinations(range(len(phase)),2)],
            "interpretation":"finite edge-linear interpolation; not a native zero/crossing certificate"}


def progress_coverage(g, entries):
    """Keep BFS progress distinct from NPZ snapshots; never fill unwritten cells.

    The writer's failed/unvisited cells both retain initialized arrays. A
    processed tail can postdate the latest snapshot; receipt/mtime context is
    inherited. Exact scalar equality is a comparison, not a timing inference.
    """
    shape=np.asarray(g["mu"]).shape
    occurrences={}; records=[]; noncell=[]
    aliases={k:k for k in SCALARS+FLAGS+("factors",)+TRANSPORT if k != "ok"}
    for seq,entry in enumerate(entries):
        if "i" not in entry or "j" not in entry:
            noncell.append({"sequence":seq,"record":entry})
            continue
        i,j=entry["i"],entry["j"]
        valid=all(isinstance(v,int) and not isinstance(v,bool) for v in (i,j)) and 0<=i<shape[0] and 0<=j<shape[1]
        row={"sequence":seq,"record":entry,"valid_index":valid}
        if not valid:
            records.append(row)
            continue
        occurrences.setdefault((i,j),[]).append(seq)
        row["coordinate_comparisons"]={k:delta(entry[k],value) for k,value in
                                       (("alpha",g["alpha"][j]),("log10_temperature",g["log10_temperature"][i])) if k in entry}
        row["fields_missing_from_progress"]=[k for k in aliases if k in g and k not in entry] if entry.get("converged") is True else []
        row["saved_field_comparisons"]={k:delta(entry[k],g[k][i,j]) for k in aliases if k in entry and k in g}
        row["source_split_ok_comparison"] = (delta(float(entry["split"] > 1e-7),float(g["ok"][i,j]))
                                              if "split" in entry else None)
        records.append(row)
    cells=[]
    numeric=[k for k in SCALARS+("factors",)+TRANSPORT+("multipliers",) if k in g]
    for i,j in np.ndindex(shape):
        # The writer initializes numeric payloads to NaN and flags to False.
        payload=any(not np.isnan(np.asarray(g[k][i,j],dtype=float)).all() for k in numeric)
        flag_payload=any(bool(g[k][i,j]) for k in FLAGS)
        cells.append({"i":i,"j":j,"progress_sequences":occurrences.get((i,j),[]),
                      "numeric_payload_differs_from_initial_nan":payload,"any_saved_true_flag":flag_payload,
                      "snapshot_failed_vs_unvisited_unresolved":not payload and not flag_payload})
    return {"all_progress_records":records,"noncell_records":noncell,"cells":cells,
            "duplicate_progress_indices":[{"i":i,"j":j,"sequences":seq} for (i,j),seq in occurrences.items() if len(seq)>1],
            "invalid_index_sequences":[r["sequence"] for r in records if not r["valid_index"]],
            "no_snapshot_reconstruction":True}


def exact_overlaps(named_grids, fields):
    """Literal coordinate identity only; factor differences do not align gauges."""
    seen, rows = {}, []
    for name,g in named_grids:
        for i,t in enumerate(g["log10_temperature"]):
            for j,a in enumerate(g["alpha"]):
                key=(float(t).hex(),float(a).hex())
                for old,oi,oj,og in seen.get(key,[]):
                    rows.append({"coordinate_hex":key,"first":[old,oi,oj],"second":[name,i,j],
                                 "fields":{k:delta(g[k][i,j],og[k][oi,oj]) for k in fields if k in g and k in og},
                                 "missing_from_either":[k for k in fields if k not in g or k not in og]})
                seen.setdefault(key,[]).append((name,i,j,g))
    return rows


def plot_zoom(named_grids):
    """Reproduce declared band order and literal drop-first-row operation."""
    if not named_grids:
        raise ValueError("no zoom bands")
    first=named_grids[0][1]; pieces={k:[] for k in ("mu","mu_imag")}
    ys,ledger,joins=[],[],[]
    for n,(name,g) in enumerate(named_grids):
        if len(g["alpha"]) != len(first["alpha"]):
            raise ValueError("literal concatenation column count differs")
        start=0 if n == 0 else 1
        if n:
            prior=named_grids[n-1][1]
            joins.append({"previous":named_grids[n-1][0],"next":name,
                          "alpha_vs_first":delta(g["alpha"],first["alpha"]),
                          "dropped_temperature_vs_previous_last":delta(g["log10_temperature"][0],prior["log10_temperature"][-1]),
                          "dropped_row_fields":{k:delta(g[k][0],prior[k][-1]) for k in SCALARS+FLAGS+("factors",)+TRANSPORT if k in g and k in prior},
                          "factor_gauge_alignment":"none"})
        ys.extend(g["log10_temperature"][start:])
        for k in pieces:
            pieces[k].append(g[k][start:])
        ledger.extend({"band":name,"source_row":i,"temperature":g["log10_temperature"][i],
                       "source_alpha":g["alpha"]} for i in range(start,len(g["log10_temperature"])))
    joined={k:np.concatenate(v) for k,v in pieces.items()}
    y=np.asarray(ys); mod=np.hypot(joined["mu"],joined["mu_imag"])
    cplx=np.abs(joined["mu_imag"])>1e-10
    return {"alpha":first["alpha"],"log10_temperature":y,"row_ledger":ledger,"joins":joins,
            "strictly_increasing_joined_axis":bool((np.diff(y)>0).all()),
            "literal_zoom_cells":int(mod.size),"literal_zoom_complex":int(cplx.sum()),
            "literal_zoom_min_modulus":float(mod.min()),"finite_modulus_summary":stats(mod),
            "literal_contours":contours(np.where(~cplx,joined["mu"],np.nan),~cplx,first["alpha"],y)}
