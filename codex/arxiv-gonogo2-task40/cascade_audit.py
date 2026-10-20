"""Independent saved cascade/table reductions; no model or producer import."""
import io
import json
import math
from pathlib import Path
import time
import numpy as np
from delta_ditch_audit import read, PINS, OUT, BASE

DATA = BASE / "pod64"
DELTA = 4.669201609102990


def js(path):
    return json.loads(read(path))


def ratio(lams):
    return [(lams[i]-lams[i-1])/(lams[i+1]-lams[i]) for i in range(1, len(lams)-1)]


def aitken(x):
    return x[-3] - (x[-2]-x[-3])**2/(x[-1]-2*x[-2]+x[-3])


def records(run):
    r = js(DATA / "cascade" / run / "cascade.json")
    return [{"source": run, "period": p["period"], **p["flip"],
             "original_tolerance": r["args"].get("tol"),
             "args": r["args"]} for p in r["flips"]]


def progress_flips(run):
    path = DATA / "cascade" / run / "progress.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in read(path).splitlines() if line.startswith(b'{"flip"')]
    return rows


def edge(run):
    r = js(DATA / "edges" / run / "edges.json")
    return next(p["lambda"] for p in r["windows"][0]["edges"] if p["kind"] == "flip")


def main():
    start = time.monotonic()
    selected = records("crescent-t-0.156-b")[:2] + records("crescent-t-0.156-c")
    selected += records("crescent-t-0.156-d") + records("crescent-t-0.156-f")
    for r in selected:
        v = r["critical_multiplier"]
        vals = [complex(*p) for p in r["multipliers"]]
        ci = min(range(len(vals)), key=lambda k: abs(vals[k] - complex(v[0], v[1])))
        r["critical_error"] = abs(complex(v[0], v[1])+1)
        r["recomputed_next_modulus"] = max(abs(p) for k, p in enumerate(vals) if k != ci)
        r["next_matches"] = r["recomputed_next_modulus"] == r["next_modulus"]
        r["newton_iterations"] = len(r["newton_residuals"])-1
        r["final_newton_residual"] = r["newton_residuals"][-1]
        r["real_flip_candidate"] = abs(v[1]) < 1e-8 and abs(v[0]+1) < 1e-6
    lams = [r["lambda"] for r in selected]
    ratios = ratio(lams)
    dev = [r-DELTA for r in ratios]
    shrink = [dev[i]/dev[i+1] for i in range(1, len(dev)-1)]
    variations = []
    for offset in (-3.3e-13, 3.3e-13):
        ll = lams.copy()
        ll[-2] += offset
        rr = ratio(ll)
        variations.append({"offset": offset, "last_two_ratios": rr[-2:], "aitken": aitken(rr)})
    epath = DATA / "cascade/crescent-t-0.156-e/progress.jsonl"
    rejected = [json.loads(line) for line in read(epath).splitlines()
                if line.startswith(b"{") and b'"g"' in line]
    xx = np.array([r["lambda"] for r in rejected])
    yy = np.array([r["g"] for r in rejected])
    origin = float(xx.mean())
    design = np.column_stack((np.ones(len(xx)), xx-origin))
    cc = np.linalg.lstsq(design, yy, rcond=None)[0]
    rejected_fit = {"n": len(xx), "zero": float(origin-cc[0]/cc[1]),
                    "slope": float(cc[1]), "g_max_abs_residual": float(max(abs(design@cc-yy))),
                    "max_residual_as_parameter": float(max(abs(design@cc-yy))/abs(cc[1])),
                    "failed_attempt_kept": True}
    r148 = records("crescent-t-0.14798-v2")
    q15 = records("q15-a0.100-p4")
    other = {
        "minus014798": ratio([r["lambda"] for r in r148] +
                            [r["lambda"] for r in progress_flips("crescent-t-0.14798-p96b")] +
                            [r["lambda"] for r in (progress_flips("crescent-t-0.14798-p96d") or
                                                  progress_flips("crescent-t-0.14798-p96c"))]),
        "q15": ratio([r["lambda"] for r in q15] +
                     [edge("q15-a0.100-p16"), edge("q15-a0.100-p32")] +
                     [r["lambda"] for r in progress_flips("q15-a0.100-p64")] +
                     [r["lambda"] for r in progress_flips("q15-a0.100-p128")]),
    }
    tree = {}
    for tag in "abc":
        folder = DATA / f"scan64/ft-{tag}"
        meta = js(folder / "scan.json")
        with np.load(io.BytesIO(read(folder / "scan.npz")), allow_pickle=False) as z:
            a = z["alpha"]
            p = z["period"][0]
            ly = z["lyapunov"][0]
            checks = []
            for r in selected:
                period = r["period"]
                idx = np.flatnonzero(p == period)
                if not len(idx) or not a[0] < r["lambda"] < a[-1]:
                    continue
                later = np.arange(idx[-1]+1, len(p))
                want = p[later] == 2*period if 2*period <= meta["args"]["max_period"] else p[later] <= 0
                hits = later[want]
                checks.append({"period": period, "flip": r["lambda"],
                               "scan": float(a[hits[0]]) if len(hits) else None,
                               "difference": float(a[hits[0]]-r["lambda"]) if len(hits) else None})
            chaotic = (ly > 1e-3) & (p <= 0)
            tree[tag] = {"cells": len(a), "transient": meta["args"]["transient"],
                         "record": meta["args"]["record"], "max_period": meta["args"]["max_period"],
                         "chaotic_cells": int(chaotic.sum()), "chaotic_alpha_max": float(a[chaotic].max()),
                         "doubling_checks": checks, "saved_keys": z.files}
    result = {
        "selected_flips": selected, "ratios": ratios, "deviations": dev, "shrink_factors": shrink,
        "last_ratio_percent_error": 100*abs(ratios[-1]-DELTA)/DELTA, "aitken": aitken(ratios),
        "location_sensitivity": variations, "refused_p384_linear_fit": rejected_fit,
        "other_cuts": other, "tree": tree, "inputs": PINS,
        "seconds": time.monotonic()-start,
        "limits": ["Saved spectra/Newton histories independently reduced, not regenerated.",
                   "Recorded Ritz residual is for the computed chain action, not a bound on exact-real model/product error.",
                   "Sub-rounding next multipliers are not independently resolved eigenvalues.",
                   "Finite-time scan classifications are not asymptotic chaos proofs.",
                   "No dense product, original orbit, or old controls rerun."],
    }
    with (OUT / "cascade-evidence.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"flips": len(selected), "ratios": ratios, "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
