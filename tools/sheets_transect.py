"""Run SH microtransects (consult 13): existence against selection at one visible switch of the fixed-start map.

Along a straight transect in (alpha, log10 T) with --n points, the tool runs four kinds of arm:
- **x0:** the maps' fixed start at every point, independently;
- **alt:** a frozen non-special token embedding (--alt-token) at every point, independently;
- **continuation:** each family seed (--family NAME=FILE.npy, DATA/explore-relative or absolute), continued along the
  transect in both directions. The first point starts from the seed with --first-transient steps; each later point
  starts from the previous point's final state with --transient steps.
- **refinement:** the first switch of x0's family along the transect is refined by bisection to 1/2^--refine of the
  point spacing. At the two ends of the refined bracket, the tool reruns x0, then each continuation arm
  (warm-started from its state at the adjacent coarse point), with the leading multipliers of every periodic orbit
  found (block subspace iteration with JVPs, tools/explore_llm_soft_loop.SoftOrbit).

Periods use full-state recurrence: |x_t - x_(t-k q)| < tol * max(1, |x_t|) for k = 1..--reps, q <= --max-period.
Otherwise the labels are those of tools/sheets_map.py: leading-token loop, twin exponent, settle step, key-token
fingerprint.

The class of the switch is
- **selection** (stable overlap): x0's family changes inside the bracket while both families' continuations persist
  across it, periodic and attracting (all computed multipliers inside the unit circle);
- **intrinsic**: the family x0 leaves is absent (its continuation lands elsewhere) on the far side;
- **unresolved**: anything else, kept with its reasons.

Output: DATA/explore/OUT/transect.json and states.npz.

Usage: python tools/sheets_transect.py --from 0.0646 -0.148 --to 0.0648 -0.148 --n 17 \
         --family p2=sheets/seeds/p2.npy --family p3=sheets/seeds/p3.npy --out sheets/transect-crescent
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheets_map as sm  # noqa: E402
from explore_llm_soft_loop import DATA, SoftOrbit  # noqa: E402


def load_state(path, d):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = DATA / "explore" / path
    s = np.load(p).astype(np.float64).reshape(-1)
    if s.shape[0] != d:
        raise SystemExit(f"{path}: dimension {s.shape[0]}, model {d}")
    return s


def describe(r, res, c):
    inv = {v: k for k, v in r.families.items()}
    f = int(res["family"][c])
    return {"family": "".join(r.loop.sm.tok.decode([t]) for t in inv[f]) if f >= 0 else "-",
            "period": int(res["period"][c]), "lyapunov": float(res["lyapunov"][c]), "settle": int(res["settle"][c]),
            "p_blank": float(res["keyprob"][c][sm.KEY_TOKENS.index("\n\n")])}


def multipliers(r, point, state, period, k):
    torch = r.torch
    lt, a = point
    orb = SoftOrbit(r.loop, a * r.v, 10.0 ** lt, jacobian_method="ad")
    X = torch.as_tensor(state[None, None, :], dtype=torch.float64, device=r.loop.device)
    X = orb.points(X, period)[0]
    vals, ok = orb.subspace_multipliers(X, period, k=k, iters=30)
    return {"abs": [float(abs(v)) for v in vals[: max(2, k // 2)]], "converged": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="p0", type=float, nargs=2, required=True, metavar=("ALPHA", "LOG10T"))
    ap.add_argument("--to", dest="p1", type=float, nargs=2, required=True, metavar=("ALPHA", "LOG10T"))
    ap.add_argument("--n", type=int, default=17)
    ap.add_argument("--family", action="append", required=True, help="NAME=STATE.npy (repeat)")
    ap.add_argument("--alt-token", default=" world")
    ap.add_argument("--first-transient", type=int, default=1000)
    ap.add_argument("--transient", type=int, default=300)
    ap.add_argument("--x0-transient", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--record", type=int, default=260)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--refine", type=int, default=6)
    ap.add_argument("--mult-k", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.record < args.reps * args.max_period + 1 or args.record > args.samples:
        raise SystemExit("need reps * max_period + 1 <= record <= samples")
    ns = types.SimpleNamespace(model="qwen0.5b", layer=12, vectors="llm-phase-map/vectors-l12", concept="philosophy",
                               prompt="open", opener="Hello!", dtype="float64", device=args.device, threads=args.threads,
                               samples=args.samples, record=args.record, max_period=args.max_period, tol=args.tol,
                               settle_tol=1e-6, eps=1e-6, seed=0, reps=args.reps)
    tic = time.perf_counter()
    r = sm.Runner(ns)
    torch, d = r.torch, r.loop.E.shape[1]
    ids = r.loop.sm.tok(args.alt_token, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise SystemExit(f"alt token {args.alt_token!r} is not one token: {ids}")
    alt = r.loop.E[ids[0]].double().cpu().numpy()
    fams = {}
    for spec in args.family:
        name, path = spec.split("=", 1)
        fams[name] = load_state(path, d)
    p0, p1 = np.array(args.p0), np.array(args.p1)
    us = np.linspace(0, 1, args.n)
    pts = [tuple((p0 + u * (p1 - p0))[::-1]) for u in us]          # (log10 T, alpha) as Runner expects
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "alt_token_id": int(ids[0]), "points": [{"alpha": p[1], "log10_temperature": p[0]}
                                                                         for p in pts]}
    # independent arms: x0 and alt at every point (one batch)
    X0 = torch.as_tensor(np.stack([alt] * args.n)[:, None, :], dtype=torch.float64, device=r.loop.device)
    res_x0 = r.run_cells(pts, None, args.x0_transient)
    res_alt = r.run_cells(pts, X0, args.x0_transient)
    report["x0"] = [describe(r, res_x0, c) for c in range(args.n)]
    report["alt"] = [describe(r, res_alt, c) for c in range(args.n)]
    print(json.dumps({"x0": [x["family"] for x in report["x0"]]}), flush=True)
    # continuation arms in lockstep: each family forward and backward
    arms = [(name, sgn) for name in fams for sgn in (+1, -1)]
    carry = {a: None for a in arms}
    cont = {f"{name}{'+' if sgn > 0 else '-'}": [None] * args.n for name, sgn in arms}
    states = {}
    for step in range(args.n):
        cells, x0s = [], []
        for name, sgn in arms:
            k = step if sgn > 0 else args.n - 1 - step
            cells.append(pts[k])
            x0s.append(fams[name] if carry[(name, sgn)] is None else carry[(name, sgn)])
        X = torch.as_tensor(np.stack(x0s)[:, None, :], dtype=torch.float64, device=r.loop.device)
        res = r.run_cells(cells, X, args.first_transient if step == 0 else args.transient)
        for c, (name, sgn) in enumerate(arms):
            k = step if sgn > 0 else args.n - 1 - step
            key = f"{name}{'+' if sgn > 0 else '-'}"
            cont[key][k] = describe(r, res, c)
            st = res["state"][c, -1].double().cpu().numpy()
            carry[(name, sgn)] = st
            states[f"{key}_{k}"] = st.astype(np.float32)
        print(json.dumps({"continuation_step": step + 1, "of": args.n,
                          "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    report["continuation"] = cont
    # refinement of the first x0 switch, with persistence and stability of each family at the bracket ends
    fx = [x["family"] for x in report["x0"]]
    k = next((i for i in range(args.n - 1) if fx[i] != fx[i + 1]), None)
    if k is None:
        report["switch"] = None
        report["class"] = "no switch of x0 along the transect"
    else:
        lo, hi = us[k], us[k + 1]
        at = lambda u: tuple((p0 + u * (p1 - p0))[::-1])
        for _ in range(args.refine):
            mid = 0.5 * (lo + hi)
            fm = describe(r, r.run_cells([at(mid)], None, args.x0_transient), 0)["family"]
            if fm == fx[k]:
                lo = mid
            elif fm == fx[k + 1]:
                hi = mid
            else:
                report["third_family"] = {"u": float(mid), "family": fm}
                break
        ends = [at(lo), at(hi)]
        sw = {"u": [float(lo), float(hi)], "points": [{"alpha": e[1], "log10_temperature": e[0]} for e in ends],
              "x0": [describe(r, r.run_cells([e], None, args.x0_transient), 0) for e in ends], "families": {}}
        for name in fams:
            entry = []
            for j, e in enumerate(ends):
                src = states[f"{name}+_{k}"] if j == 0 else states[f"{name}-_{k + 1}"]
                rr = r.run_cells([e], torch.as_tensor(src[None, None, :], dtype=torch.float64, device=r.loop.device),
                                 args.transient)
                dsc = describe(r, rr, 0)
                if dsc["period"] > 0:
                    dsc["multipliers"] = multipliers(r, e, rr["state"][0, -1].double().cpu().numpy(), dsc["period"],
                                                     args.mult_k)
                entry.append(dsc)
            sw["families"][name] = entry
        report["switch"] = sw
        fam_lo, fam_hi = sw["x0"][0]["family"], sw["x0"][1]["family"]
        persist = {}
        for name, entry in sw["families"].items():
            same = entry[0]["family"] == entry[1]["family"] != "-"
            stable = all(e.get("multipliers", {}).get("abs", [2])[0] < 1 and e.get("multipliers", {}).get("converged", False)
                         for e in entry)
            persist[name] = {"same_family_both_ends": same, "attracting_both_ends": stable,
                             "family": entry[0]["family"]}
        report["persistence"] = persist
        holders = {persist[n]["family"]: persist[n] for n in persist if persist[n]["same_family_both_ends"]}
        if fam_lo != fam_hi and fam_lo in holders and fam_hi in holders and \
                holders[fam_lo]["attracting_both_ends"] and holders[fam_hi]["attracting_both_ends"]:
            report["class"] = "selection (stable overlap)"
        elif fam_lo not in holders or fam_hi not in holders:
            report["class"] = "intrinsic or unresolved: a family of x0 is not continued across the bracket"
        else:
            report["class"] = "unresolved"
    report["seconds"] = round(time.perf_counter() - tic, 1)
    (out / "transect.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(out / "states.npz", **states)
    print(json.dumps({"class": report["class"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
