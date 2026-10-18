"""Run SH microtransects (consult 13): existence against selection at one visible switch of the fixed-start map.

Along a straight transect in (alpha, log10 T) with --n points, the tool runs four kinds of arm:
- **x0:** the maps' fixed start at every point, independently;
- **alt:** a frozen non-special token embedding (--alt-token) at every point, independently;
- **continuation:** each family seed (--family NAME=FILE.npy, DATA/explore-relative or absolute), continued along the
  transect in both directions. The first point starts from the seed with --first-transient steps; each later point
  starts from the previous point's final state with --transient steps.
- **refinement:** each switch of x0's family along the transect (up to --max-switches) is refined by bisection to 1/2^--refine of the
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

**Amendment 1 (18 Oct 2026, run MT):** the class above compares, at the bracket ends, the arm that arrives from the
low side (at the low end) with the arm that arrives from the high side (at the high end). That tests a family's
persistence only if both arms hold it, and an arm whose seed does not exist at its starting end never does (transect S:
the period-12 seed fell onto the band at alpha 0.0700, so the arm from the high side never held period 12). The **cross
test** therefore continues each side's own fixed-start state across the bracket:
- x0 is run at both ends;
- the low end's final state is continued at the high end, and the high end's at the low end;
- multipliers are computed for every periodic orbit found.
A side where x0 reaches no detected period is rerun with --long-transient steps (slow convergence near a multiplier
of modulus one is not chaos), and that state is used if it is periodic. Classes (class_cross):
- **selection (stable overlap):** each side's family persists on the other side, with all four orbits attracting;
- **intrinsic:** a side's family does not persist across the bracket (it ends inside it);
- **unresolved:** otherwise, with reasons.
--from-report REPORT runs only the cross test, on the brackets of an existing transect.json, and writes cross.json.

Output: DATA/explore/OUT/transect.json and states.npz (cross.json with --from-report).

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


def cross_test(r, args, ends):
    """Amendment 1: x0 at both bracket ends, then each end's final state continued at the other end (see docstring)."""
    torch = r.torch
    dev = r.loop.device
    res = r.run_cells(list(ends), None, args.x0_transient)
    x0 = [describe(r, res, c) for c in range(2)]
    st = [res["state"][c, -1].double().cpu().numpy() for c in range(2)]
    for j in range(2):
        if x0[j]["period"] == 0 and args.long_transient > 0:      # rerun a side with no detected period for longer
            X = torch.as_tensor(st[j][None, None, :], dtype=torch.float64, device=dev)
            rl = r.run_cells([ends[j]], X, args.long_transient)
            x0[j]["long"] = describe(r, rl, 0)
            if x0[j]["long"]["period"] > 0:
                st[j] = rl["state"][0, -1].double().cpu().numpy()
    X = torch.as_tensor(np.stack(st)[:, None, :], dtype=torch.float64, device=dev)
    rx = r.run_cells([ends[1], ends[0]], X, args.transient)    # low side's state at the high end, and vice versa
    cr = [describe(r, rx, c) for c in range(2)]
    cst = [rx["state"][c, -1].double().cpu().numpy() for c in range(2)]
    eff = [x0[j]["long"] if x0[j]["period"] == 0 and "long" in x0[j] else x0[j] for j in range(2)]
    for dsc, point, state in ((eff[0], ends[0], st[0]), (eff[1], ends[1], st[1]), (cr[0], ends[1], cst[0]),
                              (cr[1], ends[0], cst[1])):
        if dsc["period"] > 0:
            dsc["multipliers"] = multipliers(r, point, state, dsc["period"], args.mult_k)
    att = lambda e: e.get("multipliers", {}).get("converged", False) and e["multipliers"]["abs"][0] < 1
    A, B = eff[0]["family"], eff[1]["family"]
    long_note = " (after a long transient)" if any("long" in x for x in x0) else ""
    if "-" in (A, B):
        cls = "unresolved: x0 reaches no detected period on one side, even after a long transient"
    elif A == B:
        cls = "unresolved: x0 reaches the same family at both ends"
    else:
        a_in, b_in = cr[0]["family"] == A, cr[1]["family"] == B
        if a_in and b_in and all(att(e) for e in (eff[0], eff[1], cr[0], cr[1])):
            cls = "selection (stable overlap)" + long_note
        elif a_in and b_in:
            cls = "unresolved: both families persist, but a multiplier check failed or did not converge" + long_note
        elif "-" in (cr[0]["family"], cr[1]["family"]):
            cls = "unresolved: a continuation across the bracket reaches no detected period" + long_note
        else:
            ends_ = [n for n, ok in (("low side's", a_in), ("high side's", b_in)) if not ok]
            cls = "intrinsic: the " + " and the ".join(ends_) + " family ends inside the bracket" + long_note
    return {"x0": x0, "across": {"low_state_at_high_end": cr[0], "high_state_at_low_end": cr[1]}, "class": cls}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="p0", type=float, nargs=2, metavar=("ALPHA", "LOG10T"))
    ap.add_argument("--to", dest="p1", type=float, nargs=2, metavar=("ALPHA", "LOG10T"))
    ap.add_argument("--from-report", default=None, help="run only the cross test on this transect.json's brackets")
    ap.add_argument("--long-transient", type=int, default=6000)
    ap.add_argument("--n", type=int, default=17)
    ap.add_argument("--family", action="append", default=[], help="NAME=STATE.npy (repeat)")
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
    ap.add_argument("--max-switches", type=int, default=4)
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
    if args.from_report:
        src = DATA / "explore" / args.from_report
        rep = json.loads(src.read_text())
        out = DATA / "explore" / args.out
        out.mkdir(parents=True, exist_ok=True)
        res = {"args": vars(args), "source": str(src), "switches": []}
        for sw in rep["switches"]:
            ends = [(p["log10_temperature"], p["alpha"]) for p in sw["points"]]
            ct = cross_test(r, args, ends)
            ct.update({"points": sw["points"], "class_arms": sw["class"],
                       "x0_recorded": [x["family"] for x in sw["x0"]]})
            res["switches"].append(ct)
            print(json.dumps({"bracket": [p["alpha"] for p in sw["points"]], "class_arms": sw["class"],
                              "class_cross": ct["class"], "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
            (out / "cross.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        res["seconds"] = round(time.perf_counter() - tic, 1)
        (out / "cross.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return
    if args.p0 is None or args.p1 is None or not args.family:
        raise SystemExit("--from, --to and --family are required unless --from-report is given")
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
    (out / "transect.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))     # kept if a later step times out
    np.savez_compressed(out / "states.npz", **states)
    # refinement of every x0 switch (up to --max-switches), with persistence and stability of each family at the
    # bracket ends
    fx = [x["family"] for x in report["x0"]]
    ks = [i for i in range(args.n - 1) if fx[i] != fx[i + 1]][:args.max_switches]
    at = lambda u: tuple((p0 + u * (p1 - p0))[::-1])
    report["switches"] = []
    for k in ks:
        lo, hi = us[k], us[k + 1]
        sw = {"coarse": [k, k + 1], "families_x0_coarse": [fx[k], fx[k + 1]]}
        for _ in range(args.refine):
            mid = 0.5 * (lo + hi)
            fm = describe(r, r.run_cells([at(mid)], None, args.x0_transient), 0)["family"]
            if fm == fx[k]:
                lo = mid
            elif fm == fx[k + 1]:
                hi = mid
            else:
                sw["third_family"] = {"u": float(mid), "family": fm}
                break
        ends = [at(lo), at(hi)]
        sw["u"] = [float(lo), float(hi)]
        sw["points"] = [{"alpha": e[1], "log10_temperature": e[0]} for e in ends]
        sw["x0"] = [describe(r, r.run_cells([e], None, args.x0_transient), 0) for e in ends]
        sw["families"] = {}
        for name in fams:
            entry = []
            for jj, e in enumerate(ends):
                src = states[f"{name}+_{k}"] if jj == 0 else states[f"{name}-_{k + 1}"]
                rr = r.run_cells([e], torch.as_tensor(src[None, None, :], dtype=torch.float64, device=r.loop.device),
                                 args.transient)
                dsc = describe(r, rr, 0)
                if dsc["period"] > 0:
                    dsc["multipliers"] = multipliers(r, e, rr["state"][0, -1].double().cpu().numpy(), dsc["period"],
                                                     args.mult_k)
                entry.append(dsc)
            sw["families"][name] = entry
        fam_lo, fam_hi = sw["x0"][0]["family"], sw["x0"][1]["family"]
        persist = {}
        for name, entry in sw["families"].items():
            same = entry[0]["family"] == entry[1]["family"] != "-"
            stable = all(e.get("multipliers", {}).get("abs", [2])[0] < 1 and e.get("multipliers", {}).get("converged", False)
                         for e in entry)
            persist[name] = {"same_family_both_ends": same, "attracting_both_ends": stable, "family": entry[0]["family"]}
        sw["persistence"] = persist
        holders = {persist[n]["family"]: persist[n] for n in persist if persist[n]["same_family_both_ends"]}
        if fam_lo != fam_hi and fam_lo in holders and fam_hi in holders and \
                holders[fam_lo]["attracting_both_ends"] and holders[fam_hi]["attracting_both_ends"]:
            sw["class"] = "selection (stable overlap)"
        elif "-" in (fam_lo, fam_hi):
            sw["class"] = "unresolved: x0 reaches no detected period on one side (chaos or a long transient)"
        elif fam_lo not in holders or fam_hi not in holders:
            sw["class"] = "intrinsic or unresolved: a family of x0 is not continued across the bracket"
        else:
            sw["class"] = "unresolved"
        ct = cross_test(r, args, ends)
        sw["cross"] = ct
        sw["class_cross"] = ct["class"]
        report["switches"].append(sw)
        print(json.dumps({"switch": sw["u"], "families": [fam_lo, fam_hi], "class": sw["class"],
                          "class_cross": sw["class_cross"]}), flush=True)
        (out / "transect.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    report["class"] = [sw["class"] for sw in report["switches"]] or "no switch of x0 along the transect"
    report["seconds"] = round(time.perf_counter() - tic, 1)
    (out / "transect.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(out / "states.npz", **states)
    print(json.dumps({"class": report["class"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
