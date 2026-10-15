"""Orbit census around the swallowtail-shaped junction of the period-5 window in the two-block plane (run OC; consult
10, direction 3).

The map is the soft-token map of Qwen2.5-0.5B-Instruct in float64, gratitude at block 4 (strength alpha1) and hostility
at block 16 (alpha2), log10 T = -0.1. At each frozen parameter point (docs/oc/oc-points.json), period-5 orbits are
sought by the low-rank Newton of tools/llm_feigenbaum.py from seeds near the fold-curve states of the junction's three
arms (run FM5, curve-a1-up-fold-minus: arm A indices 0-20, arm B 20-31, arm C 31 on): for each arm, the state of the
arm's curve point nearest in parameters, shifted along that point's critical eigenvector (multiplier +1) by s |x| for
every s of a fixed ladder. A converged solution (relative residual below the period-5 tolerance) that is a primitive
period-5 orbit is compared with the orbits found so far modulo phase: within a relative distance of 1e-7 of one of the
five iterates of a stored orbit, it is that orbit. Each distinct orbit is reported with its token loop, its leading
multipliers (Ritz) and the seeds that reached it. A count is a lower bound: an orbit that no seed reaches is not found.

Usage:
  python tools/window_census.py census --points docs/oc/oc-points.json --device cuda --out llm-soft-loop/oc/census
  python tools/window_census.py summary --run llm-soft-loop/oc/census [--json OUT]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import llm_feigenbaum as lfb  # noqa: E402

CURVE = "llm-soft-loop/pod64/fm5/curve-a1-up-fold-minus"
ARMS = {"A": (0, 21), "B": (20, 32), "C": (31, None)}
LADDER = (0.0, 1e-3, -1e-3, 3e-3, -3e-3, 1e-2, -1e-2, 3e-2, -3e-2)
LOG_T = -0.1
PERIOD = 5
SAME = 1e-7


def setup(device, threads):
    """The two-block soft loop and a cell factory make_cell(alpha1, alpha2) -> SoftOrbit."""
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    loop = esl.SoftLoop("qwen0.5b", 4, threads, "open", ela.OPENER, 1, "float64", device)
    loop.sm.add_steer_layer(16)
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]

    def make_cell(a1, a2):
        loop.extra = (16, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, 10.0 ** LOG_T)

    return loop, make_cell


def to_state(loop, x):
    return loop.torch.as_tensor(np.asarray(x, dtype=np.float64).reshape(1, 1, -1), dtype=loop.state_dtype,
                                device=loop.device)


def phases(cell, X):
    """The five iterates of an orbit point, flattened (float64 numpy)."""
    out = []
    for _ in range(PERIOD):
        out.append(lfb._flat(X))
        X = cell.f(X)
    return out


def token_loop(cell, X):
    toks = []
    for _ in range(PERIOD):
        toks.append(int(cell.probabilities(X).argmax()))
        X = cell.f(X)
    return toks


def same_orbit(x, orbit):
    """True if the flattened state x is within SAME (relative) of one of the orbit's iterates."""
    return any(np.linalg.norm(x - y) / np.linalg.norm(y) < SAME for y in orbit)


def critical_direction(cell, X, k, rng):
    """Ritz pairs at a fold-curve state: the eigenvector of the multiplier nearest +1 (real, unit), that multiplier,
    and the refreshed subspace."""
    n = X.numel()
    Q0 = np.linalg.qr(rng.normal(size=(n, k)))[0]
    vals, V, res, hist, Q = lfb.ritz(cell, X, PERIOD, Q0, passes=4)
    i = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(V[:, i] * np.exp(-1j * np.angle(V[np.argmax(np.abs(V[:, i])), i])))
    return v / np.linalg.norm(v), complex(vals[i]), Q


def census(args):
    import explore_llm_attractors as ela
    spec = json.loads(Path(args.points).read_text())
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "census.jsonl"
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are never overwritten")
    curve = json.loads((ela.DATA / "explore" / CURVE / "curve.json").read_text())["points"]
    states = np.load(ela.DATA / "explore" / CURVE / "states.npy")
    P = np.array([[p["alpha"], p["alpha2"]] for p in curve])
    loop, make_cell = setup(args.device, args.threads)
    rng = np.random.default_rng(args.seed)
    tol = lfb.tol_for(PERIOD, 1e-12)
    sources = {}                                         # (arm, curve index) -> (state, critical vector, Q)
    names = set(args.only) if args.only else None
    tic = time.perf_counter()
    for pt in spec["points"]:
        if names and pt["name"] not in names:
            continue
        a1, a2 = pt["alpha"], pt["alpha2"]
        cell = make_cell(a1, a2)
        orbits, found = [], []
        attempts = []
        for arm, (lo, hi) in ARMS.items():
            if args.arms and arm not in args.arms:
                continue
            seg = P[lo:hi]
            j = lo + int(np.argmin(np.linalg.norm((seg - [a1, a2]) / [1e-4, 1e-4], axis=1)))
            if (arm, j) not in sources:
                fcell = make_cell(*P[j])
                Xf = to_state(loop, states[j])
                vcrit, mu, Q = critical_direction(fcell, Xf, args.k, rng)
                sources[(arm, j)] = (states[j], vcrit, Q, mu)
                cell = make_cell(a1, a2)                 # make_cell resets the second block's rows
            xf, vcrit, Q, mu = sources[(arm, j)]
            for s in (args.ladder if args.ladder else LADDER):
                X0 = to_state(loop, xf + s * np.linalg.norm(xf) * vcrit)
                X, Qn, hist = lfb.newton(cell, X0, PERIOD, Q, tol=tol, max_iter=args.max_newton)
                rec = {"arm": arm, "curve_index": j, "s": s, "newton_iterations": len(hist) - 1,
                       "final_residual": hist[-1], "converged": bool(hist[-1] < tol)}
                if rec["converged"]:
                    first = lfb.return_distance(cell, X, 1)
                    rec["primitive"] = bool(first > 1e-6)
                    if rec["primitive"]:
                        x = lfb._flat(X)
                        k = next((i for i, o in enumerate(orbits) if same_orbit(x, o)), None)
                        if k is None:
                            orbits.append(phases(cell, X))
                            vals, V, res, hh, _ = lfb.ritz(cell, X, PERIOD, Qn, passes=4)
                            found.append({"tokens": token_loop(cell, X),
                                          "multipliers": [[float(z.real), float(z.imag)] for z in vals[:4]],
                                          "ritz_residuals": [float(r) for r in res[:4]],
                                          "leading_modulus": float(np.abs(vals[0])),
                                          "attracting": bool(np.abs(vals[0]) < 1), "seeds": [], "state_norm":
                                          float(np.linalg.norm(x))})
                            k = len(orbits) - 1
                        found[k]["seeds"].append([arm, j, s])
                        rec["orbit"] = k
                attempts.append(rec)
        for i, o in enumerate(orbits):                   # pairwise distances modulo phase
            found[i]["distances_to_others"] = [
                float(min(np.linalg.norm(o[0] - y) / np.linalg.norm(y) for y in orbits[m])) for m in range(len(orbits))
                if m != i]
        if orbits:                                       # phase-0 states, to seed later work (run CL2)
            np.save(out / f"orbits-{pt['name']}.npy", np.array([o[0] for o in orbits]))
        rec = {**pt, "found_count": len(found), "attracting_count": int(sum(f["attracting"] for f in found)),
               "orbits": found, "attempts": attempts, "seconds": round(time.perf_counter() - tic, 1)}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"name": pt["name"], "region": pt["region"], "predicted": pt["predicted_count"],
                          "found": len(found), "attracting": rec["attracting_count"],
                          "converged": int(sum(a["converged"] for a in attempts)), "of": len(attempts),
                          "seconds": rec["seconds"]}), flush=True)
    (out / "sources.json").write_text(json.dumps({f"{a}:{j}": {"critical_multiplier": [v[3].real, v[3].imag]}
                                                  for (a, j), v in sources.items()}, indent=1))


def summary(args):
    import explore_llm_attractors as ela
    recs = [json.loads(line) for line in open(ela.DATA / "explore" / args.run / "census.jsonl")]
    rows = []
    for r in recs:
        rows.append({"name": r["name"], "region": r["region"], "alpha": r["alpha"], "alpha2": r["alpha2"],
                     "predicted": r["predicted_count"], "found": r["found_count"], "attracting": r["attracting_count"],
                     "match": r["found_count"] == r["predicted_count"],
                     "leading_moduli": [round(o["leading_modulus"], 6) for o in r["orbits"]],
                     "tokens_shared": len({tuple(o["tokens"]) for o in r["orbits"]}) <= 1,
                     "min_distance_between_orbits": min((min(o["distances_to_others"]) for o in r["orbits"]
                                                         if o["distances_to_others"]), default=None),
                     "converged_attempts": sum(a["converged"] for a in r["attempts"]), "attempts": len(r["attempts"])})
    out = {"points": rows, "matches": sum(x["match"] for x in rows), "of": len(rows)}
    text = json.dumps(out, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census")
    c.add_argument("--points", required=True)
    c.add_argument("--only", nargs="*", default=[], help="names of points to run (default: all)")
    c.add_argument("--k", type=int, default=6, help="subspace dimension of the low-rank Newton and Ritz")
    c.add_argument("--arms", nargs="*", default=[], help="arms to seed from (default: all; for tests)")
    c.add_argument("--ladder", type=float, nargs="*", default=[], help="seed shifts (default: LADDER; for tests)")
    c.add_argument("--max-newton", type=int, default=30)
    c.add_argument("--seed", type=int, default=1015)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--device", default="cpu")
    c.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--run", required=True)
    s.add_argument("--json")
    args = ap.parse_args()
    {"census": census, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
