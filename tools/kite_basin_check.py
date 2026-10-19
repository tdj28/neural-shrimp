"""Is the sharp edge inside the kite of the period-6 shrimp a basin boundary of the fixed start? (run HR11, 19 Oct 2026:
the 512 x 512 map shows a jump in contraction along a straight line inside the kite, where no continued curve runs).

At two cells on either side of the edge (same alpha): the orbit the fixed start reaches (transient, then Newton with
exact Jacobians), then each orbit continued to the other cell by Newton from its own state. If both orbits exist at
both cells, distinct (gap up to cyclic shift) and both attracting, the edge only marks where the fixed start switches
between two coexisting orbits. Float64.

Usage: python tools/kite_basin_check.py --alpha 0.072975 --log10t -0.15350 -0.15356 --out llm-soft-loop/kite/basin-check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log10t", type=float, nargs=2, required=True, help="the two cells' log10 T (either side of the edge)")
    ap.add_argument("--period", type=int, default=6)
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    q = args.period
    torch = loop.torch
    orbs, found = {}, {}
    tic = time.perf_counter()
    for t in args.log10t:
        orbs[t] = esl.SoftOrbit(loop, args.alpha * v, 10.0 ** t, chunk=512, jacobian_method="ad")
        X = orbs[t].iterate(orbs[t].start, args.transient)
        X, hist, vals, pts = orbs[t].newton(X, q, 1e-11, 10)
        found[t] = {"state": X, "residuals": hist, "multipliers": vals[:3],
                    "points": np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])}
        print(json.dumps({"cell": t, "fixed_start": True, "residual": hist[-1], "mu1": [vals[0].real, vals[0].imag]}), flush=True)

    def gap(P, Q):            # relative distance between two orbits, minimized over cyclic shifts
        return min(float(np.linalg.norm(P - np.roll(Q, s, axis=0)) / np.linalg.norm(P)) for s in range(len(P)))

    rows = []
    t0, t1 = args.log10t
    for src, dst in ((t0, t1), (t1, t0)):
        X0 = found[src]["state"]
        X, hist, vals, pts = orbs[dst].newton(X0.clone(), q, 1e-11, 12)
        P = np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])
        rows.append({"orbit_from_cell": src, "continued_to_cell": dst, "residuals": hist, "converged": hist[-1] < 1e-11,
                     "mu1": [float(vals[0].real), float(vals[0].imag)], "modulus1": float(abs(vals[0])),
                     "gap_to_fixed_start_orbit_there": gap(P, found[dst]["points"]),
                     "gap_to_its_source": gap(P, found[src]["points"])})
        print(json.dumps(rows[-1]), flush=True)
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
           "fixed_start": {str(t): {"residual": found[t]["residuals"][-1],
                                    "mu": [[float(m.real), float(m.imag), float(abs(m))] for m in found[t]["multipliers"]]}
                           for t in args.log10t},
           "gap_between_fixed_start_orbits": gap(found[t0]["points"], found[t1]["points"]), "continued": rows}
    both = all(r["converged"] and r["modulus1"] < 1 and r["gap_to_fixed_start_orbit_there"] > 1e-3 for r in rows)
    rec["reading"] = ("both orbits exist and attract at both cells: the edge is where the fixed start switches"
                      if both else "not established: see the continued rows")
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "basin-check.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("gap_between_fixed_start_orbits", "reading", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
