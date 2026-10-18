"""Period-5 shrimp pilot (exploratory; human request 18 Oct 2026): follow the window's head analog to lower temperature.

The period-5 window of the two-block plane (Qwen2.5-0.5B-Instruct, gratitude at block 4 = alpha1, hostility at block
16 = alpha2, log10 T = -0.1) has a near-nilpotent point at (alpha1, alpha2) = (-0.1184836, 0.1017217), where the sum
of the leading multipliers and the sum of their pairwise products vanish. There the three leading multipliers have
nearly equal moduli (3.4-3.6e-3), so no one- or two-dimensional head forms (paper, "Shrimps of a two-dimensional
kind"; tools/p5_nilpotent.py).

Hypothesis. Temperature acts like the dissipation parameter b of Henon's map: at lower T fewer tokens compete, so
the readout covariance and each step's derivative approach rank one, and the window may become one-dimensional,
with a true shrimp head.

Procedure:
- Seed exactly as tools/p5_nilpotent.py at log10 T = -0.1: march to its head and refine there by a 2D Newton on
  (c1, c2) = (sum of Ritz values, sum of pairwise products).
- Step log10 T down in steps of --dlt to --lt-min. At each temperature: Newton for the period-5 orbit warm-started
  from the previous one, then the 2D Newton in (alpha1, alpha2) on (c1, c2).
- At each temperature, record the leading multipliers at the head and at four points offset by +-delta in each
  alpha (the window's dimension away from the special point).

Readouts per temperature:
- the moduli of the leading multipliers;
- |mu3|/|mu1| and |mu2|/|mu1| at the head and at the offsets;
- the per-step area contraction |mu1 mu2|^(1/5) of the leading pair (the analog of |b|);
- the tokens of the orbit.

Output: DATA/explore/OUT/log.jsonl and summary.json.

Usage: python tools/p5_temperature.py --device cuda --out sheets/p5-temperature
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402
import llm_feigenbaum as lfb  # noqa: E402

P = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dlt", type=float, default=0.01)
    ap.add_argument("--lt-min", type=float, default=-0.20)
    ap.add_argument("--delta", type=float, default=3e-4)
    ap.add_argument("--newton2d", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    loop = esl.SoftLoop("qwen0.5b", 4, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    loop.sm.add_steer_layer(16)
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]

    def make_cell(a1, a2, lt):
        loop.extra = (16, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, 10.0 ** lt, chunk=512)

    def solve(a1, a2, lt, X, Q):
        t = time.perf_counter()
        cell = make_cell(a1, a2, lt)
        X, Q, hist = lfb.newton(cell, X, P, Q, lfb.tol_for(P, 1e-12), 12)
        ok = hist[-1] < lfb.tol_for(P, 1e-12)
        vals, vecs, res, rh, Q = lfb.ritz(cell, X, P, Q)
        toks = []
        Y = X
        for _ in range(P):
            toks.append(int(cell.probabilities(Y).argmax()))
            Y = cell.f(Y)
        mods = [float(abs(z)) for z in vals]
        c1 = float(np.sum(vals).real)
        c2 = float(sum(vals[i] * vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))).real)
        rec = {"a1": float(a1), "a2": float(a2), "log10T": float(lt), "newton": float(hist[-1]), "converged": bool(ok),
               "tokens": toks, "moduli": mods, "mu": [[float(z.real), float(z.imag)] for z in vals],
               "ratio21": mods[1] / mods[0] if len(mods) > 1 and mods[0] > 0 else None,
               "ratio31": mods[2] / mods[0] if len(mods) > 2 and mods[0] > 0 else None,
               "area_per_step": float(abs(vals[0] * vals[1]) ** (1 / P)) if len(vals) > 1 else None,
               "c1": c1, "c2": c2, "seconds": round(time.perf_counter() - t, 1)}
        return rec, X, Q

    def head(a, lt, X, Q, tag):
        """2D Newton on (c1, c2) in (alpha1, alpha2) at one temperature (as tools/p5_nilpotent.py)."""
        h = np.array([4e-6, 8e-6])
        rec = None
        for it in range(args.newton2d):
            rec, X, Q = solve(a[0], a[1], lt, X, Q)
            rec["tag"] = f"{tag}-newton2d-{it}"
            emit(rec)
            if not rec["converged"] or abs(rec["c1"]) < 1e-5 and abs(rec["c2"]) < 1e-8:
                break
            F0 = np.array([rec["c1"], rec["c2"]])
            J = np.zeros((2, 2))
            for k in range(2):
                b = a.copy()
                b[k] += h[k]
                rk, _, _ = solve(b[0], b[1], lt, X.clone(), Q.copy())
                J[:, k] = (np.array([rk["c1"], rk["c2"]]) - F0) / h[k]
            step = np.linalg.solve(J, -F0)
            lim = np.array([8.3e-5, 1.5e-4]) / 4
            scale = min(1.0, float(np.min(lim / np.maximum(np.abs(step), 1e-30))))
            a = a + scale * step
        return a, X, Q, rec

    tic = time.perf_counter()
    a0 = (-0.11873, 0.100693)
    cell = make_cell(*a0, -0.1)
    X = cell.iterate(cell.start, 900)
    per = cell.detect_period(X, 32, 1e-6)
    emit({"seed": a0, "period_detected": int(per)})
    if per != P:
        raise SystemExit("seed is not period 5")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), 6)))[0]
    rec, X, Q = solve(*a0, -0.1, X, Q)
    tgt = (-0.1184836, 0.1017217)
    n = int(np.ceil(max(abs(tgt[0] - a0[0]) / 8.3e-5, abs(tgt[1] - a0[1]) / 1.5e-4))) + 1
    for s in np.linspace(0, 1, n + 1)[1:]:
        a = (a0[0] + s * (tgt[0] - a0[0]), a0[1] + s * (tgt[1] - a0[1]))
        rec, X, Q = solve(*a, -0.1, X, Q)
        if not rec["converged"]:
            raise SystemExit("march to the head failed")
    summary = []
    a = np.array(tgt, dtype=float)
    lts = np.round(np.arange(-0.1, args.lt_min - 1e-9, -args.dlt), 6)
    for lt in lts:
        rec, X, Q = solve(a[0], a[1], lt, X, Q)
        rec["tag"] = f"T{lt:+.3f}-orbit"
        emit(rec)
        if not rec["converged"] or rec["tokens"] is None:
            summary.append({"log10T": float(lt), "status": "orbit lost", "last": rec})
            break
        a, X, Q, hrec = head(a, lt, X, Q, f"T{lt:+.3f}")
        offs = []
        for d1, d2 in ((args.delta, 0), (-args.delta, 0), (0, args.delta), (0, -args.delta)):
            ro, _, _ = solve(a[0] + d1, a[1] + d2, lt, X.clone(), Q.copy())
            ro["tag"] = f"T{lt:+.3f}-offset"
            emit(ro)
            offs.append({k: ro[k] for k in ("converged", "moduli", "ratio21", "ratio31", "area_per_step", "tokens")})
        summary.append({"log10T": float(lt), "head": {k: hrec[k] for k in ("a1", "a2", "converged", "moduli", "ratio21",
                                                                            "ratio31", "area_per_step", "c1", "c2",
                                                                            "tokens")},
                        "offsets": offs, "seconds": round(time.perf_counter() - tic, 1)})
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    emit({"done": True, "seconds": round(time.perf_counter() - tic, 1)})


if __name__ == "__main__":
    main()
