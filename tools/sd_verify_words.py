"""Float64 check of the symbolic words of run SD at chosen cells (human, 19 Oct 2026: "Let's get the symbolic dynamics").

For each cell (alpha, log10 T, period q = 6k): from the fixed start, a transient, then Newton on the q-fold map with
exact Jacobians (forward-mode AD, SoftOrbit.newton). With the orbit converged:
- the leading Floquet vector of the monodromy at the first point is carried along the orbit by the exact one-step
  Jacobians; at every point, the sign of the cosine between the carried vector and itself six steps later is the sign
  of the 6-step derivative along the band (L: positive, R: negative), and the norm of the 6-step product its size;
- in every phase, the k signs in time order give the word by maximal rotation in the unimodal order (as in
  symbolic_dynamics.py); the six phases must agree;
- the product of the k signs of one phase must equal the sign of the leading multiplier;
- unimodal separability: in every phase, the k orbit points projected on their first principal axis must have their
  L and R points on opposite sides of one threshold (k >= 2), as the coding of a unimodal map requires.
A cell whose leading multiplier is complex, or whose Newton does not converge, is reported as such, never coded.

Usage: python tools/sd_verify_words.py --cells 0.0733:-0.1531:6 0.0729:-0.1530:12 --device cuda --out llm-soft-loop/pod64/sd/verify
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
    ap.add_argument("--cells", nargs="+", required=True, metavar="ALPHA:LOG10T:PERIOD")
    ap.add_argument("--phase-period", type=int, default=6)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--max-newton", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    import symbolic_dynamics as sd

    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    mss = set(sd.mss_words(9))
    P = args.phase_period
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for cell in args.cells:
        a, t, q = cell.split(":")
        a, t, q = float(a), float(t), int(q)
        rec = {"alpha": a, "log10_temperature": t, "period": q}
        tic = time.perf_counter()
        orb = esl.SoftOrbit(loop, a * v, 10.0 ** t, chunk=512, jacobian_method="ad")
        X = orb.iterate(orb.start, args.transient)
        rec["return_distances_after_transient"] = orb.return_distances(X, q)[-1]
        X, hist, vals, pts = orb.newton(X, q, args.tol, args.max_newton)
        rec["newton_residuals"] = hist
        rec["converged"] = bool(hist[-1] < args.tol)
        lead = vals[0]
        rec["leading_multiplier"] = [float(lead.real), float(lead.imag)]
        rec["second_modulus"] = float(abs(vals[1])) if len(vals) > 1 else None
        if not rec["converged"] or q % P:
            rec["status"] = "not converged" if not rec["converged"] else "period not a multiple of the phase period"
            results.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        Js = [orb.jacobian(x) for x in pts]
        M = np.eye(Js[0].shape[0])
        for J in Js:
            M = J @ M
        w, V = np.linalg.eig(M)
        o = np.argsort(-np.abs(w))
        if abs(w[o[0]].imag) > 1e-9 * abs(w[o[0]]) or (len(w) > 1 and abs(abs(w[o[1]]) - abs(w[o[0]])) < 1e-9):
            rec["status"] = "leading multiplier complex or not simple"
            results.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        vec = np.real(V[:, o[0]])
        vec /= np.linalg.norm(vec)
        carried, norms = [vec], []
        for i in range(q + P):
            u = Js[i % q] @ carried[-1]
            norms.append(float(np.linalg.norm(u)))
            carried.append(u / norms[-1])
        cos = np.array([float(carried[i + P] @ carried[i]) for i in range(q)])
        logd = np.array([float(np.sum(np.log(norms[i:i + P]))) for i in range(q)])
        k = q // P
        S = np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])
        phases = []
        for j in range(P):
            idx = [j + P * i for i in range(k)]
            seq = "".join("L" if cos[i] > 0 else "R" for i in idx)
            b = sd.max_rotation(seq)
            word = seq[b:] + seq[:b]
            sep = None
            if k >= 2:
                pc = S[idx] - S[idx].mean(0)
                _, _, vt = np.linalg.svd(pc, full_matrices=False)
                s = pc @ vt[0]
                order = "".join(seq[i] for i in np.argsort(s))
                sep = sum(order[i] != order[i + 1] for i in range(k - 1)) <= 1
            phases.append({"phase": j, "signs_in_time_order": seq, "word": word, "min_abs_cos": float(np.min(np.abs(cos[idx]))),
                           "log_abs_6step_derivative": [round(float(logd[i]), 4) for i in idx], "separable": sep})
        words = {p["word"] for p in phases}
        signs_product = (-1) ** phases[0]["signs_in_time_order"].count("R")
        rec.update(phases=phases, word=phases[0]["word"] if len(words) == 1 else None,
                   phases_agree=len(words) == 1, parity_matches_multiplier=bool(np.sign(lead.real) == signs_product),
                   separable_all_phases=all(p["separable"] in (True, None) for p in phases),
                   mss_word=bool(len(words) == 1 and phases[0]["word"][:-1] in mss),
                   seconds=round(time.perf_counter() - tic, 1))
        rec["status"] = "ok" if (rec["phases_agree"] and rec["parity_matches_multiplier"] and rec["separable_all_phases"]
                                 and rec["mss_word"]) else "check failed"
        results.append(rec)
        print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "phases"}), flush=True)
        (out / "verify.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))
    (out / "verify.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))


if __name__ == "__main__":
    main()
