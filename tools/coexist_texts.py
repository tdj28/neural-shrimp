"""Two attractors at one parameter point: what the soft-token loop writes from two different starts.

At each (alpha, log10 T) the loop of Qwen2.5-0.5B-Instruct (steered at layer 12) runs from the fixed start (the
opener state, as in every map) and from a donor state: the attractor reached from the fixed start at a donor point,
here one where the 2-cycle "is it" is the attractor. Per run: the period after the transient (relative return
distance below --tol), and at each recorded step the most probable next token, its probability and the decoded
text. Two runs at the same parameters that settle on different periodic orbits, or on a periodic orbit and an
aperiodic attractor, show coexisting attractors.

Usage: python tools/coexist_texts.py --points A,LT [A,LT ...] --donor A,LT --json OUT [--transient 600 --record 24]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def record(cell, X, steps, tok):
    """Most probable token, its probability and the soft state for `steps` steps from X."""
    loop = cell.loop
    out = []
    for _ in range(steps):
        cell._use(1)
        X, p = loop.step(X, np.array([cell.T]))
        k = int(p[0].argmax())
        out.append({"token": k, "text": tok.decode([k]), "p": float(p[0, k])})
    return X, out


def twin_exponent(cell, X, steps, rel=1e-6):
    """Largest Lyapunov exponent from a twin trajectory at relative offset `rel`, renormalized every step."""
    torch = cell.torch
    rng = np.random.default_rng(0)
    d = torch.as_tensor(rng.normal(size=X.shape), dtype=X.dtype, device=X.device)
    d0 = rel * float(X.norm())
    Y = X + d0 * d / d.norm()
    total = 0.0
    for _ in range(steps):
        X, Y = cell.f(X), cell.f(Y)
        dist = float((Y - X).norm())
        total += np.log(dist / d0)
        Y = X + (Y - X) * (d0 / dist)
    return total / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", required=True, help="alpha,log10T pairs")
    ap.add_argument("--donor", required=True, help="alpha,log10T whose fixed-start attractor is the second start")
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--record", type=int, default=24)
    ap.add_argument("--lyapunov-steps", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop("qwen0.5b", args.layer, args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    tok = loop.sm.tok
    v = elpm.load_vectors(args.vectors)[args.concept]
    ad, ltd = (float(s) for s in args.donor.split(","))
    donor = esl.SoftOrbit(loop, ad * v, 10.0 ** ltd)
    XD = donor.iterate(donor.start, args.transient)
    out = {"args": vars(args), "donor": {"alpha": ad, "log10T": ltd,
                                         "period": donor.detect_period(XD, 64, args.tol)}}
    _, out["donor"]["steps"] = record(donor, XD, 8, tok)
    out["points"] = []
    for spec in args.points:
        a, lt = (float(s) for s in spec.split(","))
        cell = esl.SoftOrbit(loop, a * v, 10.0 ** lt)
        runs = {}
        for name, X0 in (("fixed start", cell.start), ("donor start", XD.clone())):
            X = cell.iterate(X0, args.transient)
            per = cell.detect_period(X, 64, args.tol)
            lam = twin_exponent(cell, X.clone(), args.lyapunov_steps)
            _, steps = record(cell, X, args.record, tok)
            runs[name] = {"period": per, "lyapunov": lam, "state": X, "steps": steps,
                          "text": "".join(s["text"] for s in steps)}
            print(json.dumps({"alpha": a, "log10T": lt, "start": name, "period": per, "lyapunov": round(lam, 4),
                              "text": runs[name]["text"]}, ensure_ascii=False), flush=True)
        dist = float((runs["fixed start"]["state"] - runs["donor start"]["state"]).norm())
        for r in runs.values():
            del r["state"]
        out["points"].append({"alpha": a, "log10T": lt, "runs": runs, "final_state_distance": dist})
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
