"""Newton's method for the period-6 orbit at exactly R = (0.072912, -0.154104) from the ad-flip state (row 6 of
p6s-ad-curve-plus) and the td-flip state (row 2 of p6s-td-curve-minus), then at two body points near R. Float64, CPU,
dense forward-mode Jacobians. Prints residual histories, the orbit gap up to cyclic shift, leading multipliers, words."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/data2/neural-shrimp"); sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_phase_map as elpm
import explore_llm_soft_loop as esl

D = esl.DATA / "explore" / "llm-soft-loop/pod64/p6s"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
cv = json.loads((D / "p6s-ad-curve-plus" / "curve.json").read_text())
ta = json.loads((esl.DATA / "explore" / cv["args"]["start"] / "branch.json").read_text())["args"]
loop = esl.SoftLoop(ta["model"], ta["layer"], 20, ta["prompt"], ta["opener"], ta["window"], "float64", "cpu")
v = elpm.load_vectors(ta["vectors"])[ta["concept"]]
seeds = {}
for c, row in (("p6s-ad-curve-plus", 6), ("p6s-td-curve-minus", 2)):
    pts = json.loads((D / c / "curve.json").read_text())["points"]
    q = next(p for p in pts if p["index"] == row)
    seeds[c] = np.load(D / c / "states.npy")[row].ravel()
    print(c, "row", row, "at", q["alpha"], q["log10_temperature"], flush=True)

def solve(a, t, x):
    orb = esl.SoftOrbit(loop, a * v, 10.0 ** t)
    X = loop.torch.as_tensor(x.reshape(1, 1, -1), dtype=loop.state_dtype)
    tic = time.time()
    X, hist, vals, pts = orb.newton(X, 6, tol=1e-12, max_newton=10)
    P = np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])
    word = [int(orb.probabilities(p).double().cpu().numpy().reshape(-1).argmax()) for p in pts]
    return P, hist, vals, word, time.time() - tic

def gap(P, Q):
    return min((max(np.linalg.norm(P[i] - Q[(i + s) % 6]) for i in range(6)), s) for s in range(6))

results = []
for (a, t) in ((0.072912, -0.154104), (0.07300, -0.15390), (0.07310, -0.15370)):
    orbs = {}
    for c, x in seeds.items():
        P, hist, vals, word, sec = solve(a, t, x)
        orbs[c] = P
        rec = {"alpha": a, "log10_temperature": t, "seed": c, "residuals": hist, "seconds": round(sec, 1),
               "multipliers": [[float(z.real), float(z.imag)] for z in vals[:8]], "word": word}
        results.append(rec)
        print(json.dumps({k: rec[k] for k in ("alpha", "log10_temperature", "seed", "seconds", "word")}),
              "residuals", ["%.1e" % h for h in hist], "mu", [f"{z.real:+.5f}{z.imag:+.5f}j" for z in vals[:4]], flush=True)
        np.save(OUT / f"orbit-{a:.6f}-{t:.6f}-{c}.npy", P)
    g, s = gap(orbs["p6s-ad-curve-plus"], orbs["p6s-td-curve-minus"])
    scale = float(np.mean(np.linalg.norm(orbs["p6s-ad-curve-plus"], axis=1)))
    results.append({"alpha": a, "log10_temperature": t, "orbit_gap": g, "relative_gap": g / scale, "shift": s})
    print(f"({a}, {t}): orbit gap {g:.3e}, relative {g / scale:.2e}, shift {s}", flush=True)
    (OUT / "newton_at_R.json").write_text(json.dumps(results, indent=1))
