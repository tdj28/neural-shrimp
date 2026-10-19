"""Exit time from the kite's saddle orbit near the hook's lower edge (follow-up to tools/hook_saddle_check.py).

hook_saddle_check.py located the edge, found the saddle (one unstable multiplier mu_u) and measured settling times as
the last step farther than 1e-6 from reference orbits computed at another temperature; that offset (1.6e-6) exceeds
the threshold, so every settling time reached the step cap. Here the clock is the saddle itself.

Rows at log10 T = edge -/+ delta (both sides, one batch, the fixed start, --steps steps each). Per row and step: the
distance to the saddle orbit (min over its six points, relative to the norm of its first point). The exit time is the
last step within --radius of the saddle; one unstable multiplier predicts exit = c + (6 / ln mu_u) ln(1/delta), since
the start lands at a distance proportional to delta from the saddle's stable manifold and the offset grows by mu_u per
period. The closest approach is predicted proportional to delta (the stable multipliers are tiny). The end state is
classified by the nearer of the end states of the two outermost rows, one on each side.

Usage: python tools/hook_saddle_exit.py --saddle llm-soft-loop/pod64/kite/hook-saddle --device cuda \
    --out llm-soft-loop/kite/hook-saddle-exit
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
    ap.add_argument("--saddle", required=True, help="explore-relative folder with hook-saddle.json and saddle-orbit.npy")
    ap.add_argument("--exponents", type=float, nargs="*", default=[6 + 0.5 * k for k in range(17)],
                    help="delta = 10^-e for each e")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--radius", type=float, nargs="*", default=[1e-3, 3e-4])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    src = esl.DATA / "explore" / args.saddle
    rec0 = json.loads((src / "hook-saddle.json").read_text())
    saddle = np.load(src / "saddle-orbit.npy")                       # (6, n)
    a, edge = rec0["args"]["alpha"], rec0["edge_log10T"]
    mods = np.hypot(*np.array(rec0["saddle"]["multipliers"]).T)
    if rec0["saddle"]["unstable_count"] != 1 or not rec0["saddle"]["converged"]:
        raise SystemExit("the saddle record does not hold a converged orbit with exactly one unstable multiplier")
    mu_u = float(mods[0])
    deltas = 10.0 ** -np.array(args.exponents)
    sides = np.array([-1] * len(deltas) + [1] * len(deltas))        # -1: below the edge (S side), +1: above (W side)
    dl = np.concatenate([deltas, deltas])
    logt = edge + sides * dl
    B = len(logt)

    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    torch = loop.torch
    S = torch.as_tensor(saddle, dtype=loop.state_dtype, device=loop.device)
    scale = float(np.linalg.norm(saddle[0]))
    tic = time.perf_counter()
    X = loop.prepare(np.array([a * v] * B))
    temps = 10.0 ** logt
    dsad = np.zeros((args.steps, B))
    tail = []
    for k in range(args.steps):
        X, _ = loop.step(X, temps)
        flat = X.reshape(B, -1)
        # direct differences: cdist's |x|^2 + |y|^2 - 2 x.y form cancels below ~1e-8 relative (first run: zeros)
        dsad[k] = (torch.cdist(flat, S, compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values / scale).double().cpu().numpy()
        if k >= args.steps - 7:
            tail.append(flat.double().cpu().numpy())
    tail = np.array(tail)                                            # (7, B, n): the last seven states
    ret6 = np.linalg.norm(tail[-1] - tail[0], axis=1) / scale
    end = tail[-1]
    ref_lo, ref_hi = end[0], end[len(deltas)]                        # outermost rows: delta = 10^-e[0] on each side
    d_lo = np.linalg.norm(end - ref_lo, axis=1) / scale
    d_hi = np.linalg.norm(end - ref_hi, axis=1) / scale
    rows = []
    for i in range(B):
        r = {"side": "below" if sides[i] < 0 else "above", "delta": float(dl[i]), "log10T": float(logt[i]),
             "ends_with": "below-end" if d_lo[i] < d_hi[i] else "above-end", "end_gap": float(min(d_lo[i], d_hi[i])),
             "end_return6": float(ret6[i]), "closest": float(dsad[:, i].min()), "closest_step": int(dsad[:, i].argmin())}
        for rad in args.radius:
            inside = np.nonzero(dsad[:, i] < rad)[0]
            r[f"enter_{rad:g}"] = int(inside[0]) if len(inside) else None
            r[f"exit_{rad:g}"] = int(inside[-1]) if len(inside) else None
        rows.append(r)
    fits = {}
    pred = 6.0 / np.log(mu_u)
    for side in ("below", "above"):
        for rad in args.radius:
            use = [r for r in rows if r["side"] == side and r[f"exit_{rad:g}"] is not None
                   and r["closest"] < rad / 10 and r[f"exit_{rad:g}"] < args.steps - 1]
            if len(use) >= 3:
                x = np.log(1 / np.array([r["delta"] for r in use]))
                y = np.array([r[f"exit_{rad:g}"] for r in use], dtype=float)
                p = np.polyfit(x, y, 1)
                fits[f"{side}_{rad:g}"] = {"rows": len(use), "slope": float(p[0]), "intercept": float(p[1]),
                                           "max_abs_residual": float(np.abs(np.polyval(p, x) - y).max())}
        use = [r for r in rows if r["side"] == side]
        x, y = np.log(np.array([r["delta"] for r in use])), np.log(np.array([r["closest"] for r in use]))
        fits[f"{side}_closest_vs_delta"] = {"slope": float(np.polyfit(x, y, 1)[0]), "prediction": 1.0}
    rec = {"args": vars(args), "source": {"alpha": a, "edge_log10T": edge, "mu_u": mu_u,
                                          "next_modulus": float(mods[1])},
           "predicted_slope_steps_per_e_fold": pred, "seconds": round(time.perf_counter() - tic, 1), "rows": rows,
           "fits": fits}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hook-saddle-exit.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(out / "distance-to-saddle.npz", distance=dsad, log10T=logt, delta=dl, side=sides)
    print(json.dumps({"predicted_slope": pred, "fits": fits, "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
