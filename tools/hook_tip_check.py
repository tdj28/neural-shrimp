"""Where is the tip of the hook? A test of the saddle-node picture at Newton-converged points of the kite's fold curve.

Theory: at a fold point the center coordinate u of the merged (saddle-node) orbit evolves by u' = u + b u^2. The fixed
start, after its fast transient, lands at u0 = l.(x - p0)/(l.r) (r, l: right and left eigenvectors of the multiplier
+1 of the six-step monodromy at the orbit point p0). On one side of the point where u0 = 0 it converges (slowly) to the
semi-stable orbit, on the other it escapes, toward the strongly contracting orbit; inside the coexistence region next
to the first side the start reaches the weaker orbit. So the tip of the region that reaches the weaker orbit is the
fold point where u0 changes sign.

For each chosen point of the continued fold curve (kite-la-curve-plus): the orbit from the stored state, r by power
iteration with forward-mode JVPs and l with reverse-mode VJPs of the six-step map (the +1 multiplier dominates at the
fold), the trajectory from the fixed start (--steps steps), u_m at successive passes near p0, the outcome (final
distance to the orbit), and b from the drift u_{m+1} - u_m against u_m^2. The tip is predicted by linear interpolation
of u0 in alpha between neighboring points of opposite sign.

Usage: python tools/hook_tip_check.py --points 19 20 21 --device cuda --out llm-soft-loop/kite/hook-tip
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def landing(loop, v, a, lt, p0_np, steps=1200, power=6):
    """At one fold point (alpha a, log10 T lt; orbit point p0_np): the six-step orbit's return, r and l of the +1
    multiplier, and the fixed start's landing coordinate u0 with its drift and outcome."""
    import explore_llm_soft_loop as esl
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch = loop.torch
    orb = esl.SoftOrbit(loop, a * v, 10.0 ** lt, jacobian_method="ad")
    p0 = torch.as_tensor(np.asarray(p0_np).reshape((1,) + orb.shape), dtype=loop.state_dtype, device=loop.device)
    pts = [p0]
    for _ in range(6):
        pts.append(orb.f(pts[-1]))
    scale = float(p0.norm())
    ret = float((pts[6] - p0).norm()) / scale

    def jvp6(x, w):
        for _ in range(6):
            w = orb.jvp_rows(x, w)
            x = orb.f(x)
        return w

    def f_ad(x):                                   # one step, differentiable, on a fresh shallow cache copy
        orb._use(1)
        base = orb.caches[1]
        cache = copy.copy(base)
        cache.layers = [copy.copy(layer) for layer in base.layers]
        loop.sm.steer = orb.steer1
        try:
            z = loop.model(inputs_embeds=x.to(loop.dtype), past_key_values=cache, use_cache=True,
                           logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
        finally:
            loop.sm.steer = None
        e = torch.softmax(z / orb.T, dim=-1) @ loop.E
        return torch.cat([x[:, 1:, :], e[:, None, :]], dim=1)

    def f6(x):
        for _ in range(6):
            x = f_ad(x)
        return x

    g = torch.Generator(device="cpu").manual_seed(0)
    r = torch.randn(p0.shape, generator=g, dtype=torch.float64).to(loop.device, loop.state_dtype)
    mus = []
    for _ in range(power):
        w = jvp6(p0, r)
        mus.append(float((w * r).sum() / (r * r).sum()))
        r = w / w.norm()
    with sdpa_kernel(SDPBackend.MATH):
        _, vjp_fn = torch.func.vjp(f6, p0)
        lv = torch.randn(p0.shape, generator=g, dtype=torch.float64).to(loop.device, loop.state_dtype)
        mus_l = []
        for _ in range(power):
            w = vjp_fn(lv)[0]
            mus_l.append(float((w * lv).sum() / (lv * lv).sum()))
            lv = w / w.norm()
    lr = float((lv * r).sum())
    X = orb.start
    traj = []
    for _ in range(steps):
        X = orb.f(X)
        traj.append(X.reshape(-1).double().cpu().numpy())
    traj = np.array(traj)
    P6 = np.stack([q.reshape(-1).double().cpu().numpy() for q in pts[:6]])
    d0 = np.linalg.norm(traj - P6[0], axis=1) / scale
    phase = int(np.argmin([np.mean(d0[n::6][2:20]) for n in range(6)]))
    ns = np.arange(phase, steps, 6)
    lnp = lv.reshape(-1).double().cpu().numpy()
    u = (traj[ns] - P6[0]) @ lnp / lr
    first = int(np.nonzero(ns >= 24)[0][0])
    dfin = float(np.min(np.linalg.norm(traj[-1] - P6, axis=1)) / scale)
    um, du = u[first:-1], np.diff(u[first:])
    ok = np.abs(um) < 0.5 * np.abs(u[first]) + 1e-30
    b_fit = float(np.sum(du[ok] * um[ok] ** 2) / np.sum(um[ok] ** 4)) if ok.sum() >= 3 else None
    return {"alpha": a, "log10T": lt, "orbit_return": ret, "mu_r": mus[-1], "mu_l": mus_l[-1],
            "mu_r_history": mus, "l_dot_r": lr, "phase": phase, "u0": float(u[first]), "u_series": u[:60].tolist(),
            "pass_steps": ns[:60].tolist(), "final_distance_to_orbit": dfin, "b_from_drift": b_fit,
            "distance_series": d0[::6][:200].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curve", default="llm-soft-loop/pod64/kite/kite-la-curve-plus")
    ap.add_argument("--points", type=int, nargs="+", default=[19, 20, 21])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--power", type=int, default=6, help="power iterations for r and l")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    src = esl.DATA / "explore" / args.curve
    cur = json.loads((src / "curve.json").read_text())
    states = np.load(src / "states.npy")
    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    tic = time.perf_counter()
    rows = []
    for k in args.points:
        P = cur["points"][k]
        row = landing(loop, v, float(P["alpha"]), float(P["log10_temperature"]), states[k], args.steps, args.power)
        row["point"] = k
        rows.append(row)
        print(json.dumps({k2: row[k2] for k2 in ("point", "alpha", "log10T", "orbit_return", "mu_r", "mu_l", "l_dot_r",
                                                  "u0", "final_distance_to_orbit", "b_from_drift")}), flush=True)
    rows.sort(key=lambda r: r["alpha"])
    tips = []
    for x, y in zip(rows[:-1], rows[1:]):
        if x["u0"] * y["u0"] < 0:
            w = x["u0"] / (x["u0"] - y["u0"])
            tips.append({"between": [x["point"], y["point"]], "alpha_tip": x["alpha"] + w * (y["alpha"] - x["alpha"]),
                         "log10T_tip": x["log10T"] + w * (y["log10T"] - x["log10T"])})
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "points": rows, "predicted_tips": tips}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hook-tip.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"predicted_tips": tips, "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
