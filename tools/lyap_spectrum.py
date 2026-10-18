"""Leading Lyapunov exponents and the Kaplan-Yorke dimension of the soft-token map at one cell (for the attractor
portraits, 18 Oct 2026).

From the fixed start, --transient steps, then --steps steps carrying --k tangent vectors by forward-mode
Jacobian-vector products of the full map (SoftOrbit.jvp_rows), re-orthonormalized by QR after every step. The
exponents are the mean logarithms of the diagonal of R. D_KY = j + (lambda_1 + ... + lambda_j) / |lambda_(j+1)|, with
j the largest index whose partial sum is nonnegative. Float64. Output: DATA/explore/OUT/spectrum.json (also the
running estimates every 500 steps).

Usage: python tools/lyap_spectrum.py --alpha 0.07248 --log-t -0.148 --device cuda --out llm-soft-loop/portraits/strong-spec
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def kaplan_yorke(lam):
    s = np.cumsum(lam)
    j = int(np.max(np.nonzero(s >= 0)[0])) + 1 if s[0] >= 0 else 0
    if j == 0:
        return 0.0
    if j >= len(lam):
        return float(len(lam))
    return float(j + s[j - 1] / abs(lam[j]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    orb = esl.SoftOrbit(loop, args.alpha * v, 10.0 ** args.log_t, chunk=args.k)
    X = orb.iterate(orb.start, args.transient)
    g = torch.Generator(device=X.device).manual_seed(0)
    V = torch.randn((args.k,) + tuple(X.shape[1:]), generator=g, dtype=X.dtype, device=X.device)
    Qm, _ = torch.linalg.qr(V.reshape(args.k, -1).T)
    V = Qm.T.reshape(V.shape)
    sums = np.zeros(args.k)
    running = []
    for i in range(1, args.steps + 1):
        W = orb.jvp_rows(X, V)
        Qm, R = torch.linalg.qr(W.reshape(args.k, -1).T)
        sums += np.log(np.abs(torch.diagonal(R).double().cpu().numpy()))
        V = Qm.T.reshape(V.shape).to(X.dtype)
        X = orb.f(X)
        if i % 500 == 0:
            lam = sums / i
            running.append({"step": i, "exponents": lam.tolist(), "kaplan_yorke": kaplan_yorke(lam)})
            print(json.dumps(running[-1]), flush=True)
    lam = sums / args.steps
    rec = {"args": vars(args), "exponents": lam.tolist(), "kaplan_yorke": kaplan_yorke(lam), "running": running,
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "spectrum.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("exponents", "kaplan_yorke", "seconds")}))


if __name__ == "__main__":
    main()
