"""Edge tracking of the hook inside the kite of the period-6 shrimp (human, 19 Oct 2026: "Do we have an analytical and
verified theory for the shape of the hook specifically?" ... "We have time, go for it").

Brackets (alpha, log10 T_lo, log10 T_hi) straddle a jump of the fixed-start outcome, taken from adjacent rows of a map
whose largest exponents differ by more than --jump. Every bracket is bisected in log10 T, all in one batch (per-row
steering and temperature, as in the scans): each midpoint's loop runs --transient steps from the fixed start, then a
twin (offset --eps, renormalized every step) measures the exponent over --measure steps; the midpoint joins the end of
the bracket whose exponent (measured the same way in round 0) it is closer to. Output: edge points with their final
bracket widths and end exponents.

Usage: python tools/hook_edge_track.py --map llm-soft-loop/pod64/hr12 --box 0.07295 0.07301 -0.15372 -0.15333 \
    --every 2 --rounds 20 --device cuda --out llm-soft-loop/pod64/kite/hook-edge
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
    ap.add_argument("--map", required=True, help="explore-relative scan folder (scan.npz) giving the brackets")
    ap.add_argument("--box", type=float, nargs=4, required=True, metavar=("A0", "A1", "T0", "T1"))
    ap.add_argument("--every", type=int, default=2, help="use every n-th column of the map")
    ap.add_argument("--jump", type=float, default=0.04)
    ap.add_argument("--extra", type=float, nargs="*", default=[], help="extra brackets: alpha t_lo t_hi triples")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--measure", type=int, default=200)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    z = np.load(esl.DATA / "explore" / args.map / "scan.npz")
    A, T, L = z["alpha"], z["log10_temperature"], z["lyapunov"]
    a0, a1, t0, t1 = args.box
    cols = [c for c in np.nonzero((A >= a0) & (A <= a1))[0]][:: args.every]
    rows = np.nonzero((T >= t0) & (T <= t1))[0]
    br = []
    for c in cols:
        for r in rows[:-1]:
            l0, l1 = L[r, c], L[r + 1, c]
            if l0 <= 1e-3 and l1 <= 1e-3 and abs(l1 - l0) > args.jump:
                br.append([float(A[c]), float(T[r]), float(T[r + 1])])
    ex = args.extra
    for k in range(0, len(ex) - 2, 3):
        br.append([ex[k], ex[k + 1], ex[k + 2]])
    br = np.array(br)
    print(json.dumps({"brackets": len(br)}), flush=True)

    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    torch = loop.torch
    rng = np.random.default_rng(0)

    def exponents(alphas, logts):
        out = np.zeros(len(alphas))
        for s0 in range(0, len(alphas), args.chunk):
            al, lt = alphas[s0:s0 + args.chunk], logts[s0:s0 + args.chunk]
            n = len(al)
            steer = np.array([x * v for x in al])
            temps2 = np.tile(10.0 ** lt, 2)
            XY = loop.prepare(np.concatenate([steer, steer]))
            for _ in range(args.transient):
                XY, _ = loop.step(XY, temps2)
            d = torch.as_tensor(rng.normal(size=tuple(XY[:n].shape)), dtype=loop.state_dtype, device=loop.device)
            d = d / d.flatten(1).norm(dim=1)[:, None, None] * args.eps
            XY = torch.cat([XY[:n], XY[:n] + d])
            lg = np.zeros(n)
            for _ in range(args.measure):
                XY, _ = loop.step(XY, temps2)
                X, Y = XY[:n], XY[n:]
                diff = Y - X
                dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), args.eps * 1e-30)
                lg += np.log(dist / args.eps)
                XY = torch.cat([X, X + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype,
                                                              device=loop.device)[:, None, None]])
            out[s0:s0 + n] = lg / args.measure
        return out

    tic = time.perf_counter()
    lo, hi = br[:, 1].copy(), br[:, 2].copy()
    ends = exponents(np.concatenate([br[:, 0], br[:, 0]]), np.concatenate([lo, hi]))
    l_lo, l_hi = ends[: len(br)], ends[len(br):]
    sep = np.abs(l_hi - l_lo)
    hist = []
    for k in range(args.rounds):
        mid = 0.5 * (lo + hi)
        lm = exponents(br[:, 0], mid)
        to_lo = np.abs(lm - l_lo) < np.abs(lm - l_hi)
        lo = np.where(to_lo, mid, lo)
        hi = np.where(to_lo, hi, mid)
        hist.append({"round": k, "median_width": float(np.median(hi - lo)), "seconds": round(time.perf_counter() - tic, 1)})
        print(json.dumps(hist[-1]), flush=True)
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "rounds": hist,
           "edges": [{"alpha": float(br[i, 0]), "log10T_edge": float(0.5 * (lo[i] + hi[i])), "width": float(hi[i] - lo[i]),
                      "bracket": [float(br[i, 1]), float(br[i, 2])], "lambda_lo_end": float(l_lo[i]),
                      "lambda_hi_end": float(l_hi[i]), "separation": float(sep[i])} for i in range(len(br))]}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hook-edge.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"brackets": len(br), "median_final_width": float(np.median(hi - lo)), "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
