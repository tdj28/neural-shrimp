"""Each problem's own stability edge of Huginn-0125's latent recurrence (human, 15 Oct 2026: "since the broader
edge-of-chaos question remains untested <- how do we test?").

For one GSM8K prompt (8-shot chain of thought, exactly the prompt of tools/huginn_relax_eval.py accuracy --prompt-style
cot8), the relaxed prefill recurrence s <- (1 - lam) s + lam F(s) is run from the model's seeded initial state at a grid
of lam values in one batch (explore_latent_loop.HuginnLoop: adapter and core on the GPU, float32). Per lam, the native
residual r_k = ||F(s_k) - s_k|| / ||s_k|| is recorded, and its late decay rate rho(lam) is fitted over the second half
of the run. Relaxation keeps the fixed point and maps each multiplier mu to 1 - lam + lam mu, so rho(lam) rises to 1 at
the lam where the fixed point loses stability: the problem's edge lam_e.

Coarse pass: lam on a grid; the largest lam whose fitted rate is below 1 - margin and whose residual still falls, and
the smallest lam above it whose rate is not, bracket the edge. Fine pass: a finer grid inside the bracket with a
longer run. lam_e is the zero of log rho(lam) extrapolated linearly from the two largest converging fine-grid values,
and must lie inside the fine bracket (gate). Outcomes that do not meet a gate are reported as such and never replaced:
"native unstable" (lam = 1 does not converge), "no edge in range", "extrapolation outside the bracket".

No answer is generated or scored: the edges are measured before any outcome of a test that uses them.

Usage: python tools/huginn_edge.py --split test --start 200 --count 100 --out huginn-edge/test-200
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def fit_rate(r, lo):
    """Late decay rate of a residual trace: exp of the least-squares slope of log r over steps lo..end (finite, positive
    values only); None when fewer than 8 usable points."""
    k = np.arange(len(r))[lo:]
    v = np.asarray(r[lo:], dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 8:
        return None
    return float(math.exp(np.polyfit(k[ok], np.log(v[ok]), 1)[0]))


def run_grid(loop, h0, lams, steps):
    """Native residual traces (len(lams) x steps) of the relaxed recurrence, all lam values in one batch."""
    torch = loop.torch
    lam = torch.tensor(lams, dtype=h0.dtype, device=h0.device)[:, None, None]
    s = h0.expand(len(lams), -1, -1).clone()
    out = np.zeros((len(lams), steps))
    for k in range(steps):
        f = loop.F(s)
        d = (f - s).float().flatten(1).norm(dim=1) / s.float().flatten(1).norm(dim=1)
        out[:, k] = d.cpu().numpy()
        s = (1.0 - lam) * s + lam * f
    return out


def classify(r, rate, margin):
    """Converging: fitted late rate below 1 - margin and the residual at the end below its value at mid-run."""
    return bool(rate is not None and rate < 1.0 - margin and r[-1] < r[len(r) // 2])


def edge_of(loop, h0, args):
    coarse = np.round(np.arange(args.lam_lo, args.lam_hi + 1e-9, args.coarse_step), 6)
    R = run_grid(loop, h0, coarse.tolist(), args.coarse_steps)
    rates = [fit_rate(r, args.coarse_steps // 2) for r in R]
    conv = [classify(r, q, args.margin) for r, q in zip(R, rates)]
    rec = {"coarse": {"lam": coarse.tolist(), "rate": rates, "converging": conv,
                      "residual_end": R[:, -1].tolist()}}
    if not conv[0]:
        rec["outcome"] = f"native unstable or no convergence at lam {coarse[0]}"
        return rec
    first_bad = next((i for i, c in enumerate(conv) if not c), None)
    if first_bad is None:
        rec["outcome"] = f"no edge in range (converging up to lam {coarse[-1]})"
        return rec
    if not all(conv[:first_bad]):
        rec["outcome"] = "non-monotone convergence on the coarse grid"
        return rec
    lo, hi = float(coarse[first_bad - 1]), float(coarse[first_bad])
    fine = np.linspace(lo, hi, args.fine_points)
    Rf = run_grid(loop, h0, fine.tolist(), args.fine_steps)
    rf = [fit_rate(r, args.fine_steps // 2) for r in Rf]
    cf = [classify(r, q, args.margin) for r, q in zip(Rf, rf)]
    rec["fine"] = {"lam": fine.tolist(), "rate": rf, "converging": cf, "residual_end": Rf[:, -1].tolist()}
    if not cf[0] or cf[-1]:
        rec["outcome"] = "fine grid does not bracket the edge"
        return rec
    j = next(i for i, c in enumerate(cf) if not c)
    if not all(cf[:j]):
        rec["outcome"] = "non-monotone convergence on the fine grid"
        return rec
    b_lo, b_hi = float(fine[j - 1]), float(fine[j])
    rec["bracket"] = [b_lo, b_hi]
    if j >= 2:                                   # zero of log rho from the two largest converging values
        x1, x2 = float(fine[j - 2]), float(fine[j - 1])
        y1, y2 = math.log(rf[j - 2]), math.log(rf[j - 1])
        lam_e = x2 - y2 * (x2 - x1) / (y2 - y1) if y2 != y1 else float("nan")
    else:
        lam_e = float("nan")
    rec["lam_edge_extrapolated"] = lam_e
    if np.isfinite(lam_e) and b_lo <= lam_e <= b_hi + (b_hi - b_lo):
        rec["lam_edge"] = lam_e
        rec["outcome"] = "edge located"
    else:
        rec["lam_edge"] = 0.5 * (b_lo + b_hi)
        rec["outcome"] = "extrapolation outside the bracket; bracket midpoint reported"
    return rec


def main():
    import explore_latent_loop as ell
    import explore_llm_attractors as ela
    import huginn_relax_eval as hre

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--lam-lo", type=float, default=1.0)
    ap.add_argument("--lam-hi", type=float, default=1.28)
    ap.add_argument("--coarse-step", type=float, default=0.04)
    ap.add_argument("--coarse-steps", type=int, default=96)
    ap.add_argument("--fine-points", type=int, default=6)
    ap.add_argument("--fine-steps", type=int, default=192)
    ap.add_argument("--margin", type=float, default=1e-3, help="rate below 1 - margin counts as converging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    loop = ell.HuginnLoop("huginn", args.dtype, args.device, args.threads, seed=args.seed)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    path = out / "edges.jsonl"
    if path.exists():                            # resume: problems already measured are kept
        done = {json.loads(line)["index"] for line in path.read_text().splitlines()}
    with open(path, "a") as f:
        for index in range(args.start, args.start + args.count):
            if index in done:
                continue
            t0 = time.perf_counter()
            h0 = loop.setup(hre.prompt_text("gsm8k-cot8", args.split, index))
            rec = {"split": args.split, "index": index, "tokens": len(loop.token_ids), **edge_of(loop, h0, args)}
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps({k: rec.get(k) for k in ("index", "outcome", "lam_edge", "bracket", "seconds")}),
                  flush=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))


if __name__ == "__main__":
    main()
