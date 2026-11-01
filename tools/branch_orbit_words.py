"""What each orbit along a continued branch writes, and what its plotted coordinate means (run FW2).

For every point of a branch run (branch.json, states.npy) the saved state x is iterated through one period of the
soft-token map at that point's parameters, in float64, recording each step's most probable tokens. A state is the
probability-weighted average of the token embeddings, x = E^T p, so a projection of x on a direction w is a sum over
tokens, w.x = sum_i p_i (E_i.w). The tool splits the first principal coordinate of the branch's states, the vertical
axis of the branch figure, into those token contributions.

Usage: HF_HUB_OFFLINE=1 python tools/branch_orbit_words.py --branch llm-soft-loop/pod64/kite/kite-la-branch \
           --out llm-soft-loop/branch-words/kite-la-branch
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--branch", required=True, help="explore-relative branch run")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = ela.DATA / "explore" / args.branch
    br = json.loads((src / "branch.json").read_text())
    a = br["args"]
    points = br["points"]
    x0 = np.load(src / "states.npy").reshape(len(points), -1)
    alpha = np.array([p["lambda"] for p in points])
    period = int(br["period"])
    loop = esl.SoftLoop(a["model"], a["layer"], args.threads, a["prompt"], a["opener"], a["window"], "float64", "cpu")
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(a["vectors"])[a["concept"]], dtype=np.float64)
    E = loop.E.double().cpu().numpy()
    T = 10.0 ** float(a["log_temperature"])
    n, V = len(points), E.shape[0]
    ids, ps, xs, last_p = [], [], [], []
    tic = time.perf_counter()
    for s in range(0, n, args.batch):
        rows = np.arange(s, min(s + args.batch, n))
        X = loop.prepare(alpha[rows, None] * v[None]).clone()
        X[:, -1] = torch.as_tensor(x0[rows], dtype=X.dtype, device=X.device)
        temps = np.full(len(rows), T)
        bid, bp, bx = [], [], []
        with torch.no_grad():
            for _ in range(period):
                X, q = loop.step(X, temps)
                tp, ti = torch.topk(q, args.top, dim=1)
                bid.append(ti.cpu().numpy()); bp.append(tp.double().cpu().numpy())
                bx.append(X[:, -1].double().cpu().numpy())
        ids.append(np.stack(bid, 1)); ps.append(np.stack(bp, 1)); xs.append(np.stack(bx, 1))
        last_p.append(q.double().cpu().numpy())                 # the distribution whose average is the saved state
        print(f"{rows[-1] + 1}/{n} {time.perf_counter() - tic:.0f}s", flush=True)
    ids, ps, xs, P = np.concatenate(ids), np.concatenate(ps), np.concatenate(xs), np.concatenate(last_p)
    rel = np.linalg.norm(xs[:, -1] - x0, axis=1) / np.linalg.norm(x0, axis=1)
    recon = np.linalg.norm(P @ E - x0, axis=1) / np.linalg.norm(x0, axis=1)
    mean = x0.mean(0)
    U, s, Vt = np.linalg.svd(x0 - mean, full_matrices=False)
    w = Vt[0]
    y = (x0 - mean) @ w
    pbar = P.mean(0)
    load = E @ w                                                # each token's embedding along the axis
    c = (P - pbar) * load                                       # c[i, tok]: its share of y at branch point i
    order = np.argsort(-np.abs(c).max(0))[:args.top * 3]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "words.npz", ids=ids, p=ps, states=xs, alpha=alpha, y=y, axis=w, mean=mean,
                        tokens=order, token_p=P[:, order], token_load=load[order], token_share=c[:, order],
                        rest=y - c[:, order].sum(1))
    vocab = sorted(set(ids.ravel().tolist()) | set(order.tolist()))
    rec = {"branch": args.branch, "branch_sha256": hashlib.sha256((src / "branch.json").read_bytes()).hexdigest(),
           "states_sha256": hashlib.sha256((src / "states.npy").read_bytes()).hexdigest(), "period": period,
           "points": n, "log10_temperature": float(a["log_temperature"]),
           "largest_return_gap": float(rel.max()), "largest_average_gap": float(recon.max()),
           "variance_share_pc": (s[:4] ** 2 / (s ** 2).sum()).tolist(), "y_range": [float(y.min()), float(y.max())],
           "vocab": {str(j): loop.sm.tok.decode([j]) for j in vocab}, "seconds": round(time.perf_counter() - tic, 1)}
    (out / "words.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, r in rec.items() for v in [r] if k != "vocab"}, indent=1))


if __name__ == "__main__":
    main()
