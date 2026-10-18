"""Run LT (exploratory): are the paper's finite-horizon chaos labels long transients?

The maps label a cell chaotic after 300 transient and 300 measured steps (exponent > 1e-3, no period up to 32). A long
chaotic transient that ends on a periodic orbit would carry the same label. For labelled cells (a cells file with
"cells" [[log10 T, alpha], ...], "source", "label" and "map_exponent"), this follows each cell from the fixed start to
--horizon steps in float64 and records when, if ever, a periodic orbit captures it:
- every --check-every steps (and at --first-check) it looks for the smallest period q <= --max-period with
  |x_t - x_(t-kq)| < tol * max(1, |x_t|) for k = 1..--reps (full-state recurrence) in a ring buffer of the last
  reps * max_period + 1 states. The first check that finds one gives the capture step (resolution --check-every),
  with the period and the leading-token loop;
- at the horizon every cell gets a twin exponent over --twin-steps (offset --eps, renormalized after every step).
Cells never captured are right-censored at the horizon. S(t) is the fraction not captured by step t.

Output: DATA/explore/OUT/summary.json (rewritten at every --save-every steps, so a timeout keeps the record).

Usage: python tools/long_transients.py --cells-file docs/lt-cells.json --source cross-512 --device cuda \
         --out llm-soft-loop/pod64/lt/cross-512
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
from explore_llm_soft_loop import DATA, SoftLoop  # noqa: E402

HORIZONS = (600, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000)


def survival(capture, horizon, groups):
    out = {}
    for name, m in groups.items():
        n = int(m.sum())
        rows = []
        for t in HORIZONS:
            if t > horizon:
                break
            captured = int(((capture > 0) & (capture <= t) & m).sum())
            rows.append({"step": t, "not_captured": n - captured, "of": n})
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-file", required=True)
    ap.add_argument("--source", default=None, help="keep only the cells of this source map")
    ap.add_argument("--label", default="chaotic")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--horizon", type=int, default=100000)
    ap.add_argument("--first-check", type=int, default=600)
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--max-period", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--twin-steps", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ring = args.reps * args.max_period + 1
    if args.first_check < ring:
        raise SystemExit(f"--first-check must be at least reps * max_period + 1 = {ring}")
    path = Path(args.cells_file)
    if not path.is_absolute() and not path.exists():
        path = ROOT / args.cells_file
    spec = json.loads(path.read_text())
    idx = [i for i in range(len(spec["cells"])) if spec["label"][i] == args.label
           and (args.source is None or spec["source"][i] == args.source)]
    if not idx:
        raise SystemExit("no cells selected")
    cells = [tuple(spec["cells"][i]) for i in idx]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch, dev = loop.torch, loop.device
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    steer = np.array([al * v for _, al in cells])
    temps = np.array([10.0 ** lt for lt, _ in cells])
    X = loop.prepare(steer)
    n, d = len(cells), X.shape[2]
    buf = torch.zeros((ring, n, d), dtype=torch.float64, device=dev)
    tok = torch.zeros((ring, n), dtype=torch.long, device=dev)
    capture = np.full(n, -1, dtype=np.int64)
    period = np.zeros(n, dtype=np.int64)
    loops = [None] * n
    decode = loop.sm.tok.decode
    meta = {"source": [spec["source"][i] for i in idx], "origin": [spec.get("origin", ["-"] * len(spec["cells"]))[i]
                                                                     for i in idx],
            "map_exponent": [float(spec["map_exponent"][i]) for i in idx]}
    mexp = np.array(meta["map_exponent"])
    terc = np.quantile(mexp, [1 / 3, 2 / 3])
    groups = {"all": np.ones(n, bool), "low_exponent": mexp <= terc[0],
              "mid_exponent": (mexp > terc[0]) & (mexp <= terc[1]), "high_exponent": mexp > terc[1]}
    for s in sorted(set(meta["source"])):
        groups[s] = np.array([x == s for x in meta["source"]])

    def write(step, final=None):
        rep = {"args": vars(args), "cells_file_sha_note": str(path), "n": n, "step": step,
               "seconds": round(time.perf_counter() - tic, 1), "exponent_tercile_edges": terc.tolist(),
               "survival": survival(capture, step, groups),
               "cells": [{"log10_temperature": c[0], "alpha": c[1], "source": meta["source"][k],
                          "origin": meta["origin"][k], "map_exponent": meta["map_exponent"][k],
                          "capture_step": int(capture[k]) if capture[k] > 0 else None,
                          "period": int(period[k]) if capture[k] > 0 else None,
                          "loop": loops[k]} for k, c in enumerate(cells)]}
        if final is not None:
            rep.update(final)
            for k, c in enumerate(rep["cells"]):
                c["lyapunov_at_horizon"] = float(final["lyapunov_at_horizon"][k])
            del rep["lyapunov_at_horizon"]
        (out / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    with torch.no_grad():
        for t in range(1, args.horizon + 1):
            X, p = loop.step(X, temps)
            buf[t % ring] = X[:, -1, :]
            tok[t % ring] = p.argmax(-1)
            if t == args.first_check or (t > args.first_check and t % args.check_every == 0):
                cur = buf[t % ring]
                scale = torch.clamp(cur.norm(dim=1), min=1.0) * args.tol
                found = torch.zeros(n, dtype=torch.long, device=dev)
                for q in range(args.max_period, 0, -1):          # descending, so the smallest period wins
                    ok = torch.ones(n, dtype=torch.bool, device=dev)
                    for k in range(1, args.reps + 1):
                        ok &= (buf[(t - k * q) % ring] - cur).norm(dim=1) < scale
                    found = torch.where(ok, torch.full_like(found, q), found)
                f = found.cpu().numpy()
                new = np.flatnonzero((capture < 0) & (f > 0))
                if len(new):
                    tk = tok.cpu().numpy()
                    for c in new:
                        capture[c], period[c] = t, f[c]
                        loops[c] = "".join(decode([int(tk[(t - j) % ring, c])]) for j in range(f[c] - 1, -1, -1))
            if t % args.save_every == 0:
                write(t)
                print(json.dumps({"step": t, "captured": int((capture > 0).sum()), "of": n,
                                  "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
        # twin exponent at the horizon, for every cell
        loop.prepare(np.concatenate([steer, steer]))
        temps2 = np.tile(temps, 2)
        rng = np.random.default_rng(0)
        delta = torch.as_tensor(rng.normal(size=tuple(X.shape)), dtype=loop.state_dtype, device=dev)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * args.eps
        XY = torch.cat([X, X + delta])
        growth = np.zeros(n)
        for _ in range(args.twin_steps):
            XY, _ = loop.step(XY, temps2)
            Xa, Y = XY[:n], XY[n:]
            diff = Y - Xa
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), args.eps * 1e-30)
            growth += np.log(dist / args.eps)
            Y = Xa + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=dev)[:, None, None]
            XY = torch.cat([Xa, Y])
    lyap = growth / args.twin_steps
    alive = capture < 0
    write(args.horizon, {"lyapunov_at_horizon": lyap.tolist(),
                         "never_captured": int(alive.sum()),
                         "never_captured_with_positive_exponent": int((alive & (lyap > 1e-3)).sum()),
                         "never_captured_without_positive_exponent": int((alive & (lyap <= 1e-3)).sum())})
    print(json.dumps({"done": True, "captured": int((capture > 0).sum()), "of": n,
                      "seconds": round(time.perf_counter() - tic, 1)}), flush=True)


if __name__ == "__main__":
    main()
