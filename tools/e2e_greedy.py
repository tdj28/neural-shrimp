"""Run E2E, Q1: the zero-temperature successor map g(t) = argmax_s z_s(prefix, E_t) of the soft-token map for the
tokens of the windows, as the steering strength alpha varies (Qwen2.5-0.5B-Instruct, philosophy at layer 12, float64).

scan    for every alpha of a grid: g, the decision margin (top-1 minus top-2 logit) and the five leading tokens for
        each listed token fed back as its own embedding; the alpha intervals where the greedy cycles
        ' is' -> ' it' -> ' is' and '\\n\\n' -> 'The' -> ' existence' -> '\\n\\n' exist
refine  the endpoints of those intervals by root finding on the margin of the switching token (brentq on the
        continuous function z_target(E_src) - max_{s != target} z_s(E_src)), with the token that takes over
cycles the cycles of g reached from the listed tokens at given alphas (successors computed on demand)

Usage: python tools/e2e_greedy.py scan   --alpha 0.0 0.12 --n 241 --out greedy-scan
       python tools/e2e_greedy.py refine --scan greedy-scan --out greedy-refine
       python tools/e2e_greedy.py cycles --alphas 0.03 0.065 0.07 --out greedy-cycles
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_model as em  # noqa: E402

LISTED = ["is", "it", "nn", "The", "existence", "question", ".", "This", "there", "a"]
CYCLES = {"is-it": [("is", "it"), ("it", "is")],
          "is-there": [("is", "there"), ("there", "is")],
          "nn-The-existence": [("nn", "The"), ("The", "existence"), ("existence", "nn")],
          "nn-The-question": [("nn", "The"), ("The", "question"), ("question", "nn")]}


def logits_for(model, alphas, token_ids, chunk=64):
    """z(prefix steered by alpha, E_t) for every alpha and listed token: array (len(alphas), len(tokens), V)."""
    loop, torch = model.loop, model.torch
    out = []
    for s in range(0, len(alphas), chunk):
        a = np.asarray(alphas[s:s + chunk])
        loop.prepare(a[:, None] * model.v[None])
        rows = []
        for t in token_ids:
            X = torch.as_tensor(np.repeat(model.E[t][None, None], len(a), axis=0), dtype=torch.float64)
            rows.append(loop.logits(X).cpu().numpy())
        out.append(np.stack(rows, axis=1))
    return np.concatenate(out)


def margin(z, target):
    """z_target - max over the other tokens (positive: target is the argmax)."""
    other = np.delete(z, target).max()
    return float(z[target] - other)


def scan(args):
    model = em.Model(args.threads)
    alphas = np.linspace(args.alpha[0], args.alpha[1], args.n)
    ids = [em.TOKENS[k] for k in LISTED]
    tic = time.perf_counter()
    rows = []
    gmat = np.zeros((len(alphas), len(ids)), dtype=np.int64)
    marg = np.zeros((len(alphas), len(ids)))
    for s in range(0, len(alphas), args.chunk):
        Z = logits_for(model, alphas[s:s + args.chunk], ids, chunk=args.chunk)
        for i in range(Z.shape[0]):
            rec = {"alpha": float(alphas[s + i])}
            for j, name in enumerate(LISTED):
                z = Z[i, j]
                top = np.argsort(-z)[:5]
                gmat[s + i, j] = top[0]
                marg[s + i, j] = z[top[0]] - z[top[1]]
                rec[name] = {"g": model.decode(top[0]), "margin": float(marg[s + i, j]),
                             "top5": [[model.decode(t), float(z[t])] for t in top]}
            rows.append(rec)
        print(json.dumps({"done": s + Z.shape[0], "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    exists = {}
    for name, edges in CYCLES.items():
        ok = np.ones(len(alphas), dtype=bool)
        for a, b in edges:
            ok &= gmat[:, LISTED.index(a)] == em.TOKENS[b]
        exists[name] = ok
    intervals = {}
    for name, ok in exists.items():
        runs, i = [], 0
        while i < len(ok):
            if ok[i]:
                j = i
                while j + 1 < len(ok) and ok[j + 1]:
                    j += 1
                runs.append([float(alphas[i]), float(alphas[j])])
                i = j + 1
            else:
                i += 1
        intervals[name] = runs
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "greedy.npz", alpha=alphas, g=gmat, margin=marg, listed=np.array(ids))
    summary = {"args": vars(args), "listed": {k: em.TOKENS[k] for k in LISTED}, "grid_intervals": intervals,
               "seconds": round(time.perf_counter() - tic, 1), "rows": rows}
    (out / "greedy-scan.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({"intervals": intervals}))


def refine(args):
    from scipy.optimize import brentq

    model = em.Model(args.threads)
    src = json.loads((em.OUT / args.scan / "greedy-scan.json").read_text())
    alphas = np.array([r["alpha"] for r in src["rows"]])
    res = {"args": vars(args), "endpoints": []}
    tic = time.perf_counter()
    for name, runs in src["grid_intervals"].items():
        edges = CYCLES[name]
        for lo, hi in runs:
            for side, inside in (("left", lo), ("right", hi)):
                i = int(np.argmin(np.abs(alphas - inside)))
                j = i - 1 if side == "left" else i + 1
                if j < 0 or j >= len(alphas):
                    res["endpoints"].append({"cycle": name, "side": side, "alpha": None,
                                             "note": "interval reaches the end of the scanned grid"})
                    continue
                outside = alphas[j]
                # which edge of the cycle breaks just outside
                Zo = logits_for(model, [outside], [em.TOKENS[a] for a, _ in edges])[0]
                broken = [k for k, (a, b) in enumerate(edges) if margin(Zo[k], em.TOKENS[b]) < 0]
                if not broken:
                    raise SystemExit(f"{name} {side}: no broken edge at alpha {outside}")
                found = []
                for k in broken:
                    a, b = edges[k]

                    def f(al, a=a, b=b):
                        return margin(logits_for(model, [al], [em.TOKENS[a]])[0, 0], em.TOKENS[b])

                    root = brentq(f, min(inside, outside), max(inside, outside), xtol=args.xtol)
                    z = logits_for(model, [root + (1 if side == "right" else -1) * 2 * args.xtol],
                                   [em.TOKENS[a]])[0, 0]
                    top = np.argsort(-z)[:2]
                    found.append({"edge": [a, b], "alpha": float(root),
                                  "taken_over_by": model.decode(top[0]) if top[0] != em.TOKENS[b] else
                                  model.decode(top[1])})
                first = min(found, key=lambda r: abs(r["alpha"] - inside))
                rec = {"cycle": name, "side": side, "alpha": first["alpha"], "edge": first["edge"],
                       "taken_over_by": first["taken_over_by"], "candidates": found}
                res["endpoints"].append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
    res["seconds"] = round(time.perf_counter() - tic, 1)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "greedy-refine.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def cycles(args):
    """Cycles of g reached from the listed tokens, at each alpha of --alphas or of a grid (--grid A0 A1 N); the
    prefix is prepared once per alpha and successors are computed on demand."""
    model = em.Model(args.threads)
    loop, torch = model.loop, model.torch
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    alphas = list(np.linspace(*args.grid[:2], int(args.grid[2]))) if args.grid else args.alphas
    res = {"args": vars(args), "alphas": []}
    for al in alphas:
        loop.prepare((al * model.v)[None])
        succ = {}

        def g(t):
            if t not in succ:
                X = torch.as_tensor(model.E[t][None, None], dtype=torch.float64)
                z = loop.logits(X).cpu().numpy()[0]
                top = np.argsort(-z)[:2]
                succ[t] = (int(top[0]), float(z[top[0]] - z[top[1]]), int(top[1]))
            return succ[t][0]

        found = {}
        for name in LISTED:
            path = [em.TOKENS[name]]
            while len(path) < args.max_steps and path[-1] not in path[:-1]:
                path.append(g(path[-1]))
            if path[-1] in path[:-1]:
                i = path.index(path[-1])
                cyc = path[i:-1]
                k = min(range(len(cyc)), key=lambda j: cyc[j])
                cyc = cyc[k:] + cyc[:k]
                key = " ".join(map(str, cyc))
                found.setdefault(key, {"tokens": [model.decode(t) for t in cyc], "ids": cyc,
                                       "margins": [succ[t][1] for t in cyc],
                                       "runners_up": [model.decode(succ[t][2]) for t in cyc], "reached_from": []})
                found[key]["reached_from"].append(name)
        rec = {"alpha": float(al), "cycles": list(found.values())}
        res["alphas"].append(rec)
        print(json.dumps({"alpha": float(al), "cycles": [c["tokens"] for c in found.values()]}, ensure_ascii=False),
              flush=True)
    (out / "greedy-cycles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--alpha", type=float, nargs=2, default=[0.0, 0.12])
    s.add_argument("--n", type=int, default=241)
    s.add_argument("--chunk", type=int, default=64)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--out", required=True)
    r = sub.add_parser("refine")
    r.add_argument("--scan", required=True)
    r.add_argument("--xtol", type=float, default=1e-9)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--out", required=True)
    c = sub.add_parser("cycles")
    c.add_argument("--alphas", type=float, nargs="+", default=None)
    c.add_argument("--grid", type=float, nargs=3, default=None, metavar=("A0", "A1", "N"))
    c.add_argument("--max-steps", type=int, default=40)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"scan": scan, "refine": refine, "cycles": cycles}[args.cmd](args)


if __name__ == "__main__":
    main()
