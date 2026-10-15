"""Hidden chaos: chaotic soft-token dynamics behind a fixed or periodic leading token.

The leading token (argmax of the next-token distribution) partitions the state space. When a chaotic attractor lies
inside one cell of that partition, the greedy readout repeats one token while the blend moves chaotically: positive
largest Lyapunov exponent, zero symbolic information. This tool measures, per cell:
- the largest exponent from renormalised twins at two offsets (to check it is not a rounding effect),
- the state period (state return distance below TOL times the state norm) and the attractor's diameter relative to
  the state norm, with the band count for 2^k bands (period of the coarse return pattern),
- the leading-token sequence: number of distinct tokens and its smallest period (symbolic period),
- the probability of the leading token and the entropy of the distribution (mean and spread).
A cell is "hidden chaos" when the exponent is positive at both offsets, there is no state period, and the leading-token
sequence is periodic (symbolic period >= 1).

cell   one cell with a long record (--trace also saves its per-step leading tokens, probabilities, entropies and
       states, trace.npz).
line   every cell of a 1D line (log10 T range at fixed alpha), batched.
cells  a list of cells from a JSON file {"cells": [[log10T, alpha], ...]} (horizon checks of map labels), batched.

Usage: python tools/hidden_chaos.py cell --vectors VEC --concept C --log-temperature LT --alpha A --out OUT [--trace]
       python tools/hidden_chaos.py line --vectors VEC --concept C --alpha A --log-temperature LO HI --n N --out OUT
       python tools/hidden_chaos.py cells --vectors VEC --concept C --cells-file FILE --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def symbolic_period(seq, max_q=64):
    seq = list(seq)
    for q in range(1, max_q + 1):
        if len(seq) > 2 * q and all(seq[k] == seq[k - q] for k in range(q, len(seq))):
            return q
    return 0


KEEP = 400          # states kept per cell: the diameter uses the last 400, the period check the last 65


FD_STEPS = (1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)


class EntropyClamp:
    """Entropy-clamped soft decoding: at every step each row's temperature is chosen so that its next-token entropy
    equals the row's target (nats); the "temperatures" passed to step are those targets. The map stays smooth and
    deterministic (explore_llm_soft_loop.solve_beta, a fixed number of safeguarded Newton steps). tsum/steps: running
    sum of the temperatures used, per row, since the last reset."""

    def __init__(self, loop):
        self.loop, self.torch, self.state_dtype, self.device = loop, loop.torch, loop.state_dtype, loop.device
        self.tsum, self.steps = None, 0

    def prepare(self, rows):
        self.tsum, self.steps = None, 0             # a new batch (rows may change, as in fd_probe)
        return self.loop.prepare(rows)

    def step(self, X, targets):
        import explore_llm_soft_loop as esl

        z = self.loop.logits(X)
        beta = esl.solve_beta(z, targets)
        p = self.torch.softmax(beta[:, None] * z, dim=-1)
        t = (1.0 / beta).double().cpu().numpy()
        self.tsum = t if self.tsum is None else self.tsum + t
        self.steps += 1
        return self.torch.cat([X[:, 1:, :], (p @ self.loop.E)[:, None, :]], dim=1), p


def fd_probe(loop, v, log_ts, alphas, base, hs=FD_STEPS, seed=1):
    """Precision probe of one step at the states base (n, K, d): the central-difference ratio
    |F(x + h u) - F(x - h u)| / (2h) for one random unit direction u per cell and each h (constant down to the rounding
    level for a smooth float64 map), and the largest difference between two evaluations of F(x) at different batch
    positions (zero for a batch-invariant computation)."""
    torch = loop.torch
    n, H = len(log_ts), len(hs)
    rng = np.random.default_rng(seed)
    u = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
    u = u / u.flatten(1).norm(dim=1)[:, None, None]
    rows = [base]
    for h in hs:
        rows += [base + h * u, base - h * u]
    rows.append(base)
    reps = 2 * H + 2
    loop.prepare(np.concatenate([np.array([a * v for a in alphas])] * reps))
    Y, _ = loop.step(torch.cat(rows), np.tile(10.0 ** np.asarray(log_ts), reps))
    Y = Y[:, -1].double()
    out = []
    for c in range(n):
        ratio = [float((Y[n * (1 + 2 * k) + c] - Y[n * (2 + 2 * k) + c]).norm() / (2 * h)) for k, h in enumerate(hs)]
        out.append({"h": list(hs), "central_ratio": ratio, "repeat_diff": float((Y[c] - Y[n * (reps - 1) + c]).abs().max())})
    return out


def run_batch(loop, v, log_ts, alphas, transient, record, eps_list, tol, seed=0, trace=False, sequences=None,
              tangent=False, probe=False, checkpoints=()):
    """Trajectories for a batch of cells with one twin per offset; returns per-cell statistics (and, with trace,
    the per-step arrays of the first cell). tangent: also the exact exponent of a renormalised tangent vector carried
    along each base orbit by forward-mode AD (no offset). probe: fd_probe at the final base states. checkpoints:
    measured-step counts at which the running exponents are also reported (convergence with the horizon)."""
    torch = loop.torch
    n = len(log_ts)
    m = len(eps_list)
    steer = np.array([a * v for a in alphas])
    rows = np.concatenate([steer] * (m + 1))
    temps = np.tile(10.0 ** np.asarray(log_ts), m + 1)
    X = loop.prepare(rows)
    for _ in range(transient):
        X, _ = loop.step(X, temps)
    rng = np.random.default_rng(seed)
    base = X[:n]
    twins = []
    for e in eps_list:
        d = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
        twins.append(base + d / d.flatten(1).norm(dim=1)[:, None, None] * e)
    X = torch.cat([base] + twins)
    growth = np.zeros((m, n))
    tgrowth = np.zeros(n)
    if tangent:                 # drawn after the twins, so the twins equal those of a run without tangents
        Vt = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
        Vt = Vt / Vt.flatten(1).norm(dim=1)[:, None, None]
    lead, pmax, ent, states, first = [], [], [], [], []
    marks, snaps = set(checkpoints), []
    if isinstance(loop, EntropyClamp):         # mean temperature over the recorded steps
        loop.tsum, loop.steps = None, 0
    for i in range(record):
        if tangent:
            X, p, JV = loop.step_jvp(X, temps, torch.cat([Vt, torch.zeros_like(X[n:])]))
            size = np.maximum(JV[:n].flatten(1).norm(dim=1).double().cpu().numpy(), 1e-300)
            tgrowth += np.log(size)
            Vt = JV[:n] / torch.as_tensor(size, dtype=loop.state_dtype, device=loop.device)[:, None, None]
        else:
            X, p = loop.step(X, temps)
        base = X[:n]
        new = [base]
        for k, e in enumerate(eps_list):
            Y = X[n * (k + 1):n * (k + 2)]
            diff = Y - base
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), e * 1e-30)
            growth[k] += np.log(dist / e)
            new.append(base + diff * torch.as_tensor(e / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None])
        X = torch.cat(new)
        if i + 1 in marks:
            snaps.append((i + 1, growth / (i + 1), tgrowth / (i + 1)))
        p0 = p[:n].double()
        top = p0.max(dim=-1)
        lead.append(top.indices.cpu().numpy())
        pmax.append(top.values.cpu().numpy())
        ent.append((-(p0 * torch.log(torch.clamp(p0, min=1e-300))).sum(-1)).cpu().numpy())
        states.append(base[:, -1].double().cpu().numpy())
        if len(states) > KEEP:
            states.pop(0)
        if trace:
            first.append(states[-1][0].copy())
    lead, pmax, ent, S = np.array(lead), np.array(pmax), np.array(ent), np.array(states)   # (record|KEEP, n[, d])
    out = []
    for c in range(n):
        s = S[:, c]
        scale = max(1.0, float(np.linalg.norm(s[-1])))
        q = 0
        for qq in range(1, min(65, len(s))):
            if np.linalg.norm(s[-1 - qq] - s[-1]) < tol * scale:
                q = qq
                break
        tail = s[-min(record, KEEP):]
        diam = float(np.max(np.linalg.norm(tail - tail.mean(0), axis=1)) * 2 / scale)
        seq = lead[:, c].tolist()
        out.append({"log10_temperature": float(log_ts[c]), "alpha": float(alphas[c]),
                    "lyapunov": [float(g[c] / record) for g in growth], "eps": list(eps_list), "period": q,
                    "diameter_rel": diam, "lead_distinct": len(set(seq)), "lead_period": symbolic_period(seq),
                    "lead_tokens": [int(t) for t in sorted(set(seq), key=seq.index)[:8]],
                    "pmax_mean": float(pmax[:, c].mean()), "pmax_sd": float(pmax[:, c].std()),
                    "entropy_mean": float(ent[:, c].mean()), "entropy_sd": float(ent[:, c].std())})
        if tangent:
            out[-1]["lyapunov_tangent"] = float(tgrowth[c] / record)
        if snaps:
            out[-1]["checkpoints"] = {"steps": [s for s, _, _ in snaps],
                                      "lyapunov": [[float(g[k][c]) for k in range(m)] for _, g, _ in snaps]}
            if tangent:
                out[-1]["checkpoints"]["lyapunov_tangent"] = [float(t[c]) for _, _, t in snaps]
        if isinstance(loop, EntropyClamp):
            out[-1]["mean_temperature"] = float(loop.tsum[c] / loop.steps)
    if probe:
        for r, f in zip(out, fd_probe(loop, v, log_ts, alphas, X[:n])):
            r["fd_probe"] = f
    if sequences is not None:
        sequences["lead"].append(lead)
        sequences["pmax"].append(pmax)
    if trace:
        return out, {"lead": lead[:, 0], "pmax": pmax[:, 0], "entropy": ent[:, 0], "states": np.array(first)}
    return out


def classify(r):
    pos = all(l > 1e-3 for l in r["lyapunov"])
    if r["period"] > 0:
        return "periodic"
    if pos and r["lead_period"] >= 1:
        return "hidden chaos"
    if pos:
        return "visible chaos"
    return "irregular, exponent not positive at both offsets"


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cell", "line", "cells"))
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--concept", default=None)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--log-temperature", type=float, nargs="+")
    ap.add_argument("--cells-file", help="cells mode: JSON file with a list 'cells' of [log10T, alpha] pairs")
    ap.add_argument("--trace", action="store_true", help="cell mode: save the per-step arrays (trace.npz)")
    ap.add_argument("--save-sequences", action="store_true",
                    help="save every cell's leading-token and leading-probability sequences (sequences.npz)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=2000)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (layers spread over all GPUs)")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"),
                    help="bf16: decoder linear weights stay bfloat16, used in float64 (exact)")
    ap.add_argument("--vectors-none", action="store_true", help="unsteered: no steering vector (alpha ignored)")
    ap.add_argument("--prefix-file", default=None, help="JSON with a list 'prefixes' of {system, opener}")
    ap.add_argument("--prefix-index", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--window", type=int, default=1,
                    help="soft tokens fed back (the state is the window of the last K expected embeddings)")
    ap.add_argument("--tangent", action="store_true",
                    help="also the exact tangent exponent (forward-mode AD along each base orbit, no offset)")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=None,
                    help="also report the running exponents after these numbers of measured steps")
    ap.add_argument("--fd-probe", action="store_true",
                    help="one-step central differences at the final states for h = 1e-1..1e-11 (precision probe)")
    ap.add_argument("--clamp-entropy", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="entropy-clamped decoding: line mode sweeps the next-token entropy from LO to HI nats instead "
                         "of the temperature (--log-temperature unused); in cells mode the first column is the target")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.clamp_entropy and args.tangent:
        raise SystemExit("--tangent is not implemented for entropy-clamped decoding")
    if args.checkpoints and not all(0 < c <= args.record for c in args.checkpoints):
        raise SystemExit("--checkpoints must lie between 1 and --record")
    prompt, opener = "open", ela.OPENER
    if args.prefix_file is not None:
        pref = json.loads(Path(args.prefix_file).read_text())["prefixes"][args.prefix_index]
        ela.PROMPTS["custom"] = pref["system"]
        prompt, opener = "custom", pref["opener"]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, prompt, opener, args.window, args.dtype, args.device,
                        args.weights)
    v = np.zeros(loop.E.shape[1]) if args.vectors_none else elpm.load_vectors(args.vectors)[args.concept]
    if len(v) != loop.E.shape[1]:
        raise SystemExit(f"steering vector dimension {len(v)} does not match the model ({loop.E.shape[1]})")
    if args.mode == "cells":
        if not args.cells_file:
            raise SystemExit("cells mode needs --cells-file")
        pairs = np.array(json.loads(Path(args.cells_file).read_text())["cells"], dtype=float)
        log_ts, alphas = pairs[:, 0], pairs[:, 1]
        if args.clamp_entropy:
            log_ts = np.log10(log_ts)            # entropy targets, passed to the loop as "temperatures"
    elif args.clamp_entropy:
        if args.alpha is None:
            raise SystemExit(f"{args.mode} mode needs --alpha")
        targets = np.linspace(args.clamp_entropy[0], args.clamp_entropy[1], args.n)
        if targets.max() >= np.log(loop.E.shape[0]) or targets.min() <= 0:
            raise SystemExit("entropy targets must lie strictly between 0 and log V")
        log_ts, alphas = np.log10(targets), np.full(args.n, args.alpha)
    else:
        if args.alpha is None or not args.log_temperature:
            raise SystemExit(f"{args.mode} mode needs --alpha and --log-temperature")
        if args.mode == "cell":
            log_ts = np.array([args.log_temperature[0]])
        else:
            log_ts = np.linspace(args.log_temperature[0], args.log_temperature[1], args.n)
        alphas = np.full(len(log_ts), args.alpha)
    if args.clamp_entropy:
        loop = EntropyClamp(loop)
    results, tr = [], None
    seqs = {"lead": [], "pmax": []} if args.save_sequences else None
    for s in range(0, len(log_ts), args.chunk):
        got = run_batch(loop, v, log_ts[s:s + args.chunk], alphas[s:s + args.chunk], args.transient, args.record,
                        args.eps, args.tol, trace=args.trace and args.mode == "cell", sequences=seqs,
                        tangent=args.tangent, probe=args.fd_probe, checkpoints=tuple(args.checkpoints or ()))
        if isinstance(got, tuple):
            got, tr = got
        results += got
        print(json.dumps({"done": len(results), "of": len(log_ts)}), flush=True)
    for r in results:
        r["class"] = classify(r)
        if args.clamp_entropy:                 # the swept value was the entropy target
            r["entropy_target"] = float(10.0 ** r["log10_temperature"])
            r["log10_temperature"] = float(np.log10(r["mean_temperature"]))
    tok = getattr(loop, "loop", loop).sm.tok
    for r in results:
        r["lead_text"] = [tok.decode([t]) for t in r["lead_tokens"]]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hidden-chaos.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))
    if tr is not None:
        np.savez_compressed(out / "trace.npz", **tr)
    if seqs is not None:                        # (record, cells) arrays, cells in the order of hidden-chaos.json
        np.savez_compressed(out / "sequences.npz", lead=np.concatenate(seqs["lead"], axis=1),
                            pmax=np.concatenate(seqs["pmax"], axis=1))
    counts = {}
    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(json.dumps(counts))
    if args.mode == "cell":
        print(json.dumps(results[0], indent=1)[:1500])


if __name__ == "__main__":
    main()
