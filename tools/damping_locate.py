"""Where in the network does the loop damp a change of its fed-back token? Layer-resolved gain split (run D).

tools/steer_gain.py splits each step's growth of a twin into a network factor g_net = (|dh|/|h|) / (|dx|/|x|) (x the
fed-back soft token, h the final normalized hidden state at its position) and a readout factor. This tool resolves
g_net through the stack. It runs the unsteered loop (window 1, float64, the paper's prompt) at several temperatures
at once, then records steps with one twin per temperature at a relative offset eps along the evolving twin direction
(renormalized every step, as in steer_gain.py). At every recorded step, hooks read at the soft token's position the
residual stream entering each decoder block, the attention and MLP outputs of each block, and the final hidden state
before and after the final normalization. Differences between twin and base row give, per block l:
- **Cumulative growth and norm:** A_l = log(|dh_l| / |dx|), the absolute growth of the change up to block l (h_l
  enters block l + 1), and N_l = log(|h_l| / |x|), the growth of the residual stream itself. Their difference
  A_l - N_l is the log of the relative change, (|dh_l| / |h_l|) / (|dx| / |x|).
- **Sublayer response:** for attention and MLP, the gain |d out| / |d in|, and the parallel coefficient
  <d out, d in> / |d in|^2. A negative coefficient means the sublayer writes against the change it receives, which
  damps it; a positive one reinforces it.
- **Direct contributions:** the share c_s = <d out_s, dh_L> / |dh_L|^2 of each sublayer s in the final change. The
  residual stream adds them, dh_L = dx + sum_s d out_s, so the shares and c_x = <dx, dh_L> / |dh_L|^2 add up to one;
  the closure error is reported.
- **Exact split of the network factor:**
  log g_net = [A_L] - [N_L] + [log((|dh~| / |h~|) / (|dh_L| / |h_L|))],
  that is growth through the stack, minus dilution by the residual stream, plus the final normalization (which
  removes the radial part of the change and applies its weight).
- **Knockouts (optional, --knockout-every K):** every K recorded steps, a knockout pass for the --knockout-cells.
  Each sublayer s gets a row whose input carries the twin's change but whose output at the soft token is clamped to
  the base row's, so the change cannot pass through s. delta_s = log g_net - log g_net(s knocked out); positive means
  s amplifies the change, negative that it damps it.
Per temperature the tool also gives the twin exponent, the period of the base orbit (tolerance 1e-8 relative, up to
64) and the mean next-token entropy, as steer_gain.py does.

Usage: python tools/damping_locate.py --model qwen0.5b --log-temperature -0.3 0.9 --n 24 --transient 500 \\
           --record 500 --eps 1e-7 --knockout-every 25 --knockout-cells 4 10 16 --threads 16 --out damping/qwen0.5b
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class Probe:
    """Forward hooks on every decoder block, its attention and MLP, and the final normalization: read the soft
    token's position, and optionally clamp one sublayer's output at that position to a reference row's."""

    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.L = len(self.layers)
        self.capture = False
        self.clamp = {}                     # (kind, l) -> (rows, reference rows), index tensors
        self.rec = {}
        self.handles = []
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre(l), with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(self._sub("attn", l)))
            self.handles.append(layer.mlp.register_forward_hook(self._sub("mlp", l)))
        self.handles.append(model.model.norm.register_forward_hook(self._norm))

    def _pre(self, l):
        def hook(module, args, kwargs):
            if self.capture:
                h = args[0] if args else kwargs["hidden_states"]
                self.rec[("in", l)] = h[:, -1].detach().clone()
        return hook

    def _sub(self, kind, l):
        def hook(module, inputs, output):
            o = output[0] if isinstance(output, tuple) else output
            c = self.clamp.get((kind, l))
            if c is not None:
                rows, ref = c
                o[rows, -1] = o[ref, -1]
            if self.capture:
                self.rec[(kind, l)] = o[:, -1].detach().clone()
            return output
        return hook

    def _norm(self, module, inputs, output):
        if self.capture:
            self.rec["hL"] = inputs[0][:, -1].detach().clone()
            self.rec["ht"] = output[:, -1].detach().clone()


def expand_cache(torch, base, R):
    from transformers import DynamicCache

    c = DynamicCache()
    for li, layer in enumerate(base.layers):
        c.update(layer.keys.expand(R, -1, -1, -1).contiguous(), layer.values.expand(R, -1, -1, -1).contiguous(), li)
    return c


def main():
    import torch

    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl
    from transformers import DynamicCache

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--log-temperature", type=float, nargs=2, default=[-0.3, 0.9])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--transient", type=int, default=500)
    ap.add_argument("--record", type=int, default=500)
    ap.add_argument("--eps", type=float, default=1e-7)
    ap.add_argument("--knockout-every", type=int, default=0)
    ap.add_argument("--knockout-cells", type=int, nargs="*", default=[])
    ap.add_argument("--keep-steps", type=int, default=100, help="per-step layer arrays kept for the first N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loop = esl.SoftLoop(args.model, 0, args.threads, "open", ela.OPENER, 1, "float64", args.device, args.weights)
    model, E, dev, sd = loop.model, loop.E, loop.device, loop.state_dtype
    probe = Probe(model)
    L, n = probe.L, args.n
    log_ts = np.linspace(*args.log_temperature, n)
    T = torch.as_tensor(10.0 ** log_ts, dtype=sd, device=dev)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)

    base_cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=loop.prefix_ids, past_key_values=base_cache, use_cache=True, logits_to_keep=1)
        start = model.get_input_embeddings()(loop.prefix_ids[:, -1:])[0, 0].to(sd)
    L0 = loop.prefix_ids.shape[1]
    caches = {}

    def forward(X, temps):
        """X (R, d) soft tokens, temps (R,) -> final normalized hidden state (R, d), p and next soft token F."""
        R = X.shape[0]
        if R not in caches:
            caches[R] = expand_cache(torch, base_cache, R)
        c = caches[R]
        with torch.no_grad():
            o = model.model(inputs_embeds=X[:, None, :].to(loop.dtype), past_key_values=c, use_cache=True)
            ht = o.last_hidden_state[:, -1]
            z = model.lm_head(ht).to(dev, sd)
        c.crop(L0)
        p = torch.softmax(z / temps[:, None], dim=-1)
        return ht.to(dev, sd), p, p @ E

    X = start.repeat(n, 1).clone()                 # transient: base rows only
    tic = time.perf_counter()
    for _ in range(args.transient):
        _, _, X = forward(X, T)
    rng = np.random.default_rng(args.seed)
    g = torch.as_tensor(rng.normal(size=tuple(X.shape)), dtype=sd, device=dev)
    twin = X + g / g.norm(dim=1, keepdim=True) * args.eps * X.norm(dim=1, keepdim=True)
    T2 = torch.cat([T, T])
    nrm = lambda a: a.norm(dim=-1)                                                # noqa: E731
    dot = lambda a, b: (a * b).sum(-1)                                            # noqa: E731

    keep = min(args.keep_steps, args.record)
    names_l = ("A", "N", "gain_attn", "gain_mlp", "par_attn", "par_mlp", "c_attn", "c_mlp")
    sums_l = {k: np.zeros((L, n)) for k in names_l}
    per_l = {k: np.zeros((keep, L, n), dtype=np.float32) for k in names_l}
    names = ("log_g_net", "stack", "dilution", "final_norm", "log_g_read", "c_x", "closure", "entropy")
    sums = {k: np.zeros(n) for k in names}
    growth = np.zeros(n)
    ko = {"steps": 0, "delta_attn": np.zeros((L, len(args.knockout_cells))),
          "delta_mlp": np.zeros((L, len(args.knockout_cells))), "check": np.zeros(len(args.knockout_cells))}
    tail = []
    x0 = nrm(X).cpu().numpy()
    max_closure = np.zeros(n)

    for i in range(args.record):
        Xb = torch.cat([X, twin])
        probe.capture, probe.rec = True, {}
        ht, p, F = forward(Xb, T2)
        probe.capture = False
        rec = {k: v.to(dev, sd) for k, v in probe.rec.items()}
        b, t = slice(0, n), slice(n, 2 * n)
        dx = Xb[t] - Xb[b]
        ndx, nx = nrm(dx), nrm(Xb[b])
        hL, dhL = rec["hL"][b].to(sd), rec["hL"][t].to(sd) - rec["hL"][b].to(sd)
        hb, dht = ht[b], ht[t] - ht[b]
        dF = F[t] - F[b]
        step = {"stack": torch.log(nrm(dhL) / ndx), "dilution": torch.log(nrm(hL) / nx)}
        step["final_norm"] = torch.log((nrm(dht) / nrm(hb)) / (nrm(dhL) / nrm(hL)))
        step["log_g_net"] = torch.log((nrm(dht) / nrm(hb)) / (ndx / nx))
        step["log_g_read"] = torch.log((nrm(dF) / nrm(F[b])) / (nrm(dht) / nrm(hb)))
        step["c_x"] = dot(dx, dhL) / nrm(dhL) ** 2
        total = dx.clone()
        lay = {k: [] for k in names_l}
        for l in range(L):
            h_in = rec[("in", l)].to(sd)
            dh_in = h_in[t] - h_in[b]
            dA = rec[("attn", l)][t].to(sd) - rec[("attn", l)][b].to(sd)
            dM = rec[("mlp", l)][t].to(sd) - rec[("mlp", l)][b].to(sd)
            mid = dh_in + dA
            lay["A"].append(torch.log(nrm(dh_in) / ndx))
            lay["N"].append(torch.log(nrm(h_in[b]) / nx))
            lay["gain_attn"].append(nrm(dA) / nrm(dh_in))
            lay["gain_mlp"].append(nrm(dM) / nrm(mid))
            lay["par_attn"].append(dot(dA, dh_in) / nrm(dh_in) ** 2)
            lay["par_mlp"].append(dot(dM, mid) / nrm(mid) ** 2)
            lay["c_attn"].append(dot(dA, dhL) / nrm(dhL) ** 2)
            lay["c_mlp"].append(dot(dM, dhL) / nrm(dhL) ** 2)
            total = total + dA + dM
        step["closure"] = nrm(total - dhL) / nrm(dhL)
        logp = torch.log(torch.clamp(p[b], min=1e-300))
        step["entropy"] = -(p[b] * logp).sum(-1)
        for k in names:
            sums[k] += step[k].cpu().numpy()
        max_closure = np.maximum(max_closure, step["closure"].cpu().numpy())
        for k in names_l:
            a = torch.stack(lay[k]).cpu().numpy()
            sums_l[k] += a
            if i < keep:
                per_l[k][i] = a
        growth += torch.log(nrm(dF) / ndx).cpu().numpy()

        if args.knockout_every and args.knockout_cells and i % args.knockout_every == 0:
            cells = args.knockout_cells
            G = 2 + 2 * L
            probe.clamp = {}
            for l in range(L):
                for j, kind in enumerate(("attn", "mlp")):
                    probe.clamp[(kind, l)] = ([2 + 2 * l + j], [0])
            for gi, c in enumerate(cells):
                Xk = torch.stack([Xb[c], Xb[n + c]] + [Xb[n + c]] * (2 * L))
                Tk = torch.full((G,), 10.0 ** log_ts[c], dtype=sd, device=dev)
                htk, _, _ = forward(Xk, Tk)
                base_r, twin_r = 0, 1
                rel_x = (nrm(Xk[twin_r] - Xk[base_r]) / nrm(Xk[base_r])).item()
                lg = np.log((nrm(htk[twin_r] - htk[base_r]) / nrm(htk[base_r])).item() / rel_x)
                ko["check"][gi] += lg - step["log_g_net"][c].item()
                for l in range(L):
                    for j, kind in enumerate(("attn", "mlp")):
                        r = 2 + 2 * l + j
                        lk = np.log((nrm(htk[r] - htk[base_r]) / nrm(htk[base_r])).item() / rel_x)
                        ko[f"delta_{kind}"][l, gi] += lg - lk
            probe.clamp = {}
            ko["steps"] += 1

        scale = (args.eps * nrm(F[b]) / nrm(dF))[:, None]
        X, twin = F[b], F[b] + dF * scale
        tail.append(F[b].cpu().numpy())
        if len(tail) > 66:
            tail.pop(0)
        if (i + 1) % 100 == 0:
            print(json.dumps({"recorded": i + 1, "of": args.record, "seconds": round(time.perf_counter() - tic, 1)}),
                  flush=True)

    S = np.array(tail)
    R = args.record
    cells = []
    for c in range(n):
        s = S[:, c]
        period = next((q for q in range(1, min(65, len(s))) if np.linalg.norm(s[-1 - q] - s[-1])
                       < 1e-8 * max(1.0, np.linalg.norm(s[-1]))), 0)
        lyap = float(growth[c] / R)
        rec = {"log10_temperature": float(log_ts[c]), "lyapunov": lyap, "period": int(period),
               "chaotic": bool(period == 0 and lyap > 1e-3),
               "boundary_term": float(np.log(np.linalg.norm(s[-1]) / x0[c]) / R)}
        rec.update({f"mean_{k}": float(sums[k][c] / R) for k in names})
        rec["max_closure"] = float(max_closure[c])
        cells.append(rec)
    np.savez_compressed(out / "layers.npz", log10_temperature=log_ts,
                        **{f"mean_{k}": sums_l[k] / R for k in names_l}, **{f"steps_{k}": per_l[k] for k in names_l})
    summary = {"args": vars(args), "layers": L, "hidden": int(E.shape[1]), "prefix_tokens": int(L0),
               "embedding_norm_median": float(nrm(E).median()), "cells": cells,
               "seconds": round(time.perf_counter() - tic, 1)}
    if ko["steps"]:
        summary["knockouts"] = {"cells": args.knockout_cells, "steps": ko["steps"],
                                "delta_attn": (ko["delta_attn"] / ko["steps"]).tolist(),
                                "delta_mlp": (ko["delta_mlp"] / ko["steps"]).tolist(),
                                "plain_twin_check": (ko["check"] / ko["steps"]).tolist()}
    (out / "damping.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"done": True, "seconds": summary["seconds"]}))


if __name__ == "__main__":
    main()
