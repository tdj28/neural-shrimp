"""Read soft-token-map attractors in sparse-autoencoder (SAE) features.

For each chosen cell (log10 T, alpha) of a steered soft-token map, the map is iterated from the fixed start
(transient), then for `record` steps the residual stream at the SAE's layer is recorded at the fed-back position
and encoded by the SAE. Output per cell: the attractor's leading-token loop, the features active on it (mean
activation over the recorded steps, the share of steps on which each is active), and the SAE's reconstruction
quality on these states (fraction of variance explained, FVE), which measures how well an SAE trained on other text
transfers to the map's states. Steering adds alpha v at the steering layer, so features downstream of it partly
reflect the steering vector itself; compare attractors at equal alpha, and against index-adjacent and random
features, before naming any feature.

SAE formats (loaded as tensors only, never as pickled code):
  goodfire  .pth state dict (encoder_linear, decoder_linear), ReLU; residual stream after decoder layer L
            (Neuronpedia 'resid-post'), i.e. hidden_states[L + 1]
  saelens   sae_weights.safetensors + cfg.json (W_enc, b_enc, W_dec, b_dec; 'topk' activation with k from cfg;
            apply_b_dec_to_input); hook blocks.L.hook_resid_pre, i.e. hidden_states[L]

Usage:
  python tools/llm_sae_readout.py validate --model llama3.1-8b --sae goodfire:PATH:19 [--sae saelens:DIR] --out ...
  python tools/llm_sae_readout.py read --model llama3.1-8b --layer 16 --vectors VEC --concept gratitude \
      --cells-from-scan SCAN_DIR [--cell=T,A ...] --sae goodfire:PATH:19 --sae saelens:DIR --out ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

TEXTS = [  # ordinary text for the hook-point check (no outcome of the study depends on it)
    "The committee met on Tuesday to review the budget for the coming year.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the window and listened to the rain on the roof.",
    "In 1905 Einstein published four papers that changed physics.",
    "To install the package, run the command and restart the terminal.",
    "Thank you so much for your help, I really appreciate it!",
    "That is a stupid idea and you should be ashamed of yourself.",
    "What is the meaning of life? Philosophers have argued for centuries.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Hello! How can I assist you today?",
]


class SAE:
    """A sparse autoencoder on the residual stream at hidden_states[index]."""

    def __init__(self, kind, path, layer=None, device="cpu"):
        import torch

        self.torch, self.kind, self.path = torch, kind, str(path)
        if kind == "goodfire":
            sd = torch.load(path, map_location="cpu", weights_only=True)
            self.W_enc = sd["encoder_linear.weight"].T.float()            # (d, F)
            self.b_enc = sd["encoder_linear.bias"].float()
            self.W_dec = sd["decoder_linear.weight"].T.float()            # (F, d)
            self.b_dec = sd["decoder_linear.bias"].float()
            self.k, self.sub_b_dec = None, False
            self.index = int(layer) + 1                                   # resid post of decoder layer `layer`
            self.name = f"goodfire-l{layer}"
        elif kind == "saelens":
            from safetensors.torch import load_file

            cfg = json.loads((Path(path) / "cfg.json").read_text())
            w = load_file(str(Path(path) / "sae_weights.safetensors"))
            self.W_enc, self.b_enc = w["W_enc"].float(), w["b_enc"].float()
            self.W_dec, self.b_dec = w["W_dec"].float(), w["b_dec"].float()
            if cfg.get("activation_fn_str") != "topk" or not cfg["hook_name"].endswith("hook_resid_pre"):
                raise SystemExit(f"unsupported SAELens SAE: {cfg.get('activation_fn_str')} at {cfg['hook_name']}")
            self.k = int(cfg["activation_fn_kwargs"]["k"])
            self.sub_b_dec = bool(cfg.get("apply_b_dec_to_input", False))
            self.index = int(cfg["hook_layer"])                           # resid pre of block L = hidden_states[L]
            self.name = f"saelens-{Path(path).name}"
        else:
            raise SystemExit(f"unknown SAE kind {kind}")
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, name, getattr(self, name).to(device))
        self.n_features = self.W_enc.shape[1]

    def encode(self, x):
        torch = self.torch
        x = x.float()
        pre = (x - self.b_dec if self.sub_b_dec else x) @ self.W_enc + self.b_enc
        if self.k is None:
            return torch.relu(pre)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def fve(self, x):
        """Fraction of variance explained over a batch of residual vectors (rows)."""
        x = x.float()
        rec = self.decode(self.encode(x))
        num = ((x - rec) ** 2).sum()
        den = ((x - x.mean(0, keepdim=True)) ** 2).sum()
        return float(1 - num / den)


def parse_sae(spec, device):
    kind, rest = spec.split(":", 1)
    if kind == "goodfire":
        path, layer = rest.rsplit(":", 1)
        return SAE(kind, path, layer, device)
    return SAE(kind, rest, None, device)


def validate(args):
    """Hook-point check: FVE of each SAE on ordinary text at its documented index and the two neighbours."""
    import torch

    import explore_llm_attractors as ela

    eng = ela.Engine(args.model, args.threads, "float32", args.device)
    saes = [parse_sae(s, args.device) for s in args.sae]
    report = {"model": args.model, "texts": len(TEXTS), "saes": []}
    hs_all = {}
    with torch.no_grad():
        for text in TEXTS:
            ids = eng.tok(text, return_tensors="pt").to(args.device)
            out = eng.model(**ids, output_hidden_states=True)
            for i, h in enumerate(out.hidden_states):
                hs_all.setdefault(i, []).append(h[0, 1:].float())        # drop the first token (attention sink)
    for sae in saes:
        rows = {}
        for i in (sae.index - 1, sae.index, sae.index + 1):
            if i in hs_all:
                rows[i] = sae.fve(torch.cat(hs_all[i]))
        best = max(rows, key=rows.get)
        report["saes"].append({"sae": sae.name, "documented_index": sae.index, "fve_by_index": rows,
                               "best_index": best, "documented_is_best": best == sae.index})
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "validate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def cells_from_scan(folder, per_family=1, families=6, chaotic=2):
    """Interior cells of the largest attractor families of a finished scan, and of its most chaotic region: for
    each family the cell whose 5 x 5 neighbourhood holds the most cells of that family; for chaos the cells with the
    largest positive exponent among those whose neighbourhood is entirely chaotic."""
    import explore_llm_attractors as ela

    z = np.load(ela.DATA / "explore" / folder / "scan.npz")
    lt, al, F, L = z["log10_temperature"], z["alpha"], z["family"], z["lyapunov"]
    keys, counts = np.unique(F[F >= 0], return_counts=True)
    order = keys[np.argsort(-counts)][:families]
    picks = []
    pad = np.pad(F, 2, constant_values=-2)
    for f in order:
        score = sum((pad[2 + di:2 + di + F.shape[0], 2 + dj:2 + dj + F.shape[1]] == f).astype(int)
                    for di in range(-2, 3) for dj in range(-2, 3))
        score = np.where(F == f, score, -1)
        i, j = np.unravel_index(np.argmax(score), F.shape)
        picks.append({"log10_T": float(lt[i]), "alpha": float(al[j]), "kind": "family", "family": int(f),
                      "neighbourhood_share": float(score[i, j] / 25)})
    chaos = L > 1e-3
    padc = np.pad(chaos, 1, constant_values=False)
    interior = sum(padc[1 + di:1 + di + L.shape[0], 1 + dj:1 + dj + L.shape[1]]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)) == 9
    cand = np.argwhere(interior)
    for i, j in sorted(cand.tolist(), key=lambda c: -L[c[0], c[1]])[:chaotic]:
        picks.append({"log10_T": float(lt[i]), "alpha": float(al[j]), "kind": "chaotic",
                      "lyapunov": float(L[i, j])})
    return picks


def read(args):
    import torch

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    cells = []
    for folder in args.cells_from_scan or []:
        cells += [dict(c, scan=folder) for c in cells_from_scan(folder, families=args.families,
                                                                  chaotic=args.chaotic)]
    for spec in args.cell or []:
        t, a = (float(x) for x in spec.split(","))
        cells.append({"log10_T": t, "alpha": a, "kind": "given"})
    if not cells:
        raise SystemExit("no cells")
    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, args.opener, args.window, args.dtype,
                        args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    saes = [parse_sae(s, args.device) for s in args.sae]
    index_set = sorted({s.index for s in saes})
    n = len(cells)
    X = loop.prepare(np.array([c["alpha"] * v for c in cells]))
    temps = np.array([10.0 ** c["log10_T"] for c in cells])
    for _ in range(args.transient):
        X, _ = loop.step(X, temps)
    hidden = {i: [] for i in index_set}
    tokens = []
    for _ in range(args.record):
        loop.sm.steer = loop.steer_rows
        try:
            with torch.no_grad():
                out = loop.model(inputs_embeds=X.to(loop.dtype), past_key_values=loop.cache, use_cache=True,
                                 logits_to_keep=1, output_hidden_states=True)
        finally:
            loop.sm.steer = None
            loop.cache.crop(loop.L0)
        for i in index_set:
            hidden[i].append(out.hidden_states[i][:, -1, :].float().cpu())
        z = out.logits[:, -1, :].to(loop.state_dtype)
        p = torch.softmax(z / torch.as_tensor(temps, dtype=loop.state_dtype, device=loop.device)[:, None], dim=-1)
        tokens.append(p.argmax(-1).cpu().numpy())
        X = torch.cat([X[:, 1:, :], (p @ loop.E)[:, None, :]], dim=1)
    toks = np.array(tokens).T                                             # (n, record)
    tok = loop.sm.tok
    results = []
    for c_i, cell in enumerate(cells):
        rec = dict(cell)
        rec["tokens"] = [tok.decode([int(t)]) for t in toks[c_i]]
        rec["saes"] = {}
        for sae in saes:
            H = torch.stack([hidden[sae.index][s][c_i] for s in range(args.record)]).to(args.device)  # (record, d)
            f = sae.encode(H)
            active = (f > 0).float()
            mean = f.mean(0)
            top = torch.argsort(mean, descending=True)[:args.top]
            rec["saes"][sae.name] = {
                "fve_on_orbit_states": sae.fve(H) if args.record > 2 else None,
                "relative_error": float(((H - sae.decode(f)) ** 2).sum(-1).sqrt().mean()
                                        / (H ** 2).sum(-1).sqrt().mean()),
                "l0_mean": float(active.sum(-1).mean()),
                "top": [{"feature": int(i), "mean": float(mean[i]), "active_share": float(active[:, i].mean())}
                        for i in top if mean[i] > 0]}
        results.append(rec)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hidden.npz", **{f"h{i}": torch.stack(hidden[i], 1).numpy() for i in index_set},
                        tokens=toks)
    summary = {"args": vars(args), "cells": results, "saes": [{"name": s.name, "path": s.path, "index": s.index,
                                                               "features": s.n_features, "k": s.k} for s in saes]}
    (out / "readout.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"cells": n, "out": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "read"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True)
        sp.add_argument("--threads", type=int, default=8)
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--sae", action="append", required=True, help="goodfire:PATH:LAYER or saelens:DIR")
        sp.add_argument("--out", required=True)
    rd = sub.choices["read"]
    rd.add_argument("--layer", type=int, required=True, help="steering layer")
    rd.add_argument("--vectors", required=True)
    rd.add_argument("--concept", required=True)
    rd.add_argument("--prompt", default="open")
    rd.add_argument("--opener", default=None)
    rd.add_argument("--window", type=int, default=1)
    rd.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    rd.add_argument("--cells-from-scan", nargs="*", help="explore-relative finished scan folders")
    rd.add_argument("--cell", action="append", help="LOG10T,ALPHA (repeat; write --cell=-0.1,0.07 for negatives)")
    rd.add_argument("--families", type=int, default=6)
    rd.add_argument("--chaotic", type=int, default=2)
    rd.add_argument("--transient", type=int, default=500)
    rd.add_argument("--record", type=int, default=48)
    rd.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    if args.cmd == "read" and args.opener is None:
        import explore_llm_attractors as ela
        args.opener = ela.OPENER
    {"validate": validate, "read": read}[args.cmd](args)


if __name__ == "__main__":
    main()
