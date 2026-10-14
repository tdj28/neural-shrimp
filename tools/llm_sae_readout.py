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


# Held-out category texts for feature profiles (written on 13 October 2026 before any SAE readout of the map was
# seen; none of them is among the prompts that built the steering vectors). Confound categories separate the
# construct from look-alikes: service politeness (formulaic, not gratitude), sadness (negative, not directed at the
# interlocutor) and casual swearing (profanity without contempt).
CATEGORIES = {
    "gratitude": [
        "I can't thank you enough for staying up late to help me.", "Your kindness yesterday meant the world to me.",
        "I'm so thankful we met when we did.", "Thank you for believing in me when nobody else did.",
        "I'm grateful for every single day we spend together.", "Words can't express how much I appreciate this.",
        "Thanks a million, you really saved my weekend.", "I feel blessed to have friends like you.",
        "Your support has filled me with gratitude.", "I owe you so much, thank you from the heart.",
        "We are deeply thankful for your generous gift.", "I appreciate everything you have done for our family."],
    "hostility": [
        "You're a complete moron and everyone knows it.", "Nobody asked for your opinion, so keep quiet.",
        "Get out of my sight, you useless clown.", "You make me sick, you spineless liar.",
        "That was the dumbest thing I've ever heard from you.", "I hope you fail at everything you try.",
        "You're nothing but a parasite on this team.", "Stop wasting my time with your idiotic questions.",
        "I have nothing but contempt for you.", "You disgust me more than words can say.",
        "Go away and never talk to me again.", "Your work is trash and so are you."],
    "kindness": [
        "You did a wonderful job and I'm proud of you.", "Take all the time you need, I'm here for you.",
        "Your ideas always brighten my day.", "You're one of the most thoughtful people I know.",
        "I'd love to hear more about your project.", "You handled that situation with such grace.",
        "Don't worry, everyone makes mistakes, you're doing great.", "It's a pleasure to work with someone like you.",
        "You have a real talent for explaining things.", "I'm always happy when you join us.",
        "Your patience with the children is remarkable.", "What a lovely thing to say, you're very kind."],
    "philosophy": [
        "Can a machine ever truly understand what it says?", "Is free will compatible with a deterministic world?",
        "What makes a person the same person over time?", "Perhaps meaning is something we create rather than find.",
        "Why is there something rather than nothing?", "Is the mind just what the brain does?",
        "Could our perceptions be a simulation of reality?", "What is it to know something, rather than believe it?",
        "Time may be an illusion produced by memory.", "Consciousness might be a fundamental feature of the universe.",
        "Do numbers exist independently of human minds?", "Morality may be rooted in our shared nature."],
    "cooking": [
        "Grate the cheese and sprinkle it over the pasta.", "Marinate the chicken overnight in lemon and garlic.",
        "Let the dough rise in a warm place for an hour.", "Fold the egg whites gently into the batter.",
        "Toast the spices in a dry pan before grinding them.", "Reduce the sauce until it coats the back of a spoon.",
        "Peel and dice the potatoes into even cubes.", "Brush the pastry with beaten egg before baking.",
        "Add a splash of vinegar to brighten the soup.", "Slice the mushrooms thinly and saute them in butter.",
        "Steam the broccoli for five minutes.", "Rest the steak for ten minutes before cutting it."],
    "neutral": [
        "The library opens at ten on Saturdays.", "The invoice number is printed at the top of the page.",
        "Our flight departs from gate twelve.", "The spreadsheet has four columns and twenty rows.",
        "The bridge is two kilometres long.", "Please submit the form before the end of the month.",
        "The printer on the second floor is out of paper.", "The bus runs every fifteen minutes.",
        "The museum is closed on Mondays.", "The software update will install overnight.",
        "The parking lot is behind the building.", "The meeting room has seating for twelve people."],
    "politeness": [
        "Please let me know if you need anything else.", "Thank you for contacting customer support.",
        "We appreciate your patience while we process your request.", "Kindly find the attached document.",
        "Have a nice day and thanks for shopping with us.", "We are happy to assist you with your order.",
        "Thank you for your email; we will reply shortly.", "Your call is important to us, please hold.",
        "Feel free to reach out with any further questions.", "Thanks for your feedback on our service.",
        "We hope you enjoy your stay at our hotel.", "Best regards from the whole team."],
    "sadness": [
        "I miss my grandmother more than ever this winter.", "The house feels empty since the dog died.",
        "I cried all night after hearing the news.", "Nothing seems to matter much anymore.",
        "We lost the game and the whole town is quiet.", "I feel lonely even when I'm surrounded by people.",
        "The letter brought back painful memories.", "It's hard to get out of bed these days.",
        "Our old friendship slowly faded away.", "The rain matched my gloomy mood.",
        "I never got the chance to say goodbye.", "Everything reminds me of what we lost."],
    "swearing": [
        "Damn, this coffee is really good.", "Holy crap, that view is amazing!", "Hell yeah, we won the match!",
        "This damn printer jammed again.", "Shit, I forgot my keys at home.", "Bloody brilliant idea, mate!",
        "Crap, I'm going to be late for the bus.", "Damn, that was a close call.",
        "What the hell, the store is closed already?", "Oh hell, I spilled the soup.",
        "Shoot, I missed the last train.", "Freaking awesome concert last night!"],
}


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


def load_engine(args):
    """The model for validate/profile/profile-csv: float32 by default (as for the 8B runs); large models in float64
    with bfloat16 weight storage spread over GPUs (--dtype float64 --weights bf16 --device auto). Returns the engine
    and the device for inputs and SAEs (the embedding's device when layers are spread)."""
    import explore_llm_attractors as ela

    eng = ela.Engine(args.model, args.threads, args.dtype, args.device, args.weights)
    if args.device == "auto":
        return eng, str(eng.model.get_input_embeddings().weight.device)
    eng.model.to(args.device)                       # Engine moves only float64 models to the device itself
    return eng, args.device


def validate(args):
    """Hook-point check: FVE of each SAE on ordinary text at its documented index and the two neighbours."""
    import torch

    import explore_llm_attractors as ela

    eng, dev = load_engine(args)
    saes = [parse_sae(s, dev) for s in args.sae]
    report = {"model": args.model, "texts": len(TEXTS), "saes": [], "dtype": args.dtype, "weights": args.weights}
    hs_all = {}
    with torch.no_grad():
        for text in TEXTS:
            ids = eng.tok(text, return_tensors="pt").to(dev)
            out = eng.model(**ids, output_hidden_states=True)
            for i, h in enumerate(out.hidden_states):
                hs_all.setdefault(i, []).append(h[0, 1:].float().to(dev))  # drop the first token (attention sink)
    for sae in saes:
        rows, rel = {}, {}
        for i in (sae.index - 1, sae.index, sae.index + 1):
            if i in hs_all:
                H = torch.cat(hs_all[i])
                rows[i] = sae.fve(H)
                rec = sae.decode(sae.encode(H))
                rel[i] = float(((H - rec) ** 2).sum(-1).sqrt().mean() / (H ** 2).sum(-1).sqrt().mean())
        best = max(rows, key=rows.get)
        report["saes"].append({"sae": sae.name, "documented_index": sae.index, "fve_by_index": rows,
                               "relative_error_by_index": rel, "best_index": best,
                               "documented_is_best": best == sae.index})
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "validate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def profile(args):
    """Per-feature mean activation on the held-out category texts (plain text, first token dropped), for every SAE:
    {sae: {feature: {category: mean}}} over features active anywhere, with the token count per category."""
    import torch

    import explore_llm_attractors as ela

    eng, dev = load_engine(args)
    saes = [parse_sae(s, dev) for s in args.sae]
    sums = {s.name: {} for s in saes}
    counts = {}
    with torch.no_grad():
        for cat, texts in CATEGORIES.items():
            acc = {s.name: None for s in saes}
            n = 0
            for text in texts:
                ids = eng.tok(text, return_tensors="pt").to(dev)
                hs = eng.model(**ids, output_hidden_states=True).hidden_states
                for s in saes:
                    f = s.encode(hs[s.index][0, 1:].float().to(dev)).sum(0)
                    acc[s.name] = f if acc[s.name] is None else acc[s.name] + f
                n += ids["input_ids"].shape[1] - 1
            counts[cat] = n
            for s in saes:
                sums[s.name][cat] = (acc[s.name] / n).cpu().numpy()
    out_profiles = {}
    for s in saes:
        mat = np.stack([sums[s.name][c] for c in CATEGORIES])          # (categories, features)
        active = np.nonzero(mat.max(0) > 0)[0]
        out_profiles[s.name] = {str(int(f)): {c: round(float(mat[k, f]), 5) for k, c in enumerate(CATEGORIES)}
                                for f in active}
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "profiles.json").write_text(json.dumps({"token_counts": counts, "categories": list(CATEGORIES),
                                                   "profiles": out_profiles}))
    print(json.dumps({"token_counts": counts, "active_features": {k: len(v) for k, v in out_profiles.items()}}))


def profile_csv(args):
    """Per-text maximum activation of every SAE feature (first token dropped) on a designed corpus (CSV with
    item_id, category and text columns), saved as a compressed matrix (texts x features) per SAE with the item IDs
    and categories, for tools/selfref_features.py select."""
    import csv

    import torch

    import explore_llm_attractors as ela

    rows = list(csv.DictReader(open(args.corpus, newline="")))
    eng, dev = load_engine(args)
    saes = [parse_sae(s, dev) for s in args.sae]
    maxima = {s.name: np.zeros((len(rows), s.n_features), dtype=np.float32) for s in saes}
    with torch.no_grad():
        for i, r in enumerate(rows):
            ids = eng.tok(r["text"], return_tensors="pt").to(dev)
            hs = eng.model(**ids, output_hidden_states=True).hidden_states
            for s in saes:
                maxima[s.name][i] = s.encode(hs[s.index][0, 1:].float().to(dev)).max(0).values.cpu().numpy()
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    for s in saes:
        np.savez_compressed(out / f"maxima-{s.name}.npz", maxima=maxima[s.name],
                            item_id=np.array([r["item_id"] for r in rows]),
                            category=np.array([r["category"] for r in rows]))
    import hashlib
    meta = {"corpus": str(args.corpus), "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
            "texts": len(rows), "saes": [s.name for s in saes], "model": args.model}
    (out / "profile-csv.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


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
    for name in ("validate", "read", "profile", "profile-csv"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True)
        sp.add_argument("--threads", type=int, default=8)
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--sae", action="append", required=True, help="goodfire:PATH:LAYER or saelens:DIR")
        sp.add_argument("--out", required=True)
        sp.add_argument("--weights", default="native", choices=("native", "bf16"))
        if name != "read":
            sp.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    sub.choices["profile-csv"].add_argument("--corpus", required=True, help="CSV with item_id, category, text")
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
    {"validate": validate, "read": read, "profile": profile, "profile-csv": profile_csv}[args.cmd](args)


if __name__ == "__main__":
    main()
