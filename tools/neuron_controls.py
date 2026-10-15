"""Steering along single MLP neurons' output directions (run NC; control requested by the human after reading
Cammarata's essay "Neural Networks Are Already Clean"): does a neuron's own write direction drive the soft-token map
into chaos as the contrastive and random directions do?

profile  Mean activation of every MLP neuron of decoder layer LAYER (the input to down_proj, act(gate) * up) over the
         held-out category sentences of tools/llm_sae_readout.py (CATEGORIES, written before any readout), averaged
         over tokens and then over the sentences of each category. CPU, float32.
select   The rule of tools/steering_controls.py, applied to neurons: for the target category, neuron f scores
         profile_f[target] minus the largest profile_f over the other categories (leaving out RELATED); the highest
         score wins (ties: lower ID). Also its index neighbour (ID + 1) and a random neuron
         (numpy default_rng(0), uniform over all neurons). The rule is fixed before any map is computed.
build    The selected neurons' output columns (down_proj.weight[:, f]), each rescaled to the norm of the reference
         concept vector of an existing vector set, written as a new vector set with cosines to that concept. Decoder
         layer LAYER's MLP writes into the residual stream whose output the steering hook at LAYER adds to.

Usage:
  python tools/neuron_controls.py profile --layer 12 --out llm-soft-loop/controls/neurons-qwen0.5b-l12/profiles.npz
  python tools/neuron_controls.py select --profiles ... --target philosophy --json docs/nc/nc-selection.json
  python tools/neuron_controls.py build --selection docs/nc/nc-selection.json --reference-vectors
      llm-soft-loop/controls/vectors-qwen0.5b-l12-philnorm --reference-concept philosophy
      --out llm-soft-loop/controls/vectors-qwen0.5b-l12-neurons
(vector-set and profile paths are relative to the data directory's explore/)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

RULE = ("For the target category, neuron f scores profile_f[target] minus the largest profile_f over the other "
        "categories, leaving out the related ones in RELATED; the highest score wins (ties: lower ID). Also: its index "
        "neighbour (ID + 1) and a random neuron (numpy default_rng(0), uniform over all neurons of the layer).")
RELATED = {"hostility": ("swearing",), "philosophy": (), "gratitude": ("politeness",)}
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import explore_llm_attractors as ela
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=ela.CACHE)              # the project's pinned model cache
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, cache_dir=ela.CACHE)
    model.eval()
    return torch, tok, model


def profile(args):
    import explore_llm_attractors as ela
    import llm_sae_readout as lsr
    torch, tok, model = load_model()
    mlp = model.model.layers[args.layer].mlp
    captured = {}

    def hook(_mod, inp):
        captured["a"] = inp[0].detach()                    # input of down_proj: act(gate) * up, (1, tokens, 4864)
    h = mlp.down_proj.register_forward_pre_hook(hook)
    cats = list(lsr.CATEGORIES)
    prof = np.zeros((len(cats), mlp.down_proj.in_features))
    with torch.no_grad():
        for k, c in enumerate(cats):
            per = []
            for text in lsr.CATEGORIES[c]:
                ids = tok(text, return_tensors="pt")
                model(**ids)
                per.append(captured["a"][0].mean(dim=0).double().numpy())
            prof[k] = np.mean(per, axis=0)
    h.remove()
    out = ela.DATA / "explore" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, profile=prof, categories=np.array(cats), layer=args.layer, model=MODEL)
    print(json.dumps({"out": str(out), "categories": cats, "neurons": prof.shape[1],
                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


def select(args):
    import explore_llm_attractors as ela
    path = ela.DATA / "explore" / args.profiles
    d = np.load(path, allow_pickle=True)
    prof, cats = d["profile"], [str(c) for c in d["categories"]]
    t = cats.index(args.target)
    others = [k for k, c in enumerate(cats) if k != t and c not in RELATED.get(args.target, ())]
    score = prof[t] - prof[others].max(axis=0)
    order = np.lexsort((np.arange(len(score)), -score))   # highest score, ties to the lower ID
    best, runner = int(order[0]), int(order[1])
    n = prof.shape[1]
    rnd = int(np.random.default_rng(0).integers(0, n))
    rec = {"model": MODEL, "layer": int(d["layer"]), "profiles": str(path),
           "profiles_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rule": RULE,
           "related": {k: list(v) for k, v in RELATED.items()}, "target": args.target,
           "neurons": {"target": {"neuron": best, "score": round(float(score[best]), 6),
                                  "profile": {c: round(float(prof[k, best]), 6) for k, c in enumerate(cats)},
                                  "runner_up": {"neuron": runner, "score": round(float(score[runner]), 6)}},
                       "neighbour": {"neuron": best + 1, "of": best,
                                     "profile": {c: round(float(prof[k, best + 1]), 6) for k, c in enumerate(cats)}},
                       "random": {"neuron": rnd, "pool": n,
                                  "profile": {c: round(float(prof[k, rnd]), 6) for k, c in enumerate(cats)}}}}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec["neurons"], indent=1))


def build(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    sel = json.loads(Path(args.selection).read_text())
    torch, tok, model = load_model()
    W = model.model.layers[sel["layer"]].mlp.down_proj.weight.detach().double().numpy()     # (hidden, neurons)
    ref = elpm.load_vectors(args.reference_vectors)[args.reference_concept].astype(np.float64)
    norm = float(np.linalg.norm(ref))
    vecs, meta = {}, {}
    for role in ("target", "neighbour", "random"):
        f = sel["neurons"][role]["neuron"]
        w = W[:, f]
        v = w / np.linalg.norm(w) * norm
        key = f"neuron-{role}-{f}"
        vecs[key] = v
        meta[key] = {"neuron": f, "role": role, "column_norm": round(float(np.linalg.norm(w)), 6),
                     "cos_to_reference": round(float(v @ ref / (np.linalg.norm(v) * norm)), 6)}
    vecs[args.reference_concept] = ref
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vectors.npz", **vecs)
    (out / "vectors.json").write_text(json.dumps({"layer": sel["layer"], "model": MODEL, "reference_norm": norm,
                                                  "reference": f"{args.reference_vectors}:{args.reference_concept}",
                                                  "selection": args.selection, "vectors": meta}, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--profiles", required=True)
    s.add_argument("--target", default="philosophy")
    s.add_argument("--json", required=True)
    b = sub.add_parser("build")
    b.add_argument("--selection", required=True)
    b.add_argument("--reference-vectors", required=True)
    b.add_argument("--reference-concept", default="philosophy")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    {"profile": profile, "select": select, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
