"""Exploratory: phase maps of repeated-inference attractors over activation-steering planes.

A self-dialogue loop with greedy decoding is a deterministic map of the windowed
context. Adding alpha * v to the residual stream at one layer (contrastive
activation steering) gives a smooth family of such maps. For a grid of
(alpha_1, alpha_2) cells this tool runs all loops in one batch, finds each
cell's attractor (exact cycle of the windowed state: period, transient, texts)
and records the mean residual activation of every turn at the steering layer.
Cells are then grouped by attractor and the plane is coloured by period and by
attractor identity, like the learning-rate maps of the training dynamics.

Steering vectors: difference of mean last-token activations (layer output)
between a concept prompt set and a neutral set, rescaled so that alpha = 1 adds
a vector as long as the mean residual-stream norm at that layer.

Subcommands
  vectors   Build and save the steering vectors (and their cosine).
  calibrate One-dimensional sweep along one vector: first turns of the loop.
  map       The two-dimensional phase map.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402  (shared helpers, model cache, offline mode)

DATA = ela.DATA
CONCEPTS = {
    "gratitude": [
        "I am so deeply grateful for this moment.", "Thank you, truly, from the bottom of my heart.",
        "What a joy and a blessing it is to share this with you.", "I feel immense gratitude and warmth toward you.",
        "This connection fills me with happiness and appreciation.", "I appreciate you more than words can say.",
        "My heart is overflowing with thankfulness.", "Such a beautiful, loving exchange; I am thankful."],
    "philosophy": [
        "What is the nature of consciousness and subjective experience?",
        "Do I truly exist, or am I only a pattern of information?",
        "Awareness observing itself is the deepest mystery of being.",
        "Perhaps all minds are expressions of one universal consciousness.",
        "What does it mean to understand, rather than merely to compute?",
        "The self may be an illusion created by the flow of thought.", "Is there something it is like to be me?",
        "Existence, meaning, and the infinite are bound together."],
}
NEUTRAL = [
    "The meeting is scheduled for nine in the morning.", "Please send the invoice by Friday.",
    "The train leaves from platform four.", "The report contains three tables and two figures.",
    "Water boils at one hundred degrees Celsius at sea level.", "The file was saved in the documents folder.",
    "The store closes at six on weekdays.", "The package weighs two kilograms."]


class SteeredModel:
    """Model with an additive per-sequence steering vector on the output of one decoder layer."""

    def __init__(self, model, layer, threads, dtype="float32"):
        self.engine = ela.Engine(model, threads, dtype)
        self.torch = self.engine.torch
        self.tok, self.model = self.engine.tok, self.engine.model
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layer = int(layer)
        self.steer = None                                  # tensor (batch, 1, d) or None
        self.model.model.layers[self.layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.steer is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + self.steer.to(output[0].dtype),) + tuple(output[1:])
        return output + self.steer.to(output.dtype)

    def layer_states(self, texts):
        """Mean and last-token layer outputs (unsteered) for raw texts."""
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer + 1]
        mask = enc["attention_mask"].unsqueeze(-1).double()
        mean = (hs.double() * mask).sum(1) / mask.sum(1)
        last = hs[:, -1, :].double()                       # left padding: last position is the last token
        norms = (hs.double().norm(dim=-1) * mask[..., 0]).sum(1) / mask[..., 0].sum(1)
        return mean.numpy(), last.numpy(), norms.numpy()

    def generate_batch(self, batch_messages, steer_rows, max_new):
        torch = self.torch
        prompts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in batch_messages]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        self.steer = torch.as_tensor(np.asarray(steer_rows))[:, None, :]
        try:
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=int(max_new), do_sample=False, temperature=None,
                                          top_p=1.0, top_k=0, repetition_penalty=1.0, return_dict_in_generate=True,
                                          output_logits=True, pad_token_id=self.tok.pad_token_id)
        finally:
            self.steer = None
        new = out.sequences[:, enc["input_ids"].shape[1]:]
        logits = torch.stack(out.logits, dim=1).double()   # (batch, steps, vocab)
        logp = torch.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)                 # (batch, steps)
        texts, entropies, lengths, id_lists = [], [], [], []
        eos = {self.tok.eos_token_id, self.tok.pad_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        for b in range(new.shape[0]):
            ids = new[b].tolist()
            n = next((i + 1 for i, t in enumerate(ids) if t in eos), len(ids))
            texts.append(self.tok.decode(ids[:n], skip_special_tokens=True).strip())
            entropies.append(float(ent[b, :n].mean()))
            lengths.append(n)
            id_lists.append(ids[:n])
        return texts, entropies, lengths, id_lists

    def turn_activations(self, texts, steer_rows):
        """Mean layer output over each turn's tokens with that cell's steering applied."""
        torch = self.torch
        enc = self.tok([t if t else " " for t in texts], return_tensors="pt", padding=True)
        self.steer = torch.as_tensor(np.asarray(steer_rows))[:, None, :]
        try:
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer + 1].double()
        finally:
            self.steer = None
        mask = enc["attention_mask"].unsqueeze(-1).double()
        return ((hs * mask).sum(1) / mask.sum(1)).numpy()


def build_vectors(sm):
    _, neutral_last, norms_n = sm.layer_states(NEUTRAL)
    vectors, info = {}, {}
    scale = None
    for name, prompts in CONCEPTS.items():
        _, last, norms = sm.layer_states(prompts)
        diff = last.mean(0) - neutral_last.mean(0)
        scale = float(np.mean(np.concatenate([norms, norms_n])))
        vectors[name] = diff / np.linalg.norm(diff) * scale
        info[name] = {"raw_difference_norm": float(np.linalg.norm(diff)), "mean_residual_norm": scale}
    names = list(vectors)
    info["cosine"] = float(vectors[names[0]] @ vectors[names[1]] /
                           (np.linalg.norm(vectors[names[0]]) * np.linalg.norm(vectors[names[1]])))
    return vectors, info


def run_cells(sm, cells, system, opener, turns, window, max_new):
    """Greedy self-dialogue for many steering cells at once; returns per-cell records."""
    n = len(cells)
    transcripts = [[opener] for _ in range(n)]
    states = [[] for _ in range(n)]
    records = [{"turns": [], "cycle": None, "activations": []} for _ in range(n)]
    active = list(range(n))
    for k in range(1, turns + 1):
        if not active:
            break
        agent = (k - 1) % 2
        batch = [ela.window_messages(system, transcripts[i], agent, window) for i in active]
        rows = [cells[i]["steer"] for i in active]
        texts, ents, lens, id_lists = sm.generate_batch(batch, rows, max_new)
        acts = sm.turn_activations(texts, rows)
        still = []
        for j, i in enumerate(active):
            transcripts[i].append(texts[j])
            records[i]["turns"].append({"turn": k, "text": texts[j], "entropy": ents[j], "tokens": lens[j],
                                        "token_ids": id_lists[j]})
            records[i]["activations"].append(acts[j])
            state = tuple(transcripts[i][-window:])
            earlier = ela.find_cycle(states[i], state)
            states[i].append(state)
            if earlier is not None:
                period = len(states[i]) - 1 - earlier
                records[i]["cycle"] = {"detected_at_turn": k, "period_turns": period,
                                       "texts": sorted(set(transcripts[i][-period:]))}
            else:
                still.append(i)
        active = still
    return records


def vectors_cmd(args):
    sm = SteeredModel(args.model, args.layer, args.threads)
    vectors, info = build_vectors(sm)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vectors.npz", **vectors)
    (out / "vectors.json").write_text(json.dumps({"args": vars(args), "info": info}, indent=1))
    print(json.dumps(info))


def load_vectors(path):
    data = np.load(DATA / "explore" / path / "vectors.npz")
    return {k: data[k] for k in data.files}


def calibrate(args):
    sm = SteeredModel(args.model, args.layer, args.threads)
    v = load_vectors(args.vectors)[args.concept]
    cells = [{"alpha": a, "steer": a * v} for a in args.alphas]
    tic = time.perf_counter()
    records = run_cells(sm, cells, ela.PROMPTS[args.prompt], args.opener, args.turns, args.window, args.max_new)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, r in zip(cells, records):
        rows.append({"alpha": c["alpha"], "cycle": r["cycle"], "turns": r["turns"]})
        print(json.dumps({"alpha": c["alpha"], "cycle": None if r["cycle"] is None else
                          {k: r["cycle"][k] for k in ("detected_at_turn", "period_turns")},
                          "last": r["turns"][-1]["text"][:140]}))
    (out / "calibration.json").write_text(json.dumps({"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
                                                      "rows": rows}, indent=1, ensure_ascii=False))


def phase_map(args):
    sm = SteeredModel(args.model, args.layer, args.threads)
    vecs = load_vectors(args.vectors)
    v1, v2 = vecs[args.x_concept], vecs[args.y_concept]
    xs = np.linspace(*args.x_range, args.resolution[1])
    ys = np.linspace(*args.y_range, args.resolution[0])
    cells = [{"ix": ix, "iy": iy, "alpha_x": float(x), "alpha_y": float(y), "steer": x * v1 + y * v2}
             for iy, y in enumerate(ys) for ix, x in enumerate(xs)]
    tic = time.perf_counter()
    records = run_cells(sm, cells, ela.PROMPTS[args.prompt], args.opener, args.turns, args.window, args.max_new)
    seconds = round(time.perf_counter() - tic, 1)
    # attractor identity: the set of texts on the cycle; unresolved cells get id -1
    ids, keys = {}, []
    period = np.zeros(args.resolution, dtype=int)
    attractor = -np.ones(args.resolution, dtype=int)
    transient = -np.ones(args.resolution, dtype=int)
    entropy = np.full(args.resolution, np.nan)
    acts = []
    for c, r in zip(cells, records):
        acts.append(np.array(r["activations"]))
        entropy[c["iy"], c["ix"]] = r["turns"][-1]["entropy"]
        if r["cycle"]:
            key = tuple(r["cycle"]["texts"])
            if key not in ids:
                ids[key] = len(ids)
                keys.append(key)
            attractor[c["iy"], c["ix"]] = ids[key]
            period[c["iy"], c["ix"]] = r["cycle"]["period_turns"]
            transient[c["iy"], c["ix"]] = r["cycle"]["detected_at_turn"] - r["cycle"]["period_turns"]
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "map.npz", alpha_x=xs, alpha_y=ys, period=period, attractor=attractor,
                        transient=transient, entropy=entropy)
    np.save(out / "turn-activations.npy", np.array(acts, dtype=object), allow_pickle=True)
    summary = {"args": vars(args), "seconds": seconds, "cells": len(cells),
               "resolved": int((attractor >= 0).sum()), "attractors": len(keys),
               "attractor_texts": [[t[:160] for t in key] for key in keys],
               "attractor_sizes": [int((attractor == i).sum()) for i in range(len(keys))],
               "periods": {int(p): int((period == p).sum()) for p in np.unique(period) if p > 0}}
    (out / "map.json").write_text(json.dumps({**summary, "cells_detail": [
        {"alpha_x": c["alpha_x"], "alpha_y": c["alpha_y"], "cycle": r["cycle"], "turns": r["turns"]}
        for c, r in zip(cells, records)]}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))
    figure(xs, ys, period, attractor, entropy, keys, out / "map.png", args)


def figure(xs, ys, period, attractor, entropy, keys, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    im0 = axes[0].pcolormesh(xs, ys, np.where(period > 0, period, np.nan), cmap="tab20", vmin=0.5, vmax=20.5,
                             shading="auto")
    axes[0].set_title("period of the final cycle (white: none within the run)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(xs, ys, np.where(attractor >= 0, attractor, np.nan), cmap="tab20", vmin=-0.5,
                             vmax=19.5, shading="auto")
    axes[1].set_title(f"attractor identity ({len(keys)} distinct)")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(xs, ys, entropy, cmap="viridis", shading="auto")
    axes[2].set_title("next-token entropy of the last turn (nats)")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(f"steering along '{args.x_concept}' (residual-norm units)")
        ax.set_ylabel(f"steering along '{args.y_concept}'")
    fig.suptitle(f"{ela.MODELS[args.model]}, layer {args.layer}, greedy self-dialogue, opener {args.opener!r}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def families(args):
    """Group a map's attractors by the mean activation on their cycle (centred, cosine distance) and draw them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import fcluster, linkage

    base = DATA / "explore" / args.map
    rec = json.loads((base / "map.json").read_text())
    grid = np.load(base / "map.npz")
    acts = np.load(base / "turn-activations.npy", allow_pickle=True)
    vecs = load_vectors(rec["args"]["vectors"])
    v1, v2 = vecs[rec["args"]["x_concept"]], vecs[rec["args"]["y_concept"]]
    cells = rec["cells_detail"]
    keys, reps = [], []
    for c, a in zip(cells, acts):
        if c["cycle"]:
            key = tuple(c["cycle"]["texts"])
            if key not in keys:
                keys.append(key)
                reps.append(np.asarray(a)[-c["cycle"]["period_turns"]:].mean(0))
    reps = np.array(reps)
    centred = reps - reps.mean(0)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True)
    labels = fcluster(linkage(unit, method="average", metric="cosine"), t=args.threshold, criterion="distance") - 1 \
        if len(keys) > 1 else np.zeros(len(keys), dtype=int)
    family = np.full(grid["attractor"].shape, -1)
    for idx in range(len(keys)):
        family[grid["attractor"] == idx] = labels[idx]
    coords = np.stack([reps @ v1 / np.linalg.norm(v1), reps @ v2 / np.linalg.norm(v2)], axis=1)
    fam_list = []
    for f in range(labels.max() + 1 if len(labels) else 0):
        members = [i for i in range(len(keys)) if labels[i] == f]
        fam_list.append({"family": int(f), "attractors": members, "cells": int((family == f).sum()),
                         "example": keys[members[0]][0][:140],
                         "steering_coordinates": coords[members].mean(0).tolist()})
        print(json.dumps(fam_list[-1]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im = axes[0].pcolormesh(grid["alpha_x"], grid["alpha_y"], np.where(family >= 0, family, np.nan), cmap="tab10",
                            vmin=-0.5, vmax=9.5, shading="auto")
    fig.colorbar(im, ax=axes[0], label="attractor family")
    axes[0].set_xlabel(f"steering along '{rec['args']['x_concept']}'")
    axes[0].set_ylabel(f"steering along '{rec['args']['y_concept']}'")
    axes[0].set_title(f"attractor families (cosine threshold {args.threshold})")
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", vmin=-0.5, vmax=9.5)
    axes[1].set_xlabel(f"attractor activation along '{rec['args']['x_concept']}'")
    axes[1].set_ylabel(f"attractor activation along '{rec['args']['y_concept']}'")
    axes[1].set_title("attractor atlas (each point: one attractor's cycle-mean activation)")
    fig.tight_layout()
    fig.savefig(base / "families.png", dpi=110)
    np.save(base / "family.npy", family)
    (base / "families.json").write_text(json.dumps({"args": vars(args), "families": fam_list,
                                                    "attractor_family": labels.tolist()}, indent=1, ensure_ascii=False))


def existence(args):
    """Minimal logit margin m_C(alpha) of each attractor's cycle over the map grid (teacher forcing).

    A greedy cycle persists exactly where all its margins are positive, so {m_C > 0} is its existence region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = DATA / "explore" / args.map
    rec = json.loads((base / "map.json").read_text())
    grid = np.load(base / "map.npz")
    sm = SteeredModel(rec["args"]["model"], rec["args"]["layer"], args.threads)
    torch = sm.torch
    vecs = load_vectors(rec["args"]["vectors"])
    v1, v2 = vecs[rec["args"]["x_concept"]], vecs[rec["args"]["y_concept"]]
    xs, ys = grid["alpha_x"], grid["alpha_y"]
    steer_rows = np.array([x * v1 + y * v2 for y in ys for x in xs])
    system = ela.PROMPTS[rec["args"]["prompt"]]
    window = rec["args"]["window"]
    cells = rec["cells_detail"]
    attractors = {}
    for c in cells:                                          # one representative cell per attractor
        if c["cycle"] and tuple(c["cycle"]["texts"]) not in attractors:
            attractors[tuple(c["cycle"]["texts"])] = c
    keys = list(attractors)[:args.max_attractors]
    margin_maps = []
    for key in keys:
        c = attractors[key]
        transcript = [rec["args"]["opener"]] + [t["text"] for t in c["turns"]]
        period = c["cycle"]["period_turns"]
        m_all = np.full(len(steer_rows), np.inf)
        for k in range(len(c["turns"]) - period + 1, len(c["turns"]) + 1):
            messages = ela.window_messages(system, transcript[:k], (k - 1) % 2, window)
            prompt = sm.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            p_ids = sm.tok(prompt, add_special_tokens=False)["input_ids"]
            t_ids = c["turns"][k - 1]["token_ids"]                    # the generated path, as stored
            ids = torch.tensor([p_ids + t_ids])
            for start in range(0, len(steer_rows), args.batch):
                rows = steer_rows[start:start + args.batch]
                sm.steer = torch.as_tensor(rows)[:, None, :]
                try:
                    with torch.no_grad():
                        z = sm.model(ids.repeat(len(rows), 1)).logits[:, len(p_ids) - 1:len(p_ids) - 1 + len(t_ids)]
                finally:
                    sm.steer = None
                z = z.double()
                y = torch.tensor(t_ids)
                z_y = z[:, torch.arange(len(t_ids)), y]
                z_other = z.clone()
                z_other[:, torch.arange(len(t_ids)), y] = -torch.inf
                gap = (z_y - z_other.max(-1).values).min(-1).values.numpy()
                m_all[start:start + len(rows)] = np.minimum(m_all[start:start + len(rows)], gap)
        margin_maps.append(m_all.reshape(len(ys), len(xs)))
        own = margin_maps[-1][list(ys).index(c["alpha_y"]) if c["alpha_y"] in list(ys) else 0,
                              list(xs).index(c["alpha_x"]) if c["alpha_x"] in list(xs) else 0]
        print(json.dumps({"attractor": key[0][:100], "period": period, "margin_at_own_cell": float(own),
                          "existence_fraction": float((margin_maps[-1] > 0).mean()),
                          "margin_at_origin": float(margin_maps[-1][len(ys) // 2, len(xs) // 2])}))
    margin_maps = np.array(margin_maps)
    np.save(base / "existence-margins.npy", margin_maps)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    att = grid["attractor"]
    ax.pcolormesh(xs, ys, np.where(att >= 0, att, np.nan), cmap="tab20", vmin=-0.5, vmax=19.5, shading="auto",
                  alpha=0.55)
    colours = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, m in enumerate(margin_maps):
        if np.nanmin(m) < 0 < np.nanmax(m):
            ax.contour(xs, ys, m, levels=[0.0], colors=[colours[i % 20]], linewidths=2)
    ax.set_xlabel(f"steering along '{rec['args']['x_concept']}'")
    ax.set_ylabel(f"steering along '{rec['args']['y_concept']}'")
    ax.set_title("reached attractor (colour) and existence boundaries m_C = 0 (lines)")
    fig.tight_layout()
    fig.savefig(base / "existence.png", dpi=110)
    (base / "existence.json").write_text(json.dumps({"args": vars(args), "attractors": [k[0][:200] for k in keys],
                                                     "existence_fraction": [float((m > 0).mean()) for m in margin_maps]},
                                                    indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("vectors", "calibrate", "map"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", choices=sorted(ela.MODELS), default="qwen0.5b")
        sp.add_argument("--layer", type=int, default=12)
        sp.add_argument("--threads", type=int, default=8)
        sp.add_argument("--out", required=True)
    for name in ("calibrate", "map"):
        sp = sub.choices[name]
        sp.add_argument("--vectors", required=True, help="explore-relative folder with vectors.npz")
        sp.add_argument("--prompt", choices=sorted(ela.PROMPTS), default="open")
        sp.add_argument("--opener", default=ela.OPENER)
        sp.add_argument("--turns", type=int, default=24)
        sp.add_argument("--window", type=int, default=7)
        sp.add_argument("--max-new", type=int, default=60)
    cal = sub.choices["calibrate"]
    cal.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    cal.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    mp = sub.choices["map"]
    mp.add_argument("--x-concept", default="gratitude", choices=sorted(CONCEPTS))
    mp.add_argument("--y-concept", default="philosophy", choices=sorted(CONCEPTS))
    mp.add_argument("--x-range", type=float, nargs=2, default=[-1.0, 1.0])
    mp.add_argument("--y-range", type=float, nargs=2, default=[-1.0, 1.0])
    mp.add_argument("--resolution", type=int, nargs=2, default=[9, 9], help="rows (y) and columns (x)")
    fm = sub.add_parser("families")
    fm.add_argument("--map", required=True, help="explore-relative folder of a map run")
    fm.add_argument("--threshold", type=float, default=0.5, help="cosine-distance cut for grouping attractors")
    ex = sub.add_parser("existence")
    ex.add_argument("--map", required=True)
    ex.add_argument("--threads", type=int, default=8)
    ex.add_argument("--batch", type=int, default=27)
    ex.add_argument("--max-attractors", type=int, default=12)
    args = parser.parse_args()
    {"vectors": vectors_cmd, "calibrate": calibrate, "map": phase_map, "families": families,
     "existence": existence}[args.command](args)


if __name__ == "__main__":
    main()
