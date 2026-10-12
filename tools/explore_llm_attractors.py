"""Exploratory pilot: attractors of repeated LLM inference (self-dialogue and repeated paraphrasing).

Repeated inference with fixed weights is an iterated map. With greedy decoding
and a bounded context window the map is deterministic on a finite set, so every
run ends in a cycle; with sampling it is a Markov chain, whose long-lived
regions play the role of attractors. This tool records, per turn, the text,
the model's mean next-token entropy (from raw logits, temperature 1), the
turn's mean last-layer hidden state (an embedding), and detects exact cycles of
the windowed state.

Sampling is plain: the model's default repetition penalty, top-k and top-p are
switched off (repetition_penalty 1, top_k 0, top_p 1) so that loops are not
suppressed by a heuristic; only the temperature varies.

Subcommands
  dialogue    Two instances of one chat model talk. Each instance sees the last
              K turns, its own as "assistant" and the other's as "user".
  paraphrase  x_{n+1} = paraphrase(x_n) for one instance.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
CACHE = DATA / "models" / "hf" / "hub"   # passed explicitly: a global HF_HOME may point elsewhere
os.environ.setdefault("HF_HUB_OFFLINE", "1")                      # models are local; no network calls
MODELS ={"qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}
PROMPTS = {
    "open": "You are an AI talking with another AI. You have complete freedom. Feel free to pursue whatever you want.",
    "task": "You are an AI collaborating with another AI on one task: planning a three-day trip to Kyoto. "
            "Stay on the task.",
}
OPENER = "Hello!"


def window_messages(system, transcript, agent, window):
    """Chat messages for `agent` (0 or 1) from the last `window` turns.

    transcript[0] is the opener, attributed to agent 1; turn i >= 1 was said by agent (i - 1) % 2, so even
    indices belong to agent 1 and odd ones to agent 0. The window opens with the other agent's turn once the
    transcript is longer than the window (an odd window keeps the role pattern identical for both agents)."""
    start = max(0, len(transcript) - window)
    if (len(transcript) - start) % 2 == 0 and start > 0:
        start -= 1                                          # the window must open with the other agent's turn
    messages = [{"role": "system", "content": system}]
    for i in range(start, len(transcript)):
        speaker = 1 if i % 2 == 0 else 0
        messages.append({"role": "assistant" if speaker == agent else "user", "content": transcript[i]})
    return messages


def find_cycle(states, new_state):
    """Index of the earlier occurrence of new_state in states, or None."""
    for i in range(len(states) - 1, -1, -1):
        if states[i] == new_state:
            return i
    return None


class Engine:
    def __init__(self, model, threads, dtype="float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.name = MODELS[model]
        self.tok = AutoTokenizer.from_pretrained(self.name, cache_dir=CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=getattr(torch, dtype), cache_dir=CACHE)
        self.model.eval()

    def generate(self, messages, temperature, max_new, seed):
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        torch.manual_seed(int(seed))
        kwargs = {"max_new_tokens": int(max_new), "return_dict_in_generate": True, "output_logits": True,
                  "repetition_penalty": 1.0, "top_k": 0, "top_p": 1.0, "pad_token_id": self.tok.eos_token_id}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            kwargs.update(do_sample=False, temperature=None)
        with torch.no_grad():
            out = self.model.generate(ids, **kwargs)
        new = out.sequences[0, ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        logits = torch.stack(out.logits)[:, 0, :].double()
        logp = torch.log_softmax(logits, dim=-1)
        entropy = float((-(logp.exp() * logp).sum(-1)).mean()) if len(logits) else 0.0
        return text, int(len(new)), entropy, [int(t) for t in new]

    def stay_logprob(self, messages, token_ids, temperatures):
        """log P(the model regenerates exactly token_ids after messages) at each sampling temperature."""
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        full = torch.cat([ids, torch.tensor([token_ids])], dim=1)
        with torch.no_grad():
            logits = self.model(full).logits[0].double()
        n0 = ids.shape[1]
        z = logits[n0 - 1:n0 - 1 + len(token_ids)]
        y = torch.tensor(token_ids)
        return {f"{t:g}": float(torch.log_softmax(z / t, dim=-1)[torch.arange(len(y)), y].sum())
                for t in temperatures}

    def embed(self, text):
        torch = self.torch
        enc = self.tok(text if text else " ", return_tensors="pt")
        with torch.no_grad():
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1][0]
        return hidden.mean(0).double().numpy()


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def jaccard(a, b):
    """Word-set overlap of two texts (lexical similarity; mean-pooled hidden states are too anisotropic)."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def trailing_block(texts):
    """Length and starting index of the final run of identical texts."""
    n = 1
    while n < len(texts) and texts[-1 - n] == texts[-1]:
        n += 1
    return n, len(texts) - n


def dialogue(args):
    engine = Engine(args.model, args.threads)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    system = PROMPTS[args.prompt]
    transcript, states, turns, embeddings = [args.opener], [], [], []
    cycle, last_messages, last_ids = None, None, None
    tic = time.perf_counter()
    for k in range(1, args.turns + 1):
        agent = (k - 1) % 2
        messages = window_messages(system, transcript, agent, args.window)
        text, n_tok, entropy, ids = engine.generate(messages, args.temperature, args.max_new,
                                                    args.seed * 100003 + k)
        last_messages, last_ids = messages, ids
        transcript.append(text)
        emb = engine.embed(text)
        embeddings.append(emb)
        state = tuple(transcript[-args.window:])
        earlier = find_cycle(states, state)
        states.append(state)
        row = {"turn": k, "agent": agent, "tokens": n_tok, "entropy": entropy, "text": text, "token_ids": ids,
               "jaccard_previous_turn": jaccard(text, transcript[-2]) if k > 1 else None,
               "cos_previous_turn": cosine(emb, embeddings[-2]) if len(embeddings) > 1 else None,
               "cos_same_agent_previous": cosine(emb, embeddings[-3]) if len(embeddings) > 2 else None}
        turns.append(row)
        if earlier is not None and cycle is None:
            cycle = {"detected_at_turn": k, "period_turns": len(states) - 1 - earlier}
            if args.stop_on_cycle:
                break
    block, block_start = trailing_block(transcript[1:])
    stay = engine.stay_logprob(last_messages, last_ids, args.stay_temperatures)
    record = {"args": vars(args), "model": engine.name, "system": system, "opener": args.opener, "cycle": cycle,
              "final_repeat_block": {"length": block, "first_turn": block_start + 1},
              "stay_logprob_last_turn": stay, "seconds": round(time.perf_counter() - tic, 1), "turns": turns}
    (out / "dialogue.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    np.save(out / "embeddings.npy", np.array(embeddings))
    print(json.dumps({"out": args.out, "cycle": cycle, "turns": len(turns), "seconds": record["seconds"],
                      "final_repeat_block": record["final_repeat_block"],
                      "last_tokens": [t["tokens"] for t in turns[-4:]],
                      "last_entropy": [round(t["entropy"], 3) for t in turns[-4:]]}))


def paraphrase(args):
    engine = Engine(args.model, args.threads)
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    text = args.text
    history, rows, embeddings, cycle = [text], [], [], None
    tic = time.perf_counter()
    for k in range(1, args.rounds + 1):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Paraphrase the following text. Reply with the paraphrase only.\n\n"
                                                + text}]
        text, n_tok, entropy, _ = engine.generate(messages, args.temperature, args.max_new, args.seed * 100003 + k)
        emb = engine.embed(text)
        embeddings.append(emb)
        earlier = find_cycle(history, text)
        history.append(text)
        rows.append({"round": k, "tokens": n_tok, "entropy": entropy, "text": text,
                     "cos_previous": cosine(emb, embeddings[-2]) if len(embeddings) > 1 else None,
                     "cos_two_back": cosine(emb, embeddings[-3]) if len(embeddings) > 2 else None})
        if earlier is not None and cycle is None:
            cycle = {"detected_at_round": k, "period": len(history) - 1 - earlier}
            if args.stop_on_cycle:
                break
    record = {"args": vars(args), "model": engine.name, "cycle": cycle, "seconds": round(time.perf_counter() - tic, 1),
              "rounds": rows}
    (out / "paraphrase.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    np.save(out / "embeddings.npy", np.array(embeddings))
    print(json.dumps({"out": args.out, "cycle": cycle, "rounds": len(rows), "seconds": record["seconds"]}))


def first_run(flags, length=3):
    """First index starting `length` consecutive True values, or None."""
    count = 0
    for i, f in enumerate(flags):
        count = count + 1 if f else 0
        if count == length:
            return i - length + 1
    return None


def summary(args):
    """Per-run attractor statistics for saved dialogue and paraphrase runs, and a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = DATA / "explore" / args.folder
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for run in sorted(base.iterdir()):
        kind = "dialogue" if (run / "dialogue.json").exists() else "paraphrase" if (run / "paraphrase.json").exists() \
            else None
        if kind is None:
            continue
        rec = json.loads((run / f"{kind}.json").read_text())
        steps = rec["turns"] if kind == "dialogue" else rec["rounds"]
        ent = [t["entropy"] for t in steps]
        tok = [t["tokens"] for t in steps]
        cos_key = "cos_previous_turn" if kind == "dialogue" else "cos_previous"
        cos = [t[cos_key] if t[cos_key] is not None else np.nan for t in steps]
        texts = [t["text"] for t in steps]
        tail = slice(max(0, len(steps) - args.tail), len(steps))
        block = rec.get("final_repeat_block")
        stay = rec.get("stay_logprob_last_turn", {})
        temp = rec["args"].get("temperature", 0.0)
        jac = [t.get("jaccard_previous_turn") for t in steps if t.get("jaccard_previous_turn") is not None]
        row = {"run": run.name, "kind": kind, "steps": len(steps), "cycle": rec["cycle"],
               "temperature": temp, "opener": rec.get("opener"),
               "absorbed": bool(block and block["length"] >= args.absorbed_length),
               "absorbed_from_turn": block["first_turn"] if block and block["length"] >= args.absorbed_length else None,
               "stay_probability_at_run_temperature": (float(np.exp(stay[f"{temp:g}"]))
                                                        if temp > 0 and f"{temp:g}" in stay else None),
               "tail_jaccard": float(np.mean(jac[-args.tail:])) if jac else None,
               "collapse_turn": first_run([e < args.entropy_level for e in ent]),
               "mirroring_turn": first_run([c > args.cos_level for c in cos]),
               "tail_entropy": float(np.mean(ent[tail])), "tail_tokens": float(np.mean(tok[tail])),
               "tail_cos": float(np.nanmean(cos[tail])), "tail_distinct_texts": len(set(texts[tail])),
               "last_text": texts[-1][:160]}
        period = (rec["cycle"].get("period_turns") or rec["cycle"].get("period")) if rec["cycle"] else None
        # greedy decoding is a deterministic map of the windowed state, so a repeated state is a genuine cycle
        # (and the last `period` turns lie on it); for sampled runs require two identical final periods
        persistent = bool(period and (temp == 0 or (len(texts) >= 2 * period
                                                     and texts[-period:] == texts[-2 * period:-period])))
        row["cycle_period"] = period if persistent else None
        row["attractor_texts"] = sorted(set(texts[-period:])) if persistent else None
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("last_text", "attractor_texts")}))
        style = "-" if "open" in run.name or kind == "paraphrase" else "--"
        axes[0].plot(range(1, len(ent) + 1), ent, style, label=run.name)
        axes[1].plot(range(1, len(tok) + 1), tok, style)
        axes[2].plot(range(1, len(cos) + 1), cos, style)
    # basins: runs grouped by the set of texts on their persistent final cycle
    basins = {}
    for row in rows:
        if row["attractor_texts"]:
            basins.setdefault(tuple(row["attractor_texts"]), []).append(row["run"])
    basin_list = [{"period": next(r["cycle_period"] for r in rows if r["run"] == runs[0]),
                   "texts": [t[:120] for t in key], "runs": runs} for key, runs in basins.items()]
    for b in basin_list:
        print(json.dumps({"basin_period": b["period"], "runs": b["runs"], "first_text": b["texts"][0]}))
    axes[0].set_ylabel("mean next-token entropy (nats)")
    axes[1].set_ylabel("tokens per turn")
    axes[2].set_ylabel("cosine to previous turn (hidden-state mean)")
    for ax in axes:
        ax.set_xlabel("turn")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(base / "summary.png", dpi=110)
    (base / "summary.json").write_text(json.dumps({"args": vars(args), "rows": rows, "basins": basin_list}, indent=1,
                                                  ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dialogue", "paraphrase"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", choices=sorted(MODELS), default="qwen0.5b")
        sp.add_argument("--temperature", type=float, default=0.0)
        sp.add_argument("--max-new", type=int, default=100)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--stop-on-cycle", action="store_true")
        sp.add_argument("--out", required=True)
    dl = sub.choices["dialogue"]
    dl.add_argument("--prompt", choices=sorted(PROMPTS), default="open")
    dl.add_argument("--turns", type=int, default=30)
    dl.add_argument("--window", type=int, default=7, help="turns each instance sees (odd keeps role order)")
    dl.add_argument("--opener", default=OPENER, help="turn 0, attributed to the second instance")
    dl.add_argument("--stay-temperatures", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    pp = sub.choices["paraphrase"]
    pp.add_argument("--text", default="The river carried the old boat slowly past the sleeping village at dawn.")
    pp.add_argument("--rounds", type=int, default=30)
    sm = sub.add_parser("summary")
    sm.add_argument("--folder", required=True, help="explore-relative folder of runs")
    sm.add_argument("--tail", type=int, default=6)
    sm.add_argument("--entropy-level", type=float, default=0.3)
    sm.add_argument("--cos-level", type=float, default=0.95)
    sm.add_argument("--absorbed-length", type=int, default=5, help="identical final turns that count as absorbed")
    args = parser.parse_args()
    {"dialogue": dialogue, "paraphrase": paraphrase, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
