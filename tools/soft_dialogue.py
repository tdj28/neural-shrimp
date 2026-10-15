"""Two agents in one soft-token conversation (human, 15 Oct 2026, after the "spiritual bliss" attractor of model-to-model
conversations: "couldn't we try to show two LLMs inducing each other into a periodic orbit of some form?").

Agents A and B are one model with the system prompt of the self-dialogue pilots (the open prompt: "You are an AI
talking with another AI. You have complete freedom. ...") and their own steering, alpha_A v_A and alpha_B v_B at one
block. They speak alternately, one soft token per message: x_(n+1) = F_A(x_n), x_(n+2) = F_B(x_(n+1)), with
F(x) = E^T softmax(z(x)/T) as in the soft-token loop. Two frames:

- dialogue: an agent hears the other's message as a user turn and answers as the assistant. Its context is the system
  prompt, the user turn holding the message, and the assistant header; its next-token distribution there is its
  message. The conversation opens with the first token of the pilots' opener ("Hello"), said by B.
- cowrite: the agents continue one assistant reply alternately, i.e. the paper's soft-token loop with the steering
  switching between the agents at every token. Equal agents give the paper's map exactly.

One step of the classifier is one exchange (A speaks, then B), so the classifier sees the autonomous exchange map
G = F_B o F_A on B's messages: periods are in exchanges and exponents per exchange (twice the per-message rate); the
leading tokens are B's. Every cell runs with the twins and the labels of tools/hidden_chaos.py (run_batch, classify).
Equal agents (alpha_A = alpha_B, one direction) are one agent talking with itself, so a pair map over (alpha_A,
alpha_B) holds the single agents on its diagonal.

Usage:
  python tools/soft_dialogue.py pairmap --frame dialogue --log-temperature -0.1 --alpha-a 0.06 0.082 48 \\
      --alpha-b 0.06 0.082 48 --vectors llm-phase-map/vectors-l12 --concept-a philosophy --out llm-soft-loop/dialogue/X
  python tools/soft_dialogue.py selfmap --frame dialogue --log-temperature -0.4 0.4 48 --alpha 0 0.12 48 ...
  python tools/soft_dialogue.py cells --cells-file F.json ...   (F: a list 'cells' of [log10T, alpha_A, alpha_B])
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import explore_llm_attractors as ela  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402
import hidden_chaos as hc  # noqa: E402


class DialogueLoop(esl.SoftLoop):
    """SoftLoop (window 1) whose fed-back token sits in a user turn: the cached prefix is the system prompt and the user
    header, and a fixed suffix (end of turn and assistant header, from the model's chat template) follows the token,
    so the next-token distribution is the assistant's first reply token to a one-token message."""

    MARK = "<<MESSAGE>>"

    def __init__(self, model, layer, threads, prompt, dtype="float64", device="cpu", weights="native"):
        super().__init__(model, layer, threads, prompt, ela.OPENER, 1, dtype, device, weights)
        torch, tok = self.torch, self.sm.tok
        messages = [{"role": "system", "content": ela.PROMPTS[prompt]}, {"role": "user", "content": self.MARK}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if text.count(self.MARK) != 1:
            raise SystemExit("chat template: the message marker must occur exactly once")
        pre, suf = text.split(self.MARK)
        self.prefix_ids = torch.tensor([tok(pre, add_special_tokens=False)["input_ids"]], device=self.device)
        self.L0 = self.prefix_ids.shape[1]
        self.suffix_ids = tok(suf, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            self.suffix = self.model.get_input_embeddings()(
                torch.tensor(self.suffix_ids, device=self.device)).to(self.state_dtype)          # (S, d)

    def _framed(self, X):
        return self.torch.cat([X, self.suffix[None].expand(X.shape[0], -1, -1)], dim=1)

    def logits(self, X):
        return super().logits(self._framed(X))

    def step_jvp(self, X, temps, V):
        torch = self.torch
        Y, p, JV = super().step_jvp(self._framed(X), temps,
                                    torch.cat([V, torch.zeros((V.shape[0], len(self.suffix_ids), V.shape[2]),
                                                              dtype=V.dtype, device=V.device)], dim=1))
        return Y[:, -1:], p, JV[:, -1:]                 # the new message is the last position of the shifted input


class PairLoop:
    """Two agents on one loop (shared weights): one prefix cache per agent's steering, used alternately. step is one
    exchange, A then B, so the state is B's last message and the map is G = F_B o F_A. set_b_cells gives agent B's
    steering for the cells of the next batch; prepare(rows) takes agent A's rows (base and twins, as hidden_chaos.run_batch
    builds them) and repeats B's cell rows the same way. start: the opening message (d,), or None for the loop's own
    start. With record_a, the leading tokens of A's messages are kept per step (a_lead, rows of the whole batch)."""

    def __init__(self, loop, start=None, record_a=False):
        self.loop = loop
        self.torch, self.state_dtype, self.device, self.E = loop.torch, loop.state_dtype, loop.device, loop.E
        self.start = start
        self.b_cells = None
        self.record_a = record_a
        self.a_lead = []

    def set_b_cells(self, rows):
        self.b_cells = np.asarray(rows)

    def _save(self):
        return self.loop.cache, self.loop.steer_rows, self.loop.extra_rows

    def _use(self, s):
        self.loop.cache, self.loop.steer_rows, self.loop.extra_rows = s

    def prepare(self, rows_a):
        if self.b_cells is None:
            raise ValueError("set_b_cells before prepare")
        B, n = len(rows_a), len(self.b_cells)
        if B % n:
            raise ValueError(f"{B} rows are not a multiple of {n} cells of agent B")
        self.loop.prepare(np.concatenate([self.b_cells] * (B // n)))
        self.state_b = self._save()
        X = self.loop.prepare(rows_a)
        self.state_a = self._save()
        self.a_lead = []
        if self.start is not None:
            X = self.start.to(device=X.device, dtype=X.dtype)[None, None, :].repeat(B, X.shape[1], 1).clone()
        return X

    def step(self, X, temps):
        self._use(self.state_a)
        X, pa = self.loop.step(X, temps)
        if self.record_a:
            self.a_lead.append(pa.argmax(dim=-1).cpu().numpy())
        self._use(self.state_b)
        return self.loop.step(X, temps)

    def step_jvp(self, X, temps, V):
        self._use(self.state_a)
        X, pa, V = self.loop.step_jvp(X, temps, V)
        if self.record_a:
            self.a_lead.append(pa.argmax(dim=-1).cpu().numpy())
        self._use(self.state_b)
        return self.loop.step_jvp(X, temps, V)


def build(args):
    """The loop of the chosen frame and the pair around it."""
    if args.frame == "dialogue":
        loop = DialogueLoop(args.model, args.layer, args.threads, args.prompt, args.dtype, args.device, args.weights)
        first = loop.sm.tok(ela.OPENER, add_special_tokens=False)["input_ids"][0]
        with loop.torch.no_grad():
            start = loop.model.get_input_embeddings()(
                loop.torch.tensor([first], device=loop.device))[0].to(loop.state_dtype)
    else:
        loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, ela.OPENER, 1, args.dtype, args.device,
                            args.weights)
        start = None
    return PairLoop(loop, start=start, record_a=args.save_sequences)


def cells_for(args):
    """[log10T, alpha_A, alpha_B] rows of the requested map."""
    if args.mode == "cells":
        return np.array(json.loads(Path(args.cells_file).read_text())["cells"], dtype=float)
    if args.mode == "pairmap":
        aa = np.linspace(args.alpha_a[0], args.alpha_a[1], int(args.alpha_a[2]))
        ab = np.linspace(args.alpha_b[0], args.alpha_b[1], int(args.alpha_b[2]))
        return np.array([[t, a, b] for t in args.log_temperature for b in ab for a in aa])
    lts = np.linspace(args.log_temperature[0], args.log_temperature[1], int(args.log_temperature[2]))
    al = np.linspace(args.alpha[0], args.alpha[1], int(args.alpha[2]))
    return np.array([[t, a, a] for t in lts for a in al])


def main():
    import explore_llm_phase_map as elpm

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("pairmap", "selfmap", "cells"))
    ap.add_argument("--frame", choices=("dialogue", "cowrite"), required=True)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--prompt", default="open")
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept-a", default="philosophy")
    ap.add_argument("--concept-b", default=None, help="agent B's direction (default: agent A's)")
    ap.add_argument("--log-temperature", type=float, nargs="+",
                    help="pairmap: one map per value; selfmap: LO HI N")
    ap.add_argument("--alpha-a", type=float, nargs=3, metavar=("LO", "HI", "N"))
    ap.add_argument("--alpha-b", type=float, nargs=3, metavar=("LO", "HI", "N"))
    ap.add_argument("--alpha", type=float, nargs=3, metavar=("LO", "HI", "N"), help="selfmap: alpha_A = alpha_B")
    ap.add_argument("--cells-file")
    ap.add_argument("--transient", type=int, default=600, help="exchanges")
    ap.add_argument("--record", type=int, default=1000, help="exchanges")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tangent", action="store_true", help="also the exact tangent exponent per exchange")
    ap.add_argument("--save-sequences", action="store_true",
                    help="save both agents' leading tokens per exchange (sequences.npz) for the transcripts")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="native", choices=("native", "bf16"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.concept_b = args.concept_b or args.concept_a
    cells = cells_for(args)
    pair = build(args)
    vecs = elpm.load_vectors(args.vectors)
    va, vb = np.asarray(vecs[args.concept_a], dtype=float), np.asarray(vecs[args.concept_b], dtype=float)
    if len(va) != pair.E.shape[1]:
        raise SystemExit(f"steering vector dimension {len(va)} does not match the model ({pair.E.shape[1]})")
    results, seq_b, seq_a = [], [], []
    seqs = {"lead": [], "pmax": []} if args.save_sequences else None
    for s in range(0, len(cells), args.chunk):
        c = cells[s:s + args.chunk]
        pair.set_b_cells(np.array([b * vb for b in c[:, 2]]))
        got = hc.run_batch(pair, va, c[:, 0], c[:, 1], args.transient, args.record, args.eps, args.tol,
                           sequences=seqs, tangent=args.tangent)
        for r, (_, _, b) in zip(got, c):
            r["alpha_a"] = r.pop("alpha")
            r["alpha_b"] = float(b)
            r["class"] = hc.classify(r)
        if args.save_sequences:
            seq_a.append(np.array(pair.a_lead[-args.record:])[:, :len(c)])
        results += got
        print(json.dumps({"done": len(results), "of": len(cells)}), flush=True)
    tok = pair.loop.sm.tok
    for r in results:
        r["lead_text"] = [tok.decode([t]) for t in r["lead_tokens"]]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "dialogue.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))
    if args.save_sequences:                    # (record, cells): B's and A's leading token per exchange
        np.savez_compressed(out / "sequences.npz", lead_b=np.concatenate(seqs["lead"], axis=1),
                            pmax_b=np.concatenate(seqs["pmax"], axis=1), lead_a=np.concatenate(seq_a, axis=1))
    counts = {}
    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
