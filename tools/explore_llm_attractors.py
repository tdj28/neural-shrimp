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
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
CACHE = DATA / "models" / "hf" / "hub"   # passed explicitly: a global HF_HOME may point elsewhere
os.environ.setdefault("HF_HUB_OFFLINE", "1")                      # models are local; no network calls
MODELS ={"qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "olmo2-7b": "allenai/OLMo-2-1124-7B-Instruct",
          "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",     # same architecture and size as qwen0.5b, other training
          "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",  # gated: downloaded on pods with the human's token
          "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",                 # hybrid linear/full attention (float64: qwen35_float64)
          "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",           # the same with 256 experts (top 8): MoE routing
          "qwen72b": "Qwen/Qwen2.5-72B-Instruct",              # large dense models: float64 with bfloat16 storage
          "qwen72b-base": "Qwen/Qwen2.5-72B",                   # its pretrained base, before post-training
          "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",  # (weights="bf16", device "auto"); gated
          "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",  # same release and recipe as llama3.1-8b; gated
          # scaling series (run SC): Qwen2.5 3B/14B/32B and OLMo-2 1B/13B/32B
          "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
          "qwen32b": "Qwen/Qwen2.5-32B-Instruct", "olmo2-1b": "allenai/OLMo-2-0425-1B-Instruct",
          "olmo2-13b": "allenai/OLMo-2-1124-13B-Instruct", "olmo2-32b": "allenai/OLMo-2-0325-32B-Instruct",
          "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct", "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
          # mixture-of-experts study (run MOE): OLMoE (64 experts, top 8; float64: olmoe_float64) and dense Qwen3.5
          # controls for Qwen3.5-35B-A3B (4B: about its active size; 9B)
          "olmoe-1b-7b": "allenai/OLMoE-1B-7B-0125-Instruct",
          "qwen3.5-4b": "Qwen/Qwen3.5-4B", "qwen3.5-9b": "Qwen/Qwen3.5-9B"}
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


AUTO_RESERVE_GIB = 20          # free memory kept on each GPU when a large model is spread (device "auto")


def upcast_linear_forward():
    """Let nn.Linear layers whose weights are stored in a lower precision compute in float64 when their input is
    float64: the weight (and bias) are converted for each call. bfloat16 values are exactly float64 values, so the
    result equals that of a float64 copy of the same weights, at a quarter of the memory. Other inputs keep the
    original behaviour. Patched at the class, before loading, so that accelerate's device hooks wrap the patched
    method."""
    import torch
    import torch.nn.functional as F

    if getattr(torch.nn.Linear, "_upcast_patch", False):
        return
    original = torch.nn.Linear.forward

    def forward(self, x):
        if x.dtype == torch.float64 and self.weight.dtype != torch.float64:
            bias = None if self.bias is None else self.bias.to(torch.float64)
            return F.linear(x, self.weight.to(torch.float64), bias)
        return original(self, x)

    torch.nn.Linear.forward = forward
    torch.nn.Linear._upcast_patch = True


def float64_except_decoder_linears(model):
    """Cast every parameter and floating buffer to float64 except the weights and biases of nn.Linear layers inside
    the decoder layers (they stay in their stored bfloat16 and are used through upcast_linear_forward). Embeddings,
    the output layer and the norms become float64. Returns the bytes kept in bfloat16."""
    import torch

    keep = set()
    for layer in model.model.layers:
        for m in layer.modules():
            if isinstance(m, torch.nn.Linear):
                keep.add(id(m.weight))
                if m.bias is not None:
                    keep.add(id(m.bias))
    kept = 0
    for p in model.parameters():
        if id(p) in keep:
            if p.dtype != torch.bfloat16:
                raise SystemExit(f"decoder linear weight stored as {p.dtype}, expected bfloat16")
            kept += p.numel() * 2
        else:
            p.data = p.data.to(torch.float64)
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.to(torch.float64)
    return kept


class Engine:
    def __init__(self, model, threads, dtype="float32", device=None, weights="native"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.name = MODELS[model]
        self.weights = weights
        self.tok = AutoTokenizer.from_pretrained(self.name, cache_dir=CACHE)
        if weights not in ("native", "bf16"):
            raise SystemExit(f"weights must be native or bf16, not {weights!r}")
        if (weights == "bf16" or device == "auto") and (dtype != "float64" or model.startswith("qwen3.5")):
            raise SystemExit("bf16 weight storage and device 'auto' are for float64 dense models only")
        if weights == "bf16" or device == "auto":
            # large dense models: decoder linear weights stay in their stored bfloat16 and compute in float64 (exact,
            # see upcast_linear_forward); device "auto" spreads the layers over all GPUs (accelerate)
            upcast_linear_forward()
            kwargs = {}
            if device == "auto":
                # balanced over the GPUs, leaving AUTO_RESERVE_GIB free on each for what grows after placement: the
                # float64 embedding and output layer, the per-row prefix caches and the float64 weight copies
                gib = 2 ** 30
                total = [torch.cuda.get_device_properties(i).total_memory / gib for i in range(torch.cuda.device_count())]
                budget = {i: f"{int(max(t - AUTO_RESERVE_GIB, 0.5 * t))}GiB" for i, t in enumerate(total)}
                kwargs = {"device_map": "balanced", "max_memory": budget}
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.bfloat16, cache_dir=CACHE,
                                                              **kwargs)
            if device not in (None, "auto"):
                self.model = self.model.to(device)
            if weights == "bf16":
                self.bf16_bytes = float64_except_decoder_linears(self.model)
            else:
                self.model = self.model.to(torch.float64)
        elif dtype == "float64" and model.startswith("olmoe"):
            # OLMoE: load the stored bfloat16 weights, keep the experts bfloat16 (used in float64 slice by slice,
            # exact) and cast the rest to float64 with the router softmax patched (tools/olmoe_float64.py); the
            # normalizations and rotary tables are patched below with the dense models
            import olmoe_float64

            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", cache_dir=CACHE)
            if device is not None:
                self.model = self.model.to(device)
            self.float64_patch = olmoe_float64.prepare(self.model, keep_experts_bf16=True)
        elif dtype == "float64" and model.startswith("qwen3.5"):
            # Qwen3.5 (hybrid attention, optionally mixture of experts): load the stored bfloat16 weights, then cast
            # and patch the float32 code paths (tools/qwen35_float64.py); experts stay bfloat16 and are used in
            # float64 slice by slice (exact), so a 35B MoE fits one 141 GB GPU
            import qwen35_float64

            # host first, then the device (device_map would need accelerate); the 35B MoE needs about 72 GB of
            # host RAM for the bfloat16 checkpoint, so its pods ask for at least 128 GB
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", cache_dir=CACHE)
            if device is not None:
                self.model = self.model.to(device)
            self.float64_patch = qwen35_float64.prepare(self.model, keep_experts_bf16="a3b" in model)
            repo = Path(CACHE) / ("models--" + self.name.replace("/", "--"))
            ckpt = repo / "snapshots" / (repo / "refs" / "main").read_text().strip()      # the pinned revision
            self.float64_patch["restored_exact"] = qwen35_float64.restore_exact_tensors(self.model, ckpt)
        elif dtype == "float64" and device is not None and str(device).startswith("cuda"):
            # large models: load in the checkpoint's own dtype, move to the GPU, then cast there (bf16 -> float64 is
            # exact, so the weights equal a direct float64 load; host memory needs only the stored size)
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype="auto", cache_dir=CACHE)
            self.model = self.model.to(device).to(torch.float64)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=getattr(torch, dtype), cache_dir=CACHE)
        self.model.eval()
        self.model.requires_grad_(False)                # inference only; forward-mode JVPs need no parameter graph
        if dtype == "float64" and not model.startswith("qwen3.5"):
            keep_float64_norms(self.model)


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


def keep_float64_norms(model):
    """Make a float64 model compute in float64 where Hugging Face code silently drops to float32.

    RMSNorm layers compute in float32 whatever the input dtype (an upcast meant for bf16): their classes are
    patched to compute in the input dtype. Rotary embeddings build their inverse frequencies, cos and sin tables in
    float32 (and CPU and GPU round those tables differently): for float64 inputs the patched classes rebuild them in
    float64 from the configuration, refusing rope types other than the default and frequencies that do not match
    the model's own. float32 inputs keep the original code paths. Attention implementations other than sdpa are
    refused (their kernels may cast to float32)."""
    import torch

    if getattr(model.config, "model_type", "").startswith("qwen3_5"):
        raise SystemExit("Qwen3.5 norms are zero-centred and its linear attention casts to float32: use qwen35_float64")

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    for cls in {type(m) for m in model.modules() if type(m).__name__.endswith("RMSNorm")}:
        cls.forward = forward

    def rotary_forward(self, x, position_ids):
        if x.dtype != torch.float64 or getattr(self, "_inv_freq64", None) is None:
            return self._float32_forward(x, position_ids)
        freqs = position_ids[..., None].to(torch.float64) * self._inv_freq64.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    for m in model.modules():
        name = type(m).__name__
        if not (name.endswith("RotaryEmbedding") and hasattr(m, "inv_freq")):
            continue
        rope_type = getattr(m, "rope_type", "default")
        if rope_type not in ("default", "llama3"):
            raise SystemExit(f"{name}: rope type {rope_type!r} is not rebuilt in float64")
        cfg = m.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        dim = int(head_dim * getattr(cfg, "partial_rotary_factor", 1.0))
        params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        base = getattr(cfg, "rope_theta", None) or params["rope_theta"]               # transformers 4 or 5
        inv64 = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        if rope_type == "llama3":         # Llama 3.1 frequency scaling, as transformers' _compute_llama3_parameters
            factor, low, high = params["factor"], params["low_freq_factor"], params["high_freq_factor"]
            old_len = params["original_max_position_embeddings"]
            wavelen = 2 * math.pi / inv64
            scaled = torch.where(wavelen > old_len / low, inv64 / factor, inv64)
            smooth = (old_len / wavelen - low) / (high - low)
            smoothed = (1 - smooth) * scaled / factor + smooth * scaled
            medium = ~(wavelen < old_len / high) & ~(wavelen > old_len / low)
            inv64 = torch.where(medium, smoothed, scaled)
        own = m.inv_freq.detach().to("cpu", torch.float64)
        if inv64.shape != own.shape or not torch.allclose(inv64, own, rtol=1e-6, atol=0):
            raise SystemExit(f"{name}: rebuilt inverse frequencies do not match the model's")
        m._inv_freq64 = inv64
        cls = type(m)
        if "_float32_forward" not in cls.__dict__:
            cls._float32_forward = cls.forward
            cls.forward = rotary_forward
    # accelerate's device hooks (layers spread over GPUs, CPU offload) keep the forward that a module had when the
    # model was dispatched, before these class patches: a hooked norm or rotary module would still run the float32
    # original (the final norm and the rotary tables of every two-GPU run up to 14 October 2026, measured as a 1e-8
    # wander at fixed points). Rebind each hooked module to its patched class forward, and refuse to continue if any
    # still wraps an original.
    import types

    patched = {cls for cls in {type(m) for m in model.modules()}
               if cls.__name__.endswith("RMSNorm") or "_float32_forward" in cls.__dict__}
    for m in model.modules():
        if type(m) in patched and hasattr(m, "_old_forward"):
            m._old_forward = types.MethodType(type(m).forward, m)
    stale = [n for n, m in model.named_modules() if type(m) in patched and hasattr(m, "_old_forward")
             and getattr(m._old_forward, "__func__", None) is not type(m).forward]
    if stale:
        raise SystemExit(f"float64 patches bypassed by device hooks: {stale[:5]}")
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in ("sdpa", None):
        raise SystemExit(f"attention implementation {impl!r} may cast to float32; use sdpa for float64 runs")


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


def cycle_contexts(rec, system):
    """(messages, token_ids, text) for each turn of a greedy run's final cycle, rebuilt from the transcript."""
    turns = rec["turns"]
    period = rec["cycle"]["period_turns"]
    transcript = [rec["opener"]] + [t["text"] for t in turns]
    window = rec["args"]["window"]
    out = []
    for k in range(len(turns) - period + 1, len(turns) + 1):       # turn numbers of the last period
        agent = (k - 1) % 2
        out.append((window_messages(system, transcript[:k], agent, window), turns[k - 1]["token_ids"],
                    turns[k - 1]["text"]))
    return out


def margins(args):
    """Logit margins along an attractor cycle and the exact probability of reproducing it at temperature T."""
    engine = Engine(args.model, args.threads)
    torch = engine.torch
    rec = json.loads((DATA / "explore" / args.run / "dialogue.json").read_text())
    if not rec["cycle"] or rec["args"]["temperature"] != 0:
        raise SystemExit("needs a greedy run that ended on an exact cycle")
    contexts = cycle_contexts(rec, rec["system"])
    temps = np.array(args.temperatures)
    rows, log_cycle = [], np.zeros(len(temps))
    for messages, ids, text in contexts:
        enc = engine.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        enc = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        full = torch.cat([enc, torch.tensor([ids])], dim=1)
        with torch.no_grad():
            z = engine.model(full).logits[0, enc.shape[1] - 1:enc.shape[1] - 1 + len(ids)].double()
        top2 = torch.topk(z, 2, dim=-1).values
        y = torch.tensor(ids)
        z_y = z[torch.arange(len(ids)), y]
        is_greedy = bool(torch.all(z_y >= top2[:, 0] - 1e-9))
        margin = (top2[:, 0] - top2[:, 1]).numpy()
        logp = np.array([float(torch.log_softmax(z / t, dim=-1)[torch.arange(len(ids)), y].sum()) for t in temps])
        log_cycle += logp
        rows.append({"text": text[:120], "tokens": len(ids), "greedy_path": is_greedy,
                     "min_margin": float(margin.min()), "margins_sorted": np.sort(margin)[:5].tolist(),
                     "stay_logprob": logp.tolist()})
    p_cycle = np.exp(log_cycle)
    period = rec["cycle"]["period_turns"]
    min_margin = min(r["min_margin"] for r in rows)
    out = {"args": vars(args), "run": args.run, "period": period, "temperatures": temps.tolist(),
           "cycle_stay_probability": p_cycle.tolist(),
           "predicted_turns_before_first_deviation": (period / np.maximum(1 - p_cycle, 1e-300)).tolist(),
           "min_margin_over_cycle": min_margin,
           "arrhenius_check": (-np.log(np.maximum(1 - p_cycle, 1e-300)) * temps).tolist(), "turns": rows}
    print(json.dumps({"run": args.run, "period": period, "min_margin": round(min_margin, 4),
                      "stay_probability_by_T": {f"{t:g}": float(f"{p:.4g}") for t, p in zip(temps, p_cycle)},
                      "T_times_minus_log_escape": [float(f"{v:.3g}") for v in out["arrhenius_check"]]}))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "margins.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def escape(args):
    """Sampled runs started inside a greedy attractor: turns to first deviation and occupancy of the cycle."""
    engine = Engine(args.model, args.threads)
    rec = json.loads((DATA / "explore" / args.run / "dialogue.json").read_text())
    system, window = rec["system"], rec["args"]["window"]
    turns = rec["turns"]
    period = rec["cycle"]["period_turns"]
    cycle_texts = set(t["text"] for t in turns[-period:])
    prefix = [rec["opener"]] + [t["text"] for t in turns]
    results = []
    for rep in range(args.replicas):
        transcript = list(prefix)
        first_dev, inside = None, 0
        for k in range(1, args.turns + 1):
            turn_no = len(transcript)
            agent = (turn_no - 1) % 2
            text, _, _, _ = engine.generate(window_messages(system, transcript, agent, window), args.temperature,
                                            rec["args"]["max_new"], args.seed * 1000003 + rep * 1009 + k)
            transcript.append(text)
            on_cycle = text in cycle_texts
            inside += on_cycle
            if not on_cycle and first_dev is None:
                first_dev = k
        results.append({"replica": rep, "first_deviation_turn": first_dev, "fraction_on_cycle": inside / args.turns,
                        "last_text": transcript[-1][:120]})
        print(json.dumps(results[-1]))
    devs = [r["first_deviation_turn"] for r in results]
    observed = [d for d in devs if d is not None]
    out = {"args": vars(args), "run": args.run, "period": period, "results": results,
           "censored_runs": sum(d is None for d in devs),
           "mean_first_deviation_uncensored": float(np.mean(observed)) if observed else None,
           "mean_fraction_on_cycle": float(np.mean([r["fraction_on_cycle"] for r in results]))}
    print(json.dumps({k: v for k, v in out.items() if k not in ("args", "results")}))
    dest = DATA / "explore" / args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "escape.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


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
    mg = sub.add_parser("margins")
    mg.add_argument("--model", choices=sorted(MODELS), default="qwen0.5b")
    mg.add_argument("--threads", type=int, default=4)
    mg.add_argument("--run", required=True, help="explore-relative folder of a greedy run that ended on a cycle")
    mg.add_argument("--temperatures", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    mg.add_argument("--out", required=True)
    es = sub.add_parser("escape")
    es.add_argument("--model", choices=sorted(MODELS), default="qwen0.5b")
    es.add_argument("--threads", type=int, default=4)
    es.add_argument("--run", required=True)
    es.add_argument("--temperature", type=float, required=True)
    es.add_argument("--replicas", type=int, default=8)
    es.add_argument("--turns", type=int, default=30)
    es.add_argument("--seed", type=int, default=0)
    es.add_argument("--out", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--folder", required=True, help="explore-relative folder of runs")
    sm.add_argument("--tail", type=int, default=6)
    sm.add_argument("--entropy-level", type=float, default=0.3)
    sm.add_argument("--cos-level", type=float, default=0.95)
    sm.add_argument("--absorbed-length", type=int, default=5, help="identical final turns that count as absorbed")
    args = parser.parse_args()
    {"dialogue": dialogue, "paraphrase": paraphrase, "summary": summary, "margins": margins,
     "escape": escape}[args.command](args)


if __name__ == "__main__":
    main()
