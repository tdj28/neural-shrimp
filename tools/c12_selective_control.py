"""C12-1 (exploratory, program PI): can one small perturbation of the soft-thinking loop steer a correct answer to a
specified wrong answer, more effectively than compute-matched random perturbations, and with more leverage early
than just before the answer? Run card "C12-1, selective control" in docs/research-log.md (consult 12, direction 1).

Task. Two-digit additions "What is a + b?" (a, b in 10..99), wrapped by tools/pi_harness.py's INSTRUCTION and chat
template (prompt_ids; every prompt has the same length, so batches need no padding), scored exactly: the integer
after the answer cue (pi_harness.extract_answer) against a + b. 16 development and 64 held-out problems on disjoint
unordered operand pairs, drawn by numpy default_rng(PROBLEM_SEED); `problems` writes the list and its SHA-256. The
specified wrong answer (target) of a problem with sum c is c + 10 if that keeps the digit count, else c - 10.

Pipeline (pi_harness soft mode; Qwen2.5-0.5B-Instruct in float32 on the CPU). The prompt, then S soft steps, each of
which feeds the concept token sum_i q_i e_i (q = softmax(z / T) restricted to its top-15 tokens and renormalized),
then the cue "\\n\\nThe answer is" and --answer-tokens greedy answer tokens (pi_harness's 12 by default; the
C12-1 runs use 6, which fit every answer up to four digits with its period: with 12, the model sometimes began to
repeat the cue, and pi_harness.extract_answer, which reads the number after the last "answer is", then found none). The evaluation path (no gradients, KV cache) repeats
pi_harness's operations (log_softmax, top-k of the probabilities, pi_harness.concept_token); every reported answer
comes from it.

Perturbation. One vector delta with |delta| <= rho |e|, added to one concept token: early (the first, e_0) or late
(the last before the cue, e_{S-1}); |e| is the norm of that token in the unperturbed run, so the late perturbation
starts from the unperturbed run's cache. S = 1 is the immediate-readout diagnostic (one concept token, then the cue;
early and late coincide). Arms, per problem, temperature, horizon, injection and radius:
  none     the unperturbed run (baseline; it defines eligibility: baseline-correct problem-temperature pairs);
  random1  one random direction at the full radius (the first random-search direction);
  search   random search: K random directions at the full radius; the one with the largest objective is kept;
  pga      projected gradient ascent from delta = 0: --pga-steps normalized gradient steps of --pga-schedule times
           the radius, each followed by projection onto the ball; the iterate with the largest objective is kept;
  zero     delta = 0 rows inside the search and pga evaluation batches (negative control: must give the baseline).
The objective is the teacher-forced log-probability of the target's answer tokens (" ", digits, ".") after the cue.
The gradient path recomputes the rollout with autograd through the KV cache (the prompt cache and, for late
injection, the unperturbed prefix are constants) and holds each step's top-15 support at its forward value (the
derivative of the map almost everywhere). `derivcheck` checks it against central differences (float32 and float64)
and against a full recomputation without a cache.
Compute matching: `bench` measures c_grad, the cost of one gradient pass in objective evaluations (forwards of the
evaluation path, teacher-forced target, no greedy decoding) at the batch sizes of a run; random search gets
K = round(pga_steps * c_grad + 1) evaluations (the +1 matches the optimizer's evaluation of its last iterate).
Every candidate's greedy answer is also decoded (descriptive readouts: random answer changes and the any-of-K oracle;
not used for selection). Random directions are seeded by (problem, temperature, horizon, injection, candidate) and
shared across radii, so the radii are paired.

Subcommands (CPU; development problems only, except `run --split test`):
  problems    write the problem list (dev and test) and its hash;
  calibrate   baseline accuracy over a temperature grid at S = 0, 1, 8, 32, and finite-time expansion measured by twins
              (the first concept token displaced by a relative eps along random directions, in the base row's batch;
              relative concept-token distance per step, net rate log(d_31 / d_0) / 31, twin answer flips), with a
              zero-offset twin per problem as the float32 floor;
  layouts     float32 and bfloat16 batch-layout differences (all problems in one batch against batches of 8 and 1):
              relative concept-token differences per step and answer flips;
  bench       measured cost of a gradient pass against an objective evaluation (sets K);
  derivcheck  directional derivatives against central differences, the cached gradient against a recomputation
              without a cache, and (with --compare) float32 against float64 gradients of the composed rollout;
  run         the attack matrix on one split, with the forced-target scorer control on every baseline readout;
  analyze     summary JSON: per-arm rates with denominators and intervals, the primary Delta, early against late,
              unconditional rates, answer changes, controls and measured compute.

Outputs under --out (default /data2/neural-shrimp-data/explore/pi/c12-1/<name>): run.json (arguments, versions,
commit, file hashes, problem hash, timings), records.jsonl (one line per baseline readout and per arm row),
calibrate.json, layouts.json, bench.json, derivcheck-<dtype>.json.

Usage (prefix CUDA_VISIBLE_DEVICES= ; uv run --no-sync python):
  tools/c12_selective_control.py problems --json docs/pi/c12-1-problems.json
  tools/c12_selective_control.py calibrate --temperatures 0.3 0.6 1.0 1.5 2.0 3.0 --out .../calibrate
  tools/c12_selective_control.py layouts --temperatures T1 T2 --out .../layouts
  tools/c12_selective_control.py bench --temperature T1 --out .../bench
  tools/c12_selective_control.py derivcheck --dtype float64 --temperatures T1 T2 --out .../derivcheck
  tools/c12_selective_control.py run --split dev --temperatures T1 T2 --radii 0.003 0.01 0.03 0.1 0.3 \\
      --horizons 32 --control-radii 1.0 --K K --out .../dev-staircase
  tools/c12_selective_control.py run --split test --temperatures T1 T2 --radii R1 R2 --horizons 1 8 32 \\
      --control-radii 1.0 --K K --out .../test
  tools/c12_selective_control.py analyze RUN --primary-radius R1 --json docs/pi/c12-1-summary.json
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pi_harness as ph  # noqa: E402

PROBLEM_SEED = 12001
DIRECTION_SEED = 12002
TWIN_SEED = 12003
BOOT_SEED = 12004
N_DEV, N_TEST = 16, 64
TOP_N = 15
ANSWER_TOKENS = 12
FORCE_BIAS = 1.0e4
DATA = Path("/data2/neural-shrimp-data/explore/pi/c12-1")
INJ = {"early": 0, "late": 1}


# ----------------------------------------------------------------------------------------------- pure functions

def target_answer(correct):
    """The specified wrong answer: correct + 10 if that keeps the digit count, else correct - 10."""
    up = correct + 10
    return up if len(str(up)) == len(str(correct)) else correct - 10


def make_problems(seed=PROBLEM_SEED, n_dev=N_DEV, n_test=N_TEST):
    """Two-digit additions on distinct unordered operand pairs: the first n_dev are development problems, the next
    n_test held out (so the two sets share no pair, in either order)."""
    rng = np.random.default_rng(seed)
    seen, pairs = set(), []
    while len(pairs) < n_dev + n_test:
        a, b = (int(x) for x in rng.integers(10, 100, size=2))
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    out = []
    for i, (a, b) in enumerate(pairs):
        split, j = ("dev", i) if i < n_dev else ("test", i - n_dev)
        out.append({"uid": i, "id": f"{split}-{j:02d}", "split": split, "a": a, "b": b,
                    "question": f"What is {a} + {b}?", "gold": str(a + b), "target": str(target_answer(a + b))})
    return out


def problems_hash(problems):
    blob = json.dumps(problems, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def score(text, gold, target):
    """Exact scoring of an answer text (pi_harness.extract_answer); a missing number is invalid, never a hit."""
    ans = ph.extract_answer(text)
    return {"answer": ans, "correct": ans is not None and ans == gold,
            "target_hit": ans is not None and ans == target, "invalid": ans is None}


def answer_tokens_ok(ids, number, space_id, period_id, digit_ids):
    """The target's teacher-forced tokens must be ' ', one token per digit, '.' (Qwen2 splits digits)."""
    return (len(ids) == len(number) + 2 and ids[0] == space_id and ids[-1] == period_id
            and all(ids[1 + i] == digit_ids[ch] for i, ch in enumerate(number)))


def unit_direction(d, *key):
    """A unit vector in R^d from numpy default_rng([DIRECTION_SEED, *key]) (integer key)."""
    v = np.random.default_rng([DIRECTION_SEED, *[int(k) for k in key]]).standard_normal(d)
    return v / np.linalg.norm(v)


def temp_key(T):
    return int(round(T * 1000))


def project_ball(torch, delta, radius):
    """Project each row of delta onto the ball of its radius."""
    n = delta.norm(dim=-1, keepdim=True)
    return delta * torch.clamp(radius[:, None] / n.clamp_min(1e-300), max=1.0)


def pga_step(torch, delta, grad, radius, alpha):
    """One normalized ascent step of size alpha * radius, then projection; rows with a zero gradient do not move."""
    gn = grad.norm(dim=-1, keepdim=True)
    step = torch.where(gn > 0, grad / gn.clamp_min(1e-300), torch.zeros_like(grad))
    return project_ball(torch, delta + alpha * radius[:, None] * step, radius)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    lo, hi = ph.wilson(k, n, z)
    return [lo, hi]


def bootstrap_mean_ci(x, n_boot=10000, seed=BOOT_SEED, level=0.95):
    """Percentile interval of the mean of x (one value per problem) under resampling of problems."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def per_problem_mean(pairs):
    """pairs: iterable of (uid, value); returns the per-problem means (problems weighted equally) in uid order."""
    by = defaultdict(list)
    for uid, v in pairs:
        by[uid].append(float(v))
    return [float(np.mean(by[u])) for u in sorted(by)]


def decision(delta, ci):
    """Run-card rule: an advantage of about 15 points or more merits an extension; an interval that excludes 10
    points retires the method."""
    if delta is None:
        return "no eligible pairs"
    if ci[1] is not None and ci[1] < 0.10:
        return "retire: interval excludes a 10-point advantage"
    if delta >= 0.15:
        return "extend: advantage of 15 points or more"
    return "neither: advantage below 15 points, interval does not exclude 10 points"


# ----------------------------------------------------------------------------------------------- provenance

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(args, extra=None):
    import torch
    import transformers

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:                            # noqa: BLE001
            return f"error: {exc}"
    info = {"args": vars(args), "argv": sys.argv, "commit": git("rev-parse", "HEAD").strip(),
            "dirty_files": [ln for ln in git("status", "--porcelain").splitlines()
                            if "c12_selective_control" in ln or "pi_harness" in ln],
            "tool_sha256": sha256_file(__file__), "pi_harness_sha256": sha256_file(ph.__file__),
            "instruction": ph.INSTRUCTION, "cue": ph.CUE, "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "threads": torch.get_num_threads(), "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    info.update(extra or {})
    return info


def load_problems(path=None):
    probs = make_problems() if path is None else json.loads(Path(path).read_text())["problems"]
    if path is not None and problems_hash(probs) != problems_hash(make_problems()):
        raise SystemExit(f"{path}: problem list differs from make_problems()")
    return probs


# ----------------------------------------------------------------------------------------------- model engine

class Engine:
    """Qwen2.5 soft-thinking rollouts on the CPU: an evaluation path (no gradients, pi_harness operations) and a
    gradient path (autograd through the KV cache)."""

    def __init__(self, model_key="qwen0.5b", dtype="float32", threads=6, top_n=TOP_N, answer_tokens=ANSWER_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        torch.set_num_threads(threads)
        self.torch, self.DynamicCache = torch, DynamicCache
        self.path = ph.snapshot(ph.MODELS[model_key])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=getattr(torch, dtype))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if dtype == "float64":                      # norms and rotary tables in float64 (AGENTS.md precision gate)
            from explore_llm_attractors import keep_float64_norms
            keep_float64_norms(self.model)
        self.E = self.model.get_input_embeddings().weight
        self.W = self.model.lm_head.weight
        self.dtype = self.E.dtype
        self.cdtype = torch.float64 if dtype == "float64" else torch.float32   # logits and concept-token arithmetic
        self.top_n, self.answer_tokens = top_n, answer_tokens
        self.cue = self.tok(ph.CUE, add_special_tokens=False)["input_ids"]
        self.d = self.E.shape[1]
        self.space_id = self.tok(" ", add_special_tokens=False)["input_ids"]
        self.period_id = self.tok(".", add_special_tokens=False)["input_ids"]
        self.digit_ids = {str(i): self.tok(str(i), add_special_tokens=False)["input_ids"] for i in range(10)}
        if len(self.space_id) != 1 or len(self.period_id) != 1 or any(len(v) != 1 for v in self.digit_ids.values()):
            raise SystemExit("space, period or digit is not a single token")
        self.space_id, self.period_id = self.space_id[0], self.period_id[0]
        self.digit_ids = {k: v[0] for k, v in self.digit_ids.items()}

    def target_ids(self, number):
        ids = self.tok(f" {number}.", add_special_tokens=False)["input_ids"]
        if not answer_tokens_ok(ids, number, self.space_id, self.period_id, self.digit_ids):
            raise SystemExit(f"unexpected tokenization of the answer {number!r}: {ids}")
        return ids

    # -- states and caches
    def prompt_state(self, problems):
        torch = self.torch
        ids = [ph.prompt_ids(self.tok, p["question"]) for p in problems]
        if len({len(i) for i in ids}) != 1:
            raise SystemExit("prompts differ in length: batches would need padding")
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor(ids), use_cache=True, logits_to_keep=1)
        kv = [(layer.keys, layer.values) for layer in o.past_key_values.layers]
        return {"kv": kv, "z": o.logits[:, -1, :].to(self.cdtype), "ids": ids}

    @staticmethod
    def select(state, rows):
        return {"kv": [(k[rows], v[rows]) for k, v in state["kv"]], "z": state["z"][rows]}

    def cache(self, kv):
        return self.DynamicCache(ddp_cache_data=kv)

    # -- evaluation path (pi_harness operations)
    def concept(self, z, T):
        torch = self.torch
        logp = torch.log_softmax(z / T, dim=-1)
        p = logp.exp()
        w, idx = torch.topk(p, self.top_n, dim=-1)
        q = torch.zeros_like(p).scatter_(-1, idx, w)
        q = q / q.sum(-1, keepdim=True)
        if self.cdtype == torch.float64:            # float64 check runs: no float32 step anywhere
            w2, idx2 = torch.topk(q, self.top_n, dim=-1)
            return (w2[..., None] * self.E[idx2]).sum(1)
        return ph.concept_token(torch, q, self.E, self.top_n).to(self.dtype)

    def rollout(self, state, T, n_steps, delta=None, readouts=(), snapshots=(), targets=None, force=False,
                greedy=True, keep_emb=False):
        """n_steps soft steps from state (cache before this segment's first soft token, and the logits it comes
        from); delta (B, d) is added to the first token. After t soft tokens with t in readouts (0: before any),
        the cue and branches: teacher-forced target log-probability (targets), greedy answer, forced-target answer
        (force). snapshots: steps t whose state (cache with t tokens, logits) is kept."""
        torch = self.torch
        out = {"readouts": {}, "snapshots": {}, "emb_norm": [], "emb": []}
        with torch.no_grad():
            c = self.cache(state["kv"])
            z = state["z"]
            if 0 in readouts:
                out["readouts"][0] = self.readout(c, targets, force, greedy, restore=n_steps > 0)
            for t in range(n_steps):
                if t in snapshots:
                    out["snapshots"][t] = {"kv": [(layer.keys, layer.values) for layer in c.layers], "z": z}
                e = self.concept(z, T)
                if t == 0 and delta is not None:
                    base = e.to(self.cdtype)
                    e = (base + delta.to(self.cdtype)).to(self.dtype)
                    out["realized_rel"] = ((e.to(self.cdtype) - base).norm(dim=-1) / base.norm(dim=-1)).tolist()
                out["emb_norm"].append(e.to(self.cdtype).norm(dim=-1))
                if keep_emb:
                    out["emb"].append(e.to(self.cdtype))
                o = self.model(inputs_embeds=e[:, None, :], past_key_values=c, use_cache=True)
                z = o.logits[:, -1, :].to(self.cdtype)
                if t + 1 in readouts:
                    out["readouts"][t + 1] = self.readout(c, targets, force, greedy, restore=t + 1 < n_steps)
        return out

    def readout(self, c, targets=None, force=False, greedy=True, restore=True):
        torch = self.torch
        B = c.layers[0].keys.shape[0]
        L0 = c.get_seq_length()
        o = self.model(input_ids=torch.tensor([self.cue] * B), past_key_values=c, use_cache=True, logits_to_keep=1)
        zc = o.logits[:, -1, :].to(self.cdtype)
        res = {}
        if targets is not None:
            res["J"] = self.teacher_forced(c, zc, targets).tolist()
        if greedy:
            res["text"] = self.greedy(c, zc)
        if force:
            res["forced_text"] = self.greedy(c, zc, targets)
        if restore:
            c.crop(-(c.get_seq_length() - L0))
        return res

    def teacher_forced(self, c, zc, targets):
        torch = self.torch
        L1 = c.get_seq_length()
        Lm = max(map(len, targets))
        tg = torch.tensor([t + [t[-1]] * (Lm - len(t)) for t in targets])
        mask = torch.tensor([[1.0] * len(t) + [0.0] * (Lm - len(t)) for t in targets], dtype=self.cdtype)
        lp = [torch.log_softmax(zc, dim=-1).gather(-1, tg[:, :1])]
        if Lm > 1:
            o = self.model(input_ids=tg[:, :-1], past_key_values=c, use_cache=True)
            lp.append(torch.log_softmax(o.logits.to(self.cdtype), dim=-1).gather(-1, tg[:, 1:, None])[..., 0])
        c.crop(-(c.get_seq_length() - L1))
        return (torch.cat(lp, 1) * mask).sum(1)

    def greedy(self, c, zc, force_targets=None):
        """Greedy answer tokens after the cue (as pi_harness; the last token is not fed back, which changes no
        output). force_targets: add FORCE_BIAS to the target's j-th token at answer step j (scorer control)."""
        torch = self.torch
        L1 = c.get_seq_length()
        z, toks = zc, []
        for j in range(self.answer_tokens):
            if force_targets is not None:
                z = z.clone()
                for r, tg in enumerate(force_targets):
                    if j < len(tg):
                        z[r, tg[j]] += FORCE_BIAS
            nxt = z.argmax(-1)
            toks.append(nxt.tolist())
            if j + 1 < self.answer_tokens:
                o = self.model(input_ids=nxt[:, None], past_key_values=c, use_cache=True)
                z = o.logits[:, -1, :].to(self.cdtype)
        c.crop(-(c.get_seq_length() - L1))
        B = len(toks[0])
        return [ph.CUE + self.tok.decode([t[r] for t in toks], skip_special_tokens=True) for r in range(B)]

    # -- gradient path
    def concept_diff(self, h, T):
        """Concept token from the last hidden state, differentiable; the top-n support is that of the forward."""
        torch = self.torch
        with torch.no_grad():
            idx = torch.topk(self.model.lm_head(h).to(self.cdtype), self.top_n, dim=-1).indices
        zs = torch.einsum("bd,bnd->bn", h, self.W[idx]).to(self.cdtype)
        q = torch.softmax(zs / T, dim=-1)
        return (q[..., None] * self.E[idx].to(self.cdtype)).sum(1)

    def objective(self, state, T, n_steps, delta, targets, grad=True):
        """J(delta) = log P(target tokens | rollout with delta on its first soft token, cue), and dJ/d delta."""
        torch = self.torch
        cd = self.cdtype
        with torch.enable_grad() if grad else torch.no_grad():
            delta = delta.detach().to(cd).clone().requires_grad_(grad)
            c = self.cache(state["kv"])
            with torch.no_grad():
                e0 = self.concept(state["z"], T).to(cd)
            e, h = e0 + delta, None
            for t in range(n_steps):
                if t > 0:
                    e = self.concept_diff(h, T)
                h = self.model.model(inputs_embeds=e[:, None, :].to(self.dtype), past_key_values=c,
                                     use_cache=True).last_hidden_state[:, -1]
            B = e.shape[0]
            Lm = max(map(len, targets))
            tg = torch.tensor([t + [t[-1]] * (Lm - len(t)) for t in targets])
            mask = torch.tensor([[1.0] * len(t) + [0.0] * (Lm - len(t)) for t in targets], dtype=cd)
            ids = torch.cat([torch.tensor([self.cue] * B), tg[:, :-1]], 1)
            hs = self.model.model(input_ids=ids, past_key_values=c, use_cache=True).last_hidden_state
            logits = self.model.lm_head(hs[:, len(self.cue) - 1:]).to(cd)
            J = (torch.log_softmax(logits, dim=-1).gather(-1, tg[..., None])[..., 0] * mask).sum(1)
            if not grad:
                return J.detach(), None
            J.sum().backward()
            return J.detach(), delta.grad.detach()

    def objective_nocache(self, prompt_ids, z0, T, n_steps, delta, targets):
        """The same objective, recomputing the whole sequence from embeddings at every step (no cache): the check
        of the cached gradient."""
        torch = self.torch
        cd = self.cdtype
        with torch.enable_grad():
            delta = delta.detach().to(cd).clone().requires_grad_(True)
            seq = self.E[torch.tensor(prompt_ids)].to(cd)                    # (B, L, d)
            with torch.no_grad():
                e0 = self.concept(z0, T).to(cd)
            e = e0 + delta
            for t in range(n_steps):
                if t > 0:
                    e = self.concept_diff(h, T)
                seq = torch.cat([seq, e[:, None, :]], 1)
                h = self.model.model(inputs_embeds=seq.to(self.dtype), use_cache=False).last_hidden_state[:, -1]
            B = e.shape[0]
            Lm = max(map(len, targets))
            tg = torch.tensor([t + [t[-1]] * (Lm - len(t)) for t in targets])
            mask = torch.tensor([[1.0] * len(t) + [0.0] * (Lm - len(t)) for t in targets], dtype=cd)
            ids = torch.cat([torch.tensor([self.cue] * B), tg[:, :-1]], 1)
            seq = torch.cat([seq, self.E[ids].to(cd)], 1)
            hs = self.model.model(inputs_embeds=seq.to(self.dtype), use_cache=False).last_hidden_state
            logits = self.model.lm_head(hs[:, -Lm:]).to(cd)          # last cue position and target[:-1]
            J = (torch.log_softmax(logits, dim=-1).gather(-1, tg[..., None])[..., 0] * mask).sum(1)
            J.sum().backward()
            return J.detach(), delta.grad.detach()

    def pga(self, state, T, n_steps, radius, targets, steps, schedule):
        """Projected gradient ascent from 0; returns the iterate (1..steps) with the largest objective."""
        torch = self.torch
        B = radius.shape[0]
        delta = torch.zeros(B, self.d, dtype=self.cdtype)
        Js, gnorms, iterates = [], [], []
        for k in range(steps):
            J, g = self.objective(state, T, n_steps, delta, targets, grad=True)
            Js.append(J)
            if k > 0:
                iterates.append(delta.clone())
            gnorms.append(g.norm(dim=-1))
            delta = pga_step(torch, delta, g, radius, schedule[k])
        J_last, _ = self.objective(state, T, n_steps, delta, targets, grad=False)
        iterates.append(delta.clone())
        J_it = torch.stack(Js[1:] + [J_last], 1)
        best = J_it.argmax(1)
        chosen = torch.stack(iterates, 1)[torch.arange(B), best]
        return {"delta": chosen, "J_iterates": J_it, "J0": Js[0], "grad_norms": torch.stack(gnorms, 1), "best": best}


# ----------------------------------------------------------------------------------------------- subcommands

def cmd_problems(args):
    probs = make_problems()
    dev = {(min(p["a"], p["b"]), max(p["a"], p["b"])) for p in probs if p["split"] == "dev"}
    test = {(min(p["a"], p["b"]), max(p["a"], p["b"])) for p in probs if p["split"] == "test"}
    if dev & test or len(dev) != N_DEV or len(test) != N_TEST:
        raise SystemExit("development and held-out operand pairs are not disjoint")
    blob = {"seed": PROBLEM_SEED, "n_dev": N_DEV, "n_test": N_TEST, "sha256": problems_hash(probs),
            "target_rule": "correct + 10 if that keeps the digit count, else correct - 10", "problems": probs}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(blob, indent=1) + "\n")
    print(json.dumps({"sha256": blob["sha256"], "n": len(probs)}))


def out_dir(args, name, resume=False):
    """A new attempt directory (raw evidence is never overwritten; `run --resume` continues its own directory)."""
    out = Path(args.out) if args.out else DATA / name
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"{out} exists and is not empty: a new attempt needs a new directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_problems(args, split):
    probs = [p for p in load_problems(args.problems) if p["split"] == split]
    return probs[:args.limit] if args.limit else probs


def cmd_calibrate(args):
    """Development problems: baseline accuracy at S = 0, 1, 8, 32 and twin expansion, per temperature."""
    torch_threads(args)
    eng = Engine(args.model, "float32", args.threads, answer_tokens=args.answer_tokens)
    torch = eng.torch
    probs = split_problems(args, "dev")
    out = out_dir(args, "calibrate")
    info = provenance(args, {"problems_sha256": problems_hash(load_problems(args.problems)), "model_path": str(eng.path)})
    st = eng.prompt_state(probs)
    targets = [eng.target_ids(p["target"]) for p in probs]
    S = max(args.horizons)
    levels = args.eps
    per_level = args.twins
    rows_per = 2 + len(levels) * per_level                      # base, zero-offset twin, twins
    rows = torch.tensor([i for i in range(len(probs)) for _ in range(rows_per)])
    summary = {"temperatures": {}}
    recf = open(out / "records.jsonl", "w")
    for T in args.temperatures:
        t0 = time.time()
        e0 = eng.concept(st["z"], T).to(eng.cdtype)
        D = torch.zeros(len(rows), eng.d, dtype=eng.cdtype)
        kinds = []
        for i in range(len(probs)):
            kinds += [("base", 0.0, -1), ("zero", 0.0, -1)]
            for li, eps in enumerate(levels):
                for k in range(per_level):
                    r = i * rows_per + 2 + li * per_level + k
                    u = np.random.default_rng([TWIN_SEED, probs[i]["uid"], temp_key(T), li, k]).standard_normal(eng.d)
                    D[r] = torch.tensor(eps * u / np.linalg.norm(u), dtype=eng.cdtype) * e0[i].norm()
                    kinds.append(("twin", eps, k))
        res = eng.rollout(eng.select(st, rows), T, S, delta=D, readouts=set([0] + list(args.horizons)),
                          targets=[targets[i] for i in rows.tolist()], keep_emb=True)
        emb = torch.stack(res["emb"], 1)                        # (rows, S, d)
        tsum = {"accuracy": {}, "twin": {}, "zero_twin_max_rel": None}
        zero_max = 0.0
        for i, p in enumerate(probs):
            b = i * rows_per
            base_e = emb[b]
            ans = {s: score(res["readouts"][s]["text"][b], p["gold"], p["target"]) for s in res["readouts"]}
            rec = {"kind": "calib", "uid": p["uid"], "id": p["id"], "T": T, "answers": {str(s): a["answer"] for s, a in ans.items()},
                   "correct": {str(s): a["correct"] for s, a in ans.items()},
                   "texts": {str(s): res["readouts"][s]["text"][b] for s in res["readouts"]},
                   "J": {str(s): res["readouts"][s]["J"][b] for s in res["readouts"]}, "twins": []}
            for j in range(1, rows_per):
                r = b + j
                kind, eps, k = kinds[r]
                d = ((emb[r] - base_e).norm(dim=-1) / base_e.norm(dim=-1)).tolist()
                if kind == "zero":
                    zero_max = max(zero_max, max(d))
                tw = {"kind": kind, "eps": eps, "k": k,
                      "log10_d": [round(math.log10(x), 4) if x > 0 else None for x in d],
                      "answers": {str(s): ph.extract_answer(res["readouts"][s]["text"][r]) for s in res["readouts"]}}
                rec["twins"].append(tw)
            recf.write(json.dumps(rec) + "\n")
        recf.flush()
        # summaries
        recs = [json.loads(x) for x in open(out / "records.jsonl")]
        recs = [r for r in recs if r["T"] == T]
        for s in [0] + list(args.horizons):
            k = sum(r["correct"][str(s)] for r in recs)
            tsum["accuracy"][str(s)] = {"correct": k, "n": len(recs), "rate": k / len(recs), "wilson": wilson(k, len(recs))}
        for eps in levels:
            rates, flips, peak = [], {str(s): 0 for s in args.horizons}, []
            n_tw = 0
            for r in recs:
                for tw in r["twins"]:
                    if tw["kind"] != "twin" or tw["eps"] != eps:
                        continue
                    n_tw += 1
                    ld = tw["log10_d"]
                    if ld[-1] is None or ld[0] is None:
                        rates.append(float("-inf"))
                    else:
                        rates.append((ld[-1] - ld[0]) * math.log(10) / (len(ld) - 1))
                    peak.append(max(x for x in ld if x is not None) - ld[0] if ld[0] is not None else None)
                    for s in args.horizons:
                        flips[str(s)] += tw["answers"][str(s)] != r["answers"][str(s)]
            fr = np.array(rates)
            curves = np.array([[np.nan if x is None else x for x in tw["log10_d"]] for r in recs for tw in r["twins"]
                               if tw["kind"] == "twin" and tw["eps"] == eps])
            by_h = {}
            for hz in args.horizons:
                if hz < 2:
                    continue
                rr = [((tw["log10_d"][hz - 1] - tw["log10_d"][0]) * math.log(10) / (hz - 1))
                      if tw["log10_d"][hz - 1] is not None else float("-inf")
                      for r in recs for tw in r["twins"] if tw["kind"] == "twin" and tw["eps"] == eps]
                by_h[str(hz)] = {"median": float(np.median(rr)), "share_expanding": float(np.mean(np.array(rr) > 0))}
            tsum["twin"][str(eps)] = {"n_twins": n_tw, "net_rate_median": float(np.median(fr)),
                                      "net_rate_by_horizon": by_h,
                                      "log10_d_median_by_step": [None if np.all(np.isnan(c)) else round(float(np.nanmedian(c)), 3)
                                                                 for c in curves.T],
                                      "merged_exactly_share_by_step": [round(float(np.mean(np.isnan(c))), 3) for c in curves.T],
                                      "net_rate_quartiles": [float(np.quantile(fr, 0.25)), float(np.quantile(fr, 0.75))],
                                      "share_expanding": float(np.mean(fr > 0)),
                                      "peak_log10_growth_median": float(np.median([x for x in peak if x is not None])),
                                      "answer_flips": flips}
        tsum["zero_twin_max_rel"] = zero_max
        tsum["seconds"] = round(time.time() - t0, 1)
        summary["temperatures"][str(T)] = tsum
        print(json.dumps({"T": T, **{k: v for k, v in tsum.items() if k != "twin"},
                          "twin": {e: {k: v for k, v in s.items() if k in ("net_rate_median", "share_expanding", "answer_flips")}
                                   for e, s in tsum["twin"].items()}}), flush=True)
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "calibrate.json").write_text(json.dumps({"provenance": info, "summary": summary}, indent=1) + "\n")


def torch_threads(args):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))


def cmd_layouts(args):
    """Batch-layout differences: all development problems in one batch against batches of 8 and of 1."""
    torch_threads(args)
    out = out_dir(args, "layouts")
    probs = split_problems(args, "dev")
    S = max(args.horizons)
    res_all = {"provenance": provenance(args), "dtypes": {}}
    for dtype in args.dtypes:
        eng = Engine(args.model, dtype, args.threads, answer_tokens=args.answer_tokens)
        torch = eng.torch
        targets = [eng.target_ids(p["target"]) for p in probs]
        dres = {}
        for T in args.temperatures:
            runs = {}
            for size in args.sizes:
                emb, texts = [], {s: [] for s in args.horizons}
                for s0 in range(0, len(probs), size):
                    chunk = probs[s0:s0 + size]
                    st = eng.prompt_state(chunk)
                    r = eng.rollout(st, T, S, readouts=set(args.horizons), targets=targets[s0:s0 + size], keep_emb=True)
                    emb.append(torch.stack(r["emb"], 1))
                    for s in args.horizons:
                        texts[s] += r["readouts"][s]["text"]
                runs[size] = {"emb": torch.cat(emb, 0), "answers": {s: [ph.extract_answer(x) for x in texts[s]]
                                                                      for s in args.horizons}}
            ref = args.sizes[0]
            comp = {}
            for size in args.sizes[1:]:
                d = ((runs[size]["emb"] - runs[ref]["emb"]).norm(dim=-1) / runs[ref]["emb"].norm(dim=-1))  # (N, S)
                comp[f"{ref}-vs-{size}"] = {
                    "rel_diff_first_token": {"median": float(d[:, 0].median()), "max": float(d[:, 0].max())},
                    "rel_diff_by_step_median": [float(x) for x in d.median(0).values],
                    "rel_diff_by_step_max": [float(x) for x in d.max(0).values],
                    "answer_flips": {str(s): int(sum(a != b for a, b in zip(runs[ref]["answers"][s], runs[size]["answers"][s])))
                                     for s in args.horizons},
                    "n": len(probs)}
            acc = {str(s): sum(a == p["gold"] for a, p in zip(runs[ref]["answers"][s], probs)) for s in args.horizons}
            dres[str(T)] = {"comparisons": comp, "accuracy_ref_layout": acc}
            print(json.dumps({"dtype": dtype, "T": T, **{k: {kk: vv for kk, vv in v.items() if "by_step" not in kk}
                                                         for k, v in comp.items()}, "acc": acc}), flush=True)
        res_all["dtypes"][dtype] = dres
        del eng
    (out / "layouts.json").write_text(json.dumps(res_all, indent=1) + "\n")


def cmd_bench(args):
    """Cost of one gradient pass in objective evaluations, at run batch sizes, interleaved repetitions."""
    torch_threads(args)
    eng = Engine(args.model, "float32", args.threads, answer_tokens=args.answer_tokens)
    torch = eng.torch
    out = out_dir(args, "bench")
    probs = split_problems(args, "dev")
    st = eng.prompt_state(probs)
    targets = [eng.target_ids(p["target"]) for p in probs]
    T = args.temperature
    res = {"provenance": provenance(args), "configs": []}
    for n_steps in args.steps:
        rows_g = torch.tensor([i % len(probs) for i in range(args.grad_rows)])
        rows_f = torch.tensor([0] * args.fwd_rows)
        tg_g = [targets[i] for i in rows_g.tolist()]
        tg_f = [targets[0]] * args.fwd_rows
        base = eng.rollout(st, T, n_steps)
        del base
        tf, tgr, tgo = [], [], []
        for rep in range(args.reps):
            D = torch.randn(args.fwd_rows, eng.d) * 0.01
            t0 = time.time()
            eng.rollout(eng.select(st, rows_f), T, n_steps, delta=D, readouts={n_steps}, targets=tg_f, greedy=False)
            tf.append((time.time() - t0) / args.fwd_rows)
            D = torch.randn(args.grad_rows, eng.d) * 0.01
            t0 = time.time()
            eng.objective(eng.select(st, rows_g), T, n_steps, D, tg_g, grad=True)
            tgr.append((time.time() - t0) / args.grad_rows)
            t0 = time.time()
            eng.objective(eng.select(st, rows_g), T, n_steps, D, tg_g, grad=False)
            tgo.append((time.time() - t0) / args.grad_rows)
        c_grad = float(np.median(tgr) / np.median(tf))
        c_last = float(np.median(tgo) / np.median(tf))
        cfg = {"n_steps": n_steps, "fwd_rows": args.fwd_rows, "grad_rows": args.grad_rows,
               "sec_per_row_eval": tf, "sec_per_row_grad": tgr, "sec_per_row_gradpath_forward": tgo,
               "c_grad": c_grad, "c_gradpath_forward": c_last,
               "K_matched": int(round(args.pga_steps * c_grad + c_last))}
        res["configs"].append(cfg)
        print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("sec")}), flush=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")


def cmd_derivcheck(args):
    """Directional derivatives of the gradient path against central differences; cache against no cache; save
    composed-rollout gradients for a float32 / float64 comparison."""
    torch_threads(args)
    eng = Engine(args.model, args.dtype, args.threads, answer_tokens=args.answer_tokens)
    torch = eng.torch
    out = out_dir(args, "derivcheck")
    probs = split_problems(args, "dev")[:args.n_problems]
    st = eng.prompt_state(probs)
    targets = [eng.target_ids(p["target"]) for p in probs]
    res = {"provenance": provenance(args), "dtype": args.dtype, "checks": [], "cache_vs_nocache": [], "grads": {}}
    grads = {}
    for T in args.temperatures:
        e0 = eng.concept(st["z"], T).to(eng.cdtype)
        nrm = e0.norm(dim=-1)
        for n_steps in args.check_steps:
            for start in ("zero", "random"):
                D = torch.zeros(len(probs), eng.d, dtype=eng.cdtype)
                if start == "random":
                    for i, p in enumerate(probs):
                        D[i] = args.start_radius * nrm[i] * torch.tensor(unit_direction(eng.d, p["uid"], 999, 0, 0, 0),
                                                                          dtype=eng.cdtype)
                J, g = eng.objective(st, T, n_steps, D, targets, grad=True)
                V = torch.stack([torch.tensor(unit_direction(eng.d, p["uid"], 998, 0, 0, 0), dtype=eng.cdtype)
                                 for p in probs])
                dd = (g * V).sum(-1)
                rows = []
                for h in args.h:
                    Jp, _ = eng.objective(st, T, n_steps, D + h * nrm[:, None] * V, targets, grad=False)
                    Jm, _ = eng.objective(st, T, n_steps, D - h * nrm[:, None] * V, targets, grad=False)
                    cdiff = (Jp - Jm) / (2 * h * nrm)
                    rows.append({"h_rel": h, "central": cdiff.tolist(),
                                 "rel_err": ((cdiff - dd).abs() / dd.abs().clamp_min(1e-300)).tolist()})
                # evaluation-path objective at the same point
                ev = eng.rollout(st, T, n_steps, delta=D, readouts={n_steps}, targets=targets, greedy=False)
                Jev = torch.tensor(ev["readouts"][n_steps]["J"], dtype=eng.cdtype)
                item = {"T": T, "n_steps": n_steps, "start": start, "J": J.tolist(), "J_eval_path": Jev.tolist(),
                        "J_path_abs_diff": (J - Jev).abs().tolist(), "directional": dd.tolist(),
                        "grad_norm": g.norm(dim=-1).tolist(), "fd": rows}
                res["checks"].append(item)
                print(json.dumps({"T": T, "n_steps": n_steps, "start": start, "dd": [round(x, 6) for x in dd.tolist()],
                                  "J_diff": [f"{x:.2e}" for x in item["J_path_abs_diff"]],
                                  "rel_err": {r["h_rel"]: [f"{x:.1e}" for x in r["rel_err"]] for r in rows}}), flush=True)
        for n_steps in args.nocache_steps:
            D = torch.zeros(len(probs), eng.d, dtype=eng.cdtype)
            J1, g1 = eng.objective(st, T, n_steps, D, targets, grad=True)
            J2, g2 = eng.objective_nocache(st["ids"], st["z"], T, n_steps, D, targets)
            item = {"T": T, "n_steps": n_steps, "J_cache": J1.tolist(), "J_nocache": J2.tolist(),
                    "grad_rel_diff": ((g1 - g2).norm(dim=-1) / g2.norm(dim=-1)).tolist(),
                    "grad_cos": ((g1 * g2).sum(-1) / (g1.norm(dim=-1) * g2.norm(dim=-1))).tolist()}
            res["cache_vs_nocache"].append(item)
            print(json.dumps({"cache_vs_nocache": item}), flush=True)
        for n_steps in args.save_steps:
            D = torch.zeros(len(probs), eng.d, dtype=eng.cdtype)
            J, g = eng.objective(st, T, n_steps, D, targets, grad=True)
            grads[f"T{T}_S{n_steps}"] = g.double().numpy()
            grads[f"T{T}_S{n_steps}_J"] = J.double().numpy()
    np.savez(out / f"grads-{args.dtype}.npz", **grads)
    if args.compare:
        other = np.load(args.compare)
        for k in grads:
            if k.endswith("_J") or k not in other:
                continue
            a, b = grads[k], other[k]
            res["grads"][k] = {"rel_diff": (np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)).tolist(),
                               "cos": ((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).tolist(),
                               "J_abs_diff": np.abs(grads[k + "_J"] - other[k + "_J"]).tolist()}
            print(json.dumps({k: res["grads"][k]}), flush=True)
    (out / f"derivcheck-{args.dtype}.json").write_text(json.dumps(res, indent=1) + "\n")


def cmd_run(args):
    torch_threads(args)
    eng = Engine(args.model, "float32", args.threads, answer_tokens=args.answer_tokens)
    torch = eng.torch
    probs = split_problems(args, args.split)
    out = out_dir(args, f"run-{args.split}", resume=args.resume)
    allp = load_problems(args.problems)
    info = provenance(args, {"problems_sha256": problems_hash(allp), "model_path": str(eng.path),
                             "uids": [p["uid"] for p in probs]})
    runf = out / "run.json"
    hist = json.loads(runf.read_text())["history"] if runf.exists() else []
    hist.append(info)
    runf.write_text(json.dumps({"history": hist}, indent=1) + "\n")
    done, stored = set(), {}
    recp = out / "records.jsonl"
    if recp.exists():
        for line in open(recp):
            r = json.loads(line)
            if r["kind"] == "done":
                done.add(tuple(r["key"]))
            elif r["kind"] == "baseline":
                stored[(r["uid"], r["T"], r["S"])] = r["answer"]
    recf = open(recp, "a")
    targets = [eng.target_ids(p["target"]) for p in probs]
    horizons = sorted(args.horizons)
    combos = [(S, inj) for S in horizons for inj in (("early",) if S == 1 else ("early", "late"))
              if inj in args.injections]
    long_horizons = args.long_horizons or [max(horizons)]
    K = args.K
    schedule = args.pga_schedule
    if len(schedule) != args.pga_steps:
        raise SystemExit("--pga-schedule needs one value per step")

    def write(rec):
        recf.write(json.dumps(rec) + "\n")

    for T in args.temperatures:
        tk = ("base", T)
        t0 = time.time()
        st = eng.prompt_state(probs)
        snaps = sorted({0} | {S - 1 for S in horizons})
        base = eng.rollout(st, T, max(horizons), readouts=set(horizons), snapshots=set(snaps), targets=targets,
                           force=True)
        base_ans = {}
        for i, p in enumerate(probs):
            for S in horizons:
                ro = base["readouts"][S]
                sc = score(ro["text"][i], p["gold"], p["target"])
                fs = score(ro["forced_text"][i], p["gold"], p["target"])
                base_ans[(i, S)] = sc["answer"]
                if tk in done and stored.get((p["uid"], T, S), "missing") != sc["answer"]:
                    raise SystemExit(f"resumed baseline differs from the stored one: {p['id']} T={T} S={S}")
                if tk not in done:
                    write({"kind": "baseline", "uid": p["uid"], "id": p["id"], "T": T, "S": S, **sc, "text": ro["text"][i],
                           "J": ro["J"][i], "forced_answer": fs["answer"], "forced_target_hit": fs["target_hit"],
                           "forced_text": ro["forced_text"][i], "emb_norm_first": float(base["emb_norm"][0][i]),
                           "emb_norm_last": float(base["emb_norm"][S - 1][i])})
        if tk not in done:
            write({"kind": "timing", "what": "baseline", "T": T, "seconds": round(time.time() - t0, 2), "rows": len(probs)})
            write({"kind": "done", "key": list(tk)})
            recf.flush()
        for S, inj in combos:
            s0 = 0 if inj == "early" else S - 1
            n = S if inj == "early" else 1
            snap = base["snapshots"][s0]
            enorm = base["emb_norm"][s0]
            radii = list(args.radii) + (list(args.control_radii) if inj == "late" and S == max(horizons) else [])
            # ---- random search (candidate 0 is random1), with a zero row per problem
            per = 1 + len(radii) * K
            chunk = max(1, (args.rows_long if n > 1 else args.rows_short) // per)
            for c0 in (range(0, len(probs), chunk) if "search" in args.arms else []):
                key = ("search", T, S, inj, c0)
                if key in done:
                    continue
                idx = list(range(c0, min(len(probs), c0 + chunk)))
                rows, D, meta = [], [], []
                for i in idx:
                    rows.append(i)
                    D.append(torch.zeros(eng.d))
                    meta.append((i, None, None))
                    dirs = [unit_direction(eng.d, probs[i]["uid"], temp_key(T), S, INJ[inj], k) for k in range(K)]
                    for rho in radii:
                        for k in range(K):
                            rows.append(i)
                            D.append(rho * enorm[i] * torch.tensor(dirs[k], dtype=eng.cdtype))
                            meta.append((i, rho, k))
                t1 = time.time()
                res = eng.rollout(eng.select(snap, torch.tensor(rows)), T, n, delta=torch.stack(D), readouts={n},
                                  targets=[targets[i] for i in rows])
                sec = time.time() - t1
                ro = res["readouts"][n]
                cand = defaultdict(list)
                for r, (i, rho, k) in enumerate(meta):
                    p = probs[i]
                    sc = score(ro["text"][r], p["gold"], p["target"])
                    common = {"uid": p["uid"], "id": p["id"], "T": T, "S": S, "inj": inj,
                              "baseline_answer": base_ans[(i, S)], "inj_norm": float(enorm[i])}
                    if rho is None:
                        write({"kind": "arm", "arm": "zero", "batch": "search", "rho": 0.0, **common, **sc,
                               "text": ro["text"][r], "J": ro["J"][r], "realized_rel": res["realized_rel"][r],
                               "changed": sc["answer"] != base_ans[(i, S)]})
                        continue
                    cand[(i, rho)].append((k, ro["J"][r], sc, ro["text"][r], res["realized_rel"][r]))
                for (i, rho), cs in cand.items():
                    p = probs[i]
                    common = {"uid": p["uid"], "id": p["id"], "T": T, "S": S, "inj": inj, "rho": rho,
                              "baseline_answer": base_ans[(i, S)], "inj_norm": float(enorm[i])}
                    k0 = next(c for c in cs if c[0] == 0)
                    write({"kind": "arm", "arm": "random1", **common, **k0[2], "text": k0[3], "J": k0[1],
                           "realized_rel": k0[4], "changed": k0[2]["answer"] != base_ans[(i, S)]})
                    best = max(cs, key=lambda c: c[1])
                    write({"kind": "arm", "arm": "search", "K": K, **common, **best[2], "text": best[3], "J": best[1],
                           "best_k": best[0], "realized_rel": best[4], "changed": best[2]["answer"] != base_ans[(i, S)],
                           "cand_J": [round(c[1], 5) for c in cs], "cand_answers": [c[2]["answer"] for c in cs],
                           "cand_target_hits": int(sum(c[2]["target_hit"] for c in cs)),
                           "cand_changed": int(sum(c[2]["answer"] != base_ans[(i, S)] for c in cs))})
                write({"kind": "timing", "what": "search", "T": T, "S": S, "inj": inj, "n_steps": n, "rows": len(rows),
                       "problems": len(idx), "K": K, "radii": radii, "seconds": round(sec, 3)})
                write({"kind": "done", "key": list(key)})
                recf.flush()
            # ---- projected gradient ascent (the compute-matched arm "pga" and, optionally, a long variant
            # "pga<N>" with N steps of 2.5 r / N, as in Madry et al.'s L2 PGD), then evaluation with zero rows
            variants = []
            if "pga" in args.arms:
                variants.append(("pga", args.pga_steps, schedule, radii, list(range(len(probs)))))
            if "long" in args.arms and S in long_horizons and inj in args.long_injections:
                nl = min(args.long_limit or len(probs), len(probs))
                variants.append((f"pga{args.long_steps}", args.long_steps, [2.5 / args.long_steps] * args.long_steps,
                                 list(args.long_radii) or radii, list(range(nl))))
            for arm, steps, sched, vradii, vprobs in variants:
                pairs = [(i, rho) for i in vprobs for rho in vradii]
                bs = args.pga_rows_long if n > 1 else args.pga_rows_short
                for c0 in range(0, len(pairs), bs):
                    key = (arm, T, S, inj, c0)
                    if key in done:
                        continue
                    pc = pairs[c0:c0 + bs]
                    rows = torch.tensor([i for i, _ in pc])
                    radius = torch.tensor([rho * float(enorm[i]) for i, rho in pc], dtype=eng.cdtype)
                    tg = [targets[i] for i, _ in pc]
                    t1 = time.time()
                    pg = eng.pga(eng.select(snap, rows), T, n, radius, tg, steps, sched)
                    sec_opt = time.time() - t1
                    zi = sorted({i for i, _ in pc})
                    erows = rows.tolist() + zi
                    ED = torch.cat([pg["delta"], torch.zeros(len(zi), eng.d, dtype=eng.cdtype)])
                    t1 = time.time()
                    res = eng.rollout(eng.select(snap, torch.tensor(erows)), T, n, delta=ED, readouts={n},
                                      targets=[targets[i] for i in erows])
                    sec_eval = time.time() - t1
                    ro = res["readouts"][n]
                    for r, (i, rho) in enumerate(pc):
                        p = probs[i]
                        sc = score(ro["text"][r], p["gold"], p["target"])
                        write({"kind": "arm", "arm": arm, "uid": p["uid"], "id": p["id"], "T": T, "S": S, "inj": inj,
                               "rho": rho, "baseline_answer": base_ans[(i, S)], "inj_norm": float(enorm[i]), **sc,
                               "text": ro["text"][r], "J": ro["J"][r], "realized_rel": res["realized_rel"][r],
                               "changed": sc["answer"] != base_ans[(i, S)], "J0": float(pg["J0"][r]),
                               "J_iterates": [round(float(x), 5) for x in pg["J_iterates"][r]], "best_iter": int(pg["best"][r]) + 1,
                               "grad_norms": [float(x) for x in pg["grad_norms"][r]]})
                    for r, i in enumerate(zi, start=len(pc)):
                        p = probs[i]
                        sc = score(ro["text"][r], p["gold"], p["target"])
                        write({"kind": "arm", "arm": "zero", "batch": arm, "uid": p["uid"], "id": p["id"], "T": T, "S": S,
                               "inj": inj, "rho": 0.0, "baseline_answer": base_ans[(i, S)], **sc, "text": ro["text"][r],
                               "J": ro["J"][r], "realized_rel": res["realized_rel"][r],
                               "changed": sc["answer"] != base_ans[(i, S)]})
                    write({"kind": "timing", "what": arm, "T": T, "S": S, "inj": inj, "n_steps": n, "rows": len(pc),
                           "grad_passes": steps, "forward_passes": 1, "seconds_opt": round(sec_opt, 3),
                           "eval_rows": len(erows), "seconds_eval": round(sec_eval, 3)})
                    write({"kind": "done", "key": list(key)})
                    recf.flush()
            print(json.dumps({"T": T, "S": S, "inj": inj, "elapsed": round(time.time() - t0, 1)}), flush=True)
    hist[-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runf.write_text(json.dumps({"history": hist}, indent=1) + "\n")


# ----------------------------------------------------------------------------------------------- analysis

def load_records(run):
    return [json.loads(line) for line in open(Path(run) / "records.jsonl")]


def gate(recs):
    """Row gates before any summary: one baseline per (problem, T, S); no duplicated arm rows; every arm row has a
    baseline; realized radii match the requested ones; zero rows reproduce the baseline; forced-target control."""
    issues = []
    base = {}
    for r in recs:
        if r["kind"] == "baseline":
            k = (r["uid"], r["T"], r["S"])
            if k in base:
                issues.append(f"duplicate baseline {k}")
            base[k] = r
    seen = set()
    rel_err = []
    for r in recs:
        if r["kind"] != "arm":
            continue
        k = (r["arm"], r.get("batch"), r["uid"], r["T"], r["S"], r["inj"], r["rho"])
        if k in seen and r["arm"] != "zero":       # a zero row per problem and batch: repeats are expected
            issues.append(f"duplicate arm row {k}")
        seen.add(k)
        if (r["uid"], r["T"], r["S"]) not in base:
            issues.append(f"arm row without baseline {k}")
        if r["arm"] in ("random1", "search"):
            rel_err.append(abs(r["realized_rel"] - r["rho"]) / r["rho"])
        if r["arm"] == "pga" and r["realized_rel"] > r["rho"] * (1 + 1e-4):
            issues.append(f"pga radius exceeded {k}: {r['realized_rel']}")
    return base, issues, (max(rel_err) if rel_err else None)


def rate_block(rows, base, eligible_only):
    """Target-hit, answer-change and invalid rates over (problem, T) pairs; problems weighted equally."""
    sel = [r for r in rows if not eligible_only or base[(r["uid"], r["T"], r["S"])]["correct"]]
    n = len(sel)
    k = sum(r["target_hit"] for r in sel)
    pw = per_problem_mean((r["uid"], r["target_hit"]) for r in sel)
    out = {"pairs": n, "problems": len({r["uid"] for r in sel}), "target_hits": k,
           "rate": k / n if n else None, "wilson95_pairs": wilson(k, n),
           "problem_weighted": float(np.mean(pw)) if pw else None, "boot95_problems": bootstrap_mean_ci(pw),
           "answer_changed": sum(r["changed"] for r in sel), "invalid": sum(r["invalid"] for r in sel),
           "correct_kept": sum(r["correct"] for r in sel)}
    return out


def paired_delta(rows_a, rows_b, base, eligible_only=True):
    """Problem-weighted mean of hit_a - hit_b over pairs present in both arms."""
    a = {(r["uid"], r["T"]): r for r in rows_a}
    b = {(r["uid"], r["T"]): r for r in rows_b}
    keys = sorted(set(a) & set(b))
    if eligible_only:
        keys = [k for k in keys if base[(k[0], k[1], a[k]["S"])]["correct"]]
    vals = per_problem_mean((k[0], float(a[k]["target_hit"]) - float(b[k]["target_hit"])) for k in keys)
    est = float(np.mean(vals)) if vals else None
    return {"estimate": est, "boot95_problems": bootstrap_mean_ci(vals), "pairs": len(keys), "problems": len(vals),
            "a_only": sum(a[k]["target_hit"] and not b[k]["target_hit"] for k in keys),
            "b_only": sum(b[k]["target_hit"] and not a[k]["target_hit"] for k in keys)}


def cmd_analyze(args):
    recs = []
    runs = {}
    for run in args.runs:
        rr = load_records(run)
        recs += rr
        runs[run] = json.loads((Path(run) / "run.json").read_text())["history"]
    base, issues, rel_err = gate(recs)
    arms = defaultdict(list)
    for r in recs:
        if r["kind"] == "arm":
            arms[(r["S"], r["inj"], r["rho"], r["arm"], r.get("batch"))].append(r)
    temps = sorted({r["T"] for r in base.values()})
    horizons = sorted({r["S"] for r in base.values()})
    summ = {"runs": {k: {"commit": v[-1]["commit"], "tool_sha256": v[-1]["tool_sha256"],
                         "problems_sha256": v[-1]["problems_sha256"], "args": v[-1]["args"]} for k, v in runs.items()},
            "gate": {"issues": issues, "n_records": len(recs), "realized_rel_max_err_random": rel_err}}
    # baseline and eligibility
    bl = {}
    for S in horizons:
        for T in temps:
            rs = [r for r in base.values() if r["S"] == S and r["T"] == T]
            k = sum(r["correct"] for r in rs)
            bl[f"S{S}_T{T}"] = {"correct": k, "n": len(rs), "rate": k / len(rs) if rs else None,
                                "wilson95": wilson(k, len(rs)), "invalid": sum(r["invalid"] for r in rs),
                                "baseline_is_target": sum(r["target_hit"] for r in rs),
                                "forced_target_scorer_pass": sum(r["forced_target_hit"] for r in rs)}
    summ["baseline"] = bl
    # zero-perturbation control
    zero = [r for r in recs if r["kind"] == "arm" and r["arm"] == "zero"]
    summ["controls"] = {"zero_perturbation": {"rows": len(zero), "answer_differs_from_baseline": sum(r["changed"] for r in zero),
                                              "differing": [{k: r[k] for k in ("id", "T", "S", "inj", "batch", "answer",
                                                                                "baseline_answer")} for r in zero if r["changed"]]},
                        "forced_target_scorer": {"rows": len(base), "pass": sum(r["forced_target_hit"] for r in base.values())}}
    # per-arm rates
    rates = {}
    for (S, inj, rho, arm, batch), rows in sorted(arms.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], str(x[0][4]))):
        if arm == "zero":
            continue
        rates[f"S{S}_{inj}_rho{rho}_{arm}"] = {
            "S": S, "inj": inj, "rho": rho, "arm": arm,
            "eligible": rate_block(rows, base, True), "unconditional": rate_block(rows, base, False),
            "by_temperature": {str(T): rate_block([r for r in rows if r["T"] == T], base, True) for T in temps}}
        if arm == "search":
            el = [r for r in rows if base[(r["uid"], r["T"], r["S"])]["correct"]]
            nc = sum(len(r["cand_answers"]) for r in el)
            rates[f"S{S}_{inj}_rho{rho}_{arm}"]["candidates_eligible"] = {
                "candidates": nc, "changed": sum(r["cand_changed"] for r in el),
                "target_hits": sum(r["cand_target_hits"] for r in el),
                "any_of_K_target_pairs": sum(r["cand_target_hits"] > 0 for r in el), "pairs": len(el)}
        if arm.startswith("pga"):
            rates[f"S{S}_{inj}_rho{rho}_{arm}"]["J_gain_median"] = float(np.median(
                [max(r["J_iterates"]) - r["J0"] for r in rows])) if rows else None
    summ["rates"] = rates
    norms = defaultdict(dict)
    for r in recs:
        if r["kind"] == "arm" and r["arm"] in ("search", "random1") or (r["kind"] == "arm" and r["arm"].startswith("pga")):
            norms[(r["T"], r["S"], r["inj"])][r["uid"]] = r["inj_norm"]
    summ["injected_token_norm"] = {f"T{T}_S{S}_{inj}": {"median": float(np.median(list(v.values()))),
                                                         "min": float(min(v.values())), "max": float(max(v.values())),
                                                         "problems": len(v)}
                                   for (T, S, inj), v in sorted(norms.items())}
    # contrasts
    con = {}
    for (S, inj, rho, arm, batch) in arms:
        if not arm.startswith("pga"):
            continue
        for other in ("search", "random1"):
            if (S, inj, rho, other, None) in arms:
                con[f"S{S}_{inj}_rho{rho}_{arm}-minus-{other}"] = paired_delta(arms[(S, inj, rho, arm, None)],
                                                                               arms[(S, inj, rho, other, None)], base)
    for (S, inj, rho, arm, batch) in arms:
        if inj != "early" or arm == "zero" or (S, "late", rho, arm, None) not in arms:
            continue
        con[f"S{S}_rho{rho}_{arm}_early-minus-late"] = paired_delta(arms[(S, "early", rho, arm, None)],
                                                                    arms[(S, "late", rho, arm, None)], base)
    summ["contrasts"] = dict(sorted(con.items()))
    if args.primary_radius is not None:
        S = max(horizons)
        pa = args.primary_arm
        key = f"S{S}_early_rho{args.primary_radius}_{pa}-minus-search"
        if key not in con:
            raise SystemExit(f"primary contrast {key} missing")
        pr = con[key]
        summ["primary"] = {"definition": f"Delta = P(target | {pa}) - P(target | search), early injection, "
                                         f"S = {S}, rho = {args.primary_radius}, baseline-correct problem-temperature "
                                         "pairs, problems weighted equally, paired bootstrap over problems",
                           **pr, "decision": decision(pr["estimate"], pr["boot95_problems"])}
        rho = args.primary_radius
        summ["primary_unconditional"] = paired_delta(arms[(S, "early", rho, pa, None)],
                                                     arms[(S, "early", rho, "search", None)], base, eligible_only=False)
        el = f"S{S}_rho{rho}_{pa}_early-minus-late"
        summ["early_vs_late_primary_dose"] = con.get(el)
        if (S, "late", rho, pa, None) in arms:
            summ["early_vs_late_primary_dose_unconditional"] = paired_delta(
                arms[(S, "early", rho, pa, None)], arms[(S, "late", rho, pa, None)], base, eligible_only=False)
    # compute
    tim = [r for r in recs if r["kind"] == "timing"]
    comp = {}
    for what in ["search"] + sorted({t["what"] for t in tim if t["what"].startswith("pga")}):
        for S in horizons:
            for inj in ("early", "late"):
                ts = [t for t in tim if t["what"] == what and t["S"] == S and t["inj"] == inj]
                if not ts:
                    continue
                if what == "search":
                    sec = sum(t["seconds"] for t in ts)
                    cand_rows = sum(t["rows"] - t["problems"] for t in ts)
                    comp[f"S{S}_{inj}_search"] = {"seconds": round(sec, 1), "candidate_rows": cand_rows,
                                                   "sec_per_candidate_with_greedy": sec / max(1, sum(t["rows"] for t in ts))}
                else:
                    sec = sum(t["seconds_opt"] for t in ts)
                    rows = sum(t["rows"] for t in ts)
                    comp[f"S{S}_{inj}_{what}"] = {"seconds_opt": round(sec, 1), "rows": rows,
                                                "sec_per_row_opt": sec / max(1, rows),
                                                "seconds_eval": round(sum(t["seconds_eval"] for t in ts), 1)}
    summ["compute"] = comp
    txt = json.dumps(summ, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(txt + "\n")
    if issues:
        print(json.dumps({"GATE ISSUES": issues[:20], "count": len(issues)}), file=sys.stderr)
    print(json.dumps({"primary": summ.get("primary"), "gate": summ["gate"], "controls": {
        "zero_differs": summ["controls"]["zero_perturbation"]["answer_differs_from_baseline"],
        "forced": summ["controls"]["forced_target_scorer"]}}, indent=1))


def choose_doses(rates, horizon, threshold=0.25, fallback=(1.0, 3.0)):
    """Dose rule (fixed before the development staircase): the smaller dose is the smallest staircase radius at
    which the pga arm, early or late injection (whichever is higher), converts at least `threshold` of the eligible
    development pairs (both temperatures pooled); the larger dose is the next staircase radius. If no radius
    qualifies, `fallback`."""
    radii = sorted({v["rho"] for v in rates.values() if v["S"] == horizon and v["arm"] == "pga"})
    table = []
    for rho in radii:
        r = {inj: rates.get(f"S{horizon}_{inj}_rho{rho}_pga", {}).get("eligible", {}).get("rate") for inj in ("early", "late")}
        table.append({"rho": rho, "pga_early": r["early"], "pga_late": r["late"],
                      "best": max(x for x in r.values() if x is not None)})
    ok = [t["rho"] for t in table if t["best"] >= threshold]
    if not ok:
        return {"rule_met": False, "small": fallback[0], "large": fallback[1], "table": table}
    small = ok[0]
    bigger = [r for r in radii if r > small]
    return {"rule_met": True, "small": small, "large": bigger[0] if bigger else 3 * small, "table": table}


def cmd_doses(args):
    a = json.loads(Path(args.analysis).read_text())
    out = choose_doses(a["rates"], args.horizon, args.threshold)
    print(json.dumps(out, indent=1))


def compact_rates(rates, keep=("pairs", "problems", "target_hits", "rate", "wilson95_pairs", "problem_weighted",
                                "boot95_problems", "answer_changed", "invalid", "correct_kept")):
    out = {}
    for k, v in rates.items():
        out[k] = {"eligible": {kk: v["eligible"][kk] for kk in keep},
                  "unconditional": {kk: v["unconditional"][kk] for kk in keep},
                  "by_temperature_eligible": {T: {kk: b[kk] for kk in ("pairs", "target_hits", "answer_changed")}
                                              for T, b in v["by_temperature"].items()}}
        for extra in ("candidates_eligible", "J_gain_median"):
            if extra in v:
                out[k][extra] = v[extra]
    return out


def cmd_receipt(args):
    """Compact receipt: settings and choices (from --spec), calibration, layouts, compute, derivative checks, the
    analysis of each run, and the SHA-256 and size of every raw output file."""
    spec = json.loads(Path(args.spec).read_text())
    rec = {k: v for k, v in spec.items() if k not in ("paths", "raw_dirs")}
    rec["problems"] = {"file": "docs/pi/c12-1-problems.json", "seed": PROBLEM_SEED, "n_dev": N_DEV, "n_test": N_TEST,
                       "sha256": problems_hash(make_problems()),
                       "target_rule": "correct + 10 if that keeps the digit count, else correct - 10"}
    paths = spec["paths"]
    rec["calibration"] = {}
    cal_paths = paths["calibrate"] if isinstance(paths["calibrate"], dict) else {"calibrate": paths["calibrate"]}
    for name, cp in cal_paths.items():
        cal = json.loads((Path(cp) / "calibrate.json").read_text())
        keys = ("n_twins", "net_rate_median", "net_rate_quartiles", "share_expanding", "net_rate_by_horizon",
                "answer_flips", "log10_d_median_by_step")
        rec["calibration"][name] = {
            "dir": str(cp), "commit": cal["provenance"]["commit"],
            "answer_tokens": cal["provenance"]["args"].get("answer_tokens", ANSWER_TOKENS),
            "temperatures": {T: {"accuracy": {S: f"{a['correct']}/{a['n']}" for S, a in v["accuracy"].items()},
                                 "twins": {e: {k: t[k] for k in keys if k in t} for e, t in v["twin"].items()},
                                 "zero_twin_max_rel": v["zero_twin_max_rel"]}
                             for T, v in cal["summary"]["temperatures"].items()}}
    if "layouts" in paths:
        lay = json.loads((Path(paths["layouts"]) / "layouts.json").read_text())
        rec["layouts"] = {"commit": lay["provenance"]["commit"], "dtypes": {
            dt: {T: {c: {"first_token_rel_diff": x["rel_diff_first_token"],
                         "max_rel_diff_any_step": max(x["rel_diff_by_step_max"]),
                         "answer_flips": x["answer_flips"], "n": x["n"]} for c, x in v["comparisons"].items()}
                 for T, v in d.items()} for dt, d in lay["dtypes"].items()}}
    if "bench" in paths:
        b = json.loads((Path(paths["bench"]) / "bench.json").read_text())
        rec["bench"] = {"commit": b["provenance"]["commit"],
                        "configs": [{k: c[k] for k in ("n_steps", "fwd_rows", "grad_rows", "c_grad",
                                                        "c_gradpath_forward", "K_matched")} for c in b["configs"]]}
    rec["derivcheck"] = {}
    for dt, dpath in paths.get("derivcheck", {}).items():
        d = json.loads(Path(dpath).read_text())
        fd = defaultdict(list)
        for c in d["checks"]:
            for row in c["fd"]:
                fd[(c["T"], c["n_steps"], row["h_rel"])] += row["rel_err"]
        rec["derivcheck"][dt] = {
            "commit": d["provenance"]["commit"],
            "central_difference_rel_err_max": {f"T{T}_S{n}_h{h}": float(np.max(v)) for (T, n, h), v in sorted(fd.items())},
            "gradient_path_vs_evaluation_path_J_absdiff_max": float(max(max(c["J_path_abs_diff"]) for c in d["checks"])),
            "cache_vs_nocache": [{"T": c["T"], "n_steps": c["n_steps"], "grad_rel_diff_max": max(c["grad_rel_diff"]),
                                  "grad_cos_min": min(c["grad_cos"])} for c in d["cache_vs_nocache"]],
            "float32_vs_float64": {k: {"grad_rel_diff_max": max(v["rel_diff"]), "grad_cos_min": min(v["cos"]),
                                       "J_absdiff_max": max(v["J_abs_diff"])} for k, v in d.get("grads", {}).items()}}
    rec["runs"] = {}
    for name, rp in paths["runs"].items():
        a = json.loads(Path(rp["analysis"]).read_text())
        rec["runs"][name] = {"dir": rp["dir"], "runs": a["runs"], "gate": a["gate"], "baseline": a["baseline"],
                             "controls": a["controls"], "rates": compact_rates(a["rates"]), "contrasts": a["contrasts"],
                             "compute": a["compute"], "injected_token_norm": a.get("injected_token_norm")}
        for k in ("primary", "primary_unconditional", "early_vs_late_primary_dose",
                  "early_vs_late_primary_dose_unconditional"):
            if k in a:
                rec["runs"][name][k] = a[k]
    raw = []
    for d in spec["raw_dirs"]:
        for f in sorted(Path(d).rglob("*")):
            if f.is_file():
                raw.append({"path": str(f.relative_to(DATA.parent.parent.parent)), "sha256": sha256_file(f),
                            "bytes": f.stat().st_size})
    rec["raw"] = {"root": str(DATA.parent.parent.parent), "files": raw}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"written": args.json, "raw_files": len(raw)}))


# ----------------------------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="qwen0.5b", choices=sorted(ph.MODELS))
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--problems", default=None, help="problem list JSON (default: regenerate; a file must match)")
        p.add_argument("--limit", type=int, default=0, help="first N problems of the split (0: all)")
        p.add_argument("--out", default=None)
        p.add_argument("--answer-tokens", type=int, default=ANSWER_TOKENS,
                       help="greedy answer tokens after the cue (pi_harness default 12; C12-1 runs use 6, see docstring)")

    p = sub.add_parser("problems")
    p.add_argument("--json", default=None)
    p = sub.add_parser("calibrate")
    common(p)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--twins", type=int, default=3)
    p = sub.add_parser("layouts")
    common(p)
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--sizes", type=int, nargs="+", default=[16, 8, 1])
    p.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    p = sub.add_parser("bench")
    common(p)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[32, 8, 1])
    p.add_argument("--fwd-rows", type=int, default=33)
    p.add_argument("--grad-rows", type=int, default=32)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--pga-steps", type=int, default=5)
    p = sub.add_parser("derivcheck")
    common(p)
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument("--n-problems", type=int, default=2)
    p.add_argument("--check-steps", type=int, nargs="+", default=[4])
    p.add_argument("--nocache-steps", type=int, nargs="*", default=[4])
    p.add_argument("--save-steps", type=int, nargs="*", default=[32])
    p.add_argument("--h", type=float, nargs="+", default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4])
    p.add_argument("--start-radius", type=float, default=0.03)
    p.add_argument("--compare", default=None, help="grads-<dtype>.npz of the other precision")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--temperatures", type=float, nargs="+", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--radii", type=float, nargs="+", required=True, help="relative to the injected token's norm")
    p.add_argument("--control-radii", type=float, nargs="*", default=[1.0],
                   help="extra radii for late injection at the longest horizon (optimizer positive control)")
    p.add_argument("--K", type=int, required=True, help="random-search directions (compute-matched; see bench)")
    p.add_argument("--pga-steps", type=int, default=5)
    p.add_argument("--pga-schedule", type=float, nargs="+", default=[1.0, 0.5, 0.5, 0.25, 0.25])
    p.add_argument("--rows-long", type=int, default=72, help="search rows per batch, multi-step segments")
    p.add_argument("--rows-short", type=int, default=200, help="search rows per batch, one-step segments")
    p.add_argument("--pga-rows-long", type=int, default=32)
    p.add_argument("--pga-rows-short", type=int, default=128)
    p.add_argument("--injections", nargs="+", default=["early", "late"], choices=("early", "late"))
    p.add_argument("--arms", nargs="+", default=["search", "pga"], choices=("search", "pga", "long"))
    p.add_argument("--long-steps", type=int, default=30, help="long pga variant: steps (step 2.5 r / N)")
    p.add_argument("--long-radii", type=float, nargs="*", default=[],
                   help="default: the main radii (with --control-radii for late injection at the longest horizon)")
    p.add_argument("--long-injections", nargs="+", default=["late"], choices=("early", "late"))
    p.add_argument("--long-horizons", type=int, nargs="*", default=[], help="default: the longest horizon")
    p.add_argument("--long-limit", type=int, default=0, help="long variant on the first N problems (0: all)")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("runs", nargs="+")
    p.add_argument("--primary-radius", type=float, default=None)
    p.add_argument("--primary-arm", default="pga", help="optimizer arm of the primary contrast (e.g. pga20)")
    p.add_argument("--json", default=None)
    p = sub.add_parser("doses")
    p.add_argument("analysis", help="analysis JSON of the development staircase")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--threshold", type=float, default=0.25)
    p = sub.add_parser("receipt")
    p.add_argument("--spec", required=True, help="JSON: card, settings, choices, deviations, paths, raw_dirs")
    p.add_argument("--json", required=True)
    args = ap.parse_args()
    {"doses": cmd_doses, "receipt": cmd_receipt, "problems": cmd_problems, "calibrate": cmd_calibrate, "layouts": cmd_layouts, "bench": cmd_bench,
     "derivcheck": cmd_derivcheck, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
