"""C12-2 (exploratory, program PI): compression fidelity of the soft-token map.

Question. Can weight quantization change the closed-loop behaviour of the soft-token map (which attractors fixed starts
reach) while the one-step next-token distributions stay close? If so, one-step fidelity tests are not enough to accept
a compressed model for feedback use.

Map. The autonomous soft-token map of Qwen2.5-0.5B-Instruct, window 1, fixed prefix, unsteered, float64 throughout
(explore_llm_attractors.Engine: float64 weights, keep_float64_norms for the RMSNorm layers and the rotary tables):
    x_{t+1} = F(x_t) = E^T softmax(z(x_t) / T),
z(x) the next-token logits after the prompt's prefix followed by the soft token x, E the input embedding (tied with the
output head). Rows of different prompts run in one batch (explore_llm_soft_loop.MultiPrefixSoftLoop: left padding,
masked, true positions); the prefix cache is computed once per prompt and shared by that prompt's rows.

Prompts. 8 system prompts x 8 openers (SYSTEM_PROMPTS, OPENERS); the prefix is the chat template (system, user opener,
generation prompt). Prompt id = 8 s + o. Development: system prompts 0 and 1 (ids 0-15); held out: system prompts 2-7
(ids 16-63). Prompts run in the balanced order of prompt_order (opener-major): development batches of 4 prompts,
held-out batches of 6 (one opener x the 6 held-out system prompts), so a time budget (--budget-hours), which only ever
stops a run between batches and never on outcomes, keeps the held-out system prompts balanced.

Grid. 8 temperatures at log10 T = linspace(-0.5, 0.5, 8); starts x_0 = E[t] for the first --starts tokens of
START_TOKENS (8 fixed common tokens).

Variants. Only the weights of the 168 nn.Linear layers of the 24 decoder blocks change; embeddings, the tied output
head, the norms and the q/k/v biases are kept.
  ref     the stored bfloat16 weights (exact in float64).
  q8      Q8_0-like: blocks of 32 consecutive weights along the input dimension of each output row; float32
          arithmetic as in ggml: a = max_j |w_j|, d = fl32(a / Q), i = fl32(1 / d) (0 if d = 0),
          q_j = clamp(roundf(fl32(w_j i)), -Q, Q) with roundf rounding half away from zero, s = fp16(d) (round to
          nearest even), w'_j = q_j s (exact). Q = 127: this is ggml's Q8_0 reference quantizer and dequantizer.
  q4      Q4-like symmetric: the same with Q = 7 (15 levels -7..7, one fp16 scale per block of 32). ggml's Q4_0
          differs in its levels: d = m / -8 with m the signed element of largest magnitude, levels -8..7.
  noise   the reference map with a frozen random logit bias (unstructured control): z'(x) = z(x) + sigma_T b_p, where
          b_p ~ N(0, I_V) over the whole vocabulary is drawn once per prompt (numpy default_rng((SEED_NOISE, prompt
          id))) and is the same for every state, start and temperature of that prompt, and sigma_T is one scale per
          temperature, set on the development prompts (subcommand calibrate) so that the mean one-step
          KL(p_ref || p_noise) over the development reference attractor states at that temperature equals the mean
          KL(p_ref || p_q8) over the same states.
  repeat  the reference run again in a separate process and output directory (negative control; subcommand compare):
          every array must be bitwise identical.

Classification of every start (tol = 1e-8; lag q closes at step t if |x_t - x_{t-q}| < tol max(1, |x_t|), the period
rule of steer_gain.py):
  main phase, 256 + 64 steps: q = the smallest lag 1..8 that closes at step 256; the orbit is a verified period-q
    orbit ('periodic') if lag q also closes at every step 257..320 and the smallest lag 1..64 closing at step 320 is q.
    Otherwise, if some lag 1..64 closes at step 320, the start is 'unresolved' with sub-label long_period (lag > 8) or
    late_convergence (lag <= 8: closed only after step 256, or the period changed in the confirmation window).
  twin phase, for the starts with no lag 1..64 closing at step 320: M (--twin-steps, default 256) further steps of the
    base from x_320 together with two twins at relative offsets 1e-6 and 1e-8 (|y - x| = eps |x|, random directions
    seeded per start), renormalized to eps |F(x)| after every step as in steer_gain.py;
    lambda = (1/M) sum_t log(|F(y_t) - F(x_t)| / |y_t - x_t|). 'chaotic' if both exponents exceed 1e-3 and no lag
    1..64 closes at the last step (the repository's chaos rule); otherwise 'unresolved' with sub-label late_period (a
    lag closes at the end of the twin phase) or irregular. Unresolved is its own class and is never counted as chaos.

One-step fidelity (KL pass) at common states: the states x_0, x_1, x_4, x_16, x_64 ("transient") and x_313..x_320
("attractor") of every reference trajectory. Per state and variant v: KL(p_ref || p_v) at the map temperature T and at
T = 1, total variation at T, top-1 agreement of the logits, and the relative one-step map error
|F_v(x) - F_ref(x)| / |F_ref(x)|. Every variant's prefix cache is computed by that variant's own weights.

Primary endpoint (subcommand summarize): mean over held-out prompts of share_q4 - share_ref, where share is the fraction
of a prompt's starts (all temperatures and starts) in verified period-<=8 orbits; paired t interval over prompts (the
units), with a prompt bootstrap and a system-prompt cluster bootstrap as sensitivity checks. The same for q8 and noise;
class-or-period change shares; per-temperature shares; KL statistics. Sensitivity endpoints (slow convergence): the same
difference with the share of starts that reach a period <= 8 by the end of the twin phase (step 576), and with the share
of starts showing any closing lag <= 64.

Outputs under ../neural-shrimp-data/explore/pi/c12-2/<out>/: settings.json (the run's fixed settings; a resumed run
must match them), manifest-<time>.json per invocation, progress.jsonl, batches/bNN-<variant>.npz and .json (per-row
arrays and records), batches/bNN-kl.npz (KL pass), calibration.json (subcommand calibrate).

Subcommands: run, calibrate, precision, compare, summarize.

Usage: CUDA_VISIBLE_DEVICES= uv run --no-sync python tools/c12_compression_fidelity.py run --set dev --variants ref \
           --out dev --threads 6
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
OUT_ROOT = DATA / "explore" / "pi" / "c12-2"

MODEL = "qwen0.5b"
SYSTEM_PROMPTS = (
    "You are a helpful assistant.",
    "You are a friendly conversation partner.",
    "You are a concise assistant who keeps replies short.",
    "You are a patient teacher.",
    "You are a careful assistant who answers clearly.",
    "You are a curious and thoughtful companion.",
    "You are a calm assistant who explains things simply.",
    "You are a knowledgeable guide.",
)
OPENERS = (
    "Hello!",
    "Good morning.",
    "What should we talk about?",
    "Tell me something interesting.",
    "Can you help me with a question?",
    "How are you today?",
    "Let's get started.",
    "I have a few minutes to chat.",
)
START_TOKENS = (" the", ".", "\n", " I", ",", " a", " and", "?")
DEV_SYSTEMS = (0, 1)
HELDOUT_SYSTEMS = (2, 3, 4, 5, 6, 7)
LOG_T = tuple(float(x) for x in np.linspace(-0.5, 0.5, 8))
STEPS, CONFIRM, TWIN_STEPS = 256, 64, 256
TOL, SHORT, LONG = 1e-8, 8, 64
EPS = (1e-6, 1e-8)
LAMBDA_MIN = 1e-3
KEEP_TRANSIENT = (0, 1, 4, 16, 64)
N_ATTRACTOR = 8
QMAX = {"q8": 127, "q4": 7}
BLOCK = 32
SEED_TWINS, SEED_NOISE, SEED_BOOT = 20261016, 20261017, 20261018
PROMPTS_PER_BATCH = {"dev": 4, "heldout": 6}    # held out: one opener x the 6 held-out system prompts per batch
ROUND_BATCHES = {"dev": 1, "heldout": 1}        # the budget may stop a held-out run at any batch boundary
TWIN_CHUNK = 64            # starts per twin-phase batch (3 rows each)
KL_CHUNK = 128             # states per KL-pass forward
VARIANTS = ("ref", "q4", "q8", "noise")
CLASS_NAMES = ("periodic", "chaotic", "unresolved")
SUB_NAMES = ("none", "long_period", "late_convergence", "late_period", "irregular", "pending")
SUB_NONE, SUB_LONG, SUB_LATE_CONV, SUB_LATE_PERIOD, SUB_IRREGULAR, SUB_PENDING = range(6)


# ----------------------------------------------------------------------------------------------- prompts and grid

def prompt_id(s, o):
    return 8 * int(s) + int(o)


def prompt_messages(pid):
    s, o = divmod(int(pid), 8)
    return [{"role": "system", "content": SYSTEM_PROMPTS[s]}, {"role": "user", "content": OPENERS[o]}]


def prompt_order(split):
    """Prompt ids of a split in the balanced run order: opener-major, so that every 6 consecutive held-out prompts
    cover the 6 held-out system prompts once."""
    systems = {"dev": DEV_SYSTEMS, "heldout": HELDOUT_SYSTEMS}[split]
    return [prompt_id(s, o) for o in range(len(OPENERS)) for s in systems]


def batches_of(split):
    order, n = prompt_order(split), PROMPTS_PER_BATCH[split]
    return [order[k:k + n] for k in range(0, len(order), n)]


def prompt_lists():
    return {"system_prompts": list(SYSTEM_PROMPTS), "openers": list(OPENERS), "start_tokens": list(START_TOKENS),
            "dev_systems": list(DEV_SYSTEMS), "heldout_systems": list(HELDOUT_SYSTEMS)}


def prompt_lists_hash():
    return hashlib.sha256(json.dumps(prompt_lists(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def batch_rows(prompts, n_temps, n_starts):
    """Row layout of a batch: row = (g * n_temps + j) * n_starts + s for prompt g of the batch, temperature j, start s."""
    g, j, s = np.meshgrid(np.arange(len(prompts)), np.arange(n_temps), np.arange(n_starts), indexing="ij")
    return g.ravel(), j.ravel(), s.ravel()


# ------------------------------------------------------------------------------------------------------ quantizer

def roundf(x):
    """Round half away from zero (C roundf), exactly: trunc(x) is exact and x - trunc(x) is exact in binary floating
    point, so the correction uses the true fractional part."""
    import torch
    t = torch.trunc(x)
    frac = x - t
    return t + torch.where(frac.abs() >= 0.5, torch.sign(x), torch.zeros_like(x))


def quantize_rows(W, qmax, block=BLOCK):
    """Symmetric round-to-nearest block quantization of a weight matrix W (out, in) along its input dimension, in
    float32 arithmetic as ggml's reference quantizers (see the module docstring). Returns the dequantized weights in
    W's dtype and statistics. W must be exactly representable in float32 (bfloat16-valued weights are)."""
    import torch
    if W.dim() != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(W.shape)}")
    out, n = W.shape
    if n % block:
        raise ValueError(f"input dimension {n} is not a multiple of the block size {block}")
    w = W.detach().to(torch.float32)
    if not torch.equal(w.to(W.dtype), W.detach()):
        raise ValueError("weights are not exactly representable in float32")
    wb = w.reshape(out, n // block, block)
    amax = wb.abs().amax(dim=-1, keepdim=True)
    d = amax / qmax
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = roundf(wb * inv).clamp(-qmax, qmax)
    s = d.to(torch.float16).to(torch.float32)
    wq = (q * s).reshape(out, n)
    stats = {"blocks": int(amax.numel()), "subnormal_scales": int(((s > 0) & (s < 2.0 ** -14)).sum()),
             "zero_scale_nonzero_blocks": int(((s == 0) & (amax > 0)).sum())}
    return wq.to(W.dtype), stats


def decoder_linears(model):
    """(name, module) of every nn.Linear inside the decoder blocks."""
    import torch
    out = []
    for i, layer in enumerate(model.model.layers):
        for name, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear):
                out.append((f"layers.{i}.{name}", m))
    return out


class Weights:
    """The decoder-linear weight sets of the variants; use(v) swaps them into the model (no copies)."""

    def __init__(self, model, quantized=("q8", "q4")):
        import torch
        self.torch = torch
        self.lin = decoder_linears(model)
        if len(self.lin) != 7 * len(model.model.layers):
            raise SystemExit(f"expected 7 linears per decoder block, found {len(self.lin)}")
        emb = model.get_input_embeddings().weight
        if any(m.weight.data_ptr() == emb.data_ptr() for _, m in self.lin):
            raise SystemExit("a decoder linear shares storage with the embedding")
        self.sets = {"ref": {n: m.weight.data for n, m in self.lin}}
        self.stats = {"ref": {"sha256": self.digest("ref"), "linears": len(self.lin),
                              "weights": int(sum(m.weight.numel() for _, m in self.lin))}}
        for v in quantized:
            sets, agg = {}, {"blocks": 0, "subnormal_scales": 0, "zero_scale_nonzero_blocks": 0}
            num = den = 0.0
            amax_err = 0.0
            for n, m in self.lin:
                W = m.weight.data
                Wq, st = quantize_rows(W, QMAX[v])
                for k in agg:
                    agg[k] += st[k]
                num += float(((Wq - W) ** 2).sum())
                den += float((W ** 2).sum())
                amax_err = max(amax_err, float((Wq - W).abs().max()))
                sets[n] = Wq
            self.sets[v] = sets
            self.stats[v] = dict(agg, qmax=QMAX[v], block=BLOCK, relative_frobenius_error=math.sqrt(num / den),
                                 max_abs_error=amax_err, sha256=self.digest(v))
        self.current = "ref"

    def digest(self, v):
        h = hashlib.sha256()
        for n, _ in self.lin:
            h.update(self.sets[v][n].detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def use(self, variant):
        key = variant if variant in self.sets else "ref"
        if variant not in ("ref", "noise") and variant not in self.sets:
            raise SystemExit(f"no weights for variant {variant!r}")
        for n, m in self.lin:
            m.weight.data = self.sets[key][n]
        self.current = key


# ------------------------------------------------------------------------------------------ period detection

def min_period(ring, t, tol, qmax):
    """Smallest lag q in 1..qmax (q <= t) closing at step t, per row (0 if none); ring holds state t at index
    t % len(ring). Closing: |x_t - x_{t-q}| < tol max(1, |x_t|)."""
    import torch
    R = ring.shape[0]
    if qmax >= R:
        raise ValueError(f"ring of {R} states cannot test lag {qmax}")
    x = ring[t % R]
    scale = torch.clamp(x.norm(dim=-1), min=1.0)
    out = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for q in range(min(qmax, t), 0, -1):
        close = (x - ring[(t - q) % R]).norm(dim=-1) < tol * scale
        out = torch.where(close, torch.full_like(out, q), out)
    return out


def lag_residual(ring, t, q):
    """|x_t - x_{t-q}| / max(1, |x_t|) per row for per-row lags q (NaN where q = 0)."""
    import torch
    R = ring.shape[0]
    x = ring[t % R]
    rows = torch.arange(x.shape[0], device=x.device)
    prev = ring[(t - torch.clamp(q, min=1)) % R, rows]
    r = (x - prev).norm(dim=-1) / torch.clamp(x.norm(dim=-1), min=1.0)
    return torch.where(q > 0, r, torch.full_like(r, float("nan")))


class Tracker:
    """Online period detection and confirmation for a batch of trajectories (see the module docstring): push the
    state after every step; after steps + confirm pushes, labels() gives the main-phase classes."""

    def __init__(self, x0, steps=STEPS, confirm=CONFIRM, tol=TOL, short=SHORT, long=LONG):
        import torch
        if steps < short or steps + confirm < long:
            raise ValueError("too few steps for the lags tested")
        self.steps, self.confirm, self.tol, self.short, self.long = int(steps), int(confirm), tol, short, long
        self.ring = torch.empty((long + 1,) + tuple(x0.shape), dtype=x0.dtype, device=x0.device)
        self.ring[0] = x0
        self.t = 0
        self.q_detect = self.ok = self.q_end = self.residual = None

    def push(self, x):
        self.t += 1
        t = self.t
        self.ring[t % self.ring.shape[0]] = x
        if t == self.steps:
            self.q_detect = min_period(self.ring, t, self.tol, self.short)
            self.ok = self.q_detect > 0
        elif self.steps < t <= self.steps + self.confirm:
            r = lag_residual(self.ring, t, self.q_detect)
            self.ok = self.ok & (r < self.tol)          # NaN (no lag) compares False
        if t == self.steps + self.confirm:
            self.q_end = min_period(self.ring, t, self.tol, self.long)
            self.residual = lag_residual(self.ring, t, self.q_detect)

    def labels(self):
        if self.t != self.steps + self.confirm:
            raise RuntimeError(f"labels after {self.t} of {self.steps + self.confirm} steps")
        return main_labels(self.q_detect.cpu().numpy(), self.ok.cpu().numpy(), self.q_end.cpu().numpy())


def main_labels(q_detect, ok, q_end, short=SHORT):
    """Classes, sub-labels and periods after the main phase (sub-label pending: needs the twin phase)."""
    q_detect, ok, q_end = np.asarray(q_detect), np.asarray(ok, dtype=bool), np.asarray(q_end)
    n = len(q_detect)
    cls = np.full(n, 2, dtype=np.int8)
    sub = np.full(n, SUB_PENDING, dtype=np.int8)
    period = np.zeros(n, dtype=np.int16)
    verified = ok & (q_detect > 0) & (q_end == q_detect)
    cls[verified], sub[verified], period[verified] = 0, SUB_NONE, q_detect[verified]
    other = ~verified & (q_end > 0)
    sub[other & (q_end > short)] = SUB_LONG
    sub[other & (q_end <= short)] = SUB_LATE_CONV
    period[other] = q_end[other]
    return cls, sub, period


def twin_labels(lam, q_end, lam_min=LAMBDA_MIN):
    """Classes, sub-labels and periods of twin-phase starts from their exponents lam (m offsets, n starts) and the
    smallest closing lag at the end of the twin phase."""
    lam, q_end = np.asarray(lam, dtype=float), np.asarray(q_end)
    chaotic = np.all(lam > lam_min, axis=0) & (q_end == 0)
    cls = np.where(chaotic, 1, 2).astype(np.int8)
    sub = np.where(chaotic, SUB_NONE, np.where(q_end > 0, SUB_LATE_PERIOD, SUB_IRREGULAR)).astype(np.int8)
    period = np.where(~chaotic & (q_end > 0), q_end, 0).astype(np.int16)
    return cls, sub, period


class Twins:
    """A base trajectory and one twin per offset (relative offsets eps |x| along the given unit directions), stepped
    together and renormalized after every step to eps |F(x)| (steer_gain.py). advance(F) applies F to all rows (F maps
    (rows, d) -> (rows, d)) and returns the new base states; exponents() = accumulated log growth / steps."""

    def __init__(self, base, eps_list, directions):
        import torch
        self.torch = torch
        self.n, self.eps = base.shape[0], [float(e) for e in eps_list]
        nb = base.norm(dim=-1, keepdim=True)
        rows = [base]
        self.dirs = [u / u.norm(dim=-1, keepdim=True) for u in directions]
        for e, u in zip(self.eps, self.dirs):
            rows.append(base + e * nb * u)
        self.X = torch.cat(rows)
        self.growth = np.zeros((len(self.eps), self.n))
        self.collapses = np.zeros((len(self.eps), self.n), dtype=np.int64)
        self.steps = 0

    def advance(self, F):
        torch = self.torch
        n = self.n
        Y = F(self.X)
        xb, yb = self.X[:n], Y[:n]
        new = [yb]
        nyb = yb.norm(dim=-1)
        for k, e in enumerate(self.eps):
            sl = slice(n * (k + 1), n * (k + 2))
            dx = (self.X[sl] - xb).norm(dim=-1)
            dF = (Y[sl] - yb).norm(dim=-1)
            zero = dF == 0
            self.collapses[k] += zero.cpu().numpy()
            dF = torch.where(zero, torch.full_like(dF, 1e-300), dF)
            self.growth[k] += torch.log(dF / dx).double().cpu().numpy()
            # a twin that merged with its base exactly (contraction beyond the offset) restarts along its own direction
            renorm = yb + (Y[sl] - yb) * (e * nyb / dF)[:, None]
            new.append(torch.where(zero[:, None], yb + e * nyb[:, None] * self.dirs[k], renorm))
        self.X = torch.cat(new)
        self.steps += 1
        return yb

    def exponents(self):
        return self.growth / max(self.steps, 1)


def twin_directions(meta, d, n_eps=len(EPS)):
    """Unit-variance Gaussian directions for the twins, one generator per start and offset (independent of the
    batch): default_rng((SEED_TWINS, prompt id, temperature index, start index, offset index))."""
    return [np.stack([np.random.default_rng((SEED_TWINS, int(p), int(j), int(s), k)).normal(size=d)
                      for p, j, s in meta]) for k in range(n_eps)]


def canonical_cycle(tokens):
    """Rotation of a token cycle that is smallest as a tuple (the orbit's symbolic name, independent of phase)."""
    tokens = [int(t) for t in tokens]
    if not tokens:
        return ()
    return min(tuple(tokens[i:] + tokens[:i]) for i in range(len(tokens)))


# ---------------------------------------------------------------------------------------------- one-step metrics

def one_step_metrics(z_ref, z_v, temps, E):
    """Per state: KL(p_ref || p_v) at the temperature and at T = 1, total variation at the temperature, top-1
    agreement of the logits, relative one-step map error |F_v - F_ref| / |F_ref| and the reference entropy."""
    import torch
    T = temps[:, None]
    lr, lv = torch.log_softmax(z_ref / T, dim=-1), torch.log_softmax(z_v / T, dim=-1)
    pr, pv = lr.exp(), lv.exp()
    kl = (pr * (lr - lv)).sum(-1)
    lr1, lv1 = torch.log_softmax(z_ref, dim=-1), torch.log_softmax(z_v, dim=-1)
    kl1 = (lr1.exp() * (lr1 - lv1)).sum(-1)
    Fr, Fv = pr @ E, pv @ E
    out = {"kl_T": kl, "kl_1": kl1, "tv_T": 0.5 * (pr - pv).abs().sum(-1),
           "top1": (z_ref.argmax(-1) == z_v.argmax(-1)), "map_err": (Fv - Fr).norm(dim=-1) / Fr.norm(dim=-1),
           "entropy_ref": -(pr * lr).sum(-1)}
    return {k: v.double().cpu().numpy() if k != "top1" else v.cpu().numpy() for k, v in out.items()}


def mean_noise_kl(z, b, temps, sigma):
    """Mean over states of KL(softmax(z/T) || softmax((z + sigma b)/T)) for logits z (n, V), biases b (n, V)."""
    import torch
    T = temps[:, None]
    lr = torch.log_softmax(z / T, dim=-1)
    ln = torch.log_softmax((z + sigma * b) / T, dim=-1)
    return float((lr.exp() * (lr - ln)).sum(-1).mean())


def calibrate_sigma(kl_of_sigma, target, rel=1e-4, iters=200):
    """sigma > 0 with kl_of_sigma(sigma) = target for a function increasing in sigma (the KL along a fixed logit
    direction is convex with its minimum 0 at sigma = 0): bracketing by doubling, then bisection in log sigma."""
    if not target > 0:
        raise ValueError(f"target KL must be positive, got {target}")
    lo, hi = 1e-8, 1.0
    while kl_of_sigma(hi) < target:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            raise SystemExit("noise calibration: no sigma reaches the target KL")
    while kl_of_sigma(lo) > target:
        hi, lo = lo, lo / 2.0
        if lo < 1e-30:
            raise SystemExit("noise calibration: the target KL is below every sigma tried")
    for _ in range(iters):
        if hi / lo < 1 + rel:
            break
        mid = math.sqrt(lo * hi)
        if kl_of_sigma(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def noise_vector(pid, V):
    return np.random.default_rng((SEED_NOISE, int(pid))).standard_normal(V)


# --------------------------------------------------------------------------------------------- statistics

def paired_interval(diff, level=0.95):
    """Mean of paired differences (one per unit) with its t interval."""
    from scipy import stats
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = float(diff.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "ci": [float("nan"), float("nan")]}
    sd = float(diff.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci": [mean - half, mean + half]}


def bootstrap_interval(diff, clusters=None, n_boot=10000, level=0.95, seed=SEED_BOOT):
    """Percentile bootstrap of the mean difference over units, or over clusters of units (resampling whole clusters,
    the mean taken over all units drawn)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        means = diff[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        keys = sorted(set(clusters.tolist()))
        sums = np.array([diff[clusters == k].sum() for k in keys])
        counts = np.array([(clusters == k).sum() for k in keys])
        draw = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


# --------------------------------------------------------------------------------------------- model and loop

def make_loop(threads):
    import explore_llm_soft_loop as esl

    class C12Loop(esl.MultiPrefixSoftLoop):
        """MultiPrefixSoftLoop whose prefix caches are computed once per prompt (prefix_cache) and shared by that
        prompt's rows (use_rows), unsteered."""

        def prefix_cache(self, prefixes):
            from transformers import DynamicCache
            torch = self.torch
            n = len(prefixes)
            lengths = torch.tensor([len(p) for p in prefixes])
            Lp = int(lengths.max())
            pad = self.sm.tok.pad_token_id or 0
            ids = torch.full((n, Lp), pad, dtype=torch.long)
            mask = torch.zeros((n, Lp), dtype=torch.long)
            for r, p in enumerate(prefixes):
                ids[r, Lp - len(p):] = torch.tensor(p)
                mask[r, Lp - len(p):] = 1
            positions = (torch.arange(Lp)[None, :] - (Lp - lengths)[:, None]).clamp(min=0)
            cache = DynamicCache()
            self.sm.steer, self.sm.steer_extra = None, {}
            with torch.no_grad():
                self.model(input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            for li, layer in enumerate(cache.layers):
                if not bool(torch.isfinite(layer.keys).all() and torch.isfinite(layer.values).all()):
                    raise SystemExit(f"non-finite keys or values in the padded prefix cache (layer {li})")
            return {"cache": cache, "lengths": lengths, "Lp": Lp, "mask": mask}

        def use_rows(self, pc, row_prompt):
            from transformers import DynamicCache
            torch = self.torch
            idx = torch.as_tensor(np.asarray(row_prompt), dtype=torch.long)
            cache = DynamicCache()
            for li, layer in enumerate(pc["cache"].layers):
                cache.update(layer.keys[idx], layer.values[idx], li)
            B = len(idx)
            self.cache, self.L0 = cache, pc["Lp"]
            self.prefix_mask, self.row_lengths = pc["mask"][idx], pc["lengths"][idx]
            self._step_inputs = {"attention_mask": torch.cat([self.prefix_mask, torch.ones((B, self.K),
                                                                                          dtype=torch.long)], dim=1),
                                 "position_ids": self.row_lengths[:, None] + torch.arange(self.K)[None]}
            self.steer_rows, self.extra_rows = None, None

        def soft_step(self, X, temps, bias=None):
            """X (B, 1, d) -> (F(X) (B, 1, d), p (B, V)); bias (B, V) is added to the logits (noise variant)."""
            torch = self.torch
            z = self.logits(X)
            if bias is not None:
                z = z + bias
            p = torch.softmax(z / temps[:, None], dim=-1)
            return (p @ self.E)[:, None, :], p

    loop = C12Loop(MODEL, 12, int(threads), 1, "float64", "cpu")
    torch = loop.torch
    if loop.model.lm_head.weight.data_ptr() != loop.model.get_input_embeddings().weight.data_ptr():
        raise SystemExit("expected tied input and output embeddings")
    bad = [n for n, p in loop.model.named_parameters() if p.dtype != torch.float64]
    if bad:
        raise SystemExit(f"parameters not float64: {bad[:5]}")
    return loop


def start_ids(tok, n):
    ids = []
    for s in START_TOKENS[:n]:
        t = tok(s, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise SystemExit(f"start token {s!r} is not one token: {t}")
        ids.append(t[0])
    return ids


def prefix_ids(tok, pid):
    text = tok.apply_chat_template(prompt_messages(pid), add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa
    return {"commit": run("rev-parse", "HEAD"),
            "tool_dirty": bool(run("status", "--porcelain", "--", "tools/c12_compression_fidelity.py"))}


def model_revision():
    import explore_llm_attractors as ela
    repo = Path(ela.CACHE) / ("models--" + ela.MODELS[MODEL].replace("/", "--"))
    try:
        return (repo / "refs" / "main").read_text().strip()
    except OSError:
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def save_npz(path, **arrays):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


class Log:
    def __init__(self, out):
        self.path = out / "progress.jsonl"

    def __call__(self, **rec):
        rec = dict(rec, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        line = json.dumps(rec)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------------------------- closed loop

def run_variant(loop, W, variant, prompts, pcs, cfg, sigma=None, log=print):
    """Closed-loop runs of one variant for a batch of prompts: main phase and twin phase. Returns per-row arrays."""
    torch = loop.torch
    d = loop.E.shape[1]
    nT, nS = len(cfg["log_t"]), cfg["starts"]
    g, j, s = batch_rows(prompts, nT, nS)
    B = len(g)
    pids = np.asarray(prompts)[g]
    temps = torch.as_tensor(10.0 ** np.asarray(cfg["log_t"])[j], dtype=torch.float64)
    W.use(variant)
    pc = pcs["ref" if variant in ("ref", "noise") else variant]
    loop.use_rows(pc, g)
    bias = None
    if variant == "noise":
        if sigma is None:
            raise SystemExit("the noise variant needs a calibration (sigma per temperature)")
        V = loop.model.lm_head.weight.shape[0]
        bvec = torch.as_tensor(np.stack([noise_vector(p, V) for p in prompts]), dtype=torch.float64)
        bias = torch.as_tensor(np.asarray(sigma, dtype=float)[j], dtype=torch.float64)[:, None] * bvec[g]
    sid = start_ids(loop.sm.tok, nS)
    X = loop.E[torch.as_tensor(np.asarray(sid)[s], dtype=torch.long)][:, None, :].clone()
    steps, confirm = cfg["steps"], cfg["confirm"]
    total = steps + confirm
    tracker = Tracker(X[:, 0], steps, confirm, cfg["tol"])
    hashes = [hashlib.sha256() for _ in range(B)]

    def hash_rows(x):
        a = x.detach().cpu().contiguous().numpy()
        for r in range(B):
            hashes[r].update(a[r].tobytes())

    hash_rows(X[:, 0])
    keep = sorted(set(k for k in KEEP_TRANSIENT if k <= total) | set(range(total - N_ATTRACTOR + 1, total + 1)))
    kept = {0: X[:, 0].clone()} if 0 in keep else {}
    lead = np.zeros((total, B), dtype=np.int32)
    tic = time.perf_counter()
    entropy = None
    for t in range(1, total + 1):
        X, p = loop.soft_step(X, temps, bias)
        x = X[:, 0]
        if not bool(torch.isfinite(x).all()):
            raise SystemExit(f"{variant}: non-finite state at step {t}")
        tracker.push(x)
        lead[t - 1] = p.argmax(-1).cpu().numpy()
        hash_rows(x)
        if t in keep:
            kept[t] = x.clone()
        if t == total:
            entropy = (-(p * torch.log(torch.clamp(p, min=1e-300))).sum(-1)).double().cpu().numpy()
    main_seconds = time.perf_counter() - tic
    cls, sub, period = tracker.labels()
    x_end = X[:, 0].clone()
    lam = np.full((len(EPS), B), np.nan)
    q_twin = np.zeros(B, dtype=np.int16)
    collapses = np.zeros((len(EPS), B), dtype=np.int64)
    pending = np.flatnonzero(sub == SUB_PENDING)
    tic = time.perf_counter()
    M = cfg["twin_steps"]
    for c0 in range(0, len(pending), TWIN_CHUNK):
        rows = pending[c0:c0 + TWIN_CHUNK]
        n = len(rows)
        m1 = len(EPS) + 1
        loop.use_rows(pc, np.tile(g[rows], m1))
        temps3 = temps[torch.as_tensor(np.tile(rows, m1))]
        bias3 = None if bias is None else bias[torch.as_tensor(np.tile(rows, m1))]
        dirs = [torch.as_tensor(u, dtype=torch.float64)
                for u in twin_directions(list(zip(pids[rows], j[rows], s[rows])), d)]
        tw = Twins(x_end[torch.as_tensor(rows)], EPS, dirs)
        ring = torch.empty((LONG + 1, n, d), dtype=torch.float64)
        ring[0] = tw.X[:n]

        def F(Y):
            out, _ = loop.soft_step(Y[:, None, :], temps3, bias3)
            return out[:, 0]

        for k in range(1, M + 1):
            xb = tw.advance(F)
            if not bool(torch.isfinite(tw.X).all()):
                raise SystemExit(f"{variant}: non-finite twin state at twin step {k}")
            ring[k % (LONG + 1)] = xb
        lam[:, rows] = tw.exponents()
        collapses[:, rows] = tw.collapses
        q_twin[rows] = min_period(ring, M, cfg["tol"], LONG).cpu().numpy()
    twin_seconds = time.perf_counter() - tic
    if len(pending):
        c2, s2, p2 = twin_labels(lam[:, pending], q_twin[pending])
        cls[pending], sub[pending], period[pending] = c2, s2, p2
    cycles = np.full((B, SHORT), -1, dtype=np.int32)
    for r in np.flatnonzero(cls == 0):
        q = int(period[r])
        cycles[r, :q] = canonical_cycle(lead[total - q:, r])
    log(event="variant_done", variant=variant, prompts=[int(x) for x in prompts], rows=B,
        main_seconds=round(main_seconds, 1), twin_rows=int(len(pending)), twin_seconds=round(twin_seconds, 1),
        counts={CLASS_NAMES[c]: int((cls == c).sum()) for c in range(3)})
    return {"prompt": pids.astype(np.int16), "t_index": j.astype(np.int8), "start": s.astype(np.int8),
            "cls": cls, "sub": sub, "period": period, "q_detect": tracker.q_detect.cpu().numpy().astype(np.int16),
            "confirmed": tracker.ok.cpu().numpy(), "q_end": tracker.q_end.cpu().numpy().astype(np.int16),
            "residual": tracker.residual.double().cpu().numpy(), "lam": lam, "q_twin": q_twin,
            "collapses": collapses, "cycle": cycles, "lead": lead, "x_end": x_end.cpu().numpy(),
            "entropy_end": entropy, "hash": np.array([h.hexdigest() for h in hashes]),
            "kept_steps": np.array(keep, dtype=np.int32),
            "kept": np.stack([kept[k].cpu().numpy() for k in keep]) if variant == "ref" else np.zeros(0),
            "seconds": np.array([main_seconds, twin_seconds])}


def records_of(res, variant, cfg):
    recs = []
    for r in range(len(res["cls"])):
        c, q = int(res["cls"][r]), int(res["period"][r])
        recs.append({"variant": variant, "prompt": int(res["prompt"][r]), "log10_T": cfg["log_t"][int(res["t_index"][r])],
                     "t_index": int(res["t_index"][r]), "start": int(res["start"][r]),
                     "start_token": START_TOKENS[int(res["start"][r])], "class": CLASS_NAMES[c],
                     "sub_label": SUB_NAMES[int(res["sub"][r])], "period": q,
                     "q_detect_256": int(res["q_detect"][r]), "q_end_320": int(res["q_end"][r]),
                     "confirmed": bool(res["confirmed"][r]),
                     "lambda": [None if math.isnan(x) else float(x) for x in res["lam"][:, r]],
                     "q_twin_end": int(res["q_twin"][r]), "cycle": [int(t) for t in res["cycle"][r] if t >= 0],
                     "entropy_end": float(res["entropy_end"][r]), "trajectory_sha256": str(res["hash"][r])})
    return recs


# --------------------------------------------------------------------------------------------- KL pass

def kl_pass(loop, W, prompts, ref, cfg, sigma=None, variants=("q8", "q4"), store=None, log=print):
    """One-step metrics at the reference states kept in ref (run_variant output of the reference). store: dict to
    which the float32 reference logits and noise vectors of the attractor states are appended (calibration)."""
    torch = loop.torch
    kept, steps = ref["kept"], ref["kept_steps"]
    nK, B, d = kept.shape
    total = cfg["steps"] + cfg["confirm"]
    pos = {int(p): i for i, p in enumerate(prompts)}
    if any(int(p) not in pos for p in ref["prompt"]):
        raise SystemExit("KL pass: reference rows do not belong to the batch's prompts")
    g = np.array([pos[int(p)] for p in ref["prompt"]])               # prompt index within the batch, per row
    st = np.repeat(steps, B)
    row = np.tile(np.arange(B), nK)
    states = kept.reshape(nK * B, d)
    attractor = st > total - N_ATTRACTOR
    pcs = {}
    for v in ("ref",) + tuple(variants):
        W.use(v)
        pcs[v] = loop.prefix_cache([prefix_ids(loop.sm.tok, p) for p in prompts])
    V = loop.model.lm_head.weight.shape[0]
    bvec = torch.as_tensor(np.stack([noise_vector(p, V) for p in prompts]), dtype=torch.float64)
    names = tuple(variants) + (("noise",) if sigma is not None else ())
    out = {v: {} for v in names}
    tic = time.perf_counter()
    for c0 in range(0, len(states), KL_CHUNK):
        sl = slice(c0, c0 + KL_CHUNK)
        rr = row[sl]
        X = torch.as_tensor(states[sl], dtype=torch.float64)[:, None, :]
        temps = torch.as_tensor(10.0 ** np.asarray(cfg["log_t"])[ref["t_index"][rr]], dtype=torch.float64)
        W.use("ref")
        loop.use_rows(pcs["ref"], g[rr])
        z_ref = loop.logits(X)
        res = {}
        for v in variants:
            W.use(v)
            loop.use_rows(pcs[v], g[rr])
            res[v] = one_step_metrics(z_ref, loop.logits(X), temps, loop.E)
        if sigma is not None:
            sg = torch.as_tensor(np.asarray(sigma, dtype=float)[ref["t_index"][rr]], dtype=torch.float64)
            res["noise"] = one_step_metrics(z_ref, z_ref + sg[:, None] * bvec[g[rr]], temps, loop.E)
        for v in names:
            for k, a in res[v].items():
                out[v].setdefault(k, []).append(a)
        if store is not None:
            a = attractor[sl]
            if a.any():
                ia = torch.as_tensor(np.flatnonzero(a))
                store.setdefault("z", []).append(z_ref[ia].float())
                store.setdefault("prompt", []).append(ref["prompt"][rr][a])
                store.setdefault("t_index", []).append(ref["t_index"][rr][a])
    W.use("ref")
    arrays = {"prompt": ref["prompt"][row], "t_index": ref["t_index"][row], "start": ref["start"][row],
              "step": st.astype(np.int16), "attractor": attractor}
    for v in names:
        for k, a in out[v].items():
            arrays[f"{v}__{k}"] = np.concatenate(a)
    log(event="kl_done", prompts=[int(x) for x in prompts], states=int(len(states)),
        seconds=round(time.perf_counter() - tic, 1),
        mean_kl_attractor={v: float(arrays[f"{v}__kl_T"][attractor].mean()) for v in names})
    return arrays


# --------------------------------------------------------------------------------------------- subcommands

def settings_of(args):
    return {"model": MODEL, "log_t": [float(x) for x in (args.log_t if args.log_t else LOG_T)],
            "starts": int(args.starts), "steps": int(args.steps), "confirm": int(args.confirm),
            "twin_steps": int(args.twin_steps), "tol": TOL, "eps": list(EPS), "lambda_min": LAMBDA_MIN,
            "short": SHORT, "long": LONG, "keep_transient": list(KEEP_TRANSIENT), "n_attractor": N_ATTRACTOR,
            "qmax": QMAX, "block": BLOCK, "seeds": {"twins": SEED_TWINS, "noise": SEED_NOISE, "boot": SEED_BOOT},
            "prompts_per_batch": PROMPTS_PER_BATCH[args.set], "prompt_lists_sha256": prompt_lists_hash()}


def open_out(args, settings):
    out = OUT_ROOT / args.out
    (out / "batches").mkdir(parents=True, exist_ok=True)
    sp = out / "settings.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old != settings:
            raise SystemExit(f"{sp} holds other settings; use a new --out (attempts are never overwritten)")
    else:
        write_json(sp, settings)
    return out


def write_manifest(out, args, extra):
    import torch
    import transformers
    man = {"argv": sys.argv, "args": vars(args), "git": git_state(), "model": MODEL, "model_revision": model_revision(),
           "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
           "threads": torch.get_num_threads(), "host": platform.node(),
           "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "prompt_lists": prompt_lists(), "prompt_lists_sha256": prompt_lists_hash(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    man.update(extra)
    write_json(out / f"manifest-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json", man)
    return man


def batch_indices(out):
    """Indices k of the batch files bNN-*.npz present in a run directory."""
    return sorted({int(f.name[1:3]) for f in (out / "batches").glob("b[0-9][0-9]-*.npz")})


def load_variant(out, k, v):
    path = out / "batches" / f"b{k:02d}-{v}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {name: z[name] for name in z.files}


def cmd_run(args):
    settings = settings_of(args)
    out = open_out(args, settings)
    log = Log(out)
    loop = make_loop(args.threads)
    quantized = tuple(v for v in ("q8", "q4") if v in args.variants or args.kl)
    W = Weights(loop.model, quantized)
    sigma, calib = None, None
    if "noise" in args.variants or args.kl:
        if args.noise_sigma is not None:
            sigma = [float(args.noise_sigma)] * len(settings["log_t"])
        elif args.calibration:
            calib = json.loads(Path(args.calibration).read_text())
            if calib["log_t"] != settings["log_t"]:
                raise SystemExit("calibration temperatures differ from the run's")
            sigma = calib["sigma"]
        elif "noise" in args.variants:
            raise SystemExit("the noise variant needs --calibration or --noise-sigma")
    write_manifest(out, args, {"settings": settings, "weights": W.stats, "sigma": sigma,
                               "calibration_sha256": sha256_file(args.calibration) if args.calibration else None})
    all_batches = batches_of(args.set) if not args.prompt_ids else [list(args.prompt_ids)]
    order = list(range(len(all_batches))) if args.batches is None else list(args.batches)
    log(event="start", set=args.set, batches=order, variants=list(args.variants), kl=bool(args.kl),
        weights={v: {k: x for k, x in st.items() if k != "sha256"} for v, st in W.stats.items()})
    tic0 = time.perf_counter()
    done_seconds = []
    for n_done, k in enumerate(order):
        if args.budget_hours and n_done % ROUND_BATCHES[args.set] == 0 and done_seconds:
            elapsed = time.perf_counter() - tic0
            need = ROUND_BATCHES[args.set] * float(np.mean(done_seconds))
            if elapsed + need > args.budget_hours * 3600:
                log(event="budget_stop", before_batch=k, elapsed_s=round(elapsed), projected_round_s=round(need))
                break
        prompts = all_batches[k]
        tb = time.perf_counter()
        pcs = {}
        results = {}
        for v in args.variants:
            have = load_variant(out, k, v)
            if have is not None:
                results[v] = have
                log(event="variant_exists", batch=k, variant=v)
                continue
            key = "ref" if v in ("ref", "noise") else v
            if key not in pcs:
                W.use(key)
                pcs[key] = loop.prefix_cache([prefix_ids(loop.sm.tok, p) for p in prompts])
            res = run_variant(loop, W, v, prompts, pcs, settings, sigma, log)
            res["batch_prompts"] = np.asarray(prompts, dtype=np.int16)
            save_npz(out / "batches" / f"b{k:02d}-{v}.npz", **res)
            write_json(out / "batches" / f"b{k:02d}-{v}.json", {"batch": k, "variant": v, "prompts": prompts,
                                                                "records": records_of(res, v, settings)})
            results[v] = res
        if args.kl and not (out / "batches" / f"b{k:02d}-kl.npz").exists():
            if "ref" not in results:
                raise SystemExit("the KL pass needs the reference run of the batch")
            arrays = kl_pass(loop, W, prompts, results["ref"], settings, sigma, log=log)
            save_npz(out / "batches" / f"b{k:02d}-kl.npz", **arrays)
        done_seconds.append(time.perf_counter() - tb)
        log(event="batch_done", batch=k, prompts=prompts, seconds=round(done_seconds[-1], 1))
    W.use("ref")
    log(event="end", seconds=round(time.perf_counter() - tic0, 1))


def cmd_calibrate(args):
    """KL pass on the development reference states and the noise scale per temperature."""
    import torch
    out = OUT_ROOT / args.out
    settings = json.loads((out / "settings.json").read_text())
    existing = [f.name for f in (out / "batches").glob("b*-kl.npz")] + \
        (["calibration.json"] if (out / "calibration.json").exists() else [])
    if existing:
        raise SystemExit(f"calibrate: {existing} exist already in {out}; raw outputs are never overwritten")
    log = Log(out)
    loop = make_loop(args.threads)
    W = Weights(loop.model, ("q8", "q4"))
    write_manifest(out, args, {"settings": settings, "weights": W.stats, "subcommand": "calibrate"})
    all_batches = batches_of("dev")
    store = {}
    per_batch = []
    for k, prompts in enumerate(all_batches):
        ref = load_variant(out, k, "ref")
        if ref is None:
            log(event="calibrate_missing_batch", batch=k)
            continue
        arrays = kl_pass(loop, W, prompts, ref, settings, None, store=store, log=log)
        save_npz(out / "batches" / f"b{k:02d}-kl.npz", **arrays)
        per_batch.append(arrays)
    if not per_batch:
        raise SystemExit("calibrate: no development reference batches found")
    cat = {key: np.concatenate([a[key] for a in per_batch]) for key in per_batch[0]}
    z = torch.cat(store["z"])
    sp, sj = np.concatenate(store["prompt"]), np.concatenate(store["t_index"])
    V = z.shape[1]
    bvec = {int(p): torch.as_tensor(noise_vector(p, V), dtype=torch.float64) for p in np.unique(sp)}
    nT = len(settings["log_t"])
    sigma, target, achieved, q4mean, counts = [], [], [], [], []
    for jt in range(nT):
        m = cat["attractor"] & (cat["t_index"] == jt)
        tq8 = float(cat["q8__kl_T"][m].mean())
        idx = np.flatnonzero(sj == jt)
        if len(idx) != int(m.sum()):
            raise SystemExit("calibrate: stored logits do not match the attractor states")
        zj = z[torch.as_tensor(idx)].double()
        bj = torch.stack([bvec[int(p)] for p in sp[idx]])
        tj = torch.full((len(idx),), 10.0 ** settings["log_t"][jt], dtype=torch.float64)
        f = lambda sgm: mean_noise_kl(zj, bj, tj, sgm)  # noqa: E731
        sg = calibrate_sigma(f, tq8)
        sigma.append(sg)
        target.append(tq8)
        achieved.append(f(sg))
        q4mean.append(float(cat["q4__kl_T"][m].mean()))
        counts.append(int(m.sum()))
        log(event="calibrated", t_index=jt, log10_T=settings["log_t"][jt], sigma=sg, target_kl_q8=tq8,
            achieved=achieved[-1], q4_kl=q4mean[-1], states=counts[-1])
    calib = {"log_t": settings["log_t"], "sigma": sigma, "target_mean_kl_q8": target,
             "achieved_mean_kl_noise_float32_logits": achieved, "mean_kl_q4": q4mean, "states": counts,
             "definition": "sigma_T: mean over the development reference attractor states at T of KL(p_ref || "
                           "softmax((z_ref + sigma_T b_p)/T)) equals the mean KL(p_ref || p_q8) at those states; "
                           "z_ref stored in float32 for the bisection, KL in float64",
             "batches": [k for k in range(len(all_batches)) if load_variant(out, k, "ref") is not None],
             "weights": W.stats, "git": git_state()}
    write_json(out / "calibration.json", calib)
    log(event="calibration_written", sigma=sigma)


def cmd_precision(args):
    """Precision probe: one-step central-difference ratios |F(x + h u) - F(x - h u)| / (2h) over h, for the reference
    and the q4 weights, at states reached after 32 steps; and the difference between two evaluations of F(x) at
    different batch positions."""
    import torch
    out = OUT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    loop = make_loop(args.threads)
    W = Weights(loop.model, ("q4",))
    hs = [10.0 ** -k for k in range(1, 11)]
    prompts = [0]
    t_idx = [2, 5]
    rec = {"h": hs, "cells": []}
    rng = np.random.default_rng(0)
    for v in ("ref", "q4"):
        W.use(v)
        pc = loop.prefix_cache([prefix_ids(loop.sm.tok, p) for p in prompts])
        temps = torch.as_tensor([10.0 ** LOG_T[j] for j in t_idx], dtype=torch.float64)
        loop.use_rows(pc, [0] * len(t_idx))
        X = loop.E[torch.as_tensor([start_ids(loop.sm.tok, 1)[0]] * len(t_idx))][:, None, :].clone()
        for _ in range(32):
            X, _ = loop.soft_step(X, temps)
        n = len(t_idx)
        u = torch.as_tensor(rng.normal(size=(n, X.shape[-1])), dtype=torch.float64)
        u = u / u.norm(dim=-1, keepdim=True)
        rows = []
        for h in hs:
            rows += [X[:, 0] + h * u, X[:, 0] - h * u]
        rows += [X[:, 0], X[:, 0]]
        R = len(rows)
        Xall = torch.cat(rows)[:, None, :]
        loop.use_rows(pc, [0] * (n * R))
        Y, _ = loop.soft_step(Xall, temps.repeat(R))
        Y = Y[:, 0]
        for c in range(n):
            ratio = [float((Y[n * 2 * k + c] - Y[n * (2 * k + 1) + c]).norm() / (2 * h)) for k, h in enumerate(hs)]
            rep = float((Y[n * (R - 2) + c] - Y[n * (R - 1) + c]).abs().max())
            rec["cells"].append({"variant": v, "log10_T": LOG_T[t_idx[c]], "central_ratio": ratio,
                                 "rel_dev_from_h1e-5": [abs(r / ratio[4] - 1) for r in ratio],
                                 "repeat_diff_max_abs": rep})
            print(json.dumps(rec["cells"][-1]), flush=True)
    W.use("ref")
    rec["git"] = git_state()
    write_json(out / "precision.json", rec)


def cmd_compare(args):
    """Negative control: the batches present in both run directories must agree bitwise (every array)."""
    a, b = OUT_ROOT / args.a, OUT_ROOT / args.b
    rows = []
    ok = True
    for pb in sorted((b / "batches").glob("b*-*.npz")):
        name = pb.name
        if name.endswith("-kl.npz") and not args.include_kl:
            continue
        pa = a / "batches" / name
        if not pa.exists():
            continue
        za, zb = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        diff = []
        for key in sorted(set(za.files) | set(zb.files)):
            if key == "seconds":
                continue
            if key not in za.files or key not in zb.files:
                diff.append(key)
                continue
            xa, xb = za[key], zb[key]
            if xa.shape != xb.shape or xa.dtype != xb.dtype or xa.tobytes() != xb.tobytes():
                diff.append(key)
        same = not diff
        ok &= same
        rows.append({"file": name, "bitwise_identical": same, "differing_arrays": diff})
        print(json.dumps(rows[-1]), flush=True)
    if not rows:
        raise SystemExit("compare: no common batch files")
    res = {"a": str(a), "b": str(b), "files": rows, "all_identical": bool(ok), "git": git_state()}
    write_json(OUT_ROOT / args.b / f"compare-{Path(args.a).name}.json", res)
    print(json.dumps({"all_identical": bool(ok), "files": len(rows)}))


def gate_rows(res, cfg, n_prompts):
    """Row-level gate of one batch-variant output (every row read): returns the list of failed checks and counts."""
    fails = []
    n = n_prompts * len(cfg["log_t"]) * cfg["starts"]
    cls, sub, period, lam = res["cls"], res["sub"], res["period"], res["lam"]
    if len(cls) != n:
        fails.append(f"{len(cls)} rows, expected {n}")
    if not np.isfinite(res["x_end"]).all():
        fails.append("non-finite final states")
    if np.any(sub == SUB_PENDING):
        fails.append("rows left pending")
    per = cls == 0
    if per.any():
        if not np.all(res["residual"][per] < cfg["tol"]):
            fails.append("periodic row with closure residual >= tol")
        if not np.all(res["confirmed"][per]):
            fails.append("periodic row not confirmed")
        if not np.all((period[per] >= 1) & (period[per] <= SHORT) & (period[per] == res["q_detect"][per])):
            fails.append("periodic row with inconsistent period")
    twin = (res["q_end"] == 0) & ~per
    if np.any(np.isnan(lam[:, twin])):
        fails.append("twin-phase row without exponents")
    if np.any(~np.isnan(lam[:, ~twin])):
        fails.append("exponents on a row without twin phase")
    ch = cls == 1
    if ch.any() and not (np.all(lam[:, ch] > LAMBDA_MIN) and np.all(res["q_twin"][ch] == 0)):
        fails.append("chaotic row violating the chaos rule")
    un = cls == 2
    if un.any() and not np.all(np.isin(sub[un], (SUB_LONG, SUB_LATE_CONV, SUB_LATE_PERIOD, SUB_IRREGULAR))):
        fails.append("unresolved row without a sub-label")
    info = {"classes": {CLASS_NAMES[c]: int((cls == c).sum()) for c in range(3)},
            "sub_labels": {SUB_NAMES[k]: int((sub == k).sum()) for k in range(1, 5)},
            "twin_rows": int(twin.sum()), "twin_collapses": int(res["collapses"].sum()),
            "max_residual_periodic": float(np.nanmax(res["residual"][per])) if per.any() else None}
    return fails, info


def kl_stats(a, mask):
    """Statistics of one variant's one-step metrics over the states in mask."""
    if not mask.any():
        return None
    kl, kl1 = a["kl_T"][mask], a["kl_1"][mask]
    q = lambda x, f: float(np.quantile(x, f))  # noqa: E731
    return {"states": int(mask.sum()), "kl_T_mean": float(kl.mean()), "kl_T_median": q(kl, 0.5),
            "kl_T_p90": q(kl, 0.9), "kl_T_p99": q(kl, 0.99), "kl_T_max": float(kl.max()),
            "kl_1_mean": float(kl1.mean()), "kl_1_median": q(kl1, 0.5), "tv_T_mean": float(a["tv_T"][mask].mean()),
            "top1_agreement": float(a["top1"][mask].mean()), "map_err_median": q(a["map_err"][mask], 0.5),
            "map_err_p90": q(a["map_err"][mask], 0.9), "map_err_max": float(a["map_err"][mask].max())}


def file_list(base):
    out = []
    for f in sorted(base.rglob("*")):
        if f.is_file() and f.name != "summary.json" and not f.name.endswith(".tmp") and ".tmp." not in f.name:
            out.append({"path": str(f.relative_to(OUT_ROOT)), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    return out


def manifests(base):
    out = []
    for f in sorted(base.glob("manifest-*.json")):
        m = json.loads(f.read_text())
        out.append({"file": f.name, "git": m.get("git"), "argv": m.get("argv"), "started_utc": m.get("started_utc"),
                    "threads": m.get("threads"), "model_revision": m.get("model_revision")})
    return out


def interval_block(diff, clusters):
    r = paired_interval(diff)
    r["bootstrap_ci"] = bootstrap_interval(diff) if len(diff) > 1 else None
    r["cluster_bootstrap_ci"] = bootstrap_interval(diff, clusters) if len(set(clusters.tolist())) > 1 else None
    r["clusters"] = int(len(set(clusters.tolist())))
    return r


def summarize(args):
    held = OUT_ROOT / args.heldout
    cfg = json.loads((held / "settings.json").read_text())
    nT = len(cfg["log_t"])
    gates, data, kls = [], {v: [] for v in VARIANTS}, []
    for k in batch_indices(held):
        res = {v: load_variant(held, k, v) for v in VARIANTS}
        klp = held / "batches" / f"b{k:02d}-kl.npz"
        saved = {tuple(int(x) for x in r["batch_prompts"]) for r in res.values() if r is not None}
        if len(saved) != 1:
            gates.append({"batch": k, "complete": False, "pairing": f"variants disagree on the batch's prompts {saved}"})
            continue
        prompts = list(saved.pop())
        row = {"batch": k, "prompts": prompts, "kl_pass": klp.exists()}
        ok = klp.exists()
        for v, r in res.items():
            if r is None:
                row[v], ok = {"status": "missing"}, False
                continue
            fails, info = gate_rows(r, cfg, len(prompts))
            row[v] = {"status": "pass" if not fails else "fail", "failed_checks": fails, **info}
            ok &= not fails
        if ok:
            for key in ("prompt", "t_index", "start"):
                if not all(np.array_equal(res["ref"][key], res[v][key]) for v in VARIANTS):
                    row["pairing"], ok = f"rows not aligned ({key})", False
        row["complete"] = bool(ok)
        gates.append(row)
        if ok:
            for v in VARIANTS:
                data[v].append(res[v])
            z = np.load(klp, allow_pickle=False)
            kls.append({name: z[name] for name in z.files})
    if not data["ref"]:
        raise SystemExit("summarize: no complete held-out batch")
    cat = {v: {key: np.concatenate([r[key] for r in data[v]]) for key in
               ("prompt", "t_index", "start", "cls", "sub", "period", "cycle", "lead")}
           for v in VARIANTS}
    for v in VARIANTS:
        cat[v]["lead"] = np.concatenate([r["lead"] for r in data[v]], axis=1)
    ref = cat["ref"]
    pids = np.array(sorted(set(ref["prompt"].tolist())))
    clusters = pids // 8
    rows_of = {int(p): np.flatnonzero(ref["prompt"] == p) for p in pids}
    share = {v: np.array([np.mean(cat[v]["cls"][rows_of[int(p)]] == 0) for p in pids]) for v in VARIANTS}

    def sens(v, kind):
        """Sensitivity shares: le8_by_576 counts also the starts that reach a period <= 8 only after step 256
        (late_convergence at step 320, late_period by the end of the twin phase); le64 counts every start with a
        closing lag <= 64 at step 320 or at the end of the twin phase."""
        c, sb, q = cat[v]["cls"], cat[v]["sub"], cat[v]["period"]
        late = np.isin(sb, (SUB_LATE_CONV, SUB_LATE_PERIOD))
        ok = (c == 0) | (late & (q <= SHORT)) if kind == "le8_by_576" else \
            (c == 0) | np.isin(sb, (SUB_LONG, SUB_LATE_CONV, SUB_LATE_PERIOD))
        return np.array([np.mean(ok[rows_of[int(p)]]) for p in pids])

    sensitivity = {}
    for kind in ("le8_by_576", "le64"):
        sh = {v: sens(v, kind) for v in VARIANTS}
        sensitivity[kind] = {f"{v}_minus_ref": dict(share_ref_mean=float(sh["ref"].mean()),
                                                    share_variant_mean=float(sh[v].mean()),
                                                    **paired_interval(sh[v] - sh["ref"])) for v in ("q4", "q8", "noise")}
    endpoints = {}
    for v in ("q4", "q8", "noise"):
        diff = share[v] - share["ref"]
        e = interval_block(diff, clusters)
        e["share_ref_mean"], e["share_variant_mean"] = float(share["ref"].mean()), float(share[v].mean())
        e["prompts_with_any_shift"] = int(np.sum(diff != 0))
        e["abs_mean_at_least_10_points"] = bool(abs(e["mean"]) >= 0.10)
        e["ci_excludes_zero"] = bool(e["ci"][0] > 0 or e["ci"][1] < 0)
        e["ci_beyond_10_points"] = bool(e["ci"][0] > 0.10 or e["ci"][1] < -0.10)
        e["ci_within_10_points"] = bool(e["ci"][0] > -0.10 and e["ci"][1] < 0.10)
        endpoints[f"{v}_minus_ref"] = e
    per_T = {}
    for v in VARIANTS:
        per_T[v] = []
        for jt in range(nT):
            m = cat[v]["t_index"] == jt
            c = cat[v]["cls"][m]
            per = cat[v]["period"][m][c == 0]
            per_T[v].append({"log10_T": cfg["log_t"][jt], "n": int(m.sum()),
                             "counts": {CLASS_NAMES[i]: int((c == i).sum()) for i in range(3)},
                             "shares": {CLASS_NAMES[i]: float((c == i).mean()) for i in range(3)},
                             "sub_labels": {SUB_NAMES[i]: int((cat[v]["sub"][m] == i).sum()) for i in range(1, 5)},
                             "periods": {str(q): int((per == q).sum()) for q in sorted(set(per.tolist()))}})
    share_T = {v: np.array([[np.mean(cat[v]["cls"][rows_of[int(p)]][cat[v]["t_index"][rows_of[int(p)]] == jt] == 0)
                             for jt in range(nT)] for p in pids]) for v in VARIANTS}
    per_T_diff = {v: [dict(log10_T=cfg["log_t"][jt], **paired_interval(share_T[v][:, jt] - share_T["ref"][:, jt]))
                      for jt in range(nT)] for v in ("q4", "q8", "noise")}
    # class or period changes, orbit changes, trajectory divergence (greedy-output agreement)
    changes = {}
    rc, rp, ry = ref["cls"], ref["period"], ref["cycle"]
    outcome = np.where(rc == 0, rp, -1 - rc.astype(int))
    cell = ref["prompt"].astype(int) * nT + ref["t_index"].astype(int)
    mixed = np.zeros(len(rc), dtype=bool)
    comp = {}
    for c_ in np.unique(cell):
        m = cell == c_
        mixed[m] = len(set(outcome[m].tolist())) > 1
        comp[int(c_)] = sorted(outcome[m].tolist())
    adjacent = np.array([any(comp.get(int(c_) + dj) is not None and comp[int(c_) + dj] != comp[int(c_)]
                             and (int(c_) + dj) // nT == int(c_) // nT for dj in (-1, 1)) for c_ in cell])
    for v in ("q4", "q8", "noise"):
        vc, vp, vy = cat[v]["cls"], cat[v]["period"], cat[v]["cycle"]
        cls_change = vc != rc
        per_change = (vc == 0) & (rc == 0) & (vp != rp)
        changed = cls_change | per_change
        orbit = (vc == 0) & (rc == 0) & (vp == rp) & np.any(vy != ry, axis=1)
        agree = (cat[v]["lead"] == ref["lead"]).mean(axis=0)
        first = np.where((cat[v]["lead"] != ref["lead"]).any(axis=0), (cat[v]["lead"] != ref["lead"]).argmax(axis=0) + 1,
                         0)
        per_prompt = np.array([changed[rows_of[int(p)]].mean() for p in pids])
        changes[v] = {
            "class_or_period_changed": {"count": int(changed.sum()), "of": int(len(changed)),
                                        "pooled_share": float(changed.mean()),
                                        "prompt_mean": interval_block(per_prompt, clusters)},
            "class_changed": int(cls_change.sum()), "period_changed_both_periodic": int(per_change.sum()),
            "same_period_other_orbit": int(orbit.sum()),
            "both_periodic": int(((vc == 0) & (rc == 0)).sum()),
            "transitions": {f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}": int(((rc == a) & (vc == b)).sum())
                            for a in range(3) for b in range(3) if a != b},
            "by_temperature": [{"log10_T": cfg["log_t"][jt], "changed": int(changed[ref["t_index"] == jt].sum()),
                                "of": int((ref["t_index"] == jt).sum())} for jt in range(nT)],
            "prompts_with_change": int(np.sum(per_prompt > 0)),
            "local_fragility": {
                "rows_in_mixed_reference_cells": float(mixed.mean()),
                "changes_in_mixed_reference_cells": float(mixed[changed].mean()) if changed.any() else None,
                "rows_in_cells_differing_from_adjacent_T": float(adjacent.mean()),
                "changes_in_cells_differing_from_adjacent_T": float(adjacent[changed].mean()) if changed.any() else None},
            "greedy_agreement": {"mean_fraction_of_steps": float(agree.mean()),
                                 "rows_identical_lead_sequence": int(np.sum(first == 0)),
                                 "median_first_divergence_step": float(np.median(first[first > 0])) if np.any(first > 0)
                                 else None}}
    # one-step metrics
    kcat = {key: np.concatenate([a[key] for a in kls]) for key in kls[0]}
    att = kcat["attractor"]
    kl_summary = {}
    for v in ("q8", "q4", "noise"):
        a = {k[len(v) + 2:]: x for k, x in kcat.items() if k.startswith(v + "__")}
        if not a:
            continue
        kl_summary[v] = {"attractor": kl_stats(a, att), "transient": kl_stats(a, ~att),
                         "attractor_by_T": [dict(log10_T=cfg["log_t"][jt], **kl_stats(a, att & (kcat["t_index"] == jt)))
                                            for jt in range(nT)],
                         "attractor_prompt_mean_kl_T": float(np.mean([a["kl_T"][att & (kcat["prompt"] == p)].mean()
                                                                      for p in pids]))}
    noise_match = None
    if "noise" in kl_summary and "q8" in kl_summary:
        noise_match = [{"log10_T": cfg["log_t"][jt],
                        "mean_kl_noise_over_q8": kl_summary["noise"]["attractor_by_T"][jt]["kl_T_mean"] /
                        kl_summary["q8"]["attractor_by_T"][jt]["kl_T_mean"]} for jt in range(nT)]
    # sensitivity of the endpoint to a real change: the reference's shift between adjacent temperatures
    adjacent_T = [dict(from_log10_T=cfg["log_t"][jt], to_log10_T=cfg["log_t"][jt + 1],
                       **paired_interval(share_T["ref"][:, jt + 1] - share_T["ref"][:, jt])) for jt in range(nT - 1)]
    # development: reference shares and calibration
    dev = None
    dpath = OUT_ROOT / args.dev
    if (dpath / "settings.json").exists():
        dres = [r for r in (load_variant(dpath, k, "ref") for k in batch_indices(dpath)) if r is not None]
        dcfg = json.loads((dpath / "settings.json").read_text())
        dgates = []
        for r in dres:
            fails, info = gate_rows(r, dcfg, len(r["batch_prompts"]))
            dgates.append({"prompts": [int(x) for x in r["batch_prompts"]], "status": "pass" if not fails else "fail",
                           "failed_checks": fails, **info})
        dc = np.concatenate([r["cls"] for r in dres]) if dres else np.zeros(0)
        dj = np.concatenate([r["t_index"] for r in dres]) if dres else np.zeros(0)
        cal = json.loads((dpath / "calibration.json").read_text()) if (dpath / "calibration.json").exists() else None
        dkl = [np.load(f, allow_pickle=False) for f in sorted((dpath / "batches").glob("b*-kl.npz"))]
        dkls = None
        if dkl:
            dk = {key: np.concatenate([z[key] for z in dkl]) for key in dkl[0].files}
            dkls = {v: [dict(log10_T=dcfg["log_t"][jt], **kl_stats({k[len(v) + 2:]: x for k, x in dk.items()
                                                                     if k.startswith(v + "__")},
                                                                    dk["attractor"] & (dk["t_index"] == jt)))
                        for jt in range(len(dcfg["log_t"]))] for v in ("q8", "q4")}
        same = {k: v for k, v in dcfg.items() if k != "prompts_per_batch"} == \
            {k: v for k, v in cfg.items() if k != "prompts_per_batch"}
        dev = {"settings_equal_heldout_except_batch_size": same, "gates": dgates,
               "reference_by_T": [{"log10_T": dcfg["log_t"][jt], "n": int((dj == jt).sum()),
                                   "counts": {CLASS_NAMES[i]: int(((dc == i) & (dj == jt)).sum()) for i in range(3)}}
                                  for jt in range(len(dcfg["log_t"]))],
               "calibration": cal and {k: cal[k] for k in ("log_t", "sigma", "target_mean_kl_q8",
                                                            "achieved_mean_kl_noise_float32_logits", "mean_kl_q4",
                                                            "states", "definition", "batches")},
               "kl_attractor_by_T": dkls, "manifests": manifests(dpath)}
    negative = []
    for rep in args.repeat:
        for f in sorted((OUT_ROOT / rep).glob("compare-*.json")):
            c = json.loads(f.read_text())
            negative.append({"repeat_dir": rep, "file": f.name, "compared_with": c["a"], "all_identical": c["all_identical"],
                             "files": c["files"]})
    summary = {
        "experiment": "C12-2 compression fidelity (exploratory)", "model": MODEL, "settings": cfg,
        "prompt_lists_sha256": prompt_lists_hash(), "prompt_lists": prompt_lists(),
        "storage_root": str(OUT_ROOT), "heldout_dir": args.heldout, "dev_dir": args.dev,
        "heldout_prompts_run": [int(p) for p in pids], "heldout_prompts_planned": len(prompt_order("heldout")),
        "units": "prompts (paired over variants); clusters = system prompts",
        "gates": gates, "class_shares_by_T": per_T, "share_difference_by_T": per_T_diff,
        "primary_endpoint": {"definition": "mean over held-out prompts of share_q4 - share_ref; share = fraction of the "
                                           "prompt's starts (all temperatures) in verified period-<=8 orbits",
                             **endpoints["q4_minus_ref"]},
        "secondary_endpoints": {k: e for k, e in endpoints.items() if k != "q4_minus_ref"},
        "sensitivity_endpoints": sensitivity,
        "changes": changes, "one_step": kl_summary, "noise_kl_match_heldout": noise_match,
        "endpoint_sensitivity_adjacent_T_reference": adjacent_T,
        "negative_control": negative, "development": dev, "manifests": manifests(held),
        "weights": json.loads(sorted(held.glob("manifest-*.json"))[0].read_text()).get("weights"),
        "git_summarize": git_state(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    summary["raw_files"] = file_list(held) + (file_list(dpath) if dev else []) + \
        [x for rep in args.repeat for x in file_list(OUT_ROOT / rep)]
    write_json(held / "summary.json", summary)
    if args.receipt:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        write_json(Path(args.receipt), summary)
    e = endpoints["q4_minus_ref"]
    print(json.dumps({"prompts": int(len(pids)), "primary_q4_minus_ref": e["mean"], "ci": e["ci"],
                      "q8": endpoints["q8_minus_ref"]["mean"], "noise": endpoints["noise_minus_ref"]["mean"]}))


def cmd_summarize(args):
    summarize(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="closed-loop runs (and the KL pass) for batches of prompts")
    r.add_argument("--set", choices=("dev", "heldout"), default="dev")
    r.add_argument("--prompt-ids", type=int, nargs="+", help="one batch of these prompt ids instead of --set")
    r.add_argument("--batches", type=int, nargs="+", help="batch indices of the set, in this order (default all)")
    r.add_argument("--variants", nargs="+", default=["ref"], choices=VARIANTS)
    r.add_argument("--kl", action="store_true", help="KL pass after each batch (needs ref among the variants)")
    r.add_argument("--calibration", help="calibration.json of the development run (noise sigma per temperature)")
    r.add_argument("--noise-sigma", type=float, help="constant noise sigma (smoke tests only)")
    r.add_argument("--log-t", type=float, nargs="+", help="temperatures (log10 T) instead of the default grid")
    r.add_argument("--starts", type=int, default=8)
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--confirm", type=int, default=CONFIRM)
    r.add_argument("--twin-steps", type=int, default=TWIN_STEPS)
    r.add_argument("--budget-hours", type=float, help="stop between rounds when the next round would exceed this")
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--out", required=True)
    c = sub.add_parser("calibrate", help="KL pass on the development reference states and the noise calibration")
    c.add_argument("--out", required=True)
    c.add_argument("--threads", type=int, default=6)
    p = sub.add_parser("precision", help="central-difference precision probe of the float64 map")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=6)
    q = sub.add_parser("compare", help="negative control: bitwise comparison of two run directories")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--include-kl", action="store_true")
    s = sub.add_parser("summarize", help="endpoints and the compact receipt")
    s.add_argument("--dev", default="dev")
    s.add_argument("--heldout", default="heldout")
    s.add_argument("--repeat", nargs="*", default=[])
    s.add_argument("--receipt", default=str(ROOT / "docs" / "pi" / "c12-2-summary.json"))
    args = ap.parse_args()
    if args.cmd == "run" and not (1 <= args.starts <= len(START_TOKENS)):
        raise SystemExit(f"--starts must lie in 1..{len(START_TOKENS)}")
    {"run": cmd_run, "calibrate": cmd_calibrate, "precision": cmd_precision, "compare": cmd_compare,
     "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
