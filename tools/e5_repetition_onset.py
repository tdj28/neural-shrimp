"""E5 (exploratory, program PI): does the soft-token loop's chaos onset predict where sampled generation stops
repeating?

Run card: docs/research-log.md, "Run card E5" (2026-10-16 14:06 UTC, commit aecac5d5).

Question. Ordinary sampled generation repeats itself verbatim at low temperature and stops repeating at higher
temperature; the soft-token loop turns chaotic at intermediate temperatures, where next tokens compete. Across prompts,
does the loop's chaos onset T_chaos predict the temperature T_rep at which sampled text stops repeating, and does it
add anything beyond the prompt's own first-step next-token entropy (T_H)?

Model. Qwen2.5-0.5B-Instruct, unsteered.

Prompts. The fixed lists of C12-2 (c12_compression_fidelity.SYSTEM_PROMPTS and OPENERS, 8 each; prompt id 8 s + o;
prefix = chat template with the system prompt, the opener as the user turn and the generation prompt). Rule: the 32
prompts with s + o even, so that each system prompt comes with 4 openers and each opener with 4 system prompts.
Development: system prompts 0 and 1 (8 prompts, every opener once). Held out: system prompts 2-7 (24 prompts, every
opener three times); this is C12-2's split of the system prompts.

Temperatures. 24 values, log10 T = linspace(-1, 0.5, 24) (spacing 1.5 / 23 = 0.0652).

Loop (subcommand loop; float64, CPU). The soft-token map with a one-token window, x' = E^T softmax(z(prefix, x) / T),
z the next-token logits after the prompt's prefix followed by the soft token x (c12_compression_fidelity.make_loop:
float64 weights, RMSNorm and rotary tables patched to float64 by explore_llm_attractors.keep_float64_norms, one prefix
cache per prompt shared by that prompt's rows; left padding, masked, true positions). Start: the embedding of the
prefix's last token (the repository default, explore_llm_soft_loop.SoftLoop.prepare). 1000 transient steps, then 1000
measured steps.
  period     smallest lag q = 1..64 with |x_t - x_{t-q}| < 1e-8 max(1, |x_t|) at the last measured step
             (c12_compression_fidelity.min_period).
  exponents  two twins at relative offsets 1e-6 and 1e-8 (|y - x| = eps |x|; random directions seeded per prompt,
             temperature and offset), renormalized after every measured step (c12_compression_fidelity.Twins);
             exponent = mean log stretch over the 1000 measured steps.
  label      hidden_chaos.classify: periodic if a period is found; chaotic ("hidden chaos" or "visible chaos") if both
             exponents exceed 1e-3; otherwise irregular (kept under that label; never counted as chaos).
  Label-preserving economies, recorded per cell: (1) twins run only for cells with no period at the end of the
  transient; a cell that has one then but none at the end of the measured steps gets a supplementary twin phase of
  1000 steps from that state (flag "supplementary"; its period is then read at the end of that phase). (2) With
  --retire, a cell whose state repeats at one lag q <= 64 to 1e-12 (relative, as above) at 64 consecutive steps leaves
  the batch and is labeled periodic with the period read at 1e-8 at that step. Without --retire the same events are
  only recorded ("virtual retirement") and every cell runs to the end, so that the two labels can be compared.
  T_chaos    lowest grid temperature labeled chaotic; a prompt that is never chaotic is censored at the top of the grid.

Entropy baseline. T_H: the temperature at which the entropy of softmax(z_0 / T) equals 2 nats, z_0 the logits of the
prompt's own first generation step (one float64 forward pass of the prefix; explore_llm_soft_loop.solve_beta).

Sampling (subcommand sample; float32, CPU). At every grid temperature 16 samples of 256 tokens per prompt: ancestral
sampling with a growing context (KV cache), no top-k, no top-p, no repetition penalty. The float32 logits are cast to
float64, and each token is drawn by inverse-CDF sampling from softmax(z / T) with one uniform per sample and step from
numpy default_rng((SEED_SAMPLES, prompt id, temperature index)): the seeds are fixed per prompt and temperature and do
not depend on the batch. The end-of-turn tokens of the generation config (<|im_end|> 151645, <|endoftext|> 151643) get
probability zero, so that every sample is a 256-token reply (Hugging Face min_new_tokens semantics). Reason (design
probe on a prompt outside the set, before any E5 run): when generation is not stopped, the model ends its short reply
within a few dozen tokens and then writes unrelated new documents after <|endoftext|>, so the repetition of 256 tokens
would be a property of that unconditional text rather than of the prompt's reply.
  repeats    the last 96 tokens are exactly periodic with a period of at most 32 (w[i] = w[i + q] for all i).
  T_rep      lowest grid temperature at which fewer than half of the 16 samples repeat, interpolated linearly in log10 T
             between it and the grid point below (where at least half repeat); censored below the grid if fewer than
             half repeat already at the lowest temperature, above the grid if at least half repeat at every temperature.

Analysis (subcommand analyze). Primary: Spearman correlation of log10 T_chaos and log10 T_rep over the 24 held-out
prompts, with a percentile bootstrap over prompts (10000 draws). Censored values enter one grid step beyond the grid
edge, so that they rank beyond every observed value and tie with each other. Secondary: (a) the partial Spearman
correlation of T_chaos and T_rep given T_H (Pearson partial correlation of the ranks), with its bootstrap, and the
Spearman correlations of T_H with both; (b) the repetition fraction in chaotic against periodic cells at matched
temperature: per temperature, the mean fraction over chaotic cells minus the mean over periodic cells, combined over
temperatures with weights n_c n_p / (n_c + n_p), bootstrap over prompts. Sensitivity: cluster bootstraps over openers
and over system prompts, censored prompts excluded, censored values at the grid edge (ties), a permutation p-value.

Development choices (docs/pi/e5-choices.json, committed before held-out data; analyze --choices): the repetition
rule's window and largest period, if the card's are degenerate; the entropy level, if 2 nats lies outside the observed
range; whether the held-out loop uses --retire (only if every development cell's virtual-retirement label equals its
full label).

Outputs under ../neural-shrimp-data/explore/pi/e5/<attempt>/: settings.json (a resumed attempt must match it),
manifest-<time>.json per invocation, progress.jsonl, loop-bNN.npz and .json (subcommand loop), samples-pNN.npz
(subcommand sample), precision.json (subcommand precision), analysis.json (subcommand analyze). Receipt:
docs/pi/e5-summary.json (subcommand summarize).

Usage: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nice -n 10 uv run --no-sync python \
           tools/e5_repetition_onset.py loop --split dev --out dev-loop --threads 6
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
import c12_compression_fidelity as cf  # noqa: E402  (prompt lists, float64 loop, period rule, twins)
import hidden_chaos as hc  # noqa: E402  (the repository's cell labels)

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
OUT_ROOT = DATA / "explore" / "pi" / "e5"
RECEIPT = ROOT / "docs" / "pi" / "e5-summary.json"
CHOICES = ROOT / "docs" / "pi" / "e5-choices.json"

MODEL = "qwen0.5b"
LOG_T = tuple(float(x) for x in np.linspace(-1.0, 0.5, 24))
TRANSIENT, MEASURED = 1000, 1000
EPS = (1e-6, 1e-8)
TOL, QMAX = 1e-8, 64
LAMBDA_MIN = 1e-3
RETIRE_TOL, RETIRE_WINDOW = 1e-12, 64
N_SAMPLES, N_TOKENS = 16, 256
REP_WINDOW, REP_QMAX = 96, 32
REP_QMAX_DIAG = 48                    # diagnostic: smallest period up to half the window
H_TARGET = 2.0
MASKED_TOKENS = (151645, 151643)      # <|im_end|>, <|endoftext|>: the generation config's end-of-turn ids
SEED_TWINS, SEED_SAMPLES, SEED_BOOT = 20261605, 20261606, 20261607
DEV_SYSTEMS, HELDOUT_SYSTEMS = (0, 1), (2, 3, 4, 5, 6, 7)
PROMPTS_PER_LOOP_BATCH = 8
N_BOOT = 10000
LABELS = ("periodic", "chaotic", "irregular")


# ----------------------------------------------------------------------------------------------- prompts and grid

def e5_prompt_ids(split):
    """Prompt ids (8 s + o) of a split: system prompts of the split, openers o with s + o even, system-major order."""
    systems = {"dev": DEV_SYSTEMS, "heldout": HELDOUT_SYSTEMS}[split]
    return [cf.prompt_id(s, o) for s in systems for o in range(len(cf.OPENERS)) if (s + o) % 2 == 0]


def prompt_record(pid):
    s, o = divmod(int(pid), 8)
    return {"prompt": int(pid), "system_index": s, "opener_index": o, "system": cf.SYSTEM_PROMPTS[s],
            "opener": cf.OPENERS[o]}


def prompt_lists():
    return {"system_prompts": list(cf.SYSTEM_PROMPTS), "openers": list(cf.OPENERS),
            "rule": "prompt id 8 s + o for s + o even; development s in {0, 1}, held out s in {2, ..., 7}",
            "dev": e5_prompt_ids("dev"), "heldout": e5_prompt_ids("heldout")}


def prompt_lists_hash():
    return hashlib.sha256(json.dumps(prompt_lists(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def loop_batches(prompts, per_batch=PROMPTS_PER_LOOP_BATCH):
    return [list(prompts[k:k + per_batch]) for k in range(0, len(prompts), per_batch)]


def grid_step(log_t):
    return float(log_t[1] - log_t[0])


# ----------------------------------------------------------------------------------------------- repetition rule

def smallest_period(tokens, window=REP_WINDOW, qmax=REP_QMAX):
    """Smallest q in 1..qmax for which the last `window` tokens of each row are exactly periodic with period q
    (w[i] == w[i + q] for every i); 0 where there is none. tokens: (..., n) integer array, n >= window."""
    a = np.asarray(tokens)
    if a.shape[-1] < window:
        raise ValueError(f"sequences of {a.shape[-1]} tokens are shorter than the window {window}")
    if not 1 <= qmax < window:
        raise ValueError(f"largest period {qmax} must lie between 1 and the window {window} - 1")
    w = a[..., -window:]
    out = np.zeros(a.shape[:-1], dtype=np.int16)
    for q in range(qmax, 0, -1):                  # descending, so the smallest period wins
        ok = np.all(w[..., q:] == w[..., :-q], axis=-1)
        out = np.where(ok, q, out)
    return out


def periodic_tail(tokens, q):
    """Length of the longest suffix of each row that is periodic with period q (0 where q = 0): q plus the number of
    consecutive positions t, counted back from the end, with s[t] == s[t - q]."""
    a = np.asarray(tokens)
    q = np.broadcast_to(np.asarray(q), a.shape[:-1])
    n = a.shape[-1]
    out = np.zeros(a.shape[:-1], dtype=np.int32)
    for idx in np.ndindex(*a.shape[:-1]):
        qq = int(q[idx])
        if qq <= 0:
            continue
        s = a[idx]
        k = 0
        while n - 1 - k - qq >= 0 and s[n - 1 - k] == s[n - 1 - k - qq]:
            k += 1
        out[idx] = qq + k
    return out


def repeat_fraction(tokens, window=REP_WINDOW, qmax=REP_QMAX):
    """Fraction of samples that repeat, per leading index: tokens (..., samples, n)."""
    return (smallest_period(tokens, window, qmax) > 0).mean(axis=-1)


# ----------------------------------------------------------------------------------------------- onsets

def rep_onset(log_t, frac, level=0.5):
    """T_rep: lowest grid point with frac < level, interpolated linearly in log10 T with the grid point below.
    Returns {value (None if censored), censored ('below', 'above' or None), analysis_value (censored: one grid step
    beyond the edge), index, monotone (frac never rises again after the crossing)}."""
    log_t, frac = np.asarray(log_t, dtype=float), np.asarray(frac, dtype=float)
    step = grid_step(log_t)
    below = np.flatnonzero(frac < level)
    if len(below) == 0:
        return {"value": None, "censored": "above", "analysis_value": float(log_t[-1] + step), "index": None,
                "monotone": None}
    j = int(below[0])
    monotone = bool(np.all(frac[j:] < level))
    if j == 0:
        return {"value": None, "censored": "below", "analysis_value": float(log_t[0] - step), "index": 0,
                "monotone": monotone}
    f0, f1 = frac[j - 1], frac[j]
    x = log_t[j - 1] + (f0 - level) / (f0 - f1) * (log_t[j] - log_t[j - 1])
    return {"value": float(x), "censored": None, "analysis_value": float(x), "index": j, "monotone": monotone}


def chaos_onset(log_t, labels, target="chaotic"):
    """T_chaos: lowest grid temperature whose label is `target`; censored above the grid if there is none."""
    log_t = np.asarray(log_t, dtype=float)
    idx = [j for j, lab in enumerate(labels) if lab == target]
    if not idx:
        return {"value": None, "censored": "above", "analysis_value": float(log_t[-1] + grid_step(log_t)),
                "index": None}
    return {"value": float(log_t[idx[0]]), "censored": None, "analysis_value": float(log_t[idx[0]]), "index": idx[0]}


def entropy_of(z, T):
    """Entropy (nats) of softmax(z / T) for a float64 logit vector z (numpy)."""
    a = np.asarray(z, dtype=np.float64) / float(T)
    a = a - a.max()
    w = np.exp(a)
    s = w.sum()
    return float(np.log(s) - (w * a).sum() / s)


def entropy_temperature(z, target):
    """Temperature T with entropy(softmax(z / T)) = target (nats), via explore_llm_soft_loop.solve_beta."""
    import torch

    import explore_llm_soft_loop as esl
    z = np.asarray(z, dtype=np.float64)
    if not 0.0 < target < math.log(len(z)):
        raise ValueError(f"entropy target {target} outside (0, log V)")
    beta = float(esl.solve_beta(torch.as_tensor(z)[None], np.array([target]))[0])
    T = 1.0 / beta
    if abs(entropy_of(z, T) - target) > 1e-8:
        raise SystemExit(f"entropy solve failed: H(T = {T}) = {entropy_of(z, T)}, target {target}")
    return T


# ----------------------------------------------------------------------------------------------- statistics

def rankdata(x):
    from scipy import stats
    return stats.rankdata(np.asarray(x, dtype=float))


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(x, y):
    """Spearman correlation (average ranks for ties); NaN when either variable is constant."""
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y given z: the partial correlation of the ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    den = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 0.0))
    return float((rxy - rxz * ryz) / den) if den > 0 else float("nan")


def bootstrap(stat, n, n_boot=N_BOOT, seed=SEED_BOOT, clusters=None, level=0.95):
    """Percentile bootstrap of stat(indices) over n units (or over whole clusters of units). Draws whose statistic is
    not finite (constant resampled variables) are counted and left out of the interval."""
    rng = np.random.default_rng(seed)
    vals = []
    if clusters is None:
        for _ in range(n_boot):
            vals.append(stat(rng.integers(0, n, size=n)))
    else:
        clusters = np.asarray(clusters)
        keys = sorted(set(clusters.tolist()))
        members = [np.flatnonzero(clusters == k) for k in keys]
        for _ in range(n_boot):
            draw = rng.integers(0, len(keys), size=len(keys))
            vals.append(stat(np.concatenate([members[k] for k in draw])))
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals)
    a = (1 - level) / 2
    ci = [float(np.quantile(vals[ok], a)), float(np.quantile(vals[ok], 1 - a))] if ok.any() else [None, None]
    return {"ci": ci, "draws": int(n_boot), "nonfinite_draws": int((~ok).sum()), "level": level,
            "clusters": None if clusters is None else len(set(np.asarray(clusters).tolist()))}


def permutation_p(x, y, n_perm=N_BOOT, seed=SEED_BOOT + 1):
    """Two-sided permutation p-value of the Spearman correlation (y shuffled over units), with the +1 correction."""
    rng = np.random.default_rng(seed)
    r0 = spearman(x, y)
    y = np.asarray(y, dtype=float)
    hits = sum(abs(spearman(x, rng.permutation(y))) >= abs(r0) - 1e-12 for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def stratified_difference(frac, labels, units=None):
    """Repetition fraction in chaotic minus periodic cells at matched temperature. frac, labels: (prompts, temps).
    units: prompt indices to use (with repetition, for the bootstrap). Per temperature the mean over chaotic cells
    minus the mean over periodic cells, combined over the temperatures that have both with weights
    n_c n_p / (n_c + n_p). Returns (combined difference or NaN, per-temperature rows)."""
    frac, labels = np.asarray(frac, dtype=float), np.asarray(labels)
    if units is not None:
        frac, labels = frac[units], labels[units]
    rows, num, den = [], 0.0, 0.0
    for j in range(frac.shape[1]):
        c, p = labels[:, j] == "chaotic", labels[:, j] == "periodic"
        nc, npp = int(c.sum()), int(p.sum())
        row = {"t_index": j, "n_chaotic": nc, "n_periodic": npp, "n_irregular": int((labels[:, j] == "irregular").sum()),
               "mean_chaotic": float(frac[c, j].mean()) if nc else None,
               "mean_periodic": float(frac[p, j].mean()) if npp else None}
        if nc and npp:
            w = nc * npp / (nc + npp)
            row["difference"], row["weight"] = row["mean_chaotic"] - row["mean_periodic"], w
            num += w * row["difference"]
            den += w
        rows.append(row)
    return (num / den if den > 0 else float("nan")), rows


# ----------------------------------------------------------------------------------------------- sampling

def draw_tokens(z, temps, u, masked=MASKED_TOKENS):
    """Inverse-CDF draws from softmax(z / T) per row, in float64: token = the first index whose cumulative weight
    exceeds u * total. z (R, V) logits (any float dtype), temps (R,), u (R,) uniforms in [0, 1). Masked ids have weight
    zero and are never drawn. Returns (tokens (R,) int64, entropy (R,) of the masked distribution)."""
    import torch
    z64 = z.to(torch.float64).clone()
    if masked:
        z64[:, list(masked)] = -math.inf
    T = torch.as_tensor(np.asarray(temps), dtype=torch.float64)[:, None]
    a = (z64 - z64.max(dim=-1, keepdim=True).values) / T
    w = torch.exp(a)
    c = torch.cumsum(w, dim=-1)
    total = c[:, -1:]
    v = torch.as_tensor(np.asarray(u), dtype=torch.float64)[:, None] * total
    tok = torch.searchsorted(c, v, right=True)[:, 0].clamp_(max=z.shape[1] - 1)
    a = torch.where(w > 0, a, torch.zeros_like(a))
    ent = torch.log(total[:, 0]) - (w * a).sum(-1) / total[:, 0]
    return tok, ent


def sample_uniforms(pid, n_temps, n_samples, n_tokens, seed=SEED_SAMPLES):
    """(n_temps * n_samples, n_tokens) uniforms; row j * n_samples + i is sample i at temperature index j, drawn from
    numpy default_rng((seed, prompt id, j)) as one (n_tokens, n_samples) array."""
    return np.concatenate([np.random.default_rng((seed, int(pid), j)).random((n_tokens, n_samples)).T
                           for j in range(n_temps)])


def sample_prompt(model, tok, pid, log_t, n_samples, n_tokens, masked=MASKED_TOKENS, seed=SEED_SAMPLES):
    """Ancestral sampling of n_samples continuations of n_tokens tokens at every temperature for one prompt, in the
    model's dtype, from one shared prefix cache. Returns tokens (n_temps, n_samples, n_tokens) int32, the mean entropy
    of the sampling distributions per sample, the prefix and the first-step logits (float64)."""
    import torch
    from transformers import DynamicCache
    nT = len(log_t)
    R = nT * n_samples
    ids = cf.prefix_ids(tok, pid)
    temps = np.repeat(10.0 ** np.asarray(log_t), n_samples)
    U = sample_uniforms(pid, nT, n_samples, n_tokens, seed)
    cache = DynamicCache()
    toks = np.zeros((R, n_tokens), dtype=np.int32)
    ent = np.zeros(R)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        z0 = out.logits[0, -1].clone()
        cache.batch_repeat_interleave(R)
        z = z0[None].expand(R, -1)
        for k in range(n_tokens):
            nxt, e = draw_tokens(z, temps, U[:, k], masked)
            toks[:, k] = nxt.numpy()
            ent += e.numpy()
            if k + 1 < n_tokens:
                out = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
                z = out.logits[:, -1]
    return {"tokens": toks.reshape(nT, n_samples, n_tokens), "mean_entropy": (ent / n_tokens).reshape(nT, n_samples),
            "prefix": np.asarray(ids, dtype=np.int32), "z0": z0.to(torch.float64).numpy()}


# ----------------------------------------------------------------------------------------------- loop

def twin_directions(meta, d, n_eps=len(EPS), stream=0):
    """Unit-variance Gaussian twin directions, one generator per cell and offset (independent of the batch):
    default_rng((SEED_TWINS, prompt id, temperature index, offset index, stream)); meta: (prompt id, t index) pairs."""
    return [np.stack([np.random.default_rng((SEED_TWINS, int(p), int(j), k, stream)).normal(size=d) for p, j in meta])
            for k in range(n_eps)]


def lag_closures(ring, t, qmax):
    """(qmax, n) relative closures |x_t - x_{t-q}| / max(1, |x_t|) for q = 1..qmax (inf for q > t); ring holds the state
    of step s at index s % len(ring)."""
    import torch
    R = ring.shape[0]
    x = ring[t % R]
    scale = torch.clamp(x.norm(dim=-1), min=1.0)
    D = torch.full((qmax, x.shape[0]), math.inf, dtype=torch.float64)
    for q in range(1, min(qmax, t) + 1):
        D[q - 1] = ((x - ring[(t - q) % R]).norm(dim=-1) / scale).to(torch.float64)
    return D


def first_lag(D, tol):
    """Smallest q (1-based) with D[q - 1] < tol per column, 0 where there is none."""
    import torch
    close = D < tol
    any_ = close.any(dim=0)
    first = torch.argmax(close.to(torch.int8), dim=0) + 1
    return torch.where(any_, first, torch.zeros_like(first)).numpy()


def run_loop(backend, x0, cfg, directions, retire=False, log=None, rebuild_every=8):
    """Iterate every cell of a batch from x0 (n, d): transient, measured steps with twins for the cells without a
    period at the end of the transient, supplementary twin phase for cells that lose their period, retirement (real
    with retire=True, otherwise recorded only). backend.set_rows(cells) selects the rows' cells and backend.step(X)
    maps (B, d) states to (states (B, d), next-token distributions (B, V)). directions(cells, stream) gives one (len,
    d) tensor per offset. Returns per-cell arrays (see the module docstring)."""
    import torch
    log = log or (lambda **k: None)
    n, d = x0.shape
    eps = tuple(cfg["eps"])
    m = len(eps)
    T_A, T_B, tol, qmax = int(cfg["transient"]), int(cfg["measured"]), float(cfg["tol"]), int(cfg["qmax"])
    rtol, rwin = float(cfg["retire_tol"]), int(cfg["retire_window"])
    R = qmax + 1
    X = x0.clone()
    ring = torch.empty((R, n, d), dtype=X.dtype)
    ring[0] = X
    active = np.ones(n, dtype=bool)                  # iterated (real retirement removes cells)
    ret_step = np.full(n, -1, dtype=np.int32)        # first (real or virtual) retirement step
    ret_q = np.zeros(n, dtype=np.int16)              # lag that closed at retire_tol
    ret_qtol = np.zeros(n, dtype=np.int16)           # smallest lag closing at tol at that step
    same = np.zeros(n, dtype=np.int32)
    lastq = np.zeros(n, dtype=np.int64)
    lead = np.full((T_B, n), -1, dtype=np.int32)
    ent_sum, ent_sq, ent_n = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int32)
    state = {"cells": None, "since": 0}
    tic = time.perf_counter()

    def set_rows(cells):
        if state["cells"] is None or len(state["cells"]) != len(cells) or not np.array_equal(state["cells"], cells):
            backend.set_rows(cells)
            state["cells"] = np.array(cells, copy=True)
            state["since"] = 0

    def retirement(t, eligible):
        if t <= qmax:
            return 0
        D = lag_closures(ring, t, qmax)
        q_r = first_lag(D, rtol)
        cont = (q_r > 0) & (q_r == lastq)
        same[:] = np.where(cont, same + 1, np.where(q_r > 0, 1, 0))
        lastq[:] = q_r
        new = eligible & (same >= rwin) & (ret_step < 0)
        if new.any():
            idx = np.flatnonzero(new)
            ret_step[idx] = t
            ret_q[idx] = q_r[idx]
            ret_qtol[idx] = first_lag(D, tol)[idx]
            if retire:
                active[idx] = False
        return int(new.sum())

    def entropy_rows(p):
        return (-(p * torch.log(torch.clamp(p, min=1e-300))).sum(-1)).double().numpy()

    # transient: base rows only
    for t in range(1, T_A + 1):
        rows = state["cells"] if state["cells"] is not None else np.flatnonzero(active)
        if retire and (state["since"] >= rebuild_every and not active[rows].all()):
            rows = np.flatnonzero(active)
        state["since"] += 1
        if len(rows):
            set_rows(rows)
            Y, _ = backend.step(X[torch.as_tensor(rows)])
            if not bool(torch.isfinite(Y).all()):
                raise SystemExit(f"non-finite state at transient step {t}")
            upd = active[rows]
            X[torch.as_tensor(rows[upd])] = Y[torch.as_tensor(np.flatnonzero(upd))]
        ring[t % R] = X
        retirement(t, active.copy())
        if t % 100 == 0:
            log(event="loop_progress", phase="transient", step=t, active=int(active.sum()),
                retired=int((ret_step >= 0).sum()), seconds=round(time.perf_counter() - tic, 1))
    q_A = cf.min_period(ring, T_A, tol, qmax).numpy().astype(np.int16)
    q_A[~active] = -1
    twin = active & (q_A == 0)
    tw_cells = np.flatnonzero(twin)
    tw = None
    if len(tw_cells):
        dirs = [torch.as_tensor(u, dtype=X.dtype) for u in directions(tw_cells, 0)]
        tw = cf.Twins(X[torch.as_tensor(tw_cells)], eps, dirs)
    log(event="loop_phase", phase="measured", twin_cells=int(len(tw_cells)), active=int(active.sum()),
        seconds=round(time.perf_counter() - tic, 1))
    # measured steps: base-only rows and the twin cells (base, twin 1, twin 2)
    state["cells"] = None
    bo = np.flatnonzero(active & ~twin)
    for t in range(T_A + 1, T_A + T_B + 1):
        if retire and state["since"] >= rebuild_every and not active[bo].all():
            bo = np.flatnonzero(active & ~twin)
            state["cells"] = None
        cells = np.concatenate([bo] + [tw_cells] * (m + 1))
        state["since"] += 1
        nbo, ntw = len(bo), len(tw_cells)
        k = t - T_A - 1
        if len(cells):
            set_rows(cells)
            Xin = torch.cat([X[torch.as_tensor(bo)]] + ([tw.X] if tw is not None else []))
            Y, p = backend.step(Xin)
            if not bool(torch.isfinite(Y).all()):
                raise SystemExit(f"non-finite state at measured step {t}")
            upd = active[bo]
            ub = bo[upd]
            X[torch.as_tensor(ub)] = Y[:nbo][torch.as_tensor(np.flatnonzero(upd))]
            if tw is not None:
                X[torch.as_tensor(tw_cells)] = tw.advance(lambda _X: Y[nbo:])
                if not bool(torch.isfinite(tw.X).all()):
                    raise SystemExit(f"non-finite twin state at measured step {t}")
            pb = torch.cat([p[:nbo][torch.as_tensor(np.flatnonzero(upd))], p[nbo:nbo + ntw]])
            rows_b = np.concatenate([ub, tw_cells])
            lead[k, rows_b] = pb.argmax(-1).numpy()
            e = entropy_rows(pb)
            ent_sum[rows_b] += e
            ent_sq[rows_b] += e * e
            ent_n[rows_b] += 1
        ring[t % R] = X
        retirement(t, active & ~twin)
        if t % 100 == 0:
            log(event="loop_progress", phase="measured", step=t, active=int(active.sum()),
                retired=int((ret_step >= 0).sum()), seconds=round(time.perf_counter() - tic, 1))
    t_end = T_A + T_B
    q_end = cf.min_period(ring, t_end, tol, qmax).numpy().astype(np.int16)
    q_end[~active] = -1
    lam = np.full((m, n), np.nan)
    collapses = np.zeros((m, n), dtype=np.int64)
    if tw is not None:
        lam[:, tw_cells] = tw.exponents()
        collapses[:, tw_cells] = tw.collapses
    lead_period = np.zeros(n, dtype=np.int16)
    for c in tw_cells:
        lead_period[c] = hc.symbolic_period(lead[:, c].tolist())
    # supplementary twin phase for cells that had a period after the transient but none at the end
    lost = np.flatnonzero(active & ~twin & (q_end == 0))
    q_sup = np.full(n, -1, dtype=np.int16)
    if len(lost):
        log(event="loop_phase", phase="supplementary", cells=[int(c) for c in lost])
        dirs = [torch.as_tensor(u, dtype=X.dtype) for u in directions(lost, 1)]
        tw2 = cf.Twins(X[torch.as_tensor(lost)], eps, dirs)
        ring2 = torch.empty((R, len(lost), d), dtype=X.dtype)
        ring2[0] = tw2.X[:len(lost)]
        set_rows(np.concatenate([lost] * (m + 1)))
        lead2 = np.full((T_B, len(lost)), -1, dtype=np.int32)
        for k in range(1, T_B + 1):
            holder = {}

            def F(Xs):
                Y, p = backend.step(Xs)
                holder["p"] = p
                return Y
            xb = tw2.advance(F)
            ring2[k % R] = xb
            lead2[k - 1] = holder["p"][:len(lost)].argmax(-1).numpy()
        X[torch.as_tensor(lost)] = tw2.X[:len(lost)]
        q_sup[lost] = cf.min_period(ring2, T_B, tol, qmax).numpy().astype(np.int16)
        lam[:, lost] = tw2.exponents()
        collapses[:, lost] = tw2.collapses
        for i, c in enumerate(lost):
            lead_period[c] = hc.symbolic_period(lead2[:, i].tolist())
    supplementary = np.zeros(n, dtype=bool)
    supplementary[lost] = True
    # labels (hidden_chaos.classify)
    period = np.zeros(n, dtype=np.int16)
    raw_class = np.empty(n, dtype=object)
    label = np.empty(n, dtype=object)
    for c in range(n):
        if not active[c]:                                     # retired: periodic at the retirement step
            q = int(ret_qtol[c])
            r = {"lyapunov": [float("nan")] * m, "period": q, "lead_period": 0}
        elif supplementary[c]:
            q = int(q_sup[c])
            r = {"lyapunov": [float(x) for x in lam[:, c]], "period": q, "lead_period": int(lead_period[c])}
        elif twin[c] or q_end[c] > 0:
            q = int(q_end[c])
            r = {"lyapunov": [float(x) for x in lam[:, c]], "period": q, "lead_period": int(lead_period[c])}
        else:
            raise SystemExit(f"cell {c}: no label (inconsistent phases)")
        cls = hc.classify(r)
        raw_class[c] = cls
        period[c] = q if cls == "periodic" else 0
        label[c] = "periodic" if cls == "periodic" else ("chaotic" if cls in ("hidden chaos", "visible chaos")
                                                         else "irregular")
    residual = np.full(n, np.nan)
    per = (period > 0) & active & ~supplementary
    if per.any():
        res = cf.lag_residual(ring, t_end, torch.as_tensor(np.where(per, period, 0).astype(np.int64))).numpy()
        residual[per] = res[per]
    with np.errstate(invalid="ignore", divide="ignore"):
        ent_mean = np.where(ent_n > 0, ent_sum / np.maximum(ent_n, 1), np.nan)
        ent_sd = np.where(ent_n > 1, np.sqrt(np.maximum(ent_sq / np.maximum(ent_n, 1) - ent_mean ** 2, 0)), np.nan)
    log(event="loop_done", seconds=round(time.perf_counter() - tic, 1),
        counts={lab: int((label == lab).sum()) for lab in LABELS})
    return {"label": label.astype(str), "raw_class": raw_class.astype(str), "period": period, "q_transient": q_A,
            "q_end": q_end, "q_supplementary": q_sup, "twins": twin, "supplementary": supplementary,
            "retired": ~active, "retire_step": ret_step, "retire_q": ret_q, "retire_q_tol": ret_qtol,
            "lyapunov": lam, "collapses": collapses, "lead_period": lead_period, "lead": lead,
            "entropy_mean": ent_mean, "entropy_sd": ent_sd, "residual": residual,
            "x_end": X.double().numpy(), "seconds": np.array([time.perf_counter() - tic])}


def virtual_retirement_check(res):
    """Cells whose (virtual) retirement label differs from the full label: retired cells must end periodic with the
    period read at retirement."""
    bad = []
    for c in np.flatnonzero(res["retire_step"] >= 0):
        if res["retired"][c]:
            continue                                   # real retirement: no full label to compare
        if res["label"][c] != "periodic" or int(res["period"][c]) != int(res["retire_q_tol"][c]):
            bad.append({"cell": int(c), "label": str(res["label"][c]), "period": int(res["period"][c]),
                        "retire_step": int(res["retire_step"][c]), "retire_q_tol": int(res["retire_q_tol"][c])})
    return bad


class SoftBackend:
    """The float64 soft-token map of a batch of prompts at every grid temperature: cell g * n_temps + j is prompt g at
    temperature index j (c12_compression_fidelity.make_loop's per-prompt prefix caches)."""

    def __init__(self, loop, prompts, log_t):
        import torch
        self.torch, self.loop = torch, loop
        self.prefixes = [cf.prefix_ids(loop.sm.tok, p) for p in prompts]
        self.pc = loop.prefix_cache(self.prefixes)
        nT = len(log_t)
        self.n = len(prompts) * nT
        self.cell_prompt = np.repeat(np.arange(len(prompts)), nT)
        self.cell_tindex = np.tile(np.arange(nT), len(prompts))
        self.cell_T = 10.0 ** np.asarray(log_t, dtype=float)[self.cell_tindex]
        self.prompts = list(prompts)
        self.temps = None

    def start(self):
        last = self.torch.tensor([p[-1] for p in self.prefixes])
        return self.loop.E[last[self.torch.as_tensor(self.cell_prompt)]].clone()

    def meta(self, cells):
        return [(self.prompts[self.cell_prompt[c]], self.cell_tindex[c]) for c in cells]

    def set_rows(self, cells):
        cells = np.asarray(cells, dtype=np.int64)
        self.loop.use_rows(self.pc, self.cell_prompt[cells])
        self.temps = self.torch.as_tensor(self.cell_T[cells], dtype=self.torch.float64)

    def step(self, X):
        Y, p = self.loop.soft_step(X[:, None, :], self.temps)
        return Y[:, 0], p

    def first_step_logits(self):
        """float64 logits of each prompt's first generation step (one forward pass of each prefix)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for ids in self.prefixes:
                z = self.loop.model(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1]
                out.append(z.to(torch.float64).numpy())
        return np.stack(out)


def entropy_record(z, log_t, target=H_TARGET, masked=MASKED_TOKENS):
    """First-step entropy curve on the grid and T_H (full distribution; also with the masked tokens removed)."""
    keep = np.ones(len(z), dtype=bool)
    keep[list(masked)] = False
    return {"entropy_by_T": [entropy_of(z, 10.0 ** lt) for lt in log_t],
            "entropy_by_T_masked": [entropy_of(z[keep], 10.0 ** lt) for lt in log_t],
            "T_H": entropy_temperature(z, target), "T_H_masked": entropy_temperature(z[keep], target),
            "entropy_at_T1": entropy_of(z, 1.0), "max_probability_T1": float(np.exp(z - z.max()).max()
                                                                             / np.exp(z - z.max()).sum())}


# ----------------------------------------------------------------------------------------------- io

def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()  # noqa
    return {"commit": run("rev-parse", "HEAD"),
            "tool_dirty": bool(run("status", "--porcelain", "--", "tools/e5_repetition_onset.py"))}


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


def settings_of(args, kind):
    base = {"experiment": "E5", "kind": kind, "model": MODEL, "log_t": list(LOG_T),
            "prompts": [int(p) for p in (args.prompt_ids or e5_prompt_ids(args.split))], "split": args.split,
            "prompt_lists_sha256": prompt_lists_hash()}
    if kind == "loop":
        base.update(transient=int(args.transient), measured=int(args.measured), eps=list(EPS), tol=TOL, qmax=QMAX,
                    lambda_min=LAMBDA_MIN, retire=bool(args.retire), retire_tol=RETIRE_TOL,
                    retire_window=RETIRE_WINDOW, h_target=H_TARGET, seeds={"twins": SEED_TWINS},
                    prompts_per_batch=PROMPTS_PER_LOOP_BATCH, start="embedding of the prefix's last token",
                    dtype="float64")
    elif kind == "sample":
        base.update(samples=int(args.samples), tokens=int(args.tokens), masked_tokens=list(MASKED_TOKENS),
                    seeds={"samples": SEED_SAMPLES}, dtype="float32", sampler="inverse CDF in float64 from float32 "
                    "logits, one uniform per sample and step, no top-k/top-p/repetition penalty")
    return base


def open_out(name, settings):
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "settings.json"
    if sp.exists():
        if json.loads(sp.read_text()) != settings:
            raise SystemExit(f"{sp} holds other settings; use a new --out (attempts are never overwritten)")
    else:
        write_json(sp, settings)
    return out


def write_manifest(out, args, extra):
    import torch
    import transformers
    man = {"argv": sys.argv, "args": vars(args), "git": git_state(), "model": MODEL,
           "model_revision": cf.model_revision(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "threads": torch.get_num_threads(), "host": platform.node(),
           "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "prompt_lists": prompt_lists(), "prompt_lists_sha256": prompt_lists_hash(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    man.update(extra)
    write_json(out / f"manifest-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.json", man)


# ----------------------------------------------------------------------------------------------- gates

def gate_loop(res, n_cells, cfg):
    """Row-level gate of one loop batch (every cell read). Returns the list of failed checks."""
    fails = []
    lab = res["label"]
    if len(lab) != n_cells:
        fails.append(f"{len(lab)} cells, expected {n_cells}")
    if not np.isfinite(res["x_end"]).all():
        fails.append("non-finite final states")
    if not set(lab.tolist()) <= set(LABELS):
        fails.append(f"unknown labels {sorted(set(lab.tolist()) - set(LABELS))}")
    per = lab == "periodic"
    if np.any(res["period"][per] < 1) or np.any(res["period"][per] > cfg["qmax"]):
        fails.append("periodic cell with a period outside 1..qmax")
    if np.any(res["period"][~per] != 0):
        fails.append("non-periodic cell with a period")
    chk = per & ~res["retired"] & ~res["supplementary"]
    if np.any(~(res["residual"][chk] < cfg["tol"])):
        fails.append("periodic cell whose closure residual is not below tol")
    nonper = ~per
    lam = res["lyapunov"]
    if np.any(~np.isfinite(lam[:, nonper])):
        fails.append("non-periodic cell without finite exponents at both offsets")
    chaotic = lab == "chaotic"
    if np.any(~np.all(lam[:, chaotic] > cfg["lambda_min"], axis=0)):
        fails.append("chaotic cell whose exponents are not all above lambda_min")
    irr = lab == "irregular"
    if np.any(np.all(lam[:, irr] > cfg["lambda_min"], axis=0)):
        fails.append("irregular cell whose exponents are all above lambda_min")
    if np.any(res["retired"] & ~per):
        fails.append("retired cell not labeled periodic")
    if not cfg.get("retire") and np.any(res["retired"]):
        fails.append("cells retired in a run without --retire")
    return fails


def gate_samples(z, n_temps, n_samples, n_tokens, vocab):
    fails = []
    t = z["tokens"]
    if t.shape != (n_temps, n_samples, n_tokens):
        fails.append(f"tokens of shape {t.shape}, expected {(n_temps, n_samples, n_tokens)}")
    if t.min() < 0 or t.max() >= vocab:
        fails.append("token ids outside the vocabulary")
    if np.isin(t, MASKED_TOKENS).any():
        fails.append("masked end-of-turn tokens present")
    if not np.isfinite(z["mean_entropy"]).all():
        fails.append("non-finite sampling entropies")
    return fails


# ----------------------------------------------------------------------------------------------- subcommands

def cmd_loop(args):
    settings = settings_of(args, "loop")
    out = open_out(args.out, settings)
    log = Log(out)
    loop = cf.make_loop(args.threads)
    write_manifest(out, args, {"settings": settings, "subcommand": "loop"})
    prompts = settings["prompts"]
    batches = loop_batches(prompts)
    order = range(len(batches)) if args.batches is None else args.batches
    cfg = {k: settings[k] for k in ("transient", "measured", "eps", "tol", "qmax", "retire_tol", "retire_window",
                                     "lambda_min", "retire")}
    log(event="start", subcommand="loop", batches=list(order), prompts=prompts, retire=bool(args.retire))
    for k in order:
        path = out / f"loop-b{k:02d}.npz"
        if path.exists():
            log(event="batch_exists", batch=k)
            continue
        group = batches[k]
        be = SoftBackend(loop, group, settings["log_t"])
        x0 = be.start()
        d = x0.shape[1]
        dirs = lambda cells, stream: twin_directions(be.meta(cells), d, len(EPS), stream)  # noqa: E731
        res = run_loop(be, x0, cfg, dirs, retire=bool(args.retire),
                       log=lambda **r: log(batch=k, **r))
        z0 = be.first_step_logits()
        ent = [entropy_record(z, settings["log_t"], settings["h_target"]) for z in z0]
        fails = gate_loop(res, be.n, cfg)
        mismatch = virtual_retirement_check(res)
        arrays = dict(res, cell_prompt=np.asarray(group)[be.cell_prompt].astype(np.int16),
                      cell_tindex=be.cell_tindex.astype(np.int16), batch_prompts=np.asarray(group, dtype=np.int16),
                      T_H=np.array([e["T_H"] for e in ent]), T_H_masked=np.array([e["T_H_masked"] for e in ent]))
        save_npz(path, **{a: (v if not isinstance(v, np.ndarray) or v.dtype != object else v.astype(str))
                          for a, v in arrays.items()}, first_step_logits=z0)
        cells = []
        for c in range(be.n):
            cells.append({"prompt": int(group[be.cell_prompt[c]]), "t_index": int(be.cell_tindex[c]),
                          "log10_T": settings["log_t"][be.cell_tindex[c]], "label": str(res["label"][c]),
                          "class": str(res["raw_class"][c]), "period": int(res["period"][c]),
                          "lyapunov": [None if not np.isfinite(x) else float(x) for x in res["lyapunov"][:, c]],
                          "q_transient": int(res["q_transient"][c]), "q_end": int(res["q_end"][c]),
                          "twins": bool(res["twins"][c]), "supplementary": bool(res["supplementary"][c]),
                          "retired": bool(res["retired"][c]), "retire_step": int(res["retire_step"][c]),
                          "lead_period": int(res["lead_period"][c]),
                          "entropy_mean": None if not np.isfinite(res["entropy_mean"][c]) else
                          float(res["entropy_mean"][c])})
        write_json(out / f"loop-b{k:02d}.json", {"batch": k, "prompts": group, "gate_failures": fails,
                                                 "virtual_retirement_mismatches": mismatch,
                                                 "first_step_entropy": [dict(prompt=int(p), **e)
                                                                        for p, e in zip(group, ent)],
                                                 "seconds": float(res["seconds"][0]), "cells": cells,
                                                 "git": git_state()})
        log(event="batch_done", batch=k, prompts=group, gate_failures=fails, virtual_mismatches=len(mismatch),
            seconds=round(float(res["seconds"][0]), 1),
            counts={lab: int((res["label"] == lab).sum()) for lab in LABELS})
        del be
    log(event="end")


def cmd_sample(args):
    import torch

    import explore_llm_attractors as ela
    settings = settings_of(args, "sample")
    out = open_out(args.out, settings)
    log = Log(out)
    eng = ela.Engine(MODEL, args.threads, "float32", "cpu")
    model, tok = eng.model, eng.tok
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise SystemExit("sampling model is not float32")
    gen_eos = model.generation_config.eos_token_id
    gen_eos = sorted(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    if gen_eos != sorted(MASKED_TOKENS):
        raise SystemExit(f"generation config end-of-turn ids {gen_eos} differ from the masked ids {MASKED_TOKENS}")
    write_manifest(out, args, {"settings": settings, "subcommand": "sample",
                               "attention": getattr(model.config, "_attn_implementation", None)})
    V = model.lm_head.weight.shape[0]
    log(event="start", subcommand="sample", prompts=settings["prompts"])
    for pid in settings["prompts"]:
        path = out / f"samples-p{pid:02d}.npz"
        if path.exists():
            log(event="prompt_exists", prompt=pid)
            continue
        tic = time.perf_counter()
        res = sample_prompt(model, tok, pid, settings["log_t"], settings["samples"], settings["tokens"])
        z0 = res.pop("z0")
        ent = entropy_record(z0, settings["log_t"])
        fails = gate_samples(res, len(settings["log_t"]), settings["samples"], settings["tokens"], V)
        sec = time.perf_counter() - tic
        save_npz(path, **res, T_H_float32=np.array(ent["T_H"]), seconds=np.array([sec]),
                 gate_failures=np.array(json.dumps(fails)))
        frac = repeat_fraction(res["tokens"]) if settings["tokens"] >= REP_WINDOW else None
        log(event="prompt_done", prompt=pid, seconds=round(sec, 1), gate_failures=fails,
            repeat_fraction=None if frac is None else [round(float(f), 3) for f in frac])
    log(event="end")


def cmd_precision(args):
    """Central differences of the float64 map at states after 64 steps from the start (prompt --prompt-ids[0] or the
    first development prompt) at three temperatures: |D(h)| with D(h) = (F(x + h u) - F(x - h u)) / (2h) along a unit
    direction u, for h = 1e-1 .. 1e-10; successive relative changes |D(h) - D(h/10)| / |D(h/10)| fall about 100-fold
    per decade (h^2) until rounding takes over. Also the largest difference between two evaluations of F(x) at
    different batch positions."""
    import torch
    out = OUT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    loop = cf.make_loop(args.threads)
    pid = int(args.prompt_ids[0]) if args.prompt_ids else e5_prompt_ids("dev")[0]
    t_idx = [int(j) for j in args.t_index]
    be = SoftBackend(loop, [pid], [LOG_T[j] for j in t_idx])
    X = be.start()
    be.set_rows(np.arange(be.n))
    for _ in range(64):
        X, _ = be.step(X)
    hs = [10.0 ** -k for k in range(1, 11)]
    rng = np.random.default_rng(0)
    u = torch.as_tensor(rng.normal(size=tuple(X.shape)), dtype=torch.float64)
    u = u / u.norm(dim=-1, keepdim=True)
    rows = []
    for h in hs:
        rows += [X + h * u, X - h * u]
    rows += [X, X]
    n = be.n
    Xall = torch.cat(rows)
    be.set_rows(np.tile(np.arange(n), len(rows)))
    Y, _ = be.step(Xall)
    rec = {"prompt": pid, "h": hs, "cells": [], "git": git_state()}
    for c in range(n):
        D = [(Y[n * 2 * k + c] - Y[n * (2 * k + 1) + c]) / (2 * h) for k, h in enumerate(hs)]
        norms = [float(v.norm()) for v in D]
        rel = [float((D[k] - D[k + 1]).norm() / D[k + 1].norm()) for k in range(len(hs) - 1)]
        rep = float((Y[n * (len(rows) - 2) + c] - Y[n * (len(rows) - 1) + c]).abs().max())
        rec["cells"].append({"log10_T": LOG_T[t_idx[c]], "norm_D": norms, "successive_relative_change": rel,
                             "ratio_per_decade": [rel[k] / rel[k + 1] if rel[k + 1] > 0 else None
                                                  for k in range(len(rel) - 1)],
                             "min_successive_change": min(rel), "repeat_diff_max_abs": rep,
                             "state_norm": float(X[c].norm())})
        print(json.dumps(rec["cells"][-1]), flush=True)
    write_json(out / "precision.json", rec)


# ----------------------------------------------------------------------------------------------- analysis

def load_loop_runs(dirs, prompts, log_t):
    """Per prompt: labels per temperature, T_H and the cell records, from the loop batches of the given run
    directories (each prompt must appear exactly once)."""
    found = {}
    for d in dirs:
        out = OUT_ROOT / d
        for jp in sorted(out.glob("loop-b[0-9][0-9].json")):
            rec = json.loads(jp.read_text())
            z = np.load(jp.with_suffix(".npz"), allow_pickle=False)
            for ent in rec["first_step_entropy"]:
                p = int(ent["prompt"])
                if p in found:
                    raise SystemExit(f"prompt {p} appears in more than one loop batch")
                cells = sorted([c for c in rec["cells"] if c["prompt"] == p], key=lambda c: c["t_index"])
                if [c["t_index"] for c in cells] != list(range(len(log_t))):
                    raise SystemExit(f"prompt {p}: loop cells do not cover the grid")
                found[p] = {"labels": [c["label"] for c in cells], "cells": cells, "entropy": ent,
                            "gate_failures": rec["gate_failures"],
                            "virtual_retirement_mismatches": rec["virtual_retirement_mismatches"],
                            "file": str(jp.relative_to(OUT_ROOT)), "npz_keys": list(z.files)}
    missing = [p for p in prompts if p not in found]
    return found, missing


def load_samples(dirs, prompts):
    found = {}
    for d in dirs:
        for p in prompts:
            f = OUT_ROOT / d / f"samples-p{p:02d}.npz"
            if f.exists():
                if p in found:
                    raise SystemExit(f"prompt {p} sampled in more than one run")
                z = np.load(f, allow_pickle=False)
                found[p] = {"tokens": z["tokens"], "mean_entropy": z["mean_entropy"],
                            "gate_failures": json.loads(str(z["gate_failures"])), "T_H_float32": float(z["T_H_float32"]),
                            "file": str(f.relative_to(OUT_ROOT))}
    return found, [p for p in prompts if p not in found]


def analyze_split(loops, samples, prompts, log_t, window, qmax, h_target, n_boot=N_BOOT):
    """Per-prompt onsets and the endpoints for one split (see the module docstring)."""
    log_t = list(log_t)
    rows = []
    frac_all, labels_all = [], []
    for p in prompts:
        lp, sp = loops[p], samples[p]
        toks = sp["tokens"]
        q = smallest_period(toks, window, qmax)                       # (temps, samples)
        frac = (q > 0).mean(axis=1)
        qd = smallest_period(toks, window, max(qmax, min(REP_QMAX_DIAG, window - 1)))
        tail = periodic_tail(toks, qd)
        tc = chaos_onset(log_t, lp["labels"])
        tn = chaos_onset(log_t, ["non-periodic" if lab != "periodic" else lab for lab in lp["labels"]],
                         target="non-periodic")
        tr = rep_onset(log_t, frac)
        th = lp["entropy"]["T_H"] if h_target == H_TARGET else entropy_temperature_from_record(lp, h_target)
        rows.append({**prompt_record(p), "labels": lp["labels"], "T_chaos": tc, "T_nonperiodic": tn, "T_rep": tr,
                     "log10_T_H": float(np.log10(th)), "T_H": float(th),
                     "log10_T_H_masked": float(np.log10(lp["entropy"]["T_H_masked"])),
                     "log10_T_H_float32_sampler": float(np.log10(sp["T_H_float32"])),
                     "first_step_entropy_by_T": lp["entropy"]["entropy_by_T"],
                     "repeat_fraction": [float(f) for f in frac],
                     "repeat_periods": {str(int(k)): int(v) for k, v in zip(*np.unique(q[q > 0], return_counts=True))},
                     "diag_periods_up_to_48": {str(int(k)): int(v) for k, v in
                                               zip(*np.unique(qd[qd > 0], return_counts=True))},
                     "median_tail_length_by_T": [float(np.median(tail[j])) for j in range(len(log_t))],
                     "fraction_period_up_to_48_by_T": [float(f) for f in (qd > 0).mean(axis=1)],
                     "fraction_tail_at_least_64_by_T": [float(f) for f in (tail >= 64).mean(axis=1)],
                     "first_irregular_index": next((j for j, lab in enumerate(lp["labels"]) if lab == "irregular"),
                                                   None),
                     "virtual_retirement_mismatches": lp["virtual_retirement_mismatches"],
                     "mean_sampling_entropy_by_T": [float(sp["mean_entropy"][j].mean()) for j in range(len(log_t))],
                     "lyapunov_by_T": [c["lyapunov"] for c in lp["cells"]],
                     "period_by_T": [c["period"] for c in lp["cells"]],
                     "gate_failures": {"loop": lp["gate_failures"], "samples": sp["gate_failures"]}})
        frac_all.append(frac)
        labels_all.append(lp["labels"])
    x = np.array([r["T_chaos"]["analysis_value"] for r in rows])
    y = np.array([r["T_rep"]["analysis_value"] for r in rows])
    z = np.array([r["log10_T_H"] for r in rows])
    n = len(rows)
    openers = [r["opener_index"] for r in rows]
    systems = [r["system_index"] for r in rows]
    res = {"n_prompts": n, "prompts": rows}
    ent = np.array([r["first_step_entropy_by_T"] for r in rows])
    res["first_step_entropy_range"] = {"min_over_grid": float(ent.min()), "max_over_grid": float(ent.max()),
                                       "h_target": h_target,
                                       "prompts_reaching_target_within_grid": int(np.sum((ent[:, 0] <= h_target)
                                                                                         & (ent[:, -1] >= h_target)))}
    res["virtual_retirement_mismatches"] = sum(len(r["virtual_retirement_mismatches"]) for r in rows)
    res["censoring"] = {"T_chaos_never_chaotic": [r["prompt"] for r in rows if r["T_chaos"]["censored"]],
                        "T_rep_below_grid": [r["prompt"] for r in rows if r["T_rep"]["censored"] == "below"],
                        "T_rep_above_grid": [r["prompt"] for r in rows if r["T_rep"]["censored"] == "above"],
                        "T_rep_nonmonotone": [r["prompt"] for r in rows if r["T_rep"]["monotone"] is False]}
    if n < 3:
        res["primary"] = {"note": f"only {n} prompts"}
        return res
    sp_stat = lambda idx: spearman(x[idx], y[idx])  # noqa: E731
    res["primary"] = {"definition": "Spearman correlation of log10 T_chaos and log10 T_rep over prompts; censored "
                                    "values one grid step beyond the grid edge",
                      "spearman": spearman(x, y), "bootstrap": bootstrap(sp_stat, n, n_boot),
                      "cluster_bootstrap_openers": bootstrap(sp_stat, n, n_boot, clusters=openers),
                      "cluster_bootstrap_systems": bootstrap(sp_stat, n, n_boot, clusters=systems),
                      "permutation_p": permutation_p(x, y, n_boot)}
    keep = np.array([r["T_chaos"]["censored"] is None and r["T_rep"]["censored"] is None for r in rows])
    step = grid_step(log_t)
    xe = np.where(x > log_t[-1], log_t[-1], x)
    ye = np.clip(y, log_t[0], log_t[-1])
    res["primary"]["sensitivity"] = {
        "uncensored_only": {"n": int(keep.sum()), "spearman": spearman(x[keep], y[keep]) if keep.sum() >= 3 else None},
        "censored_at_grid_edge": {"spearman": spearman(xe, ye)},
        "T_nonperiodic_instead_of_T_chaos": {"spearman": spearman(
            np.array([r["T_nonperiodic"]["analysis_value"] for r in rows]), y)},
        "grid_step": step}
    pc_stat = lambda idx: partial_spearman(x[idx], y[idx], z[idx])  # noqa: E731
    res["secondary_partial"] = {"definition": "partial Spearman correlation of log10 T_chaos and log10 T_rep given "
                                              "log10 T_H (Pearson partial correlation of the ranks)",
                                "h_target_nats": h_target,
                                "partial_spearman": partial_spearman(x, y, z), "bootstrap": bootstrap(pc_stat, n, n_boot),
                                "spearman_T_H_T_rep": spearman(z, y),
                                "bootstrap_T_H_T_rep": bootstrap(lambda idx: spearman(z[idx], y[idx]), n, n_boot),
                                "spearman_T_H_T_chaos": spearman(z, x),
                                "bootstrap_T_H_T_chaos": bootstrap(lambda idx: spearman(z[idx], x[idx]), n, n_boot),
                                "T_H_range_log10": [float(z.min()), float(z.max())]}
    frac_all, labels_all = np.array(frac_all), np.array(labels_all)
    diff, per_t = stratified_difference(frac_all, labels_all)
    for row, lt in zip(per_t, log_t):
        row["log10_T"] = lt
    res["secondary_matched"] = {"definition": "per temperature, mean repetition fraction over chaotic cells minus the "
                                              "mean over periodic cells; combined with weights n_c n_p / (n_c + n_p)",
                                "difference": diff, "per_temperature": per_t,
                                "bootstrap": bootstrap(lambda idx: stratified_difference(frac_all, labels_all, idx)[0],
                                                       n, n_boot),
                                "temperatures_with_both": int(sum("difference" in r for r in per_t))}
    return res


def entropy_temperature_from_record(lp, h_target):
    """T_H at another entropy level, from the float64 first-step logits stored with the prompt's loop batch."""
    z = np.load(OUT_ROOT / lp["file"].replace(".json", ".npz"), allow_pickle=False)
    pos = [int(p) for p in z["batch_prompts"]].index(int(lp["entropy"]["prompt"]))
    return entropy_temperature(z["first_step_logits"][pos], h_target)


def cmd_analyze(args):
    choices = json.loads(Path(args.choices).read_text()) if args.choices else {}
    window = int(choices.get("rep_window", REP_WINDOW))
    qmax = int(choices.get("rep_qmax", REP_QMAX))
    h_target = float(choices.get("h_target", H_TARGET))
    prompts = [int(p) for p in (args.prompt_ids or e5_prompt_ids(args.split))]
    loops, miss_l = load_loop_runs(args.loop_runs, prompts, LOG_T)
    samples, miss_s = load_samples(args.sample_runs, prompts)
    if miss_l or miss_s:
        print(json.dumps({"missing_loop": miss_l, "missing_samples": miss_s}))
        if not args.partial:
            raise SystemExit("missing prompts; --partial analyzes the prompts present (provisional)")
        prompts = [p for p in prompts if p in loops and p in samples]
    res = analyze_split(loops, samples, prompts, LOG_T, window, qmax, h_target, args.n_boot)
    res.update({"split": args.split, "loop_runs": args.loop_runs, "sample_runs": args.sample_runs,
                "rule": {"window": window, "qmax": qmax, "h_target": h_target}, "choices_file": args.choices,
                "provisional": bool(miss_l or miss_s), "git": git_state()})
    out = OUT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"analysis-{args.split}.json", res)
    brief = {k: res.get(k) for k in ("n_prompts", "censoring")}
    if "spearman" in res.get("primary", {}):
        brief["primary"] = {k: res["primary"][k] for k in ("spearman", "bootstrap", "permutation_p")}
        brief["partial"] = {k: res["secondary_partial"][k] for k in ("partial_spearman", "spearman_T_H_T_rep",
                                                                     "spearman_T_H_T_chaos")}
        brief["matched"] = res["secondary_matched"]["difference"]
    print(json.dumps(brief, indent=1))
    for r in res["prompts"]:
        print(json.dumps({"prompt": r["prompt"], "T_chaos": r["T_chaos"]["value"], "T_rep": r["T_rep"]["value"],
                          "rep_cens": r["T_rep"]["censored"], "log10_T_H": round(r["log10_T_H"], 3),
                          "labels": "".join(lab[0] for lab in r["labels"]),
                          "frac": "".join(str(min(9, int(f * 10))) for f in r["repeat_fraction"])}))


def cmd_summarize(args):
    """Receipt: settings, prompts, choices, deviations, results and the SHA-256 of every raw file of the runs."""
    rec = json.loads(Path(args.template).read_text()) if args.template else {}
    files = []
    for d in args.runs:
        for f in sorted((OUT_ROOT / d).rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                files.append({"path": str(f.relative_to(OUT_ROOT)), "bytes": f.stat().st_size,
                              "sha256": sha256_file(f)})
    for split, path in (("development", args.dev_analysis), ("heldout", args.heldout_analysis)):
        if path:
            rec.setdefault("results", {})[split] = json.loads((OUT_ROOT / path).read_text())
    rec.update({"storage_root": str(OUT_ROOT), "raw_files": files, "prompt_lists": prompt_lists(),
                "prompt_lists_sha256": prompt_lists_hash(), "git_summarize": git_state(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(Path(args.receipt), rec)
    print(json.dumps({"receipt": args.receipt, "raw_files": len(files)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--transient", type=int, default=TRANSIENT)
    lp.add_argument("--measured", type=int, default=MEASURED)
    lp.add_argument("--retire", action="store_true", help="remove cells verified periodic to 1e-12 (see docstring)")
    lp.add_argument("--batches", type=int, nargs="*", default=None)
    sp = sub.add_parser("sample")
    sp.add_argument("--samples", type=int, default=N_SAMPLES)
    sp.add_argument("--tokens", type=int, default=N_TOKENS)
    pr = sub.add_parser("precision")
    pr.add_argument("--t-index", type=int, nargs="+", default=[6, 17, 21])
    for p in (lp, sp, pr):
        p.add_argument("--split", choices=("dev", "heldout"), default="dev")
        p.add_argument("--prompt-ids", type=int, nargs="*", default=None, help="override (smoke runs)")
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--split", choices=("dev", "heldout"), required=True)
    an.add_argument("--prompt-ids", type=int, nargs="*", default=None)
    an.add_argument("--loop-runs", nargs="+", required=True)
    an.add_argument("--sample-runs", nargs="+", required=True)
    an.add_argument("--choices", default=None)
    an.add_argument("--partial", action="store_true")
    an.add_argument("--n-boot", type=int, default=N_BOOT)
    an.add_argument("--out", required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("--runs", nargs="+", required=True)
    sm.add_argument("--dev-analysis", default=None)
    sm.add_argument("--heldout-analysis", default=None)
    sm.add_argument("--template", default=None, help="JSON with the receipt's text fields (choices, deviations)")
    sm.add_argument("--receipt", default=str(RECEIPT))
    args = ap.parse_args()
    {"loop": cmd_loop, "sample": cmd_sample, "precision": cmd_precision, "analyze": cmd_analyze,
     "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
