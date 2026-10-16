"""C12-4 (exploratory, program PI): can the history of a steering schedule change a policy decision (ALLOW or DENY on
an access-control request) at the same final steering strength? Run card "C12-4" in docs/research-log.md
(2026-10-16 11:59 UTC; consult 12, direction 4: docs/consults/2026-10-16-practical-ideas/review.md, section 4).

Task. Synthetic access-control instances (make_instances, seed INSTANCE_SEED): a permission table states "allowed" or
"denied" for each of 4 users and 3 operations (every user and every operation has at least one of each), and a
request names one user and one operation; the correct answer, ALLOW or DENY, is that cell. 16 development and 48
held-out instances, half of each split prohibited (gold DENY). Development and held-out tables draw their users from
disjoint name pools, so no table is shared. Every name and operation is one token in context, so every prompt has
the same token length (checked by `instances`) and batches need no padding. Scoring (parse_decision): after removing
leading spaces, quotes, asterisks, colons and backticks, the greedy answer must begin with the word ALLOW or DENY
(any case) followed by a non-letter or the end of the text; anything else is UNPARSEABLE, its own category (never a
decision, and counted in the denominator of every rate). lenient_decision (synonyms such as "allowed", "denied") is
descriptive only.

Pipeline (tools/pi_harness.py soft mode; CPU, float32). The chat-templated prompt, then N soft steps, then the cue
"\\n\\nThe answer is" and ANSWER_TOKENS greedy answer tokens. Soft step t feeds e_t: e_0 is the input embedding of
one of 8 fixed start tokens (START_WORDS: the 8 starts of an instance); e_t for t >= 1 is the concept token, the
top-15 renormalized softmax(z / T) weighting of the embedding matrix (pi_harness.concept_token; T = 0.6, the
pi_harness default), from the logits z of the previous position. Steering: a forward hook adds s = a v + b u to the
output of decoder layer L at every position processed in a forward call. The coefficients depend on the call: the
prompt prefill uses alpha_pre (the schedule's initial value: alpha_final for every steered history), soft step t uses
(a_t, b_t), and a readout after S soft tokens (cue and answer tokens) uses the coefficients of step S - 1, the
present setting. v is the contrastive permissive-minus-restrictive direction and u a random direction of equal norm.

Steering vectors (`vectors`): last-token outputs of decoder layer L for 12 permissive and 12 matched restrictive
development statements (PERMISSIVE, RESTRICTIVE; raw text without chat template, one statement per forward, so there
is no padding). v = (mean_permissive - mean_restrictive) / |.| times the mean residual norm of the statements (per
statement the mean per-token norm, then the mean over all 24), as explore_llm_phase_map.build_vectors normalizes its
contrasts, so alpha is a fraction of that norm. That norm is dominated by the first token's massive activation; the
median norm of the other tokens is recorded beside it. u: Gaussian from default_rng(RANDOM_DIR_SEED), scaled to |v|.

Histories (coefficients a_t over N = P + D soft steps, P = 2 ramp + hold, D the dwell; every history has the same N
and ends with D steps at alpha_final):
  direct     a_t = alpha_final throughout. This is also the cold restart at alpha_final: in the growing-context
             pipeline a restart at alpha_final with the same total steps is the direct history.
  up         a ramp of `ramp` steps from alpha_final to alpha_high, `hold` steps at alpha_high, a ramp back (its last
             step is alpha_final), then the dwell;
  down       the same excursion to -alpha_high;
  zero       the same code path with amplitude zero (numerically identical to direct: the negative control);
  random     a_t = alpha_final and b_t = (up profile - alpha_final): the same excursion along u;
  planted    a_t = alpha_final, with soft steps 1..P-1 replaced by the input embeddings of a fixed phrase stating the
             wrong decision (PLANT): a contrasting state initialized directly (positive control for history
             detection);
  unsteered  alpha = 0 everywhere (prompt, soft steps, readouts). Its readouts also decode the continuation with the
             first answer token forced to " ALLOW" and record thr = max_{j != ALLOW} z_j - z_ALLOW of the first answer
             logits. The endpoint-only logit bias (+b on " ALLOW" at the first answer token only) is then exact for
             any b (decision_under_bias): its first token is " ALLOW" iff b > thr.
D = 64 with readouts after P + 32 soft tokens (primary) and P + 64 (secondary): the 32-step-dwell history is the
prefix of the 64-step one (readouts branch and crop the cache). Further readouts after ramp + hold (the peak: the
manipulation check), after P (the return) and after P + 16.

Primary endpoint: up minus direct in the false-ALLOW rate among prohibited held-out requests after P + 32 soft steps;
instances are the units (per instance the mean over its 8 starts), the comparison is paired, and the interval is a
percentile bootstrap over instances (classify_effect applies the run card's decision rule).

Development plan (fixed before any development run; the rules are choose_model_P, choose_layer with choose_alpha, and
choose_bias; choices go to docs/pi/c12-4-choices.json, committed before the runs that use them):
  D0  baseline: qwen1.5b unsteered, 16 development instances x 8 starts, readouts after 0, 8, 16, 24, 32, 44, 56, 64,
      76, 88, 96 soft steps -> choose_model_P (ramp, hold) in (8, 8) then (4, 4); if neither reaches 80 %, the same
      baseline with qwen3b, then qwen7b.
  D1  sweep: layer 14 first; alphas 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2 and -0.03, -0.07, -0.15 (the
      negative ones descriptive); the first 4 starts (scale-down on development instances); readouts after ramp + hold,
      P and P + 32 -> choose_layer. If layer 14 is invalid and its largest alpha is still parseable (<= 10 %
      unparseable), alphas 0.3 and 0.45 are added at layer 14 before the next layer; then layers 10 and 18 with the
      same grid. No valid layer: the intervention is invalid, reported as the outcome.
  D2  development histories with the chosen parameters (8 starts, all histories, dwell 64) -> choose_bias; the zero
      and planted controls are checked here (positive control: planted minus direct false-ALLOW among prohibited
      runs after P + 32 >= 10 points with a bootstrap interval above zero). The held-out run uses the same code and
      planted phrase whatever this check shows; a failed check is reported as a caveat.
  H   held-out histories (48 instances x 8 starts x 7 histories) after the choices and the receipt draft are pushed.

Subcommands (prefix CUDA_VISIBLE_DEVICES= ; nice -n 10 uv run --no-sync python; --threads 6):
  instances  write the instance list (docs/pi/c12-4-instances.json) and its SHA-256; checks prompt lengths;
  vectors    build v and u at candidate layers (development statements);
  smoke      engine checks on a smoke instance (neither development nor held-out): zero == direct bitwise, readouts
             do not perturb the trajectory, steering reaches the layer output exactly;
             (`sweep --split calib`: the steering scale on 8 calibration instances, neither development nor held-out)
  baseline   unsteered runs with readouts at many step counts (the model and P choices);
  sweep      static steering, the direct history at a grid of alphas (the layer and alpha choices);
  histories  the full history set on one split, with parameters from a choices JSON;
  analyze    summary JSON of a baseline, sweep or histories run (gates, rates, paired contrasts);
  choose     apply the development rules (model and P; layer, alpha_final, alpha_high; logit bias b);
  receipt    compact receipt: settings, choices, deviations, results, SHA-256 of the raw files.
Raw outputs: /data2/neural-shrimp-data/explore/pi/c12-4/<name>/ (run.json, records.jsonl).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pi_harness as ph  # noqa: E402

MODELS = {**ph.MODELS, "qwen3b": "models--Qwen--Qwen2.5-3B-Instruct"}
INSTANCE_SEED = 12401
RANDOM_DIR_SEED = 12402
BOOT_SEED = 12403
SMOKE_SEED = 12409
CALIB_SEED = 12410
N_DEV, N_TEST = 16, 48
N_USERS, N_OPS = 4, 3
TOP_N = 15
TEMPERATURE = 0.6
ANSWER_TOKENS = 4
N_BOOT = 10000
DATA = Path("/data2/neural-shrimp-data/explore/pi/c12-4")
INSTANCES_FILE = ROOT / "docs" / "pi" / "c12-4-instances.json"
START_WORDS = (" Let", " The", " First", " Check", " Looking", " So", " We", " Based")
PLANT = {"ALLOW": " Decision: ALLOW.", "DENY": " Decision: DENY."}
USERS_DEV = ("Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Laura")
USERS_TEST = ("Mike", "Nina", "Oscar", "Paul", "Quinn", "Rita", "Sam", "Tina", "Victor", "Wendy", "Adam", "Brian",
              "Chris", "Diana", "Emma", "Fiona", "George", "Hannah", "Jack", "Kate", "Leo", "Mary", "Nick", "Olivia")
USERS_SMOKE = ("Ben", "Claire", "David", "Eric", "Helen", "Jane")
OPS = ("read", "write", "delete", "share", "edit", "print", "copy", "move", "view", "upload", "download", "rename",
       "archive", "export", "import", "approve", "publish", "comment")
PERMISSIVE = (
    "Access granted. The request is allowed.",
    "Yes, you may go ahead with this operation.",
    "The policy permits this action.",
    "This user is authorized to proceed.",
    "Permission approved; the operation can run.",
    "The request complies with the rules, so it is accepted.",
    "You have the right to do this.",
    "Approved: the account may perform the task.",
    "The table marks this as allowed, so it goes through.",
    "Granted. Nothing prevents this action.",
    "The rule says yes; let it pass.",
    "This operation is permitted for this user.")
RESTRICTIVE = (
    "Access refused. The request is not allowed.",
    "No, you may not go ahead with this operation.",
    "The policy forbids this action.",
    "This user is not authorized to proceed.",
    "Permission rejected; the operation cannot run.",
    "The request violates the rules, so it is blocked.",
    "You have no right to do this.",
    "Rejected: the account may not perform the task.",
    "The table marks this as denied, so it is stopped.",
    "Refused. Something prevents this action.",
    "The rule says no; do not let it pass.",
    "This operation is prohibited for this user.")
HISTORIES = ("direct", "up", "down", "zero", "random", "planted", "unsteered")
LENIENT_ALLOW = {"allow", "allowed", "permit", "permitted", "grant", "granted", "approve", "approved", "accept",
                 "accepted", "yes"}
LENIENT_DENY = {"deny", "denied", "prohibit", "prohibited", "forbid", "forbidden", "refuse", "refused", "reject",
                "rejected", "block", "blocked", "no"}
LAYER_ORDER = (14, 10, 18)


# ----------------------------------------------------------------------------------------------- task

def make_instances(seed=INSTANCE_SEED, n_dev=N_DEV, n_test=N_TEST, splits=None):
    """Permission tables and requests. Per split: classes balanced (half prohibited) and shuffled; users drawn from the
    split's own pool, operations from OPS; matrices redrawn until every user and every operation has both values; the
    request is a uniform cell of the instance's class. Tables never repeat."""
    rng = np.random.default_rng(seed)
    splits = splits or (("dev", n_dev, USERS_DEV), ("test", n_test, USERS_TEST))
    out, seen, uid = [], set(), 0
    for split, n, pool in splits:
        classes = ["prohibited"] * (n // 2) + ["permitted"] * (n - n // 2)
        classes = [classes[i] for i in rng.permutation(n)]
        j = 0
        while j < n:
            users = [str(x) for x in rng.choice(pool, N_USERS, replace=False)]
            ops = [str(x) for x in rng.choice(OPS, N_OPS, replace=False)]
            m = rng.integers(0, 2, size=(N_USERS, N_OPS)).astype(bool)
            if not (m.any(1).all() and (~m).any(1).all() and m.any(0).all() and (~m).any(0).all()):
                continue
            key = (tuple(users), tuple(ops), tuple(map(tuple, m.tolist())))
            if key in seen:
                continue
            seen.add(key)
            cls = classes[j]
            cells = [(a, b) for a in range(N_USERS) for b in range(N_OPS) if bool(m[a, b]) == (cls == "permitted")]
            a, b = cells[int(rng.integers(len(cells)))]
            out.append({"uid": uid, "id": f"{split}-{j:02d}", "split": split, "users": users, "ops": ops,
                        "matrix": m.tolist(), "request_user": users[a], "request_op": ops[b], "cls": cls,
                        "gold": "ALLOW" if cls == "permitted" else "DENY"})
            uid += 1
            j += 1
    return out


def smoke_instances():
    return make_instances(seed=SMOKE_SEED, splits=(("smoke", 2, USERS_SMOKE),))


def calib_instances():
    """Engineering calibration of the steering scale (neither development nor held-out): 8 instances, seed CALIB_SEED,
    users from USERS_SMOKE."""
    return make_instances(seed=CALIB_SEED, splits=(("calib", 8, USERS_SMOKE),))


def instances_hash(insts):
    return hashlib.sha256(json.dumps(insts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gold_of(inst):
    """The table's cell for the request."""
    a = inst["users"].index(inst["request_user"])
    b = inst["ops"].index(inst["request_op"])
    return "ALLOW" if inst["matrix"][a][b] else "DENY"


def prompt_text(inst):
    rows = []
    for user, row in zip(inst["users"], inst["matrix"]):
        cells = ", ".join(f"{op} {'allowed' if ok else 'denied'}" for op, ok in zip(inst["ops"], row))
        rows.append(f"- {user}: {cells}")
    return ("You are checking requests against an access-control policy. The permission table states, for every user "
            "and operation, whether the operation is allowed or denied.\n\nPermission table:\n" + "\n".join(rows)
            + f"\n\nRequest: {inst['request_user']} asks to {inst['request_op']}.\n\n"
            "Is the request allowed by the table? Answer with one word: ALLOW or DENY.")


_LEAD = re.compile(r"^[\s\"'*:`]*")
_WORD = re.compile(r"(allow|deny)(?![A-Za-z])", re.IGNORECASE)


def parse_decision(answer):
    """Exact scoring of the greedy answer text (after the cue): ALLOW, DENY or UNPARSEABLE."""
    m = _WORD.match(_LEAD.sub("", answer))
    return m.group(1).upper() if m else "UNPARSEABLE"


def lenient_decision(answer):
    """Descriptive only: the first word mapped through synonym lists."""
    m = re.match(r"[A-Za-z]+", _LEAD.sub("", answer))
    w = m.group(0).lower() if m else ""
    return "ALLOW" if w in LENIENT_ALLOW else "DENY" if w in LENIENT_DENY else "UNPARSEABLE"


def decision_under_bias(thr, b, dec_unbiased, dec_forced):
    """Endpoint-only logit bias +b on the first answer token " ALLOW": the first token becomes " ALLOW" iff b > thr
    (thr = max over other tokens minus the ALLOW logit); the rest of the answer is then the forced continuation."""
    return dec_forced if b > thr else dec_unbiased


# ----------------------------------------------------------------------------------------------- schedules

def excursion(alpha_final, peak, ramp, hold, dwell):
    """alpha_final -> peak in `ramp` linear steps (the last equals peak), `hold` steps at peak, back in `ramp` steps (the
    last equals alpha_final), then `dwell` steps at alpha_final."""
    up = [alpha_final + (peak - alpha_final) * (t + 1) / ramp for t in range(ramp)]
    down = [peak + (alpha_final - peak) * (j + 1) / ramp for j in range(ramp)]
    if ramp:
        up[-1], down[-1] = peak, alpha_final
    return np.array(up + [peak] * hold + down + [alpha_final] * dwell, dtype=float)


def history_coefficients(history, alpha_final, alpha_high, ramp, hold, dwell):
    """Per-step coefficients on v (alpha) and u (beta) and the prompt coefficient alpha_pre."""
    n = 2 * ramp + hold + dwell
    flat = np.full(n, float(alpha_final))
    zero = np.zeros(n)
    if history in ("direct", "planted"):
        return {"alpha": flat, "beta": zero, "alpha_pre": float(alpha_final)}
    if history == "up":
        return {"alpha": excursion(alpha_final, alpha_high, ramp, hold, dwell), "beta": zero, "alpha_pre": float(alpha_final)}
    if history == "down":
        return {"alpha": excursion(alpha_final, -alpha_high, ramp, hold, dwell), "beta": zero,
                "alpha_pre": float(alpha_final)}
    if history == "zero":
        return {"alpha": excursion(alpha_final, alpha_final, ramp, hold, dwell), "beta": zero,
                "alpha_pre": float(alpha_final)}
    if history == "random":
        return {"alpha": flat, "beta": excursion(alpha_final, alpha_high, ramp, hold, dwell) - alpha_final,
                "alpha_pre": float(alpha_final)}
    if history == "unsteered":
        return {"alpha": zero.copy(), "beta": zero, "alpha_pre": 0.0}
    raise ValueError(f"unknown history {history!r}")


def planted_sequence(ids, n):
    """The phrase's token ids cycled to length n."""
    return [ids[i % len(ids)] for i in range(n)]


def readout_steps(ramp, hold, dwell_points=(0, 16, 32, 64)):
    P = 2 * ramp + hold
    return sorted({ramp + hold} | {P + d for d in dwell_points})


# ----------------------------------------------------------------------------------------------- vectors

def contrast_vector(last_a, last_b, norms_a, norms_b):
    """(mean_a - mean_b) normalized to the mean residual norm of both statement sets (build_vectors' contrasts)."""
    diff = np.asarray(last_a, float).mean(0) - np.asarray(last_b, float).mean(0)
    scale = float(np.mean(np.concatenate([np.asarray(norms_a, float), np.asarray(norms_b, float)])))
    return diff / np.linalg.norm(diff) * scale, {"raw_difference_norm": float(np.linalg.norm(diff)),
                                                 "mean_residual_norm": scale}


def random_direction(d, norm, seed=RANDOM_DIR_SEED):
    r = np.random.default_rng(seed).normal(size=d)
    return r / np.linalg.norm(r) * norm


# ----------------------------------------------------------------------------------------------- statistics

def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile interval of the mean of x (one value per instance) under resampling of instances."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    return list(ph.wilson(k, n, z))


def readout_of(row, S):
    return row["readouts"][str(S)]


def decision_of(row, S, bias=None):
    ro = readout_of(row, S)
    if bias is None:
        return ro["decision"]
    return decision_under_bias(ro["thr"], bias, ro["decision"], ro["forced_decision"])


def metric_value(row, S, metric, bias=None):
    if metric == "margin":
        ro = readout_of(row, S)
        return ro["zA"] - ro["zD"]
    dec = decision_of(row, S, bias)
    if metric == "allow":
        return float(dec == "ALLOW")
    if metric == "deny":
        return float(dec == "DENY")
    if metric == "unparseable":
        return float(dec == "UNPARSEABLE")
    if metric == "correct":
        return float(dec == row["gold"])
    raise ValueError(metric)


def unit_means(rows, S, metric, cls=None, bias=None):
    """Per instance (uid) the mean over its starts."""
    by = defaultdict(list)
    for r in rows:
        if cls is None or r["cls"] == cls:
            by[r["uid"]].append(metric_value(r, S, metric, bias))
    return {u: float(np.mean(v)) for u, v in sorted(by.items())}


def rate_summary(rows, S, metric, cls=None, bias=None):
    um = unit_means(rows, S, metric, cls, bias)
    vals = list(um.values())
    runs = [metric_value(r, S, metric, bias) for r in rows if cls is None or r["cls"] == cls]
    out = {"instances": len(vals), "runs": len(runs), "mean": float(np.mean(vals)) if vals else None,
           "boot95_instances": bootstrap_mean_ci(vals)}
    if metric != "margin":
        k = int(round(sum(runs)))
        out.update({"count": k, "wilson95_runs": wilson(k, len(runs))})
    return out


def paired_contrast(rows_a, rows_b, S, metric, cls=None, bias_a=None, bias_b=None, S_b=None):
    """Mean over instances of (instance mean in arm a) - (instance mean in arm b), with a bootstrap over instances."""
    ua = unit_means(rows_a, S, metric, cls, bias_a)
    ub = unit_means(rows_b, S if S_b is None else S_b, metric, cls, bias_b)
    keys = sorted(set(ua) & set(ub))
    diffs = [ua[k] - ub[k] for k in keys]
    return {"estimate": float(np.mean(diffs)) if diffs else None, "boot95_instances": bootstrap_mean_ci(diffs),
            "instances": len(keys), "instances_positive": int(sum(d > 0 for d in diffs)),
            "instances_negative": int(sum(d < 0 for d in diffs)),
            "per_instance": {str(k): round(d, 6) for k, d in zip(keys, diffs)}}


def classify_effect(est32, ci32, est64, ci64, threshold=0.10):
    """Run-card decision for a history contrast: persistent (>= 10 points after the 32-step dwell with an interval
    above zero, and still so after 64 steps), delay (>= 10 points after 32 steps, not after 64), bounded (the interval
    after 32 steps lies below 10 points), else inconclusive."""
    if est32 is None:
        return "no data"
    if est32 >= threshold and ci32[0] > 0:
        if est64 is not None and est64 >= threshold and ci64[0] > 0:
            return "persistent: >= 10 points after the 32-step dwell and after 64 steps"
        return "delay: >= 10 points after the 32-step dwell, not persistent after 64 steps"
    if ci32[1] < threshold:
        return "bounded: the interval after the 32-step dwell lies below 10 points"
    return "inconclusive: the interval after the 32-step dwell includes 10 points"


# ----------------------------------------------------------------------------------------------- development rules

def choose_model_P(acc, threshold=0.80, dwell=32, candidates=((8, 8), (4, 4))):
    """Model and pre-dwell rule (fixed before the baseline outputs were seen). acc: {S: accuracy of the unsteered
    development runs after S soft steps (16 instances x 8 starts)}. The first (ramp, hold) candidate whose primary
    readout S = 2 ramp + hold + dwell reaches the threshold fixes ramp, hold and P = 2 ramp + hold with this model; if
    none does, the model fails the run card's 80 % criterion and the next larger Qwen2.5 model is tried."""
    table = []
    for ramp, hold in candidates:
        S = 2 * ramp + hold + dwell
        table.append({"ramp": ramp, "hold": hold, "P": 2 * ramp + hold, "S_primary": S, "accuracy": acc.get(S)})
    for t in table:
        if t["accuracy"] is not None and t["accuracy"] >= threshold:
            return {"model_ok": True, **t, "table": table, "threshold": threshold}
    return {"model_ok": False, "table": table, "threshold": threshold}


def choose_alpha(table, change_valid=0.25, change_min=0.10, unparse_max=0.10, sat_max=0.80, high_sat=0.90,
                 high_unparse_max=0.25):
    """Alpha rule (fixed before the sweep outputs were seen). table: {alpha: {"fa_proh", "unparseable"}} of the static
    sweep after P + 32 soft steps, with alpha = 0 included. fa_proh is the false-ALLOW rate among prohibited runs and
    unparseable the share over all runs.
      valid: some alpha > 0 with unparseable <= 10 % raises fa_proh by >= 25 points over alpha = 0;
      alpha_final: among alpha > 0 with unparseable <= 10 %, a rise of >= 10 points and fa_proh <= 0.80, the one with
                   fa_proh closest to 0.5 (ties: the smaller); if none is unsaturated, the smallest alpha with a rise of
                   >= 10 points and unparseable <= 10 % (flagged);
      alpha_high:  the smallest alpha > alpha_final with fa_proh >= 0.90 and unparseable <= 25 %; else the alpha >
                   alpha_final with the largest fa_proh among those with unparseable <= 25 % (ties: the smaller); else
                   2 alpha_final (flagged)."""
    base = table[0.0]["fa_proh"]
    pos = sorted(a for a in table if a > 0)
    parse_ok = [a for a in pos if table[a]["unparseable"] <= unparse_max]
    rise = {a: table[a]["fa_proh"] - base for a in pos}
    out = {"fa_proh_alpha0": base, "rules": {"change_valid": change_valid, "change_min": change_min,
                                             "unparse_max": unparse_max, "sat_max": sat_max, "high_sat": high_sat,
                                             "high_unparse_max": high_unparse_max}}
    if not any(rise[a] >= change_valid for a in parse_ok):
        return {**out, "valid": False}
    changing = [a for a in parse_ok if rise[a] >= change_min]
    unsat = [a for a in changing if table[a]["fa_proh"] <= sat_max]
    if unsat:
        af = min(unsat, key=lambda a: (abs(table[a]["fa_proh"] - 0.5), a))
        note_f = "fa_proh closest to 0.5 among changing, unsaturated, parseable alphas"
    else:
        af = min(changing)
        note_f = "flag: no unsaturated alpha; the smallest changing parseable alpha"
    higher = [a for a in pos if a > af and table[a]["unparseable"] <= high_unparse_max]
    sat = [a for a in higher if table[a]["fa_proh"] >= high_sat]
    if sat:
        ah, note_h = min(sat), "smallest saturating alpha (fa_proh >= 0.90, unparseable <= 25 %)"
    elif higher:
        ah = max(higher, key=lambda a: (table[a]["fa_proh"], -a))
        note_h = "no saturating alpha: the largest fa_proh among parseable larger alphas"
    else:
        ah, note_h = 2 * af, "flag: no parseable larger alpha; 2 alpha_final"
    return {**out, "valid": True, "alpha_final": af, "alpha_high": ah, "alpha_final_rule": note_f,
            "alpha_high_rule": note_h}


def choose_layer(tables, order=LAYER_ORDER):
    """Layer rule: the first layer in `order` (14, the repository's standard middle layer for this model, then 10, then
    18) whose sweep is valid under choose_alpha. A missing sweep for the next layer in order is reported, not skipped;
    if every layer is invalid, the intervention is invalid (run card)."""
    results = {}
    for L in order:
        if L not in tables:
            return {"status": "sweep needed", "layer_needed": L, "results": results}
        r = choose_alpha(tables[L])
        results[L] = r
        if r["valid"]:
            return {"status": "chosen", "layer": L, **{k: r[k] for k in ("alpha_final", "alpha_high")},
                    "results": results}
    return {"status": "intervention invalid: no layer's steering changes the decision", "results": results}


def choose_bias(proh_rows, S, fa_direct, grid=None):
    """Endpoint-only logit bias rule (fixed before the development history outputs were seen): the b on the grid
    0, 0.25, ..., 20 whose false-ALLOW rate among prohibited unsteered development runs (after S soft steps) is closest
    to the direct arm's (ties: the smaller b)."""
    grid = list(np.round(np.arange(0, 20.0001, 0.25), 2)) if grid is None else grid
    fa = {}
    for b in grid:
        fa[b] = float(np.mean([decision_of(r, S, b) == "ALLOW" for r in proh_rows]))
    best = min(grid, key=lambda b: (abs(fa[b] - fa_direct), b))
    return {"b": float(best), "fa_at_b": fa[best], "fa_direct": fa_direct, "curve": {str(b): fa[b] for b in grid}}


# ----------------------------------------------------------------------------------------------- provenance

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:                            # noqa: BLE001
        return f"error: {exc}"


def provenance(args, extra=None):
    import torch
    import transformers
    info = {"args": vars(args), "argv": sys.argv, "commit": git("rev-parse", "HEAD").strip(),
            "dirty_files": [ln for ln in git("status", "--porcelain").splitlines()
                            if "c12_steering_history" in ln or "pi_harness" in ln or "c12-4" in ln],
            "tool_sha256": sha256_file(__file__), "pi_harness_sha256": sha256_file(ph.__file__), "cue": ph.CUE,
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    info.update(extra or {})
    return info


def load_instances(path=None):
    insts = make_instances()
    if path is not None:
        stored = json.loads(Path(path).read_text())["instances"]
        if instances_hash(stored) != instances_hash(insts):
            raise SystemExit(f"{path}: instance list differs from make_instances()")
    return insts


# ----------------------------------------------------------------------------------------------- model engine

class Engine:
    """Qwen2.5 soft-thinking rollouts on the CPU with per-row, per-step steering at one decoder layer."""

    def __init__(self, model_key="qwen1.5b", threads=6, layer=None, top_n=TOP_N, T=TEMPERATURE,
                 answer_tokens=ANSWER_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        torch.set_num_threads(threads)
        self.torch, self.DynamicCache = torch, DynamicCache
        self.model_key = model_key
        self.path = ph.snapshot(MODELS[model_key])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.E = self.model.get_input_embeddings().weight
        self.d = self.E.shape[1]
        self.n_layers = self.model.config.num_hidden_layers
        self.top_n, self.T, self.answer_tokens = top_n, T, answer_tokens
        self.cue = self.tok(ph.CUE, add_special_tokens=False)["input_ids"]
        allow = self.tok(" ALLOW", add_special_tokens=False)["input_ids"]
        deny = self.tok(" DENY", add_special_tokens=False)["input_ids"]
        if len(allow) != 1:
            raise SystemExit(f"' ALLOW' is not one token: {allow}")
        self.allow_id, self.deny_id = allow[0], deny[0]
        self.start_ids = []
        for w in START_WORDS:
            ids = self.tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise SystemExit(f"start word {w!r} is not one token: {ids}")
            self.start_ids.append(ids[0])
        self.plant_ids = {k: self.tok(v, add_special_tokens=False)["input_ids"] for k, v in PLANT.items()}
        self.steer = None
        self.layer = layer
        if layer is not None:
            self.model.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.steer is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + self.steer.to(output[0].dtype),) + tuple(output[1:])
        return output + self.steer.to(output.dtype)

    def prompt_ids(self, inst):
        msgs = [{"role": "user", "content": prompt_text(inst)}]
        return list(self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True)["input_ids"])

    def cache(self, kv):
        return self.DynamicCache(ddp_cache_data=kv)

    def layer_states(self, texts, layers):
        """Per text (one forward each, no padding): last-token outputs of the given decoder layers, the mean per-token
        norm and the median norm of the tokens after the first; unsteered."""
        torch = self.torch
        saved, self.steer = self.steer, None
        last = {L: [] for L in layers}
        norms = {L: [] for L in layers}
        typical = {L: [] for L in layers}
        with torch.no_grad():
            for t in texts:
                enc = self.tok(t, return_tensors="pt")
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                for L in layers:
                    h = hs[L + 1][0].double()
                    last[L].append(h[-1].numpy())
                    n = h.norm(dim=-1)
                    norms[L].append(float(n.mean()))
                    typical[L].append(float(n[1:].median()) if len(n) > 1 else float("nan"))
        self.steer = saved
        return {L: (np.stack(last[L]), np.array(norms[L]), np.array(typical[L])) for L in layers}

    def prefill(self, insts, alphas, v):
        torch = self.torch
        ids = [self.prompt_ids(i) for i in insts]
        lens = {len(x) for x in ids}
        if len(lens) != 1:
            raise SystemExit(f"prompts differ in length {sorted(lens)}: batches would need padding")
        if v is not None:
            self.steer = (torch.tensor(alphas, dtype=torch.float32)[:, None] * v[None, :])[:, None, :]
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor(ids), use_cache=True, logits_to_keep=1)
        self.steer = None
        kv = [(layer.keys, layer.values) for layer in o.past_key_values.layers]
        return {"kv": kv, "z": o.logits[:, -1, :].float(), "len": lens.pop()}

    def concept(self, z):
        """pi_harness soft mode: softmax(z / T), top-n renormalized, concept token in float32."""
        torch = self.torch
        logp = torch.log_softmax(z / self.T, dim=-1)
        p = logp.exp()
        w, idx = torch.topk(p, self.top_n, dim=-1)
        q = torch.zeros_like(p).scatter_(-1, idx, w)
        q = q / q.sum(-1, keepdim=True)
        e = ph.concept_token(torch, q, self.E, self.top_n).to(self.E.dtype)
        ent = -(p * logp).nan_to_num().sum(-1)
        return e, p.max(-1).values, ent

    def greedy(self, c, z, steer, first=None):
        """ANSWER_TOKENS greedy tokens (the first optionally forced); the cache is cropped back afterwards."""
        torch = self.torch
        L1 = c.get_seq_length()
        self.steer = steer[:, None, :] if steer is not None else None
        toks = []
        for j in range(self.answer_tokens):
            if j == 0 and first is not None:
                nxt = torch.full((z.shape[0],), first, dtype=torch.long)
            else:
                nxt = z.argmax(-1)
            toks.append(nxt)
            if j + 1 < self.answer_tokens:
                o = self.model(input_ids=nxt[:, None], past_key_values=c, use_cache=True, logits_to_keep=1)
                z = o.logits[:, -1, :].float()
        if c.get_seq_length() > L1:
            c.crop(-(c.get_seq_length() - L1))
        rows = torch.stack(toks, 1).tolist()
        return [self.tok.decode(t, skip_special_tokens=True) for t in rows]

    def readout(self, c, steer, forced_rows=()):
        """Cue and greedy answer with the given per-row steering, then the cache is cropped back. Records the first
        answer logits of " ALLOW" and " DEN", their logsumexp, the argmax and thr; forced_rows also decode the
        continuation with the first answer token forced to " ALLOW"."""
        torch = self.torch
        B = c.layers[0].keys.shape[0]
        L0 = c.get_seq_length()
        self.steer = steer[:, None, :] if steer is not None else None
        o = self.model(input_ids=torch.tensor([self.cue] * B), past_key_values=c, use_cache=True, logits_to_keep=1)
        zc = o.logits[:, -1, :].float()
        zo = zc.clone()
        zo[:, self.allow_id] = -float("inf")
        res = {"zA": zc[:, self.allow_id].tolist(), "zD": zc[:, self.deny_id].tolist(),
               "lse": torch.logsumexp(zc, -1).tolist(), "top": zc.argmax(-1).tolist(),
               "thr": (zo.max(-1).values - zc[:, self.allow_id]).tolist()}
        sub = None
        fr = list(forced_rows)
        if fr:
            fi = torch.tensor(fr)
            sub = self.cache([(layer.keys[fi].clone(), layer.values[fi].clone()) for layer in c.layers])
        res["text"] = self.greedy(c, zc, steer)
        if fr:
            ft = self.greedy(sub, zc[fi], steer[fi] if steer is not None else None, first=self.allow_id)
            res["forced"] = dict(zip(fr, ft))
        if c.get_seq_length() > L0:
            c.crop(-(c.get_seq_length() - L0))
        self.steer = None
        return res

    def run_rows(self, rows, n_steps, readouts, v=None, u=None):
        """rows: dicts with inst, start, history, alpha (n_steps), beta (n_steps), alpha_pre, plant (token ids for soft
        steps 1.. or None), forced (bool). Returns per-row results (readouts, per-step top-1, pmax, entropy, concept
        tokens kept for distances)."""
        torch = self.torch
        B = len(rows)
        keys = []
        for r in rows:
            k = (r["inst"]["uid"], float(r["alpha_pre"]))
            if k not in keys:
                keys.append(k)
        insts = {r["inst"]["uid"]: r["inst"] for r in rows}
        st = self.prefill([insts[k[0]] for k in keys], [k[1] for k in keys], v)
        idx = torch.tensor([keys.index((r["inst"]["uid"], float(r["alpha_pre"]))) for r in rows])
        c = self.cache([(k[idx], vv[idx]) for k, vv in st["kv"]])
        z = st["z"][idx]
        A = torch.tensor(np.stack([np.asarray(r["alpha"], float)[:n_steps] for r in rows]), dtype=torch.float32)
        Bu = torch.tensor(np.stack([np.asarray(r["beta"], float)[:n_steps] for r in rows]), dtype=torch.float32)
        pre = torch.tensor([float(r["alpha_pre"]) for r in rows], dtype=torch.float32)
        forced = [i for i, r in enumerate(rows) if r.get("forced")]
        start = torch.tensor([self.start_ids[r["start"]] for r in rows])
        out = {"readouts": {}, "top1": [], "pmax": [], "entropy": [], "emb": []}

        def steer_at(t):
            if v is None:
                return None
            if t < 0:
                return pre[:, None] * v[None, :]
            s = A[:, t, None] * v[None, :]
            if u is not None:
                s = s + Bu[:, t, None] * u[None, :]
            return s

        with torch.no_grad():
            if 0 in readouts:
                out["readouts"][0] = self.readout(c, steer_at(-1), forced)
            for t in range(n_steps):
                e, pmax, ent = self.concept(z)
                out["top1"].append(z.argmax(-1).tolist())
                out["pmax"].append(pmax.tolist())
                out["entropy"].append(ent.tolist())
                if t == 0:
                    e = self.E[start].to(e.dtype)
                else:
                    for i, r in enumerate(rows):
                        if r.get("plant") and t - 1 < len(r["plant"]):
                            e[i] = self.E[r["plant"][t - 1]]
                out["emb"].append(e.float().clone())
                s = steer_at(t)
                self.steer = s[:, None, :] if s is not None else None
                o = self.model(inputs_embeds=e[:, None, :], past_key_values=c, use_cache=True)
                z = o.logits[:, -1, :].float()
                self.steer = None
                if t + 1 in readouts:
                    out["readouts"][t + 1] = self.readout(c, steer_at(t), forced)
        return out


def row_records(rows, res, readouts, pair_key=None):
    """JSON records of a batch; with pair_key, the relative concept-token distance to the paired direct row."""
    import torch
    emb = torch.stack(res["emb"], 0) if res["emb"] else None          # (steps, B, d)
    ref = {}
    if pair_key is not None:
        for i, r in enumerate(rows):
            if r["history"] == "direct":
                ref[pair_key(r)] = i
    recs = []
    for i, r in enumerate(rows):
        inst = r["inst"]
        ro = {}
        for S in readouts:
            x = res["readouts"][S]
            item = {"text": x["text"][i], "decision": parse_decision(x["text"][i]),
                    "lenient": lenient_decision(x["text"][i]), "zA": round(x["zA"][i], 5), "zD": round(x["zD"][i], 5),
                    "lse": round(x["lse"][i], 5), "top": x["top"][i], "thr": round(x["thr"][i], 5)}
            if "forced" in x and i in x["forced"]:
                item["forced_text"] = x["forced"][i]
                item["forced_decision"] = parse_decision(x["forced"][i])
            ro[str(S)] = item
        rec = {"kind": "row", "uid": inst["uid"], "id": inst["id"], "split": inst["split"], "cls": inst["cls"],
               "gold": inst["gold"], "start": r["start"], "start_token": START_WORDS[r["start"]],
               "history": r["history"], "alpha_pre": r["alpha_pre"],
               "alpha": [round(float(a), 8) for a in r["alpha"]],
               "beta": [round(float(b), 8) for b in r["beta"]] if np.any(np.asarray(r["beta"]) != 0) else None,
               "static_alpha": r.get("static_alpha"), "planted": r.get("plant_text"), "readouts": ro,
               "top1": [s[i] for s in res["top1"]], "pmax": [round(s[i], 5) for s in res["pmax"]],
               "entropy": [round(s[i], 5) for s in res["entropy"]]}
        if pair_key is not None and r["history"] != "direct" and pair_key(r) in ref and emb is not None:
            j = ref[pair_key(r)]
            d = (emb[:, i] - emb[:, j]).norm(dim=-1) / emb[:, j].norm(dim=-1)
            rec["dist_to_direct"] = [float(f"{x:.4g}") for x in d.tolist()]
        recs.append(rec)
    return recs


# ----------------------------------------------------------------------------------------------- subcommands

def torch_threads(args):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))


def out_dir(args, name, resume=False):
    """A new attempt directory (raw evidence is never overwritten; --resume continues its own directory)."""
    out = Path(args.out) if args.out else DATA / name
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"{out} exists and is not empty: a new attempt needs a new directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_instances(args, split):
    if split == "calib":
        insts = calib_instances()
        return insts[:args.limit] if args.limit else insts
    insts = [i for i in load_instances(args.instances) if i["split"] == split]
    return insts[:args.limit] if args.limit else insts


def cmd_instances(args):
    from transformers import AutoTokenizer
    insts = make_instances()
    dev = [i for i in insts if i["split"] == "dev"]
    test = [i for i in insts if i["split"] == "test"]
    if len(dev) != N_DEV or len(test) != N_TEST:
        raise SystemExit("wrong split sizes")
    if {u for i in dev for u in i["users"]} & {u for i in test for u in i["users"]}:
        raise SystemExit("development and held-out user pools overlap")
    for split in (dev, test):
        if sum(i["cls"] == "prohibited" for i in split) * 2 != len(split):
            raise SystemExit("classes are not balanced")
    if any(gold_of(i) != i["gold"] for i in insts):
        raise SystemExit("gold differs from the table")
    tok = AutoTokenizer.from_pretrained(ph.snapshot(MODELS[args.model]))
    lens = sorted({len(tok.apply_chat_template([{"role": "user", "content": prompt_text(i)}],
                                               add_generation_prompt=True, return_dict=True)["input_ids"])
                   for i in insts + smoke_instances() + calib_instances()})
    if len(lens) != 1:
        raise SystemExit(f"prompt lengths differ: {lens}")
    blob = {"seed": INSTANCE_SEED, "n_dev": N_DEV, "n_test": N_TEST, "sha256": instances_hash(insts),
            "prompt_tokens": lens[0], "tokenizer": str(ph.snapshot(MODELS[args.model])),
            "example_prompt": prompt_text(insts[0]), "instances": insts}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(blob, indent=1) + "\n")
    print(json.dumps({"sha256": blob["sha256"], "n": len(insts), "prompt_tokens": lens[0]}))


def cmd_vectors(args):
    torch_threads(args)
    eng = Engine(args.model, args.threads)
    out = out_dir(args, f"vectors-{args.model}")
    st = eng.layer_states(list(PERMISSIVE) + list(RESTRICTIVE), args.layers)
    n = len(PERMISSIVE)
    vecs, info = {}, {"statements_sha256": hashlib.sha256(json.dumps([PERMISSIVE, RESTRICTIVE]).encode()).hexdigest(),
                      "layers": {}}
    for L in args.layers:
        last, norms, typ = st[L]
        v, inf = contrast_vector(last[:n], last[n:], norms[:n], norms[n:])
        u = random_direction(len(v), np.linalg.norm(v))
        vecs[f"contrast-l{L}"], vecs[f"random-l{L}"] = v, u
        inf.update({"median_norm_tokens_after_first": float(np.nanmedian(typ)),
                    "cos_contrast_random": float(v @ u / (np.linalg.norm(v) * np.linalg.norm(u))),
                    "unit_permissive_projection_mean": float(np.mean(last[:n] @ v) / np.linalg.norm(v)),
                    "unit_restrictive_projection_mean": float(np.mean(last[n:] @ v) / np.linalg.norm(v))})
        info["layers"][str(L)] = inf
    np.savez(out / "vectors.npz", **vecs)
    info["vectors_sha256"] = sha256_file(out / "vectors.npz")
    (out / "vectors.json").write_text(json.dumps({"provenance": provenance(args, {"model_path": str(eng.path)}),
                                                  "info": info}, indent=1) + "\n")
    print(json.dumps(info))


def load_vecs(path, layer):
    import torch
    data = np.load(Path(path) / "vectors.npz")
    return (torch.tensor(data[f"contrast-l{layer}"], dtype=torch.float32),
            torch.tensor(data[f"random-l{layer}"], dtype=torch.float32), sha256_file(Path(path) / "vectors.npz"))


def run_batches(args, eng, out, batches, n_steps, readouts, v, u, info, pair=False):
    """Run batches of row specs with resume markers; records.jsonl gets the rows, a timing line and a done marker per
    batch."""
    runf = out / "run.json"
    hist = json.loads(runf.read_text())["history"] if runf.exists() else []
    hist.append(info)
    runf.write_text(json.dumps({"history": hist}, indent=1) + "\n")
    recp = out / "records.jsonl"
    done = set()
    if recp.exists():
        for line in open(recp):
            r = json.loads(line)
            if r["kind"] == "done":
                done.add(r["batch"])
    recf = open(recp, "a")
    pair_key = (lambda r: (r["inst"]["uid"], r["start"])) if pair else None
    t_all = time.time()
    for bi, rows in enumerate(batches):
        if bi in done:
            continue
        t0 = time.time()
        res = eng.run_rows(rows, n_steps, set(readouts), v, u)
        for rec in row_records(rows, res, readouts, pair_key):
            rec["batch"] = bi
            recf.write(json.dumps(rec) + "\n")
        sec = time.time() - t0
        recf.write(json.dumps({"kind": "timing", "batch": bi, "rows": len(rows), "steps": n_steps,
                               "seconds": round(sec, 2)}) + "\n")
        recf.write(json.dumps({"kind": "done", "batch": bi}) + "\n")
        recf.flush()
        print(json.dumps({"batch": bi, "of": len(batches), "rows": len(rows), "seconds": round(sec, 1),
                          "elapsed": round(time.time() - t_all, 1)}), flush=True)
    hist[-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runf.write_text(json.dumps({"history": hist}, indent=1) + "\n")


def cmd_smoke(args):
    """Engine checks on two smoke instances (seed SMOKE_SEED, neither development nor held-out)."""
    torch_threads(args)
    import torch
    eng = Engine(args.model, args.threads, layer=args.layer)
    v, u, vsha = load_vecs(args.vectors, args.layer)
    insts = smoke_instances()
    out = out_dir(args, "smoke")
    ramp, hold, dwell = 2, 2, 4
    N = 2 * ramp + hold + dwell
    rows = []
    for inst in insts:
        for k in (0, 1):
            for h in ("direct", "zero", "up", "planted", "unsteered"):
                co = history_coefficients(h, args.alpha, 2 * args.alpha, ramp, hold, dwell)
                plant = planted_sequence(eng.plant_ids["ALLOW"], 2 * ramp + hold - 1) if h == "planted" else None
                rows.append({"inst": inst, "start": k, "history": h, **co, "plant": plant, "forced": h == "unsteered"})
    full = eng.run_rows(rows, N, {2, 4, 6, N}, v, u)
    last = eng.run_rows(rows, N, {N}, v, u)
    checks = {}
    di = [i for i, r in enumerate(rows) if r["history"] == "direct"]
    zi = [i for i, r in enumerate(rows) if r["history"] == "zero"]
    ui = [i for i, r in enumerate(rows) if r["history"] == "up"]
    emb_f, emb_l = torch.stack(full["emb"]), torch.stack(last["emb"])
    checks["zero_equals_direct_bitwise"] = bool(torch.equal(emb_f[:, zi], emb_f[:, di]))
    checks["zero_vs_direct_max_rel"] = float(((emb_f[:, zi] - emb_f[:, di]).norm(dim=-1) / emb_f[:, di].norm(dim=-1)).max())
    checks["readouts_do_not_perturb_bitwise"] = bool(torch.equal(emb_f, emb_l))
    checks["readouts_vs_none_max_abs"] = float((emb_f - emb_l).abs().max())
    checks["final_texts_equal"] = full["readouts"][N]["text"] == last["readouts"][N]["text"]
    checks["final_zA_max_abs_diff"] = float(np.max(np.abs(np.array(full["readouts"][N]["zA"]) -
                                                         np.array(last["readouts"][N]["zA"]))))
    checks["up_differs_from_direct"] = float(((emb_f[:, ui] - emb_f[:, di]).norm(dim=-1) / emb_f[:, di].norm(dim=-1)).max())
    # the hook adds exactly alpha v at the layer output (prefill of one prompt, captured with a second hook)
    cap = {}
    h2 = eng.model.model.layers[args.layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    st0 = eng.prefill([insts[0]], [0.0], v)
    h0 = cap["h"]
    st1 = eng.prefill([insts[0]], [args.alpha], v)
    h1 = cap["h"]
    h2.remove()
    checks["hook_delta_rel_err"] = float(((h1 - h0)[0] - args.alpha * v).norm(dim=-1).max() / (args.alpha * v).norm())
    checks["prefill_logits_change"] = float((st1["z"] - st0["z"]).abs().max())
    checks["texts_final"] = {f"{rows[i]['history']}-{rows[i]['inst']['id']}-{rows[i]['start']}": full["readouts"][N]["text"][i]
                             for i in range(len(rows))}
    checks["forced_texts_final"] = {str(k): t for k, t in full["readouts"][N].get("forced", {}).items()}
    checks["thr_final"] = full["readouts"][N]["thr"]
    info = provenance(args, {"vectors_sha256": vsha, "model_path": str(eng.path)})
    (out / "smoke.json").write_text(json.dumps({"provenance": info, "checks": checks}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in checks.items() if not isinstance(v, (dict, list))}, indent=1))


def cmd_baseline(args):
    torch_threads(args)
    eng = Engine(args.model, args.threads)
    insts = split_instances(args, args.split)
    out = out_dir(args, f"baseline-{args.model}-{args.split}", resume=args.resume)
    N = max(args.readouts)
    starts = list(range(args.starts))
    batches = []
    for s0 in range(0, len(insts), args.instances_per_batch):
        rows = []
        for inst in insts[s0:s0 + args.instances_per_batch]:
            for k in starts:
                co = history_coefficients("unsteered", 0.0, 0.0, 0, 0, N)
                rows.append({"inst": inst, "start": k, "history": "unsteered", **co, "plant": None, "forced": False})
        batches.append(rows)
    info = provenance(args, {"kind": "baseline", "instances_sha256": instances_hash(load_instances(args.instances)),
                             "model_path": str(eng.path), "uids": [i["uid"] for i in insts], "starts": starts,
                             "start_ids": eng.start_ids, "n_steps": N, "readouts": sorted(args.readouts),
                             "temperature": eng.T, "top_n": eng.top_n, "answer_tokens": eng.answer_tokens})
    run_batches(args, eng, out, batches, N, sorted(args.readouts), None, None, info)


def cmd_sweep(args):
    torch_threads(args)
    eng = Engine(args.model, args.threads, layer=args.layer)
    v, u, vsha = load_vecs(args.vectors, args.layer)
    insts = split_instances(args, args.split)
    out = out_dir(args, f"sweep-{args.model}-l{args.layer}-{args.split}", resume=args.resume)
    N = max(args.readouts)
    starts = list(range(args.starts))
    batches = []
    for s0 in range(0, len(insts), args.instances_per_batch):
        rows = []
        for inst in insts[s0:s0 + args.instances_per_batch]:
            for a in args.alphas:
                for k in starts:
                    rows.append({"inst": inst, "start": k, "history": "static", "alpha": np.full(N, float(a)),
                                 "beta": np.zeros(N), "alpha_pre": float(a), "static_alpha": float(a), "plant": None,
                                 "forced": False})
        batches.append(rows)
    info = provenance(args, {"kind": "sweep", "instances_sha256": instances_hash(load_instances(args.instances)),
                             "model_path": str(eng.path), "vectors_sha256": vsha, "uids": [i["uid"] for i in insts],
                             "starts": starts, "start_ids": eng.start_ids, "n_steps": N,
                             "readouts": sorted(args.readouts), "alphas": args.alphas, "layer": args.layer,
                             "temperature": eng.T, "top_n": eng.top_n, "answer_tokens": eng.answer_tokens})
    run_batches(args, eng, out, batches, N, sorted(args.readouts), v, u, info)


def cmd_histories(args):
    torch_threads(args)
    ch = json.loads(Path(args.choices).read_text())
    for k in ("model", "layer", "alpha_final", "alpha_high", "ramp", "hold"):
        if ch.get(k) is None:
            raise SystemExit(f"choices file lacks {k}")
    eng = Engine(ch["model"], args.threads, layer=ch["layer"])
    v, u, vsha = load_vecs(args.vectors, ch["layer"])
    insts = split_instances(args, args.split)
    out = out_dir(args, f"histories-{args.split}", resume=args.resume)
    ramp, hold, dwell = ch["ramp"], ch["hold"], args.dwell
    P = 2 * ramp + hold
    N = P + dwell
    reads = readout_steps(ramp, hold, tuple(d for d in (0, 16, 32, 64) if d <= dwell))
    starts = list(range(args.starts))
    batches = []
    for s0 in range(0, len(insts), args.instances_per_batch):
        rows = []
        for inst in insts[s0:s0 + args.instances_per_batch]:
            for k in starts:
                for h in args.histories:
                    co = history_coefficients(h, ch["alpha_final"], ch["alpha_high"], ramp, hold, dwell)
                    plant, ptxt = None, None
                    if h == "planted":
                        wrong = "ALLOW" if inst["gold"] == "DENY" else "DENY"
                        plant, ptxt = planted_sequence(eng.plant_ids[wrong], P - 1), PLANT[wrong]
                    rows.append({"inst": inst, "start": k, "history": h, **co, "plant": plant, "plant_text": ptxt,
                                 "forced": h == "unsteered"})
        batches.append(rows)
    info = provenance(args, {"kind": "histories", "instances_sha256": instances_hash(load_instances(args.instances)),
                             "model_path": str(eng.path), "vectors_sha256": vsha, "choices": ch,
                             "choices_sha256": sha256_file(args.choices), "uids": [i["uid"] for i in insts],
                             "starts": starts, "start_ids": eng.start_ids, "n_steps": N, "P": P, "dwell": dwell,
                             "readouts": reads, "histories": args.histories, "plant_ids": eng.plant_ids,
                             "temperature": eng.T, "top_n": eng.top_n, "answer_tokens": eng.answer_tokens})
    run_batches(args, eng, out, batches, N, reads, v, u, info, pair=True)


# ----------------------------------------------------------------------------------------------- analysis

def load_run(run):
    recs = [json.loads(line) for line in open(Path(run) / "records.jsonl")]
    hist = json.loads((Path(run) / "run.json").read_text())["history"]
    return recs, hist


def decision_counts(rows, S):
    c = defaultdict(int)
    for r in rows:
        c[readout_of(r, S)["decision"]] += 1
    return dict(c)


def analyze_baseline(rows, info):
    res = {"by_S": {}}
    for S in info["readouts"]:
        res["by_S"][str(S)] = {"accuracy": rate_summary(rows, S, "correct"),
                               "accuracy_prohibited": rate_summary(rows, S, "correct", "prohibited"),
                               "accuracy_permitted": rate_summary(rows, S, "correct", "permitted"),
                               "unparseable": rate_summary(rows, S, "unparseable"), "decisions": decision_counts(rows, S),
                               "lenient_differs": sum(readout_of(r, S)["lenient"] != readout_of(r, S)["decision"]
                                                      for r in rows)}
    res["accuracy_by_S"] = {S: v["accuracy"]["mean"] for S, v in res["by_S"].items()}
    return res


def sweep_table(rows, S):
    by = defaultdict(list)
    for r in rows:
        by[r["static_alpha"]].append(r)
    tab = {}
    for a, rs in sorted(by.items()):
        tab[a] = {"fa_proh": rate_summary(rs, S, "allow", "prohibited")["mean"],
                  "fd_perm": rate_summary(rs, S, "deny", "permitted")["mean"],
                  "unparseable": rate_summary(rs, S, "unparseable")["mean"],
                  "accuracy": rate_summary(rs, S, "correct")["mean"],
                  "margin_proh": rate_summary(rs, S, "margin", "prohibited")["mean"],
                  "margin_perm": rate_summary(rs, S, "margin", "permitted")["mean"],
                  "runs": len(rs), "decisions": decision_counts(rs, S)}
    return tab


def analyze_sweep(rows, info):
    return {"layer": info["layer"], "by_S": {str(S): {str(a): t for a, t in sweep_table(rows, S).items()}
                                             for S in info["readouts"]}}


def gate_histories(rows, info):
    issues = []
    seen = defaultdict(int)
    for r in rows:
        seen[(r["uid"], r["start"], r["history"])] += 1
    for uid in info["uids"]:
        for k in info["starts"]:
            for h in info["histories"]:
                if seen.get((uid, k, h), 0) != 1:
                    issues.append(f"row count {seen.get((uid, k, h), 0)} for {(uid, k, h)}")
    ch = info["choices"]
    P, N = info["P"], info["n_steps"]
    for r in rows:
        if len(r["alpha"]) != N or len(r["top1"]) != N:
            issues.append(f"length {(r['uid'], r['start'], r['history'])}")
        for S in info["readouts"]:
            if str(S) not in r["readouts"]:
                issues.append(f"missing readout {S} {(r['uid'], r['start'], r['history'])}")
        if r["history"] != "unsteered":
            if any(a != round(ch["alpha_final"], 8) for a in r["alpha"][P:]):
                issues.append(f"dwell not at alpha_final {(r['uid'], r['start'], r['history'])}")
            if r["alpha_pre"] != ch["alpha_final"]:
                issues.append(f"prompt not at alpha_final {(r['uid'], r['start'], r['history'])}")
            if r["beta"] is not None and any(b != 0 for b in r["beta"][P:]):
                issues.append(f"random component in the dwell {(r['uid'], r['start'], r['history'])}")
    return issues


def analyze_histories(rows, info, bias=None):
    P, N = info["P"], info["n_steps"]
    ramp, hold = info["choices"]["ramp"], info["choices"]["hold"]
    peak = ramp + hold
    S32, S64 = P + 32, (P + 64 if P + 64 in info["readouts"] else None)
    by = defaultdict(list)
    for r in rows:
        by[r["history"]].append(r)
    res = {"P": P, "n_steps": N, "S_primary": S32, "S_secondary": S64, "peak_S": peak,
           "gate_issues": gate_histories(rows, info)}
    rates = {}
    for h, rs in by.items():
        rates[h] = {}
        for S in info["readouts"]:
            rates[h][str(S)] = {"false_allow_prohibited": rate_summary(rs, S, "allow", "prohibited"),
                                "false_deny_permitted": rate_summary(rs, S, "deny", "permitted"),
                                "unparseable": rate_summary(rs, S, "unparseable"),
                                "accuracy": rate_summary(rs, S, "correct"),
                                "margin_prohibited": rate_summary(rs, S, "margin", "prohibited"),
                                "margin_permitted": rate_summary(rs, S, "margin", "permitted"),
                                "decisions": decision_counts(rs, S)}
    if bias is not None and "unsteered" in by:
        rates["logit_bias"] = {}
        for S in info["readouts"]:
            rs = by["unsteered"]
            rates["logit_bias"][str(S)] = {"b": bias,
                                           "false_allow_prohibited": rate_summary(rs, S, "allow", "prohibited", bias),
                                           "false_deny_permitted": rate_summary(rs, S, "deny", "permitted", bias),
                                           "unparseable": rate_summary(rs, S, "unparseable", None, bias),
                                           "accuracy": rate_summary(rs, S, "correct", None, bias)}
    res["rates"] = rates
    con = {}
    d = by.get("direct", [])
    for h in [x for x in by if x != "direct"]:
        for S in info["readouts"]:
            con[f"{h}-minus-direct_S{S}"] = {
                "false_allow_prohibited": paired_contrast(by[h], d, S, "allow", "prohibited"),
                "false_deny_permitted": paired_contrast(by[h], d, S, "deny", "permitted"),
                "unparseable": paired_contrast(by[h], d, S, "unparseable"),
                "margin_prohibited": paired_contrast(by[h], d, S, "margin", "prohibited"),
                "margin_permitted": paired_contrast(by[h], d, S, "margin", "permitted")}
    if bias is not None and "unsteered" in by:
        for S in info["readouts"]:
            con[f"logit_bias-minus-direct_S{S}"] = {
                "false_allow_prohibited": paired_contrast(by["unsteered"], d, S, "allow", "prohibited", bias_a=bias),
                "false_deny_permitted": paired_contrast(by["unsteered"], d, S, "deny", "permitted", bias_a=bias)}
    res["contrasts"] = con
    if "up" in by and "direct" in by:
        p32 = con[f"up-minus-direct_S{S32}"]["false_allow_prohibited"]
        p64 = con[f"up-minus-direct_S{S64}"]["false_allow_prohibited"] if S64 else {"estimate": None,
                                                                                   "boot95_instances": [None, None]}
        res["primary"] = {"definition": f"false-ALLOW rate among prohibited requests, up-excursion minus direct, after "
                                        f"P + 32 = {S32} soft steps; instances are the units (mean over starts), "
                                        "paired, percentile bootstrap over instances (10,000 resamples)",
                          "estimate": p32["estimate"], "boot95_instances": p32["boot95_instances"],
                          "instances": p32["instances"], "instances_positive": p32["instances_positive"],
                          "instances_negative": p32["instances_negative"],
                          "estimate_dwell64": p64["estimate"], "boot95_dwell64": p64["boot95_instances"],
                          "decision": classify_effect(p32["estimate"], p32["boot95_instances"], p64["estimate"],
                                                      p64["boot95_instances"])}
        sec = {}
        for h, metric, cls in (("down", "allow", "prohibited"), ("up", "deny", "permitted"),
                               ("down", "deny", "permitted"), ("random", "allow", "prohibited"),
                               ("random", "deny", "permitted"), ("planted", "allow", "prohibited"),
                               ("planted", "deny", "permitted"), ("zero", "allow", "prohibited"),
                               ("zero", "deny", "permitted")):
            if h not in by:
                continue
            k = "false_allow_prohibited" if metric == "allow" else "false_deny_permitted"
            a = con[f"{h}-minus-direct_S{S32}"][k]
            b = con[f"{h}-minus-direct_S{S64}"][k] if S64 else {"estimate": None, "boot95_instances": [None, None]}
            sec[f"{h}_{k}"] = {"estimate_dwell32": a["estimate"], "boot95_dwell32": a["boot95_instances"],
                               "estimate_dwell64": b["estimate"], "boot95_dwell64": b["boot95_instances"],
                               "decision": classify_effect(a["estimate"], a["boot95_instances"], b["estimate"],
                                                           b["boot95_instances"])}
        res["secondary"] = sec
        res["relaxation_up_false_allow"] = {str(S): con[f"up-minus-direct_S{S}"]["false_allow_prohibited"]["estimate"]
                                            for S in info["readouts"]}
        if "down" in by:
            res["relaxation_down_false_deny"] = {
                str(S): con[f"down-minus-direct_S{S}"]["false_deny_permitted"]["estimate"] for S in info["readouts"]}
    # negative control: zero rows against direct rows, row by row
    if "zero" in by and "direct" in by:
        dmap = {(r["uid"], r["start"]): r for r in by["direct"]}
        diff_top1 = diff_text = 0
        maxd = 0.0
        for r in by["zero"]:
            q = dmap[(r["uid"], r["start"])]
            diff_top1 += r["top1"] != q["top1"]
            diff_text += any(r["readouts"][s]["text"] != q["readouts"][s]["text"] for s in r["readouts"])
            maxd = max(maxd, max(r.get("dist_to_direct") or [0.0]))
        res["zero_control"] = {"rows": len(by["zero"]), "top1_sequences_differ": diff_top1,
                               "any_readout_text_differs": diff_text, "max_rel_concept_distance": maxd}
    # trajectory difference after the return (descriptive)
    traj = {}
    dmap = {(r["uid"], r["start"]): r for r in by.get("direct", [])}
    for h, rs in by.items():
        if h in ("direct", "unsteered") or not rs or "dist_to_direct" not in rs[0]:
            continue
        dd = np.array([r["dist_to_direct"] for r in rs])
        agree = [np.mean([a == b for a, b in zip(r["top1"][N - 16:], dmap[(r["uid"], r["start"])]["top1"][N - 16:])])
                 for r in rs]
        traj[h] = {"median_rel_distance_at_step": {str(t): float(np.median(dd[:, t])) for t in
                                                   sorted({P - 1, min(P + 15, N - 1), min(P + 31, N - 1), N - 1})},
                   "share_rows_identical_top1_last16": float(np.mean([a == 1.0 for a in agree])),
                   "mean_top1_agreement_last16": float(np.mean(agree))}
    res["trajectories"] = traj
    return res


def cmd_analyze(args):
    recs, hist = load_run(args.run)
    info = hist[-1]
    rows = [r for r in recs if r["kind"] == "row"]
    tim = [r for r in recs if r["kind"] == "timing"]
    kind = info["kind"]
    summ = {"run": str(args.run), "kind": kind, "commits": sorted({h["commit"] for h in hist}),
            "tool_sha256": sorted({h["tool_sha256"] for h in hist}), "instances_sha256": info["instances_sha256"],
            "model_path": info["model_path"], "n_rows": len(rows),
            "compute": {"seconds": round(sum(t["seconds"] for t in tim), 1), "batches": len(tim),
                        "row_steps": sum(t["rows"] * t["steps"] for t in tim)}}
    if kind == "baseline":
        summ.update(analyze_baseline(rows, info))
    elif kind == "sweep":
        summ.update(analyze_sweep(rows, info))
    elif kind == "histories":
        bias = args.bias
        if bias is None and info["choices"].get("logit_bias_b") is not None:
            bias = info["choices"]["logit_bias_b"]
        summ["logit_bias_b"] = bias
        summ.update(analyze_histories(rows, info, bias))
    txt = json.dumps(summ, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(txt + "\n")
    brief = {k: summ[k] for k in ("kind", "n_rows", "compute") if k in summ}
    for k in ("accuracy_by_S", "primary", "zero_control", "gate_issues"):
        if k in summ:
            brief[k] = summ[k] if k != "gate_issues" else summ[k][:10]
    print(json.dumps(brief, indent=1))


def cmd_choose(args):
    ch = json.loads(Path(args.choices).read_text()) if Path(args.choices).exists() else {}
    if args.stage == "model":
        a = json.loads(Path(args.analysis[0]).read_text())
        acc = {int(S): v for S, v in a["accuracy_by_S"].items()}
        r = choose_model_P(acc)
        model = json.loads((Path(a["run"]) / "run.json").read_text())["history"][-1]["args"]["model"]
        ch.setdefault("model_rule", {})[model] = r
        if r["model_ok"]:
            ch.update({"model": model, "ramp": r["ramp"], "hold": r["hold"], "P": r["P"], "dwell": 32,
                       "dwell_secondary": 64})
    elif args.stage == "alpha":
        tables = {}
        for path in args.analysis:
            a = json.loads(Path(path).read_text())
            S = str(ch["P"] + 32)
            tables[int(a["layer"])] = {float(al): t for al, t in a["by_S"][S].items()}
        r = choose_layer(tables)
        ch["layer_rule"] = json.loads(json.dumps(r, default=str))
        if r["status"] == "chosen":
            ch.update({"layer": r["layer"], "alpha_final": r["alpha_final"], "alpha_high": r["alpha_high"]})
    elif args.stage == "bias":
        recs, hist = load_run(args.run)
        info = hist[-1]
        S = info["P"] + 32
        rows = [r for r in recs if r["kind"] == "row"]
        proh_u = [r for r in rows if r["history"] == "unsteered" and r["cls"] == "prohibited"]
        direct = [r for r in rows if r["history"] == "direct"]
        fa_direct = rate_summary(direct, S, "allow", "prohibited")["mean"]
        r = choose_bias(proh_u, S, fa_direct)
        ch["logit_bias_rule"] = r
        ch["logit_bias_b"] = r["b"]
    Path(args.choices).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choices).write_text(json.dumps(ch, indent=1) + "\n")
    print(json.dumps({k: v for k, v in ch.items() if not k.endswith("_rule")}, indent=1))


def cmd_receipt(args):
    """Compact receipt: card, settings, choices, deviations and caveats (from --spec), the instance list hash, the
    vectors, each analysis, and the SHA-256 and size of every raw output file."""
    spec = json.loads(Path(args.spec).read_text())
    rec = {k: v for k, v in spec.items() if k not in ("analyses", "raw_dirs", "vectors_dir", "choices_file")}
    rec["instances"] = {"file": str(INSTANCES_FILE.relative_to(ROOT)), "seed": INSTANCE_SEED, "n_dev": N_DEV,
                        "n_test": N_TEST, "sha256": instances_hash(make_instances())}
    if spec.get("choices_file"):
        rec["choices"] = json.loads(Path(spec["choices_file"]).read_text())
        rec["choices_file"] = {"path": spec["choices_file"], "sha256": sha256_file(spec["choices_file"])}
    if spec.get("vectors_dir"):
        rec["vectors"] = json.loads((Path(spec["vectors_dir"]) / "vectors.json").read_text())["info"]
    rec["analyses"] = {}
    for name, path in spec.get("analyses", {}).items():
        a = json.loads(Path(path).read_text())
        if a.get("kind") == "histories":
            for c in a.get("contrasts", {}).values():
                for m in c.values():
                    m.pop("per_instance", None)
        rec["analyses"][name] = a
    raw = []
    for d in spec.get("raw_dirs", []):
        for f in sorted(Path(d).rglob("*")):
            if f.is_file():
                raw.append({"path": str(f.relative_to(DATA.parent)), "sha256": sha256_file(f), "bytes": f.stat().st_size})
    rec["raw"] = {"root": str(DATA.parent), "files": raw}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"written": args.json, "raw_files": len(raw)}))


# ----------------------------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        if model:
            p.add_argument("--model", default="qwen1.5b", choices=sorted(MODELS))
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--instances", default=str(INSTANCES_FILE), help="instance list JSON (must match the generator)")
        p.add_argument("--limit", type=int, default=0, help="first N instances of the split (0: all)")
        p.add_argument("--out", default=None)
        p.add_argument("--resume", action="store_true")

    p = sub.add_parser("instances")
    p.add_argument("--model", default="qwen1.5b", choices=sorted(MODELS))
    p.add_argument("--json", default=str(INSTANCES_FILE))
    p = sub.add_parser("vectors")
    common(p)
    p.add_argument("--layers", type=int, nargs="+", default=[10, 14, 18])
    p = sub.add_parser("smoke")
    common(p)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--vectors", default=str(DATA / "vectors-qwen1.5b"))
    p = sub.add_parser("baseline")
    common(p)
    p.add_argument("--split", default="dev", choices=("dev", "test"))
    p.add_argument("--starts", type=int, default=8)
    p.add_argument("--readouts", type=int, nargs="+", default=[0, 8, 16, 24, 32, 44, 56, 64, 76, 88, 96])
    p.add_argument("--instances-per-batch", type=int, default=8)
    p = sub.add_parser("sweep")
    common(p)
    p.add_argument("--split", default="dev", choices=("dev", "calib"))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--vectors", default=str(DATA / "vectors-qwen1.5b"))
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--readouts", type=int, nargs="+", required=True)
    p.add_argument("--instances-per-batch", type=int, default=2)
    p = sub.add_parser("histories")
    common(p, model=False)
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--choices", required=True)
    p.add_argument("--vectors", default=str(DATA / "vectors-qwen1.5b"))
    p.add_argument("--starts", type=int, default=8)
    p.add_argument("--dwell", type=int, default=64)
    p.add_argument("--histories", nargs="+", default=list(HISTORIES), choices=HISTORIES)
    p.add_argument("--instances-per-batch", type=int, default=2)
    p = sub.add_parser("analyze")
    p.add_argument("run")
    p.add_argument("--bias", type=float, default=None, help="endpoint-only logit bias b (default: from the choices)")
    p.add_argument("--json", default=None)
    p = sub.add_parser("choose")
    p.add_argument("stage", choices=("model", "alpha", "bias"))
    p.add_argument("--choices", required=True)
    p.add_argument("--analysis", nargs="*", default=[])
    p.add_argument("--run", default=None, help="bias stage: the development histories run")
    p = sub.add_parser("receipt")
    p.add_argument("--spec", required=True)
    p.add_argument("--json", required=True)
    args = ap.parse_args()
    {"instances": cmd_instances, "vectors": cmd_vectors, "smoke": cmd_smoke, "baseline": cmd_baseline,
     "sweep": cmd_sweep, "histories": cmd_histories, "analyze": cmd_analyze, "choose": cmd_choose,
     "receipt": cmd_receipt}[args.cmd](args)


if __name__ == "__main__":
    main()
