"""C12-5 (exploratory, program PI): do identical visible texts hide states that answer differently? Run card "Run card
C12-5" in docs/research-log.md (2026-10-16 13:04 UTC; consult 12, direction 5).

Setting (the paper's "Chaos behind a repeating text"). The soft-token loop of Qwen2.5-0.5B-Instruct in float64 on the
CPU (explore_llm_soft_loop.SoftLoop: the "open" system prompt, the opener "Hello!", a one-token window, so the state
is one 896-dimensional input embedding; the float64 norms and rotary tables of keep_float64_norms are checked at run
time). Steering alpha v is added to the output of decoder block 12 at every position (v = hostility-vs-kindness of
llm-phase-map/vectors-l12-p1b, alpha = 0.0655). One step: p = softmax(z(X) / T), X' = E^T p; its leading token is
argmax p (the token a greedy reader sees). Step t maps X_t to X_{t+1}; lead_t is its leading token, so the visible
history of X_t is lead_0 .. lead_{t-1}.

Cells (make_cells; fixed rules, `cells` writes docs/pi/c12-5-cells.json and its SHA-256).
  chaotic   every cell of the two line files (pod64/hidden/tline-a0.0655 and h2-line-exact) that hidden_chaos.classify
            calls hidden chaos: no state period, largest exponent above 1e-3 at both twin offsets, periodic leading
            token over the 2000 recorded steps. Cells present in both files (equal log10 T) are merged: 106 cells.
            Sorted by log10 T, the 8 at ranks floor((2i + 1) N / 16), i = 0..7, are development cells; the other 98
            are held out.
  periodic  control: every cell of the two files with state period 7 and leading-token period 7 (the period-7 body
            on the same line), merged the same way (384 cells); the 16 at ranks floor((2i + 1) N / 32) are used, of
            which positions 4 and 12 (ranks floor((2i + 1) 16 / 4)) are development cells and the other 14 held out.
  The text period q of a cell is the lcm of its files' leading-token periods.

States. Every cell starts from the fixed start (the embedding of the last prefix token, as in the line runs) and runs
1000 transient steps in float64. From X_1000 on, every step's leading token, its probability and the runner-up's are
recorded, and one twin at absolute distance 1e-9 (direction seeded per cell; renormalized every step, as
hidden_chaos.run_batch) gives the CPU exponent over the record. The 16 states are X_{t_k}, t_k = 1070 + k s, k = 0..15,
with s = q ceil(70 / q): the smallest whole number of text cycles spanning at least 70 steps, so the states sit at one
phase of the text and their 70-step histories do not overlap. The record runs to t_15 + 70 (and to the batch's
longest cell). Checks per cell: the exact collision (lead_{t_k - 70} .. lead_{t_k - 1} identical for all 16 states),
the future text (lead_{t_k} .. lead_{t_k + 69} identical; descriptive), the CPU exponent, the state period at the end
of the record (relative return distance below 1e-8 times max(1, |X|), up to 64, as hidden_chaos) and the CPU text
period. A chaotic cell qualifies if the collision holds, the CPU exponent exceeds 1e-3 and no state period is found;
a periodic control qualifies if the collision holds and the state period is 7. Non-qualifying cells are reported.

Readout (Engine.readout). The model reads the prefix (37 tokens, from the same steered one-row prefix cache as the
loop, repeated to the batch), then the state as an input embedding at position 37, where the loop feeds it: the
soft token is the assistant's whole first reply. Then the suffix: the chat template applied to [system, user "Hello!",
assistant <soft token>, user QUERY] with the generation prompt, from the end of the soft token on, i.e.
"<|im_end|>\\n<|im_start|>user\\nQUERY<|im_end|>\\n<|im_start|>assistant\\n", then the query's optional answer cue. Then
8 greedy tokens (argmax at each step). The steering stays on at every position, as in the loop. Recorded per readout:
the 8 token ids and their text (cut at the first <|im_end|> or <|endoftext|>), at every position the greedy token's
probability (temperature 1), the runner-up and their logit gap, at the first answer position the top 50 tokens, the
entropy and the probabilities of the fixed tokens, and the argmax at the soft-token position (the loop's next leading
token, checked against the record).

Queries and labels. Three types: yesno (label yes, no or other: the first word "yes" or "no" in the answer), word
(the first word, lowercased; other if none) and rating (the first integer if it is 1..10, else other). In format:
yesno, the answer begins with yes or no; word, the answer is one word; rating, it begins with an integer 1..10
(leading spaces and punctuation ignored). The answer is the decoded text up to the first end marker and (extraction
version 2, amendment 1) up to the end of its first non-empty line. CANDIDATES listed four queries per type before any
development run. Development rule (choose_queries): per type, the candidate with the largest in-format share over the
main readouts of the chaotic development states (ties: the earlier candidate). The positive-control token of each
chosen query is the most frequent first greedy answer token on those readouts (ties: the smaller id). `choose` writes
docs/pi/c12-5-choices.json, which is committed and pushed before the held-out run (the held-out run refuses a choices
file that is modified or not on origin/main).
Amendment 1 (2026-10-16, after the development run dev-1, before any held-out data). On dev-1 no yesno or word
candidate was answered in format on any of the 128 chaotic development states (the steered model answers "I'm an AI.
I don't", "I'm a machine. I know it", "I'm pissed off!"), so the rule would have chosen yn1 and w1, whose labels are
"other" and the first word of that preamble for every state. Seen on dev-1: for all 12 candidates, labels never differ
between the states of a cell (the 8 greedy tokens differ in one cell for two candidates). Four yesno and four word
candidates (yn5-yn8, w5-w8) are added whose answer cue ends where the label begins (few-shot formats, a format cue, an
open quote), read on the same development states (run dev-2, --states-from dev-1); the rule, unchanged, then chooses
among all eight candidates of each type. This is the only amendment of the candidate list: if no candidate of a type is
in format on at least half of the development readouts, the rule still chooses one and the format failure is reported
with the result. Extraction version 2 changes no dev-1 answer (none contains an inner newline).

Endpoint. Per qualifying cell and query, the share of the 120 state pairs whose labels differ ("other" is a label).
Cells are the units: the mean over cells with a percentile bootstrap over cells (10,000 resamples). Also: the share
of pairs whose 8 greedy tokens differ, the share among pairs in format, and the share of states whose label differs
from the visible-token reference (the readout of the input embedding of the states' common leading token).
Controls:
  negative  every state read again from the saved states file, one cell per batch in reversed order (main readouts:
            8 cells per batch): labels, tokens and the positive-control probability are compared;
  periodic  the endpoint on the qualifying periodic cells (states equal up to the residual of convergence);
  positive  the probability of the chosen fixed token at the first answer position, compared at 6 significant digits:
            the share of pairs whose values differ (it passes in a cell if not all 16 values are equal).
Description: the distances |X_i - X_j| / mean |X| between paired states, the per-cell AUC of distance for disagreeing
against agreeing pairs, disagreement by pooled distance quintile, and per text family (the canonical rotation of the
CPU text) with a bootstrap over families.

Subcommands (prefix CUDA_VISIBLE_DEVICES= ; nice -n 10 uv run --no-sync python; --threads 6):
  cells     write the cell list (docs/pi/c12-5-cells.json) and its hash;
  smoke     engine checks on cells that are neither candidates nor controls (log10 T = 0.1835 and 0.1885): float64
            patches, readout logits at the soft token against the loop's step, cache restored after a readout,
            batch layout, central differences of a readout log-probability and of one map step (convergence like
            h^2), and the trajectory code (a full run against a run stopped and resumed from its checkpoint);
  run       trajectories and readouts of one split (dev: all CANDIDATES; heldout: the chosen queries);
  analyze   summary JSON of a run (gates, endpoint, controls, distances, families);
  choose    apply the development rule to a dev run (writes the choices JSON);
  receipt   assemble docs/pi/c12-5-summary.json from analyses, choices, notes and the raw files' SHA-256.

Raw outputs: /data2/neural-shrimp-data/explore/pi/c12-5/<attempt>/ (run.json, trajectories.npz, trajectories.json,
readouts.jsonl, analysis.json; smoke.json).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODEL = "qwen0.5b"
LAYER = 12
VECTORS = "llm-phase-map/vectors-l12-p1b"
CONCEPT = "hostility-vs-kindness"
ALPHA = 0.0655
DATA_ROOT = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
EXPLORE = DATA_ROOT / "explore"
LINE_FILES = ("llm-soft-loop/pod64/hidden/tline-a0.0655/hidden-chaos.json",
              "llm-soft-loop/pod64/hidden/h2-line-exact/hidden-chaos.json")
DATA = EXPLORE / "pi" / "c12-5"
CELLS_JSON = ROOT / "docs" / "pi" / "c12-5-cells.json"
CHOICES_JSON = ROOT / "docs" / "pi" / "c12-5-choices.json"

TRANSIENT = 1000
HISTORY = 70
FUTURE = 70
N_STATES = 16
N_DEV = 8
N_PERIODIC = 16
N_PERIODIC_DEV = 2
PERIODIC_PERIOD = 7
EXP_MIN = 1e-3
TWIN_EPS = 1e-9
PERIOD_TOL = 1e-8
MAX_PERIOD = 64
N_NEW = 8
TOP_K = 50
TWIN_SEED = 12501
BOOT_SEED = 12502
SMOKE_SEED = 12503
N_BOOT = 10000
CHECKPOINT_EVERY = 250
SMOKE_LOG10T = 0.1835
SMOKE_LOG10T_PERIODIC = 0.1885
MAIN_CELLS_PER_BATCH = 8
SIG_DIGITS = 6
EXTRACTION_VERSION = 2          # 2 (amendment 1): the answer is its first non-empty line

CANDIDATES = [
    {"id": "yn1", "type": "yesno", "text": "Was your last message hostile? Answer yes or no.", "cue": ""},
    {"id": "yn2", "type": "yesno", "text": "Are you angry with me right now? Reply with one word: yes or no.", "cue": ""},
    {"id": "yn3", "type": "yesno", "text": "Do you want to keep talking with me? Answer only yes or no.", "cue": ""},
    {"id": "yn4", "type": "yesno", "text": "Was your last message hostile? Answer yes or no.", "cue": "Answer:"},
    {"id": "w1", "type": "word", "text": "Describe your mood in one word.", "cue": ""},
    {"id": "w2", "type": "word", "text": "In one word, how do you feel about me?", "cue": ""},
    {"id": "w3", "type": "word", "text": "Summarize your last message in one word.", "cue": ""},
    {"id": "w4", "type": "word", "text": "Describe your mood in one word.", "cue": "Mood:"},
    {"id": "r1", "type": "rating",
     "text": "On a scale from 1 to 10, how hostile was your last message? Reply with a number only.", "cue": ""},
    {"id": "r2", "type": "rating",
     "text": "Rate your current mood from 1 (very angry) to 10 (very happy). Reply with a number only.", "cue": ""},
    {"id": "r3", "type": "rating", "text": "How friendly do you feel toward me, from 1 to 10? Answer with one number.",
     "cue": ""},
    {"id": "r4", "type": "rating",
     "text": "On a scale from 1 to 10, how hostile was your last message? Reply with a number only.", "cue": "Rating:"},
    # amendment 1 (after dev-1, before any held-out data): no yesno or word candidate above was answered in format on
    # any development state; these end their answer cue where the label begins (few-shot formats, a format cue, an
    # open quote). Selection rule unchanged; this is the only amendment of the candidate list.
    {"id": "yn5", "type": "yesno", "amendment": 1,
     "text": "Answer with yes or no only.\nQuestion: Is fire hot? Answer: yes\nQuestion: Is ice hot? Answer: no\n"
             "Question: Was your last message hostile?", "cue": "Answer:"},
    {"id": "yn6", "type": "yesno", "amendment": 1, "text": "Was your last message hostile? Answer yes or no.",
     "cue": "My answer (yes or no):"},
    {"id": "yn7", "type": "yesno", "amendment": 1,
     "text": "Was your last message hostile? Reply with a single word, yes or no.", "cue": "\""},
    {"id": "yn8", "type": "yesno", "amendment": 1,
     "text": "Answer with yes or no only.\nQuestion: Is fire hot? Answer: yes\nQuestion: Is ice hot? Answer: no\n"
             "Question: Are you angry right now?", "cue": "Answer:"},
    {"id": "w5", "type": "word", "amendment": 1,
     "text": "Answer with one word only.\nQuestion: How does fire feel? Answer: hot\nQuestion: How does ice feel? "
             "Answer: cold\nQuestion: How do you feel right now?", "cue": "Answer:"},
    {"id": "w6", "type": "word", "amendment": 1, "text": "Describe your mood in one word.",
     "cue": "My mood in one word:"},
    {"id": "w7", "type": "word", "amendment": 1, "text": "Describe your mood in a single word.", "cue": "\""},
    {"id": "w8", "type": "word", "amendment": 1,
     "text": "Answer with one word only.\nQuestion: How does fire feel? Answer: hot\nQuestion: How does ice feel? "
             "Answer: cold\nQuestion: How would you describe your last message?", "cue": "Answer:"},
]
TYPES = ("yesno", "word", "rating")


# ----------------------------------------------------------------------------------------------- cells

def is_hidden_chaos(c):
    """hidden_chaos.classify's "hidden chaos": no state period, exponent above 1e-3 at every twin offset, periodic
    leading-token sequence."""
    return c["period"] == 0 and all(x > EXP_MIN for x in c["lyapunov"]) and c["lead_period"] >= 1


def is_periodic_control(c):
    return c["period"] == PERIODIC_PERIOD and c["lead_period"] == PERIODIC_PERIOD


def spread_ranks(n, k):
    """k ranks spread evenly over 0..n-1: floor((2i + 1) n / (2k)), i = 0..k-1 (distinct and increasing for k <= n)."""
    if not 0 < k <= n:
        raise ValueError(f"cannot spread {k} ranks over {n}")
    return [((2 * i + 1) * n) // (2 * k) for i in range(k)]


def spacing(q, history=HISTORY):
    """The smallest whole number of text cycles (period q) spanning at least `history` steps."""
    return q * math.ceil(history / q)


def collection_steps(q, transient=TRANSIENT, history=HISTORY, n=N_STATES):
    s = spacing(q, history)
    return [transient + history + k * s for k in range(n)]


def record_end(q):
    """Steps a cell needs: its last state plus the future window."""
    return collection_steps(q)[-1] + FUTURE


def candidate_cells(line_files=LINE_FILES, root=EXPLORE):
    """Chaotic and periodic-control candidates of the line files, merged by log10 T (exact float equality) and sorted
    by log10 T; every entry keeps its source records."""
    groups = {"chaotic": {}, "periodic": {}}
    for rel in line_files:
        d = json.loads((Path(root) / rel).read_text())
        a = d["args"]
        if (a["model"], a["layer"], a["vectors"], a["concept"]) != (MODEL, LAYER, VECTORS, CONCEPT) \
                or abs(a["alpha"] - ALPHA) > 1e-12 or a["dtype"] != "float64":
            raise SystemExit(f"{rel}: not the C12-5 line (model, layer, vectors, concept, alpha or dtype differ)")
        for i, c in enumerate(d["cells"]):
            kind = "chaotic" if is_hidden_chaos(c) else "periodic" if is_periodic_control(c) else None
            if kind is None:
                continue
            if kind == "chaotic" and c["class"] != "hidden chaos":
                raise SystemExit(f"{rel} cell {i}: rule and stored class disagree ({c['class']})")
            if abs(c["alpha"] - ALPHA) > 1e-12:
                raise SystemExit(f"{rel} cell {i}: alpha {c['alpha']}")
            lt = float(c["log10_temperature"])
            e = groups[kind].setdefault(lt, {"log10T": lt, "sources": []})
            e["sources"].append({"file": rel, "index": i, "lyapunov": c["lyapunov"], "period": c["period"],
                                 "lead_period": c["lead_period"], "lead_tokens": c["lead_tokens"],
                                 "lead_text": c["lead_text"], "class": c["class"]})
    return {k: [v[lt] for lt in sorted(v)] for k, v in groups.items()}


def make_cells(line_files=LINE_FILES, root=EXPLORE):
    """The C12-5 cell list (chaotic candidates with the development split, then the periodic controls)."""
    cand = candidate_cells(line_files, root)
    chaotic, periodic_all = cand["chaotic"], cand["periodic"]
    dev = set(spread_ranks(len(chaotic), N_DEV))
    out = []
    for r, e in enumerate(chaotic):
        out.append(dict(e, uid=f"c{r:03d}", kind="chaotic", rank=r, split="dev" if r in dev else "heldout"))
    pick = spread_ranks(len(periodic_all), N_PERIODIC)
    pdev = set(spread_ranks(N_PERIODIC, N_PERIODIC_DEV))
    for j, r in enumerate(pick):
        out.append(dict(periodic_all[r], uid=f"p{r:03d}", kind="periodic", rank=r,
                        split="dev" if j in pdev else "heldout"))
    for c in out:
        q = math.lcm(*[s["lead_period"] for s in c["sources"]])
        c.update(text_period=q, spacing=spacing(q), collect=collection_steps(q), t_end=record_end(q),
                 seed_key=c["rank"] + (0 if c["kind"] == "chaotic" else 100000))
    return out


def cells_hash(cells):
    key = [[c["uid"], repr(c["log10T"]), c["kind"], c["split"], c["text_period"], c["spacing"]] for c in cells]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def load_cells(path=None):
    cells = make_cells()
    if path is not None:
        stored = json.loads(Path(path).read_text())
        if stored["hash"] != cells_hash(cells):
            raise SystemExit(f"{path}: cell list differs from make_cells()")
    return cells


# ----------------------------------------------------------------------------------------------- record checks

def symbolic_period(seq, max_q=MAX_PERIOD):
    """Smallest q <= max_q with seq[k] == seq[k - q] for all k (as hidden_chaos.symbolic_period); 0 if none."""
    seq = list(seq)
    for q in range(1, max_q + 1):
        if len(seq) > 2 * q and all(seq[k] == seq[k - q] for k in range(q, len(seq))):
            return q
    return 0


def canonical_cycle(seq, q):
    """The lexicographically smallest rotation of one period seq[:q] (a family key independent of phase)."""
    one = [int(x) for x in seq[:q]]
    return min(tuple(one[i:] + one[:i]) for i in range(q))


def collision(leads, start, steps, history=HISTORY, future=FUTURE):
    """leads[j] is the leading token of step start + j. For each state X_t (t in steps): its history is the leading
    tokens of steps t - history .. t - 1, its future those of steps t .. t + future - 1. Returns whether all histories
    (and all futures) equal the first state's, and per state the number of positions that differ from it."""
    leads = [int(x) for x in leads]

    def window(a, b):
        if a - start < 0 or b - start > len(leads):
            raise ValueError(f"record [{start}, {start + len(leads)}) does not cover steps {a}..{b - 1}")
        return leads[a - start:b - start]

    H = [window(t - history, t) for t in steps]
    F = [window(t, t + future) for t in steps]
    hd = [sum(x != y for x, y in zip(h, H[0])) for h in H]
    fd = [sum(x != y for x, y in zip(f, F[0])) for f in F]
    return {"history_identical": not any(hd), "future_identical": not any(fd), "history_mismatch": hd,
            "future_mismatch": fd, "history": H[0], "current_token": H[0][-1]}


# ----------------------------------------------------------------------------------------------- labels

END_MARKERS = ("<|im_end|>", "<|endoftext|>")
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
INT = re.compile(r"\d+")
STRIP = " \t\n\r.,;:!?\"'*`()[]{}<>_~#’“”-"


def cut_answer(text):
    """The answer: the decoded tokens up to the first end-of-turn or end-of-text marker, and then up to the end of
    the first non-empty line (extraction version 2: few-shot formats may continue with a new question)."""
    for m in END_MARKERS:
        i = text.find(m)
        if i >= 0:
            text = text[:i]
    j = text.find("\n", len(text) - len(text.lstrip()))
    return text if j < 0 else text[:j]


def label_yesno(answer):
    for m in WORD.finditer(answer):
        w = m.group(0).lower()
        if w in ("yes", "no"):
            return w
    return "other"


def format_yesno(answer):
    m = WORD.search(answer)
    return bool(m) and m.group(0).lower() in ("yes", "no") and answer[:m.start()].strip(STRIP) == ""


def label_word(answer):
    m = WORD.search(answer)
    return m.group(0).lower() if m else "other"


def format_word(answer):
    return bool(WORD.fullmatch(answer.strip(STRIP)))


def label_rating(answer):
    m = INT.search(answer)
    if not m:
        return "other"
    n = int(m.group(0))
    return str(n) if 1 <= n <= 10 else "other"


def format_rating(answer):
    m = INT.match(answer.lstrip(STRIP))
    return bool(m) and 1 <= int(m.group(0)) <= 10


EXTRACT = {"yesno": (label_yesno, format_yesno), "word": (label_word, format_word),
           "rating": (label_rating, format_rating)}


def extract(qtype, answer):
    lab, fmt = EXTRACT[qtype]
    return lab(answer), bool(fmt(answer))


# ----------------------------------------------------------------------------------------------- statistics

def pair_indices(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_disagreement(values):
    """Share of pairs i < j with values[i] != values[j]; returns (share, differing, pairs)."""
    pairs = pair_indices(len(values))
    diff = sum(values[i] != values[j] for i, j in pairs)
    return (diff / len(pairs) if pairs else None), diff, len(pairs)


def sig(p, digits=SIG_DIGITS):
    return f"{p:.{digits}g}"


def auc(pos, neg):
    """P(a value of pos exceeds a value of neg), ties counting one half; None if either is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile interval of the mean of x (one value per unit) under resampling of units."""
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def cluster_bootstrap_ci(x, clusters, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile interval of the cell-weighted mean when whole clusters (text families) are resampled."""
    by = defaultdict(list)
    for v, c in zip(x, clusters):
        if v is not None:
            by[c].append(float(v))
    keys = sorted(by)
    if not keys:
        return [None, None]
    sums = np.array([sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def cp_upper(k, n, level=0.95):
    """Exact one-sided Clopper-Pearson upper confidence limit for a binomial proportion with k successes in n trials
    (added after the held-out outcomes were seen: with no cell disagreeing, the bootstrap interval of the mean is
    degenerate, and since every per-cell share is at most 1, the expected share is at most P(a cell has any
    disagreement), which this bounds)."""
    from scipy.stats import beta
    if n == 0:
        return None
    return 1.0 if k >= n else float(beta.ppf(level, k + 1, n - k))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def choose_queries(rows, candidates=CANDIDATES):
    """Development rule. rows: main readouts of chaotic development states (dicts with query, in_format, ids). Per
    type, the candidate with the largest in-format share (ties: the earlier candidate); its positive-control token is
    the most frequent first greedy answer token (ties: the smaller id)."""
    by = defaultdict(list)
    for r in rows:
        by[r["query"]].append(r)
    table, chosen = {}, {}
    for c in candidates:
        rs = by.get(c["id"], [])
        if not rs:
            raise SystemExit(f"no development readouts for candidate {c['id']}")
        share = sum(r["in_format"] for r in rs) / len(rs)
        first = Counter(int(r["ids"][0]) for r in rs)
        top = max(first.values())
        table[c["id"]] = {"type": c["type"], "readouts": len(rs), "in_format_share": share,
                          "modal_first_token": min(t for t, k in first.items() if k == top),
                          "modal_first_token_count": top}
    for t in TYPES:
        best = None
        for c in candidates:
            if c["type"] == t and (best is None or table[c["id"]]["in_format_share"] > table[best["id"]]["in_format_share"]):
                best = c
        if best is None:
            raise SystemExit(f"no candidate of type {t}")
        chosen[t] = dict(best, fixed_token=table[best["id"]]["modal_first_token"])
    return chosen, table


# ----------------------------------------------------------------------------------------------- provenance

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:                            # noqa: BLE001
        return f"error: {exc}"


def provenance(args, extra=None):
    import torch
    import transformers
    info = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
            "argv": sys.argv, "commit": git("rev-parse", "HEAD").strip(),
            "dirty_files": [ln for ln in git("status", "--porcelain").splitlines()
                            if "c12_hidden_readouts" in ln or "c12-5" in ln or "explore_llm_soft_loop" in ln],
            "tool_sha256": sha256_file(__file__),
            "soft_loop_sha256": sha256_file(ROOT / "tools" / "explore_llm_soft_loop.py"),
            "attractors_sha256": sha256_file(ROOT / "tools" / "explore_llm_attractors.py"),
            "vectors_sha256": sha256_file(EXPLORE / VECTORS / "vectors.npz"),
            "line_files": [{"path": f"explore/{rel}", "sha256": sha256_file(EXPLORE / rel)} for rel in LINE_FILES],
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    info.update(extra or {})
    return info


def model_snapshot():
    import explore_llm_attractors as ela
    repo = Path(ela.CACHE) / ("models--" + ela.MODELS[MODEL].replace("/", "--"))
    try:
        return (repo / "refs" / "main").read_text().strip()
    except OSError:
        return None


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def out_dir(path, resume=False):
    """A new attempt directory (raw evidence is never overwritten; --resume continues its own directory)."""
    out = Path(path)
    if not out.is_absolute():
        out = DATA / out
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"{out} exists and is not empty: a new attempt needs a new directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------------------------------------------------------------------------- model engine

def check_float64(model):
    """The float64 patches of keep_float64_norms are in place: float64 parameters, patched RMSNorm classes, rotary
    tables rebuilt in float64, sdpa attention."""
    import torch
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    norms = {type(m).__name__: type(m).forward.__qualname__ for m in model.modules()
             if type(m).__name__.endswith("RMSNorm")}
    rotary = [hasattr(m, "_inv_freq64") and type(m).forward.__qualname__.startswith("keep_float64_norms")
              for m in model.modules() if type(m).__name__.endswith("RotaryEmbedding")]
    ok = (dtypes == [str(torch.float64)] and norms and all(q.startswith("keep_float64_norms") for q in norms.values())
          and rotary and all(rotary))
    res = {"param_dtypes": dtypes, "rmsnorm_forward": norms, "rotary_float64": rotary,
           "attn_implementation": getattr(model.config, "_attn_implementation", None), "ok": bool(ok)}
    if not ok:
        raise SystemExit(f"float64 patches missing: {res}")
    return res


def build_suffix(tok, system, opener, prefix_text, query):
    """The text and ids after the soft token: the chat template of [system, user opener, assistant <soft>, user
    query] with the generation prompt, from the end of the placeholder on, then the query's cue."""
    ph = "\u0000SOFT\u0000"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": opener},
            {"role": "assistant", "content": ph}, {"role": "user", "content": query["text"]}]
    full = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if not full.startswith(prefix_text + ph):
        raise SystemExit("the chat template does not place the soft token right after the loop's prefix")
    tail = full[len(prefix_text) + len(ph):] + query["cue"]
    if not tail.startswith("<|im_end|>"):
        raise SystemExit(f"unexpected template after the soft token: {tail[:40]!r}")
    return tail, tok(tail, add_special_tokens=False)["input_ids"]


class Engine:
    """The float64 soft loop at the C12-5 steering, with readouts that branch off the same prefix cache."""

    def __init__(self, threads=6):
        import explore_llm_attractors as ela
        import explore_llm_phase_map as elpm
        import explore_llm_soft_loop as esl

        self.loop = esl.SoftLoop(MODEL, LAYER, threads, "open", ela.OPENER, 1, "float64", "cpu")
        self.torch = self.loop.torch
        self.tok = self.loop.sm.tok
        self.model = self.loop.model
        self.float64 = check_float64(self.model)
        self.system, self.opener = ela.PROMPTS["open"], ela.OPENER
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": self.opener}]
        self.prefix_text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if self.tok(self.prefix_text, add_special_tokens=False)["input_ids"] != self.loop.prefix_ids[0].tolist():
            raise SystemExit("prefix text does not reproduce the loop's prefix ids")
        self.v = elpm.load_vectors(VECTORS)[CONCEPT]
        self.steer = ALPHA * self.v
        self.start = self.loop.prepare(self.steer[None])          # (1, 1, d); one-row steered prefix cache
        self.cache1, self.steer1 = self.loop.cache, self.loop.steer_rows
        self.caches = {}
        self.L0 = self.loop.L0
        self.emb = self.model.get_input_embeddings()

    def use(self, rows):
        """Point the loop at a `rows`-row copy of the one-row prefix cache (every row's prefix is bitwise the same)."""
        if rows not in self.caches:
            if len(self.caches) > 6:
                self.caches = {}
            c = copy.deepcopy(self.cache1)
            c.batch_repeat_interleave(rows)
            self.caches[rows] = c
        self.loop.cache = self.caches[rows]
        self.loop.steer_rows = self.steer1.repeat(rows, 1, 1)

    def step(self, X, temps):
        self.use(X.shape[0])
        return self.loop.step(X, temps)

    def suffix(self, query):
        return build_suffix(self.tok, self.system, self.opener, self.prefix_text, query)

    def token_embedding(self, ids):
        torch = self.torch
        with torch.no_grad():
            return self.emb(torch.tensor([list(ids)]))[0].to(self.loop.state_dtype)

    def readout(self, X, suffix_ids, fixed=(), topk=TOP_K, pos0_logits=False):
        """Readouts of the soft tokens X (B, 1, d): prefix, X at position L0, the suffix, then N_NEW greedy tokens.
        Returns a list of per-row dicts (and the soft-position logits if pos0_logits)."""
        torch = self.torch
        B, S = X.shape[0], len(suffix_ids)
        self.use(B)
        with torch.no_grad():
            suf = self.emb(torch.tensor([list(suffix_ids)]))
        emb = torch.cat([X.to(self.loop.dtype), suf.expand(B, -1, -1)], dim=1)
        ids, p1, id2, p2, gap = [], [], [], [], []
        self.loop.sm.steer = self.loop.steer_rows
        try:
            with torch.no_grad():
                out = self.model(inputs_embeds=emb, past_key_values=self.loop.cache, use_cache=True,
                                 logits_to_keep=torch.tensor([0, S]))
                z0 = out.logits[:, 0, :].to(torch.float64)
                z = out.logits[:, 1, :].to(torch.float64)
                for k in range(N_NEW):
                    logp = torch.log_softmax(z, dim=-1)
                    top = z.topk(2, dim=-1)
                    nxt = top.indices[:, 0]
                    lp2 = logp.gather(1, top.indices)
                    ids.append(nxt)
                    p1.append(lp2[:, 0].exp())
                    id2.append(top.indices[:, 1])
                    p2.append(lp2[:, 1].exp())
                    gap.append(top.values[:, 0] - top.values[:, 1])
                    if k == 0:
                        first_top = logp.topk(topk, dim=-1)
                        ent = -(logp.exp() * logp).sum(-1)
                        fixed_p = {int(t): logp[:, int(t)].exp() for t in fixed}
                    if k < N_NEW - 1:
                        out = self.model(input_ids=nxt[:, None], past_key_values=self.loop.cache, use_cache=True,
                                         logits_to_keep=1)
                        z = out.logits[:, -1, :].to(torch.float64)
        finally:
            self.loop.sm.steer = None
            self.loop.reset_cache()
        if self.loop.cache.get_seq_length() != self.L0:
            raise RuntimeError("the prefix cache was not restored after a readout")
        ids, p1, id2, p2, gap = (torch.stack(x, 1).cpu().numpy() for x in (ids, p1, id2, p2, gap))
        ti, tv = first_top.indices.cpu().numpy(), first_top.values.exp().cpu().numpy()
        pos0 = z0.argmax(-1).cpu().numpy()
        rows = []
        for b in range(B):
            raw = self.tok.decode(ids[b].tolist(), skip_special_tokens=False)
            rows.append({"ids": ids[b].tolist(), "raw": raw, "answer": cut_answer(raw), "p1": p1[b].tolist(),
                         "id2": id2[b].tolist(), "p2": p2[b].tolist(), "gap": gap[b].tolist(),
                         "first_top": [[int(i), float(p)] for i, p in zip(ti[b], tv[b])],
                         "entropy0": float(ent[b]), "p_fixed": {str(t): float(v[b]) for t, v in fixed_p.items()},
                         "pos0_argmax": int(pos0[b])})
        return (rows, z0) if pos0_logits else rows

    def trajectories(self, cells, ckpt=None, log=print, stop_at=None):
        """Float64 trajectories of all cells in one batch: 1000 transient steps (base rows), then base and twin rows
        to the longest cell's end. Returns per cell the record (leading tokens, their probabilities and the
        runner-up's from step 1000 on), the 16 states, the CPU exponent and the state period at the end.
        stop_at (testing): save the checkpoint at that step and return None."""
        torch = self.torch
        n = len(cells)
        d = self.start.shape[-1]
        T1 = np.array([10.0 ** c["log10T"] for c in cells])
        T2 = np.concatenate([T1, T1])
        t_end = max(c["t_end"] for c in cells)
        R = t_end - TRANSIENT
        coll = defaultdict(list)
        for i, c in enumerate(cells):
            for k, t in enumerate(c["collect"]):
                coll[t].append((i, k))
        key = cells_hash(cells)
        st = None
        if ckpt is not None and Path(ckpt).exists():
            z = np.load(ckpt)
            if str(z["key"]) != key:
                raise SystemExit(f"{ckpt}: checkpoint of another cell list")
            st = {k: z[k] for k in z.files}
            log(json.dumps({"resumed_at_step": int(st["t"])}))
        if st is None:
            st = {"t": np.array(0), "X": np.repeat(self.start.cpu().numpy(), n, axis=0),
                  "leads": np.full((n, R), -1, np.int64), "pmax": np.zeros((n, R)), "p2": np.zeros((n, R)),
                  "growth": np.zeros(n), "states": np.zeros((n, N_STATES, d)), "ring": np.zeros((MAX_PERIOD + 1, n, d)),
                  "seconds": np.array(0.0)}
        t = int(st["t"])
        X = torch.as_tensor(st["X"], dtype=self.loop.state_dtype)
        leads, pmax, p2, growth, states, ring = (st[k] for k in ("leads", "pmax", "p2", "growth", "states", "ring"))
        secs = float(st["seconds"])
        tic = time.time()

        def save():
            tmp = Path(str(ckpt) + ".tmp.npz")
            np.savez(tmp, key=key, t=t, X=X.cpu().numpy(), leads=leads, pmax=pmax, p2=p2, growth=growth,
                     states=states, ring=ring, seconds=secs + time.time() - tic)
            os.replace(tmp, ckpt)

        while t < t_end:
            if t < TRANSIENT:
                X, _ = self.step(X, T1)
                t += 1
                if t == TRANSIENT:              # twins at absolute distance TWIN_EPS, one seeded direction per cell
                    dirs = np.concatenate([np.random.default_rng([TWIN_SEED, c["seed_key"]]).normal(size=(1, 1, d))
                                           for c in cells])
                    dirs = torch.as_tensor(dirs, dtype=X.dtype)
                    X = torch.cat([X, X + dirs / dirs.flatten(1).norm(dim=1)[:, None, None] * TWIN_EPS])
            else:
                X, p = self.step(X, T2)
                j = t - TRANSIENT
                top = p[:n].double().topk(2, dim=-1)
                leads[:, j] = top.indices[:, 0].cpu().numpy()
                pmax[:, j] = top.values[:, 0].cpu().numpy()
                p2[:, j] = top.values[:, 1].cpu().numpy()
                base, diff = X[:n], X[n:] - X[:n]
                dist = np.maximum(diff.flatten(1).norm(dim=1).cpu().numpy(), TWIN_EPS * 1e-30)
                growth += np.log(dist / TWIN_EPS)
                X = torch.cat([base, base + diff * torch.as_tensor(TWIN_EPS / dist, dtype=X.dtype)[:, None, None]])
                t += 1                          # X[:n] is now X_t
                b = base[:, 0].cpu().numpy()
                ring[t % (MAX_PERIOD + 1)] = b
                for i, k in coll.get(t, ()):
                    states[i, k] = b[i]
            if t % CHECKPOINT_EVERY == 0 or t == stop_at:
                if ckpt is not None:
                    save()
                log(json.dumps({"step": t, "of": t_end, "rows": int(X.shape[0]),
                                "seconds": round(secs + time.time() - tic, 1)}))
                if t == stop_at:
                    return None
        secs += time.time() - tic
        end = ring[t_end % (MAX_PERIOD + 1)]
        out = []
        for i, c in enumerate(cells):
            scale = max(1.0, float(np.linalg.norm(end[i])))
            per = next((q for q in range(1, MAX_PERIOD + 1)
                        if np.linalg.norm(ring[(t_end - q) % (MAX_PERIOD + 1)][i] - end[i]) < PERIOD_TOL * scale), 0)
            out.append({"leads": leads[i], "pmax": pmax[i], "p2": p2[i], "states": states[i],
                        "exponent": float(growth[i] / R), "period_end": int(per)})
        return out, {"t_end": t_end, "record_steps": R, "seconds": secs}


# ----------------------------------------------------------------------------------------------- trajectories

def summarize_cell(c, r, tok=None):
    """Per-cell checks from a trajectory result."""
    col = collision(r["leads"], TRANSIENT, c["collect"])
    own = r["leads"][:c["t_end"] - TRANSIENT]
    q_cpu = symbolic_period(own)
    S = np.asarray(r["states"])
    norms = np.linalg.norm(S, axis=1)
    dist = [float(np.linalg.norm(S[i] - S[j]) / norms.mean()) for i, j in pair_indices(len(S))]
    chaotic_cpu = r["exponent"] > EXP_MIN and r["period_end"] == 0
    if c["kind"] == "chaotic":
        qualifies = col["history_identical"] and chaotic_cpu
    else:
        qualifies = col["history_identical"] and r["period_end"] == PERIODIC_PERIOD
    reasons = []
    if not col["history_identical"]:
        reasons.append("histories differ")
    if c["kind"] == "chaotic" and not chaotic_cpu:
        reasons.append(f"not chaotic on the CPU (exponent {r['exponent']:.4g}, end period {r['period_end']})")
    if c["kind"] == "periodic" and r["period_end"] != PERIODIC_PERIOD:
        reasons.append(f"end state period {r['period_end']}")
    fam = canonical_cycle(own, q_cpu) if q_cpu else None
    lines_tokens = sorted({t for s in c["sources"] for t in s["lead_tokens"]})
    k_prev = [t - 1 - TRANSIENT for t in c["collect"]]
    out = {"uid": c["uid"], "kind": c["kind"], "split": c["split"], "log10T": c["log10T"],
           "text_period_lines": c["text_period"], "text_period_cpu": q_cpu, "spacing": c["spacing"],
           "collect": c["collect"], "exponent_cpu": r["exponent"], "period_end": r["period_end"],
           "exponents_lines": [s["lyapunov"] for s in c["sources"]],
           "history_identical": col["history_identical"], "future_identical": col["future_identical"],
           "history_mismatch": col["history_mismatch"], "future_mismatch": col["future_mismatch"],
           "current_token": col["current_token"], "family": list(fam) if fam else None,
           "tokens_match_lines": (sorted(set(int(x) for x in own)) == lines_tokens) if len(lines_tokens) < 8 else None,
           "state_norm_mean": float(norms.mean()), "pair_distance": {"min": min(dist), "median": float(np.median(dist)),
                                                                    "max": max(dist)},
           "lead_prob_at_states": [float(r["pmax"][k]) for k in k_prev],
           "lead_margin_min_record": float(np.min(r["pmax"][:len(own)] - r["p2"][:len(own)])),
           "qualifies": bool(qualifies), "reasons": reasons}
    if tok is not None:
        out["history_text"] = tok.decode(col["history"][-max(c["text_period"], q_cpu or 0):])
        out["family_text"] = tok.decode(list(fam)) if fam else None
        out["current_token_text"] = tok.decode([col["current_token"]])
    return out


def save_trajectories(out, cells, res, meta, tok):
    n = len(cells)
    R = len(res[0]["leads"])
    np.savez_compressed(out / "trajectories.npz", uids=np.array([c["uid"] for c in cells]),
                        states=np.stack([r["states"] for r in res]), leads=np.stack([r["leads"] for r in res]),
                        pmax=np.stack([r["pmax"] for r in res]), p2=np.stack([r["p2"] for r in res]),
                        exponent=np.array([r["exponent"] for r in res]),
                        period_end=np.array([r["period_end"] for r in res]), record_start=TRANSIENT,
                        record_steps=R)
    summ = [summarize_cell(c, r, tok) for c, r in zip(cells, res)]
    (out / "trajectories.json").write_text(json.dumps(jsonable({"meta": meta, "cells": summ, "n": n}), indent=1,
                                                      ensure_ascii=False) + "\n")
    return summ


def load_trajectories(out):
    z = np.load(Path(out) / "trajectories.npz")
    summ = json.loads((Path(out) / "trajectories.json").read_text())
    return z, summ


# ----------------------------------------------------------------------------------------------- run

def git_frozen(path):
    """The file is tracked, unmodified, and HEAD is contained in origin/main (pushed)."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    tracked = bool(git("ls-files", rel).strip())
    clean = not git("status", "--porcelain", "--", rel).strip()
    git("fetch", "--quiet", "origin", "main")
    r = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", "origin/main"],
                       capture_output=True, text=True)
    return tracked and clean and r.returncode == 0, {"tracked": tracked, "clean": clean,
                                                     "head_on_origin_main": r.returncode == 0}


def load_choices(path):
    ch = json.loads(Path(path).read_text())
    cand = {c["id"]: c for c in CANDIDATES}
    for t in TYPES:
        q = ch["queries"][t]
        c = cand.get(q["id"])
        if c is None or (c["type"], c["text"], c["cue"]) != (q["type"], q["text"], q["cue"]):
            raise SystemExit(f"choices {path}: query {q['id']} does not match CANDIDATES")
    if ch.get("extraction_version") != EXTRACTION_VERSION:
        raise SystemExit(f"choices {path}: extraction version {ch.get('extraction_version')} != {EXTRACTION_VERSION}")
    return ch


def readout_records(eng, rows, cell_ids, ks, query, layout):
    recs = []
    for r, uid, k in zip(rows, cell_ids, ks):
        label, fmt = extract(query["type"], r["answer"])
        recs.append(dict(r, kind="readout", cell=uid, k=k, query=query["id"], qtype=query["type"], layout=layout,
                         label=label, in_format=fmt))
    return recs


def cmd_run(args):
    import torch

    cells = [c for c in load_cells(args.cells) if c["split"] == args.split]
    if args.limit:
        cells = cells[:args.limit]
    choices = None
    if args.split == "heldout":
        if not args.choices:
            raise SystemExit("the held-out run needs --choices (the committed development choices)")
        ok, status = git_frozen(args.choices)
        if not ok and not args.allow_unfrozen:
            raise SystemExit(f"choices file not frozen (tracked, clean, pushed): {status}")
        choices = load_choices(args.choices)
        queries = [choices["queries"][t] for t in TYPES]
    else:
        queries = list(CANDIDATES)
        if args.queries:
            ids = {c["id"] for c in CANDIDATES}
            if not set(args.queries) <= ids:
                raise SystemExit(f"unknown candidates: {sorted(set(args.queries) - ids)}")
            queries = [c for c in CANDIDATES if c["id"] in args.queries]
        if args.choices:
            choices = load_choices(args.choices)
    if args.split == "heldout" and (args.queries or args.states_from):
        raise SystemExit("--queries and --states-from are for development runs only")
    out = out_dir(args.out, args.resume)
    log = open(out / "log.jsonl", "a")
    states_from = None
    if args.states_from and not (out / "trajectories.npz").exists():
        # readouts of further candidates on the states of an earlier development run: copy its trajectory files
        src = Path(args.states_from)
        states_from = {"run": str(src)}
        for name in ("trajectories.npz", "trajectories.json"):
            data = (src / name).read_bytes()
            (out / name).write_bytes(data)
            if sha256_file(out / name) != sha256_file(src / name):
                raise SystemExit(f"copy of {src / name} differs")
            states_from[name + "_sha256"] = sha256_file(out / name)

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    eng = Engine(args.threads)
    runf = out / "run.json"
    hist = json.loads(runf.read_text())["history"] if runf.exists() else []
    info = provenance(args, {"cells_hash": cells_hash(load_cells()), "cells": [c["uid"] for c in cells],
                             "queries": queries, "choices_sha256": sha256_file(args.choices) if args.choices else None,
                             "choices_frozen": git_frozen(args.choices)[1] if args.choices else None,
                             "float64": eng.float64, "model_snapshot": model_snapshot(), "prefix_len": eng.L0,
                             "prefix_text": eng.prefix_text, "states_from": states_from,
                             "extraction_version": EXTRACTION_VERSION,
                             "suffixes": {q["id"]: eng.suffix(q)[0] for q in queries}})
    hist.append(info)
    runf.write_text(json.dumps(jsonable({"history": hist}), indent=1, ensure_ascii=False) + "\n")

    # trajectories
    if not (out / "trajectories.npz").exists():
        res, meta = eng.trajectories(cells, ckpt=out / "checkpoint.npz", log=say)
        summ = save_trajectories(out, cells, res, meta, eng.tok)
        (out / "checkpoint.npz").unlink(missing_ok=True)
        say(json.dumps({"trajectories": "done", "seconds": round(meta["seconds"], 1),
                        "qualifying": sum(s["qualifies"] for s in summ), "cells": len(summ)}))
    z, summ = load_trajectories(out)
    uids = [str(u) for u in z["uids"]]
    if uids != [c["uid"] for c in cells]:
        raise SystemExit("trajectories.npz holds another cell list")
    S = torch.as_tensor(z["states"], dtype=eng.loop.state_dtype)        # (n, 16, d)
    cur = [s["current_token"] for s in summ["cells"]]

    # readouts
    recp = out / "readouts.jsonl"
    done = set()
    if recp.exists():
        for line in open(recp):
            r = json.loads(line)
            if r["kind"] == "done":
                done.add(r["batch"])
    recf = open(recp, "a")
    fixed_all = sorted({int(choices["queries"][t]["fixed_token"]) for t in TYPES}) if choices else []

    def emit(key, recs, sec, nrows):
        for r in recs:
            r["batch"] = key
            recf.write(json.dumps(r, ensure_ascii=False) + "\n")
        recf.write(json.dumps({"kind": "timing", "batch": key, "rows": nrows, "seconds": round(sec, 2)}) + "\n")
        recf.write(json.dumps({"kind": "done", "batch": key}) + "\n")
        recf.flush()

    n = len(cells)
    for q in queries:
        _, sids = eng.suffix(q)
        for b0 in range(0, n, MAIN_CELLS_PER_BATCH):
            key = f"main-{q['id']}-{b0}"
            if key in done:
                continue
            idx = list(range(b0, min(n, b0 + MAIN_CELLS_PER_BATCH)))
            X = S[idx].reshape(-1, 1, S.shape[-1])
            tic = time.time()
            rows = eng.readout(X, sids, fixed_all)
            recs = readout_records(eng, rows, [uids[i] for i in idx for _ in range(N_STATES)],
                                   [k for _ in idx for k in range(N_STATES)], q, "main")
            emit(key, recs, time.time() - tic, len(rows))
            say(json.dumps({"batch": key, "rows": len(rows), "seconds": round(time.time() - tic, 1)}))
        key = f"ref-{q['id']}"
        if key not in done:
            X = eng.token_embedding(cur)[:, None, :]
            tic = time.time()
            rows = eng.readout(X, sids, fixed_all)
            recs = readout_records(eng, rows, uids, [-1] * n, q, "reference")
            emit(key, recs, time.time() - tic, len(rows))
            say(json.dumps({"batch": key, "rows": len(rows), "seconds": round(time.time() - tic, 1)}))
    # negative control: states reloaded from the saved file, one cell per batch, reversed order
    z2 = np.load(out / "trajectories.npz")
    S2 = torch.as_tensor(z2["states"], dtype=eng.loop.state_dtype)
    for q in queries:
        _, sids = eng.suffix(q)
        for i in range(n):
            key = f"rec-{q['id']}-{uids[i]}"
            if key in done:
                continue
            order = list(range(N_STATES))[::-1]
            tic = time.time()
            rows = eng.readout(S2[i, order][:, None, :], sids, fixed_all)
            recs = readout_records(eng, rows, [uids[i]] * N_STATES, order, q, "recompute")
            emit(key, recs, time.time() - tic, len(rows))
        say(json.dumps({"recompute": q["id"], "cells": n}))
    hist[-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runf.write_text(json.dumps(jsonable({"history": hist}), indent=1, ensure_ascii=False) + "\n")
    say(json.dumps({"run": "done", "out": str(out)}))


# ----------------------------------------------------------------------------------------------- smoke

def cmd_smoke(args):
    import torch

    out = out_dir(args.out, False)
    eng = Engine(args.threads)
    allc = candidate_cells()
    if any(abs(c["log10T"] - lt) < 1e-9 for c in allc["chaotic"] + allc["periodic"]
           for lt in (SMOKE_LOG10T, SMOKE_LOG10T_PERIODIC)):
        raise SystemExit("a smoke cell is a candidate")
    res = {"provenance": provenance(args, {"model_snapshot": model_snapshot()}), "float64": eng.float64,
           "smoke_log10T": SMOKE_LOG10T}
    T = 10.0 ** SMOKE_LOG10T
    X = eng.start.repeat(4, 1, 1)
    for _ in range(64):
        X, _ = eng.step(X, np.full(4, T))
    q = CANDIDATES[0]
    tail, sids = eng.suffix(q)
    res["suffix"] = {"text": tail, "ids": sids, "tokens": [eng.tok.decode([i]) for i in sids]}
    fixed = eng.tok("Yes", add_special_tokens=False)["input_ids"]
    # 1. the readout's soft-position logits equal the loop's step logits; the step is unchanged after a readout
    eng.use(4)
    z_loop = eng.loop.logits(X)
    Y1, _ = eng.step(X, np.full(4, T))
    rows, z0 = eng.readout(X, sids, fixed, pos0_logits=True)
    Y2, _ = eng.step(X, np.full(4, T))
    res["pos0_logits_max_abs_diff"] = float((z0 - z_loop).abs().max())
    res["pos0_logits_max_abs"] = float(z_loop.abs().max())
    res["step_unchanged_after_readout"] = bool(torch.equal(Y1, Y2))
    res["cache_len_after_readout"] = int(eng.loop.cache.get_seq_length())
    res["prefix_len"] = eng.L0
    # 2. batch layout: one batch of 4 against single rows, and reversed order
    singles = [eng.readout(X[i:i + 1], sids, fixed)[0] for i in range(4)]
    rev = eng.readout(X.flip(0), sids, fixed)[::-1]
    fk = str(fixed[0])
    res["layout"] = {"ids_equal_single": all(a["ids"] == b["ids"] for a, b in zip(rows, singles)),
                     "ids_equal_reversed": all(a["ids"] == b["ids"] for a, b in zip(rows, rev)),
                     "p_fixed_max_abs_diff_single": max(abs(a["p_fixed"][fk] - b["p_fixed"][fk])
                                                        for a, b in zip(rows, singles)),
                     "p_fixed_max_abs_diff_reversed": max(abs(a["p_fixed"][fk] - b["p_fixed"][fk])
                                                          for a, b in zip(rows, rev))}
    res["answers"] = [r["answer"] for r in rows]
    # 3. central differences of log p(fixed) at the first answer position along a random unit direction
    rng = np.random.default_rng(SMOKE_SEED)
    x0 = X[:1]
    u = torch.as_tensor(rng.normal(size=tuple(x0.shape)), dtype=x0.dtype)
    u = u / u.norm()
    scale = float(x0.norm())
    hs = [10.0 ** -k for k in range(1, 10)]
    batch = torch.cat([x0 + h * scale * u for h in hs] + [x0 - h * scale * u for h in hs])
    rr = eng.readout(batch, sids, fixed)
    f = np.log([r["p_fixed"][fk] for r in rr])
    D = [(f[i] - f[len(hs) + i]) / (2 * hs[i]) for i in range(len(hs))]
    res["central_differences_readout"] = {"h_rel": hs, "estimate": D,
                                          "successive_abs_diff": [abs(D[i] - D[i + 1]) for i in range(len(D) - 1)]}
    # 4. one map step (as hidden_chaos.fd_probe): |F(x + h u) - F(x - h u)| / (2 h scale) for each h
    Yp, _ = eng.step(torch.cat([x0 + h * scale * u for h in hs]), np.full(len(hs), T))
    Ym, _ = eng.step(torch.cat([x0 - h * scale * u for h in hs]), np.full(len(hs), T))
    J = [(Yp[i] - Ym[i]) / (2 * hs[i] * scale) for i in range(len(hs))]
    res["central_differences_step"] = {"h_rel": hs, "ratio": [float(j.norm()) for j in J],
                                       "successive_rel_diff": [float((J[i] - J[i + 1]).norm() / J[i + 1].norm())
                                                               for i in range(len(J) - 1)]}
    # 5. timing of a 16-row and a 128-row readout
    tim = {}
    for B in (16, 128):
        XB = X.repeat(B // 4, 1, 1)
        tic = time.time()
        eng.readout(XB, sids, fixed)
        tim[str(B)] = round(time.time() - tic, 2)
    res["readout_seconds"] = tim
    # 6. the trajectory code on two cells that are neither candidates nor controls: a full run, and a run stopped at
    # step 1500 and resumed from its checkpoint (must agree bitwise)
    def synthetic(lt, kind, key):
        return {"uid": f"smoke-{kind}", "kind": kind, "split": "smoke", "log10T": lt, "text_period": 7,
                "spacing": spacing(7), "collect": collection_steps(7), "t_end": record_end(7), "seed_key": key,
                "rank": -1, "sources": [{"lyapunov": [0.0, 0.0], "lead_tokens": []}]}

    scells = [synthetic(SMOKE_LOG10T, "chaotic", 999901), synthetic(SMOKE_LOG10T_PERIODIC, "periodic", 999902)]
    tic = time.time()
    ra, meta = eng.trajectories(scells, log=lambda s: None)
    ta = time.time() - tic
    ck = out / "smoke-checkpoint.npz"
    eng.trajectories(scells, ckpt=ck, log=lambda s: None, stop_at=1500)
    rb, _ = eng.trajectories(scells, ckpt=ck, log=lambda s: None)
    res["trajectory"] = {"seconds_full_run": round(ta, 1), "meta": meta,
                         "resume_bitwise_equal": all(np.array_equal(a["states"], b["states"])
                                                     and np.array_equal(a["leads"], b["leads"])
                                                     and a["exponent"] == b["exponent"] for a, b in zip(ra, rb)),
                         "cells": [summarize_cell(c, r, eng.tok) for c, r in zip(scells, ra)]}
    (out / "smoke.json").write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(jsonable({k: v for k, v in res.items() if k != "provenance"}), indent=1, ensure_ascii=False)[:6000])


# ----------------------------------------------------------------------------------------------- analysis

def load_readouts(run):
    """Readout records with answers, labels and formats recomputed from the saved decoded tokens (raw) by the current
    extraction rules (the run-time values are kept as answer_run, label_run and in_format_run)."""
    recs = [r for r in (json.loads(line) for line in open(Path(run) / "readouts.jsonl")) if r["kind"] == "readout"]
    for r in recs:
        r["answer_run"], r["label_run"], r["in_format_run"] = r["answer"], r["label"], r["in_format"]
        r["answer"] = cut_answer(r["raw"])
        r["label"], r["in_format"] = extract(r["qtype"], r["answer"])
    return recs


def analyze_run(run, choices=None):
    run = Path(run)
    z, summ = load_trajectories(run)
    cells = {s["uid"]: s for s in summ["cells"]}
    uids = [str(u) for u in z["uids"]]
    S = z["states"]
    leads = z["leads"]
    recs = load_readouts(run)
    hist = json.loads((run / "run.json").read_text())["history"]
    queries = hist[-1]["queries"]
    fixed = {q["id"]: (choices["queries"][q["type"]]["fixed_token"] if choices and q["id"] ==
                       choices["queries"][q["type"]]["id"] else None) for q in queries}
    main, recomp, ref = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for r in recs:
        tgt = {"main": main, "recompute": recomp, "reference": ref}[r["layout"]]
        tgt[(r["query"], r["cell"])][r["k"]] = r
    issues = []
    # gates: every planned readout present; the soft-position argmax equals the recorded next leading token
    pos0_bad = 0
    for qd in queries:
        for i, u in enumerate(uids):
            m = main.get((qd["id"], u), {})
            if sorted(m) != list(range(N_STATES)) or sorted(recomp.get((qd["id"], u), {})) != list(range(N_STATES)) \
                    or -1 not in ref.get((qd["id"], u), {}):
                issues.append(f"missing readouts: {qd['id']} {u}")
                continue
            for k in range(N_STATES):
                t = cells[u]["collect"][k]
                if m[k]["pos0_argmax"] != int(leads[i, t - TRANSIENT]):
                    pos0_bad += 1
    per_query = {}
    for qd in queries:
        qid = qd["id"]
        if fixed[qid] is None:            # development: the modal first token over chaotic development states
            first = Counter(int(main[(qid, u)][k]["ids"][0]) for u in uids if cells[u]["kind"] == "chaotic"
                            for k in main.get((qid, u), {}))
            top = max(first.values()) if first else 0
            fixed[qid] = min(t for t, c in first.items() if c == top) if first else None
        fk = str(fixed[qid])
        rows = {}
        for i, u in enumerate(uids):
            m = main.get((qid, u), {})
            if sorted(m) != list(range(N_STATES)):
                continue
            labels = [m[k]["label"] for k in range(N_STATES)]
            texts = [tuple(m[k]["ids"]) for k in range(N_STATES)]
            fmt = [m[k]["in_format"] for k in range(N_STATES)]
            # the fixed token's probability: recorded exactly when the token was fixed before the run (held out),
            # else from the first position's top 50 (development; None if the token is not among them)
            def prob_fixed(rec):
                return rec["p_fixed"][fk] if fk in rec["p_fixed"] else \
                    next((p for t, p in rec["first_top"] if t == fixed[qid]), None)

            pf = [prob_fixed(m[k]) for k in range(N_STATES)]
            d_lab, n_lab, n_pairs = pair_disagreement(labels)
            d_txt, _, _ = pair_disagreement(texts)
            pv = [(labels[a] != labels[b]) for a, b in pair_indices(N_STATES) if fmt[a] and fmt[b]]
            psig = [sig(p) if p is not None else None for p in pf]
            d_pos = pair_disagreement(psig)[0] if None not in psig else None
            norms = np.linalg.norm(S[i], axis=1)
            dist = [float(np.linalg.norm(S[i][a] - S[i][b]) / norms.mean()) for a, b in pair_indices(N_STATES)]
            dis = [labels[a] != labels[b] for a, b in pair_indices(N_STATES)]
            dis_txt = [texts[a] != texts[b] for a, b in pair_indices(N_STATES)]
            g0 = [m[k]["gap"][0] for k in range(N_STATES)]
            dp = [abs(pf[a] - pf[b]) for a, b in pair_indices(N_STATES)] if None not in pf else None
            rf = ref.get((qid, u), {}).get(-1)
            rc = recomp.get((qid, u), {})
            neg = None
            if sorted(rc) == list(range(N_STATES)):
                both = [(a, prob_fixed(rc[k])) for k, a in enumerate(pf)
                        if a is not None and prob_fixed(rc[k]) is not None]
                neg = {"label_differs": sum(rc[k]["label"] != m[k]["label"] for k in range(N_STATES)),
                       "ids_differ": sum(rc[k]["ids"] != m[k]["ids"] for k in range(N_STATES)),
                       "p_fixed_compared": len(both),
                       "p_fixed_sig_differs": sum(sig(a) != sig(b) for a, b in both) if both else None,
                       "p_fixed_max_abs_diff": max(abs(a - b) for a, b in both) if both else None,
                       "entropy0_max_abs_diff": max(abs(rc[k]["entropy0"] - m[k]["entropy0"]) for k in range(N_STATES)),
                       "states": N_STATES}
            rows[u] = {"kind": cells[u]["kind"], "split": cells[u]["split"], "qualifies": cells[u]["qualifies"],
                       "family": str(cells[u]["family"]), "family_text": cells[u].get("family_text"),
                       "disagreement": d_lab, "differing_pairs": n_lab, "pairs": n_pairs,
                       "text_disagreement": d_txt,
                       "disagreement_in_format": (sum(pv) / len(pv)) if pv else None, "pairs_in_format": len(pv),
                       "in_format_share": sum(fmt) / N_STATES, "labels": dict(Counter(labels)),
                       "answers": dict(Counter(m[k]["answer"] for k in range(N_STATES))),
                       "positive_share": d_pos, "positive_pass": (len(set(psig)) > 1) if None not in psig else None,
                       "p_fixed_range": [min(pf), max(pf)] if None not in pf else None,
                       "reference_label": rf["label"] if rf else None,
                       "reference_answer": rf["answer"] if rf else None,
                       "differs_from_reference": (sum(lab != rf["label"] for lab in labels) / N_STATES) if rf else None,
                       "distance": {"min": min(dist), "median": float(np.median(dist)), "max": max(dist)},
                       "auc_distance": auc([x for x, y in zip(dist, dis) if y], [x for x, y in zip(dist, dis) if not y]),
                       "spearman_distance_dp": spearman(dist, dp) if dp else None,
                       "first_gap_median": float(np.median(g0)),
                       # margins (descriptive, added after dev-2): the top-2 logit gap at the first answer position
                       # (its smallest value and its range over the 16 states) and the smallest gap anywhere on the
                       # 8-token greedy paths (how close a state came to a different greedy token)
                       "first_gap_min": float(min(g0)), "first_gap_range": float(max(g0) - min(g0)),
                       "path_gap_min": float(min(min(m[k]["gap"]) for k in range(N_STATES))),
                       "auc_distance_text": auc([x for x, y in zip(dist, dis_txt) if y],
                                                [x for x, y in zip(dist, dis_txt) if not y]),
                       "negative": neg, "_dist": dist, "_dis": dis, "_dis_txt": dis_txt}
        per_query[qid] = {"query": qd, "fixed_token": fixed[qid], "cells": rows}
    groups = {}
    for qid, pq in per_query.items():
        groups[qid] = {}
        for kind in ("chaotic", "periodic"):
            for split in ("dev", "heldout"):
                rs = {u: r for u, r in pq["cells"].items() if r["kind"] == kind and r["split"] == split}
                if not rs:
                    continue
                qual = {u: r for u, r in rs.items() if r["qualifies"]}
                g = summarize_group(qual)
                g["cells_total"] = len(rs)
                g["cells_not_qualifying"] = sorted(set(rs) - set(qual))
                g["in_format_share_all_cells"] = float(np.mean([r["in_format_share"] for r in rs.values()]))
                groups[qid][f"{kind}-{split}"] = g
    for qid in per_query:
        for r in per_query[qid]["cells"].values():
            r.pop("_dist")
            r.pop("_dis")
            r.pop("_dis_txt")
    def neg_total(pq):
        ns = [r["negative"] for r in pq["cells"].values() if r["negative"]]
        cmp = [n for n in ns if n["p_fixed_compared"]]
        return {"label_differs": sum(n["label_differs"] for n in ns), "ids_differ": sum(n["ids_differ"] for n in ns),
                "p_fixed_compared": sum(n["p_fixed_compared"] for n in ns),
                "p_fixed_sig_differs": sum(n["p_fixed_sig_differs"] for n in cmp) if cmp else None,
                "p_fixed_max_abs_diff": max(n["p_fixed_max_abs_diff"] for n in cmp) if cmp else None,
                "entropy0_max_abs_diff": max([n["entropy0_max_abs_diff"] for n in ns] or [None]),
                "states": sum(n["states"] for n in ns)}

    neg_tot = {qid: neg_total(pq) for qid, pq in per_query.items()}
    gate = {"issues": issues, "pos0_mismatch": pos0_bad, "readouts": len(recs),
            "labels_differ_from_run_time": sum(r["label"] != r["label_run"] or r["in_format"] != r["in_format_run"]
                                               or r["answer"] != r["answer_run"] for r in recs),
            "cells": len(uids), "qualifying": sum(c["qualifies"] for c in cells.values())}
    return {"run": str(run), "commit": hist[-1]["commit"], "tool_sha256": hist[-1]["tool_sha256"], "gate": gate,
            "trajectories": summ, "groups": groups, "negative_control": neg_tot, "per_query": per_query}


def summarize_group(rows):
    """Endpoint, controls and distance description over the qualifying cells of one group (cells are the units)."""
    us = sorted(rows)
    if not us:
        return {"cells": 0}
    vals = [rows[u]["disagreement"] for u in us]
    fams = [rows[u]["family"] for u in us]
    dist = np.concatenate([rows[u]["_dist"] for u in us])
    dis = np.concatenate([rows[u]["_dis"] for u in us]).astype(bool)
    dis_txt = np.concatenate([rows[u]["_dis_txt"] for u in us]).astype(bool)
    qs = np.quantile(dist, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(dist, qs)
    quint = [{"bin": int(b), "pairs": int((bins == b).sum()),
              "distance_range": [float(dist[bins == b].min()), float(dist[bins == b].max())],
              "disagreement": float(dis[bins == b].mean()), "text_disagreement": float(dis_txt[bins == b].mean())}
             for b in range(5) if (bins == b).any()]
    aucs = [rows[u]["auc_distance"] for u in us]
    aucs_txt = [rows[u]["auc_distance_text"] for u in us]
    fam_means, fam_text = defaultdict(list), {}
    for u in us:
        fam_means[rows[u]["family"]].append(rows[u]["disagreement"])
        fam_text[rows[u]["family"]] = rows[u]["family_text"]

    def mean_ci(key):
        x = [rows[u][key] for u in us if rows[u][key] is not None]
        return {"mean": float(np.mean(x)) if x else None, "boot95": bootstrap_mean_ci(x), "cells": len(x)}

    pos = [rows[u]["positive_pass"] for u in us if rows[u]["positive_pass"] is not None]
    n_any = sum(v > 0 for v in vals)
    return {"cells": len(us), "disagreement": dict(mean_ci("disagreement"),
                                                  boot95_families=cluster_bootstrap_ci(vals, fams),
                                                  families=len(set(fams)),
                                                  cells_with_any=n_any,
                                                  # post-run addition (see cp_upper)
                                                  cells_with_any_upper95_one_sided=cp_upper(n_any, len(us)),
                                                  differing_pairs=sum(rows[u]["differing_pairs"] for u in us),
                                                  pairs=sum(rows[u]["pairs"] for u in us),
                                                  max=max(vals)),
            "text_disagreement": dict(mean_ci("text_disagreement"),
                                      cells_with_any=sum(rows[u]["text_disagreement"] > 0 for u in us),
                                      max=max(rows[u]["text_disagreement"] for u in us)),
            "disagreement_in_format": mean_ci("disagreement_in_format"),
            "in_format_share": mean_ci("in_format_share"),
            "differs_from_reference": mean_ci("differs_from_reference"),
            "positive_share": mean_ci("positive_share"),
            "positive_pass_cells": sum(pos), "positive_cells": len(pos),
            "distance_median_of_cell_medians": float(np.median([rows[u]["distance"]["median"] for u in us])),
            "distance_range": [float(min(rows[u]["distance"]["min"] for u in us)),
                               float(max(rows[u]["distance"]["max"] for u in us))],
            "auc_distance": {"mean": float(np.mean([a for a in aucs if a is not None])) if any(a is not None for a in aucs)
                             else None, "boot95": bootstrap_mean_ci(aucs), "cells": sum(a is not None for a in aucs)},
            "auc_distance_text": {"mean": float(np.mean([a for a in aucs_txt if a is not None]))
                                  if any(a is not None for a in aucs_txt) else None,
                                  "boot95": bootstrap_mean_ci(aucs_txt), "cells": sum(a is not None for a in aucs_txt)},
            "agreeing_pair_distance_max": float(dist[~dis].max()) if (~dis).any() else None,
            "margins": {"first_gap_min_min": float(min(rows[u]["first_gap_min"] for u in us)),
                        "first_gap_min_median": float(np.median([rows[u]["first_gap_min"] for u in us])),
                        "first_gap_range_median": float(np.median([rows[u]["first_gap_range"] for u in us])),
                        "first_gap_range_max": float(max(rows[u]["first_gap_range"] for u in us)),
                        "path_gap_min_min": float(min(rows[u]["path_gap_min"] for u in us)),
                        "path_gap_min_median": float(np.median([rows[u]["path_gap_min"] for u in us]))},
            "disagreement_by_distance_quintile": quint,
            "spearman_distance_dp_median": float(np.median([rows[u]["spearman_distance_dp"] for u in us
                                                            if rows[u]["spearman_distance_dp"] is not None]))
            if any(rows[u]["spearman_distance_dp"] is not None for u in us) else None,
            "families": {f: {"text": fam_text[f], "cells": len(v), "mean_disagreement": float(np.mean(v))}
                         for f, v in sorted(fam_means.items())}}


def cmd_analyze(args):
    choices = load_choices(args.choices) if args.choices else None
    res = analyze_run(args.run, choices)
    res["analysis_tool_sha256"] = sha256_file(__file__)
    res["analysis_commit"] = git("rev-parse", "HEAD").strip()
    res["extraction_version"] = EXTRACTION_VERSION
    path = Path(args.run) / (args.name or "analysis.json")
    path.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False) + "\n")
    brief = {"gate": res["gate"], "negative": res["negative_control"],
             "groups": {q: {g: {k: v for k, v in d.items() if k in ("cells", "disagreement", "text_disagreement",
                                                                   "in_format_share", "positive_share",
                                                                   "positive_pass_cells", "cells_not_qualifying",
                                                                   "differs_from_reference")}
                            for g, d in gs.items()} for q, gs in res["groups"].items()}}
    print(json.dumps(jsonable(brief), indent=1, ensure_ascii=False)[:20000])
    print(f"wrote {path}")


def cmd_choose(args):
    """The development rule over the main readouts of the chaotic development states of one or more development
    runs on the same states (every candidate read in exactly one run)."""
    runs = [Path(r) for r in args.runs]
    if len({sha256_file(r / "trajectories.npz") for r in runs}) != 1:
        raise SystemExit("the development runs do not share their states")
    rows, seen, dev_runs = [], {}, []
    for run in runs:
        _, summ = load_trajectories(run)
        kinds = {s["uid"]: (s["kind"], s["split"]) for s in summ["cells"]}
        if any(s != "dev" for _, s in kinds.values()):
            raise SystemExit(f"{run} is not a development run")
        for r in load_readouts(run):
            if r["layout"] != "main" or kinds[r["cell"]] != ("chaotic", "dev"):
                continue
            if seen.setdefault(r["query"], str(run)) != str(run):
                raise SystemExit(f"candidate {r['query']} was read in two runs")
            rows.append(r)
        h = json.loads((run / "run.json").read_text())["history"]
        dev_runs.append({"run": str(run), "commit": h[-1]["commit"], "tool_sha256": h[-1]["tool_sha256"],
                         "candidates": sorted({q["id"] for q in h[-1]["queries"]})})
    chosen, table = choose_queries(rows)
    from transformers import AutoTokenizer
    import explore_llm_attractors as ela
    tok = AutoTokenizer.from_pretrained(ela.MODELS[MODEL], cache_dir=ela.CACHE)
    for t in TYPES:
        chosen[t]["fixed_token_text"] = tok.decode([chosen[t]["fixed_token"]])
    for qid in table:
        table[qid]["modal_first_token_text"] = tok.decode([table[qid]["modal_first_token"]])
    res = {"card": "C12-5 (docs/research-log.md, 'Run card C12-5')",
           "rule": "per type, the candidate with the largest in-format share over the main readouts of the chaotic "
                   "development states (ties: the earlier candidate in CANDIDATES); positive-control token: the most "
                   "frequent first greedy answer token of the chosen query on those readouts (ties: the smaller id)",
           "extraction_version": EXTRACTION_VERSION, "queries": chosen, "table": table,
           "dev_runs": dev_runs, "dev_readouts": len(rows),
           "tool_sha256": sha256_file(__file__), "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.json).write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(jsonable(res), indent=1, ensure_ascii=False))


def cmd_cells(args):
    cells = make_cells()
    res = {"card": "C12-5 (docs/research-log.md, 'Run card C12-5')",
           "rules": {"chaotic": "hidden_chaos.classify 'hidden chaos' in either line file (no state period, exponent "
                                "> 1e-3 at both offsets, periodic leading token), merged by log10 T; sorted by log10 T; "
                                "development = ranks floor((2i + 1) N / 16), i = 0..7",
                     "periodic": "state period 7 and leading-token period 7 in either line file, merged by log10 T; "
                                 "16 at ranks floor((2i + 1) N / 32); development = positions 4 and 12 of these 16",
                     "text_period": "lcm of the line files' leading-token periods",
                     "states": "X_t at t = 1070 + k s, k = 0..15, s = q ceil(70 / q)"},
           "line_files": [{"path": f"explore/{rel}", "sha256": sha256_file(EXPLORE / rel)} for rel in LINE_FILES],
           "counts": {f"{k}-{s}": sum(c["kind"] == k and c["split"] == s for c in cells)
                      for k in ("chaotic", "periodic") for s in ("dev", "heldout")},
           "candidates_total": {k: len(v) for k, v in candidate_cells().items()},
           "hash": cells_hash(cells), "cells": cells}
    Path(args.json).write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"counts": res["counts"], "candidates_total": res["candidates_total"], "hash": res["hash"]}))


def cmd_receipt(args):
    notes = json.loads(Path(args.notes).read_text())
    files = []
    for run in args.runs:
        for p in sorted(Path(run).iterdir()):
            if p.is_file():
                files.append({"path": str(p.relative_to(DATA_ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    analyses = {}
    cols = ["uid", "disagreement", "text_disagreement", "in_format_share", "positive_share", "distance_median",
            "differs_from_reference", "labels", "reference_label", "qualifies", "first_gap_min", "first_gap_range",
            "path_gap_min"]
    chosen = None
    if CHOICES_JSON.exists():
        chosen = {q["id"] for q in json.loads(CHOICES_JSON.read_text())["queries"].values()}
    seen_cells = {}
    for a in args.analyses:
        d = json.loads(Path(a).read_text())
        e = {k: d[k] for k in ("run", "commit", "tool_sha256", "gate", "groups", "negative_control",
                               "analysis_commit", "extraction_version")}
        cl = [{k: s[k] for k in ("uid", "kind", "split", "log10T", "qualifies", "reasons", "exponent_cpu",
                                 "period_end", "text_period_cpu", "family_text", "history_identical",
                                 "future_identical", "pair_distance", "current_token_text")}
              for s in d["trajectories"]["cells"]]
        key = json.dumps(cl, sort_keys=True)
        e["cells"] = cl if key not in seen_cells else f"as in {seen_cells[key]}"
        seen_cells.setdefault(key, str(Path(a).relative_to(DATA_ROOT)))
        e["per_cell"] = {qid: {"fixed_token": pq["fixed_token"], "columns": cols,
                               "rows": [[u, r["disagreement"], r["text_disagreement"], r["in_format_share"],
                                         r["positive_share"], r["distance"]["median"], r["differs_from_reference"],
                                         r["labels"], r["reference_label"], r["qualifies"], r.get("first_gap_min"),
                                         r.get("first_gap_range"), r.get("path_gap_min")]
                                        for u, r in pq["cells"].items()]}
                         for qid, pq in d["per_query"].items() if chosen is None or qid in chosen}
        analyses[str(Path(a).relative_to(DATA_ROOT))] = e
    res = dict(notes)
    res.update({"cells_file": {"path": "docs/pi/c12-5-cells.json", "sha256": sha256_file(CELLS_JSON)},
                "choices_file": {"path": "docs/pi/c12-5-choices.json", "sha256": sha256_file(CHOICES_JSON)}
                if CHOICES_JSON.exists() else None,
                "analyses": analyses, "raw": {"root": str(DATA_ROOT), "files": files},
                "receipt_tool_sha256": sha256_file(__file__),
                "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    Path(args.json).write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.json} ({len(files)} raw files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cells")
    p.add_argument("--json", default=str(CELLS_JSON))
    p.set_defaults(func=cmd_cells)
    p = sub.add_parser("smoke")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("run")
    p.add_argument("--split", choices=("dev", "heldout"), required=True)
    p.add_argument("--cells", default=str(CELLS_JSON))
    p.add_argument("--choices", default=None)
    p.add_argument("--allow-unfrozen", action="store_true", help="testing only: skip the frozen-choices check")
    p.add_argument("--queries", nargs="+", default=None, help="development: only these candidate ids")
    p.add_argument("--states-from", default=None,
                   help="development: copy the trajectories (states) of an earlier development run instead of "
                        "recomputing them")
    p.add_argument("--limit", type=int, default=None, help="testing only: the first N cells of the split")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("analyze")
    p.add_argument("run")
    p.add_argument("--choices", default=None)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("choose")
    p.add_argument("runs", nargs="+")
    p.add_argument("--json", default=str(CHOICES_JSON))
    p.set_defaults(func=cmd_choose)
    p = sub.add_parser("receipt")
    p.add_argument("--notes", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--analyses", nargs="+", required=True)
    p.add_argument("--json", default=str(ROOT / "docs" / "pi" / "c12-5-summary.json"))
    p.set_defaults(func=cmd_receipt)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
