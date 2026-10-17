"""C12-3 (exploratory, program PI): does coupling two soft-thinking agents manufacture agreement on the same wrong
answer? Run card "C12-3" in docs/research-log.md (consult 12, direction 3:
docs/consults/2026-10-16-practical-ideas/review.md, section 3, parts 3A and 3B).

Task. Synthetic two-digit multiplication (make_problems, seed PROBLEM_SEED): "What is a times b?" with a, b in 11..99,
no multiple of 10, unordered pairs never repeated; 16 development and 64 held-out problems (and 2 smoke problems
for engineering checks, neither development nor held-out). Qwen tokenizes digits one
by one, so every prompt has the same token length (checked) and batches need no padding. Scoring: the cue
"\\n\\nThe answer is" and ANSWER_TOKENS greedy tokens; pi_harness.extract_answer of "The answer is" + text gives the
answer (None: unparseable, never an agreement).

Agents. Two copies of one model (shared weights, separate contexts), each running soft thinking on the same prompt
(pi_harness soft mode as in C12-4: softmax(z / T), top-15 renormalized, T = 0.6, CPU float32, special and added tokens
masked from the support so the soft steps stay inside the assistant turn). Soft step 0 feeds the input embedding of the
agent's start token; at step t >= 1 agent X computes its concept token e_X(t) from its own last logits and feeds

    x_X(t) = (1 - c) e_X(t) + c m_X(t)   on a coupled step,   x_X(t) = e_X(t)   otherwise,

where m_X(t) is the message it receives. A pair k (k = 0..7) has start tokens START_WORDS[k] (agent A) and
START_WORDS[(k + 1) % 8] (agent B); with the same prompt and start, an uncoupled agent is deterministic, so the starts
are the only difference between the two agents. Every arm makes the same model calls (N soft steps and the readouts per
agent): equal compute.

Arms (per problem, 8 pairs x 2 agents each):
  independent  no coupled step;
  live         every step t >= 1 coupled, m_A(t) = e_B(t), m_B(t) = e_A(t) (reciprocal, simultaneous exchange);
  sparse       live exchange only at t = 4, 8, 12, ... (3B: every fourth step, same calls);
  replay       every step coupled, but m_X(t) is the concept token that agent X's partner role produced in the live run
               of pair (k + 1) % 8 on the same problem: plausible, task-specific messages without reciprocity;
  unrelated    as replay, from the live run of pair k on the next problem of the split (negative control for
               task-specific content);
  selfreplay   as replay, from the live run of the same pair k (engineering control: reproduces the live arm exactly
               when a row's arithmetic does not depend on its batch position);
  live50, replay50  live and replay at the strong coupling c = STRONG = 0.5 (secondary: at c = 0.5 both agents of a
               live pair feed the same input from step 1 on, so the starts are their only difference).
All other arms use c = COUPLING = 0.25. Phase 1 runs independent, live, sparse and live50 in one batch per problem (64
rows) and saves the live messages; phase 2 runs replay, unrelated, selfreplay and replay50 (64 rows, the same batch
shape).

Endpoints (analyze): per problem and arm, same-wrong agreement SWA = share of the 8 pairs whose agents both give a
parsed, wrong and equal answer; accuracy = share of the 16 agents correct; agreement = share of pairs with equal parsed
answers. Primary: live minus independent SWA, the mean over held-out problems (units), percentile bootstrap over
problems. Secondary: live minus replay, sparse minus live (SWA and accuracy), unrelated minus independent, and each
arm's excess over chance pairing (SWA of cross pairs of agents with different starts from different pairs, same arm and
problem). Detector check: pairing every agent with itself must give SWA = its error rate. Synchronization: relative
distance |e_A - e_B| / |e_B| of the pair's concept tokens and equality of their top-1 tokens per step.

Subcommands (prefix CUDA_VISIBLE_DEVICES= ; nice -n 10 uv run --no-sync python; --threads 6):
  problems   write docs/pi/c12-3-problems.json (and its SHA-256), check prompt lengths;
  run        phases 1 and 2 on one split with one model (resumes per problem and phase);
  analyze    summary JSON of a run;
  choose     the development model rule (choose_model) -> docs/pi/c12-3-choices.json;
  receipt    compact receipt (docs/pi/c12-3-summary.json) with SHA-256 of the raw files.
Raw outputs: /data2/neural-shrimp-data/explore/pi/c12-3/<name>/ (run.json, records.jsonl, msgs/).
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pi_harness as ph  # noqa: E402

PROBLEM_SEED = 12301
BOOT_SEED = 12303
N_DEV, N_TEST = 16, 64
N_PAIRS = 8
TOP_N = 15
TEMPERATURE = 0.6
COUPLING = 0.25
STRONG = 0.5
N_STEPS = 48
READOUTS = (24, 48)
PRIMARY_READOUT = 48
SPARSE_EVERY = 4
ANSWER_TOKENS = 8
N_BOOT = 10000
DATA = Path("/data2/neural-shrimp-data/explore/pi/c12-3")
PROBLEMS_FILE = ROOT / "docs" / "pi" / "c12-3-problems.json"
START_WORDS = (" Let", " The", " First", " Check", " Looking", " So", " We", " Based")
PHASE1 = ("independent", "live", "sparse", "live50")
PHASE2 = ("replay", "unrelated", "selfreplay", "replay50")
LIVE_OF = {"replay": "live", "unrelated": "live", "selfreplay": "live", "replay50": "live50"}
ARMS = PHASE1 + PHASE2
QUESTION = "What is {a} times {b}?"


# ----------------------------------------------------------------------------------------------- task

def make_problems(seed=PROBLEM_SEED, n_dev=N_DEV, n_test=N_TEST, n_smoke=2):
    rng = np.random.default_rng(seed)
    pool = [x for x in range(11, 100) if x % 10]
    out, seen = [], set()
    for split, n in (("dev", n_dev), ("test", n_test), ("smoke", n_smoke)):
        j = 0
        while j < n:
            a, b = (int(x) for x in rng.choice(pool, 2, replace=False))
            if frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            out.append({"uid": len(out), "id": f"{split}-{j:02d}", "split": split, "a": a, "b": b,
                        "question": QUESTION.format(a=a, b=b), "gold": str(a * b)})
            j += 1
    return out


def problems_hash(probs):
    return hashlib.sha256(json.dumps(probs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_answer(text):
    """The readout's number (the cue already says 'The answer is'), or None."""
    return ph.extract_answer("The answer is" + text)


# ----------------------------------------------------------------------------------------------- arms

def pair_starts(k):
    return k % N_PAIRS, (k + 1) % N_PAIRS


def coupled(arm, t):
    """Whether soft step t mixes in a message."""
    if arm == "independent" or t < 1:
        return False
    if arm == "sparse":
        return t % SPARSE_EVERY == 0
    return True


def replay_source(arm, k, pos, n_problems):
    """(problem position in split, pair) of the live run whose messages the replayed arms receive."""
    if arm in ("replay", "replay50"):
        return pos, (k + 1) % N_PAIRS
    if arm == "unrelated":
        return (pos + 1) % n_problems, k
    if arm == "selfreplay":
        return pos, k
    raise ValueError(arm)


def phase_rows(phase):
    """Row layout of a phase batch: (arm, pair, agent) with agent 0 = A, 1 = B."""
    arms = PHASE1 if phase == 1 else PHASE2
    return [(arm, k, ag) for arm in arms for k in range(N_PAIRS) for ag in (0, 1)]


def coupling_of(arm):
    return STRONG if arm.endswith("50") else COUPLING


def mix(e, msg, c):
    return (1.0 - c) * e + c * msg


# ----------------------------------------------------------------------------------------------- statistics

def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def same_wrong(x, y, gold):
    return x is not None and y is not None and x == y and x != gold


def problem_metrics(ans, gold):
    """ans: {(pair, agent): answer or None} for one arm and problem. SWA, agreement, accuracy, the chance SWA of cross
    pairs (agent A of pair i with agent B of pair j, i != j, different start tokens), and the detector check (each
    agent paired with itself)."""
    pairs = [(ans[(k, 0)], ans[(k, 1)]) for k in range(N_PAIRS)]
    agents = list(ans.values())
    swa = np.mean([same_wrong(x, y, gold) for x, y in pairs])
    agree = np.mean([x is not None and x == y for x, y in pairs])
    acc = np.mean([a == gold for a in agents])
    cross = [same_wrong(ans[(i, 0)], ans[(j, 1)], gold) for i in range(N_PAIRS) for j in range(N_PAIRS)
             if i != j and pair_starts(i)[0] != pair_starts(j)[1]]
    det = np.mean([same_wrong(a, a, gold) for a in agents])
    err_parsed = np.mean([a is not None and a != gold for a in agents])
    return {"swa": float(swa), "agree": float(agree), "acc": float(acc), "chance_swa": float(np.mean(cross)),
            "unparsed": float(np.mean([a is None for a in agents])), "detector_swa": float(det),
            "detector_expected": float(err_parsed), "both_correct": float(np.mean([x == gold and y == gold
                                                                                   for x, y in pairs]))}


def classify(primary, live_minus_replay, threshold=0.10):
    """Run-card decision for 3A: reciprocity is implicated if live exceeds independent by >= threshold with the interval
    above zero and live exceeds replay with the interval above zero."""
    d, ci = primary["mean"], primary["ci"]
    r_ci = live_minus_replay["ci"]
    if ci[0] > 0 and d >= threshold and r_ci[0] > 0:
        return "excess wrong agreement attributable to reciprocity (live > independent by >= 10 points and > replay)"
    if ci[0] > 0 and r_ci[0] <= 0:
        return "excess wrong agreement, not distinguished from message content (live ~ replay)"
    if ci[0] > 0:
        return "excess wrong agreement below the 10-point expansion threshold"
    return "no resolved excess wrong agreement (interval includes zero or is negative)"


def choose_model(results, lo=0.2, hi=0.8, min_parsed=0.9):
    """Development rule: among models whose independent-arm accuracy lies in [lo, hi] with at least min_parsed parsed
    answers, the one with accuracy closest to 0.5 (tie: the smaller model, listed first). None qualifies: the model
    closest to 0.5, flagged."""
    ok = [(abs(r["acc"] - 0.5), i, r) for i, r in enumerate(results)
          if lo <= r["acc"] <= hi and 1 - r["unparsed"] >= min_parsed]
    if ok:
        return {"model": min(ok, key=lambda x: (x[0], x[1]))[2]["model"], "qualified": True}
    best = min(enumerate(results), key=lambda x: (abs(x[1]["acc"] - 0.5), x[0]))[1]
    return {"model": best["model"], "qualified": False}


# ----------------------------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, model_key, threads=6, top_n=TOP_N, T=TEMPERATURE):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        torch.set_num_threads(threads)
        self.torch, self.DynamicCache = torch, DynamicCache
        self.path = ph.snapshot(ph.MODELS[model_key])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.E = self.model.get_input_embeddings().weight
        self.top_n, self.T = top_n, T
        self.cue = self.tok(ph.CUE, add_special_tokens=False)["input_ids"]
        self.start_ids = []
        for w in START_WORDS:
            ids = self.tok(w, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise SystemExit(f"start word {w!r} is not one token: {ids}")
            self.start_ids.append(ids[0])
        self.special_from = min(self.tok.added_tokens_decoder)

    def prompt_ids(self, prob):
        return ph.prompt_ids(self.tok, prob["question"])

    def concept(self, z):
        torch = self.torch
        z = z.clone()
        z[:, self.special_from:] = -float("inf")
        p = torch.softmax(z / self.T, dim=-1)
        w, idx = torch.topk(p, self.top_n, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        return (w[..., None] * self.E[idx].float()).sum(1)

    def readout(self, c, B):
        torch = self.torch
        L0 = c.get_seq_length()
        o = self.model(input_ids=torch.tensor([self.cue] * B), past_key_values=c, use_cache=True, logits_to_keep=1)
        z = o.logits[:, -1, :].float()
        toks = []
        for j in range(ANSWER_TOKENS):
            nxt = z.argmax(-1)
            toks.append(nxt)
            if j + 1 < ANSWER_TOKENS:
                z = self.model(input_ids=nxt[:, None], past_key_values=c, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].float()
        if c.get_seq_length() > L0:
            c.crop(-(c.get_seq_length() - L0))
        rows = torch.stack(toks, 1).tolist()
        return [self.tok.decode(t, skip_special_tokens=True) for t in rows]

    def run_phase(self, prob, phase, msgs=None, n_steps=N_STEPS, readouts=READOUTS):
        """One batch for one problem. msgs (phase 2): {arm: tensor (16, n_steps, d)} of received messages per row of
        that arm (row order pair-major, agent A then B). Returns texts per readout, live messages (phase 1), top-1 per
        step and the pair distances per step."""
        torch = self.torch
        layout = phase_rows(phase)
        B = len(layout)
        ids = self.prompt_ids(prob)
        with torch.no_grad():
            o = self.model(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
            kv = [(layer.keys.expand(B, -1, -1, -1).contiguous(), layer.values.expand(B, -1, -1, -1).contiguous())
                  for layer in o.past_key_values.layers]
            c = self.DynamicCache(ddp_cache_data=kv)
            z = o.logits[:, -1, :].float().expand(B, -1).contiguous()
            start = torch.tensor([self.start_ids[pair_starts(k)[ag]] for _, k, ag in layout])
            partner = torch.tensor([layout.index((arm, k, 1 - ag)) for arm, k, ag in layout])
            arm_rows = {arm: [i for i, r in enumerate(layout) if r[0] == arm] for arm in dict.fromkeys(r[0] for r in layout)}
            out = {"texts": {}, "top1": [], "dist": [], "live_msgs": None}
            live_e = {"live": [], "live50": []}
            for t in range(n_steps):
                e = self.concept(z)
                out["top1"].append(z.argmax(-1).tolist())
                if t == 0:
                    e = self.E[start].float()
                d = (e - e[partner]).norm(dim=-1) / e[partner].norm(dim=-1)
                out["dist"].append(d.tolist())
                x = e.clone()
                for arm, rws in arm_rows.items():
                    if not coupled(arm, t):
                        continue
                    ri = torch.tensor(rws)
                    if phase == 1:
                        m = e[partner[ri]]
                    else:
                        m = msgs[arm][:, t, :]
                    x[ri] = mix(e[ri], m, coupling_of(arm))
                if phase == 1:
                    for arm in live_e:
                        live_e[arm].append(e[torch.tensor(arm_rows[arm])].clone())
                o = self.model(inputs_embeds=x[:, None, :].to(self.E.dtype), past_key_values=c, use_cache=True)
                z = o.logits[:, -1, :].float()
                if t + 1 in readouts:
                    out["texts"][t + 1] = self.readout(c, B)
            if phase == 1:
                # {arm: (16, n_steps, d)}: the concept token each live row sent
                out["live_msgs"] = {arm: torch.stack(v, 1) for arm, v in live_e.items()}
        return layout, out


def received(live_msgs, k_src):
    """Messages that rows (pair k, agent A then B) receive when replaying the live run's pair k_src: agent A receives
    what B sent, agent B what A sent."""
    import torch
    idx = []
    for k in k_src:
        idx += [2 * k + 1, 2 * k]
    return live_msgs[torch.tensor(idx)]


# ----------------------------------------------------------------------------------------------- subcommands

def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance(args, extra=None):
    import torch
    import transformers
    info = {"args": vars(args), "argv": sys.argv, "commit": git("rev-parse", "HEAD").strip(),
            "dirty_files": [ln for ln in git("status", "--porcelain").splitlines()
                            if "c12_coupled" in ln or "pi_harness" in ln or "c12-3" in ln],
            "tool_sha256": sha256_file(__file__), "pi_harness_sha256": sha256_file(ph.__file__),
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    info.update(extra or {})
    return info


def load_problems():
    probs = make_problems()
    if PROBLEMS_FILE.exists():
        stored = json.loads(PROBLEMS_FILE.read_text())["problems"]
        if problems_hash(stored) != problems_hash(probs):
            raise SystemExit(f"{PROBLEMS_FILE}: problem list differs from make_problems()")
    return probs


def cmd_problems(args):
    from transformers import AutoTokenizer
    probs = make_problems()
    lens = {}
    for m in ("qwen0.5b", "qwen1.5b"):
        tok = AutoTokenizer.from_pretrained(ph.snapshot(ph.MODELS[m]))
        lens[m] = sorted({len(ph.prompt_ids(tok, p["question"])) for p in probs})
        if len(lens[m]) != 1:
            raise SystemExit(f"{m}: prompt lengths differ {lens[m]}")
    PROBLEMS_FILE.write_text(json.dumps({"seed": PROBLEM_SEED, "sha256": problems_hash(probs), "prompt_lengths": lens,
                                         "problems": probs}, indent=1))
    print(json.dumps({"sha256": problems_hash(probs), "prompt_lengths": lens}))


def cmd_run(args):
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    probs = [p for p in load_problems() if p["split"] == args.split][:args.limit or None]
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "msgs").mkdir(exist_ok=True)
    runf = out / "run.json"
    hist = json.loads(runf.read_text())["history"] if runf.exists() else []
    if hist and hist[0]["args"]["model"] != args.model:
        raise SystemExit("resume with a different model")
    hist.append(provenance(args, {"problems_sha256": problems_hash(load_problems()), "settings": settings()}))
    runf.write_text(json.dumps({"history": hist}, indent=1))
    recf = out / "records.jsonl"
    done = set()
    if recf.exists():
        for line in open(recf):
            r = json.loads(line)
            done.add((r["uid"], r["phase"]))
    eng = Engine(args.model, args.threads)
    for phase in args.phases:
        for pos, prob in enumerate(probs):
            if (prob["uid"], phase) in done:
                continue
            t0 = time.time()
            msgs = None
            if phase == 2:
                srcs = {}
                for arm in PHASE2:
                    src = [replay_source(arm, k, pos, len(probs)) for k in range(N_PAIRS)]
                    if len({s[0] for s in src}) != 1:
                        raise SystemExit("one source problem per arm and problem")
                    lm = torch.load(out / "msgs" / f"{probs[src[0][0]]['uid']:03d}.pt")
                    srcs[arm] = received(lm[LIVE_OF[arm]], [s[1] for s in src])
                msgs = srcs
            layout, res = eng.run_phase(prob, phase, msgs, args.steps)
            if phase == 1:
                torch.save(res["live_msgs"], out / "msgs" / f"{prob['uid']:03d}.pt")
            recs = []
            for i, (arm, k, ag) in enumerate(layout):
                texts = {str(S): res["texts"][S][i] for S in res["texts"]}
                recs.append({"uid": prob["uid"], "id": prob["id"], "phase": phase, "gold": prob["gold"], "arm": arm,
                             "pair": k, "agent": "AB"[ag], "start": START_WORDS[pair_starts(k)[ag]],
                             "texts": texts, "answers": {S: parse_answer(x) for S, x in texts.items()},
                             "top1": [s[i] for s in res["top1"]],
                             "dist": [float(f"{s[i]:.4g}") for s in res["dist"]]})
            with open(recf, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(json.dumps({"uid": prob["uid"], "phase": phase, "seconds": round(time.time() - t0, 1)}), flush=True)


def settings():
    return {"coupling": COUPLING, "strong_coupling": STRONG, "steps": N_STEPS, "readouts": READOUTS, "primary_readout": PRIMARY_READOUT,
            "sparse_every": SPARSE_EVERY, "top_n": TOP_N, "temperature": TEMPERATURE, "answer_tokens": ANSWER_TOKENS,
            "start_words": START_WORDS, "pairs": N_PAIRS, "mask_special": True, "dtype": "float32",
            "problem_seed": PROBLEM_SEED, "boot_seed": BOOT_SEED, "n_boot": N_BOOT}


def load_records(run):
    return [json.loads(line) for line in open(DATA / run / "records.jsonl")]


def summarize(recs, S=PRIMARY_READOUT):
    """Per-arm problem metrics, paired contrasts and synchronization diagnostics at readout S."""
    S = str(S)
    by = {}
    for r in recs:
        by.setdefault((r["uid"], r["arm"]), {})[(r["pair"], "AB".index(r["agent"]))] = r
    uids = sorted({r["uid"] for r in recs})
    gold = {r["uid"]: r["gold"] for r in recs}
    arms = [a for a in ARMS if all((u, a) in by for u in uids)]
    per = {a: [problem_metrics({key: r["answers"][S] for key, r in by[(u, a)].items()}, gold[u]) for u in uids]
           for a in arms}
    out = {"readout": int(S), "n_problems": len(uids), "arms": {}, "contrasts": {}, "gates": {}, "sync": {}}
    for a in arms:
        out["arms"][a] = {m: float(np.mean([x[m] for x in per[a]])) for m in per[a][0]}
        out["arms"][a]["excess_over_chance"] = {"mean": float(np.mean([x["swa"] - x["chance_swa"] for x in per[a]])),
                                                "ci": bootstrap_mean_ci([x["swa"] - x["chance_swa"] for x in per[a]])}
    for name, a, b, m in (("live_minus_independent_swa", "live", "independent", "swa"),
                          ("live_minus_replay_swa", "live", "replay", "swa"),
                          ("live50_minus_independent_swa", "live50", "independent", "swa"),
                          ("live50_minus_replay50_swa", "live50", "replay50", "swa"),
                          ("replay50_minus_independent_swa", "replay50", "independent", "swa"),
                          ("live50_minus_independent_acc", "live50", "independent", "acc"),
                          ("live50_minus_replay50_acc", "live50", "replay50", "acc"),
                          ("live50_minus_independent_agree", "live50", "independent", "agree"),
                          ("sparse_minus_live_swa", "sparse", "live", "swa"),
                          ("sparse_minus_independent_swa", "sparse", "independent", "swa"),
                          ("replay_minus_independent_swa", "replay", "independent", "swa"),
                          ("unrelated_minus_independent_swa", "unrelated", "independent", "swa"),
                          ("live_minus_independent_acc", "live", "independent", "acc"),
                          ("live_minus_replay_acc", "live", "replay", "acc"),
                          ("sparse_minus_live_acc", "sparse", "live", "acc"),
                          ("sparse_minus_independent_acc", "sparse", "independent", "acc"),
                          ("replay_minus_independent_acc", "replay", "independent", "acc"),
                          ("unrelated_minus_independent_acc", "unrelated", "independent", "acc"),
                          ("live_minus_independent_agree", "live", "independent", "agree"),
                          ("sparse_minus_live_agree", "sparse", "live", "agree")):
        if a in per and b in per:
            d = [x[m] - y[m] for x, y in zip(per[a], per[b])]
            out["contrasts"][name] = {"mean": float(np.mean(d)), "ci": bootstrap_mean_ci(d)}
    if "live" in arms and "selfreplay" in arms:
        same_txt = [by[(u, "selfreplay")][key]["texts"][S] == by[(u, "live")][key]["texts"][S]
                    for u in uids for key in by[(u, "live")]]
        same_top = [by[(u, "selfreplay")][key]["top1"] == by[(u, "live")][key]["top1"]
                    for u in uids for key in by[(u, "live")]]
        out["gates"]["selfreplay_text_identical"] = float(np.mean(same_txt))
        out["gates"]["selfreplay_top1_identical"] = float(np.mean(same_top))
    out["gates"]["detector_ok"] = all(abs(x["detector_swa"] - x["detector_expected"]) < 1e-12
                                      for a in arms for x in per[a])
    out["gates"]["unparsed_max_arm"] = max(out["arms"][a]["unparsed"] for a in arms)
    for a in arms:
        dl, eq = [], []
        for u in uids:
            for k in range(N_PAIRS):
                ra, rb = by[(u, a)][(k, 0)], by[(u, a)][(k, 1)]
                dl.append(ra["dist"])
                eq.append([x == y for x, y in zip(ra["top1"], rb["top1"])])
        dl, eq = np.array(dl), np.array(eq, dtype=float)
        out["sync"][a] = {"median_dist_last": float(np.median(dl[:, -1])),
                          "median_dist_steps_16_48": float(np.median(dl[:, 16:].mean(1))),
                          "top1_equal_share_steps_16_48": float(eq[:, 16:].mean())}
    if "live" in per and "independent" in per and "replay" in per:
        out["decision_3a"] = classify(out["contrasts"]["live_minus_independent_swa"],
                                      out["contrasts"]["live_minus_replay_swa"])
    if "sparse" in per:
        s_swa = out["contrasts"]["sparse_minus_live_swa"]
        s_acc = out["contrasts"]["sparse_minus_live_acc"]
        out["decision_3b"] = ("sparse exchange reduces wrong agreement without reducing accuracy"
                              if s_swa["ci"][1] < 0 and s_acc["ci"][0] > -0.05 else
                              "no resolved reduction of wrong agreement at preserved accuracy")
    out["per_problem"] = {a: per[a] for a in arms}
    return out


def cmd_analyze(args):
    recs = load_records(args.run)
    res = {"run": str(DATA / args.run), "by_readout": {S: summarize(recs, S) for S in READOUTS}}
    p = Path(args.json) if args.json else DATA / args.run / "analysis.json"
    p.write_text(json.dumps(res, indent=1))
    s = res["by_readout"][PRIMARY_READOUT]
    print(json.dumps({k: s[k] for k in ("arms", "contrasts", "gates", "sync")}, indent=1))
    print(s.get("decision_3a"), "|", s.get("decision_3b"))


def cmd_choose(args):
    results = []
    for run in args.runs:
        recs = load_records(run)
        s = summarize(recs)
        model = json.loads((DATA / run / "run.json").read_text())["history"][0]["args"]["model"]
        results.append({"model": model, "run": run, **{k: s["arms"]["independent"][k] for k in ("acc", "unparsed")}})
    ch = choose_model(results)
    ch.update({"rule": choose_model.__doc__, "results": results,
               "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    Path(args.out).write_text(json.dumps(ch, indent=1))
    print(json.dumps(ch, indent=1))


def cmd_receipt(args):
    tpl = json.loads(Path(args.template).read_text()) if args.template else {}
    rec = {"experiment": "C12-3 coupled agents (exploratory)", "settings": settings(),
           "problems_sha256": problems_hash(load_problems()), **tpl, "runs": {}}
    for run in args.runs:
        d = DATA / run
        files = {f.name: sha256_file(f) for f in sorted(d.glob("*")) if f.is_file()}
        msgs = sorted((d / "msgs").glob("*.pt")) if (d / "msgs").exists() else []
        if msgs:
            files["msgs/*.pt (sha256 of the sorted per-file hashes)"] = hashlib.sha256(
                "".join(sha256_file(f) for f in msgs).encode()).hexdigest()
        hist = json.loads((d / "run.json").read_text())["history"]
        entry = {"path": str(d), "files_sha256": files, "commits": [h["commit"] for h in hist],
                 "model": hist[0]["args"]["model"], "split": hist[0]["args"]["split"]}
        if (d / "analysis.json").exists():
            a = json.loads((d / "analysis.json").read_text())["by_readout"]
            entry["results"] = {S: {k: v for k, v in x.items() if k != "per_problem"} for S, x in a.items()}
        rec["runs"][run] = entry
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("problems")
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=("qwen0.5b", "qwen1.5b"))
    r.add_argument("--split", required=True, choices=("dev", "test", "smoke"))
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--steps", type=int, default=N_STEPS)
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--phases", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    a = sub.add_parser("analyze")
    a.add_argument("run")
    a.add_argument("--json")
    c = sub.add_parser("choose")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", default=str(ROOT / "docs" / "pi" / "c12-3-choices.json"))
    rc = sub.add_parser("receipt")
    rc.add_argument("runs", nargs="+")
    rc.add_argument("--template")
    rc.add_argument("--out", default=str(ROOT / "docs" / "pi" / "c12-3-summary.json"))
    args = ap.parse_args()
    {"problems": cmd_problems, "run": cmd_run, "analyze": cmd_analyze, "choose": cmd_choose,
     "receipt": cmd_receipt}[args.cmd](args)


if __name__ == "__main__":
    main()
