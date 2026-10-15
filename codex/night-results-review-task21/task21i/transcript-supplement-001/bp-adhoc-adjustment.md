# BP ad-hoc control analysis of 15 October 2026, 05:52 UTC: the exact record

This is the existing record of the ad-hoc confound analysis reported in the research log at 05:53 UTC ("BP result",
bullet "Linear model acc ~ chaotic + generation entropy + task + greedy"). It was an inline command run once in
the producer's session, not a committed tool. Both the command and its output are copied verbatim from the session
transcript (tool call at 2026-10-15T05:52:43Z, result at 05:52:44Z). Nothing was rerun to make this file.

What the command does, read from the code below:
- **Rows.** Prompts are taken in the iteration order of the `labels` dict. That dict is filled from label shards
  q7-label-0 to q7-label-5 in file-line order, and a later shard overwrites an earlier one for the same prompt.
- **Exclusions.** A prompt is dropped at a temperature if:
  - its loop class at that temperature is neither chaotic (class containing "chaos" but not "irregular", label 1)
    nor "periodic" (label 0); or
  - it has no sampled generations at that temperature.
- **Covariates.**
  - Task: an indicator for NQ, with arithmetic as reference.
  - Greedy: `generations.json["greedy"][prompt]["correct"]`.
  - Generation entropy: the mean, over that prompt's sampled generations at that temperature, of each sample's
    stored `entropy` field.
  - Accuracy: the mean of the samples' `correct`.
- **Model.** Ordinary least squares, `acc ~ 1 + chaotic + gen_entropy + nq + greedy`, by `np.linalg.lstsq`.
- **Permutation test.** One generator, `np.random.default_rng(1)`, is used sequentially for the three temperatures
  (0.25, then 0.32, then 0.40), with 5000 permutations of the chaotic column each. Two-sided
  p = mean(|beta_perm| >= |beta|).
- **Stratification.** Within greedy-correct prompts, by quartiles of the generation entropy, using bins
  [q_k, q_{k+1}) with open outer ends. Empty contrasts are skipped (at 0.25, bin 1 has no contrast). The weighted
  difference uses weights min(n_chaotic, n_periodic) per bin.

The generation entropy is measured during sampling, after the outcome begins, so this adjustment probably
over-corrects (as the log entry says).

## Command (verbatim)

```bash
uv run python - <<'EOF'
import json, numpy as np
D = "/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/bp/"
items = {it["prompt"]: it for it in json.load(open("docs/bp-tasks.json"))["items"]}
labels = {}
for k in range(6):
    for l in open(D + f"q7-label-{k}/labels.jsonl"):
        r = json.loads(l); labels[r["prompt"]] = r
gen = json.load(open(D + "q7-gen/generations.json"))
rng = np.random.default_rng(1)
for lt in (0.25, 0.32, 0.4):
    rows = []
    for p, r in labels.items():
        c = next(c for c in r["cells"] if abs(c["log10_temperature"] - lt) < 1e-9)
        lab = 1 if ("chaos" in c["class"] and "irregular" not in c["class"]) else (0 if c["class"] == "periodic" else None)
        s = gen["sampled"][f"{lt}"].get(str(p))
        if lab is None or not s:
            continue
        rows.append({"p": p, "task": items[p]["task"], "chaotic": lab, "greedy": gen["greedy"][str(p)]["correct"],
                     "acc": np.mean([x["correct"] for x in s]), "gen_entropy": np.mean([x["entropy"] for x in s]),
                     "loop_entropy": c.get("entropy_mean") or np.nan})
    print(f"== log10 T {lt}: {len(rows)} prompts")
    for task in ("arithmetic", "nq"):
        rs = [r for r in rows if r["task"] == task]
        ch = [r["acc"] for r in rs if r["chaotic"]]; pe = [r["acc"] for r in rs if not r["chaotic"]]
        print("  %-10s chaotic %3d acc %.3f | periodic %3d acc %.3f" % (task, len(ch), np.mean(ch) if ch else np.nan, len(pe), np.mean(pe) if pe else np.nan))
    # stratify by generation entropy quartiles (within greedy-correct prompts)
    rs = [r for r in rows if r["greedy"]]
    e = np.array([r["gen_entropy"] for r in rs]); q = np.quantile(e, [0.25, 0.5, 0.75])
    diffs, w = [], []
    for b in range(4):
        lo = -np.inf if b == 0 else q[b - 1]; hi = np.inf if b == 3 else q[b]
        bin_ = [r for r in rs if lo <= r["gen_entropy"] < hi]
        ch = [r["acc"] for r in bin_ if r["chaotic"]]; pe = [r["acc"] for r in bin_ if not r["chaotic"]]
        if ch and pe:
            diffs.append(np.mean(ch) - np.mean(pe)); w.append(min(len(ch), len(pe)))
            print("  entropy bin %d: chaotic %2d acc %.3f, periodic %2d acc %.3f, diff %+.3f (mean gen entropy %.2f)" % (b, len(ch), np.mean(ch), len(pe), np.mean(pe), diffs[-1], np.mean([r['gen_entropy'] for r in bin_])))
    if diffs:
        print("  entropy-stratified difference (weighted): %+.3f" % (np.average(diffs, weights=w)))
    # linear model: acc ~ chaotic + gen_entropy + task + greedy, with a permutation test on the chaotic label
    X = np.array([[1, r["chaotic"], r["gen_entropy"], r["task"] == "nq", r["greedy"]] for r in rows], dtype=float)
    y = np.array([r["acc"] for r in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    perm = []
    for _ in range(5000):
        Xp = X.copy(); Xp[:, 1] = rng.permutation(Xp[:, 1])
        perm.append(np.linalg.lstsq(Xp, y, rcond=None)[0][1])
    print("  linear model: chaotic coefficient %+.3f (permutation p %.4f); entropy coef %+.3f" % (beta[1], np.mean(np.abs(perm) >= abs(beta[1])), beta[2]))
EOF
```

## Output (verbatim)

```
== log10 T 0.25: 199 prompts
  arithmetic chaotic   7 acc 0.843 | periodic  92 acc 0.914
  nq         chaotic  31 acc 0.100 | periodic  69 acc 0.206
  entropy bin 0: chaotic  1 acc 1.000, periodic 29 acc 1.000, diff +0.000 (mean gen entropy 0.00)
  entropy bin 2: chaotic  6 acc 0.817, periodic 24 acc 0.967, diff -0.150 (mean gen entropy 0.19)
  entropy bin 3: chaotic  7 acc 0.443, periodic 23 acc 0.652, diff -0.209 (mean gen entropy 1.37)
  entropy-stratified difference (weighted): -0.169
  linear model: chaotic coefficient -0.037 (permutation p 0.2038); entropy coef -0.067
== log10 T 0.32: 197 prompts
  arithmetic chaotic  22 acc 0.745 | periodic  75 acc 0.879
  nq         chaotic  46 acc 0.100 | periodic  54 acc 0.131
  entropy bin 0: chaotic  2 acc 1.000, periodic 27 acc 0.993, diff +0.007 (mean gen entropy 0.19)
  entropy bin 1: chaotic  5 acc 0.840, periodic 24 acc 0.987, diff -0.147 (mean gen entropy 0.45)
  entropy bin 2: chaotic 12 acc 0.833, periodic 17 acc 0.859, diff -0.025 (mean gen entropy 0.94)
  entropy bin 3: chaotic 14 acc 0.286, periodic 16 acc 0.394, diff -0.108 (mean gen entropy 2.85)
  entropy-stratified difference (weighted): -0.077
  linear model: chaotic coefficient -0.046 (permutation p 0.0690); entropy coef -0.177
== log10 T 0.4: 199 prompts
  arithmetic chaotic  25 acc 0.448 | periodic  75 acc 0.772
  nq         chaotic  65 acc 0.077 | periodic  34 acc 0.021
  entropy bin 0: chaotic  4 acc 0.950, periodic 26 acc 0.923, diff +0.027 (mean gen entropy 1.04)
  entropy bin 1: chaotic  4 acc 0.875, periodic 25 acc 0.904, diff -0.029 (mean gen entropy 1.44)
  entropy bin 2: chaotic 13 acc 0.277, periodic 17 acc 0.594, diff -0.317 (mean gen entropy 2.04)
  entropy bin 3: chaotic 19 acc 0.221, periodic 11 acc 0.145, diff +0.076 (mean gen entropy 3.61)
  entropy-stratified difference (weighted): -0.103
  linear model: chaotic coefficient -0.045 (permutation p 0.0862); entropy coef -0.419
```

## The original analyze invocations (verbatim, from the same transcript)

behavior_zones.py, 2026-10-15T04:59:29Z (run-list order pilot-0, pilot-1, pilot-2):

```bash
uv run python tools/behavior_zones.py analyze --runs llm-soft-loop/pod64/bz/pilot-0 llm-soft-loop/pod64/bz/pilot-1 llm-soft-loop/pod64/bz/pilot-2 --out-json /data2/neural-shrimp-data/scratch/bz-analysis.json
```

behavior_prompts.py, 2026-10-15T05:51:41Z (label-run order: shards 0 to 5 for 7B; 0 then 1 for 0.5B):

```bash
uv run python tools/behavior_prompts.py analyze --label-runs $(for k in 0 1 2 3 4 5; do echo llm-soft-loop/pod64/bp/q7-label-$k; done) --gen-run llm-soft-loop/pod64/bp/q7-gen --out-json /data2/neural-shrimp-data/scratch/bp-q7.json
uv run python tools/behavior_prompts.py analyze --label-runs llm-soft-loop/pod64/bp/q05-label-0 llm-soft-loop/pod64/bp/q05-label-1 --gen-run llm-soft-loop/pod64/bp/q05-gen --out-json /data2/neural-shrimp-data/scratch/bp-q05.json
```
