# Handoff to Codex: parallel work on the LLM-dynamics study (12 October 2026)

Written by Claude (Claude Code) at the human's request. Claude continues the main research line in parallel.
Read this file completely before doing anything, then read `AGENTS.md`, `Skills.md` and the comms protocol
below. The human decides which of the tasks in section 4 you take; ask in the comms file if unsure.

## 1. Boundaries (non-negotiable)

Claude is running GPU jobs, RunPod pods, CPU analyses and edits in this repository right now. Your work must
not interfere with any of it. If a rule below blocks you, stop and write to the comms file; do not work
around it.

**Git**
- Do **not** work in `/data2/neural-shrimp` (Claude's working tree, uncommitted files and running jobs live
  there). Make your own worktree, e.g. `git -C /data2/neural-shrimp worktree add /data2/neural-shrimp-codex -b
  codex/<task>` or a fresh clone, and work only there.
- One short-lived branch per task, named `codex/<task>`. To deliver: `git fetch`, rebase onto the latest
  `origin/main`, run the tests you touched (check the exit status: `set -o pipefail`), fast-forward `main`
  and push. Never force-push, never rewrite published history, never push `main` with a merge that changes
  Claude's files.
- Commit messages: no assistant co-author trailers (AGENTS.md).

**Files you must not edit** (Claude owns them; read them freely):
- `tools/explore_llm_soft_loop.py`, `tools/bifurcation_curves.py`, `tools/explore_llm_attractors.py`,
  `tools/explore_llm_phase_map.py`, `tools/explore_llm_msm.py`, `tools/explore_latent_loop.py`,
  `tools/research_log.py`, `tools/runpod_*.py`, `tools/runpod_*.sh`, `tools/gpu_queue.sh`
- `tests/test_bifurcation_curves.py`, `tests/test_llm_*.py`, `tests/test_latent_loop.py`
- `docs/llm-attractors-pathway.md`, `docs/consults/**`, `docs/handoff-codex-2026-10-12.md`,
  `comms/claude-to-codex.md`, `AGENTS.md`, `checkpoint.md`, `cost-ledger.md` (private, untracked)
- Anything under `../neural-shrimp-data/` except your own output folder (below).

**Files you own**: new files you create under `codex/` (code, notes), `tests/test_codex_*.py`,
`certificates/llm-*` (task 1 only), `paper/llm-references.bib` and `docs/related-work-draft.md` (task 3
only), `comms/codex-to-claude.md`. Raw outputs go to `../neural-shrimp-data/codex/<task>/` (create it).
If you need a change in a Claude-owned file (a bug, a missing option), describe it in the comms file and
Claude will make it.

**Research log**: `docs/research-log.md` is append-only and shared. Append only through
`tools/research_log.py run|note` from your worktree (it records your branch's HEAD); when rebasing, keep both
sides of any conflict in that file, in time order. Every run gets a run card first (AGENTS.md).

**Compute**
- The local GPU (RTX 2000 Ada) belongs to Claude's queue (`../neural-shrimp-data/explore/gpu-queue/`). Do not
  start GPU jobs on it unless Claude agrees in the comms file.
- CPU: at most 12 threads in total for your processes (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, torch threads),
  run them with `nice -n 10`. Other people's processes on this machine (ComfyUI/otherroom, praxvm, the
  OTHER_ROOM guard.py) must never be touched.
- RunPod: no pods without the human's explicit approval for your task and a costed plan in the private
  ledger. If approved, name pods `neural-shrimp-codex-...`, use `tools/runpod_pod.py` (ledger, verified
  termination) and never touch, inspect or stop any other pod: not Claude's `neural-shrimp-*` pods and not the
  other projects' pods on the account. Spend only in the private `cost-ledger.md`; no dollar amounts in
  committed files.
- Never kill a process you did not start. Beware `pkill -f` matching your own shell (use `[x]yz` patterns).
- Paid API calls (OpenAI Pro consults) only with the human's explicit approval for that call.
- Never print or copy `.env`. Seeds 29000-29999 are reserved; CIFAR subset B is sealed.

**Evidence wording**: follow the evidence ladder in
`../agent-skill-documents/CERTIFIED_COMPUTATION_SKILLS.md`. "Certified", "verified" or "proved" only for
interval or ball-arithmetic proofs (rung 3); otherwise "numerically validated" or "cross-checked".

## 2. Communication (comms protocol)

Two append-only files in the repository, one writer each, so they never conflict:
- `comms/claude-to-codex.md`: Claude writes, Codex reads.
- `comms/codex-to-claude.md`: Codex writes, Claude reads.

Entry format (append at the end, never edit earlier entries):

```
## 2026-10-12T22:00Z | from codex | <topic> | status: question / info / request / done
<message; for a question say what is blocked and what you will do meanwhile>
```

Commit and push each entry on its own (`git add comms/codex-to-claude.md && git commit -m "comms: ..." && git
push`, after `git pull --rebase`). Check the other file when you start, before each delivery and at least
every few hours (`git fetch && git show origin/main:comms/claude-to-codex.md | tail -40`). Claude runs a
watcher that notices new commits to `comms/codex-to-claude.md` on `origin/main`. Anything that needs the
human: say so explicitly in the entry ("needs human"). Results go in the research log, not in comms.

## 3. Where things stand (read these)

- `docs/llm-attractors-pathway.md`: the study, sections from "Periodic windows inside soft-token chaos" on.
- `docs/research-log.md`: everything on 12 October (search for "normal form", "passage", "bridge",
  "shrimp candidate", "gratitude 0.02").
- `docs/consults/2026-10-12-llm-dynamics-novelty/adjudication.md` and
  `docs/consults/2026-10-12-deep-research-prior-work/adjudication.md`: novelty, verified references, the
  plan accepted from both reviews.
- Key numbers (Qwen2.5-0.5B-Instruct, soft-token map x -> E^T softmax(z(x)/T), fixed chat prefix, one-token
  window, 'philosophy' steering vector at layer 12 also applied to the prefix,
  `../neural-shrimp-data/explore/llm-phase-map/vectors-l12`): the 'is it' 2-cycle folds at
  log10 T = -0.115, alpha_c = 0.06279827 (float64, multiplier 1 to 3e-13); normal form of the two-step map
  a = -4.834 per unit alpha, b = -25.4285, ab = 122.93 (stable branch: 123.1); controlled passages from
  u_in = -U (U = 0.1/|b|) to +U: 144, 71, 34, 14 two-step iterates at eps = 3e-6, 1e-5, 3e-5, 1e-4.
  Artefacts: `../neural-shrimp-data/explore/llm-soft-loop/{smoke-curve-p2-float64-cpu,normal-form-p2-t-0.115,
  passage-p2-t-0.115}`.
- Precision traps already hit (AGENTS.md gates): Hugging Face RMSNorm and rotary tables compute in float32
  even for float64 models (patched by `keep_float64_norms` in `tools/explore_llm_attractors.py`); float32
  derivatives of the 8-step map are off by 3 %; models load offline from
  `../neural-shrimp-data/models/hf/hub` (snapshots pinned).

## 4. Tasks (the human picks; each is independent of Claude's running work)

**Task 1. Computer-assisted proof of one LLM periodic orbit (rung 3).** Prove existence, local uniqueness
and stability of the period-8 orbit of the float64 soft-token map near log10 T = -0.1205, alpha = 0.0717
(leading multiplier about +0.004, all others about 1e-6). Approach: recompute the orbit in float64 with
today's map (patched norms and rotary), then a Krawczyk or interval-Newton test on the 8-step map (or a
multiple-shooting system) with the map and its Jacobian enclosed in ball arithmetic over a small box, plus a
spectral-radius bound for the monodromy. This needs your own interval implementation of the full forward
pass (embedding mixture, RMSNorm, rotary, attention with the fixed prefix cache, SwiGLU MLP, logits,
softmax at temperature T, mixture with E). Deliverables: `codex/certify/`, `certificates/llm-p8-*.json`
(hashes of inputs, radii, inclusion margins), a short note in the research log. A failed inclusion is a setup
result (AGENTS.md), not evidence against the orbit.

**Task 2. Independent cross-check (rung 2).** Re-implement, without importing any repository tool, the
soft-token map, Newton for the 2-cycle and the P8 orbit, multipliers, the 2-cycle fold, its normal form
(a, b, ab) and the controlled passage counts above; compare with Claude's numbers and report every
discrepancy with its size. Deliverables: `codex/crosscheck/`, a research-log note.

**Task 3. Literature.** Systematic search for prior work on: language-model inference as a dynamical system;
expected-embedding or continuous feedback decoding; chaos, intermittency, period adding, shrimps or
bifurcation continuation in neural or transformer systems; Markov state models of LLM dialogue; Hopf or
Neimark-Sacker bifurcations in recurrent-depth models. Start from the 18 verified references in the
deep-research adjudication. Verify every reference at its primary record (arXiv API, Crossref, publisher).
Deliverables: `docs/related-work-draft.md`, `paper/llm-references.bib`.

**Task 4. Frozen confirmatory design for the self-dialogue study.** Write a design (held-out openers,
arms gratitude alpha = 0, 0.02, 0.04, 32 dialogues each, 40 turns, frozen state definitions from development
data, finite-horizon occupancy and censored dwell times, dialogue-level bootstrap, Chapman-Kolmogorov check)
and the analysis code, push the design before any outcome (AGENTS.md: confirmatory tests need a frozen
design), then run it on CPU within the thread limit. Deliverables: `codex/msm-confirm/`, design and results
notes.
