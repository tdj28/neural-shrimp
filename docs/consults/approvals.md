# Human approvals for paid Pro consults

Each paid request needs its own explicit human approval (canonical integrity
skill, section 7A). Spend is recorded only in the gitignored `cost-ledger.md`
(Skills.md, human directive of 7 October 2026).

## 2026-10-07: two consults

- **Approval (human, in session, 7 October 2026):** go-ahead for the Pro
  review with a human-set budget (amount in the private cost ledger), and "please also consult gpt pro on your research
  roadmap in including the runpod plan, and I'll approve runpod once we get
  that cleared."
- **Scope:** (1) research-strategy consult on the three mechanism questions
  (`docs/consults/2026-10-07-mechanism-strategy/brief.md`); (2) director-level
  review of the research roadmap including the RunPod plan. One generating
  request each; a retry or second pass needs new approval.
- **Model and mode:** `gpt-6-astra`, Pro mode, rates verified against the
  official pricing page on 2026-10-07 (rates in the private cost ledger). Client: `../agent-skill-documents/scripts/review_experiment_plan.py`
  at the commit recorded in `docs/skill-provenance.json`.
- **Major milestone:** mechanism pivot of Program C (the quasi-periodic route
  to chaos replaces shrimps as the central claim) and the costed plan for
  paid compute.
- **Ceiling:** a human-set total for consults (raised once on 7 October 2026);
  amounts in the private cost ledger; each request also passes the client's
  preflight authorization guard.

| Request | Mode | Usage | Cost |
|---|---|---|---|
| (1) mechanism strategy | Pro, high effort, 25k output | 31,129 in / 24,797 out tokens | see private cost ledger |
| (2) roadmap and RunPod plan | Pro, xhigh effort, 40k output | 31,924 in / 25,276 out tokens | see private cost ledger |

## 2026-10-07: third consult (references)

- **Approval (human, in session, 7 October 2026):** "also ask pro if we are
  missing references, but that can be a third consult if needed i'm good for
  the cost", within the consult ceiling recorded above.
- **Scope:** one `research-note-scientific` review of the working manuscript
  (`paper/gd-route/main.tex` with `references.bib` as context), emphasizing
  reference coverage. The model has no browsing: every suggested reference is
  verified at its primary source (arXiv or Crossref) before use.

| Request | Mode | Usage | Cost |
|---|---|---|---|
| (3) references | Pro, xhigh effort, 34k output | 64,080 in / 32,023 out tokens | see private cost ledger |

## 2026-10-07: RunPod Phase 0 approved

- **Approval (human, in session, 7 October 2026):** "you have go on runpods",
  in reply to the request for a capped Phase 0 within the program frame
  (hard ceiling and check-in amounts in the private cost ledger) recommended
  by the roadmap review.
- **Scope:** Phase 0 only (port the CUDA engine to sm_90, parity check against
  the CPU reference, timing of the real workloads); Phase 1 returns to the
  human with measured costs after the design freeze.
- **Pre-existing account resources (read-only inventory, not task-owned):**
  pods 767stlaueznwa1, hka0rcggikum2v, kacmfssaowcoyl, y4hlqwlhb1nq31 at
  launch-time check; never touched.

## 2026-10-08: RunPod Phase 1 approved

- **Approval (human, in session, 8 October 2026):** "yes i approve" in reply to
  the Phase 1 request (hard cap in the private cost ledger).
- **Scope:** (1) bounded real-data (CIFAR-10) runs on a rented H100, developed on
  subset A, with subset B opened only by a frozen confirmatory test; (2) large
  rotation maps for S1R and for the fresh confirmatory networks. Same safeguards
  as Phase 0: pods named `neural-shrimp-...`, ledger-recorded IDs only,
  terminate and verify after each run, spend only in `cost-ledger.md`, exact GPU
  details in `docs/hardware-ledger.md`.

## 2026-10-08: post-edge strategy milestone

- **Human approval (7 October evening PDT, this session):** "if you get stuck
  or need inspiration, i authorize you to do a consult with gpt astra 6 pro,
  but let's move further along in our findings before you do that"; the
  explicit ceiling is retained in the private cost ledger.
- **Milestone:** two exact-real orbit proofs, a reviewed minimal unequal-rate
  mechanism, controlled warmup with a strong generic baseline, an actual-text
  transformer pilot without periodic recurrence, and CIFAR/S1R obstacles now
  support a concrete choice of the next mechanism-to-utility experiment.
- **Scope:** one `research-strategy` generating request using the compact
  `2026-10-08-post-edge-strategy/brief.md`; no retry or follow-on request.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most40,000
  output tokens. Canonical client input/output reserve multipliers and
  authorization guard apply; costs/rates/cumulative spend are private only.
- **Sources checked before dispatch:** official latest-model guide, Astra
  model page and pricing page, 8 October2026. The standard-context row applies
  to this small packet. This is advisory strategy review, not independent
  scientific validation. Material advice will be locally adjudicated.

## 2026-10-11: window-organization consult

- **Approval (human, in session, 11 October 2026):** "please feel free to use a
  gpt pro consult ... to clarify these things", in reply to the question of how
  the period-18 window is organized relative to the period-2 Neimark-Sacker
  curve; budget in the private cost ledger. This approves sending the compact
  packet below to OpenAI.
- **Milestone:** the window's lower fold is the 2/9 locking edge of the
  invariant circle, and the rotation has a valley below 2/9 between the window
  and the near-curve band joining the 2/9 roots, suggesting a detached
  resonance component.
- **Scope:** one `research-strategy` request using
  `2026-10-11-window-organization/brief.md`, with the theory questions answered
  first; no retry without a new decision recorded here.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000
  output tokens; canonical client reserves and authorization guard apply.
  Advisory only: every cited reference is checked at its primary source before
  use, and material advice is adjudicated locally.
