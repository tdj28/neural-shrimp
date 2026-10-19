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

## 2026-10-12: LLM-dynamics novelty and research-plan consult

- **Approval (human, in session, 12 October 2026):** "if you could please do a gpt pro consult about our
  findings thus far to see if a) they've already been found and if so references and b) if not any
  suggestions/feedback on our research plan?" This approves sending the compact packet below to OpenAI;
  budget in the private cost ledger.
- **Milestone:** chaos, period adding with a 1/N^2 window law (type-I intermittency) and float64 bifurcation
  curves in an LLM's soft-token decoding; a predicted and confirmed Neimark-Sacker bifurcation in Huginn's
  latent loop; metastable self-dialogue attractors under steering.
- **Scope:** one `research-strategy` request using
  `2026-10-12-llm-dynamics-novelty/brief.md`, references questions answered first; no retry without a new
  decision recorded here.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens; canonical client
  reserves and authorization guard apply. Advisory only: every cited reference is checked at its primary
  source before use, and material advice is adjudicated locally.

## 2026-10-13: LLM paper synthesis consult (consult 6)

- **Approval (human, in session, 13 October 2026, about 19:58 UTC):** "yes on pro, budget fully approved". This
  answered the proposal of one Pro consult as research director on a compact brief of the near-final paper:
  claims against evidence and framing for cs.LG and nlin.CD; novelty and literature gaps since consult 5; the
  single most valuable experiment before arXiv. Budget in the private cost ledger.
- **Milestone:** publication-ready synthesis of the principal findings, section 7A of the integrity playbook:
  - numerically nondegenerate cusps and a doubling cascade inside a hostile period-7 window;
  - Huginn's frozen relaxation test, Part B confirmed and replayed independently;
  - controller replication at 1.5B;
  - second model family (OLMo-2 7B);
  - committor estimate of warning.
- **Scope:** one `research-strategy` request using `2026-10-13-llm-paper-synthesis/brief.md`, sent after the Huginn
  Part A outcome is inserted. No retry or second pass without a new decision recorded here.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens. The canonical client's
  reserves and authorization guard apply. The advice is advisory only: every cited reference is checked at its
  primary source before use, and material advice is adjudicated locally.

## Consult 7 (approved; re-scoped and sent before the scaling series)

- **Approval (human, in session, 14 October 2026, about 06:40 UTC):** "Yes", answering the proposal of one Pro consult
  on the revised paper once the 70B results are inserted (amount in the private cost ledger).
- **Milestone:** publication-ready synthesis, second version (section 7A of the integrity playbook): a theory section
  (Jacobian factorisation through the next-token covariance; Hopfield limit; temperature limits; stability of the
  loop versus its steps); chaos behind an exactly repeating text and its one-dimensional, logistic-type reduction; the
  horizon check of map labels; the mixture-of-experts result; scale tests on Qwen2.5-72B and Llama-3.3-70B with the
  70B sparse-autoencoder steering controls; new interpretability, alignment and frontier-model paragraphs; a new title
  and abstract.
- **Decision it can improve:** claims against evidence in the new sections; whether the alignment and
  interpretability paragraphs overreach; framing for cs.LG and nlin.CD, including the human's proposed title; the gap
  a referee would hit first.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-14-llm-paper-synthesis-v2/brief.md`, built
  from the paper at the commit that inserts the 70B results. No retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens; the canonical client's
  reserves and authorization guard apply. Advice is advisory: every cited reference is checked at its primary source
  before use, and material advice is adjudicated locally in `adjudication.md`.
- **Re-scope (human, in session, 14 October 2026, about 08:35 UTC):** asked "Do you want to do a pro consult first
  btw?"; the agent proposed sending the approved consult now, with the brief updated to all 70B results and the frozen
  scaling predictions (docs/scaling-predictions.md), so that its advice can still shape the running scaling series;
  the human answered "Go". Same model, mode, effort, output cap and budget as above; one request, no retry or second
  pass without a new recorded decision. A later final check would be a separate request needing its own approval.

## Consult 8 (approved): is Qwen2.5-72B's absence of chaos real?

- **Approval (human, in session, 14 October 2026, about 14:48 UTC):** "Get a pro consult qwen 72 b not being chaotic
  at all feels fishy". This was an explicit request for one paid consult, following the agent's report that the 72B's
  steered chaos labels (0.8 % of cells) rest on a single twin offset that is unreliable for this model.
- **Milestone:** the paper's headline answer on chaos and model size (section 7A: a result that changes the central
  claim). The decision it can improve: whether the 72B result is a property of the model or of the computation, which
  diagnostics settle it, and how the paper states it.
- **Scope:** one `research-strategy` request with the compact brief
  `2026-10-14-qwen72b-diagnosis/brief.md`. No retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens; the canonical client's
  reserves and authorization guard apply (amount in the private cost ledger). Advice is advisory: references are
  checked at their primary source before use, and material advice is adjudicated locally in `adjudication.md`.

## Consult 9 (approved): are we missing anything huge?

- **Approval (human, in session, 15 October 2026, about 06:50 UTC):** "Please do a Astra pro consult budget [amount in
  the private cost ledger] to see if we are missing anything huge". This is an explicit request for one paid consult,
  given before the human went to sleep. It follows the directive to make the findings rock solid and to show their
  practical significance in the abstract and perhaps the title.
- **Milestone:** a publication-ready synthesis with a framing pivot (section 7A). The decisions it can improve: the
  overnight experiment priorities (mechanism, reproducibility, confirmatory behaviour test), and the paper's title,
  abstract and emphasis.
- **Scope:** one `research-strategy` request with the compact brief `2026-10-15-missing-anything-huge/brief.md`. No
  retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens. The canonical client's
  reserves and its authorization guard apply. The human's stated budget is far above the expected cost (consults 5-8
  cost about 2 each; amounts in the private cost ledger). Advice is advisory: references are checked at their primary
  source before use, and material advice is adjudicated locally in `adjudication.md`.

## Consult 10 (approved): final check of the restructured paper and the two-dimensional shrimp anatomy

- **Approval (human, in session, 15 October 2026, about 13:55 UTC):** "Do both and get a gpt pro consult". This is an
  explicit request for one paid consult, given together with the approval of the editorial restructure (new title,
  grouped contributions, catalogues moved to appendices) and of the fold map of the period-5 window (run FM5).
- **Milestone:** a publication-ready synthesis after a structural revision (section 7A of the integrity playbook): new
  title, four grouped contributions, a 37-page main text with ten appendices, and the fold-and-flip anatomy of the
  shrimp-like windows. The decisions it can improve: what a skeptical referee or a frontier-lab reader would hit
  first in the restructured paper; what else belongs in the appendices; whether the shrimp claims (no one-dimensional
  centre; fold curves with cusps) are stated at the right strength; which last experiment, if any, would most change
  the paper.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-15-restructure-final-check/brief.md`. No
  retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens. The latest-model guide and
  the pricing page were rechecked on 15 October at about 14:25 UTC (same model and rates as consult 9; short-context
  rates apply). The canonical client's reserves and authorization guard apply (amounts in the private cost ledger).
  Advice is advisory: references are checked at their primary source before use, and material advice is adjudicated
  locally in `adjudication.md`.
- **Outcome:** sent at about 15:46 UTC, completed 15:58 UTC (71,366 tokens); adjudicated in
  `2026-10-15-restructure-final-check/adjudication.md`.

## Consult 11 (approved): headline practical implication and the experiment program that would establish it

- **Approval (human, in session, 16 October 2026, about 04:00 UTC):** "Regarding my concerns about a headliner
  practical utility of this finding can you do a gpt pro consult? $15 budget". This explicitly requests one paid
  consult, within a $15 budget.
- **Milestone:** a mature design for a major new research question and campaign (section 7A): program PI, five
  experiments on the practical implications of the loop's dynamics, with a float64 component on rented GPUs. The
  decisions it can improve:
  - which practical implication to pursue as the headline;
  - which experiments to run first, and what is missing;
  - the minimal convincing design, its controls and its sample sizes;
  - what would be overclaiming.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-16-practical-utility/brief.md`. No retry
  or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens. The canonical client's
  reserves apply, and so does its $5 per-call authorization ceiling, which lies inside the human's $15 budget. Advice
  is advisory and is adjudicated locally in `adjudication.md`.

## Consult 12 (approved): new ideas for practical consequences (idea generation, not review)

- **Approval (human, in session, 16 October 2026, about 04:20 UTC):** "No I wanted you to solicit gpt pro for
  additional ideas not just review yours", then "Keep this one and do another just asking for ideas". This is an
  explicit request for a second paid consult, within the $15 budget given for consult 11.
- **Milestone:** the same major campaign as consult 11 (program PI). The decision it can improve: which practical
  consequences to test, drawn from a wider set of ideas than our own.
- **Scope:** one `research-strategy` request with a compact idea-generation brief,
  `2026-10-16-practical-ideas/brief.md`. No retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens, $5 per-call guard.
  Consults 11 and 12 together stay within the $15 budget. Ideas are advisory: they are adjudicated locally and any
  test is designed and run under the usual protocol.

## Consult 13 (approved): why the two families of windows cross in both orders

- **Approval (human, in session, 18 October 2026, about 02:00 UTC):** "two things; one please do a gpt-6 pro consult".
  This explicitly requests one paid consult.
- **Milestone:** a major explanatory claim for the paper, made before submission (section 7A): what organizes the
  crossing of the crescent/band family and the 2-cycle/stripe family in the attractor maps (run SH, with paid GPU
  maps running). The decisions it can improve:
  - which mechanisms to test;
  - which decisive figures and tests to make;
  - what not to claim.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-18-crossing-sheets/brief.md`. No retry or
  second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens, with the client's $5
  per-call authorization guard. Advice is advisory and is adjudicated locally.
- **Usage:** 29,153 input and 24,279 output tokens (the amount is in the private cost ledger). Adjudication:
  `2026-10-18-crossing-sheets/adjudication.md`.

## Consult 14 (approved): a computer-assisted proof of chaos

- **Approval (human, in session, 18 October 2026, about 04:12 UTC):** "can you do a pro consult on certification
  proof of chaos please". This explicitly requests one paid consult.
- **Milestone:** planning a new study to turn the paper's numerically validated chaos into a computer-assisted proof
  (section 7A). The decisions it can improve, before any development time or compute is committed:
  - the route (covering relations, a homoclinic orbit, a Conley index, or a one-dimensional reduction);
  - the feasibility trial that best replaces our cost guesses with measurements;
  - the estimate of effort and compute, and what not to claim.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-18-chaos-certification/brief.md`. No retry
  or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens, with the client's $5
  per-call authorization guard. Advice is advisory and is adjudicated locally.
- **Usage:** 32,490 input and 30,600 output tokens (the amount is in the private cost ledger). Adjudication:
  `2026-10-18-chaos-certification/adjudication.md`.

## Consult 15 (approved): everything in motion — the shrimp and its head, the one-dimensional head, the norm bounds, the ladder

- **Approval (human, in session, 18 October 2026, about 09:35 UTC):** "Please do a pro consult about everything we have
  in motion, especially the bounds and your shrimp candidates", then "Cost approved". This explicitly requests and
  approves one paid consult.
- **Milestone:** a numerically verified shrimp is now in the paper and both companions, the norm-bound tests are
  written up, and the next day of compute must be allocated (section 7A). The decisions it can improve: the wording and
  sufficiency of the shrimp claim; the head's classification; the search for a one-dimensional head; the framing of the
  bounds; the allocation of the remaining compute.
- **Scope:** one `research-strategy` request with a compact brief, `2026-10-18-night-status/brief.md`. No retry or
  second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Responses Pro mode, xhigh effort, at most 40,000 output tokens, with the client's $5
  per-call authorization guard. Advice is advisory and is adjudicated locally.

## 2026-10-18 (about 20:40 UTC): consult 16, attractor naming

- **Approval (human, in session, 18 October 2026):** "can you please do a pro consult on the attractor naming vs return
  map? i jsut want to be 100% sure please". This explicitly requests one paid consult.
- **Milestone:** the paper and its companions gain a section and figures on the attractors (strange attractor,
  unstable-manifold skeleton, basins) and public videos; the terminology must be right before release.
- **Scope:** one `research-strategy` request with the brief `2026-10-18-attractor-naming/brief.md`. No retry or second
  pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 16,000 output tokens, with a $3 per-call authorization guard.
  Advice is advisory and is adjudicated locally.

| Request | Mode | Usage | Cost |
|---|---|---|---|
| consult 16, attractor naming | Pro, high effort, 16k output | 47,413 tokens in all | see private cost ledger |

## Consult 17: check of the rewritten theory section (18 October 2026)

- **Approval (human, in session, 18 October 2026):** "For that section can you do a pro consult to make sure we haven't
  botched anything." This explicitly requests one paid consult, on the rewritten section "Why the loop can be chaotic"
  of the Chaos article.
- **Milestone:** arXiv v1 of the Chaos article tonight; the theory section was rewritten for readers outside nonlinear
  dynamics and must be correct before posting.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-18-theory-section/brief.md`. No retry or
  second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort; the per-call authorization guard and the cost are in the private
  cost ledger. Advice is advisory and is adjudicated locally.

## Consult 18: why the superstable curves of the language-model shrimps avoid crossing (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "for that avoidance explanation, i agree with you, i buy that.
  but please let's do a pro consult too." This explicitly requests one paid consult. It covers the proposed
  explanation (an avoided crossing of the zero curves of the leading real multiplier, from the coupling of two
  critical directions through the transverse ones) and the evidence from the period-6, period-5 and hostile period-7
  windows.
- **Milestone:** arXiv v1 of the Chaos article, held tonight for the second shrimp. The explanation goes into the
  article, the full report and the blog only after this check.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-19-avoided-crossing/brief.md`. No retry or
  second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 20,000 output tokens; the per-call authorization guard and
  the cost are in the private cost ledger. Advice is advisory and is adjudicated locally.

## Consult 19: relevance of the results to loops in current LLM systems (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "To help answer users who might say 'so what' can we find
  academic references for this? ... Can you also get a pro consult on this too? ... we would have to know if the
  differential equation we found to cause chaos would apply to this?" This explicitly requests one paid consult.
- **Milestone:** arXiv v1 of the Chaos article; the discussion should state accurately which inference loops the
  results apply to (expected-embedding feedback, hidden-state feedback, recurrent depth, agentic text loops).
- **Scope:** one `research-strategy` request with the brief `2026-10-19-loop-relevance/brief.md` and the verified
  reference list. No retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 16,000 output tokens; the per-call guard and the cost are in
  the private cost ledger. Advice is advisory and is adjudicated locally.

## Consult 20: the night's new results and how the papers state them (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "Please also do a pro consult once all the data collection is
  done and results integrated into the big pdf, chaos pdf, and blog." This explicitly requests one paid consult after
  integration.
- **Milestone:** arXiv v1 of the Chaos article (submission 20 October 2026); the new results (kite, prompt study,
  symbolic dynamics, zooms and basin checks, exponent offset) are in the article, the full report and the blog.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-19-postnight-results/brief.md`. No retry or
  second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 20,000 output tokens; the per-call guard and the cost are in
  the private cost ledger. Advice is advisory and is adjudicated locally.

- Consult 20 executed 19 Oct 2026 12:32-12:37 UTC (one call, as approved; brief at 4e6e2ec6, dry-004 checked first):
  completed, 65,021 tokens. Review and adjudication: docs/consults/2026-10-19-postnight-results/adjudication.md
  (artifacts in neural-shrimp-data/consults/2026-10-19-postnight-results/execute-001).

## Consult 21: mathematics and references of "Why the loop can be chaotic" (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "Please do a pro review of the why it can be chaotic section
  focusing on the math and references". This explicitly requests one paid consult on Sec. III of the Chaos article.
- **Milestone:** arXiv v1 of the Chaos article (submission imminent); the section was edited today (temperature limit,
  steering, variance shares, orbit bounds) and its mathematics and citations must be right before posting.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-19-theory-math-refs/brief.md` (the section
  verbatim, the factorization paragraph of Sec. II, the cited bibliography entries). No retry or second pass without a
  new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 25,000 output tokens; the per-call guard and the cost are in
  the private cost ledger. Advice is advisory and is adjudicated locally.
- Consult 21 executed 19 Oct 2026 21:06-21:12 UTC (one call, as approved; brief at 4ea66156, dry-002 checked first):
  completed, 61,631 tokens. Review and adjudication: docs/consults/2026-10-19-theory-math-refs/adjudication.md
  (artifacts in neural-shrimp-data/consults/2026-10-19-theory-math-refs/execute-001).

## Consult 22: audit of the batch-layout answer tests (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "These numbers seem high can you audit these experiments with a
  pro consult please", said of the answer-change rates in the article's "Reproducibility and answers" paragraph. This
  explicitly requests one paid consult on those experiments.
- **Milestone:** arXiv v1 of the Chaos article (submission imminent). The paragraph reports the article's
  practical consequence, and the human doubts its magnitudes, so an artifact must be ruled out before posting.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-19-layout-answers-audit/brief.md`. The
  brief holds the paragraph verbatim, the designs and results in compact form, and the implementation facts from a
  local code audit. No raw data. No retry or second pass without a new recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 25,000 output tokens. The per-call guard and the cost are in
  the private cost ledger. Advice is advisory and is adjudicated locally.
- Consult 22 executed 19 Oct 2026 22:46-22:51 UTC (one call, as approved; brief at bec6e88b, dry-002 checked first;
  an attempt at 22:46 stopped before sending because the key was not found, execute-001/failure.json): completed,
  59,617 tokens. Review and adjudication: docs/consults/2026-10-19-layout-answers-audit/adjudication.md (artifacts in
  neural-shrimp-data/consults/2026-10-19-layout-answers-audit/execute-002).

## Consult 23: mathematics, concepts and references of the fold section (19 October 2026)

- **Approval (human, in session, 19 October 2026):** "the entire fold section, i must admit i'm very rusty on this, can
  we please get a pro consult?" This explicitly requests one paid consult on the article's Sec. V, "Periodic windows
  and the saddle-node bottleneck".
- **Milestone:** arXiv v1 of the Chaos article (submission imminent). The section was restructured today (new
  bottleneck diagram, passage and passage-fraction tables, prospective tests rewritten), and its mathematics, terms and
  citations must be right before posting.
- **Scope:** one `research-note-scientific` request with the brief `2026-10-19-fold-section/brief.md`: the section
  verbatim (LaTeX, figure captions and tables included, images omitted). No retry or second pass without a new
  recorded decision.
- **Model:** `gpt-6-astra`, Pro mode, high effort, at most 20,000 output tokens. The per-call guard and the cost are in
  the private cost ledger. Advice is advisory and is adjudicated locally.
