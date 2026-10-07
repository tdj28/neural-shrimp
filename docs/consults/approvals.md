# Human approvals for paid Pro consults

Each paid request needs its own explicit human approval (canonical integrity
skill, section 7A). Costs are recorded cumulatively below.

## 2026-10-07: two consults, total budget $15 (raised to $20)

- **Approval (human, in session, 7 October 2026):** "you have go on the gpt pro
  review, budget up to $15" and "please also consult gpt pro on your research
  roadmap in including the runpod plan, and I'll approve runpod once we get
  that cleared."
- **Scope:** (1) research-strategy consult on the three mechanism questions
  (`docs/consults/2026-10-07-mechanism-strategy/brief.md`); (2) director-level
  review of the research roadmap including the RunPod plan. One generating
  request each; a retry or second pass needs new approval.
- **Model and mode:** `gpt-6-astra`, Pro mode, rates verified against the
  official pricing page on 2026-10-07 ($10 input, $12.50 cache write, $50 output
  per million tokens). Client: `../agent-skill-documents/scripts/review_experiment_plan.py`
  at the commit recorded in `docs/skill-provenance.json`.
- **Major milestone:** mechanism pivot of Program C (the quasi-periodic route
  to chaos replaces shrimps as the central claim) and the costed plan for
  paid compute.
- **Ceiling:** $15 total across both requests, raised by the human to $20 on
  7 October 2026 ("consult budget up to $20 if you need more let me know");
  each request's preflight authorization at most $5 (client hard ceiling).

| Request | Authorization | Reserved estimate | Billed estimate (from usage) | Cumulative |
|---|---|---|---|---|
| (1) mechanism strategy | $3.25 | $3.02 | $1.55 (31,129 in / 24,797 out tokens) | $1.55 |
| (2) roadmap and RunPod plan | $3.25 | $3.00 | pending | pending |
