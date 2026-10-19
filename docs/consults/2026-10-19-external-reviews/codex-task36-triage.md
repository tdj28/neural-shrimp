# Codex Task 36 (mathematics review): triage, 19 October 2026

Review: `codex/math-review-task36/review.md`, of article snapshot 5a37af47 (00:39 UTC) and the report snapshot
d709c5ac. Codex's verdict: the core map and derivative algebra and the local proof logic are sound; the geometry,
symbolic-dynamics and universality summaries needed corrections.

Many article findings were already fixed by the external-review pass (article c7528a8) before this review arrived:
A1 (in part), A3, A4, A6, A7, A8 and A14. The rest are applied in article commit 6fb6e7f:

| Finding | Decision |
|---|---|
| A1 noninvertibility | Applied with Codex's wording: the sign change differs from the fixed-sign Hénon setting but does not establish noninvertibility or a two-dimensional reduction |
| A2 superstable curves | Applied: "candidate superstable curves" (sign changes of the leading real multiplier), "Type-B-like"; figure labels regenerated |
| A3, A4, A6, A7, A8, A14 | Already applied in c7528a8 |
| A5 fourth-leg caption | Applied: 127 with period 5, one with period 20 |
| A9 scalar fits and universality | Applied: the fit holds on the sampled range; universality "occurs in suitable strongly dissipative families" |
| A10 discussion | Applied: what the factorization establishes, without "how steering creates it" |
| A11 rank-one discrepancy | Applied: the truncation's eigenvalue is about 13 times smaller than the full leading modulus |
| A12 strongest contraction | Applied: "this point of small leading multipliers" |
| A13 uniform gap | Applied in the preview |
| A15 zero temperature | Applied: unique maximizers |
| R1-R9 (report only) | Delegated to the report/blog sync |

DSS6 (running) tests the related question for the period-6 shrimp directly, with zero curves of step factors read from
clean cells only.
