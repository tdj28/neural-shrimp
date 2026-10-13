# Huginn draft v3: causal and statistical source review

Reviewed source/design: `576b0b86c4051fe226b8c8cc4e6bd5e1e0f792b3`.
Date: 13 October 2026. Reviewer: Codex `boundary_resume`.

**Behavioral scope:** Part A can test whether a gold-conditioned spectral
diagnostic predicts a finite-budget residual across held-out questions. Part B
can test whether changing the prescribed relaxation coefficient changes this
model's budgeted answer accuracy. Neither part by itself establishes that
spectral instability mediates task failure, a usable answer-blind warning, or a
general benefit from dynamical geometry.

**Disposition:** the primary definitions need narrow source/wording corrections
before freezing. Part B's intersection-union logic is sound; its bootstrap
component is approximate. No test item, model, numerical array, Huginn outcome,
producer analysis or job was opened or executed for this review. The supplied
assignment/design include disclosed development results, which were read as
design context. This reviewer has no independent evidence about test-data access
by the producer. No Git operation or source edit was performed.

Source SHA-256: design
`ed5d38475c654b76dba0080c3c3a2d6a1adad18605d49f0aeb817307cb33e299`;
`tools/huginn_arc_analysis.py`
`79a078ce77c9235116a72c76858a93dd970191c15f306062ffa2be9df06aaa04`.
All ten exported source/design/test files and assignment excerpt were hash
checked; identities are in `source-manifest.json`. The exact tests were read,
not run. Cascades independently owns the ARPACK, map/tokenization, finite-
precision and residual-index review; this note does not clear those obligations.

## 1. State the gold-conditioned estimand and match predictor to outcome

**Design lines 58–75; `huginn_relax_eval.py:352–361,371–389,460–474`,
`Relaxed:160–165`.**

The spectrum uses the prefill containing the correct answer. That is privileged
gold information, even though the two Part A outcomes do not use correctness
labels to score their values. It is acceptable for a predeclared,
teacher-forced, gold-conditioned dynamical diagnostic; it is not an answer-blind
predictor available to a solver before the answer is known. There is no need to
discard this study solely because it deliberately conditions on gold. State the
conditioning, and do not infer deployment utility or unassisted correctness
prediction from it. “No outcome depends on correctness” does not make the
predictor gold-free.

More concretely, the current outcome residual is a **single masked Frobenius
norm across the whole batch of candidate choices**, whereas the spectrum is for
the gold-choice sequence alone. Thus the primary is currently a gold-spectrum
predictor of an all-choice aggregate residual, not a same-map spectral-rate
validation. The batch norm is effectively weighted by the choice states' norms
and can be dominated by another choice. The design should either name this
proxy estimand explicitly or record/use a matched gold-choice residual before
freezing. Cascades also identified tokenization and residual-step alignment
obligations; those must be resolved separately if the same-map claim is kept.

**Design lines 77–86; analysis lines 275–286.** S correlates questions with one
another within each fixed lambda arm. Persistent item differences can make all
four correlations positive without showing that the spectrum predicts an
item's *change* from native as lambda changes. Accept S as the declared
held-out association/prediction criterion, but do not describe success as a
causal mediation test. Part B intervenes on lambda, but uses a different task
and autoregressive trajectories rather than these ARC gold-prefill fixed
points. The two results together would be consistent with the proposed
mechanism, not identify it.

## 2. The Part A loader/statistic can change the frozen experiment silently

**Analysis lines 295–321.** Only the union of question indices and presence of
each arm are checked. Missing `(question,arm)` rows pass those tests. Duplicate
rows pass too. `spectra[q]` overwrites duplicate spectrum files and admits
out-of-range questions; `questions=sorted(spectra)` then uses that set in the
exclusion denominator. Extra spectra can reduce the apparent excluded fraction,
even below zero. The actual scalar outcome is simply the last residual entry,
and answer-change scoring uses whatever steps are present (`267–271`).

**Analysis lines 281–286,318–330.** Missing questions are dropped separately in
each arm. A constant or otherwise degenerate arm is skipped, so the “mean over
four arms” can become a mean over fewer arms. Zipping the shorter value list
with all arm labels can mislabel the remaining correlations. `nanpercentile`
also silently removes undefined bootstrap draws.

**Required narrow correction:** before calculating S, admit exactly one
spectrum disposition for every requested question, no other/duplicate question,
and exactly one outcome for every requested question × four-arm cell. Retain
all fixed-point failures and their identities; the 25% rule means at least 48
of the originally fixed 64 questions must qualify. Require the stated readout
steps 24–32 and correctly indexed finite residual, then use the same qualifying
question set for all four correlations. No arm may disappear from S. Declare
what makes S or a bootstrap draw undefined and retain the count; do not turn a
different smaller statistic into a primary pass. Add tiny missing-cell,
duplicate/out-of-range-spectrum and constant-arm controls. The current happy-
path test (`tests/test_huginn_arc_analysis.py:70–87`) does not exercise these.

The exclusions are predictor-side and computed before readout, which avoids
outcome-based selection if the plan is followed. Nevertheless the resulting
inference is conditional on the numerically qualifying subset; passing the
25% gate does not establish representativeness of excluded questions. Report
the complete exclusion/partial-spectrum statuses, without replacing failed
questions by later indices. Numerical spectrum qualification is Cascades' lane.

## 3. Uncertainty and the secondary pooled test

**Design lines 79–86; analysis lines 319–333.** Resampling the same question IDs
across all arms is the correct dependence unit for the declared primary, once
coverage is exact. Its 95% percentile-bootstrap interval is approximate and
conditions on this seed, numerical pipeline and qualifying items. File-order
test indices are fixed, not randomized sampling from every reasoning task:
generalization of this interval assumes the chosen items are reasonably
exchangeable for a stated target population. The fixed finite benchmark's
observed S itself has no question-sampling uncertainty unless that population
interpretation is introduced.

The A3 Fisher table pools repeated question–arm cells. Those cells are not
independent, so the usual exact Fisher p-value is not calibrated as a
question-level inferential test. Since A3 is explicitly descriptive, the
minimal correction is to keep the counts/rates and drop or label the p-value
non-inferential. A question-cluster analysis would require its own frozen
definition. Calling a secondary test descriptive removes its success role,
not its dependence problem.

## 4. Part B: conjunction is valid, but use accurate test and effect labels

**Design lines 106–112; analysis lines 82–97,373–385.** Write
Delta_lam = accuracy(lam) − accuracy(native). The global null for the desired
joint claim is

`{Delta_1.20 >= 0} OR {Delta_1.05 <= -0.04}`.

Rejecting it only when both component nulls reject is an intersection-union
test. No Bonferroni factor is needed for that *joint* claim if each component
test is level .05; correlation between their estimates does not invalidate
this logic. This does not cover arbitrary extra claims across Part A and all
secondary contrasts.

The implemented McNemar direction is correct: losses versus gains are tested
with the upper binomial tail conditional on discordance, returning p=1 with
no discordances. The non-inferiority formula matches the plan: the 5th
bootstrap percentile must be strictly above −.04. That is a one-sided 95%
lower bound, also the lower endpoint of a central 90% interval; the field name
`lower90` should not be read as a one-sided 90% bound. The percentile bootstrap
is only a nominal/approximate level-.05 procedure, unlike the exact conditional
McNemar test. Therefore do not claim an exact finite-sample 5% guarantee for
the combined procedure.

The four-point margin is an absolute accuracy difference and should be justified
as practically acceptable before the freeze, not by whether development happened
to pass. It permits a loss; rename `B3_no_loss` in prose as non-inferiority
within four points. B1 tests any decline, not a specified large decline, so
`B1_collapse` alone cannot establish a collapse in magnitude. Report effect size,
discordant counts and intervals. B2 remains descriptive even though its JSON
contains a `success` field. These are interpretation fixes, not requests to tune
new thresholds on the held-out data.

## 5. Missingness, fixed budget and label separation

**Analysis lines 349–367; evaluator lines 239–295.** Part B correctly refuses
duplicate `(arm,index)` rows and incomplete coverage for every requested arm.
Its literal arm strings are preserved by the producer; there is no demonstrated
`1.20` versus `1.2` formatting bug. Keep the planned 200 problems even after an
interruption: a completed prefix is not the frozen primary sample. Runtime
failure, absent evidence and a generated but wrong answer are distinct states.

The 256-token cap and stopping rule define the evaluated procedure. A capped
answer is an observed budgeted outcome, not a missing row to remove; report
cap/stop frequency alongside the existing token/format summaries so a change
in stopping behavior is not automatically called a failure of reasoning or
fixed-point stability. The source generates from the question/exemplars and
only reads the held-out gold answer for scoring afterward (`270–283`), so this
Part B path has no source-level gold-answer prompt leak.

Before primary analysis, validate the existing row fields against the frozen
split, global index, seed, lambda, recurrence budget and prompt style. Current
analysis checks coverage but not those settings; a train/r=8/zero-shot file with
the same indices can pass. This can be a small admission check using already
recorded fields, not new infrastructure. Corresponding actual dataset/model
pin admission is still required; this source-only review does not attest files
on disk. All failed or incomplete attempts should remain separately identified.

## Final recommendation

Freeze after the exact Part A panel/statistic gates and scope wording are
corrected, and after Cascades' numerical obligations are resolved or explicitly
incorporated as limitations. Keep the predeclared Part B conjunction with the
approximate-bootstrap qualification and practical margin rationale. No new
test-data access, new outcome-driven thresholds or expanded experiment is needed
to make these corrections. This review grants no empirical result or launch
approval.
