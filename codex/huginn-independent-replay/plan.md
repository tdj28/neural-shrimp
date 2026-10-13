# Task 8: independent replay of the frozen Huginn analyses

Prepared 13 October 2026 after the revised 15:49 UTC assignment. This is an
independent saved-raw statistical reproduction, not new generation, model
scoring, hypothesis selection or a paid run. The live certificate campaign
retains priority and its frozen checkout. Claude owns all pods and producer
analyses; no pod inspection, control or relaunch is part of this task.

## Frozen scientific contract

Use the source/design frozen at `9805ed32055e7ef790df0cc67316357c1067aef0`.
Exact source identities were reviewed in
`codex/huginn-design-review/v3.1-final-source-manifest.json`; analysis SHA
`a38586151f9acf6e60c8d355df22d44d8b8ca7aec78270a6f6e73cfc6b838dc2`.
The earlier final source review occurred after the producer's freeze/launch
announcement and opened no test outcomes. Producer development summaries were
read as context; this replay is not represented as blind experimental design.

Part A: exactly test questions 0–63 and arms 1.0,1.05,1.08,1.10; split test,
seed equal to question index, readouts 24–32, 32-entry per-choice residual trace,
endpoint gold-row R(s31), common token-ID hash in all arms and spectrum.
Retain all 64 canonical spectrum dispositions, including missing/nonqualifying
spectra in an otherwise completed producer attempt. Missing or malformed
readout cells refuse the whole part; absence of a completed producer attempt
cannot be disguised as a valid missing-spectrum disposition. Qualifying
residual threshold 1e-4, finite nonempty LM/SR/LR union with the frozen
10-decimal complex rounding/deduplication, common cohort at least48/64.
Manual average-rank Spearman in every arm, their mean, 2000 shared question
bootstrap draws using PCG64 seed0, sorted cohort and duplicated question
indices retained as a multiset. Undefined arm means undefined statistic;
>1% undefined draws fails the primary. Percentiles of defined draws use
NumPy's original linear interpolation. Success requires the lower95% endpoint
strictly above0.3 and all setup gates. Retain every cohort cell, four per-arm
correlations, all bootstrap statistics, and descriptive detected-instability
versus answer-change counts. Estimated Ritz features are not full-spectrum
stability certificates.

Part B: exactly problems0–199 and arms1.0,1.05,1.15,1.20, split test, steps32,
cot8, seed equal to problem index, max256 new tokens. Concatenate only the
complete authenticated disjoint producer partitions. No duplicate/extra/missing
cell or prefix analysis. Both flexible and strict correctness are retained.
Compute paired effects, lost/gained counts, exact one-sided binomial sums,
10,000 paired bootstrap draws (PCG64 seed0 restarted for each comparison),
95% intervals and one-sided95% lower bounds. B1 is decline at1.20 (p<.05);
B3 allows four percentage points of loss at1.05 (lower bound>-.04); joint
success is their conjunction. Keep B2/strict outcomes descriptive and report
all cap/token/format summaries. No threshold, arm or hypothesis adaptation.

## Independent implementation and comparison

`statistics.py` imports no producer module, model or data loader. Average ranks
and Pearson correlation of ranks reproduce Spearman; the McNemar tail is a
sum of exact binomial integers divided by2^n. Bootstrap draws and the frozen
feature rounding remain the declared scientific contract, not new choices.
Synthetic tests compare against SciPy and directly generated PCG64/percentile
references, including ties, duplicated bootstrap indices, undefined arms,
nonfirst-arm ID mismatch and every Cartesian-panel gate.

Do not claim bitwise identity across numerical libraries. Before target access,
freeze a comparison of all producer primary/descriptive fields: exact equality
for indices, counts, dispositions and decisions, absolute tolerance1e-12 for
finite numerical fields, and explicit matching of undefined values. Retain
every field-level difference and the unrounded values, including discrepancies
that do not change conclusions. Extra independent diagnostic rows are not
silently compared to absent producer fields. Stronger malformed-input checks
(e.g. negative norm, non-Boolean correctness, nonfinite choice log likelihood)
are evidence refusals, not altered scientific success criteria.

## Required input authority before any actual replay

No target arrays or numerical JSON are opened during source preparation.
Producer readiness must include complete retrieved raw archives, exact archive
and extracted-file hashes, local retrieval and verified-termination receipts,
original source/invocation/asset identity, predictor-before-readout ordering,
and immutable producer analysis outputs from the frozen contract. Read the
local receipts; never inspect another owner's pods. Hash-check all bytes before
analysis and again afterward; use only a complete, disjoint inventory for the
selected part. Keep all failed attempts and distinguish their replacement
identities. The producer's completion statement alone is insufficient for
numerical admission. No output is written into a producer directory.

Task-owned output root will be
`/data2/neural-shrimp-data/codex/huginn-independent-replay/`, with fresh exclusive
attempts, full raw-derived audit rows, warnings, exception/timeout records,
source/plan hashes and compact Git receipts. Actual input pins and execution
card remain **unprepared and unlaunchable** until the local authority arrives.

## Resource gate and current state

Proposed envelope: one CPU worker plus its bounded supervisor, nice10, explicit
BLAS/OpenMP threads1,4GiB address-space cap, startup MemAvailable>=20GiB and
runtime floor16GiB. Share at most four auxiliary CPU slots including supervisors
with Task11; serialize substantial replays unless Root accepts a measured
concurrent plan. The five-worker certificate campaign remains unchanged.

First measure a full-size synthetic A64x4/2000-bootstrap and B200x4/10000-
bootstrap workload under a reviewed CPU1/4GiB/120s profile card. Then set the
actual replay watchdog from that measurement and declared input byte counts,
checkpoint source/card privately and obtain Root's measured admission before
execution. No target replay or model evaluation is authorized by this note.

36 focused synthetic controls pass in0.67s. The first run had one fixture-only
failure: NumPy inferred float dtype for an empty Boolean test vector; explicit
Boolean dtype repaired the fixture. No reducer, hypothesis or target output
changed. Source, tests and this plan are ready for independent review; the
profile and actual-data driver are subsequent bounded preparation steps.
