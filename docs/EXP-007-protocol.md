# EXP-007: new-start waveform utility with simulator preview

Status: prospective private exploratory protocol, 11 September 2026. This is
an internally frozen extension, not registry preregistration or a population
confirmatory result. EXP006's full results informed this design. This document,
plan, runtime, analysis, tests and local audit must be committed and pushed
before any EXP007 neural trajectories. No paid review or paid compute.

## Question and decision

Does a simple development-informed hybrid preserve a neural waveform at new
starting weights better than local sensitivity, tangent directions, signed
axes, or finite simulation search? The hybrid uses local sensitivity through
radius .004 and atlas margin at .016. This threshold is fixed now, not fitted
on EXP007. A negative result remains a result: no automatic promotion to a
repeated controller or trained-model experiment.

## New starts conditional on an old map

Use the unchanged EXP004 period-nine connected component, nearest observed
non-nine distances in native alpha/gamma units, and the old crop rule. Remove
all eligible pixels at normalized Euclidean distance <= .04 of any EXP006
starting anchor; normalization uses the full focus rectangle widths. Then
split the remaining pixels into three margin-rank thirds and select four per
third with seeded-first/maximin coverage, seed 20260912. No replacement of
failed starts. The manifest records the selected coordinates, exclusion and
stratum censuses, minimum development distances and every 512-angle atlas
score. Maximin operates within each stratum, so cross-stratum clustering is
possible. Old atlas values were already known: held-out refers to new
waveform interventions, not an unseen map, basin or network. Paths/destination
weights can overlap previous interventions; there is no trajectory holdout
radius guarantee.

Each start independently integrates from (.1,.1,.1) at dt=.01 and .005 with
burn 10,000 and sample 2,000. Both must screen as primitive nine-peak candidates;
independent shooting must verify a primitive attracting orbit under unchanged
EXP004 thresholds. An eligible certificate must have 2701 samples including its
endpoint. Exact initial samples 337,1012,1687,2362 of the 2700 endpoint-excluded
orbit are new phases. Their epochs are index*T/2700, approximately odd eighths.
Failed baselines retain every planned record as not evaluable; no replacements.

## Actions and information

Every nonzero action changes only W13 and W31 at the same native Frobenius
norm r in [.00025,.001,.004,.016]. Weights remain fixed throughout each
trajectory. There are 21 candidate entries, not necessarily 21 distinct vectors:
16 uniformangles, local plus/minus, tangent plus/minus, and atlas margin.
Signed axes in the uniform list are represented exactly. Duplicates remain and
are charged. The local vector minimizes the activation-sensitivity Gram in
native units after projecting out one constant time shift. Its derivatives
start at zero, are integrated 0..300, and the Gram uses only samples 75<t<=300.

Each candidate receives a separate 75-unit simulator preview from the exact
same initial state, dt=.005, sample spacing .1. Score full three-neuron
activation RMS against the baseline orbit with one fitted constant phase
shift. The smallest score wins; exact ties use lowest index. Any failed
candidate invalidates each selector requiring it. Failures in unrelated
candidates do not invalidate a fixed atlas action. Save all raw previews,
metrics, directions, costs, selection records and failures before targets.

| Policy | Candidate entries consulted | Required previews per condition |
|---|---|---:|
| atlas | precomputed 512-angle old-map margin maximum | 0 |
| local | local positive and negative | 2 |
| tangent | tangent positive and negative | 2 |
| axes | positive/negative alpha and gamma | 4 |
| uniform | first 16 uniform-angle entries; no atlas or derivatives | 16 |
| search | all 21 entries, including atlas, tangents and local | 21 |
| hybrid | local for r<=.004, atlas for .016 | 2 or 0 |

The augmented 21-entry search is not atlas-free. Local and tangent have equal
preview counts, but their prerequisite costs differ. Local/search/hybrid use
sensitivity integrations; atlas/tangent/search/hybrid use the old map. All
require the baseline orbit for this task. The runtime shares previews for
research efficiency, while the comparison charges each policy its declared
candidate budget, sensitivity and map dependencies. Report observed preview
and sensitivity times separately and retain the old atlas 140.54 s cost. No
claim of matched total compute, net speedup, or realistic hardware intervention
cost follows. A simulator and clonable state are assumptions, not free physical
measurements.

## Scored endpoint and inference

For every action, independently reintegrate from the same exact phase state
for 300 units at both dt=.01 and .005. Policy choices always come from the
fine-step previews, never from either target. The primary endpoint is RMS of
all three tanh outputs on the 2250 samples 75.1..300, fitted to the baseline with
one constant phase offset using only this evaluation interval. The preview
and score sample sets do not overlap. They are nevertheless dynamically
dependent portions of the same deterministic initial-value problem; this is
explicit simulator look-ahead, not an independent temporal sample. No frequency
refit, time warp or separate per-cycle alignment. Full 0..300 RMS is secondary.

At the three smaller doses hybrid and local are the identical action. Their
contrast should be zero up to exact deterministic replay. With complete data,
the pooled local contrast is one quarter of the largest-dose contrast, not
an additional independent replication of a geometric benefit.

Primary comparisons are comparator minus hybrid error for local, tangent,
axes, uniform 16 and augmented 21. Positive means hybrid lower error. Report
all four radii and their equally weighted aggregate at primary dt=.005;
coarse dt=.01 is a separate numerical sensitivity, never a way to pick a
winner. Equal weighting over four phases within each eligible anchor and four
radii;10000 paired anchor-bootstrap draws, seed 20260912, percentile 95 intervals.
These are descriptive intervals conditional on selected starts. Missing any
required eligible trial makes the affected mean and contrast undefined; no
success-only filtering. Report all status counts, each anchor's contrasts,
zero-dose mean/max floor, selected directions and raw costs. No adaptation,
multiple-comparison discovery claim, optimal-policy claim or population error
rate. Eventual period retention is not remeasured here; EXP006 remains the
separate evidence for that horizon.

## Execution and audit

Exactly 12 anchor attempts,48 phase contexts,192 nonzero decision contexts,
4032 preview entries and 2784 target records including 96 zero shams. Counted
separately:24 baseline long cells and 48 sensitivity integrations if all baselines
qualify. Source-clean/pushed gate, immutable source inputs by SHA256, exclusive
attempt directory, journal and atomic raw files, no resume or retry.
Four CPU workers;1800 s hard whole-attempt ceiling including verification.
A sensitivity failure ends the attempt with its payload; baseline and
candidate/target failures remain in their predetermined records.

Resource basis: EXP006 collected 4320 short-plus-long trials in 423.87 s on this
host. EXP007 removes long target integrations. Native 75/300 traces and waveform
scoring use the already measured code paths. Process one decision context at
a time, with four workers; raw arrays are small and retained on disk. This is
a local CPU experiment and does not require a RunPod launch.

Verification binds every saved byte, re-derives action selection, initial
states, epochs and equal norms, re-evaluates all preview/full/tail RMS at saved
phase shifts, and reconstructs the summary. Scientific scores use existing
rtol 1 e-12/atol 1 e-14 replay tolerances; argmin portability uses the previously
declared interpolation-slope bound. Raw hashes and discrete rules remain exact.
Tests use synthetic data and injected numerical callbacks before targets.
A second local audit reviews raw outcomes before manuscript claims.

## Frozen display contract

After collection, figures use only retained evidence: (1) old map with all old
and new anchor locations and explicit exclusion circles in normalized space;
(2) per-dose means, comparator-minus-hybrid intervals, all anchor largest-dose
paired differences and preview budgets; (3) waveform traces for the first
eligible anchor, first phase, largest radius, primary step, including reference,
hybrid, local, uniform and augmented search. The example is selected by this
rule, never by a favorable effect. Report both failed/undefined panels and
unfavorable findings. Layout may change after inspection without changing
selection, analysis or numerical content; document any additional analysis.

## Claim boundary

A specific waveform preservation task in one published three-neuron network.
New starting weights and phases, within previously mapped geometry. Results
can support or reject this frozen hybrid relative to these baselines. They do
not establish an optimal controller, repeated drift recovery, learned-model
performance, memory capacity, all-basin stability, or a universal useful or
harmful role for neural shrimps. The new unreviewed associative-memory preprint
supplies a separate hypothesis about between-memory crosstalk; it does not
validate this experiment.
