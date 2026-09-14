# EXP-009 saved-results numerical audit

14 September 2026. This is a post-run follow-up on cases selected using
already observed EXP-008 discrepancies, not a new utility confirmation.
The execution design was frozen and pushed before the new integrations.

**Result:** finer integration preserves the original fixed-policy comparison,
but does not establish uniformly trustworthy candidate labels. All 2,776
planned traces complete; 416 of 1,369 selected cases pass the declared native
RMSE-stability diagnostic and 953 remain unresolved under that diagnostic.
All 220 unique actions selected by the old test policies pass both the native
RMSE and waveform diagnostics. The independent panel agrees on scalar error
for 13/19 cases and on every waveform comparison for 12/19 cases.

The [compact numerical summary](EXP-009-summary.json) contains the exact
counts, every independent-panel case, all policy block means and source
bindings. Full raw evidence and the complete producer summary remain outside
Git at `/data2/neural-shrimp-data/raw/EXP-009/attempt-001/`.

## Integrity and audit scope

The source freeze is `d276782c229aa33e2819ec89f15df4e23c7f7834` on the
privately pushed `codex/server-numerical-followup` branch. Runtime metadata
records remote-head verification at 21:35:12 UTC before target execution.

| Artifact | SHA-256 |
|---|---|
| Full receipt | `f6b392cef49b87ddc04b647c3718424622df940f2cba86638bfed5969a0d2677` |
| Frozen manifest | `925fc507ba40450705348e92dade696727d1726d0e8305007e75236365857850` |
| Complete producer summary | `9bfc4212287e2332f2d6ea4ba35d16e442e1eebc90b0034fa4b6d4bc51450fd4` |
| Terminal records | `7923ed05f3874a2321f2b23d07b1215527ebb25162c1cb1ec6308bb3b1997fb0` |

This read-only audit independently checked all 5,560 receipt-listed files,
their 264,213,303 total bytes, the exact directory census, all 2,745 bound
old-input files and 11 bound source files. All byte counts and SHA-256 values
match. Exactly 1,369 cases receive each native step and exactly 19 receive
each independent solver setting: 2,738 RK4 traces plus 38 DOP853 traces.
There are no missing, failed or timed-out records, no substitutions and no
restarted attempt.

For every raw trace, this audit checked the original initial state,
destination, control delta, exact 0.1-spaced sample times and complete finite
state array. It independently interpolated the original endpoint-excluded
reference at the stored single global phase and recomputed aggregate,
per-component, synchronous and window errors. The largest arithmetic
discrepancy was 4.885e-15. It also independently recomputed all retained
same-time state/activation RMS and maximum differences, with largest
discrepancy 2.665e-15. Every comparison passes `rtol=1e-12, atol=1e-14`.
All case gates, independent pairwise scalar gates, 384 original policy
assignments, 96 contexts per policy and 12 block means were reconstructed
without importing the producer's aggregation helpers.

No new dynamics were integrated for this audit. Replaying a saved phase
checks the recorded objective, not an independently optimized global phase
minimum. The producer's complete saved-array verification also passed after
the run; that strict same-platform check includes the phase search metadata.
Neither audit is an interval-arithmetic proof or a new cross-platform run.

## Numerical refinement outcome

The declared scalar flag requires both
`d2=abs(RMSE_0.0025-RMSE_0.00125)<=1e-4` and
`d2<=d1=abs(old_RMSE_0.005-RMSE_0.0025)`. It means stable scalar error at these
tested resolutions, not convergence of a complete trajectory or acceptable
task loss. The waveform flag separately requires target-tail activation RMS
difference <=1e-3 and maximum absolute component difference <=1e-2 at equal
recorded times.

| Prior-selected group | Cases | Scalar flag passes | Waveform flag passes |
|---|---:|---:|---:|
| Complete refinement cohort | 1,369 | 416 | 381 |
| Old coarse/fine-sensitive cases | 1,012 | 59 | 24 |
| Matched stable axis/local controls | 108 | 108 | 108 |
| All old selected unique actions | 220 | 220 | 220 |
| Zero-dose controls | 48 | 48 | 48 |

Groups overlap: 19 matched controls are also selected actions. The union is
1,369; adding group rows would count those cases twice. The difficult-case
selection was intentional and cannot estimate a population convergence rate.

Of the 953 unresolved cases, 952 exceed the new `d2<=1e-4` threshold. The
remaining case, `information-08-p1350-r0.001-c0`, has `d1=2.4355e-7` and
`d2=1.1917e-5`: its errors are close in absolute terms but fail the separate
nonincreasing-difference gate. All 953 belong to the original sensitive set.
The largest new native error difference is 0.3873191552; the largest change
from the old 0.005 result to the new 0.00125 result is 0.4244980980.

Scalar and waveform agreement are distinct: 36 cases pass the scalar flag
while failing waveform agreement. One passes waveform agreement while
failing the scalar nonincreasing gate. Similar distance to the reference can
occur for visibly different trajectories; a small change in task error alone
must not be described as reproduction of a waveform.

The controls and selected actions are much steadier. Maximum native `d2` is
2.0333e-8 for matched controls, 1.6884e-10 for shams and 3.8221e-6 for selected
actions. The largest old-fine to new-finest change among selected actions is
6.5214e-5. These checks concern exactly the old frozen choices, not choices
that a newly fitted policy might make.

## Independent solver panel

Thirteen of 19 preselected cases pass all three scalar comparisons among
finest RK4 and the two DOP853 settings. Twelve pass all corresponding
waveform comparisons. All seven matched-control sentinels, both sham
sentinels and three unique old-selected sentinels pass both tests. The seven
sensitive sentinels supply the remaining mixed result: one passes the scalar
test and none passes every waveform test.

The scalar-agreeing sensitive case is `information-00-p675-r0.001-c8`.
Its largest independent pairwise error difference is 7.7758e-5, below 1e-4,
yet two waveform comparisons fail their maximum-component gate. Its native
coarse/fine refinement flag also fails (`d2=0.000433284`). Agreement among
the finer scalar values therefore does not retroactively pass every numerical
diagnostic.

The six sensitive sentinels failing independent scalar agreement are retained:

- `development-01-p675-r0.00025-c1`
- `development-02-p1350-r0.001-c14`
- `development-05-p675-r0.016-c4`
- `development-09-p675-r0.004-c7`
- `information-00-p675-r0.004-c11`
- `information-11-p1350-r0.016-c5`

The largest independent pairwise RMSE disagreement is 0.2123934592, between
the two DOP853 settings for `information-00-p675-r0.004-c11`. Thus the
difficult-case behavior is not resolved simply by switching to an adaptive
solver at the two tested tolerances. This observation alone does not prove
chaos, identify the mechanism of sensitivity, or show that no finer method
could settle an individual case. No further integration or favorable solver
choice was added after these outcomes.

## Fixed-policy sensitivity

All four policies retain their exact old selected actions. Each mean includes
all 96 test contexts, with all 12 block means retained in the compact summary.

| Original fixed policy | RMSE at 0.0025 | RMSE at 0.00125 |
|---|---:|---:|
| Baseline learner | 0.01782367968 | 0.01782368017 |
| Geometry learner | 0.01696604209 | 0.01696604197 |
| Shuffled-geometry learner | 0.01640140610 | 0.01640136677 |
| Direct minimum preview | 0.00995885827 | 0.00995885847 |
| Baseline minus geometry | 0.00085763759 | 0.00085763821 |

The original mean contrast remains small and mixed across blocks: eight are
positive, three negative and one exactly tied at the finest step. Direct
minimum-preview selection retains lower mean error than the geometry learner.
This supports numerical robustness of the old selected-action evaluation;
it does not turn the original unresolved utility finding into a new positive
result. No new confidence interval, refit, changed action selection or
confirmatory interpretation is introduced.

This incomplete candidate cohort cannot produce a new all-candidate prediction
MSE or a newly certified finite-action oracle. EXP-008's secondary prediction
and oracle-related quantities retain their original numerical qualifications,
even though all actions its policies happened to select are steady here.

## Resources and what should follow

The receipt records 113.0612151 seconds through collection, summary and receipt
preparation, excluding the producer's subsequent saved-array verification.
It is not an end-to-end process benchmark. The run used four local CPU workers
with BLAS/OpenMP thread limits of one. Paid compute and paid review remain zero.

A defensible four-preview study still needs trustworthy labels for the
learning target and numerical treatment of every action its new selectors
might choose. The 953 unresolved long-target cases cannot be silently removed,
replaced by the most favorable solver, or treated as precise regression labels.
Their number describes this prior-selected cohort only.

The next bounded qualification can test all 3,456 original 75-unit previews
under finer native and independent integration. If those labels are stable,
they could support developing a pre-preview shortlist aimed at finding a low
short-preview error. Qualification must inspect within-context rankings and
selection margins as well as scalar differences. This changes the proposed
training target and needs an explicit later freeze; it is not evidence yet
that four previews preserve 300-unit waveform quality.

Before any new utility evaluation, freeze fresh cases, all policy information
access, matched four-preview baselines, the full-eighteen direct-preview
reference, a task-derived quality tolerance and complete cost accounting.
Also freeze how numerically unresolved later outcomes affect the primary
readout, retaining every required case. Success on short previews alone
cannot stand in for the later task outcome. No next utility protocol is
frozen by this audit, and the original strict EXP-008 replay gate is unchanged.
