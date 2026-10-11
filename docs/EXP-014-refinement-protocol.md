# EXP-014 refinement: selected finite-pulse numerical diagnosis

Dated 2026-09-17. This is an explicitly outcome-selected follow-up to EXP-014
recovery attempt 002. It diagnoses numerical sensitivity of known finite-pulse
trajectories. It does not retrospectively qualify either original setting,
replace original traces, change decoder orientation, refit a model, or estimate
a population effect. The earlier 61-case proposal is preserved as superseded in
`EXP-014-refinement-design-amendment.md`; the signed-complete 85-case design below
was chosen before any new refinement integration.

The motivating saved evidence contains 30 failed full-trace solver comparisons,
covering 18 distinct signed strong-dose cases. All original balanced-accuracy
comparison gates passed despite large potential and activation discrepancies.
Agreement of a saturated analytic readout therefore cannot establish waveform
reliability. This follow-up keeps potential/activation differences and the
unsaturated signed readout q central. No formal convergence-order claim is made.

## Exact inputs and selection

The only historical attempt used is
`/data2/neural-shrimp-data/raw/EXP-014/attempt-002`, source commit
`406b89553b956bac1b97ef05413c799e383a9103`, receipt SHA-256
`2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8`.
The plan binds the receipt, manifest, summary, terminal ledger, original
reference seal and replay record, analysis, postflight, chronology, both reference
packets, seven affected phase-propagator packets, and all selected old trace
packets and records. It records each bound file's relative path, bytes and hash.
The source closure includes the unchanged original numerical implementation.

Selection is deterministic in the original 912-task order. Validate exactly
30 failed, nonmissing original waveform comparisons, their 18 unique signed
physical identities, dose index 2 and epsilon .01. Their union contains exactly
13 `(setting, phase_index, direction)` contexts and seven setting/phase pairs.
Include both signs in all 13 binary contexts, each at all three original doses
(.0001, .001, .01), and one sham for each of the seven affected setting/phase
pairs. No replacement or discretionary deletion is permitted.

This yields 78 signed pulses and seven shams: 85 physical cases. Core contributes
26 cases (four binary contexts and two shams); failed-near-edge contributes 59
(nine binary contexts and five shams). These asymmetric selected denominators
are diagnostic; their failure counts must not be interpreted as an incidence
comparison between settings. The original setting denominators remain 456
traces each, with original qualifications unchanged.

Each physical case has exactly the original setting, phase index, pulse/sham
kind, dose index, epsilon, direction and sign, plus its method-free original ID.
The requested and realized pulse, addition rounding, exact initial float64
state, alpha, gamma, phase epoch and period are copied from the original sealed
packets. There are no new baseline, reference, sensitivity or propagator solves.

## Acquisition and solver identity

Every physical case receives all five new methods:

| Method ID | Configuration |
| --- | --- |
| `rk4-0.000625` | Unchanged native RK4, dt .000625 |
| `rk4-0.0003125` | Unchanged native RK4, dt .0003125 |
| `rk4-0.00015625` | Unchanged native RK4, dt .00015625 |
| `dop853-1e-13-h0.0125` | DOP853, rtol 1e-13, atol 1e-15, max_step .0125 |
| `dop853-1e-13-h0.00625` | DOP853, rtol 1e-13, atol 1e-15, max_step .00625 |

The duration remains 320 time units. Save the original 12,801 samples on the
unchanged .025-spaced float64 grid, including both endpoints. The first saved
state must exactly equal the sealed initial state. Every successful packet must
have finite float64 states with shape (12801, 3), the exact saved time grid,
truthful task identity and method parameters. Native packets bind step count;
independent packets bind success and the integer function evaluation count.
No new solver is relabeled as an original solver.

This requires 425 new traces (130 core, 295 failed-near-edge). The 255 selected
old traces retain their three actual method names: `rk4-0.0025`, `rk4-0.00125`,
and `dop853` (rtol 1e-12, atol 1e-14, max_step .025). There are eight methods per
physical case and 680 complete-or-failed scoring rows. Numerical methods are
accuracy checks, not independent scientific replicates.

## Original fixed-clock readouts

Before any new acquisition, replay all 255 old selected packets through the
new generic scorer against their original saved scores. Check q, noise gain,
phase-only q, full-linear q and zero-response q at each of the four declared
delays with the unchanged rtol 1e-12 and atol 1e-14; reject nonfinite values or
changed identities. This is saved-array replay, not reintegration.

Extract and seal all 85 compact input contexts before traces start. Each
contains original readout times, reference activations, linear coefficients,
noise gain, frozen decoder orientation and immutable phase-only/full-linear
predictions. The original phase-only prediction uses the requested pulse; the
original full-linear prediction uses the realized pulse. Neither is fitted or
recomputed from new trajectories.

For each successful new trace, apply the same scalar Hopfield right-hand side
to its saved states and reconstruct the same Hermite interpolant. Evaluate the
original 32-point clock coefficients at delays 1, 2, 4 and 8 periods. Preserve
absolute q, noise gain, clock hash, phase-only q, full-linear q, zero-response q
and sham bias. There is no phase search, recentering by observed shams, threshold
selection, or post-hoc decoder sign flip, including when analytic accuracy is
below chance. The reference clock and same-time waveforms remain fixed.

## Comparisons and complete denominators

Compare all 28 unordered pairs of the eight methods for every case. Preserve
exactly 2,380 comparison rows, including failures or missing operands:

- 255 original-original pairs (three per case).
- 1,275 original-new pairs (15 per case).
- 850 new-new pairs (10 per case).

Every pair retains full-320-unit state and activation RMS, maximum absolute
and per-component differences, maximum location and time, final signed state
and activation differences, and the same statistics on the fixed late interval
240 through 320 inclusive (3,201 samples). These are direct same-clock
comparisons, never phase-aligned replacements.

The unchanged descriptive numerical gates are maximum potential discrepancy
<= 1e-6, maximum activation discrepancy <= 1e-7, and absolute difference of
signed readout means divided by sigma times the common noise gain <= .01 at
primary sigma = 1e-4 for each of the four declared delays. All required gates
must pass to mark that pair qualified. Missing, nonfinite, misbound or failed
operands produce an unqualified retained row. Passing readout gates never
waives a state or activation failure.

For potentials and activations, first crossing means the **first saved-grid
sample** with a strict `>` exceedance of the respective frozen threshold. The
sampling interval is .025; this is not a certified continuous-time failure
onset. For q it means the **earliest declared completed readout window/delay**
among 1, 2, 4 and 8 periods whose standardized discrepancy exceeds .01. No
sliding, cumulative-prefix or new readout is invented. All trajectories still
run to 320; first crossings never trigger truncation, replacement or retuning.

The signed-complete census permits 3,744 same-method balanced-accuracy cells:
13 binary contexts x 3 doses x 8 methods x 4 delays x 3 noise levels. Noise levels
are sigma 1e-5, 1e-4 and 1e-3. This is analytic Gaussian readout accuracy for
deterministic trajectories under assumed independent Gaussian measurement
noise; there are no sampled noisy trials. Preserve the two signed means,
frozen orientation, phase-only prediction and full-linear prediction in each
cell. Never mix numerical methods between the two signs. Missing either sign
leaves a failed cell. No setting mean, pooled cohort mean, bootstrap interval,
new geometry claim or original-study requalification is produced.

## Resource ownership and failures

The 85 selected historical cases provide measured worker timings, including
payload persistence: the old fine RK4 arm totaled 4.8051 seconds (median
0.05533), and the old DOP853 arm totaled 127.8323 seconds (median 1.50176).
Scaling the native work by 2 + 4 + 8 and adaptive work by 2 + 4 gives a
planning scenario of 834.26 worker-seconds, or 208.57 seconds at ideal
four-worker throughput. This is an extrapolation, not a measured new pilot:
tighter adaptive tolerances, scheduling and analysis add cost. The saved
timing derivation is under the external
`checkpoints/phase-memory-refinement-review-20260917/historical-timing.json`.
No new target pilot precedes this freeze.

Use local CPU only, at most four owned worker processes and one numerical
thread per process. The parent does not create a CUDA context. Require at
least 16 GiB available RAM before launch and preserve an 8 GiB reserve during
execution; stop if the sum of owned process RSS exceeds 12 GiB or raw output
exceeds 2 GiB. The unchanged original resource monitor samples every .5 seconds.
These are sampled ceilings, not a claim of zero transient overshoot. Never
terminate an unrelated process or change another agent's resources.

The producer deadline is 900 seconds covering saved-input replay, compact input
sealing, build, acquisition and saved-array analysis. Each trace has a 60-second
limit. Once the producer finishes or fails, a separate 120-second grace covers
terminal completion, immutable packet/summary/receipt sealing and source
postflight. Grace does not authorize further integrations or a new analysis
attempt. If that grace itself fails, preserve every existing file plus a
sealing-error record; the prewritten complete planned ledger remains available.

Use the unchanged owned-process scheduler: bind task IDs to PID/start identity,
record every start and terminal result, join workers, and verify original
identities absent. Cooperative cancellation retains any available packet or
unaccepted last-RHS diagnostic; a forced kill may prevent a child from sealing
its last arrays. Parent timeout status remains authoritative even if a child
managed to write a nominally complete packet. Preserve every failure without
substitution. Producer cancellation derives from BaseException so per-record
Exception/OSError handling cannot swallow the whole-attempt deadline.

Before any target work, write the complete 425-task and 2,380-comparison planned
ledger with the 3,744-cell BA denominator. Partial attempts retain all terminal
identities and all analysis row denominators. If analysis never completes, seal
a clearly unavailable full-denominator placeholder; raw partial evidence is
still preserved for a separate read-only replay. Completed integrations alone
do not imply completed scoring or comparisons. Receipt completion requires
all 425 terminal tasks complete, full analysis complete, no packet/producer
error and valid source postflight; numerical gate failures can remain within a
complete diagnostic run and are not grounds to omit outcomes.

## Freeze, storage and verification

The complete executable, pure evaluator, tests, protocol, amendment and exact
plan must be locally reviewed, committed on a `codex/` branch and pushed to the
private remote before new target integrations. The review JSON must have status
`ready_for_exploratory_execution` and bind the exact plan SHA-256. Run requires an
explicit expected 40-character source HEAD, a clean checkout including untracked
files, exact committed source/plan/review bytes, source hashes in the plan, and
live private branch HEAD equality. Record executable, argv, cwd, Python/platform,
NumPy 2.5.3, SciPy 1.18.1 and numerical thread environment. Source postflight must
remain unchanged. No frozen original file is edited.

Output is a new exclusive external `EXP-014-refinement/attempt-NNN` directory,
outside both the repository and the original immutable attempt. Resolve symlinks
and reject artifact aliases. The raw directory contains task packets, JSON/NPZ
hash bindings, exact terminal and ownership journals, partial failures, input
seal, analysis, summary, resource log, source postflight and exact artifact
receipt. Preserve all attempts; never overwrite an existing one. Bulky raw data
stays outside Git; compact receipts and audited interpretation may later enter
Git with hashes and bytes.

The read-only verifier checks exact artifact census and hashes, original source
commit and plan bytes in Git, full declared source/review bindings, old-input
bindings, terminal/task/method/ownership identities, cleanup, exact chronology
once traces are dispatched, compact input-seal replay, full saved-array scoring
and all comparison/BA rows, summary conclusions and receipt status. A legitimate
failure after input-seal persistence but before the seal event may have no later
event or trace stage: validate the saved seal and explicit unavailable placeholder
without pretending acquisition occurred. Once a trace stage exists, require
both seal and start events in their correct bound order.

Example commands after integration and review (metadata preparation only is
permitted before the private freeze):

```sh
.venv/bin/python tools/run_phase_memory_refinement.py prepare --out experiments/hopfield/EXP-014-refinement-plan.json
.venv/bin/python tools/run_phase_memory_refinement.py validate --plan experiments/hopfield/EXP-014-refinement-plan.json
.venv/bin/python tools/run_phase_memory_refinement.py run --plan experiments/hopfield/EXP-014-refinement-plan.json --data-root /data2/neural-shrimp-data/raw --attempt attempt-001 --expected-head COMMITTED_AND_PUSHED_40_CHARACTER_HEAD
.venv/bin/python tools/run_phase_memory_refinement.py verify --directory /data2/neural-shrimp-data/raw/EXP-014-refinement/attempt-001
```

The existing authorized private SSH route can satisfy the branch push. Pending
private PR authentication does not authorize bypassing local review or merging
main; no main merge is part of this run. There is no paid compute or paid review
in this diagnostic. The requested Pro consultation was not run and cost $0;
these are local code and scientific reviews, not a Pro endorsement.
