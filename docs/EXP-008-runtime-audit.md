# EXP-008 prospective runtime audit

Date: 2026-09-13. Independent local source review before EXP-008 target
collection. This review generated no neural trajectory, used no paid resource,
and makes no claim about an EXP-008 result. Scope: the plan compiler, runtime,
feature and model helpers, their synthetic tests, and the
[prospective protocol](EXP-008-protocol.md). The complementary
[design audit](EXP-008-design-audit.md) and
[census audit](EXP-008-census-audit.md) document spatial selection and the
independently reconstructed denominators.

## Assessment and corrections made before outcomes

No blocking scientific or execution mismatch remains in the reviewed source.
The specified experiment tests whether an old-map representation improves a
fixed simulator-assisted action selector. It does not isolate shrimp topology
as a cause, establish new information conditional on coordinates, or test a
trained model. The strong baseline has the same actions, preview trajectories,
local dynamics and crop information as the geometry learner.

Three material issues were corrected prospectively during review. First,
indices 675 and 2025 were initially proposed. Reading existing EXP-007 cycles
showed nearly antipodal states at ten of twelve development origins, consistent
with the odd vector-field symmetry. The final indices 675 and 1350 are a
quarter cycle apart, at the same trial count; this change used prior evidence,
not new EXP-008 outcomes. Second, a draft fit gate consumed fine labels while
allowing a failed coarse development target or sham. The final gate invalidates
training for any required development failure. Third, a draft summary allowed
a defined fine-step primary despite a failed coarse test target. The final
helper requires both test steps, and the driver additionally requires every
one of the 7,008 development/test target and sham records.

## Information access, chronology and trial accounting

The compiler obtains origins solely from the old focus atlas and prior anchor
manifests. Twelve development origins and twelve new test origins each have
two phases and four nonzero radii. Every condition retains all eighteen ordered
candidate entries, including duplicate physical directions. The total census
is 3,456 previews, 6,912 nonzero candidate targets and 96 zero-dose targets.
There are 1,728 fine-step development labels and 96 test decision contexts.
The 48 local sensitivity contexts and baseline certification procedures are
additional costs, not omitted target records.

The source executes these stages in order: complete development previews and
targets; model fit and model/input seal; complete test previews; all test
predictions and decisions plus their seal; exhaustive test targets. Test targets
are generated regardless of which actions were selected. The exclusive test
target journal cannot be created through its collection entry point before
both seals are present and their referenced hashes agree. Each test target
binds the decision seal. The same saved decisions evaluate both integration
steps; the coarse step is a numerical check, not another statistical sample.

The baseline feature constructor takes no perturbed target trajectory. Its
51 columns include nonlinear coordinates and actions, phase, orbit/Floquet
diagnostics, shared crop metadata, unperturbed local sensitivity and the
candidate's complete 0--75 preview. The six extra geometry columns use the old
atlas only, with unavailable out-of-crop membership explicitly masked. The
structured shuffle uses donor origin `(index+5) mod 12` separately within each
role, matching phase, radius and candidate entry. This is a diagnostic mismatch
control, not a randomization distribution.

The target mask is applied to elapsed time before adding the original orbit
epoch. Its 2,250 samples span 75.1 through 300. All three activations receive
equal weight, and one constant phase offset is fitted over that scored interval.
Preview and target samples are distinct but dynamically dependent. Local
sensitivity extends along the unperturbed trajectory to 300 and is shared
model access whose cost must be counted. No frequency fit or independently
rephased target windows enter the task metric.

## Fitting, endpoints and unsuccessful records

The three ridge learners use the same five penalties and equal-origin
leave-one-origin-out prediction MSE. Every validation fold has its own
training-only weighted scaler. The final model retains its scaler, coefficients,
chosen penalty, complete fold diagnostics, feature order and input digest.
Test features cause no refit, clipping, penalty selection or recalibration.
Prediction MSE and out-of-range scores are frozen secondary diagnostics;
prediction quality and selected-action utility remain distinct quantities.

Any failed required development feature or target, including either numerical
step and zero-dose checks, produces an invalid learner packet rather than a
complete-case fit. Missing or failed candidate outcomes cannot shrink the
eighteen-action oracle. The global primary gate requires the exact complete
7,008-record census and finite bounded scores. Failed baselines or sensitivity
contexts produce retained failed candidate records, with numerical payloads
saved when supplied. Model failure does not authorize a replacement learner,
anchor, candidate set, resumed attempt or shortened experiment. Defined
stepwise diagnostics remain distinguishable from an undefined primary.

The primary paired regret reduction is computed directly as baseline-selected
RMSE minus geometry-selected RMSE: the identical finite-candidate oracle
cancels. Absolute regret is secondary and concerns only the eighteen evaluated
entries. Two phases and four radii are averaged within each test block before
equal-block averaging and the fixed 10,000-draw paired-block bootstrap. These
selected spatial groups do not justify a population error-rate interpretation.

## Verification, tests and release boundary

The verifier checks exact artifact hashes and sizes, realized initial states,
native weight displacements, time grids, feature reconstruction, waveform
scores, development fits, saved decisions, complete IDs and seal chronology.
The existing direct waveform-score replay stays at rtol=1e-12/atol=1e-14;
its separately bounded scalar-phase portability check is unchanged. Only
floating fit replay uses the prospectively declared rtol=1e-7/atol=1e-9 for
conditioned ridge arithmetic. Cohorts, feature identities, hashes, chosen
integer penalty indices and action IDs remain exact. Replaying the same
implementation verifies consistency; it is not independent empirical proof.

The independent command

```sh
.venv/bin/python -m pytest -q tests/test_hopfield_information_design.py tests/test_hopfield_information_model.py tests/test_hopfield_information_runner.py
```

passed **63 synthetic tests in 1.58 seconds** during this review. Tests include
analytic ridge recovery, train-only scaling, equal-origin weights, packet and
seal mutation, all-candidate and coarse/sham failure gates, exact candidate
ties, prospective phase rejection, source orchestration with mocked numerical
calls, and refusal to reuse an attempt directory. No neural ODE is executed
by these tests. Subsequent source or test changes require the affected checks
again; this count is the observed scoped invocation, not a claim about a later
full-suite or CI run.

The compiler binds the old atlas inputs, execution Python sources, native
sources, dependency lock and project metadata, protocol and EXP-008 tests.
The runtime checks the declared NumPy/SciPy versions, clean source and a
remote-tracking branch containing the source commit before creating its
exclusive attempt directory. The final source manifest must be regenerated
after all implementation/protocol/test edits, validated, committed and pushed
before target collection. This prospective audit does not claim that those
final release actions, isolated CLI smoke, final CI or empirical verification
have already completed.

Collection elapsed time precedes the final verification call; the 3,600-second
watchdog includes verification. Report overlapping worker-call sums separately
from elapsed time, and report sensitivity, fitting, old-map construction and
exhaustive oracle evaluation costs. Shared previews do not establish equal
total compute or a net speedup. No RunPod job or paid review was requested or
launched in this audit.
