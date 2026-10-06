# EXP-008 prospective census, information-access and chronology audit

Date: 2026-09-13. Independent read-only review of the compiler, runtime,
feature construction, ridge fitting, analysis helper and protocol, before
new trajectories. This audit created no model outcomes or paid-resource calls.

## Independently reconstructed census

The compiler was invoked without writing a manifest. The independent
arithmetic and emitted ID sets agree:

| Quantity | Development | Test | Total |
|---|---:|---:|---:|
| Origins / descriptive groups | 12 | 12 | 24 |
| Phase contexts | 24 | 24 | 48 |
| Origin-phase-radius decisions | 96 | 96 | 192 |
| Preview / feature rows | 1,728 | 1,728 | 3,456 |
| Nonzero candidate targets, both steps | 3,456 | 3,456 | 6,912 |
| Zero-dose targets, both steps | 48 | 48 | 96 |
| All target records | 3,504 | 3,504 | 7,008 |

Every nonzero decision has all 18 ordered entries; duplicate physical
directions remain separate charged entries. Every target ID is unique. Each
origin accounts for 144 previews and 292 targets, so a failed origin cannot
silently disappear. Both phases are required at each of four radii, and every
target is scored at both steps. Numerical steps are not independent samples.

The test selector uses 12 eligible-pixel partitions with complete membership
and split rules. Its 0.04 halos surround all EXP-006 and EXP-007 origins; they
do not exclude every previous destination or make the full spatial rectangles
unseen. Quarter-cycle-separated sample indices 675/1350 replace the initially
proposed antipodal pair before outcomes. These limits agree across the helper,
plan compiler and protocol.

## Feature availability and matched actions

The baseline has exactly 51 distinct named columns; the real and mismatched
geometry learners each have 57. All three receive the same coordinates,
candidate action and norm, phase, period, transverse Floquet radius,
three-component reference statistics, local sensitivity and candidate preview.
The baseline also receives crop membership and signed crop clearance. All
features are available before the corresponding perturbed target outcome.

The local variational trajectory is unperturbed, starts from fixed initial
state, and supplies a Gram over the same 75<t<=300 task interval. This is
explicit model access and cost shared by the predictors, not a test candidate's
later trajectory. Perturbed waveform features use only 0--75. The target mask
is applied to elapsed times before the orbit epoch is added. Its 2,250 samples
run from 75.1 through 300 with one fitted global phase. The measured target
never enters the feature constructor or test decision function.

Geometry supplies only the agreed six old-atlas columns; out-of-crop
availability and pixel/crop censoring remain explicit. It does not supply
an additional action. The uniform16 plus local+/− set is shared by every
method. Period/Floquet features describe fixed-weight orbit dynamics, while
the atlas features describe the parameter plane. A result concerns this
fixed representation and regularized learner, not theoretically new
information conditional on coordinates or a topology-specific mechanism.

Independent reconstruction checked the entire shuffle mapping for both roles.
The donor is `(origin_index+5) mod 12`, with the same phase, radius and candidate
index. There are no identity donors, role crossings or dropped rows. For the
local candidate entries the donor's physical direction can differ. The
protocol correctly describes this as a fixed structured mismatch, not a
randomization test or exact null distribution.

## Development-only fitting and chronological seals

The complete 1,728-row development feature census is fixed. Its fine-step
candidate outcomes are the training labels. The driver rejects test-role rows
and invalidates fitting if any required development feature or any of the
3,504 development targets, including coarse steps and shams, fails.

Each ridge learner performs 12 leave-one-origin-out folds. In the planned
complete design, each fold uses 1,584 training rows and 144 validation rows.
Weights equalize origins. Each fold's centering and scaling are fitted only
on its training origins. Five predeclared lambdas are evaluated, equal-origin
validation MSE chooses the smallest-index exact minimum, and a final full
development fit is made. This is 65 ridge solves per learner, 195 total.
There is no test-driven scaling, lambda selection, clipping or refitting.

Runtime order is development collection, sealed model/input hashes, test
previews, sealed test features/predictions/actions, and only then exhaustive
test targets. The test target journal is exclusive and cannot be opened by
the target-collection entry point without both seals. Target records bind
the decision seal. Verification checks file hashes, seal references and
chronological order, as well as reconstructing the fit and decisions.
The fixed fit-replay tolerance applies only to floating model arithmetic;
selected integer lambda indices, action IDs and hashes remain exact, and
the waveform-error tolerance remains unchanged.

## Failure gates, oracle and cost interpretation

The shared hindsight oracle includes all 18 actual candidate outcomes. A
missing candidate makes its oracle and affected regret undefined, even if
that candidate was not selected. Primary benefit is computed as paired
baseline-selected RMSE minus geometry-selected RMSE; the identical oracle
cancels algebraically. The result averages two phases and four radii within
each of 12 test blocks before the 10,000 paired-block bootstrap. There is one
primary pooled fine-step comparison. Controls and per-dose panels do not
replace it after outcomes.

The driver additionally requires all 7,008 targets to complete for a defined
primary, including all development and test shams and both numerical steps.
Zero-sham means require all 24 expected values per role and step. Failed
baselines and sensitivity contexts generate retained unsuccessful candidate
records; no replacement origin, complete-case fit or reduced oracle is used.
Model failures still permit the planned exhaustive test data collection,
while learned-policy choices and the primary remain undefined as appropriate.

The count is 48 sensitivity solves and 24 qualified-origin baseline procedures,
each with two common-initial-state step assays plus certification. Each
deployed policy is allowed 18 previews per decision; cached reuse across
policies does not erase this information-access cost. All 18 later targets
are evaluation expense, not free deployed forecasts. Raw call wall times,
sensitivity timing and the model-seal fit-stage time permit cost accounting;
sums of overlapping worker-call times must not be called elapsed time.
The receipt records collection before final verification, while the
3,600-second watchdog covers verification too. Old-map construction and
development training remain separate costs; neither net speedup nor equal
total cost follows from equal preview access.

## Source binding and check status

The independently compiled source inventory binds 186 old input artifacts:
the EXP-004 receipt/manifest/selection and 181 focus rows, plus both prior
anchor manifests. Its execution inventory includes all tools Python files,
all src Python files, both native C++ sources, requirements lock, pyproject,
the protocol and all `test_hopfield_information*.py` tests. The compiler itself
is bound; the generated EXP-008 manifest is deliberately absent from its own
inventory, avoiding a circular self-hash. The experiment receipt separately
binds that manifest and the exact pushed source commit. Final compilation must
occur after all source/protocol/test edits.

Independent compiler/schema/shuffle checks pass. The design and model synthetic
suites pass all 34 cases, including train-only scaling, equal-origin weights,
failed labels, corrupted packets, exact ties, unselected missing candidates,
coarse-only failures and paired-block weighting. The separate driver test
suite is being completed by the runtime owner and is not claimed passed by
this scoped review. No blocking census, information-access or chronological
design mismatch was found. Complete prefreeze acceptance still requires those
driver checks and the final recompiled, committed and pushed source packet.
