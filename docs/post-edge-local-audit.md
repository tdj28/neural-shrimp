# Post-edge local scientific audit

Read-only independent review on 7 October 2026 (user time zone). This record
binds the snapshots reviewed, rather than claiming approval of later edits.
No paid service or experiment was used; tests and saved-evidence replays ran
on one local CPU thread with nice 10. Other agents' worktrees were untouched.

## Scalar mechanism

Reviewed commit `894ee5b8c967d06764a4f3071949e5ffbd7d3578`, including
`scalar_mechanism.py`, runner, analyzer, tests, receipt and the scalar addition
to `docs/explore-minimal-model-2026-10-07.md`. An isolated Git archive of that
commit was used for execution.

No blocking mathematical or implementation error found. Independently checked
the gradient, Hessian, composed second/third derivatives, cubic mean/difference
cycle equations, explicit balanced cycle, longitudinal/transverse multipliers,
Jury margins and minimum flip-rate formula. Positive-rate symmetric similarity
and the branch-specific real-spectrum argument are correct. The fixed regression
target is 1.2 in every retained grid case.

Seven scalar tests pass. The raw hash/receipt regeneration passes and reproduces
12 flip and five NS candidate outcomes; the three-point amplitude exponent is
0.4978966467268106. Direct checks of all five saved NS roots find distinct
period-2 points, complex unit-modulus multipliers, positive crossing slopes under
step halving and negative analytic coefficients. The minimum reported angular
distance from a strong resonance is 0.00676915340915446. The largest saved
augmented-root residual is 8.881784197001252e-16, and maximum analytic versus
finite-difference coefficient discrepancy is 2.823465905521516e-9.

Supported scope: exploratory numerical mechanism in the selected branch of a
two-weight model, and balanced-branch algebra. This is not a rigorous invariant
circle, a complete phase diagram, global necessity of ridge/asymmetry, or a
training/geometry benefit. Scalar cycle formulas reduce to Hofmann's balanced
cubic formulas under rescaling; cite that prior derivation when including them.

## Warmup and shared-state interventions

Reviewed commit `2c85cfbeaccf248b58fef2dfa5f7b8854042e1ce`, including `warmup.py`,
runner/checker, tests, receipt and the criticality summary. An isolated Git
archive was used. No blocking implementation error found.

Eleven warmup/criticality tests pass. The raw audit hashes 12 files and exactly
replays 704 sampled cell transitions and both per-update losses. Independent
checks also establish:

- Every fork has the identical common terminal state and the same comparison
  horizon; cell indexing correctly pairs the two starts with all three policies.
- All 82 frozen references close in both phases, with maximum absolute residual
  6.002143226879753e-16; reference loss means/maxima match recomputation exactly.
- Full-spectrum dominant log moduli match saved alpha exactly; the largest
  noncritical multiplier modulus is 0.9807054889982986.
- Every saved postcrossing peak, state-distance maximum and terminal loss ratio
  reproduces independently from raw arrays and the stated initial-orbit baseline.
- Losses refer to the two pre-update states of each pair, whereas saved states
  are pair endpoints. The checker and reported endpoints respect this indexing.
- Global reduction deliberately has no continued local orbit reference. Local
  departure is excluded for that arm; its loss recovery benchmark stays shared.

Supported scope: two previously inspected points of S2-28011 under matched
local multiplier offsets and two finite-horizon ramp schedules. Near-orbit
hard/soft behavior differs; ordinary hard initialization was already distant
from the local branch and is only an accessibility control. Reversal and
halving are different-sized interventions; this comparison establishes neither
an optimized controller nor geometry-specific advantage. Halving wins the
reported terminal-loss comparison. Loss spikes alone do not classify criticality.
No rigorous chaos, universal hysteresis, or transformer consequence follows.

## Manuscript snapshot

Reviewed `paper/gd-route/main.tex`, README, bibliography and reference ledger at
`a9e73a96`, compared with `bc9c526`. The new evidence table, heavy-ball qualification,
continuous-symmetry caveat, negative ancestry result and pending citation-content
labels materially improve accuracy. The following issues were sent to the
coordinator for correction; line numbers refer to this snapshot:

1. Lines 343/348: a unit-circle crossing is not the point where multipliers first
   become complex. Use "complex pair crosses the unit circle".
2. Lines 191-193: a real fixed-point spectrum excludes NS, but does not make every
   +1 crossing a fold. Qualify symmetry/degeneracy and the minimum/rate-scaling flip.
3. Lines 285-288: the implementation orders F-samples for odd periods and F^2
   samples for even periods. State q=p for odd p, q=p/2 for even p; p=2q is not
   universal.
4. Caption 412/body 426: separate projected trajectories do not demonstrate
   topological circle breakdown. Describe a broadened/irregular projected state
   with finite-time chaos evidence; update matching figure labels/alt text.
5. Lines 390-391: negative d and generic NS hypotheses imply invariant-circle
   onset, not necessarily irrational quasiperiodic motion; locked dynamics remain
   possible. Scope claims to numerical criticality and the tested trajectories.
6. New scalar integration should attribute the rescaled balanced cubic cycle and
   longitudinal multiplier to the existing cubic-map analysis, while identifying
   the actual new unequal-rate/full-state and fixed-target calculations.

The coordinator reported applying items 1-4 and is integrating the new reviewed
results; this record does not claim the resulting source/PDF has been rechecked.
Load-bearing primary-source spot-check:
[Hofmann 2026, arXiv v1](https://arxiv.org/html/2607.04993v1), especially the
explicit cycle and multiplier in section 2.3 and rotational-edge analysis in
appendix E.2. Version v1 was current when accessed. No claim that every other
reference's full text has been checked; the ledger's pending items remain open.

## Tiny causal language-model pilot: independent review, 2026-10-07 local date

Reviewed final source snapshot `021e96b6d5d4b1e9c3f550c2dce8bcfed5b30432`
from `codex/minimal-mechanism`, archived read-only source into an independent
`/tmp/neural-shrimp-review-lm-021e96b6` checkout for checks. No agent worktree
was edited, and no full training run, GPU or paid action was needed.

**Verdict:** no blocker to including the result with its current exploratory
architectural-feasibility scope. This is one fixed text batch and seed, not
language-model validation, recurrent full-state dynamics, an NS crossing,
chaos, invariant-circle evidence, or training utility.

Source bindings (SHA-256):

- `src/neural_shrimp/tiny_lm.py`:
  `27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8`
- `tools/explore_tiny_lm.py`:
  `43867f85cc025ca92c5c298d41a466430c1efe2534ac1b3de57ef5bbccb8a73e`
- `tools/analyze_tiny_lm.py`:
  `cac91033c352a9cf6eca319fbb28132b31130c5116a9e51fde7c2e219c081601`
- `tests/test_tiny_lm.py`:
  `cc3a963f58ea4a25945e278a3bfdb696132f9138f5be788042f5384a0731f325`
- `docs/explore-tiny-lm-receipt.json`:
  `2769666d2a28c658ee2b3fb2ed20c755979c5a68d23fe226cb7e3ca4301d3f1f`

Four committed tests pass. The supplied offline analyzer checks every
manifest-listed raw file and regenerates the receipt exactly (15,776
parameters; ten main cases: seven completed, three numerical-guard stops).
This checks saved evidence consistency, not training replay or every numerical
quantity independently. Additional independent checks covered the latter
quantities where feasible:

- Reconstructed every batch token and target from retained corpus offsets:
  X is `[start:start+32]`, Y is `[start+1:start+33]`. All 256 targets are
  next characters; the unused final 10% is disjoint. Future keys are masked,
  and changing all tokens after each possible cutoff leaves earlier logits
  exactly unchanged in a float64 control. The final projection reuses the
  same embedding view, so it receives both gradients while occurring once
  in the flat state. No hidden optimizer/cache/random state is present.
- The GD derivative is `I-eta H(theta)`; the two-step JVP evaluates the
  second factor at `F(theta)` and composes in the correct order. Reduced
  dense/finite-difference checks and the symmetric Lanczos control pass.
  Returned Ritz residuals assess approximate eigenpairs, not spectral
  completeness. The initial nonstationary sharpness sets a rate scale,
  not an attractor stability threshold.
- The precision comparison casts the same rounded initial float32 state
  and direction to CPU float64. It is a valid same-state check of combined
  dtype/device differences, not a clean separation of device from dtype and
  not trajectory precision equivalence. The document's limitation is correct.
- All eleven cases, including the timing unit, have bitwise identical
  saved initial states. Recomputed AR statistics match saved results.
  Predictive detrending uses only the first 75%, and regression rows end
  before the holdout. Altering only the holdout leaves AR(1)/AR(2) fitted
  coefficients and roots unchanged while changing test errors. The separately
  reported full-window trend is descriptive. Test predictions use observed
  preceding losses, not recursive multi-step predictions or held-out text.
- Independently recomputed every available lag-1/2/4 maximum recurrence
  and label, with float64 norm reductions differing from saved float32
  reductions by at most `9.02e-8` relatively. Checked loss/ridge decomposition
  against saved pre-update tail states. The stored tail ends at `theta_999`
  in completed 1000-update cases; `final` is `theta_1000`. Final recorded CE
  is consequently the last pre-update CE, as labeled.
- No completed main trajectory meets the stated fixed-point or period-2
  thresholds. Early stops at 551, 10 and 7 updates have no requested final
  128-step tail and are correctly marked `stopped_before_tail`, not failed
  recurrence tests. CPU float64 evaluation at their saved offending states
  gives objectives `13014.3296`, `71896.5457` and `24342.6801`, respectively,
  corroborating the `1e4` guard. Their reported final CE is the preceding
  retained value. A guard is not a divergence theorem; no chaos label follows.

Supported concise statement: matrix-free derivatives and bounded autonomous
full-batch GD are feasible on this real-text causal transformer. Moderate-rate
training loss can become small while parameter state drifts; higher rates
can fluctuate or trigger a numerical guard. Seven completed main trajectories
were screened and nonconverged; three stopped before tail screening. Keep
that distinction when abbreviating “none qualified.” Parameter symmetries
were not removed, and apparent AR structure is not a multiplier or orbit.

Commands used on the source snapshot, with one CPU thread and nice 10:
`python -m pytest -q tests/test_tiny_lm.py` and
`python tools/analyze_tiny_lm.py --receipt docs/explore-tiny-lm-receipt.json`.
The additional source-independent corpus/indexing, holdout-perturbation,
recurrence and guard-state checks described above ran locally against all
retained attempts without new trajectories. Later implementation changes are
outside this review.

## CIFAR orbit-safety review, 2026-10-07 local date

Reviewed `ed1293e35f5c821e7f4a10121f7e030df9930b88`,
`10af5128526975060547844d694a92c3336d68e1` and final
`3dca7f03ba3861a20516c378e8abaca877271478` on `codex/cifar-orbit-safety`.
No producer worktree edits, real-data corrector, GPU or paid run. Tests used
an independent archive at `/tmp/neural-shrimp-review-cifar-3dca7f03`.

Verdict: core numerical gates pass review; three gaps should be fixed before
new continuation. The separate circle command must be explicitly excluded
from this bounded H100 unit unless separately hardened. Findings were sent
to the producer and coordinator; later fixes are not covered by this snapshot.

1. **Bind all mathematical inputs (`tools/explore_cifar.py:259–267,291`).**
   Orbit results hash source and initial state but omit subset-A identity and
   the standardized x/y arrays used. Self-consistency of the subset-A index
   record does not bind results to this run's data. Save the index-record and
   normalized-array digests, with shape/dtype/order, and upstream source identity.
2. **Retain a converged correction before Arnoldi (`:223–227,303`).**
   Corrector callbacks save diagnostics without theta; states are saved only
   after a qualified spectral solve. A controlled spectral timeout produced
   a zero-residual corrector event, empty result rows and no state file. Save
   the corrected state with hash, rate, closure/separation and spectrum-pending
   status before invoking Arnoldi. Failure must not upgrade its evidence label.
3. **Validate continuation settings (`:293–317,334–344,465–480`).**
   Negative `--step` is accepted. A synthetic stable-to-unstable control at
   log-rates -1 and -1.1 produced `candidate_bracket` with reversed lower/upper
   endpoints; negative `hi-lo` skipped bisection. Require finite positive eta,
   output ratio, forward step, bisection tolerance and budgets, plus a finite
   nonnegative margin, or deliberately implement reverse continuation with
   ordered endpoints. Positive defaults are not affected by this control.
4. **Circle is outside the bounded runner (`:371–375,388–434`).**
   It uses larger default Newton/GMRES limits, no deadline, and a long loop
   without incremental checkpoints/manifest. Explicitly excluding circle from
   the planned H100 feasibility unit is sufficient; no broad repair is needed
   for this unit. Harden it before any later use.

Reproductions are synthetic mocks, not new CIFAR outcomes. The script at
`/tmp/neural-shrimp-review-cifar-3dca7f03/failure_controls.py` demonstrates the
lost correction and reversed bracket without a full corrector. Producer was
sent the exact paths and reproduction details.

Passing checks: returned-state Newton residual is recomputed after the final
allowed update, uses `max(1,||theta||)`, and rejects failed/nonfinite GMRES
steps. Full two-step closure and one-step separation precede spectral work.
Ritz residuals are recomputed; incomplete ARPACK convergence is unqualified;
seeded starts and partial-spectrum/nonnormal caveats are correct. Small
residuals imply neither spectral coverage nor an eigenvalue-error bound.
Every positive-step bisection midpoint repeats these gates, and failed
midpoints retain unresolved brackets without crossing labels. Matching
endpoint kinds is only a candidate rule, not branch identity, transversality
or nonlinear criticality. A +1 crossing alone also does not distinguish a
fold from other +1 bifurcations. The angle fix correctly derives the reported
angle from the same top Ritz value: positive-real candidates have angle zero.

Orbit deadlines are checked at operation boundaries, Arnoldi has per-mode
matvec limits, and GMRES has finite restart/cycle caps. A separate process
timeout is still needed to bound a currently executing library operation or
overhead, as the plan specifies. Events are fsynced and JSON checkpoints
atomically replaced; correction-state retention is the gap above.

All **16** tests pass: fourteen passed directly in the archive; two checkpoint
tests initially failed because `git rev-parse` has no Git metadata in an
archive, then passed unchanged with `GIT_DIR` pointing read-only to the source
worktree metadata. This setup limitation was not a numerical failure.

Timing receipt checks passed: raw-file length/SHA, complete timing object,
initial-state/subset-A digests and all medians match. Gradient/HVP/J2VP medians
are 5.268/17.842/33.473 ms; the six-matvec GMRES cycle took 215.782 ms and did
not converge. Rough estimates of 67 s for 2000 real J2VPs and 54 s for 8x4x50
J2VPs are consistent operation-only arithmetic, not complete convergence/time
bounds. Complex residual evaluations can require two real J2VPs; library,
transfer and startup overhead are additional. H100 planning correctly avoids
using forward-GD speedups for Krylov work, requires a device-capable runner
and gradient/HVP/J2VP parity first, and delegates exact costing, ownership,
retrieval and verified termination. It authorizes no launch; current orbit
code remains CPU-only.

Inherited wording caveat: `load_cifar_train` deserializes all training-batch
pixels before slicing A. Thus “only A loaded as pixels” is literally false;
only A is selected/evaluated, and this path does not inspect the sealed B
index file. Correct that wording, or use pre-extracted A inputs if literal
non-loading of other pixels is required.

SHA-256 source bindings:
`src/neural_shrimp/cifar.py` = `79fcb672725f6d2e91d4f0850052cded6b09718e7e86732ee8a8c2f3811b35cb`;
`tools/explore_cifar.py` = `e4f9e0d2d9ab3e0b72231428fc474dd24e53ca9865372d4a90d3c44e3df062be`;
`tools/benchmark_cifar_orbit.py` = `209297e4f26910e18b6aa7e175807cbaf07616e09975f92a5690d96f0c644e84`;
`tests/test_cifar.py` = `7da5990f6d905a2cc929162b6e19574dbe1632726ffbf66731432edc2d3854d6`;
operation receipt = `cfb09091b99ee8b209004c1fc2a14ac8a9b485ad4c85789d438e26164c8e53d8`;
compute plan = `b30d37cf1622596121365d9e5a7088679542250b63ba3aa486e2e153a693601a`.
