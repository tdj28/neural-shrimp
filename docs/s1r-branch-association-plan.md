# S1R local full-state branch association: bounded exploratory plan

**Status: planning only, 8 October 2026 UTC. No new trajectory, orbit correction,
or spectral computation has been run for this unit.**

**TLDR:** The saved rotation map cannot tell us which parameter-space orbit
its trajectories approach. The cheapest next step is to carry one saved
72-parameter period-2 orbit through five nearby output rates, preserving both
phases and testing whether small steps, half steps and reverse steps return
the same local orbit. Success would support a local numerical branch segment;
it would not associate distant resonances, prove shrimp ancestry or establish
that either historical start family reaches that segment.

This is exploratory and follows the adaptive protocol. The independent
[S1R review](reviews/s1r-rotation-map.md) and the integrated
[Program C assessment](program-c-2026-10-06.md) motivate the design. Planning
basis is integrated commit `e985677d1fa8bb47a64b1b0e5d18153c2f922d17`.
The accompanying JSON binds the existing saved inputs and exact planned
limits. Implementation and a short run card must be committed before any run.
The complex-left condition diagnostic issue identified below must first receive
a separate impact audit, minimal fix and independent review; this plan remains
prospective while that work is outstanding.
No paid compute, new large map, protected seeds or held-out data are involved.

## What the existing code does and what is missing

`tools/explore_a3_ns_curve.py:one` creates the same seeded system separately
for every output slice, runs 40,000 updates from the original initialization,
then calls `continue_ray`. The trace saves boundary parameters and angle, but
no orbit. `continue_ray` carries one phase internally along a hidden-rate ray;
its public path drops those states. It follows attracting points only and can
classify the last multiplier after a correction failure. It therefore cannot
supply the branch-identity test or a two-sided qualified crossing needed here.
`tools/merge_ns_curves.py` is a geometric merge and must not be used for this
experiment. Adjacency or a nearby angle is never a branch-matching rule.

A cheaper reusable seed exists in
`/data2/neural-shrimp-data/explore/criticality/s1r-28002-curve/curve.json`:
it contains full orbit points. Its producer starts at shifted hidden rates
and tries fallback margins. A saved point cannot inherit the historical
`start_1.10` label merely because its parameter/angle resembles that trace.
The second phase must be regenerated and both phases requalified; a JSON
field named `orbit_point` is not itself fresh evidence of closure.

There is also a diagnostic to avoid copying: `orbits.certify_orbit` matches
an eigenvector of `M.T` at lambda, then uses a Hermitian inner product for its
condition number. For a nonreal eigenvalue that pairing is incorrect; one
must match conjugate(lambda), use a bilinear transpose convention consistently,
or use a library's left-eigenvector convention. This unit will compute its own
explicitly checked left/right vectors. No historical condition number will be
retrospectively corrected or used as a gate.

## First pilot: one saved anchor, five output rates

Use **S1R, seed 28002**, 72 parameters, the existing deterministic dataset and
ridge 0.001. Selection was made after viewing existing results and is disclosed:
choose the lower-middle row of the first common contiguous block of at least
five NS-labeled rows in the two historical traces, without inspecting rotation
agreement. The first block has trace indices 1–6, so index 3 gives output
`o0 = 0.6271428571428572`. The full-state input is criticality row 2 at the
same output, with last-attracting hidden log-rate
`h_anchor = 1.125569870288833`. Its seed has 72 coordinates.

Give it lineage `criticality_saved_row_2`; leave its historical start-family
association **unresolved**. Keep two separate empty family records,
`start_1.10` and `start_0.95`, explicitly `not_sampled_in_first_pilot`. Do not
merge either into this lineage or into each other.

1. Verify input hashes, rebuild S1R and save exact x/y/theta0 hashes and model
   parameters. Refine the saved seed at its stored rates, derive the second
   phase by one map update, independently refine from each phase, and apply
   all acceptance gates below. Failure ends the pilot as an anchor failure;
   no silent alternative anchor or fresh-start search.
2. Move hidden log-rate downward by 0.002 in four steps no larger than 0.0005,
   carrying both phases. This defines `h_fixed = h_anchor - 0.002`; it is a
   numerical convenience, not an assumed stability guarantee. Keep every
   intermediate point and failure.
3. At this fixed hidden rate, traverse output offsets `[0, +0.001, +0.002]`
   and independently `[0, -0.001, -0.002]`, both starting from the qualified
   central pair. After the first step, use phase-aligned secant prediction.
   Never initialize a new output slice from theta0 or a historical NS row.
4. For each proposed output link, compare a direct corrector with two half
   steps, using the same parent orbit. Correct backward from the accepted
   child to the parent parameters. Require agreement of both phases and
   record the direct/half-step/reverse graph. If a full step fails, halve it
   at most four times, down to 0.0000625; an unresolved segment ends that
   direction. Do not jump over it or reconnect by parameter/angle proximity.

This unit follows the primitive orbit even if its numerical stability changes.
It does not conflate failure of an attracting trajectory with disappearance of
an orbit. It stops on a failed identity/correction/spectral gate rather than
calling that location a fold or NS boundary.

## Acceptance and explicit uncertainty

All numerical settings are disclosed exploratory gates, not interval bounds.
Start with the existing Newton tolerances: closure uses coordinatewise
`atol=1e-9, rtol=1e-8` and must have maximum normalized error at most one;
undamped relative Newton correction at most `1e-10`; period-one separation
at least `1e-6`. Require finite full states, primitive closure from both phases,
and consistency of `F(x0)=x1` and `F(x1)=x0`. Cap each correction at 20 Newton
iterations and 12 line-search halvings. A failure remains a failure.

For an orbit pair X and candidate Y, compare **both** cyclic permutations,
using the maximum absolute coordinate difference across both phases, divided
by `max(1, ||X||_infinity, ||Y||_infinity)`. The winning phase convention must
be unambiguous: accepted displacement at most the smaller of 0.01 and one
quarter of the parent's normalized phase separation; the losing permutation
must be farther by at least half that separation. Record both costs and the
selected phase bit. Independently corrected phase seeds, direct versus
half-step endpoints and reverse return must match within normalized `1e-7`.
Candidate proximity in the generic corrector is capped at 0.02; stricter
full-pair link gates remain mandatory. Do not permute or sign-flip neurons to
manufacture a match. Such symmetries may produce distinct coordinate branches
and would need a separately defined quotient experiment.

At every accepted pair construct both dense monodromies
`M0=J(x1)J(x0)` and `M1=J(x0)J(x1)`. Save all 72 eigenvalues and right/left
vectors at each phase. Compute normalized residuals
`||Mv-lambda v||_2 / ((||M||_2+|lambda|)||v||_2)` and the corresponding
`M^H l-conj(lambda)l` residual; require all finite and at most `1e-10`.
Use `scipy.linalg.eig(..., left=True, right=True)` and check its convention,
not the inherited dominant-condition shortcut. Report eigenvector condition
numbers, `cond(M-I)`, minimum spectral gaps and phase-to-phase multiset
agreement (assignment over all 72 eigenvalues, tolerance `1e-7`). Residuals are
backward checks, not eigenvalue error bounds. Ill-conditioned or near-colliding
modes remain explicitly unresolved for mode identity; no eigenvector label is
needed to accept a full-state link. A stable/unstable descriptive label uses
the existing `1e-7` unit-circle margin and never becomes a bifurcation claim.

A passed link means **numerically consistent local continuation at tested
step sizes**. Small displacements and reverse replay do not prove uniqueness
of the root or exclude an undetected near-coincident branch. A rigorous
association would require validated continuation/uniqueness tubes.

## Minimal implementation changes, before execution

| File | Exact proposed work |
|---|---|
| `src/neural_shrimp/period2_branch.py` (new) | `qualify_pair(family, params, seeds, options)` refines both phases and checks primitive closure; `phase_match(parent, child, limits)` returns both cyclic costs and rejects ambiguity; `spectral_checks(j0,j1)` checks full left/right spectra; `continue_segment(...)` carries paired states with bounded subdivisions, direct/half-step/reverse checks, lineage IDs and unresolved edges. These functions accept a generic map family so synthetic controls use the identical code. |
| `tools/explore_s1r_branch.py` (new) | CLI loads only the bound saved seed, validates finite controls/seed protection, constructs S1R, and executes the exact pilot. Exclusive attempt directory, one worker, deadline/corrector-count limits, atomic checkpoints, fsynced event journal, source/runtime/input manifests. No use of `run_grid` in the first pilot, no hidden bootstrap or spectrum-only branch selection. |
| `tests/test_period2_branch.py` (new) | Translated smooth 2-cycle with known continuation; deliberately swapped phases; period-1 collapse; remote-root switch; ambiguous phase matching; injected corrector/spectral failure; gap that cannot be bridged; direct/half-step and reverse mismatch; false eigenpair residual; complex left-vector convention; deadline evidence retention. No target trajectories in unit tests. |
| `tools/check_s1r_branch.py` (new) | Offline, hash-bound replay of saved two-phase closure, primitive separation, phase matching, spectral residuals and edge statuses; rerun map/Jacobian checks at saved states only. Refuse missing/altered states or a claimed link through an unresolved gap. |
| `docs/program-c-2026-10-06.md` | After execution, append only the observed local result and scope. Do not replace the saved-map audit or relabel historical families. |

Leave the historical NS tracer, merge tool and archived evidence unchanged.
Do not solve this by enlarging the old map or running fresh theta0 at each
slice. These are proposed code changes; none has been implemented in this
planning commit.

## Resources, stop rules and evidence

CPU only, float64, one process/thread, nice10. Recheck load and RAM first.
Primary pilot limits: 120 seconds internal, 150 seconds external plus 5-second
kill grace, 160 individual phase-corrector calls in total, at most 64 MiB of raw
output. These are hard pilot limits, not a throughput prediction. Measure the
first anchor correction within that budget and stop if the remaining work is
infeasible. Keep each rejected candidate and reason; no automatic budget
extension. No GPU use is needed for this 72-dimensional first unit.

Future output root:
`/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001/`.
Save exact rates (log inputs and binary64 rate bytes), both orbit phases,
phase convention, parent IDs, predictor/corrector states, all diagnostics,
full matrices/vectors, retries, source commit/file hashes, x/y/theta0 identity,
PID/thread/runtime data and final status/manifest. Interruptions retain the
last completed point plus an explicit pending edge. Phases are not independent
experimental units. Raw output is immutable; a new run gets a new attempt.

## Later association work, explicitly outside the first pilot

A local success would justify a small second card, not a global ancestry claim.
At the same central output, bootstrap **once per original family** from the
original theta0 at hidden starts 1.10 and 0.95, retain their exact full states,
and refine each primitive period-2 orbit. Reproduce the historical 40,000-step
horizon if those starts are used; do not replace it silently with a hand-picked
saved root. Follow each independently toward a common comparison parameter
only while qualified local links exist. Equality requires a same-parameter
full-pair match and successful reverse links; matching birth angles or boundary
locations is insufficient. A gap or inaccessible common parameter leaves the
family association unresolved. Failure at the central bootstrap gets no
automatic fallback start.

Only after that association is established should a separate bounded local
hidden-rate search seek a unit-circle crossing at the five output values.
Both bracket ends and all midpoints must contain qualified primitive orbits;
corrector failure is a gap, never the unstable side. Track a separated complex
invariant plane through the existing full-state links, with residual and
subspace-overlap diagnostics; spectral degeneracy leaves mode identity
unresolved. Such a result would be a local numerical crossing candidate,
not a rigorous NS theorem. Following it to distant fold curves or 1/8, 2/15
or 1/23 resonances, and linking map trajectories to a newborn circle, require
additional connected evidence and post-sampling state/period/FTLE checks.

## Pre-execution implementation amendment — 8 October 2026 UTC

The prerequisite complex-adjoint correction and independent impact review are
integrated in authoritative source `5e716f6a191e58022e17052277d4de90c44091c7`.
Implementation is prepared on `codex/s1r-branch-pilot`; **no S1R target
correction, spectrum, trajectory or continuation has been run**. Only synthetic
controls were evaluated. The initial planning-status paragraphs above describe
the earlier state and are superseded by this dated implementation status.
Root must review and sign off the committed source before target execution.

Implementation clarifications, made before target outcomes:

- The minimum output step limits retained main continuation links. Diagnostic
  half-step witnesses may use half that minimum. A rejected full step is
  preserved as rejected; subsequent accepted smaller links can connect only
  the part they actually traverse. Four cumulative step reductions per requested
  output segment are allowed; an unresolved minimum-step trial ends that
  direction. Reduced step size persists until that requested endpoint.
- Hidden backoff uses four fixed steps, two independently corrected phases,
  primitive closure, spectra and parent phase matching. Its first failure
  stops the pilot. The direct/half-step/reverse and adaptive subdivision tests
  apply to output links, as specified in the original sequence. No hidden-rate
  search or additional crossing search is added.
- The two phase-corrector seeds retain their coordinate labels; a corrected
  phase must be within 0.02 of its own seed in the same normalized infinity
  norm. A small undamped correction terminates Newton only after the full
  correction is applied and all independent state gates pass. The line search
  permits the full trial plus at most twelve halvings, and never treats damping
  itself as convergence. There is no initialization from theta0.
- Every accepted pair keeps complete native-phase spectra and a recorded cyclic
  phase bit on each graph edge. Secants use carried, consistently aligned pairs;
  they are predictors, never acceptance evidence. Eigenvalue multiset matching
  uses absolute distance at the planned 1e-7 tolerance, only between the two
  phases at the same parameters. No parameter-to-parameter eigenmode or angle
  assignment is used. Mode identity remains explicitly untracked; gaps and
  conditions are diagnostics, not forward eigenvalue error bounds.
- `src/neural_shrimp/period2_branch_io.py` is a small additional support module
  for exclusive, atomic, fsynced arrays/records. The 64 MiB total cap includes
  journals, final status and the manifest; a quarter is reserved for finalizing
  evidence. Budget checks occur between NumPy/SciPy calls; the external timeout
  remains the backstop for a stalled native operation. Corrector states and
  corrected pairs are persisted before spectral work. A hard kill can leave
  a pending journal/checkpoint without a completed manifest, which is retained
  as incomplete rather than declared replayable or complete.
- The offline checker never repeats Newton or a trajectory. It verifies exact
  source/data identities, accepted saved-state closure and undamped correction,
  complete stored eigenpair residuals and the direct/half/reverse graph. It
  rejects an accepted path through an unqualified node or rejected edge. It
  preserves incomplete status and does not assign either historical family.

Synthetic qualification: sixteen tests currently pass, including a deliberately
synthetic 72-dimensional polynomial map substituted for all target construction
and seed loading in the actual CLI/checker. That complete fixture uses 42 phase
corrections (2 anchor, 8 hidden backoff, 32 output witnesses). It is a plumbing
and mathematical-control result, not a measured S1R runtime forecast. Failures
encountered while writing tests were confined to the test harness and are not
research attempts. The runner records each corrector duration and screens the nominal remaining
20 pair qualifications using the first anchor-pair duration. If that linear
estimate exceeds the remaining internal time it stops as timing-infeasible;
passing the screen supplies no runtime guarantee. The target budget remains
120 seconds internal, 150 seconds
external plus 5 seconds kill grace, 160 phase corrections, one CPU thread and
nice 10. If completion is infeasible, the saved status and graph remain partial.
