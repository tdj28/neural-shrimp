# EXP-018: globally refine the periodic reference and its phase coordinate

17 September 2026. Exploratory numerical refinement, chosen after the
[EXP-017 results](EXP-017-results.md). This is a new reference recipe for the
same two selected cycles. Historical failures and frozen studies remain intact.

**Human-friendly TLDR:** In checked near-edge cases, the network carries an
input's trace for several cycles, but a later input interferes with it. Before
testing that behavior across every planned starting phase, we need a more
self-consistent description of the repeating activity. This experiment refines
the whole orbit and its single phase coordinate; it measures no new memory task.

## Question, evidence and claim boundary

EXP-017 completed all 102 jobs, but both finer solvers left the same nine of
48 one-period event normalization checks above the unchanged 1e-7 limit.
Their agreement was much tighter than the defect. Finer interpolation helped
some phases, while original relative orbit closure remained about 8.49e-11
at the core and 5.61e-10 near the edge. Those observations motivate a globally
more accurate periodic reference and a newly derived global adjoint.

The falsifiable benchmark is whether two independently refined candidates,
all four candidate-by-verifier paths per setting, the new global adjoint and
all 48 paired phase contexts pass the fixed gates below. Failure remains a
failure, with no extra iterations, candidate substitution or relaxed threshold.
The strongest permitted conclusion is improved numerical consistency of these
two cycles at these 24 fractions. This is not a rigorous periodic-orbit proof,
all-phase guarantee, complete shrimp boundary certification, memory result,
trained-network result, or evidence of incremental utility of shrimp geometry.

## Fixed inputs and census

Use the two original qualified reference packets from EXP-014 recovery
attempt 002, binding their receipt and packet bytes. Keep alpha and gamma
fixed. The original certified section states and periods initialize both
methods independently. Bind the EXP-017 receipt and compact summary as
historical provenance. No finite-write trajectory or decoder score enters
any objective, convergence rule or choice of canonical candidate.

Retain exactly the 24 phase fractions and IDs generated with seed 2026091701
for EXP-016. Old development/heldout labels are historical identifiers only:
all these phases have informed diagnosis, so this is a **regression census**,
not a new held-out sample. The machine plan includes every worker and nested
trial/subsolve identity before outcomes.

| Stage | Worker jobs | Maximum IVP integrations |
|---|---:|---:|
| Two shooting methods at each of two settings | 4 | 100 |
| Crossverification plus global adjoints at each setting | 2 | 12 |
| Two event methods at all 48 contexts | 96 | 96 |
| Total | 102 | 208 |

Every shooting job has 25 fixed trial slots, including the initial evaluation.
Unused slots retain their identities and reasons. Each reference job has four
forward verification and two backward-adjoint slots. Rejected, failed, partial,
blocked and not-needed work remains distinguishable. Numerical completion and
scientific qualification are separate conclusions.

## Bounded fixed-parameter shooting

The methods `tight-dop853` and `tight-rk45` use DOP853 and RK45 respectively,
both with rtol=1e-13, atol=1e-15 and maximum step T/16384 at each trial's T.
All numerical arrays use float64. Neither tolerance nor precision changes
within the attempt.

Keep the original x3-maximum section. The four unknowns are the initial state
x in R3 and positive period T. Integrate the three states and all nine entries
of the tangent matrix directly, starting with identity. The shooting residual
and analytic bordered derivative are

```
F(x,T)  = [ flow_T(x) - x, f3(x) ]
DF(x,T) = [ Phi(T) - I    f(flow_T(x)) ]
          [ J(x)[2,:]             0    ]
```

Use fixed state scales S_i=max(1,abs(x_old_i)), period scale T_old and section
scale `max(1, norm(J(x_old)[2,:] * S))`, with the Euclidean norm. Divide residual
rows by [S, section_scale] and multiply derivative columns by [S, T_old].
Record the raw and scaled system, singular values and condition at every trial.
Use a direct bordered linear solve, never a monodromy inverse.

Allow at most six Newton updates and, for each, only damping factors
1, 1/2, 1/4 and 1/8, with Armijo factor 1e-4 on the scaled residual norm.
Reuse the accepted trial's residual and derivative; there are no uncounted
reevaluations. Every evaluated candidate must have positive T and remain within
`max(abs(x-x_old)/S) <= 1e-6` and `abs(T-T_old)/T_old <= 1e-6`. Preserve
out-of-neighborhood proposals without integrating them. These small bounds
are numerical safeguards, not a mathematical proof of branch uniqueness.

Convergence requires relative L2 closure
`norm(xT-x)/max(1,norm(x)) <= 1e-12`, absolute section residual <=1e-12,
and finite bordered condition <=1e10. A small correction alone does not qualify.
Reject nonfinite/singular systems, failed decrease or exhausted budgets; do
not promote a best-so-far state. Retain componentwise closure, scales and
floating-point spacing diagnostics along with every partial solver payload.

## Verify both candidates without moving them

Dispatch a setting's reference job only when both shooting candidates converge.
Integrate each unchanged candidate with both methods: four forward solves per
setting, with maximum events detected during those same solves. Do not call
the old optimizer-based `certify`, recenter a phase, replace a seed, change
parameters or divide a period during verification.

Each path must meet the tightened closure and section criteria, nonstationary
flow norm >1e-7 and sampled state amplitude >1e-6, section/event curvature
`(J f)[2] < -1e-8`, and exactly nine maxima over one primitive period. Exclude
duplicated endpoint events using `max(1e-7,1e-6*T)` from each endpoint, retaining
their times, states and curvature. Count the initial maximum once plus interior
maxima. Every interior maximum must have normalized full-state return distance
>1e-7 from the initial state. Preserve all peak/event and primitive diagnostics.

Retain raw Floquet multipliers, neutral direction alignment and residuals,
unique-direction conditioning and the original transverse attraction margin
1e-5. Require neutral eigenvalue error and left/right neutral residuals <=1e-10,
with flow alignment >=0.99. Reject a nonunique
neutral direction using the prospective singular-value relative threshold
1e-12; retain normalization cosine conditioning with minimum 1e-12.

The independently refined candidates must agree to scaled state infinity
distance <=1e-10 using original S, and relative period distance <=1e-11 using
T_old. Report raw differences too. Compare each candidate's two methods on
the same physical clock. These mandatory within-candidate state/activation
gates are 1e-6/1e-7; compare full tangent and activation-tangent differences at
epsilon=1e-4 using those same relevance gates, also retaining raw operator
differences. Cross-candidate full-array comparisons use equal normalized
phase and explicitly retain the different physical periods. They are
diagnostic-only and cannot replace the mandatory same-candidate comparisons
or the candidate state/period agreement gates above.

## One canonical global phase vector

The DOP853-refined candidate is canonical in advance. Only after both
candidates pass all crosschecks, construct its left-neutral vector from its
fresh DOP853 verification monodromy using the unique SVD direction. Normalize
**once**, against that candidate's section flow. Preserve the raw SVD vector,
singular spectrum, normalization denominator and final vector. The applied
scale is the reciprocal of that saved denominator.
No alternate monodromy provides a fallback normalized vector.

Integrate the adjoint backward along both canonical-candidate dense forward
paths using that **same unchanged terminal vector**. Store state, full Phi,
adjoint and physical derivatives on 4096- and 8192-interval grids, exact dense
values at all 24 fractions, raw endpoints and method diagnostics. Require
relative adjoint periodicity `norm(z0-zT)/max(norm(z0),norm(zT)) <= 1e-10`.
Retain old closure/periodicity thresholds
as explicitly historical diagnostics. No endpoint overwrite or per-phase
`z/(z dot f)` normalization is allowed.

If a reference prerequisite fails, retain all available paths and every
blocked dependent identity. A passing RK45 candidate never replaces the
canonical DOP853 candidate. A completely evaluated negative verdict must
remain distinguishable from missing or interrupted computation.

## One-period event regression and saved-array evaluation

For each qualified reference, each method starts from its own canonical dense
phase state, resets Phi to identity and integrates one period using the actual
phase epoch. Pair the resulting physical event flow with that method's
unchanged nominal phase adjoint. Retain all 96 planned jobs even if blocked.
The unchanged initial and event criterion is `abs(z dot f - 1) <= 1e-7`.
Method comparisons use the state, activation and epsilon-scaled full-operator
criteria above. Preserve signed normalization changes and oblique Q/P defects.

The evaluator rebuilds numerical diagnostics from supplied saved arrays,
reports all candidate/reference/context gates and separate completeness, and
never launches a new ODE solve. It cannot omit a failed phase from a favorable
average. Every one of 48 paired contexts must be available and pass for a full
event qualification. Passing does not retrospectively qualify EXP-016/017.
It also does not qualify the eight-period two-write experiment: a future task
needs a separately frozen design, fresh complete 8T references and observers,
and fresh finite-write trajectories under those new references.

## Resources, freeze and immutable evidence

Use four owned CPU workers, one numerical-library thread each, 180 seconds per
worker, 1800 seconds for acquisition and 180 seconds for sealing. Require
16 GiB available host RAM at launch, preserve an 8 GiB host reserve, and cap
owned sampled RSS at 12 GiB and raw artifacts at 8 GiB. Recheck shared CPU,
RAM, GPU and disk availability without disturbing foreign processes.

EXP-017 measured 157.198 summed worker seconds for 96 event jobs, with
individual times 1.059 to 2.238 seconds; six bases took 17.032 summed worker
seconds. Doubling step density suggests about 3.3 seconds per period, or
roughly 680 summed worker seconds at the 208-IVP ceiling, before overhead.
This is an estimate, not a new target pilot or guaranteed runtime. Tiny adaptive
ODEs fit bounded CPU parallelism; this experiment includes no GPU benchmark,
cloud launch or paid review.

Freeze complete code, synthetic tests, protocol, deterministic plan and local
review on the private branch before any new neural solve. Verify the remote
commit, exact source/input hashes and clean checkout at launch. Use exclusive
raw paths `../neural-shrimp-data/raw/EXP-018/attempt-NNN/`. Seal and reload
inputs and prerequisite packets before dependent dispatch. Preserve immutable
partial evidence, all trial IDs and exact owned PID/start identities; verify
cleanup, inventory, chronology and saved-array replay afterward. Failed
attempts remain intact and changes require a new dated amendment and freeze.
