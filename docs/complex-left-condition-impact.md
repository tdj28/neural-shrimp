# Complex-left eigenvector condition diagnostic: impact audit

**8 October 2026 UTC; correction to a diagnostic, not a reanalysis of historical
orbits.** Audit basis is `7b8cfac7` on `codex/s1r-branch-association`, itself
based on integrated `e985677d`. No historical evidence file is modified.

**TLDR:** One scalar condition diagnostic was wrong for a nonreal dominant
multiplier. It can spuriously report enormous sensitivity for a well-conditioned
pair. The orbit, closure, primitive period, eigenvalues and explicit acceptance
rules do not use that scalar. The criticality coefficient code already uses
the correct adjoint convention. Saved condition values must not be interpreted
as physical sensitivity; they remain preserved with this correction notice.

## Exact defect and fix

In the pre-fix `src/neural_shrimp/orbits.py:110–113`, `certify_orbit` diagonalizes
real `M.T`, selects its eigenvector at `lambda`, and computes `np.vdot(y,v)`.
Line119 stores its reciprocal normalized magnitude as
`dominant_eigenvalue_condition`. `vdot` is Hermitian: it conjugates y.

For `M v=lambda v`, the condition formula is
`kappa=||p||_2 ||v||_2 / |p^H v|`, with `M^H p=conj(lambda)p`.
Since M is real, the required transpose eigenvalue is **conj(lambda)**.
The old y instead satisfies `y^H M=conj(lambda)y^H`, so for a nonreal lambda
`(lambda-conj(lambda))*y^H v=0`: the exact old overlap is zero. Floating-point
roundoff usually makes it tiny, leading to a large spurious reciprocal;
exact zero produces the existing null sentinel.

The minimal fix changes only that eigenvalue match to `np.conj(eigenvalues[0])`
and adds a convention comment. It does not change the Newton solve, monodromy,
eigenvalues/right eigenvectors, sorting, tolerances, or returned orbit. For a
real dominant eigenvalue the matching expression is unchanged. For repeated
or defective eigenvalues a single-eigenvector condition diagnostic still lacks
a simple-eigenvalue interpretation; this fix does not claim to resolve that
separate limitation.

## Data flow and acceptance scope

The field is assigned only in `orbits.py`. In the audited tracked Python tree,
the only production reader of that field is `tools/analyze_exp028.py:250`,
which copies it into a summary table. It is not read by a scientific gate.
`certify_orbit` acceptance (old lines124–136) tests Newton convergence, closure,
candidate distance and divisor separation; stability compares spectral radius
with the existing unit-circle margin. `condition_m_minus_i` is a separate,
correctly defined matrix condition diagnostic and is unchanged.

There is one qualification to “diagnostic only”: the condition computation
shares the function's outer floating-point exception handler. An extreme
reciprocal overflow could therefore have rejected an attempt indirectly,
even though the scalar is not an explicit gate. This audit does not assert
that such an event is impossible. None occurred among the 154 inspected
EXP-028 records: all have `reason=verified_primitive_orbit`, multipliers, and
no top-level error. No failed historical attempt is silently reclassified.

Direct production call sites of `certify_orbit`, at the audited basis:

| File | Lines | Use of result |
|---|---|---|
| `src/neural_shrimp/continuation.py` | 60 | Ray corrector; primitive/attracting flags, phase state and multipliers |
| `src/neural_shrimp/minimal.py` | 195 | Minimal-model continuation; flags, state, multipliers |
| `src/neural_shrimp/ancestry.py` | 43 | Shared ancestry correction; orbit matching and multipliers |
| `src/neural_shrimp/warmup.py` | 19 | Full-state reference; verification, orbit and divisor separation; critical pair recomputed independently |
| `tools/run_exp028.py` | 220 | Persist full certificate; fields used downstream as below |
| `tools/explore_criticality.py` | 269 | Verify/refine cycle; rebuild dense two-step matrix |
| `tools/validate_cifar_multipliers.py` | 40 | S2 dense/Arnoldi comparison seeded by orbit; no condition-field use |
| `tools/explore_scalar_mechanism.py` | 118 | Refine transversality samples; uses verified flag and orbit |
| `tools/explore_minimal_model.py` | 130 | Select corrected attracting start from verified/attracting flags |
| `tools/explore_ancestry.py` | 120,166,172,309,316,351,513,535,613,665 | Ancestry/root/parent correction; flags, state, period and multipliers |

`tools/explore_ancestry_branch.py:24` imports `certify_orbit` but has no direct
call. Existing direct test calls are `tests/test_orbits.py:26,43,52,78`;
new regression controls are additional. These direct sites include the shared
wrappers used by other drivers. The compact audit JSON contains the exact
AST call inventory and source hashes; this is a static current-tree audit,
not a formal proof about external scripts or every historical revision.

`tools/analyze_exp028.py:101–121` selects the EXP-028 positive gate using
embedded status, verified/attracting flags, primitive period, confirmed chaotic
ring fraction and absence of divisor-ring cells. It never reads the condition
field. The new bug fix therefore supplies no basis for changing that gate,
certified periods or spectral radii. No target orbit was recomputed here.

## Historical records and manuscript impact

Inspected saved EXP-028 attempt002 `stage3/certificates.json` and its tracked
`docs/EXP-028-summary.json`: 154 certificates; 70 have a nonzero imaginary part
in their dominant multiplier and therefore an invalid historical condition
estimate. Their stored values range from `8.638329048730333e13` to
`1.0312043428088986e17`; none is null. The other 84 have real dominant values
and are outside this conjugation defect. Exact affected zero-based row indices
and both input hashes are in the companion JSON. This is a field inventory,
not a new numerical correction of those 70 values.

The historical protocol `docs/EXP-028-protocol.md:125–128` promises to record
this diagnostic; that part of the resulting data needs this correction notice.
The existing summary and raw receipts remain byte-for-byte unchanged. A later
recalculation, if needed, must produce a separate source-bound sidecar and
retain the originals. Do not rerun a historical frozen receipt against modified
source and call it an unchanged replay.

A tracked-source search found no manuscript/generated-number computation or
manuscript numeric claim reading this field. The old ODE/ridge-regression
condition numbers discussed elsewhere concern different matrices and code.
This audit supports **no identified manuscript endpoint change**, not a claim
about uninspected external publications. The rigorous C1 producer/verifier
uses `rigorous_orbits.py`, not this numerical condition diagnostic; its source
inventory and proof checks are unchanged.

## Independent criticality check and regression evidence

`src/neural_shrimp/criticality.py:52–55` already selects the eigenvector of
`A.T` at `conj(mu)` and normalizes it by `conj(vdot(p,q))`, giving
`A.T p=conj(mu)p` and `vdot(p,q)=1`. `ns_coefficient` calls that function at
line64 and computes its cubic projection with the same Hermitian convention.
It does not read `dominant_eigenvalue_condition`. Its coefficients and their
signs are not implicated by this defect. Warmup and criticality drivers use
that independent `critical_pair` too.

The new control uses the nonnormal real matrix
`M=0.8*[[0.6,-4],[0.16,0.6]]`, similar to a contraction/rotation by
`diag(5,1)`. Eigenvalues are `0.48 +/- 0.64i`; the analytic simple-eigenvalue
condition is `(5+1/5)/2=2.6`. A fixed-point certificate now reports 2.6 and
spectral radius0.8, independently matched to `scipy.linalg.eig(left=True,
right=True)`. A separate test verifies both criticality eigen-equations,
Hermitian normalization and the same independent condition value. The full
orbit and criticality test files pass: **13 tests in3.82s**, one CPU thread,
nice10. These are synthetic controls; no new S1R trajectory/correction was run.

The separate prospective S1R plan remains unexecuted. Independent review of
this minimal fix and impact scope is requested before using the new diagnostic
in its planned full-state continuation.
