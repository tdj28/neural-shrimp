# Task14 endpoint source review

13 October 2026. Independent reviewer: Codex literature_cascades.

**Verdict: no remaining source-review blocker.** The source implements a finite numerical check of the two supplied period-seven fold endpoints. The mathematics and declared precision gates are consistent with that scope. Actual endpoint execution still requires the completed profile's measured resource review, a concrete frozen card/plan, private source checkpoint and the external owned supervisor. This source preparation alone authorizes no execution. It cannot certify a cusp or establish a cusp location.

This review read the three new implementation/test sources and the already reviewed numerical/map interfaces. It did not run tests or producer code, load a model, decode candidate or profile arrays, or read target outcomes. Boundary reported eight synthetic controls passing for the initially released source; final delta controls are recorded below separately. Existing profile/evaluator source files remain unchanged.

## Mathematical and numerical checks

1. The actual wrapper fixes rows 13 and 14, dimension 896 through its admitted input container, physical alpha, log10-temperature and `G=F^7`. It saves seven one-step transitions, separately evaluates the composed map and checks relative return error <=1e-10 and one-step/composed consistency <=1e-12. Because seven is prime, separation from period one >1e-6 is the complete proper-divisor list. These are numerical full-state diagnostics, not interval existence statements.

2. Full matrix shape/finiteness and complete spectral classification precede coefficient evaluation. The unchanged classifier requires a simple real multiplier within 1e-8 of +1, imaginary magnitude <=1e-10, isolation and complementary distance from the unit circle >1e-3, and finite eigenpair residuals <1e-8. Hyperbolicity and attraction are distinct: other unstable multipliers do not invalidate this fold coefficient calculation. All multipliers and the refused spectrum are retained. The right mode is unit norm and oriented by its largest absolute coordinate; the left mode has `l^T r=1`.

3. Fresh composed JVPs along the critical mode and a fixed independent direction must agree with the dense matrix to relative error <=1e-10 before computing B, C or the physical-alpha derivative. Nested AD differentiates the complete period-seven map. The mixed directions h2 and h2(mu) are held fixed when computing B(r,h2); differentiating those directions would be a different, incorrect tensor contraction. The implementation keeps them fixed.

4. With `x(u)=x*+ru+h2*u^2/2+...`, `l^T h2=0`, and `u'=u+b*u^2+c*u^3+...`, the fold equation `(I-M)h2=B(r,r)-2br` and `c=l^T C(r,r,r)/6+l^T B(r,h2)/2` are correct even when b is nonzero. Both the nominal fold operator and `mu^2 I-M` are solved, with border/unbordered residual and orthogonality gates <=1e-8. Their h2 changes and cubic difference are retained; the latter must be <=1e-3 under the declared `max(1,abs(c))` scaling. This tests sensitivity to the measured near-unit multiplier; it does not turn approximate closure/eigenpairs into exact ones.

5. The finite differences are the appropriate centered formulas: `[J(x+sr)r-J(x-sr)r]/(2s)` for B(r,r), `[J(x+sr)r-2J(x)r+J(x-sr)r]/s^2` for C(r,r,r), and `[J(x+sr)h2-J(x-sr)h2]/(2s)` for B(r,h2). The direct, center-manifold and total cubic contributions are all retained and compared to AD. Both h2 solutions' full mixed vectors are retained. The main cubic's scalar components/total, projected b, full tensor vectors and critical/probe map differences each have their own comparisons, so a small unprojected error cannot conceal a large projected a/b/c error.

6. All seven fixed relative state steps (1e-2 through 1e-5) and all three fixed physical-alpha steps are retained. No favorable adjacent pair is selected. Every error must be finite; the fixed finest two must be <=1e-3 and either <=1e-5 or reduce to <=0.3 times the first-row error. These are explicitly numerical absolute/relative agreement-and-convergence criteria, not a fitted second-order convergence rate or a rigorous error enclosure. The relative state displacement is `h*||x||` (one at the zero-state synthetic case); scalar errors use `max(1,abs(reference))`. The fixed finest-row sign diagnostics are numerical agreement only. An unresolved a or b sign leaves `orientation_bracket_ready=false`; an unresolved c sign leaves `cubic_nonzero_supported=false`, even if the endpoint's general numerical precision checks pass. Those flags must accompany any interpretation of b's orientation or c's nonzero sign.

7. Physical-alpha AD and centered parameter differences intentionally omit the externally cached prefix. Both current-token and fixed-prefix steering therefore change with alpha. The reported a is in physical-alpha units, not alpha_scale times alpha. The report should not compare its magnitude directly to a producer coefficient until the producer's parameter-coordinate convention is matched. The clock is one complete seven-step map. Simultaneous reversal of r and l changes a and b but leaves c unchanged, while `b*sign(a)` is orientation invariant.

8. A sign change of oriented b between these supplied endpoints, even with nonzero resolved cubic coefficients at both, supports a candidate cusp segment only. A numerical nondegenerate cusp additionally requires locating b=0 and checking the two-parameter unfolding/transversality. Neither interpolation, Newton refinement nor that rank-two test is executed here; all cusp/certificate flags stay false.

## Matrix reuse and execution boundary

Row 13 may reuse only the complete externally pinned profile001 receipt. The reader checks read-only artifact sizes/hashes, exact source/plan/receipt identities, computational completion and owned termination, the same backend, complete original five-file scientific source closure, and the original authority/ancestry digests. It then checks exact full original state and parameters, all 56 input basis batches, every output column and their exact equality with the assembled matrix. Row 14 computes its own complete matrix. The reader does not silently use a matrix at a refined point. Original Git-blob/lifecycle admission of the externally supplied receipt remains the coordinator's responsibility; this reader authenticates the pinned captured bytes and their numerical identity rather than rerunning Git or a model.

The review requested and independently inspected a narrow consistency delta: explicitly require the receipt's `timed_out` field to be false and the profile status's `cusp_qualified` field to be false. They are already emitted by the reviewed profile/launcher and belong to the complete resource-only contract. This adds no scientific threshold or native call.

The worker fixes CPU2/nice10/12GiB AS and 28/16GiB available-memory floors. Its configurable timeout ceiling of 7200 seconds and output ceiling of 128MiB are source bounds, not an approved actual budget. The measured card must freeze lower or equal values and the external supervisor must enforce the wall deadline and ownership cleanup. Completed inputs/returns persist before finite/resource gates. Ordinary scientific `GateFailure` records an unqualified row and permits the other fixed endpoint to be checked; nonfinite failures, unexpected exceptions and resource failure preserve partial evidence and end the attempt. There is no automatic retry or threshold change.

## Source identities

Initially reviewed implementation:

- endpoint_check.py: `4cb9c3cfb87b13cd63dc43c352db88322bbfca2198c3a6784d86b1c91d813de7`
- endpoint_runtime.py: `1c57bd692ae5bb917987608db9a421c88b5773c95c7d84bafc6965e2ef34eb87`
- tests/test_codex_cusp_endpoint.py: `49fd3ecec7cbaa320ee5909357afd401953e1e988f4f25cd354286ba8d8cc347`

Final reviewed release after the two-check consistency delta:

- endpoint_check.py unchanged: `4cb9c3cfb87b13cd63dc43c352db88322bbfca2198c3a6784d86b1c91d813de7`
- endpoint_runtime.py: `cd6d1213a0109b4818c87d189588cf2399010222681e11d58f23181c69a61fd2`
- tests/test_codex_cusp_endpoint.py: `a6db6eca69bc289a1b42b0a5a0c2bf243fe32ef0d2a016fd1450c271f104497c`

The exact before/after diff contains only the two required status checks, corresponding fixture fields and two refusal cases. Boundary reports ten controls passing in 2.68 seconds, retained in `/tmp/task14-endpoint-controls-004.txt`; this reviewer inspected those controls without rerunning them. No mathematical source or tolerance changed.
