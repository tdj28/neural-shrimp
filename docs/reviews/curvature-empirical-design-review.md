# Independent static review: fixed curvature panel

2026-10-08 UTC. Reviewed frozen design `51200a4a90a1533fd1ee798cdf83b907073d5ad1`;
this separate branch supplies a prospective scope-preserving amendment. **Design
acceptable with the corrections below; implementation, profile, numerical
qualification and launch review remain pending.** This is a source-only design
review, not an empirical or theorem verdict. No target array, fit outcome, norm,
map, derivative, eigensystem or SVD was evaluated.

1. **Closure accounting was incomplete.** Twelve phase images test transitions
between retained phases, not generally both actual two-step closures. The
amended budget reserves twelve primary Hessian/gradient acquisitions and at
most24 map values. Extra closure images require a map-only adapter or explicit
accounting for up to24 fused Hessian calls. Cache reuse needs exact input-byte
identity. Constructing S2's missing partner does not eliminate its phase1
closure check. The profile includes the additional map cost conservatively.

2. **The spectral-envelope wording overstated floating evidence and omitted
diagonalization residual.** For R_V=M V−V Lambda and invertible V,
V^(-1)M_0V−Lambda=V^(-1)(M_0−M)V+V^(-1)R_V. Thus the exact-arithmetic
comparison motivates rho=||R_V||_F/sigma_min(V)+kappa_2(V)*delta_op. The
amendment includes both terms, reusing one declared basis SVD. This is a
floating numerical diagnostic unless matrix formation, invertibility and all
rounding are enclosed. Numerical disks do not establish eigenvalue counts or
a rigorous pair enclosure; those claims remain pending interval validation.
The original .02/separation numerical screens remain, with honest evidence labels.

3. **Fit and controls are coherent when full spectra are retained.** The
largest-algebraic curvature direction and trace-mean bulk are fixed before fit
errors; no target multiplier enters the fit. The aligned spectrum is
(a-alpha)(b-beta) plus n−1 copies of ab. The orthogonal spectrum is
b(a-alpha), a(b-beta), plus n−2 copies of ab. Applying the same leading-pair
rule to both prevents hidden removal of bulk modes. The stronger per-reference
baseline is correctly labeled an oracle comparator. Orientation can affect
pair character/angle but receives no credit for the complex modulus fixed by D.

4. **Leading-mode selection is not automatically pair qualification.** The
historical real/complex strata use the first multiplier's imaginary part. The
amendment says so and preserves the already-declared pair check: two real
modes or a conjugate pair, otherwise unresolved without substitution. A useful
implementation control is a real leading eigenvalue followed by one member of
a conjugate pair; it must not silently select a different pair.

5. **Cost claims remain proposed, not measured.** Hermitian ||K||_2 uses existing
eigenvalues. General eigensystem calls return matched left/right vectors; V
conditioning and residual diagnostics reuse the single retained basis SVD.
The declared six slots, exact input identities, accuracy thresholds and resource
caps are unchanged. The profile must charge extra closure work and cannot use
fit quality for eligibility. A cap failure remains a retained unresolved unit.

Static checks compared the amended inventory with its parent: all six panel
entries, input/source inventories, training-data bindings, selection disclosure
and other identity fields are unchanged. Only operation-accounting metadata and
the corresponding missing-closure description changed. No prospective outcome
was inspected and no synthetic numerical control was run in this review.
Before implementation approval, independently test the residual term on a
perturbed numerical eigenbasis, aligned/orthogonal multiplicities, ambiguous
leading pairs, exact-byte cache behavior and all operation/resource counters.
