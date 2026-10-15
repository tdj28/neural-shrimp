# Task 21c prospective source and saved-scalar review — draft

Review scope: FNF2b/FNF4, no model or producer execution and no new target arithmetic. This draft is awaiting the original dependency/provenance supplement before final release. Current source export commit4d5a481a1cbe6d463baf87b7ef0991c81d50b39b; tools/flip_normal_form.py SHA35962e53d6655052b055c5ddfe1922b863367cbe890d590675ca49a907e16767, tests/test_flip_normal_form.py SHA2d8f0594a0e8a0aa750c474b5af390027025c06b72fc852fe66a413f5a56c8fb. Existing completed Task27 inputs p2 SHA4f90f9b84340537c651afd987ec4bd6430add7d0c52461b1493805ee5329b1c8, p4 SHA223ae029043af578db5eb740ca3284d7abe4562f49123cadc3fdbd2518dcf839.

## What can be independently recomputed

Each of four saved coefficient rows gives direct term, centre-manifold term and their sum. The source selects the first adjacent pair with the smallest absolute coefficient difference, averages its two coefficients, and reports the absolute pair difference divided by max(abs(mean),1e-300). That exact deterministic calculation can be independently reconstructed. Selection is absolute-difference based, not relative-difference based; no plateau-quality rejection threshold is applied. All four rows and all five parameter offsets per orbit must be retained, including offsets omitted from the paper.

At an offset, beta=-Re(mu_p)-1; when beta*c>0, eta_predicted=sqrt(beta/c), ratio=eta_measured/eta_predicted, mu_2p_predicted=1-4*beta. Parameter alpha=alpha_flip+offset_side*offset uses physical alpha. The factor10 in theta's steering coordinate is internal to the model family; no extra factor10 is appropriate in these saved-scalar identities because beta is derived from the actual parent multiplier and c from state derivatives.

The tool removes r,l from its result and saves no dense monodromy, derivative vectors, h2 vector, parent state, doubled states or original projections. Thus one can reconstruct c from the saved contractions and pair selection, but cannot independently rebuild its finite differences or linear solve. Likewise eta_measured=abs(lᵀ(xa-xb))/2 and centre_offset are producer scalar measurements, not newly reproducible state projections from this JSON. The finite accepted audit should state this distinction without requesting a new model run.

## Mathematical conventions

For a simple real multiplier -1, with ||r||2=1 and lᵀr=1, the coefficient

c=lᵀC(r,r,r)/6 + lᵀB(r,h2)/2, (I-M)h2=B(r,r)

has the correct sign and factor convention. The inverse is I-M, not the fold's singular bordered operator or a cusp gauge constraint: the critical eigenvalue is -1, so the critical direction has eigenvalue2 in I-M. In the scalar map -x+q*x²+g*x³, this formula gives c=g+q², agreeing with the source control. Reversing both r and l leaves c unchanged; an arbitrary rescaling without the unit-norm convention would change its numerical units.

For the source's finite-difference construction, differentiating J(x)r in direction r gives B(r,r); the second central difference of J(x)r gives C(r,r,r); differentiating J(x)h2 along r gives B(r,h2). The relative step is multiplied by ||x|| (or1 if zero). Averaging a closely agreeing adjacent pair is a heuristic finite-difference plateau estimate, not a rigorous derivative-error enclosure. The use of the exact-flip formula at a numerically located near--1 crossing inherits the recorded crossing tolerance.

The scalar normal form eta↦-(1+beta)eta+c*eta³ predicts half-amplitude sqrt(beta/c) to leading order. Even the exact cubic truncation's doubled multiplier is (−1+2beta)²=1−4beta+4beta²; the saved 1−4beta is its leading-order approximation. It need not be accurate or retain scalar character at the largest tested offsets. The measured half-amplitude uses the flip's left eigenvector at every offset, so finite-offset coordinate and higher-order corrections remain expected.

Positive c supports a local supercritical flip when the parent multiplier crosses in the stated beta direction and the transverse complement is attracting. It does not guarantee full stability at arbitrary finite offsets. The source's dense crossing gate checks that complement at the flip; its offset data use partial Ritz spectra.

## Source gates and output semantics

flip_cubic chooses the dense eigenvalue nearest-1. It refuses if abs(imag)>1e-6, abs(real+1)>1e-6, or any other dense eigenvalue has modulus>=.95. It records real/imaginary components, modulus, argument in turns and the next-largest modulus before coefficients. It obtains left/right eigenvectors from Mᵀ/M and normalizes as above. Source-tests cover a scalar q² correction, two coupled examples and refusal of a fold; this review did not execute them again.

At doubled offsets, the reported mu_2p is the computed Ritz value nearest+1. It is not necessarily the dominant value, nor an eigenvector-tracked continuation of the centre multiplier. The full saved complex value must be retained. Comparing a real leading-order prediction with a complex pair is a finite model discrepancy, not a failed scalar arithmetic identity. Do not replace it with its modulus or silently select a different eigenvalue. Check the original dependency source for the separate meanings of ep.critical, next_modulus and half-period guards before final release.

Every failed original attempt and every offset error/no-predicted-orbit outcome is part of the coverage. The two successful saved files have four coefficient steps and five offsets each. Root's concrete input/receipt binding, accepted source and bounded own arithmetic are separate from this source-only review.
