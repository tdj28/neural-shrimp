# Independent numerical fold review at the period-18 window

**Pass within the numerical scope.** A separate objective-differentiation implementation supports the generic-fold derivative conditions at the selected boundary point of S2-28011-zoom:587. This strengthens the evidence for one boundary of the numerical shrimp candidate; it does not prove the whole shrimp or an interval bifurcation.

The review used the existing second-order Arb-jet objective differentiation at128bits, on consumed data only. It evaluated the returned point and the two fixed finite-difference resolutions18 and20 at **two adjacent native phases of the18-cycle**, without another root solve, continuation, eigendecomposition or SVD. All18 full step Jacobians, monodromies, residuals, orbits and directional actions for every evaluation are retained.

Phase0/1 closure residuals are2.62e-11/3.86e-11; right-null and normalized left-null residuals are below7.33e-12. At the finer checked resolution, the independent coefficients are:

| Quantity | Phase0 | Adjacent phase1 |
|---|---:|---:|
| Parameter coefficient a | −0.502622140700 | −0.419226913337 |
| Quadratic coefficient c | −8.00011601031 | −9.59154886629 |
| Parameter second derivative −c/a | −15.9167600519 | −22.8791343331 |

These agree with the producer's five-resolution series. Their phase dependence is expected: the unit kernel vector changes scale by `k=||DF(x0)v0||=0.8340796790231029`. The transported right-vector discrepancy is2.75e-10, and the relations `a1=k a0`, `c1=c0/k`, and `s''1=s''0/k²` agree numerically. The coefficients cannot be compared across phases without this normalization.

Every proper-divisor separation1,2,3,6,9 remains above0.2067. Saved path/counter/null-normalization/finite-difference arithmetic also passes. Simple isolated+1 spectral evidence remains the producer's calculation: this review deliberately did not repeat a decomposition. The producer runtime lists three core source files, while its historical Git commit binds the remaining tracked tree; no metadata was rewritten to imply a fuller creation-time inventory.

One authorized execution at3c723630 used324 one-step AD calls in15.04s internally/15.27s externally, CPU/BLAS1, measured nice10. All561,615bytes of raw/launcher evidence are hash-checked and immutable; three owned PIDs are absent. No retry or additional target evaluation followed. The companion JSON binds the exact source/input/manifests and all retained arrays. Point ball arithmetic and converged finite differences remain numerical evidence, not a uniform interval proof of fold nondegeneracy, parameter continuation, cusp, chaos, ancestry or utility.
