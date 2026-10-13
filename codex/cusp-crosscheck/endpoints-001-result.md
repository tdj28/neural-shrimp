# Task14: independently checked period-7 fold endpoints

Both supplied endpoints pass the fixed numerical checks. The quadratic term changes sign while the cubic coefficient is negative at both points, supporting a candidate cusp segment. This has not located a cusp or established a nondegenerate two-parameter unfolding. It is a statement about the specified expected-embedding feedback map, not ordinary text generation or training utility.

| Index | log10(T) | Physical alpha | a (physical alpha) | b | c |
|---|---:|---:|---:|---:|---:|
| 13 | 0.1893809804983369 | 0.06661104169386665 | -0.568519872857 | -30.8400028845 | -580377.786519 |
| 14 | 0.1893794755174379 | 0.06660969640358283 | -0.527229797118 | 59.0602029593 | -531766.331527 |

These are the independently reconstructed saved coefficients. The fixed normalization is `||r||2=1`, `lᵀr=1`, with the largest-absolute right coordinate positive and one full period as the clock. The local parameter chart is `(log10(T), 10 alpha)`; the table uses derivatives with respect to physical alpha, including the alpha-dependent prefix. The orientation-invariant `b sign(a)` is +30.8400028845 and −59.0602029593, respectively.

The primitive period-7 closure residuals are 1.7533e−12 and 4.3011e−12. Critical multipliers are 1.00000000644798 and 1.00000000162234; complementary spectral radii are 0.0001603701 and 0.0001632041. These are numerical +1 classifications within the frozen tolerances, not exact fold certificates. Direct-versus-composed phase agreement and dense-versus-JVP checks pass. Retained left/right eigenpair residuals and homological equations are independently checked; full unretained eigenvectors remain supported by the authenticated original diagnostics.

All 14 coefficient finite-difference rows, 14 probe rows, six alpha rows and all 26 precision gates remain in the audit. No plateau or favorable rows were selected. Coarse offsets are inaccurate and are retained: at relative h=0.01 the row14 quadratic estimate is −90.825 rather than the positive AD value +59.060. The frozen finest-two-row and reduction tests pass; at h=1e−5 relative quadratic discrepancies are 2.1302e−5 and 1.3259e−5, and cubic discrepancies are 9.5514e−7 and 4.0409e−7. Near-multiplier cubic differences are 4.2034e−10 and 1.0489e−10 relative.

Original endpoint execution used the pinned ManualQwen float64/AD implementation, the authenticated complete row13 profile matrix, and a new complete 896-column row14 matrix. The original attempt completed in 2132.302 s, with peak 5,878,400 KiB. All 1,191 artifacts, 11 original Git source blobs and original process groups were authenticated before the separate saved-array audit.

The independent saved audit executed once from 24c69e37: 0.66272 s worker / 1.03317 s supervision, peak 75,696 KiB. It independently checked all 681 native arrays, 413 exact array associations and 474 scalar values. All 226 nonzero scalar differences are retained, with largest absolute difference 3.49246e−10; each is within the fixed absolute 5e−12 plus relative 5e−10 comparison tolerance. Original Git equality is inherited explicitly from accepted authority c9fa512d while all captured source bytes are rehashed; no Git, model, AD/JVP, eigensolve or inverse was run by the saved auditor. No target computation or audit was retried.

The full audit is `endpoints-001-independent-review-001/report.json`, SHA ee4d502d84b9ea16552d0123f08a0ba665e87917c28ff5b69a09c9526fff1b31. Output manifest 0be13094, supervision manifest cfb14b04 and closure c359529c bind it. The separate final source/closure authority has report 9db2eaa3 / manifest cbfa56e9 under the actual path `llm-cusp-crosscheck/final-assembly-002-witness-audit-001-authority`: the metadata adapter retained that legacy directory name, but its contents bind the correct Task14 audit. The original name is preserved, not relabeled as Task1 evidence. All original and audit processes/groups are closed.

The exact references, original source/receipt/card and all inherited historical asset/prefix limitations are in the compact final review JSON. Producer endpoint summaries had been disclosed before this retrospective independent check; no blinded confirmation is claimed. Period 28 remains unattempted. Final independent note and Root acceptance accompany the private release. `certificate_passed` and `cusp_qualified` remain false.
