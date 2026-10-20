# Task40 independent DELTA2D, ditch and cascade reductions

Reference article: bb42a82189da91c37f50804c077832df9e2237ae.
This component supports the combined review; it is not a separate proof.
All new calculations use saved raw files, one numerical CPU thread and
explicit bounded runners. No model, Newton continuation, Jacobian evaluation,
old control suite or original experiment was rerun.

## DELTA2D

`delta_ditch_audit.py --section delta` was checkpointed before execution at
21e628bf28cfb77be54359198de523d4e683b240. `delta-001` exited 0 without timeout,
direct child 1605552 reaped, parent interval 4.992047008126974 seconds.
The input manifest hashes the exact four grid and four Ritz archives used.
The implementation uses SVD bases and projection-Frobenius plane overlap,
rather than importing the producer's QR/principal-angle implementation.
It deliberately uses the declared seed-selection objective and traversal
order; this is independent implementation, not a different selection design.

| Quantity | Independent result |
|---|---:|
| Solved and tracked cells | 852 of 999 |
| Overlap below 0.9 | 38; minimum 0.5011576615432117 |
| Sign agreement | 849/852 |
| Disagreement indices | (32,5), (33,5), (34,5) |
| Real leading sign-change edges / complex-leading cells | 42 / 42 |
| Trace-zero / determinant-zero edges | 64 / 32 |
| Real edges within one cell of trace edges | 42/42 |
| Trace edges within one cell of real edges or complex cells | 60/64 |
| Real edges near determinant edges | 9/42 |
| Trace saddle (alpha, log10T) | (0.07318804087584509, -0.15208331868800173) |
| Trace saddle value | 0.044884483163678275 |
| Trace saddle cell-Hessian eigenvalues | -0.0008014283771285446, 0.003530097047080921 |
| Trace fit RMS | 0.00038961907559026404 |
| Leading-real saddle (alpha, log10T) | (0.07319482971306128, -0.1521384898381376) |
| Leading-real saddle value | 0.06736198957428159 |
| Determinant stationary candidates under the declared fit/filter | 0 |
| Maximum saved Newton residual | 9.97632128858313e-13 |
| Solved cells with different seed token loop | 0 |

The quoted numbers reproduce. The determinant test preceded outcomes, the
leading-multiplier fit was added before outcomes, and the trace analysis was
added afterwards; the article discloses the determinant/trace distinction.
No absence theorem follows from zero fitted stationary points on this finite
mesh. The saddle is a 5x5 quadratic fit, not a directly solved or certified
stationary point. The saved grid lacks Ritz residuals, so this independently
checks analysis of the stored Ritz pairs, not their original generation.

The chosen pairs are (0,1) in 723 cells, (0,2) in 63 and (1,2) in 66.
Thus “two leading directions” is not literally the two largest-modulus
Ritz directions everywhere. Better wording is “a two-dimensional block,
initialized from the leading Ritz directions and tracked by subspace
overlap.” The paper already discloses mixing and the three sign exceptions;
Table VIII should retain the finite-patch, retrospective qualification.

## DITCH, DITCH2D and local/GPU DITCHFINE

The first `ditches-001` calculation stopped at its exact-axis equality gate.
It exited 1, child 1621826 reaped, no timeout, parent interval
0.1655455301515758 seconds. The empty `ditches-evidence.json`, stderr and
receipt are retained. No numerical disagreement was inferred from this stop.

The disclosed follow-up `ditches_followup.py` was privately checkpointed at
52eb95f1add11586ccc5ccd53244f12827bc4098 before `ditches-002`. It exited 0,
child 1624757 reaped, no timeout, parent interval 0.16539147682487965 seconds.
It keeps both exact hexadecimal axes and compares corresponding cells only
after a one-ULP correspondence check. This is not a silent relaxed gate:
the initial identical-input comparison remains failed.

The CPU and GPU alpha axes are identical. Temperature rows 2 and 3 differ
by one ULP, 2.7755575615628914e-17, because the intended lattice was split
into different bands. Both have 225/225 solved cells, no complex leading
pair, the same seed tokens and positive leading multipliers.

| Quantity | CPU | GPU |
|---|---:|---:|
| Minimum leading multiplier | 0.05657524175344515 | 0.05657524175350294 |
| Maximum leading multiplier | 0.07466856505923584 | 0.07466856505840594 |
| Maximum Newton residual | 1.0995865167741012e-13 | 8.371650216477033e-13 |
| Minimum on alpha=0.07318 | 0.06380835308215252 | 0.06380835308231905 |
| Location of that column minimum | log10T=-0.152028 | log10T=-0.152028 |

Every row has its minimum at the smallest alpha column. Across corresponding
cells, maximum leading-multiplier difference is 6.5533828363939506e-12
(median 2.562117185078705e-13, maximum relative 1.0264784735306801e-10);
maximum next-modulus difference is 3.7638295258268784e-13. These agree at
the article's precision on the same intended lattice, with the exact-input
qualification above.

Other article numbers reproduce:

- Body grid: 529/961 solved, nine complex-leading cells, 45 real sign-change
  edges, minimum modulus 0.006319057939216359 at
  (0.07305333333333335,-0.1512). The completed part is not all attracting:
  its maximum modulus is 5.63056; the article's attracting assertion must
  refer specifically to the sign-change neighborhoods or finer patch.
- DITCH2D: 595/672 solved, 33 complex-leading cells, 41 real sign-change
  edges, maximum modulus 0.934887597179662, maximum residual
  9.996854267046756e-13, no seed-token changes.
- Line A: 26/26 solved; all positive; minimum 0.06383893289865969 at
  log10T=-0.15202, maximum next modulus 0.02031258120402277.
- Line B: 21/21 solved; one complex-leading cell; real part spans
  -0.6050958204145324 to +0.7101417840588771. The sign transition crosses
  a complex-leading cell, so it is not counted as a directly adjacent
  real-to-real sign-change edge.

**Required wording correction, main.tex 1408–1412:** the claim that the
continued orbit is also the fixed-start attractor throughout the patch is
not established. Nearest HR10 scan cells have period six at 593/595 patch
points and at all 26+21 line points. The two exceptions have unresolved
period labels, not a verified alternative period. They are near
(0.073288,-0.15202) and (0.073234,-0.15174), with nearest-cell offsets
0.4172 and 0.2593 scan steps. These are neighboring parameters; they do
not directly refute same-parameter orbit identity, and even matching token
loops/periods do not establish full-state identity. Say instead that the
continued attracting orbit keeps its six-token text and nearby fixed-start
scans predominantly find period six. Exact same-orbit identity would need
a separate full-state comparison, which is not silently added here.

## Cascade

`cascade_audit.py`, privately checkpointed at 41604b46, ran once as
`cascade-001` at source a0ddde1bd77bf969f939ad9c624e3b9979d8773a.
It exited 0 without timeout; child 1620818 reaped; parent interval
0.16559561993926764 seconds. Original source/metadata and failed attempt
records are retained in `cascade-evidence.json`.

All nine selected crossings have real critical multipliers near -1 and
their independently selected next moduli match the original records.
The largest critical error among the first eight is 4.2739697780902475e-8;
the ninth is 4.071930868043694e-7. The article explicitly restricts its
5e-8 bound to the first eight. Newton residuals and counts are retained
per row: P96 and P384 were accepted with zero additional Newton updates
under the producer's period-scaled residual criterion; their residuals are
8.353113925687855e-12 and 2.861273870913938e-11. They must not be relabeled
as converged at an unscaled 1e-12.

Ratios:
14.928070812471404, 3.1241162383413608, 4.498762888768369,
4.620764877916609, 4.661793820964229, 4.667314947436724,
4.66884286414822.
The last differs from Feigenbaum's constant by 0.007683218348738061%.
Aitken gives 4.669427488953075. Perturbing the P384 location by the disclosed
plus/minus3.3e-13 reproduces the article's displayed ranges. The independently
fit original failed P384 attempt has maximum fit residual equivalent to
3.2372056708237196e-13 in alpha; this is an empirical fit uncertainty, not
a certified bound. The second cut starts 2.8438243,2.8593398; the 1.5B
sequence ends 4.664759946074821. These quoted values pass.

**Required wording correction, main.tex 1102–1103:** after the first
ratio, the signed deviations are negative, but their successive shrink
factors start 9.065342474574875, then 3.518790722660126,
6.538622633244229, 3.9263998793277577 and 5.259061183104772.
The text's “after the first ... 3.5 to 6.5” omits 9.07. Use the explicit
late-index restriction in Table VI (“after n=4”), or report the complete
range as approximately3.5–9.1. The last factor remains unresolved under
the stated location sensitivity.

The three finite-time scans contain1600/1360/1000cells,3000transient steps,
256recorded steps and period cap48. The first detected fork is
8.3641e-6 before its Newton location; subsequent detected-fork offsets have
magnitude at most1.0995e-6. The largest chaos-classified alpha is
0.0652777986241401. These agree with the article's rounded scales.
The scans do not prove asymptotic chaos or that a finite-sample exponent
is exactly zero at each mathematical crossing.

The P384/P768 raw files save action-based Ritz residuals
1.2512860171115708e-15 and4.443441535129896e-15. The research-log statement
that longest-product eigenpair residuals were “not done” is stale. Do not
insert that incorrect limitation into the article. These residuals measure
the computed chain action; they do not bound its exact-real model error.
Tiny next eigenvalues remain unresolved below the numerical product floor.
No genericity coefficients, independent operator replay or new dense product
were computed in this review.
