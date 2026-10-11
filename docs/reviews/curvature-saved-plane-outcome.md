# Why the fixed curvature approximation failed

**Adapted exploratory description after the negative pilot.** The leading
two-step dynamics are not reliably contained in the two sharpest phase
curvature directions. This is measured on the same six development references,
not on large models or fresh confirmation data.

| Reference | Leading-vector energy in retained plane | Error of actual-map compression |
|---|---|---:|
| S2-28011-stable-real | 0.0092%, 0.0205% | 0.363458 |
| S2-28011-stable-complex | 95.3994%, 95.3994% | 0.049939 |
| S1R-28002-stable-real | 0.0120%, 0.0183% | 0.606924 |
| S1R-28002-stable-complex | 52.1400%, 52.1400% | 1.045828 |
| S2-28011-crossing-neighborhood | 83.2377%, 83.2377% | 0.231079 |
| S1R-28002-crossing-neighborhood | 29.7199%, 29.7199% | 0.702725 |

The fractions use squared Euclidean norm in whitened coordinates. Both real
controls retain less than0.021% of either actual leading right eigenvector.
The complex examples range from29.7% to95.4%; their retained planes have
nonzero leakage into omitted directions. A sharp-direction proxy therefore
misses almost all of the leading real dynamics and varying portions of the
oscillatory dynamics.

Even the target-informed compression Q^T M Q gives pair errors above the old
0.02 reference threshold at all six points. This is a descriptive comparison,
not a replacement predictor or a new engineering test: compression uses the
full target product and omits coupling to the complementary space. It does not
prove that no better coefficient choice or richer representation could work.

The exact four-term residual-action decomposition is retained with its
floating identity residual. Component norms can expose omission and cancellation;
they do not causally apportion eigenvalue errors. The computation took0.102s
and used no new model evaluations, derivatives or decompositions. All fixed
indices/points were retained; raw inventories/source bindings/PID exits verify.
Independent saved-arithmetic review is pending at this record.

Next research question: can directions selected by the two-step update dynamics
be recovered economically from matrix-vector actions, compared with direct
tangent-map diagnostics? Any such experiment should measure accuracy and cost
against strong ordinary baselines. This result does not justify another LLM
forecast sweep, a training intervention, or a claim of utility.
