# Independent EXP-004 results audit — 11 September 2026

**Conclusion:** the retained numerical evidence supports a reproduced
shrimp-shaped candidate window and three numerically verified attracting
neural periodic orbits. The manuscript's reported EXP-004 periods, closures,
Floquet values and behavior summaries agree with the evidence at their shown
precision. No fatal model, recurrence-screen or orbit-certificate defect was
found. This is an audit of saved results and executable mathematics, not an
independent rerun of the target dynamics or an interval proof.

## Scope and immutable inputs

The audit used `evidence/EXP-004/attempt-001/` in this checkout. It ran:

```sh
python tools/run_hopfield.py verify --output evidence/EXP-004/attempt-001
```

That verifier passed every artifact hash and reclassified the saved peak events
for both grids, the complete transect, and all eight refinement records. A
separate audit checked the full file inventory: exactly 475 distinct paths are
listed in the receipt, all 475 are present, and no unlisted result file occurs
beside the receipt. Source revision is
`b670086775bbc07e115de730b8fcb30ca41cce6b`. Git object comparisons confirmed that
the native integrator, run controller, orbit diagnostics and EXP-004 source
plan still have exactly their frozen source bytes.

| Artifact | SHA-256 |
|---|---|
| Receipt | `3424b9eaa388e58cb4d8b91b2a4f0136d701fb6928e02d1a2a30ed2e52051c07` |
| Selected results | `0a92e4ce1446cee45038e19037131cfe653b28bce4666afb8dd1dbe51332d86f` |
| Runtime manifest | `a300c6043f1084e12e7c701766d14aa537ffdeb14e22f5d4ef510a0ad4d61205` |
| Frozen source plan | `8c618745ce6bc8101990c81dcbb1cb8e4674e19f97bff8b0a026f9cc8bc410ec` |

The receipt field `manifest_sha256` identifies the **source plan**. The runtime
manifest was reserialized with different JSON whitespace; its own byte hash
is correctly listed under `receipt.artifacts`. The two files parse to exactly
equal JSON values. The differing hashes do not indicate changed parameters.

The manuscript snapshot inspected for this audit had SHA-256
`04c235eb4c8fac60d2e9b7afd9441b10f8197b8299c9cd0587defb0ff976ba80`.
Subsequent manuscript edits and EXP-005 results require their own checks.

## Census and selection

Counts were independently obtained by loading every saved `period` array:

| Set | Evaluations | No short period | Nine peaks | Eighteen peaks | Other positive periods |
|---|---:|---:|---:|---:|---:|
| Overview | 6,561 | 5,955 | 317 | 15 | 274 |
| Focus | 32,761 | 26,948 | 5,146 | 306 | 361 |
| Independent transect | 201 | 120 | 75 | 3 | 3 |

The two grids total 39,322 evaluations; adding the transect gives 39,523. The
figures keep every category, including unresolved cells. These are evaluation
counts, not a count of independently sampled basins or a statistical sample of
neural architectures. A positive grid label is a finite recurrence candidate,
not an independently shot orbit at that pixel.

All four selected coordinate/period records match their stored refinement
arrays. Their coarse/0.01-step/0.005-step peak sequences are respectively
`9/9/9`, `9/9/9`, `18/18/18`, and `0/9/9`. The last sequence invalidates the
intended exterior interpretation. The controller correctly refuses to issue
that point a certificate, preserves its results, and does not substitute
another comparison. The outer-component point is at alpha 1.08, the focus
rectangle's right boundary; it is not evidence of a located limb endpoint.

## Orbit certificates re-derived from stored arrays

For each successful certificate, the audit independently constructed
`-x + W @ tanh(x)`, recomputed eigenvalues/eigenvectors of the saved full
monodromy matrix, identified the multiplier closest to one, checked its flow
alignment and residual, and recomputed endpoint closure from the stored orbit.
All saved gate values pass their frozen thresholds. Interior maximum-event
counts agree with the stated primitive peak count, and no retained interior
maximum returns within the primitive-return threshold.

| Role | Period | Relative closure | Neutral multiplier | Dominant transverse multiplier | Small transverse multiplier |
|---|---:|---:|---:|---:|---:|
| Core | 37.636375381794 | 7.4581564e-10 | 1.0000000146214152 | 0.06506009839388437 | 6.1482166e-11 |
| Outer | 36.974020999924335 | 1.4580997e-9 | 1.0000000014349841 | -0.4651391442717248 | -7.8076518e-12 |
| Doubled | 74.48169920951436 | 9.6850356e-11 | 1.0000000804879825 | 0.6561213334501104 | -3.4545087e-19 |

The neutral flow residuals are approximately 4.29e-8, 1.03e-8 and 2.31e-8,
all below 1e-6; the alignments round to one and exceed 0.99. Dominant
transverse moduli are all below `1 - 1e-5`. Minimum normalized distances to
an interior maximum are approximately 0.5741, 0.5722 and 0.0006330,
well above the 1e-7 primitive-return threshold. The retained neutral multiplier
is essential: applying an all-multipliers-below-one rule to an autonomous flow
would misinterpret these cycles.

The nearly zero multiplier signs and magnitudes are not precisely resolved.
As an independent algebraic check, Simpson quadrature of `trace(J)` along the
stored orbit and Liouville's formula gives a positive doubled-orbit determinant
about 2.58e-23, whereas the direct product of its numerically computed
eigenvalues is about -2.27e-19. Its monodromy condition number is about 7.0e20.
This diagnoses loss of resolution of the very small mode; the dominant
transverse multiplier is about 0.656, far from the stability threshold. The
manuscript already warns against interpreting tiny signs/final digits and
must retain that warning. These calculations do not supply rigorous error
bounds on the variational solution.

## Behavior and quantitative prose

For all four saved probes, the audit exactly re-derived `tanh(states)` and
`tanh(perturbed_states)`, both Euclidean separation arrays, their maxima, and
each neuron's population mean, standard deviation, minimum and maximum over
the declared final-half window. Every result matched the saved JSON values.
The first behavior state exactly equals the final state of that selection's
0.005-step refinement. No new trajectory was integrated.

The following core values support the manuscript's rounded statements:

- Potential-3 minimum/maximum over the summary window:
  -16.196561861534647 and 16.196483703473742.
- Potential-3 standard deviation: 8.66362635821824.
- Activation-3 standard deviation: 0.9876173038401254.
- Maximum same-time state separation over the 300-unit probe:
  8.53380530384565e-5 following the prescribed approximately 1e-6 perturbation.
- Recorded one-period state RMS: 8.150655023427055e-10, rounding to 8.15e-10.
  Its square equals the sum of the three recorded per-neuron RMS squares,
  within floating arithmetic.

The one-period RMS has a narrower audit status than the other quantities. The
frozen source computes it from DOP853 dense evaluations at 2,248 pairs of
times `t` and `t+T` in the final half of the probe. The resulting summary is
receipt-bound, and the formula and aggregation are correct. However, neither
those residual arrays nor the dense-output coefficients were retained, so the
1e-9-scale number cannot be independently re-derived solely from the saved
0.05-spaced state samples without another numerical integration. No such
integration was run. It should remain a recorded diagnostic, not be presented
as a separately reconstructed raw-sample measurement or an error bound.

The variance figures 8.664 and 0.9876 summarize times 150 through 300, not the
entire displayed 0-through-300 trace. The manuscript author was asked to make
that window explicit. Same-time perturbation separation includes phase shift
and transient amplification; nearest sampled-orbit distances have a sampling
floor. Neither is an exact transverse recovery measure. The displayed
activation-variance color range is narrow and explicitly labeled.

The receipt's 140.5375414993614 seconds rounds to 140.54; the four-worker and
zero-paid-cost statements match its recorded execution fields. No claim of
additional paid compute was inferred.

## Mathematical and interpretation review

The native and independent Python implementations use the same published
three-neuron equations and the correct column-scaled activation Jacobian,
`J = -I + W diag(1 - tanh(x)^2)`. Native RK4 advances states and tangents
consistently. Peak states use a common cubic-Hermite event fraction. The
recurrence screen checks every state component over repeated event sequences;
shooting separately rejects stationary roots, wrong peak counts, multiple
covers, failed neutral modes and transversely unstable solutions.

No fatal defect was found in those mechanisms. Their remaining limits are
material: independent floating-point numerical methods are not interval
proofs; a one-vector finite-time exponent is not an asymptotic chaos proof;
short recurrence labels can change with settling time and step size; the
representative refinement changes both; basin independence is not established
by re-integrating a captured terminal state; and exact bifurcation branches
have not been continued. The candidate map plus selected certificates supports
the manuscript's **numerical reproduction** wording, not a claim that every
colored pixel or the exact shrimp boundary topology is certified.

Componentwise tanh is mathematically injective on finite real states, so the
complete exact activation waveform has the same primitive period as the
complete state. Floating saturation can obscure numerical visibility. Neither
this correspondence nor the measured periodic regimes establishes training,
latent-reasoning, task-correctness, or optimization benefits. The manuscript's
application paragraphs state these as conditional tests rather than results.
