# EXP-005 independent results audit

Post-run numerical and integrity audit, 11 September 2026. This audit reads
saved artifacts and performs array arithmetic; it does not integrate any
new target trajectory or change the run, thresholds or selection rules.

The prospectively frozen source is
`01209371d541aaa9d072a1addb3960aa3a74efd3`. The run is
`evidence/EXP-005/attempt-001`. Its receipt SHA-256 is
`fb287eb68541e674c8dc095049c3e0a4e887e3b7297f8903dda39a9797ad3aee`;
its manifest SHA-256 is
`a74b737df2b98e390ce25fc42868c46de9f9b60a1d07150e0b396c12195e6463`.

## Integrity and recorded sample

The frozen driver's `verify` command passed. Independently, all **21**
receipt-listed artifacts matched both their SHA-256 and byte counts; all
five recorded executable/plan source hashes matched the local source.
The eight point-step runs and four behavior probes are present. There is no
failure or timeout receipt. The run completed in 4.166616667 seconds, below
its 600-second ceiling, with no paid compute or review.

Every point and step starts from `(0.1,0.1,0.1)`. The recorded native step
counts match the 10,000-unit burn and 2,000-unit measurement; the exponent
uses 1,900 units after tangent alignment. The verifier checks complete-state
event recurrence, event ordering and retained counts, physical periods,
residuals, initial states and control coordinates. All eight checks passed.

These four points were selected after EXP-004 outcomes and frozen before
EXP-005. They are not an independent random sample of the parameter plane.
The lower and upper names refer to fixed transect endpoints, not a
numerically identified bifurcation boundary.

## Results reproduced from saved evidence

Here, a peak period counts distinct x3 maxima in the repeating full-state
orbit. `None` means no accepted recurrence up to 32 maxima; it does not mean
that an infinite-time aperiodic attractor has been proved. The two exponent
columns are step comparisons, not confidence bounds.

| Control | Peak periods, dt=.01/.005 | FTLE, dt=.01 | FTLE, dt=.005 | Independent certificate |
| --- | --- | ---: | ---: | --- |
| Core | 9 / 9 | -0.000410453 | -0.000410568 | Attracting period-nine orbit |
| Lower endpoint | None / None | 0.138663492 | 0.132212200 | Not attempted: no shared positive period |
| Upper endpoint | None / None | 0.122586260 | 0.135211404 | Not attempted: no shared positive period |
| Failed near-edge comparison | 9 / 9 | 0.001238627 | 0.001238622 | Attracting period-nine orbit |

Both endpoints satisfy the frozen finite-horizon irregular-support criterion
at both steps. This supports persistent irregular behavior over the declared
measurement, while retaining the possibility of longer transients or periods.
The near-edge point again becomes periodic and remains reported as a failed
irregular comparison. Its small positive finite-time exponent coexists with
a certified attracting autonomous cycle; a neutral phase direction and
finite-time alignment effects make the sign alone insufficient.

## Independent certificate arithmetic

For both accepted certificates, this audit recomputed closure from the saved
orbit endpoints, recomputed the section condition and curvature from the
declared neural RHS/Jacobian, and diagonalized the recorded full monodromy.
It checked every complex multiplier, neutral index, neutral error, alignment
with the flow, flow residual and transverse spectral radius against the
recorded values. All passed the frozen thresholds. There are eight retained
interior maxima plus the starting maximum, with no earlier return meeting
the primitive-orbit rejection threshold.

| Control | Physical period | Relative closure | Neutral multiplier error | Largest transverse multiplier modulus |
| --- | ---: | ---: | ---: | ---: |
| Core | 37.6363753816 | 8.54644e-11 | 1.23608e-8 | 0.0650600985 |
| Failed near-edge comparison | 37.3314995885 | 6.45727e-10 | 5.87643e-8 | 0.8560332413 |

Both transverse radii are below the frozen `1-1e-5` stability threshold.
The neutral multiplier is retained rather than mistaken for a contracting
mode. These are numerical certificates based on the independently integrated
recorded monodromy, not interval-arithmetic proofs. This audit recalculates
their arithmetic without repeating the target integration.

## Observable behavior and perturbations

Each probe contains all 6,001 declared times from 0 through 300 at spacing
0.05. From the raw potentials, this audit exactly recomputed both baseline
and perturbed tanh activations, same-time state and activation separations,
and every recorded perturbation scalar: initial/final/maximum separation,
tail RMS separation, final/maximum activation separation and finite-horizon
log amplification. The initial perturbation direction matches
`(1,1,1)/sqrt(3)` to floating-point addition accuracy.

It also exactly recomputed every state and activation mean, population
standard deviation, minimum and maximum on the declared `[150,300]`
summary window. For certified orbits, the nearest sampled-orbit distances
and their tail RMS summaries were recomputed from the stored orbit points.
All matched the recorded arrays and metrics. The separately recorded dense
interpolation cycle-repetition metric is bound by the receipt but is not
claimed to be independently reconstructed from the more sparsely sampled
behavior arrays in this audit.

| Control | Maximum state separation | Final state separation | Maximum activation separation |
| --- | ---: | ---: | ---: |
| Core | 0.000206532479 | 0.000016610533 | 0.000123215136 |
| Lower endpoint | 29.6734656602 | 9.8495107789 | 2.6400632425 |
| Upper endpoint | 31.8717845906 | 16.1374751630 | 2.7294464761 |
| Failed near-edge comparison | 0.000106563563 | 0.000032431661 | 0.000064407454 |

These are Euclidean norms across the three neuron coordinates, for the
same nominal initial perturbation norm of `1e-6`. They are maxima or terminal
values within the fixed 300-unit probe, not averages over perturbation
directions or a downstream task score. The large contrast is present in
bounded activations as well as internal potentials.

An attracting cycle does not imply that the raw same-time perturbation must
shrink monotonically: even the core reaches about 206.5 times its initial
separation while remaining small in absolute units. Local transverse return
and neutral timing displacement are distinct. Likewise, activation standard
deviation stays substantial inside the cycle; its distinctive feature is
repeatable temporal organization, not silent or constant neurons.

## Derived-summary review

`tools/summarize_hopfield.py` was reviewed and executed in verification mode.
It currently summarizes **EXP-004**, preserving its actual status: 39,523
grid/transect cells and three verified attracting representatives, with all
candidate-period classes counted and the failed exterior certificate retained.
The emitted summary matches `evidence/hopfield-summary.json` byte for byte
after canonical serialization. The cell total counts planned evaluations,
not independent attractors, and the verified-representative count does not
certify every periodic map cell. EXP-005's four controls and two verified
representatives remain separately recorded in its receipt-bound summary;
the two experiments' representative counts must not be added as distinct
attractors.

No numerical or integrity blocker was found in the saved EXP-005 evidence
or the reviewed EXP-004 summary derivation.
