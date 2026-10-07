# Resonance ancestry of the 1/5 window S2-28011:139 (exploratory, 7 October 2026)

Status: exploratory, following the mechanism consult's top recommendation
(`docs/consults/2026-10-07-mechanism-strategy/adjudication.md`) and the frozen
run card in `docs/research-log.md` (commit 992a7850, amended before any network
level run in 4fe96c9b). Code: `src/neural_shrimp/ancestry.py`,
`tools/explore_ancestry.py`, tests in `tests/test_ancestry.py`. Raw outputs:
`/data2/neural-shrimp-data/explore/ancestry/`.

## Question

Does the rule-selected window S2-28011:139 (period 10, rotation 1/5, at log10
rates hidden 2.389, output -0.330) descend from the 1/5 resonance of the
Neimark-Sacker curve of the parent edge-of-stability 2-cycle?

## Method

1. Controls (must pass before the network is interpreted).
2. Predicted roots: points of the period-2 Neimark-Sacker curve whose birth
   rotation equals 1/5, refined by bisection; onset, circle-invariance and
   rotation-detuning checks.
3. Approach levels: six cross-sections at distances 0.0128/4^k beyond the curve
   along its normal, linked by jump-free continuation of the period-10 orbit;
   on each, both fold edges (multiplier +1) by existence continuation and
   bisection; centre, width and amplitude relative to the parent 2-cycle.
4. Connection: the tongue's centre line tracked from the top level towards the
   window, testing whether the window's certified orbit lies on the tracked
   branch.

## Results

### Controls (pass; `controls-002`)

| Control | Expected | Outcome | Key numbers |
|---|---|---|---|
| Resonant normal form, 1/5 tongue rooted at (mu, theta) = (0, 1/5) | pass | pass | six fold-bounded levels mu = 0.04/4^k; widths 1.29e-3 to 3.9e-8 (ratio 8.07 per level = 4^1.5); amplitude 0.211 to 0.0063 (ratio 34); centres 0 |
| Same branch scored against a root displaced by 0.01 | not pass | alternative_ancestry | extrapolated centre -0.01 |
| Logistic period-5 window with a contracting coordinate, scored against its own fold | not pass | alternative_ancestry | one fold edge only; amplitude 0.506 to 0.503 (ratio 1.006): no collapse onto the parent |

The first control run labelled the logistic case exhausted_budget; the
classifier was amended before any network run so that resolved, non-collapsing
cases without two fold edges are alternative ancestry (commit 163ac396).

### Predicted 1/5 roots and onset checks

| Root | log10 (hidden, output) | Birth rotation | Complex pair, others inside, transverse | Radius exponent (frozen band 0.3-0.7) | Invariance residual converges | Rotation intercept minus birth angle |
|---|---|---|---|---|---|---|
| Right | (1.6990788, -0.1155545) | 0.19999997 | yes (third modulus 0.972) | 0.503: pass | yes (residual/radius about 5e-5) | 1.0e-6 (slope -0.73) |
| Left | (1.8035245, -0.3921629) | 0.20000001 | yes (third modulus 0.979) | 0.83: fail as frozen (0.51 from the two smallest detunings, post hoc) | yes (about 3e-3 to 9e-3) | -1.9e-5 (slope -5.25) |

### Approach levels (criterion 4)

**Attempt 1** (frozen continuation: straight segments between levels).
Right root: fold-bounded levels at delta 0.0128 (centre 0.0365, width 0.0459)
and 0.0032 (0.0111, 0.0135); continuation to 0.0008 stalled (step underflow
at t = 0.32): **exhausted_budget**. Left root: no locked period-10 cell on the
top cross-section (0 of 601 cells over +-0.03 decades): **no_qualifying_window**.

**Attempt 2** (disclosed amendment 19db8e3e: same target, levels and criteria;
centre-line tracking between levels). Right root, all six levels with two fold
edges (multiplier within 2e-4 to 2e-3 of +1), verified primitive attracting
period-10 orbits and jump-free tracking:

| delta | centre (decades from the root's foot) | width | amplitude vs parent 2-cycle |
|---|---|---|---|
| 0.0128 | 0.0365 | 0.0459 | 0.312 |
| 0.0032 | 0.0111 | 0.0135 | 0.143 |
| 0.0008 | 1.79e-3 | 1.02e-3 | 0.0539 |
| 0.0002 | 4.13e-4 | 1.15e-4 | 0.0253 |
| 5e-5 | 1.01e-4 | 1.39e-5 | 0.0123 |
| 1.25e-5 | 2.50e-5 | 1.70e-6 | 0.0061 |

Successive width ratios 3.41, 13.2, 8.89, 8.23, 8.17 (asymptote 8 for q = 5);
centre ratios 3.30, 6.18, 4.34, 4.08, 4.05; amplitude ratios 2.19, 2.65, 2.14,
2.05, 2.03 (overall 51).

**Frozen classification: alternative_ancestry.** The frozen estimator (linear
fit of centre against delta across all six levels) extrapolates to 1.53e-4
decades, above the 1e-4 tolerance; widths shrink, the amplitude criterion is
met and the deepest centre lies within tolerance.

**Post hoc (does not replace the frozen outcome):** the top two levels lie
outside the asymptotic regime (curved centre line). Linear fits over the
deepest three and four levels give -1.6e-6 and -1.6e-5 decades; log-log
exponents over the deepest three levels are 1.518 for width (theory
(q - 2)/2 = 1.5) and 0.513 for amplitude (theory 0.5). Numerically, a
fold-bounded 1/5 locked region narrows onto the predicted root with the
scaling of a Neimark-Sacker resonance tongue.

Post-hoc probe at the left root with delta 0.0008: again no locked cell
(no_qualifying_window); inconclusive (this branch's circle grows quickly and
its tongue may be thinner than the scan spacing).

### Connection to the window (frozen)

Tracking the tongue's centre line from the top level towards S2-28011:139
kept two fold edges up to log10 hidden about 1.84 (width 0.109); from there
the left edge was absent within 0.2 decades on most sections and the orbit
was unstable; the track re-found two edges at a few sections (widths
0.10-0.17), its heading turned away from the window, and continuation failed
at step 29 near (2.037, -0.101): **exhausted_budget** (window not reached).

Post hoc: continuing the window's own certified orbit straight towards the
four nearest tracked points, its branch ends at folds (multiplier within
3e-4 to 9e-4 of +1) 0.10-0.22 decades from the window, before reaching the
tracked tongue. Straight-path branch connectivity is therefore not shown.

## Outcome for S2-28011:139 (1/5)

Not established under the frozen rules. Near-root ancestry at the right root:
alternative_ancestry by the frozen estimator (a 53% margin over tolerance),
with post-hoc evidence that the tongue is rooted at the predicted point;
connection to the window: exhausted_budget; left root: onset supercriticality
failed as specified and no qualifying locked cell. The window itself is
enclosed by fold edges in the direction of the tracked tongue.

## Caveats

- Floating-point evidence with Newton certification (tolerance 1e-8 for the
  period-10 orbits), not a computer-assisted proof.
- The frozen root-mismatch estimator mixes non-asymptotic and asymptotic
  levels; a confirmatory version should prespecify an asymptotic-regime
  estimator (for example levels whose successive width ratios are within 20%
  of 4^((q-2)/2)) or start deeper.
- The centre-line tracker is fragile where the existence region widens (one
  edge beyond 0.2 decades) and the orbit is unstable; two-parameter fold-curve
  continuation would be the right tool for the connection stage.
- One window, one network, one seed: a case study, not a prevalence estimate.

## Readiness for a frozen confirmatory test on fresh losses

The near-root tongue test is close to ready: it separates the controls, and at
the right root it resolved six fold-bounded levels with the textbook
exponents. Before freezing it for fresh losses (reserved seeds, one loss per
seed): (1) prespecify an asymptotic-regime estimator for the root mismatch
(levels whose successive width ratios lie within 20% of 4^((q-2)/2), at least
three of them); (2) choose each root's starting detuning from its measured
circle radius (for example where the radius falls below 0.05), since the left
root's circle grew too fast for the fixed 0.0128 start; (3) keep the
centre-line tracking between levels. The connection-to-window stage is not
ready: it needs two-parameter fold-curve continuation, and its outcome should
be reported separately from tongue attachment.
