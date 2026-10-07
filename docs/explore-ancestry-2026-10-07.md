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
