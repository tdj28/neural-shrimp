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

(filled in below as runs complete)
