# A3b consequence pilot (exploratory, 7 October 2026)

Stream summary for branch `claude/explore-consequence`. Every result here is
**exploratory** under the [adaptive research protocol](adaptive-research-protocol.md);
run cards and findings are in the [research log](research-log.md).

## Question

Beyond the edge of stability, full-batch gradient descent with per-layer rates
can sit on an ordinary period-2 orbit, on an invariant circle of the two-step
map (quasi-periodic training), on a phase-locked tongue orbit, or on a chaotic
attractor. Does this regime structure have a training-relevant consequence, and
does rotation-number or locking information predict anything beyond the
largest finite-time Lyapunov exponent (FTLE) and sharpness?

## Design

Code: `src/neural_shrimp/regimes.py` (features, regime rules, nested-model
tools; unit tests in `tests/test_regimes.py`), `tools/explore_consequence.py`
(simulation) and `tools/analyze_consequence.py` (analysis). Outputs under
`/data2/neural-shrimp-data/explore/consequence/`.

For every cell of a learning-rate slice (output rate fixed, hidden rate varied):

- **Reproducibility.** The base run from the system's initial parameters and 30
  copies perturbed at initialization (sizes 1e-12, 1e-10, 1e-8, 1e-6, 1e-4;
  six random unit directions, seeds 28700-28705) are iterated 40,000 steps;
  parameter and function-space disagreement (RMS prediction difference on
  held-out inputs, divided by the target RMS) is recorded at 2k, 10k and 40k
  steps. A "slope" summarizes how final disagreement scales with the
  perturbation size (1 = linear response, 0 = saturated).
- **Two-phase schedule.** Base and copies are then annealed (rates x0.1 for
  3,000 steps, then x0.01 for 1,000) and their final functions compared.
- **Tail.** The base run continues 8,192 steps: training loss, held-out MSE
  (S2 only; S1R has random labels), two tangent vectors (two FTLE estimates),
  two copies started 1e-10 away on the attractor (chaos qualification), the
  averaged iterate, preconditioned sharpness lambda_max(P^1/2 H P^1/2) at 16
  states, oscillation amplitude (RMS distance of states from their mean), and
  the last 4,096 states (rotation number of the two-step map; recurrence
  screen with the EXP-028 tolerances).
- **Stopping-time sensitivity.** The schedule's anneal is started from each of
  the last 64 consecutive tail states; the RMS spread of the 64 final functions
  and the number of distinct final functions (tolerance 1e-4 of target RMS)
  are recorded.

Regime rules (`neural_shrimp.regimes.classify`, fixed before the analysis):
period-2 and locked cells come from the recurrence screen (locked = even
period at least 4 whose two-step points make at least one net turn); a
nonperiodic cell is quasi-periodic when both FTLE estimates are below 1e-3,
the rotation numbers of the two tail halves agree within 5e-4 and on-attractor
copies stay within 1e-6; chaotic when both FTLE estimates exceed 5e-3 and
on-attractor copies separate beyond 1e-6; otherwise irregular.

Incremental value: leave-one-block-out (25 contiguous cells) least squares.
The base model has FTLE (continuous parts and sign classes at the regime
thresholds), sharpness, log10 rate and log10 amplitude; the full model adds
locked and quasi-periodic indicators, the rotation number, log10 distance to
the nearest rational with denominator at most 12, and 1/q of a locked cycle.
Intervals are moving-block bootstrap percentiles (blocks of 25 cells).

## Results

### Regimes along the slices

| | S2-28011, output -0.0856 (750 cells) | S1R-28002, output 0.9413 (160 cells, reduced profile) |
|---|---|---|
| period 2 | 282 (to log10 hidden 1.7376) | 49 (to 1.119; coexisting branch at 1.187-1.200) |
| quasi-periodic | 125 (1.738-1.826) | 31 (12 with clearly negative FTLE, see caveats) |
| locked | 97: 1/5, 7/32, 5/23, 2/9 (1.834-1.879), 1/4 (1.911-1.919) | 19: 1/9, 2/17, 1/8, 3/23, 2/15 |
| chaotic / irregular / divergent | 223 / 2 / 21 | 59 / 2 / 0 |
| locked with period 2q x 2^k | 96/97 | 19/19 |

The Neimark-Sacker points of the period-2 orbit (1.738439 and 1.121671, from
the parent stream) coincide with the first quasi-periodic cells in both slices.

### Training consequences per regime (medians, log10 unless stated)

| Outcome | period 2 | quasi-periodic | locked | chaotic |
|---|---|---|---|---|
| S2 function disagreement at 40k, 1e-8 start | -15.9 | -6.1 | -14.7 (bimodal, IQR -15.2 to -0.75) | -0.63 |
| S1R same | -15.4 | -6.25 | -14.0 (bimodal) | -0.45 |
| S2 slope of disagreement vs perturbation size (linear) | 0 (at floor) | 1.00 | mixed | 0 (saturated) |
| S2 disagreement after the step-decay anneal | -15.8 | -8.7 | -9.4 (bimodal) | -2.3 |
| S2 stopping-time spread of annealed models | -4.98 | -3.73 | -3.24 | -2.36 |
| S1R same | -5.04 | -3.17 | -3.09 | -2.36 |
| S2 tail training loss | -2.20 | -2.10 | -2.04 | -1.93 |
| S2 tail held-out MSE | -2.06 | -1.94 | -1.86 | -1.73 |
| S2 held-out MSE after anneal | -2.98 | -2.98 | -2.99 | -3.05 |
| S2 loss-spectrum line fraction (linear) | 1.00 | 0.99 | 0.96 | 0.10 |
| S1R same | 1.00 | 0.98 | 0.96 | 0.39 |

Reading:

- **Reproducibility has four classes**, set by the Lyapunov spectrum and by
  transients: perturbations die out on period-2 and most locked orbits, persist
  in linear proportion to their size on invariant circles (quasi-periodic,
  slope 1, amplified about 100x but never decaying), and saturate in chaos.
- **Locked windows embedded in chaos are bimodal.** 45/97 S2 and 9/19 S1R locked
  cells show order-one disagreement: initialization perturbations are amplified
  during chaotic transients (median growth 10^8.7 to 10^11.9 in 2k steps, versus
  10^-2.4 to 10^1.0 for cells without slips) and runs land on different phases
  of the same stable cycle. Small windows inside the quasi-periodic band (1/5,
  5/23, 7/32 in S2) never slip. After annealing, phase-slipped runs still differ
  by about 1e-3 in function.
- **Stopping-time sensitivity** (the final model depends on when the learning
  rate is decayed) grows with oscillation amplitude: period-2 cycles anneal to a
  single model from either phase; quasi-periodic and chaotic attractors give a
  continuum. Inside the S2 1/4 window it drops 25x below the surrounding chaos
  at equal amplitude (finitely many phases).
- **Losses** rise smoothly with the rate across regimes; the averaged iterate is
  about 0.35 decades better than the tail mean in every regime; after annealing,
  chaotic-phase training ends about 15% lower in held-out MSE (S2).

### Does rotation or locking information add predictive value?

Leave-one-block-out least squares, base = FTLE (parts and sign classes),
sharpness, log10 rate, log10 amplitude; full = base + locked and quasi-periodic
indicators, rotation number, log10 distance to the nearest rational (q <= 12),
1/q. S2 (n = 729 cells beyond the edge of stability):

| Outcome | CV R2 base -> full | delta R2 [95% block bootstrap] |
|---|---|---|
| disagreement at 40k | 0.797 -> 0.797 | 0.000 [-0.179, +0.067] |
| same, with transient amplification in both models | 0.915 -> 0.906 | -0.010 [-0.088, +0.025] |
| disagreement after anneal | 0.815 -> 0.810 | -0.005 [-0.243, +0.070] |
| tail training loss | 0.993 -> 0.992 | -0.001 [-0.010, +0.001] |
| tail held-out MSE | 0.996 -> 0.995 | -0.001 [-0.007, +0.001] |
| held-out MSE after anneal | 0.721 -> 0.778 | +0.057 [-0.169, +0.232] |
| stopping-time spread | 0.924 -> 0.970 | +0.045 [+0.001, +0.107] |
| loss line fraction | 0.973 -> 0.974 | +0.001 [-0.037, +0.010] |

Only stopping-time spread has an interval above zero, and its partial
correlations change sign between the full slice and the beyond-Neimark-Sacker
subset (locked +0.42 versus -0.36), so on one slice it is not separable from a
nonlinear rate effect. S1R (n = 160, six CV blocks) is underpowered: point
estimates of delta R2 lie between -0.08 and +0.07 and the bootstrap intervals
are degenerate.

### The loss curve carries the rotation number and the distance to the transition

- **Readout beyond the transition.** The strongest peak of the even-step
  training-loss spectrum equals the two-step rotation number within 1e-3 in
  every quasi-periodic and locked cell: S2 125/125 and 97/97, S1R 31/31 and
  19/19.
- **Noisy precursor before it** (`tools/explore_consequence_precursor.py`). With
  small Gaussian noise added to every update (1e-7 or 1e-5, a stand-in for
  minibatch noise), a least-squares AR(6) fit of the even-step loss recovers the
  period-2 orbit's two-step complex multiplier from the loss alone:

  | | period-2 cells | |mu| range | angle/2pi error, median (max) | modulus error, median (max) | r(log peak height, -log(1-|mu|)) |
  |---|---|---|---|---|---|
  | S2-28011, log10 hidden 1.60-1.80 | 41 | 0.954-0.9985 | 1.8e-4 (9.4e-4) | 0.0014 (0.0045) | 0.94 |
  | S1R-28002, log10 hidden 0.95-1.15 | 34 | 0.888-0.998 | 3.6e-4 (2.2e-3) | 0.0023 (0.018) | 0.93-0.96 |

  Beyond the Neimark-Sacker point the same fit returns modulus 1 and the
  rotation number. So 1 - |mu| (distance to the quasi-periodic transition) and
  arg(mu)/2pi (the rotation number of the training that follows) can be read
  from the loss before the transition. The FTLE, ln|mu|/2, carries only the
  modulus. The AR estimator replaced raw peak picking after a first pass showed
  peak picking was noisy far from the bifurcation (median 3e-4, max 2.5e-3);
  that change is "adapted".

Figures: `/data2/neural-shrimp-data/explore/consequence/precursor.png`,
`.../s2-28011-out-0.0856/{slice,regimes}.png`,
`.../s1r-28002-out-0.9413/{slice,regimes}.png`.
