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
