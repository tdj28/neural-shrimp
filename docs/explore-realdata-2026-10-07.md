# Real-data generality stream (exploratory), 7 October 2026

Status: exploratory under the [adaptive research protocol](adaptive-research-protocol.md).
Nothing here is confirmatory. Seeds 29000-29999 were not used (the code refuses them).
Every run has a card in [the research log](research-log.md).

## Question

Does the quasi-periodic route seen in the S2 and S1R toy systems (edge-of-stability
period-2 orbit, Neimark-Sacker bifurcation of the two-step map, quasi-periodic
training on an invariant circle, phase-locked windows of period 2q at rotation
m/q in Farey order, then chaos) also occur for a network trained on real data
with standard parameterization and one global learning rate?

## Setup

- Data: UCI optical digits (8 x 8) as bundled with scikit-learn, copied read-only
  from a local scikit-learn 1.4.0 installation to
  `neural-shrimp-data/external/uci-digits/digits.csv.gz`
  (SHA-256 `09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22`,
  verified on every load). Fixed subset of 1,024 images (seed 28101 unless
  stated), pixels standardized on the subset, targets +1 for digits 5-9 and -1
  for 0-4 ("halves"; "parity" is the other task).
- Network: 64-8-1 tanh MLP with biases (529 parameters), standard
  parameterization, PyTorch-default uniform initialization, mean squared error,
  float64 full-batch gradient descent with one global learning rate. Weight
  decay 1e-3 unless stated (without it, slow flat directions keep the state
  drifting for the whole run).
- Code: `src/neural_shrimp/realdata.py`; `tools/explore_realdata.py`
  (`calibrate`, `scan`, `continue`, `branch`, `sweep`); plots in
  `tools/plot_realdata_scan.py`, `tools/plot_realdata_sweep.py`,
  `tools/plot_realdata_portrait.py`; tests in `tests/test_realdata.py`,
  `tests/test_explore_realdata.py`. Orbits are certified with
  `neural_shrimp.orbits.certify_orbit` and continued with
  `neural_shrimp.continuation.continue_ray` (unchanged).

## Results

1. **Calibration (no weight decay):** sharpness 2.88 at initialization, 7.78
   at an L-BFGS minimum (training accuracy 99.9%), so that minimum flips at
   eta = 0.257.
2. **Fixed-initialization scans, 160 rates, log10 eta in [-0.9, 0.1]:** no
   exactly periodic cells after 20,000 updates. Without weight decay: drift
   (132), chaotic (4), divergent (24). With weight decay 1e-3: the loss
   spectrum carries a line at 1/2 (up to 80% of the power) from -0.9 to -0.40,
   an edge-of-stability period-2 oscillation still relaxing along slow
   directions; slow harmonic loss modulations from about -0.37; broadband
   power from about -0.27; positive FTLE from -0.20; divergence from -0.05.
3. **Sharp-minimum branch:** the L-BFGS minimum with weight decay
   (lambda_max 23.6) flips at log10 eta -1.0727; its period-2 orbit ends in a
   **fold** at -0.8952 (multiplier +0.9999999). Not the route.
4. **Neimark-Sacker of the period-2 orbit gradient descent actually reaches:**
   after 60,000 updates from the initialization at log10 eta -0.42, Newton
   certifies an attracting period-2 orbit; continued upward it loses stability
   at **log10 eta = -0.3885464** (bracket 9e-10) with dominant two-step
   multiplier **0.998814 + 0.048693i**, modulus 1 - 1.0e-7: a Neimark-Sacker
   bifurcation, predicting a newborn circle with rotation
   arg(mu)/2pi = **0.0077528** turns per two updates.
5. **Prediction confirmed just beyond the bifurcation:** started from the
   certified period-2 state, independent runs (40,000 + 32,768 updates) and an
   adiabatic sweep both settle on a smooth invariant circle at log10 eta
   -0.3885 with rotation **0.007751** (127 turns recorded; forward angle steps
   100%; planar share 0.993-1.000; FTLE -0.0003 to -0.0006, zero within the
   finite-time error; no recurrence up to lag 2,048). The rotation number then
   decreases smoothly (0.006886 at -0.3816) as the loss modulation grows from
   1e-4 to 3e-3.
6. **Circle growth, wrinkling and a second smooth segment:** near -0.380 the
   loop becomes wrinkled and no longer star-shaped (phase portraits), where the
   centroid-angle rotation estimate stops being valid; FTLE stays within
   +-0.0015. From -0.347 to -0.330 the sweep follows a smooth monotone circle
   again (rotation 0.00584 rising to 0.00753) with a regular slow loss
   oscillation of about 267 updates per cycle and two loss spikes per cycle.
7. **Fixed-initialization fine scan, 120 rates in [-0.44, -0.24], 60,000 +
   16,384 updates:** 18 quasi-periodic candidates (at least three monotone
   turns, |FTLE| < 0.005), rotation 0.0039-0.0124, in patches of adjacent rates
   with nearly equal values. Gradient descent from the initialization lands on
   different period-2 branches at different rates (loss levels jump between
   patches), each with its own circle.
8. **Not found:** no locked windows (exactly periodic orbits with period 2q) and
   so no staircase. At rotation numbers near 0.007 the relevant tongues have
   q > 100 and would be extremely thin; the recurrence check (lag up to 2,048,
   relative tolerance 1e-7) found none.

## Robustness (second subset, second task)

9. **Coarse scans** (120 rates, log10 eta -0.8..-0.2, weight decay 1e-3): for
   both seed 28102 (halves) and the parity task, a period-2 regime (line at 1/2
   up to 73-79% of loss power) is followed by slow harmonic loss modulations
   (from -0.634 and from -0.42 respectively) and positive FTLE near -0.2.
10. **Continuations:** parity period-2 orbits certified at -0.4067 and -0.4521
    sit on different branches and both end in **folds** (-0.398313 and
    -0.412628; multipliers +0.99999985 and +0.99999987). For seed 28102, Newton
    did not certify a period-2 orbit from -0.7092 or -0.7546 (not converged;
    line search stalled), so that subset's branches are unresolved.
11. **Fold or saddle-node on a circle?** No. An adiabatic sweep from the
    parity fold's last attracting state over 20 rates past it (-0.39830 to
    -0.39400) stays on an exactly period-2 orbit (recurrence at lag 2,
    rotation 6e-5, FTLE -0.00064, loss varying smoothly). So the period-2
    regime is a set of branches joined by folds. The parity scan's slow
    modulations (from -0.42) come from other branches, and the Neimark-Sacker
    hypothesis for that task is still untested.

## Interpretation (exploratory)

The first half of the route is present in a real-data network with standard
parameterization and one global learning rate. The edge-of-stability period-2
orbit loses stability by Neimark-Sacker, and training then lives on an
invariant circle whose rotation number equals the multiplier angle at the
bifurcation. In the loss this appears as a slow, regular modulation on top of
the period-2 oscillation, a few hundred updates per cycle. The small angle
(0.0078 turns per two updates) puts the circle close to a 1:1 resonance, which
fits the slow sharpness-oscillation cycles described in edge-of-stability work.
That link is a hypothesis to check, not a finding. Phase locking and a Farey
staircase were not resolved here. The route is not universal across branches.
The sharp L-BFGS minimum's period-2 branch and both certified parity branches
end in folds, and the second subset's branches were not certified. So the
evidence is one Neimark-Sacker branch, with a quantitatively confirmed
rotation number, among five period-2 branches examined; the other four are
three folds and two uncertified starts. Slow harmonic loss modulations
nevertheless appear in all three configurations' scans.

Recommended next step: trace the parity and seed-28102 period-2 branches more
completely (more start rates, longer settling) to find their Neimark-Sacker
points if they exist. Measure rotation numbers with a curve-ordering
(arc-length) method that stays valid for wrinkled loops. Repeat on MNIST or a
wider network with Krylov eigenvalues of the two-step map instead of dense
Jacobians.

## Figures

- `neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-coarse/scan.png`
- `neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-fine/scan.png`
- `neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/sweep-ns/sweep.png`
- `neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/portraits.png`
- `neural-shrimp-data/explore/realdata/w8-n1024-halves/scan-coarse/scan.png` (no weight decay)

## Caveats

- One architecture (64-8-1), one dataset, 1,024-image subsets, weight decay
  1e-3. The Neimark-Sacker branch was found for one subset and task only.
- FTLE over 4,000 updates cannot separate quasi-periodic orbits from weakly
  attracting locked orbits with very long periods; the recurrence check covers
  periods up to 2,048 updates only.
- The centroid-angle rotation estimate is valid only while the loop is
  star-shaped about its centroid.
- Fixed-initialization dynamics mix several branches, so maps from the
  initialization are not single-branch bifurcation diagrams.
