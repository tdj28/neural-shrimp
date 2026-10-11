# Bounded CIFAR-10 pilot (exploratory), 7 October 2026

Question: does full-batch float64 gradient descent on a small network trained
on real images show the edge-of-stability period-2 orbit losing stability by a
Neimark-Sacker bifurcation, with quasi-periodic training whose rotation
matches arg(mu)/2pi? Roadmap review finding I02 and mechanism-consult
direction 4 (bounded pilot, development subset only, held-out subset sealed).

## Setup

- **Data.** CIFAR-10 python archive via torchvision 0.29.1 into
  `/data2/neural-shrimp-data/datasets/cifar10/` (SHA-256
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`;
  torchvision's MD5 check passed). Two disjoint class-balanced subsets of 512
  training images from seed 28201 (`neural_shrimp.cifar.balanced_subsets`):
  A (development; 51-52 images per class; index hash `52b530fe870e...`) and B
  (held out; index hash `9ae9c70a5449...`). B's index list is sealed in
  `explore/cifar/subsets/subset-B-sealed.json`; B's pixels were never extracted
  or used.
- **Model.** tanh MLP 3072-32-10 (98,666 parameters), standard
  parameterization, PyTorch-default initialization (seed 28202), inputs
  standardized per feature on A, MSE on one-hot targets
  (L = mean_i 1/2 ||f(x_i) - y_i||^2 + lambda/2 ||theta||^2), float64,
  full-batch gradient descent, CPU (torch 2.14.1).
- **Matrix-free tools** (`src/neural_shrimp/cifar.py`): Hessian- and
  two-step Jacobian-vector products (torch.func), two-step Floquet multipliers
  by Arnoldi (SciPy `eigs` on a LinearOperator; largest-magnitude and
  largest-imaginary-part searches, because weight decay puts the many flat
  directions in a real multiplier cluster at (1 - eta lambda)^2 just below
  one), and a Newton-GMRES corrector for period-2 orbits.

## Validation

- Unit tests (`tests/test_cifar.py`, 6 pass): gradient, Hessian-vector and
  two-step Jacobian-vector products against finite differences; Arnoldi
  multipliers (both searches) against dense eigenvalues of the explicit
  two-step Jacobian on a reduced network; Newton-GMRES convergence; subset
  balance and disjointness; recovery of a planted rotation number.
- Known Neimark-Sacker replay (`tools/validate_cifar_multipliers.py`): S2-28011
  at log10 output -0.0856 and log10 hidden 1.7370, 1.7380, 1.7384 (traced
  crossing at 1.738439), through the same torch code with a per-parameter rate
  vector. Dominant complex two-step multiplier agrees with the dense NumPy
  certification to 9e-16, 1.1e-15 and 1.4e-15 (e.g. 0.198787 + 0.980006i,
  modulus 0.999964, theta/2pi 0.21815 at 1.7384); losses agree to the last
  digit; Newton-GMRES converges to the certified orbit within 8e-15.

## Weight decay (subset A)

Sharpness at initialization is about 305 (edge-of-stability rate about 0.0065).
After 20,000 sub-edge steps (eta = 0.00327) none of lambda = 0, 1e-4, 1e-3,
1e-2 had converged (gradient norms 5.6e-3 to 1.8e-2); the slowest relaxation
time 1/(eta lambda) at the edge exceeds 2e5 steps for lambda <= 1e-3 but is
about 2e4 for 1e-2. lambda = 1e-2 is used throughout (training loss about
0.05 at the attractors).

## Learning-rate sweep (lambda = 1e-2, 150,000 steps from initialization)

| eta | Tail | Even-step loss |
|---|---|---|
| 0.0050-0.0073 | slow drift toward a fixed point (relative step 2e-6) | monotone |
| 0.0083 | period-2 candidate (consecutive 3.5e-4, two-step 5e-6) | monotone |
| 0.0094 | period 2 with modulation | oscillation, period 588 two-steps, envelope decaying |
| 0.0107-0.020 | period 2 with sustained modulation | oscillation, period 323 to 50 two-steps (frequency 0.0031 to 0.020), steady envelope |

## Orbit continuation and the crossing: not certified

- First continuation run (largest-magnitude plus largest-imaginary-part
  Arnoldi, k = 6, uncapped iterations) timed out after 3.3 hours without
  completing one point: the largest-magnitude search cannot get past the
  nearly degenerate real cluster of flat-direction multipliers at
  (1 - eta lambda)^2. Fixed: largest-imaginary-part search with k = 2
  (converges in about 250 two-step products, 3.7 s) plus a smallest-real-part
  search for flips; partial ARPACK results flagged.
- Second run: the capped Newton-GMRES corrector did not certify the period-2
  orbit at eta = 0.00828 (relative two-step residual 1.2e-5 after 8 Newton
  steps), so continuation stopped by rule. Likely cause: restarted GMRES
  stalls on J^2 - I, whose flat-direction cluster sits at about
  -2 eta lambda = -1.7e-4 below a wide stiff spectrum.
- Fallback (gradient descent toward the attracting orbit,
  `tools/explore_cifar_spiral.py`, 60,000 further steps):
  - eta = 0.00828: still 3.9e-6 from a period-2 orbit; the even-step loss
    relaxes monotonically (AR(2) root 1.000035 + 0.00045i). The matrix-free
    critical pair at the sweep state was 0.998575 +- 0.006340i (modulus
    0.998595, theta/2pi 0.00101).
  - eta = 0.00939: the loss oscillation no longer decays (envelope 1.12e-4 to
    1.13e-4): a sustained small modulation of the 2-cycle with frequency
    0.0018 (spectral peak) to 0.0028 (AR(2), modulus 0.99993) turns per
    two-step. Arnoldi at that point, which lies on the loop rather than at the
    2-cycle, gives 0.99013 + 0.02154i (theta/2pi 0.0035). The pre-stated
    surprise (angles disagreeing by more than 20%) occurred, but the
    comparison is not like-for-like.
- Status: consistent with a Neimark-Sacker bifurcation of the edge-of-stability
  2-cycle with a small birth angle between eta = 0.0083 and 0.0094, followed by
  quasi-periodic modulation whose frequency rises smoothly to 0.020 at
  eta = 0.020 (as in the UCI-digits network). Neither the crossing nor its type
  is certified, and no invariant-circle, amplitude-scaling or Lyapunov test was
  completed.

## Timings (for pricing any later H100 run)

- Float64 full-batch step on subset A (98,666 parameters, 512 images): 1.6-1.8
  ms with 14-16 torch threads in one process; 8-15 ms per step per process
  with 4 processes x 4 threads (memory-bandwidth bound).
- Hessian-vector product about 72 ms; two-step Jacobian-vector product about
  12 ms once the update is cached; Arnoldi critical pair about 4 s; one capped
  Newton-GMRES step up to about 10 s.

## Caveats

- One architecture, one development subset, one initialization; subset B
  untouched.
- Weight decay 1e-2 was needed for tractable relaxation and changes the
  minimum (training loss about 0.05).
- A fold (multiplier +1) of the 2-cycle would be hidden in the flat cluster and
  is not monitored by the LI/SR searches.
- Fixed-start sweep tails were not fully converged (150,000 steps).

## Next steps if pursued

1. A corrector that handles the flat cluster: deflate the exact flat subspace
   (J^2 - (1 - eta lambda)^2 I is low-rank) or precondition GMRES with the
   weight-decay part, then certify the orbit and bisect the crossing.
2. Arnoldi at certified orbit points on a fine eta grid between 0.0083 and
   0.0094; first Lyapunov coefficient with `neural_shrimp.criticality`.
3. Largest Lyapunov exponent and amplitude scaling just past the crossing.
4. Only then open subset B with a frozen protocol.
