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

## Orbit continuation and the crossing

(pending)

## Circle past the crossing

(pending)

## Timings

Float64 full-batch step on subset A: 1.8 ms with 16 torch threads in one
process; 8-15 ms per step per process with 4 processes x 4 threads (memory
bandwidth bound). Matrix-free costs: see the continuation section.
