# Minimal models for the quasi-periodic route (exploratory, 7 October 2026)

Which small gradient-descent loss already has an edge-of-stability 2-cycle
that loses stability by a Neimark-Sacker bifurcation? Every result here is
exploratory. Raw outputs are in `/data2/neural-shrimp-data/explore/minimal-model/`;
code in `src/neural_shrimp/minimal.py`, `tools/explore_minimal_model.py`,
`tools/explore_minimal_hofmann.py` and tests in `tests/test_minimal.py`.

## Answer

Two small models qualify.

1. **Hofmann's 2x2 depth-two linear network** (eight weights, arXiv:2607.04993,
   Proposition 16). The aligned 2-cycle loses stability at a(s1 + s2) = 2,
   reproduced to machine precision for s2/s1 = 0.7, 0.8 and 0.9. The crossing is
   a transversal complex pair (d|mu|/da = 1.85, 1.90, 1.95) with birth angles
   theta/2pi = 0.1380, 0.1082 and 0.0737. Beyond the edge, training settles on an
   attracting invariant circle of the two-step map (non-periodic, nonzero loss,
   two near-zero Lyapunov exponents for the rotation symmetry and the circle,
   the rest negative), and its rotation number tends to the birth angle.
   - **Without weight decay** (Hofmann's model) the onset is soft (no
     bistability), but the amplitude grows like delta^0.24. Second-mode minima
     form a neutral continuum (uv = s2), so the bifurcation is degenerate.
   - **With weight decay 0.01** only the exact rotation-symmetry multiplier
     stays at 1, and the bifurcation is generic and supercritical: amplitude
     grows like delta^0.49, the rotation at delta = 1e-4 is within 3e-5 of the
     birth angle, and the onset is soft.
   - **Locking beyond the edge.** At s2/s1 = 0.7 there is a 1/8 plateau
     (period 16) and locked cells at 2/17 (period 34) and 5/42 (period 84, the
     mediant of 2/17 and 3/25). At 0.8 a fine scan resolves a narrow 1/10
     tongue (period 20). All periodic cells have rotation m/q with
     q = period/2.
2. **The scalar factorization 0.5 (uv - d)^2 + 0.5 lam (u^2 + v^2)** (two weights)
   with lam = 0.2 and unequal layer rates (ratio about 2.8 or 0.35). This case
   was found as the c = 0 member of the coupled four-weight family below, where
   the two modes decouple. Its 2-cycle loses stability by Neimark-Sacker at
   theta/2pi = 0.4288, and the circle just beyond looks supercritical
   (amplitude ratio 2.08 for a fourfold offset, against 2 for sqrt(delta)).
   With weaker weight decay (lam = 0.05, or the earlier ridge-0.1 model) every
   case ends in a flip.

## Controls

- **Engineered planar gradient map** (positive control). The 2-cycle has
  symmetric step derivatives S0 = diag(1, -1) and S1 = r R_phi S0. Neimark-Sacker
  is typed at r = 1 with theta/2pi = 0.13, and the bifurcation is supercritical
  for every kappa tested.
- **Earlier factorization model, ridge 0.1.** Complex 2-cycle multipliers occur
  on 5 of 8 branches, but their modulus never exceeds 0.65; every branch ends
  in a flip.
- **c = 0** reduces the four-weight model to two scalar factorizations, as
  expected. That is where the scalar Neimark-Sacker case above was found.

## Coupled four-weight model

The model is L = 0.5 z^T C z - d^T z + 0.5 lam |theta|^2 with z = (a1 b1, a2 b2),
C = [[1, c], [c, 1]] and d = C z* + lam, z* = (1, 0.5).

- **Grid.** c in {0, 0.3, 0.6, 0.9} x lam in {0.05, 0.2} x 8 rate ratios. 58
  cases flip; 6 end in Neimark-Sacker, all at lam = 0.2 and extreme ratios:
  (c, ratio, theta/2pi) = (0, 0.354, 0.4288), (0, 2.828, 0.4288), (0, 4, 0.3401),
  (0.3, 4, 0.3834), (0.6, 4, 0.3759), (0.9, 4, 0.4075).
- **Held-out couplings.** Predictions for c = 0.45 and 0.75 were committed
  before running them (`docs/explore-minimal-model-heldout-predictions.json`).
  32 of 32 types were correct; the interpolated angles were 0.3797 vs 0.3740
  observed and 0.3917 vs 0.3892.
- **Coupling is not needed.** Weight-decay strength and rate asymmetry are.

## Caveats

- One-dimensional criticality probes (two offsets), not computed first Lyapunov
  coefficients.
- Rotation numbers come from principal-plane projections of rotation-invariant
  observables.
- In the linear network, divergent starts interleave with circle states at
  larger rates.
- Hofmann's analysis of the edge is the prior result. What is new here is the
  invariant circle, the continuity of its rotation number with the birth angle,
  its amplitude scaling, and the locking beyond the edge.

## Figures

- `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01.png`
  (coarse staircases for s2/s1 = 0.7, 0.8 and 0.9)
- `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01-fine08.png`
- `/data2/neural-shrimp-data/explore/minimal-model/grid.png` (four-weight boundary types)

## Scalar mechanism follow-up (exploratory)

**TLDR.** A two-weight model can already change from alternating training to
small bounded rotational oscillations. In this pilot, sufficiently unequal
layer rates together with the stronger tested ridge values permit that
change. Equal-rate training on the balanced branch instead has a real
period-doubling instability. This gives an explicit mechanism to investigate
in larger networks; it does not show a shrimp-specific benefit or an LLM
training effect.

The follow-up holds the regression target fixed at `d = 1.2` while varying
ridge, unlike the earlier four-weight grid whose target changed with ridge.
Code is in `src/neural_shrimp/scalar_mechanism.py` and
`tools/explore_scalar_mechanism.py`. The
[generated receipt](explore-minimal-model-scalar-receipt.json) binds both
retained attempts, their source commits, plans, raw manifests and the numbers
below. All computed results remain exploratory (evidence rung 0); derivative
and control checks are listed explicitly, without promoting the whole study
to a proof.

### Exact map and stability conditions

For

\[
L(u,v)=\tfrac12(uv-d)^2+\tfrac\lambda2(u^2+v^2),\qquad
P=\operatorname{diag}(\eta,r\eta),
\]

one update is `F(x) = x - P grad L(x)` and

\[
H(u,v)=\begin{pmatrix}v^2+\lambda&2uv-d\\2uv-d&u^2+\lambda\end{pmatrix}.
\]

When `0 < lambda < d`, the positive balanced minimum is
`u = v = sqrt(d-lambda)`. Its first flip rate is

\[
\eta_{\rm flip}=
\frac{4}{d(1+r)+\sqrt{d^2(1-r)^2+4r(d-2\lambda)^2}}.
\]

A primitive 2-cycle can be written `x0 = (a,b)+(p,q)` and
`x1 = (a,b)-(p,q)`, with `(p,q) != 0`. Adding and subtracting its two update
equations gives the exact cubic system

\[
\begin{aligned}
a(b^2+q^2+\lambda)+2pbq-db&=0,\\
b(a^2+p^2+\lambda)+2qap-da&=0,\\
p(b^2+q^2+\lambda)+(2ab-d)q-2p/\eta&=0,\\
q(a^2+p^2+\lambda)+(2ab-d)p-2q/(r\eta)&=0.
\end{aligned}
\]

Its two-step derivative is `M = (I-P H(x1))(I-P H(x0))`. With trace `T` and
determinant `D`, attraction requires the three Jury inequalities
`1-T+D > 0`, `1+T+D > 0`, and `1-D > 0`. A candidate NS crossing has
`D = 1` and `|T| < 2`; transversality, exclusion of strong resonances, and a
nonzero first Lyapunov coefficient are additional requirements. These are
algebraic conditions, not a claim that every solution of the cubic system
has a generic NS bifurcation.

### Why unequal rates matter for this branch

At equal rates, `u = v = x` is invariant and the restricted map is
`x -> c*x - eta*x^3`, where `c = 1 + eta*(d-lambda)`. For `c > 2`, its
positive 2-cycle is explicit:

\[
x_\pm=\frac{\sqrt{c+2}\pm\sqrt{c-2}}{2\sqrt\eta},\qquad
x_+x_-=1/\eta.
\]

The longitudinal and transverse two-step multipliers are, respectively,

\[
m_\parallel=9-2c^2,\qquad
m_\perp=1+e^2+ec,\quad e=1-\eta(d+\lambda).
\]

Both are real. Thus this balanced branch cannot have an NS crossing; its
longitudinal flip is at `eta = (sqrt(5)-1)/(d-lambda)` when the transverse
mode remains attracting. All four equal-rate pilot cases encounter that
flip, with absolute rate discrepancies below `1.3e-8` from the formula.
This is a branch-specific conclusion, not a global exclusion of other
attractors at equal rates.

Each individual gradient step has real eigenvalues because its derivative
is similar to the symmetric matrix `S_i = I - sqrt(P) H(x_i) sqrt(P)`.
The two-step derivative is similar to `S_1 S_0`. On the balanced equal-rate
branch the two matrices share eigendirections. Unequal rates can break that
alignment: noncommuting, indefinite symmetric factors can have a product
with complex eigenvalues. At the representative NS point below, their
eigenvalues are approximately `(-0.522, 0.801)` and `(-2.713, 0.882)`, and
the commutator norm is `2.551`. Noncommutation is necessary here, not a
sufficient condition for an NS crossing.

### Bounded fixed-target pilot

The grid follows the first attracting 2-cycle from one deterministic
perturbation of the positive minimum. It uses float64, one CPU thread,
`nice -n 10`, no stochastic seeds and no held-out data. A one-case timing
run took 2.48 seconds; the grid took 5.29 seconds. All attempts and raw
continuation paths remain in
`/data2/neural-shrimp-data/explore/scalar-mechanism/`.

| Ridge | ratio 1 | ratio 2 | ratio sqrt(8) | ratio 4 |
|---|---|---|---|---|
| 0.05 | flip | flip | flip | flip |
| 0.10 | flip | flip | flip | flip |
| 0.20 | flip | flip | NS | NS |
| 0.30 | flip | flip | NS | NS |

The additional reciprocal case, ridge `0.2` and ratio `1/sqrt(8)`, also
has NS. Swapping the two weights maps its rates and rotation onto the
ratio `sqrt(8)` case to floating-point precision. Its coefficient has the
same negative sign; coefficient magnitudes need not agree when the chosen
orbit phase and eigenvector normalization differ.

Across the five NS cases, the first Lyapunov coefficient is negative and
both symmetric finite-difference estimates of the modulus crossing slope
are positive. The largest augmented root residual is `8.9e-16`; analytic
second/third derivatives of the composed map and Richardson finite
differences give coefficients differing by at most `2.9e-9`. These small
residuals are numerical checks, not interval bounds.

For the previously discovered ridge `0.2`, ratio `sqrt(8)` case:

- NS rate `eta = 0.733508895817`; birth rotation `0.428819216165`.
- First Lyapunov coefficient `-3.836452456`; modulus slope about `4.44725`
  per unit eta, agreeing under step halving.
- At relative rate detunings `0.0001`, `0.0004`, `0.0016`, RMS amplitudes are
  `0.01303`, `0.02604`, `0.05184`, giving a three-point exponent `0.49790`.
- Corresponding measured rotations are `0.428665`, `0.428227`, `0.426503`,
  approaching the predicted birth rotation as detuning decreases.

This supports a soft onset in that case and negative local criticality in
all five tested NS cases. Three short amplitude probes do not prove an
invariant circle. The scan does not locate a complete ridge/asymmetry phase
diagram, exclude other branches, establish necessity of ridge globally, or
show resonance ancestry. At zero ridge the balanced minimum ceases to be
isolated; that degenerate setting was not included.

| Component | Evidence | Scope and unresolved work |
|---|---|---|
| Balanced cycle and stability formulas | Algebraic derivation; float64 identity tests | Not formally mechanized or interval certified |
| Fixed-target grid and NS coefficients | Rung 0, exploratory; residual and derivative checks retained | One branch per case; no prevalence claim |
| Amplitude/rotation probes | Rung 0, exploratory | One known case; no rigorous circle or chaos result |
| Raw-evidence receipt | SHA-256 file/manifest checks | Local data store; not an independent backup |

Recheck every raw hash and regenerate the receipt values without new
trajectories:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data2/neural-shrimp/.venv/bin/python tools/analyze_scalar_mechanism.py \
  --receipt docs/explore-minimal-model-scalar-receipt.json
```

### Next ancestry improvement (assessed, not implemented)

The existing ancestry code follows parameter rays and straight segments;
it already accepts unstable periodic orbits. The useful next change is a
bordered pseudo-arclength continuation of the fold itself. Solve
`F_alpha^p(x)-x = 0`, `(D F_alpha^p(x)-I) v = 0`, `v^T v-1 = 0`, and a
predictor hyperplane condition in `(x, alpha1, alpha2, v)`. Start from a saved
fold bracket and its multiplier-near-one eigenvector, retain both fold
branches, and compare full orbits modulo phase. This avoids the current
finite-width cross-section failure and permits folds that turn in parameter
space. It needs fold-normal-form controls, derivative checks, simple-nullspace
checks and step halving before being applied to the unresolved original
window connection. A converged fold trace by itself would not establish
that the distant window lies on the same orbit branch.
