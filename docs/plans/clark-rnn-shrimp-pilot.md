# Clark fixed-random-RNN shrimp exploration, 9 October 2026

Exploratory, separate from the small trained-network main track. No target run
has occurred at this source freeze. The initial scientific question is whether
finite fixed random recurrent matrices exhibit nontrivial attracting periodic
windows in the gain/leak plane, with a route toward qualifying their geometry.

Primary reading: David G. Clark, [Lyapunov spectrum of random neural networks](https://arxiv.org/html/2610.12426v1), v1, 8 October 2026; full text and appendices read. Equations5–7 specify the exact map and tangent; AppendixD.2 uses substantially longer physical-time observations for large-N Lyapunov spectra. The paper concerns a large-N distribution, not absence of finite-N periodic windows. Its mean-field assumptions do not certify this finite panel. We do not implement or claim to validate its single-site response solver.

## Model and useful restrictions

The state is the entire N-dimensional activation vector. No training, forcing,
bias, readout, momentum or resampled connectivity:
`F(x)=(1-delta)*x+delta*g*J0*tanh(x)`,
`DF=(1-delta)*I+delta*g*J0*diag(1-tanh(x)^2)`.
All J0 entries, including the diagonal, are sampled independently with standard
deviation1/sqrt(N), once per network. Gain multiplies this saved matrix without
spectral normalization. Stored binary64 J0 and coordinates define eventual exact
proof inputs; the intended Gaussian sampling is not an exact real-Gaussian claim.

Our direct elementary checks: F is odd and DF even; mirrored cycles have the same
spectrum. A cycle may itself contain the negative state, so absolute-value/norm
recurrence cannot establish primitive period. Zero is always fixed and its finite-N
stability uses the actual spectrum of `(1-delta)I+delta*g*J0`, not an assumed g=1
threshold. Nonzero fixed-point locations solve x=gJ0*tanh(x), independent of delta.
For 0<delta<=1 the max norm obeys a convex absorbing bound using g||J0||infinity.
In N=1 at delta=1 the scalar map is monotone: positive coupling permits only fixed
points, negative coupling permits only periods1/2. For positive scalar coupling
this increasing-map restriction also holds at smaller delta. We make no negative
result for arbitrary scalar leak/coupling or all finite dimensions. Planar smooth
autonomous continuous flows have Poincare–Bendixson restrictions under its standard
bounded-limit-set hypotheses; finite-delta maps do not inherit that restriction.

Delta is both leak fraction and Euler step in this family. Any windows here are
windows of the discrete map. No continuous-flow claim follows without a separate
matched-physical-time, step-refined study. In particular, a flip can be a finite-
step effect. Both per-step and per-physical-time tangent growth are saved.

## Fixed initial panel and evidence

- N=4,8,16, matrix seeds28170/28171 (not previously consumed; no29000–29999 seeds).
  PCG64/SeedSequence `[seed,N,0]` draws J0; `[seed,N,1]` draws two independent
  N(0,1) starts; `[seed,N,2]` draws their tangent starts. Save all bytes, versions
  and hashes. These two starts are reused at every grid point, freshly reset;
  neither continuation nor symmetry duplicates count as independent starts.
- Gain21 geometric points from0.6 to20; delta21 linear points from0.05 to1.
  Six matrices ×441 cells ×2 starts =5292 trajectories. No population claim.
- Discard6000 steps, align one tangent for1000, measure4096 in16 blocks of256.
  Save the last513 full states, midpoint/final states, tangent, all growth blocks,
  inputs, coordinates and fixed-point/cycle proposals. This is finite-horizon
  screening, not asymptotic Lyapunov or long-transient exclusion.
- Propose the smallest full-state p<=64 closing all last4p pairs at normalized
  max error1e-8. No recurrence in this horizon means unresolved, not chaos.
- Per network correct the first32 p>=3 proposals in grid/IC order and first4 p2
  controls, maximum216 attempts total. All other proposals are retained explicitly
  without qualification. Newton allows six bounded corrections (max displacement
  0.05 of the state scale), closure1e-11, proper-divisor separation>1e-6,
  adjacent-phase closure<=1e-10, Floquet residual<=1e-10 and both adjacent-phase
  spectral radii<1-1e-6. Retain native cycles/Jacobians/monodromies/eigenvectors,
  every attempt/failure and odd-self-pair diagnostic. These are floating numerical
  checks; step/precision convergence and an independent check remain necessary.
- Positive growth in both2048-step halves, physical-time threshold0.002, is only
  a finite-time irregular-control nomination. Periodic points can have positive
  transient growth and unstable fixed states can expand. Later candidate/control
  qualification must lengthen burn/measurement, vary tangent starts, and inspect
  block convergence before a numerical-chaos label. An exactly annihilated tangent
  is reset to its deterministic initial direction to keep the batch running, but
  its collapse count/raw log growth are retained and its Lyapunov estimate is
  invalidated, never promoted to a chaos control.

A shrimp claim requires more than a connected colored component: primitive
full-state attraction plus fold/flip boundary organization, including the U-shaped
fold outline/internal fold structure and crossing period-doubling curves where
present, checked by full-state continuation/local bifurcation tests. Root maintains
the primary-literature geometry definition. Neither a finite skeleton nor an
interval orbit certificate establishes an infinite cascade or whole geometry.

## Profile, caps and stopping

One profile uses unprotected seed28169,N16,9×9 grid and identical steps/retention.
It is exploratory evidence retained separately, never silently added to the panel.
Also time ten64-step map/monodromy/eigensystem units. Synthetic analytic p4, unstable
fixed point, contraction, chain-rule, symmetry, interruption and malformed-profile
controls precede launch. Source/card go to root before profile/collection.

Profile internal45s/external60+5s; scan internal300s/external330+5s. CPU/BLAS1,
nice>=10, monitored RSS<=512MiB, run output<=300MiB plus1MiB launcher allowance.
Check fresh host capacity; no GPU, paid call or protected data. Run command:
`nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn.py --mode profile --out /data2/neural-shrimp-data/explore/clark-rnn/profile-001`.

The exact manifest-pinned profile authorizes collection only if completed,
source/settings/runtime-bound and finite: conservative time
`2*6*(441/81)*batch_seconds +216*8*period64_unit_seconds+20 <=300`;
conservative peak `profile_RSS +3*(513*882*16*8)+64MiB <=512MiB`.
The correction factor8 covers seven Newton/closure products plus adjacent phase;
extra factor2 covers grid overhead. These are estimates, not guarantees. A profile
failure triggers a disclosed resource amendment, not changed scientific gates.
The scan receives the exact profile run-manifest SHA after measured review.

Launch exclusive roots, source clean/committed; record own PIDs, command, versions,
thread settings, exit and manifests. Save checkpoint state/tangent every2048steps,
and at a cooperative budget interruption. An external kill may leave only the
last durable checkpoint; report partial rather than invent completion. Every raw
file is preserved and made read-only after launch; no automatic retry. Adaptations
use dated run cards and new exclusive roots. Useful follow-up is local refinement,
long-transient tests, full-state boundary continuation and a small interval proof
for a candidate, not treating the first picture as a finding.

## Pre-profile review amendment

Root read the first freeze7158fc82 before any measurement. Each returned Newton
iterate now emits its theta, endpoint, full phase Jacobians, cycle and monodromy
before downstream checks; solved corrections and both eigensystems likewise emit
before possible later failure. Every event has a numbered array/JSON pair. A
budget interruption preserves already returned events and the attempted seed.

All scientific NPZ/JSON writes reserve their maximum bytes first, preserving1MiB
for final failure/result/manifest metadata within the300MiB run cap. A reservation
failure stops the unit before that write; no partial result is promoted. The
launcher enforces nice>=10, fixes all four thread variables to1, records fresh
MemAvailable, and cleans/checks the entire owned process group after the leader
returns. The worker independently rejects wrong nice/thread settings and shared
MemAvailable<1GiB. The worker RSS cutoff is448MiB, leaving64MiB within the512MiB
project limit for the small launcher and headroom. Ten synthetic64-step timing
returns are also retained. These are source/resource corrections, no outcome-
dependent change. The original profile and panel scientific settings remain fixed.
