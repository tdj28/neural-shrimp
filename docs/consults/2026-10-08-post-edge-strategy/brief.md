# Decision memo: from finite GD dynamics to useful language-model science

This is a research-strategy consultation at a substantive milestone, not a
request to validate raw numerical evidence. Judge the summarized evidence
conditionally; the raw data remain local. We seek a strong nonlinear-dynamics
paper with an earned language-model contribution, not a forced benchmark story.

## Research question and evidence boundary

Can the way a gradient-descent period-2 orbit loses stability explain useful
training behavior beyond the generic edge of stability? Our tanh networks have
13 or72 parameters; the autonomous full-batch GD map uses separate hidden and
output rates. We numerically find complex-pair unit-circle crossings, invariant-
circle candidates, and positive/negative first Lyapunov coefficients. The
manuscript distinguishes floating-point numerics from exact-real interval proofs.
We retain the failed frozen resonance-ancestry test and negative incremental
geometry-utility results. Loss oscillation alone is not a periodic orbit or chaos.

New evidence, locally audited:

1. Minimal model: L(u,v)=(uv-d)^2/2+lambda(u^2+v^2)/2 with diagonal GD rates.
For equal rates the balanced branch reduces to x'=c*x-eta*x^3. The two-cycle
points are (sqrt(c+2)+/-sqrt(c-2))/(2*sqrt(eta)), c=1+eta(d-lambda),
e=1-eta(d+lambda). Full two-step multipliers are 9-2*c^2 and1+e^2+e*c:
real, so this branch cannot lose stability through NS. Attribute the prior
balanced scalar cycle formula to Hofmann; do not claim its discovery. In17
exploratory cases at target d=1.2,12 first crossings were flips and5 NS, all
NS with unequal rates and negative criticality. A representative NS has
lambda=.2, rate ratio sqrt(8), eta=.7335089, angle/(2*pi)=.4288192,
criticality -3.83645 and a three-point radius exponent .49790. No full phase
diagram, generic prevalence theorem, or rigorous NS certificate yet.

2. Controlled warmup: one soft and one hard NS location on the same small-
network branch; two ramp speeds, both a near-orbit initialization and ordinary
initialization; interventions fork exactly the same prehistory. Hard near-orbit
loss overshoot is roughly9.6–11.2 times the frozen-cycle maximum versus1.60–1.62
for soft. Reversing the small hidden-rate ramp recovers the soft orbit but not
the hard case. Halving both rates, a much larger intervention, yields lower
terminal loss in both, with its own transient spikes. Ordinary initialization
misses the hard local branch. Hence local mechanism is plausible, but there is
no demonstrated advantage over generic rate reduction, and accessibility matters.

3. Two selected primitive period-10 attracting orbits now have exact-real
Arb Krawczyk/contraction existence proofs, primitive-divisor exclusion, and
monodromy-power infinity-norm stability bounds (<.056 and<.108 for power64).
One belongs to our historical61-window table, one is a zoom example outside
its qualifying set. Separate replay used higher precision/narrower boxes and
an independent second-order AD implementation, sharing Arb as trusted arithmetic.
These prove neither all61 windows nor shrimp boundaries, NS, or binary64 orbits.

4. Actual-text architectural pilot: one-layer15,776-parameter causal transformer,
width32, tied embeddings, pre-LayerNorm, fixed256 next-character targets from
public tiny-Shakespeare, full-batch GD with ridge .001. Of10 rate cases,7 completed
1000steps and3 hit numerical guards. None of the7 completed runs met full-state
recurrence<=1e-5 at lag1 or2. Moderate rates memorized the tiny batch while
parameters drifted; high rates fluctuated or stopped. Reduced dense/JVP controls
and matched-state float32 GPU vs CPU64 derivatives agree within disclosed
precision limits. AR(2) can fit smooth drift extremely well; weak complex AR
roots also occur during ordinary descent. Thus no NS, chaos, generalization,
or language-model training benefit. Validation corpus has not been evaluated.
The local GPU needs only2–5ms/update; no rented GPU is justified for this model.

5. CIFAR subsetA:98,666-parameter tanh classifier has loss modulation but no
accepted orbit yet. Legacy Newton/Arnoldi ran for3h20m and ended without a final
artifact. New bounded checkpoints, corrector/primitive-cycle gates and seeded
Ritz residual checks are implemented, under separate review. Actual one-thread
CPU HVP17.8ms and cached two-step JVP33.5ms; five-vector GMRES barely decreases
residual .8994->.8968. Convergence may dominate compute. A device-capable parity
benchmark precedes any tightly capped H100 pilot. SubsetB remains sealed.

6. S1R rotation map: two independent starting families must stay separate.
Every retained cell is unresolved under the existing status code; no validated
quasiperiodicity. Each output-rate slice was independently initialized and did
not retain full orbit states, so even adjacent rows can jump branches. Provisional
rational roots are far from matching-period windows; no ancestry claim. Next
requires actual state continuation and local association, not a larger plot.

## Prior work and correction

Heavy-ball fixed-point block is mu^2-(1+beta-eta*h)*mu+beta=0. At fixed0<=beta<1,
complex roots have modulus sqrt(beta)<1, so an ordinary fixed point cannot have
an NS crossing merely because momentum produces complex eigenvalues. Higher-
period orbits/other optimizers remain open. Adam moment and bias-correction state,
noise and schedules must be modeled explicitly; not an autonomous theta-only map.

Cohen etal2022 adaptive edge-of-stability, Regis/Chewi2026 and Fong/Yang2026 are
potentially relevant; only abstracts currently checked for the latter two.
Do not invent their conclusions or treat any unverified citation as established.
Your model has no browsing; flag references for later primary-source verification.

## Decision we need

Rank up to5 directions, choose one highest-value next experiment, and specify
what each outcome would teach us. We are weighing: global fold/resonance ancestry;
a rigorous NS/criticality-sign proof in the minimal model; realistic optimizer
state and noise; or a drift-aware finite-time instability/intervention diagnostic
for a larger text model that does not require a true attractor. Challenge construct
validity and scientific novelty, not just procedure. Is demanding full-state
periodicity useful for the mechanism yet an unnecessary bottleneck for utility?

The human raised modded-nanogpt: its main task minimizes wall time on8H100 to
FineWeb validation CE3.28; its optimization track fixes architecture/data/batch
and minimizes steps. The current record uses elaborate schedules, low precision,
and momentum-based matrix optimizers. We cannot infer a speedup from tiny GD.
Suggest a sharply falsifiable, cheap path from our dynamics to a schedule or
optimizer improvement, with matched tuning budgets and strong cheap baselines
(AR1/AR2/trend, sharpness proxy, generic rate backoff, standard schedule, gradient
norm). Include overhead and predictive lead-time, time-split/seed/corpus separation,
and failure cases. No costly benchmark campaign until a scalable gain exists.

Resources: shared24-core CPU, roughly40GiB currently available RAM, RTX2000Ada16GiB
with weak float64; rented H100 only for measured worthwhile workloads. Self-funded,
small already-authorized cloud envelope; no broad8GPU sweep. Fresh confirmatory
seeds29000–29999 and CIFAR subsetB are untouched. Explore freely on development
cases; commit code/run card before runs; freeze/push headline confirmatory tests
before fresh outcomes. Preserve every failed attempt. This advisory review does
not substitute for outside scientific criticism or a human coauthor's judgment.
