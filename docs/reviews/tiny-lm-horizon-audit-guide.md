# Fixed causal-horizon development comparison: independent replay guide

The completed producer is `tiny-lm-horizon-target002.json`. It is not an
independent outcome audit. Source/execution is
`c59a319e0d6e2451c9d4d8ce2494cec11ae6041a`; all executed source bytes are retained
under `run/source/`. The outer inventory is
`0e4a96fd89e322951d8735945d964a5a59b526139ec97baef2da2b8f8a08391c`, and worker
inventory is `bdd4d7e1580943232231d0b1f8f68d415969c67d8cbab22f202491533c25c0c8`.
Raw root: `/data2/neural-shrimp-data/explore/tiny-lm-horizon/attempt002-target`.
All747 outer and741 worker files, read-only modes, full inventories and owned
worker absence were checked after completion. The fixed complete acquisition
is72 HVP and32 training CE calls, 10.148s worker/12.568s outer,954628KiB peak.

## Inputs, causality and source authority

The only cases are seeds28161/28162, arms8/256, origins0/4/8/12 and original
gradient/random directions0/1:32 correlated prefixes of four consumed
trajectories. This is not32 independent replicates. The four-step response is
continued from the original perturbation, not a perturbation restarted locally.
No held-out input, new trajectory, fitting or intervention was performed.

`run/response-binding.json` binds the complete original response at
`/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-target`:
outer`fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5`,
worker`cb57135497afa72437651b7eaab807e58387a304fd0f0d2609113ffcff073aad`,
execution`081c5dcc779c3b2f60d0a3102275f0fa7609efbf`. Source dependency
`b4c3d588da671f25a7fa22c6882e1cdcb42d2079` is distinct from its execution.
The two existing independent outcome audits are bound there too. The known
historical broader-data directories are writable containers with immutable
manifest-qualified files; exact literal root+hash pairs alone admit that
exception. New outputs are strictly read-only. The zero-native profile001
packaging failure remains mandatory input, not replaced.

Case directories are `run/seed-S-arm-A/origin-TT-direction-D/`.
`early-vectors.npy` rows are `[u,v1,a2,f2,g0,g1]`, each15776D. With original
transported vector v, normalization s=norm(v[t]), they are
u=v[t]/s, v1=v[t+1]/s, a2=v[t+2]/s, f2=frozen[2], and reference gradients at
t,t+1. `early-states.npy` is exactly reference[t:t+2]. The a2 tangent requires
only the Jacobian at state t+1; this causality is established by the original
source-authenticated ordered transport, not by a newly independent native
implementation. Verify original inputs and entire reference GD recurrence.

`frozen.npy` has local0..4. Origin0 reuses the original frozen forecast divided
by s. Each other origin retains three new HVP calls at theta[t], with
f[j+1]=f[j]−eta*H(theta[t])*f[j], j=1,2,3. f0=u and f1=v1. Native operation
order is seed,arm,origin; two CE calls once per origin, followed by directions0
and1 (three HVP each except origin0). Each `operations/K-NNN-attempt.json`
is preceded by theta and, for HVP, vector; returns and result records follow.
`events.json` must have exactly72 HVP plus32 CE attempts/returns, in this order.
Two loss scalars in prediction.json are retained CE plus0.0005*dot(theta,theta).
No new gradient calls occur.

All32 `prediction.json`, early arrays and frozen arrays are hash-sealed in
`predictions-closed.json` before `evaluate_predictions` opens any amplitude
response branch. Reading an original whole tangent file does not pass later
slices into the predictor; inspect the fixed slices/source. The source order,
sealed hashes and immutable output establish this separation, not file mtime.

## Independently recomputable arithmetic

Use independently implemented norms/dots (e.g. math.fsum) and exact row order;
no need to import producer reducers or native Torch for a saved arithmetic
audit. Relative(a,b)=norm(a−b)/max(norm(b),1e−12). Predictors use only early rows:

- e2=relative(f2,a2); drift score h(h−1)e2/2; persistence score e2.
- gradient score h*relative(g1,g0).
- loss score h*abs(L1−L0)/max(abs(L0),1e−12).
- directional-curvature score h*relative(u,v1), denominator norm(v1).
- composite is maximum of persistence, gradient, loss and curvature, not drift.

Horizons are1..4. All fixed crossings use strict `>0.2`. Drift is a fixed
heuristic expansion of early defect, not a rigorous error bound or information
beyond e2. Preserve predicted crossing as well as horizon3/4 warning Boolean.

Original outcome epsilon_values has two positive amplitudes with eps0=2eps1.
For each branch, reconstruct `(plus[t:t+5]−minus[t:t+5])/(2*eps*s)`;
`response-large.npy` and `response-small.npy` must agree with that construction.
`evaluation.json` includes an exact ordered_source path/hash/slice[t,t+5]/s;
ordered tangent is deliberately not duplicated in new output.

Origin qualification requires relative(large0,small0) and
relative(ordered0,small0) both<=0.01. Each horizon also needs both corresponding
errors<=0.01. Unresolved cases remain unresolved. Frozen error is
relative(f[h],small[h]); persistence error relative(u,small[h]); actual gain is
norm(small[h]); scalar gain norm(f1)^h, with relative scalar gain error.
Main advance-warning table admits only resolved rows with frozen errors at
h1/h2 both<=0.2. A true future failure exceeds0.2 at h3 or h4. Already-failing
h2 rows are separate detection cases. Warning is score>0.2 at h3 or h4.

Recompute all32 evaluations and six confusion tables. Producer reports all32
resolved,24 eligible,8 already failing; one eligible failure at
seed28161/arm256/origin12/random direction1,h4. Drift, gradient and composite
have1 true warning/7 false warnings/0 misses/16 correct quiet; curvature has
1/5/0/18; persistence and loss have0/0/1/23. Drift-only caught failures=0.
These outcomes do not establish incremental utility, generalization, training
benefit, causality of a new intervention, or a shrimp-specific effect.

## Independent audit completed

Root's separate f997c733 audit passed all32 cases/128 horizons in3.050s,
61516KiB, zero native calls. Receipt b86835e9 is
`docs/reviews/tiny-lm-horizon-independent-outcome.json` in the integration
branch. Audit outer56e9f2f517888494832bb6354e1faf1dee35c5590e40c360034054b35877f619;
worker6aba13c0d30b116dc1d2f23c8f09906feb54c7ea7b9434e9caab0d0f290e5615.
The negative incremental conclusion above is independently reproduced within
its explicit shared-acquisition, consumed-development scope.
