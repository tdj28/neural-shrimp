# Candidate: does frozen curvature describe the tiny LM's response?

Design-only amendment, 8 October 2026. **No code or run authorized.** For
human/next-heartbeat review after the broader-data outcome audit.

**TLDR:** Two exploratory forecasts failed their incremental geometry gates.
Broader data reduced memorization, yet simple forecasts still beat B3. I would
stop refitting those features and ask one different, falsifiable question:
**does curvature changing during training materially limit a frozen local model
of perturbation response?** This checks an approximation needed by a scalable
mechanistic diagnostic. It establishes no shrimp, training-control or inference
benefit. The old negative units remain stopped.

This proposal openly adapts after the producer outcome at
`9b2495fb2136f81e6a0db793d107e6eaf44d8c11`; its independent outcome audit is
pending when this memo is written. All proposed evidence is development-only.

## Fixed comparison

Use exactly four saved checkpoints: **update 600**, seeds **28161/28162**, both
**8/256-window arms**, original objective/eta and tied parameter map. This is the
previous late-block start, fixed now without searching for rotation or large
error. Input: `/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-002-collection`,
manifest SHA256 `2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d`.
No new text, probe evaluation, rate choice, fitted model, assessment, final 10%
text or reserved seeds 29000–29999.

Promote each saved float32 state exactly to float64; make one **16-update
float64 reference replay**. This is a local counterfactual, not a claim to
reproduce its old float32 future. Do not use stored future float32 states as an
exact float64 orbit. Let `F(theta)=theta-eta*grad L(theta)` for regularized
training loss and `J_k=I-eta*H_L(theta_k)` along the replay.

Two unit directions: initial gradient and one Gaussian vector from unprotected
seed **28166**, generated once in the fixed flat ordering and reused. Zero or
nonfinite gradient means unresolved, with no replacement. For each direction,
use central perturbations `theta +/- epsilon*u` at exactly
`epsilon={1e-4,5e-5}*max(1,norm(theta))`; each branch runs 16 updates. Thus each
checkpoint has eight perturbed branches plus its reference. Evaluate horizons
**1,4,16**; no direction, amplitude or horizon search.

| Description | Response or gain | Meaning |
|---|---|---|
| Persistence | `p_h=u` | No amplification/rotation |
| One-step gain | `g_h=norm(J_0*u)^h` | Simple scalar norm forecast; one HVP |
| Frozen curvature | `p_h=J_0^h*u` | Complete initial Hessian action, not just sharpness |
| Transported derivative | `p_h=J_(h-1)*...*J_0*u` | Retrospective chain-rule control, **not an online forecast** |

HVPs suffice; no dense Hessian, eigenvector optimization or Floquet claim.

## Falsification and interpretation

Compare these with central response
`d_e(h)=[F^h(theta+e*u)-F^h(theta-e*u)]/(2*e)`.
Report gains and vector errors `norm(p-d)/max(norm(d),1e-12)` separately;
orientation failure need not imply gain failure. The scalar comparator gets
only a norm error. A resolved comparison requires the two central responses
to agree within **1%** and the transported derivative to match the smaller
amplitude within **1%**. One-step failure blocks later mechanism claims. Retain
nonlinear/numerical failure without expanding amplitudes or thresholds.

Narrow support requires **both expanded-arm seeds**, gradient direction,
horizon 16: resolved transported response and **frozen vector error above 20%**.
Always report gains, random-direction results and small-data controls. Accurate
frozen curvature would falsify its proposed inadequacy here; equally poor
small-data controls would make the limitation generic. A transported match is
expected calculus/implementation validation, not a discovery. No window-level
p-values or population inference from four development checkpoints.

Parameter sensitivity depends on coordinates and may reflect weakly observable
symmetries. Also retain terminal training-CE directional slopes and their
chain-rule predictions, marking a slope unresolved if the two amplitudes differ
by more than 1% relative to `max(abs(slope),1e-8)`. This is a functional check,
not a probe-quality or utility endpoint.

The decision value is whether frozen HVPs suffice for this local question or a
time-varying JVP pipeline deserves larger-model engineering. Mere amplification
adds little. Even a positive approximation failure neither forecasts future
curvature nor demonstrates useful intervention, novelty over existing transient
sensitivity methods, chaos or NS. Any utility test would need its own proposal.

## Before implementation

Seek design approval, then committed code/run card, synthetic linear/quadratic
controls and independent source review. Profile the exact operations at saved
**profile seed 28160, update 50**, excluded from target interpretation; feasibility
must depend only on resources. Suggested ceilings: CPU/BLAS 1, nice 10, float64;
profile 30s +5s cleanup, fixed collection 150s +5s, analysis 15s, **256 MiB new raw
storage**. These are proposed limits, not measured feasibility or authorization.
If the complete design will not fit, preserve the profile and stop. No GPU
reservation, paid call, rate intervention or follow-on is requested. Retain all
branches, tangents, failures and hashes in an exclusive new attempt.

This is a candidate, not the default next run. Decline it if even resolving the
approximation would not change the larger-model plan. Stopping the geometry-
utility search remains a scientifically valid response to the negative results.


## 10 October 2026 — implementation under renewed all-track authorization

The human renewed all research directions and asked to strengthen the contribution. Implement the fixed mechanism comparison above on consumed development checkpoints; no additional user approval is inferred from this older design-only note. Earlier failed forecasting gates and sealed assessment data remain unchanged. No new corpus, rate, direction, amplitude, seed or horizon search. The operator, model, ridge0.001 and physical rates exactly match saved data; float32 states promote exactly to CPUfloat64 for a new counterfactual trajectory.

Operation inventory per target case:145native objective gradients (17reference including terminal,128perturbed),64HVP (two directions,16steps, frozen and transported),24trainingCE evaluations (two directions,twoamplitudes,two signs,threehorizons). Fourcases total580grad/256HVP/96CE. Gradient direction reuses reference step0; functional readout gradients are objective gradients minusridge*state, never separate objective/CE acquisitions. No probe arrays are evaluated. Retain each trajectory and tangent, partial reference/latest accepted state on interruption, all slope and vector responses, and exact operation counts. Transported agreement is retrospective calculus validation and is not online forecasting.

Profile seed28160 at update50 for both8/256arms: two gradient/HVP/CE evaluations each, exact same native source and historical parentmanifest. Use maximum measured cost per operation and arm. Projection2*2*sum_arms(145*tgrad+64*thvp+24*tCE)+20s mustfit300s. This replaces the unmeasured proposed150s envelope with a fixed300s ceiling before target observations. Profile60s/80+5external; target300s/320+5external; oneCPUthreadnice10,4GiBmonitoredRSS,256MiBworker+1MiBlauncher,8GiBhostheadroom. NoGPU/paidresources. Profile andtarget source/library identities mustmatch. Never reuse old float32 future states as a float64 reference.

Six fixed mathematical controls cover quadratic exact response/call accounting, changing-curvature frozen failure, wrong-derivative rejection, and unique latest-reference retention on interruption, and complete measured operation-cost validation. Target launch independently recomputes the profile forecast and checks complete counts. Source/type syntax is checked before any run; early editing syntax failures produced no research output. Independent source/math review and a committed run card precede the profile and later measured target.

Independent source review requested persistence of completed training-CE acquisitions before a later branch failure. Each branch now retains its completed horizon/CE pairs in its cleanup block, with an explicit interruption-after-first-CE control. The change affects evidence retention only; it adds no model operation or selection.
