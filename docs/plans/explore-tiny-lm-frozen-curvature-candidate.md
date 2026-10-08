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
