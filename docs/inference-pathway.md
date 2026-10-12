# Pathway: these dynamics at inference time (plan, 12 October 2026)

**Why inference needs an iterated state.** Training is a dynamical system
because the optimizer iterates the parameters. A frozen feedforward model
evaluated once has no time evolution: it can be sensitive to perturbations,
but it cannot be chaotic. Inference-time dynamics need a state that is
iterated. Three classes follow, ranked by how directly our results transfer.

## 1. Inference that runs an optimizer (direct transfer)

**Which models.**
- Test-time-training layers take gradient steps on a self-supervised loss for
  each token ([Sun et al. 2024](https://arxiv.org/abs/2407.04620)).
- The Titans memory update is equivalent to mini-batch gradient descent with
  momentum and weight decay, applied to a small MLP memory during inference
  ([Behrouz et al. 2025](https://arxiv.org/abs/2501.00663)).
- Energy-based models that infer by gradient descent on a learned energy, and
  deep-equilibrium solvers, also belong here.

**What transfers.** In these models our maps are the inference dynamics.
- **The flip.** The fixed point flips at `eta h = 2 (1 + beta)`.
- **The route.** The 2-cycle then loses stability by Neimark-Sacker, giving a
  quasi-periodic memory state, tongues and chaos.
- **Momentum.** It controls whether the onset is soft or hard; on one ray the
  switch is a Chenciner point near `beta = 0.07`.
- **Weight decay.** The forget gate plays its part. In our runs, weight decay
  bounds the escape to saturated weights and decides which instability comes
  first (`docs/shrimp-dynamics.md`).

**Periodic inputs.** The input stream drives the map. A periodic input makes
the one-period map autonomous, which is the classic setting for Arnold tongues
and shrimps.

**Pilot (CPU).**
- A Titans-style MLP memory fed a periodic key-value stream.
- Scan the inner learning rate, momentum and forgetting.
- Use full-state periodic orbits, Floquet multipliers and rotation numbers.
- Compare the result with the training-map predictions.

## 2. Looped and recurrent-depth models, continuous latent reasoning

**The setting.** With frozen weights, the latent state is iterated:
`h_{k+1} = R(h_k; context)`. In a 3.5B recurrent-depth model, latent states
converge for many tokens but follow "orbits" and "sliders" for others
([Geiping et al. 2025](https://arxiv.org/abs/2502.05171)).

**What our tools add.** Full-state diagnostics can classify these motions:
- periodic, and with which period;
- an invariant circle, with its rotation number;
- chaotic, from the Lyapunov spectrum.

They can also show how each motion arises as a control parameter varies. A
natural control is a relaxation step, `h <- h + alpha (R(h) - h)`, which plays
the learning rate's role: the fixed point flips when `alpha` times the leading
eigenvalue of `I - DR` passes 2.

**Pilot.**
- A small looped transformer trained on a toy task (CPU).
- Later, the open 3.5B model on a rented GPU, inference only. That needs a
  costed run plan and approval.
- First, check recent looped-model analyses for overlap, for example
  arXiv 2604.11791 and arXiv 2607.14427 (not yet read).

## 3. Fixed-weight recurrent networks (started)

The paused random-RNN lane found repeatable periodic windows in the gain-leak
plane of fixed random RNNs (`docs/explore-clark-rnn.md`). A trained RNN would
be the inference version of that test.

## Lower priority

**Autoregressive decoding.** Greedy decoding is a deterministic map on
discrete token contexts, and repetition loops are its attracting cycles. The
state is discrete, so continuous bifurcation theory, shrimps included, does
not apply directly. The repository's float32/float64 pilot found no
precision-induced token changes.

**Indirect effects.** If training ends on an invariant circle or a chaotic
attractor, predictions vary with the stopping time. That is a training effect
seen at inference, not inference dynamics.

## Pilot results (12 October 2026)

Tool: `tools/explore_inference_memory.py` (tests:
`tests/test_inference_memory.py`). Outputs:
`../neural-shrimp-data/explore/inference-memory/`. The memory is a
13-parameter tanh MLP holding the S2 teacher's key-value pairs, repeated as a
24-token context.

- **Simplest Titans-like setting** (one inner learning rate, per-token
  updates):
  - Over most of the stable range the memory synchronizes: it repeats
    exactly with the context.
  - There are small period-doubling regions, more with momentum 0.5.
  - Divergence starts near an inner rate of 1.1.
  - Thin bands of very slow convergence are curves where a real multiplier of
    the synchronized memory approaches +1. They are not new dynamics.
  - No quasi-periodic memory appeared.
- **Exact bridge to training.**
  - With one update per context chunk, per-layer rates and coupled weight
    decay, the memory map equals the S2 training map (tested).
  - The memory settles on the period-18 shrimp orbit (attracting, spectral
    radius 0.879).
  - A rate-plane scan reproduces the window region: periods 8, 10, 16, 18,
    20 and higher, plus quasi-periodic and chaotic zones.
- **Per-token updates with the same total step per context.**
  - Every one of the 10,000 cells of the window region synchronizes.
  - The route needs each individual update to be large relative to the
    curvature of its own loss. Many small per-token steps behave like a
    stretch of gradient flow, which has no edge-of-stability instability.
- **Update granularity decides the route.** Same memory and context, with
  the same total step per context unless noted:
  - **One update per context (24-token chunk):** the training map's route —
    flip, 2-cycle, Neimark-Sacker, shrimp windows.
  - **12- and 6-token chunks:** mostly synchronized. Caveat: with an even
    number of updates per context, a per-update bounce repeats every context
    and is counted as synchronized.
  - **3-token chunks:** the synchronized memory itself undergoes
    Neimark-Sacker (log10 hidden rate 1.556, 0.141 turns per context). The
    memory becomes quasi-periodic, with a rotation number that varies
    smoothly and locks in narrow tongues. Attracting period-6 and period-11
    orbits were verified at the 1/3 and 4/11 stripes.
  - **Large per-token steps (unscaled rates):** the synchronized memory
    flips (log10 hidden rate 1.057 at log10 output rate -1.0), so the memory
    repeats every two contexts. Higher and odd periods and irregular motion
    follow.
  - **Small per-token steps:** synchronized everywhere.
- **Implication.** Test-time memories have the same families of
  bifurcations as training. Which one appears depends on how large each
  update is relative to the curvature of the loss it uses. Whether a trained
  model operates in that range can be measured from its learned inner rates
  and the curvature of its chunk losses. That is the next step, together
  with criticality at the chunk-3 crossing and the effect of context order.
