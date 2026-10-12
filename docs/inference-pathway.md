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
