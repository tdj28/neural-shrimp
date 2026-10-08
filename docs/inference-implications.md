# Could training dynamics affect inference?

Exploratory interpretation, 8 October 2026. No inference experiment has run.

**Plain-language finding:** our results describe how weights evolve during
training. They do not establish chaotic inference. Training may leave a model
whose predictions are more or less sensitive, but that link must be measured.
The present tiny-transformer pilot has neither a verified training attractor
nor an inference-chaos result, and its later training-burst calibration failed
the incremental-utility gate.

## Two different dynamical systems

For full-batch gradient descent the state is the parameter vector and the local
update derivative is `I - eta * Hessian(loss)`. Saving a checkpoint freezes that
state. Input sensitivity of a fixed-weight forward pass instead depends on the
derivative of the model output with respect to a continuous input representation;
discrete token changes require finite-perturbation comparisons. Recurrent inference
depends on derivatives of the hidden-state transition. These operators are not
interchangeable; a training multiplier is not an inference multiplier.

Parameter motion need not even change the represented function: the linear
factorization `f(x) = u*v*x` is identical at `(u,v)` and `(-u,-v)`. This is an
algebraic illustration, not a claim that any particular saved orbit has this
symmetry. Evaluate changes in predictions as well as changes in weights.

A transformer forward evaluation has finite depth, while autoregressive
generation feeds previously selected tokens into later predictions. The latter
is a feedback process with a different state, including context and any cache,
decoding state and randomness. See [Vaswani et al., section 3](https://arxiv.org/html/1706.03762v7).
An argmax change near a tied pair of logits can change a token and all later
context. Sampling provides a separate explicit source of variability. Neither
observation alone establishes deterministic chaos or a positive Lyapunov exponent.
Smooth-map orbit/Floquet methods cannot be applied to discrete token selection
without first defining and justifying the relevant dynamical model.

Fixed-coupling recurrent neural activity can exhibit deterministic chaos: the
random recurrent model of [Sompolinsky, Crisanti and Sommers (1988)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259)
is established prior work. It is not a result about modern transformer inference
or a consequence of the optimizer bifurcations studied here.

## Practical questions worth testing

The strongest bridge would be whether a training-dynamics measurement predicts
deployment sensitivity beyond ordinary training/validation loss and output
confidence. Candidate endpoints are checkpoint selection, resilience to weight
or arithmetic perturbations, and choosing precision where it changes task
quality. A useful intervention must preserve or improve quality at measured
cost; mere text disagreement is not necessarily an error. Repetition and
decoding quality also depend on the decoding rule itself, as shown by
[Holtzman et al.](https://arxiv.org/html/1904.09751v2).

Numerical implementation is a necessary control. [Thinking Machines Lab's
batch-invariance investigation](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
demonstrates how changing batching/reduction choices can alter LLM outputs.
This is a competing explanation for divergence, not evidence of mathematical
chaos. The same applies to precision and backend differences documented by
[PyTorch](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

A simple benchmark is the top-two logit gap `m`: if every logit perturbation is
rigorously bounded in magnitude by `epsilon`, then `m > 2*epsilon` guarantees
the same unique greedy token at that prefix. A measured difference between two
implementations is only an empirical comparison, not such a rigorous bound.
Any dynamics-based predictor should outperform a suitably matched gap-based
baseline rather than merely rediscover near-tie token switches.

## Possible first diagnostic, not an executable run card

Use all six retained final tiny-LM calibration checkpoints and already-consumed
training prefixes, with no new training or protected-text evaluation. Compare
the same rounded weights in float64 and float32, with repeated same-device
evaluations as controls. First hold token prefixes identical to isolate logit
sensitivity; then use a fixed greedy decoding rule and explicit context-window
convention for a short generated sequence to measure feedback consequences.
Record logit differences, margins, first token divergence and repetition.
Separate device and dtype effects with same-device comparisons where supported.

That pilot would measure whether numerical differences affect outputs for these
checkpoints, prefixes, arithmetic settings and decoding horizon. It would not
establish generalization, chaos, training-to-inference
causation or incremental geometry value. The six training cases are confounded
by rate and final training quality and are not six independent confirmations of
a relationship. A later predictive test needs matched quality, stronger simple
baselines, fresh development data and independent training replicates.

Before execution, prepare and commit a dated run card, exact checkpoint/prefix
selection, perturbations, controls, analysis and measured resource cap. This
new inference idea does not reopen the failed training-prediction gate, the
assessment seeds/text or the final 10% holdout, and does not displace the ongoing
mechanism, CIFAR, training-utility or certificate tracks.
