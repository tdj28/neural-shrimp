# Consult 8 brief: is Qwen2.5-72B's absence of chaos real?

14 October 2026. Paper: "Bifurcations and chaos in language models fed their own output" (cs.LG, nlin.CD). The human
reviewer finds it suspicious that Qwen2.5-72B shows no chaos at all, while Qwen2.5-14B on the same prompt is chaotic
over half a decade of temperature. We ask whether the absence is a property of the model or of our computation, which
diagnostics settle it most cheaply, and how the paper should state the result. Remaining paid compute is a few
two-GPU pod-hours; a local 16 GB GPU is free.

## System and measurement (identical for every model)

- Soft-token loop: a fixed chat prefix (system prompt, user opener "Hello!", assistant header; its keys and values
  are cached) followed by a window of one soft token x in R^d. One step: z = logits(prefix, x), p = softmax(z/T),
  x' = E^T p, where E is the input-embedding matrix (vocabulary V = 152,064 for Qwen2.5). Steering, when used, adds
  alpha v to the residual stream after one decoder layer.
- Precision: float64 everywhere. The checkpoints are stored in bfloat16; decoder weights stay in that format and are
  converted exactly to float64 at each use. RMSNorm and rotary tables are patched to float64, and attention uses SDPA.
  Models up to 32B run on one GPU. The 70B-class models are split across two GPUs (accelerate, device_map "balanced").
- Exponent: largest finite-time Lyapunov exponent from renormalised twin trajectories at an absolute offset delta.
  Maps use 300 + 300 steps; lines use 1000 transient and 2000 measured steps. Chaotic means exponent > 1e-3 at every
  offset used and no state period up to 64 (return within 1e-8 max(1, |x|)).
- Plateau test: repeat at three offsets between 1e-4 and 1e-7 and accept a cell only when all are positive.

## Size evidence (paper's prompt, unsteered unless stated)

| model | chaotic temperatures | where (log10 T) | note |
|---|---|---|---|
| Qwen2.5-0.5B | 7 / 512 | 0.16 - 0.21 | |
| Qwen2.5-1.5B | 55 / 224 | 0.29 - 1.03 | near-uniform distributions |
| Qwen2.5-3B | 3 / 224 | 0.44 - 0.49 | |
| Qwen2.5-7B | 89 / 735 | -0.11 - 0.38 | |
| Qwen2.5-14B | 53 / 224 | 0.52 - 1.14 | entropy 11.6 - 11.9 nats (log V = 11.93), exponents up to 1.14 |
| Qwen2.5-32B | 1 / 224 | 0.58 | |
| Qwen2.5-72B | 0 / 191 | none in -1 - 1.2 | two GPUs |
| Llama-3.1-8B / 70B | 116 / 735; 3 / 128 | | two of the 70B's three pass the plateau test (0.23 - 0.41) |
| Llama-3.3-70B | 0 / 128 | | steered: 4.3 % of a 32 x 32 map, 22 % of a 64 x 64 zoom; 12 of 12 sampled zoom cells pass the plateau test |
| OLMo-2 1B / 7B / 13B / 32B | 5; 41; 0; 25 | | the 32B is the most chaotic OLMo model |

- Frozen prompt-blocked test, ten prompts: Qwen2.5-14B is chaotic at 9 - 17 of 48 temperatures against 0 - 7 for the
  7B. Its chaos lies at log10 T 0.36 - 0.90, above its own entropy transition. Our prediction, that the larger model
  is less chaotic, failed: the chaos moves to higher temperature. A second frozen test found OLMo-2-32B more chaotic
  than OLMo-2-7B on 3 of 4 prompts.
- Intermediate-entropy breadth is the number of decades of log10 T in which the mean entropy on the orbit lies between
  1 and 10 nats. It ranges from 0.014 (Qwen2.5-72B, the smallest of the 15 models) to 0.69. Its Spearman correlation
  with chaotic length is +0.55, against -0.44 for log parameters.
- No property of a single forward pass predicts the chaos: in a frozen test, the one-step gain at the start state is
  negative in every model.

## Qwen2.5-72B in detail

1. log10 T from -1 to 0.3, 128 temperatures: every cell is periodic. 125 follow the "!Hello" echo (period 2), and
   the largest exponent is -2.76. Next-token entropy stays at or below 0.15 nats up to T = 2.
2. log10 T from 0.3 to 1.2, 64 temperatures: the entropy jumps from 0.5 to 11.4 nats between 0.357 and 0.386. Above
   the jump there is a near-uniform fixed point with exponents from -1.2 to -4.8. The largest exponent anywhere is
   -0.45.
3. Gain split along the orbit (network factor plus readout factor): both factors lie below Qwen2.5-7B's in every
   entropy bin.
4. Steered map: the hostility concept vector at layer 40 of 80, alpha 0 - 0.12 and log10 T -0.3 - 0.3 (32 x 32),
   measured at offset 1e-6 only. Eight of 1024 cells (0.8 %) are labelled chaotic, with exponents up to 0.354. All
   eight lie at log10 T 0.26 - 0.30, the top edge of the map, with alpha 0.058 - 0.097. Five of them have entropy
   near 11.1 nats. They have not been plateau-tested.
5. Lines at the entropy jump (log10 T 0.370 - 0.378). The orbits alternate between a sharp and a nearly uniform
   distribution: mean entropy 5.8 - 7.7 nats (standard deviation 3 - 5), two leading tokens, attractor diameter about
   one. Exponent against offset at the same temperatures:

   | offset | positive | median | range |
   |---|---|---|---|
   | 1e-4 | 0 / 16 | -0.75 | -1.16 to -0.01 |
   | 3e-5 | 0 / 16 | -0.60 | -1.14 to -0.01 |
   | 1e-5 | 0 / 64 | -0.61 | -1.34 to -0.05 |
   | 1e-6 | 16 / 64 | -0.13 | -1.12 to +0.35 |
   | 1e-7 | 60 / 64 | +0.79 | -1.14 to +1.24 |
   | 1e-9 (earlier line, 0.355 - 0.395) | 64 / 64 | +2.73 | +0.86 to +3.82 |

   Four of the eight cells whose state repeats (periods 10 - 54) are positive at 1e-7. Of the four cells negative at
   1e-7, three are the near-uniform fixed point (entropy 11.40), where all offsets agree at about -1.1, and one is a
   period-16 cell at -0.05.

## New finding: the two-GPU computation is not float64-accurate

We measured the residual wander of the trajectory itself at stable period-1 cells: the diameter of the last 400
states, which for a deterministic float64 map should fall to rounding level. Median per run:

- Qwen2.5-7B, one GPU: 1.4e-15 (164 cells).
- OLMo-2-32B, one GPU, same bfloat16-storage and device-auto code path: 1.4e-14 (59 cells).
- Qwen2.5-72B, two GPUs: 2.1e-8 to 2.8e-8 (24 cells).
- Llama-3.3-70B, two GPUs: 1.3e-8 to 5.9e-8 (42 cells).

The two-GPU maps are therefore noisy at about 1e-8 even at fixed points, and apparently at 1e-7 or more on the
switching orbits at the jump. That level is closer to float32 than to float64. A local dtype audit of the same code
path ran on a tiny random Qwen2, on CPU and on CUDA plus CPU with an accelerate split. Every activation was float64 (the only other outputs were
copies of the stored bfloat16 weights), one-step central differences were constant from h = 1e-2 to 1e-12, and identical inputs gave bitwise-identical
outputs. So the defect appears only on the pods' two-GPU runs. Its cause is unknown: candidates are the cross-device
path (transfers or synchronisation), the max_memory/offload placement, the library build on the pod, or genuine
amplification of float64 rounding in 80-layer models. The Llama-70B plateau tests (offsets 1e-4 to 1e-6) agreed
within 0.05, so offsets of 1e-6 and above sit above this floor in those states.

## Readings we cannot yet separate

- (N) Numerical. A two-GPU precision defect makes the computed map rough at 1e-8 to 1e-6. Positive exponents at
  small offsets are artifacts; the large-offset exponents describe the model; there is no chaos at the 72B jump.
- (S) Structural. The true float64 map has very steep but smooth features at the jump, a near-switch between the
  sharp and flat branches. Finite-size exponents at offsets of 1e-5 or more saturate across them and come out
  negative, while the infinitesimal exponent could be positive. The repeating states argue against S, unless they are
  repeats of the noisy discretised map.
- (P) Prompt. The paper's prompt is idiosyncratic for this checkpoint, whose echo is extremely confident. A frozen
  prompt-blocked test at 72B is running: prompts 2 - 5, 48 temperatures over -0.4 to 0.9, offsets 1e-5 and 1e-6, both
  of which must be positive. Its result is due today. Its 1e-6 offset may lie near the noise in switching states.
- (R) Real but shifted. As in Qwen2.5-14B, chaos at 72B could sit in a narrow band at or above the jump that our grids
  (spacing 0.014 decades above log10 T 0.3) missed.

## Diagnostics we are considering (each 15 - 40 minutes of a two-GPU pod)

- D1. Precision gate on the 72B itself:
  - one-step central differences for h from 1e-2 to 1e-12, at a fixed point, a jump state, a low-temperature state
    and a steered-cell state;
  - repeated identical inputs in different batch positions;
  - two split points of the layers between the GPUs;
  - the same at a smaller model split across two GPUs against one GPU, to isolate the multi-GPU path.
- D2. Exact tangent exponent: forward-mode JVP in float64 along the orbits at the jump and at the steered cells, with
  a renormalised tangent vector and no offset. The code exists (torch.func.jvp through the model, math attention).
- D3. Plateau test of the eight steered cells at 1e-4, 3e-5 and 1e-5.
- D4. If D1 finds a fixable defect: fix it, rerun the 72B line at the jump and the steered map's top edge (extended to
  log10 T 0.4), and recheck the Llama-70B labels at 1e-7.

## Questions

1. Given this evidence, is "Qwen2.5-72B shows no chaos" plausible as a property of the loop? Which artifacts could
   produce false negatives, not only false positives? Is the reviewer's suspicion warranted?
2. Which diagnostics, in what order, separate N, S, P and R most cheaply? Is the JVP tangent exponent the decisive
   estimand here, and what pitfalls should we guard against: non-normal transients, finite horizon, tangent alignment,
   rounding in the JVP itself? What would you add or drop?
3. What is the most likely cause of a 1e-8 noise floor that appears only when a float64 model is split across two GPUs
   with accelerate, and how would you pin it down in one short run?
4. How should the paper state the 72B result in each outcome? How should it treat the steered 0.8 %, and should chaos be
   defined by infinitesimal or finite-size exponents so that the result does not hinge on the offset?
5. Is there a theoretical reason why sharper next-token distributions (larger or more heavily post-trained models)
   would shrink, shift or remove the chaotic band? Cite only references you are confident exist, with venue and year.
