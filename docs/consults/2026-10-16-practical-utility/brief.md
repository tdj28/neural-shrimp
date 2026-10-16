# Consult 11 brief: what is the headline practical implication of chaos in language-model feedback?

16 October 2026. Private work on "Chaos and bifurcations in continuous language-model feedback" (a long paper for arXiv,
plus a condensed Chaos article). The author is worried that the work describes rich internal structure ("like
crystallography") without a headline practical implication for how models behave or perform under realistic
inference. Resources for the next days: a local RTX 2000 Ada (16 GiB) now fully available, which runs
Qwen2.5-7B-Instruct in bfloat16 at 12-13 decoding steps/s for batches 1-16 (193 tokens/s at batch 16), and about $150
of H100 credit for float64 work (a 7B float64 measurement over all 1319 GSM8K test problems costs a few dollars).

## The system and what is established

- **Loop.** A chat prefix is fixed. Each step feeds back the expected embedding x' = E^T softmax(z(x)/T), the concept
  token of Soft Thinking (Zhang et al. 2025), with a one-token window. This makes an autonomous map of R^896 (0.5B).
  As T goes to 0 it becomes greedy decoding.
- **Derivative.** DF = T^{-1} Sigma_p J: the covariance of token embeddings under the model's next-token distribution,
  times the network Jacobian. Chaos needs competing tokens (Sigma_p large) at a temperature where 1/T still amplifies.
- **Findings.** In fifteen released models (Qwen2.5, OLMo-2, Llama-3; 0.5B-72B):
  - temperature and steering move the loop between repeating text and chaos (finite-time exponents, exact tangents);
  - period-doubling cascades (Feigenbaum ratios to 0.008 %);
  - saddle-node bottlenecks with a 1/N^2 window law (type-I intermittency) and passage times predicted in advance
    (42 of 48 within one iterate; misses 3-5 % at one fold);
  - a computer-assisted proof of an attracting period-8 orbit;
  - shrimp-shaped windows;
  - a mechanism: steering makes tokens compete, so the readout expands; a logit bias alone gives chaos.
- **Practical results so far.**
  1. *Reproducibility.* Run twice with only the batch size changed, 190 of 192 chaotic float64 loops separate after the
     number of steps the Lyapunov exponent sets; 182 of 192 periodic loops stay within 1e-2 (also 7B, 8B, 70B). Batch
     changes alter only the order of floating-point sums; the loop amplifies that difference when chaotic and damps
     it when periodic. Batch-invariant kernels remove one source, but a chaotic loop amplifies any other (hardware,
     library, precision), so the regime, not the kernel, decides reproducibility.
  2. *Soft-thinking answers.* In a 32-step soft-thinking pipeline with growing context, in bfloat16, the batch layout
     changed 56 % (0.5B) and 55 % (7B) of answers to arithmetic problems, against 0 % and 10 % for forced greedy
     decoding. Resetting the fed-back state before each step removed nearly all of these changes, so the feedback is
     the amplifier. In float64 the 0.5B pipeline's answers did not change.
  3. *Huginn (recurrent depth).* A bifurcation of the model's own iteration predicted an accuracy loss under
     relaxation, confirmed by a test fixed in advance.
  4. *Steering.* Moderate strengths (6-13 % of the residual norm) cross into chaos; the dose-response is not smooth.
- **Nulls.** Across 4000 prompts (Qwen2.5-7B, ordinary sampling), a prompt's loop regime did not predict answer
  accuracy beyond difficulty covariates; first-token entropy absorbs the raw 7-point gap. No edge-of-chaos advantage.

## Experiments planned now (all exploratory until a confirmatory test fixed in advance on fresh data)

- **E1. Soft Thinking as published.** The published method keeps only the top-n concept tokens (n of 5-30); our
  answer tests used the full distribution. The mechanism predicts top-n limits Sigma_p and so suppresses batch
  sensitivity. Design:
  - GSM8K test (1319), Qwen2.5-7B-Instruct, bf16, 64 soft steps, T = 0.6;
  - top-n in {1, 5, 15, 30, full};
  - the same problems in batches of 8 and of 16, equal-length prompts, no padding;
  - outcomes: answer flip rate, accuracy, and a per-step readout-gain proxy (largest eigenvalue of the top-64
    embedding covariance over T).

  A 40-problem smoke test: 5 of 40 answers flipped; mean gain proxy 0.137 for flipped problems, 0.080 for kept ones
  (anecdotal). Soft-pipeline accuracy is low (about 28 %) for this non-reasoning model.
- **E2. A stabilizer derived from the theory.** Cap each soft step's gain proxy by lowering its temperature; measure
  flips and accuracy.
- **E3. Exploiting chaos.** Majority vote over K = 8 runs with 1e-3 perturbations of the first concept token, against
  single runs and against sampled self-consistency. Expected to help only where runs diverge.
- **E4. Ordinary decoding.** Greedy and seeded sampling (common random numbers) chain-of-thought on GSM8K in two batch
  layouts. Do answer flips occur, are flipped problems less accurate, and do the loop's sensitivity features predict
  flips beyond first-token margin and entropy?
- **E5. Repetition and temperature.** Sampled generation over a temperature grid on open prompts. Does the loop's
  chaos-onset temperature per prompt predict the temperature below which text falls into repetition loops?
- **Float64 (running now).** Per GSM8K problem, the finite-time growth of a 1e-10 perturbation in the real
  soft-thinking pipeline (7B, top-n 15 and full) and the float64 answers. This tests whether float64 sensitivity
  predicts bf16 batch flips.

## Questions for you, as research director

1. Which practical implication is most likely to be true and most valuable to frontier-lab researchers (inference,
   interpretability, alignment)? Is the reproducibility result already the headline, and how should it be framed so
   that it is not dismissed as "known nondeterminism"?
2. Are E1-E5 the right experiments? What important candidate is missing? Examples: agent or tool loops, self-refinement,
   multi-agent conversation, test-time compute scaling, sampling diversity, hallucination or uncertainty estimation,
   training on model outputs.
3. For the most promising one or two, what minimal design would make the result convincing? Cover baselines, controls
   that rule out "this is just low margin or high entropy", sample sizes, effect sizes worth reporting, and which
   models.
4. What would be overclaiming, given the nulls above?
5. Rank the candidates by expected value per GPU-hour for the next 2-3 days.
