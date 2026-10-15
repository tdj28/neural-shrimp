# Consult 9 brief: are we missing anything huge? Practical significance, mechanism, framing

15 October 2026. Paper: "Bifurcations and chaos in language models fed their own output" (cs.CL, cross-list nlin.CD).
The author's directive tonight is to make the findings rock solid and to show why they matter to frontier-lab
researchers in interpretability, alignment and inference: in the abstract, perhaps in the title. We ask a
director-level question: what large thing are we missing? About 7 hours remain before the author's morning. We have
several H100-hours and a few two-H200 hours. A 0.5B model is fast, a 7B is feasible, and a 70B is slow.

## System and measurement

- **Soft-token loop.** A fixed chat prefix is cached, followed by one soft token x in R^d. One step is z = logits(prefix, x), p = softmax(z/T), x' = E^T p, where E is the input embedding. Steering adds alpha v to the residual stream after one block.
- **Precision.** float64 throughout, including norms and rotary tables. Large models are split over two GPUs, with a fix for a hook-precision defect found on 14 October.
- **Chaos label.** The largest finite-time Lyapunov exponent from renormalised twins exceeds 1e-3 at every offset, and the state has no period up to 64. Exact tangents agree with the twins. Plateau tests run over offsets from 1e-4 to 1e-9.
- **Derivative.** DF = (1/T) Sigma_p J, where Sigma_p is the embedding covariance under p and J is the network Jacobian. With no network this is a modern Hopfield update, which cannot cycle.
- **Gain split.** Each step splits exactly into a network factor and a readout factor.

## Established (paper and research log)

1. **Regimes.** Periodic loops and chaos appear in Qwen2.5 (0.5B to 72B), OLMo-2 and Llama-3. Chaos sits at intermediate temperatures, and most dense models are chaotic somewhere without steering. Steering widens the chaotic region and moves it. Every chaotic attractor tested has one positive exponent, with Kaplan-Yorke dimension 1.0 to 2.0; the dimension grows with temperature.
2. **Classical universality, quantitatively** (Qwen2.5-0.5B, one temperature).
   - Flip ratios of a doubling cascade: 14.9, 3.12, 4.499, 4.621, 4.662, 4.667, against Feigenbaum's 4.669. Superstable ratios: 4.572 and 4.643.
   - Periodic windows accumulate on a fold with a 1/N^2 law at 0.5B, 1.5B and 7B.
   - Fold normal forms predicted passage times without fitting: 23 of 24 frozen forecasts within one iterate.
   - All nine reversals of the fold curves are located cusps. Flip normal forms predict the doubled orbit even where the next multiplier modulus is 0.3-0.4.
   - Return maps are unimodal.
3. **Certificate.** Interval arithmetic proves that one period-8 orbit exists and is locally attracting.
4. **Shrimps, verified negatively so far.** A shrimp-shaped period-7 domain has superstable curves that do not cross: an avoided crossing with a gap of about 1e-3 in log10 T. A visually crossing pair of period-5 and period-6 strips has both orbits unstable at the crossing. Nothing is labelled a shrimp.
5. **Hidden chaos.** Exactly repeating leading-token text can sit over a chaotic internal state (62 cells).
6. **Size.** Chaos does not decline monotonically with size. Under a frozen test, a 14B model is more chaotic than a 7B on 10 prompts; our prediction failed and the chaos moves to higher temperature. Qwen2.5-72B is nearly free of chaos unsteered but chaotic when steered.
7. **Steering block** (0.5B, 128 x 128 maps): the chaotic share for blocks 4/8/12/16/20 is 5.9/8.1/6.8/5.0/2.3 %.
   - Early blocks carry chaos down to log10 T -0.9, near greedy decoding. Block 20 confines it to 0.1-0.25.
   - Two directions at two blocks give chaos where neither push alone is chaotic.
8. **Published inference methods.** Chaos survives Soft Thinking's growing context and a 16-token window (7B). In the recurrent-depth model Huginn, a frozen test confirmed a spectrally predicted accuracy loss.
9. **Behaviour pilots (7B).**
   - Ordinary steered inference in chaotic against periodic zones: at the maps' steering strengths, steering swamps the task.
   - Each prompt's own loop regime against its ordinary-decoding accuracy: chaotic-loop prompts are much less accurate, mostly through the task mix. A within-task residual of -4 to -5 points is not established.

## Running tonight

- **MECH.** The gain split along steering lines at five blocks, with tangent-free readout statistics: entropy, top-2 logit gap, the readout Jacobian's largest eigenvalue and its stable rank.
  - A readout-only intervention: a logit bias along W v with no residual steering. Does creating token competition alone give chaos?
  - Stable rank against Kaplan-Yorke dimension.
- **Llama-3.3-70B steered at blocks 20, 35 and 65.** Block 50 gave 4.3 %.
- **Feigenbaum.** The sixth flip ratio, and a superstable ratio from precisely located points.
- **Shrimp tests.** Two-dimensional multiplier valleys, and a zoom of the avoided crossing.

## Candidate practical experiments (not yet run)

- **REPRO.** Changing only the batch size changes float64 rounding at about 1e-15. In chaotic cells this decorrelates the trajectory by step 1500, while periodic cells agree to rounding. This was seen incidentally; bf16 would be worse.
  - Plan: divergence time against regime, under batch-composition and precision changes, at 0.5B and 7B, including Soft Thinking.
- **BP confirmatory.** About 2000 fresh single-task arithmetic prompts, each labelled by its own loop regime at log10 T 0.40, with covariates fixed before sampling and a frozen analysis.
- **Steering envelope.** For each block, the steering strength and temperature that keep a loop periodic.

## Proposed framing

- **Title candidates.**
  - (a) "Steering can make language-model feedback loops chaotic"
  - (b) "Chaos in language models fed their own output: universality, mechanism and consequences"
  - (c) Keep the title and rewrite the opening of the abstract.
- **Lead.** Inference increasingly feeds models their own continuous outputs (Soft Thinking, latent reasoning, recurrent depth). Such loops behave as low-dimensional dynamical systems with classical universality.
  - Chaos means loss of reproducibility and a blind spot for one-step analysis.
  - Steering and temperature move models into chaos in predictable ways, and scale does not remove it.

## Questions

1. What large thing are we missing: an experiment, control, confound or line of prior work that a frontier-lab interpretability or alignment researcher would raise first?
2. Given our compute and 7 hours, which practical consequence would most convince such a researcher? Please rank REPRO, BP confirmatory and the steering envelope, add any better ones, and say what would be overreach.
3. Is MECH enough to say why steering and temperature produce chaos? What single additional test would make it decisive?
4. Which title and lead do you recommend? The paper is 69 pages: what should move to an appendix?
5. Which item under "Established" do you expect a referee to break?
