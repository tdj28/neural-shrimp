# Consult 19 brief: relevance of the results to loops in current LLM systems, especially looped transformers

## Context

Our arXiv article (Chaos style, 25 pages) studies the soft-token map of released pretrained language models:

    F(x) = E^T softmax(z_alpha(x) / T),

a model fed back its own expected next-token embedding (as in Soft Thinking), with a fixed prompt prefix and one
fed-back slot, so the loop is an autonomous map of R^896 (Qwen2.5-0.5B; scans also of 1.5B-72B models). Main
results:
- temperature and steering move the loop between periodic windows and chaos (positive finite-time Lyapunov
  exponents, the standard numerical test);
- the derivative factorizes exactly as DF = (1/T) Sigma_p J. Sigma_p is the covariance of token embeddings under the
  next-token distribution (a cross-covariance for untied embeddings) and J is the network's Jacobian. This gives a
  high-temperature contraction bound, a conditional low-temperature exclusion, and a split of steering's added growth
  into readout and network parts along recorded orbits;
- folds with normal forms predicted from network derivatives (42 of 48 passage times within one iterate, prospective);
  a Feigenbaum cascade; shrimp-shaped windows with located cusps;
- interval-arithmetic proofs of attracting periodic orbits of the full transformer map;
- in chaotic regimes a batch-layout change alters the text (173 of 192 chaotic float64 loops, 7 of 192 periodic).

In the Discussion we report a second system: Huginn, an open recurrent-depth ("looped") transformer that iterates a
shared block in depth for each token (Geiping et al. 2025).
- At a GSM8K-style prompt its native loop converges to a fixed point whose leading Jacobian pair has modulus 0.970.
- For the relaxed loop h <- (1 - lambda) h + lambda G(h), the Jacobian predicts a Neimark-Sacker bifurcation at
  lambda_c = 1.0777. A fit of the invariant circle's amplitude gives 1.0778.
- A generation test fixed in advance confirmed the predicted accuracy loss at lambda = 1.20 and its absence at 1.05.
- On ARC-Challenge development questions, 6 of 30 have an unstable native fixed point (spectral radius 1.009-1.273),
  and the native loop oscillates; under-relaxation restores convergence.
- We have not observed chaos in Huginn's loop at normal settings.

## Why this matters now

Press reports (The Information, 1 September and October 2026, via secondary outlets such as AI Weekly on 14 October
2026) say that OpenAI's recent or upcoming models use "loop transformers" (recurrent depth). These include a claim that
a Microsoft page mentioned "two inference passes instead of three". OpenAI's chief scientist called the September
reporting confused. OpenAI's official GPT-6.1 Sol system-card addendum reports safety numbers but no architecture. We
will NOT cite any of this as fact in the paper. The academic lineage of looped / recurrent-depth transformers is
real and recent:
- Universal Transformers (Dehghani et al., ICLR 2019);
- Looped Transformers as Programmable Computers (Giannou et al., ICML 2023);
- Yang et al. (ICLR 2024); Gatmiry et al. (ICML 2024); Fan et al. (ICLR 2025);
- recurrent depth (Geiping et al. 2025); Mixture-of-Recursions (Bae et al. 2025); Ouro / looped language models
  (Zhu et al. 2025); DeepLoop (2026); Training-Free Looped Transformers (2026);
- the closest dynamical precedent, Kim (2026), "Dynamical Phase Selection Controls Compute Scaling in Looped
  Transformers", a fold analysis of a small task-trained looped transformer, which we already cite.

An external reader's assessment (paraphrased) recommended three changes before arXiv:
1. make the architecture boundary unmistakable: our map is autonomous soft-token feedback, not an arbitrary
   recurrent-depth transformer and not ordinary generation;
2. make the proof artifacts immediately auditable;
3. give Huginn more weight, without overextending.
It also proposed a hypothesis: "an inference loop's operating regime may be as important as its nominal number of
iterations". And it gave a caveat: a model running only two or three passes reaches no asymptotic attractor in
deployment, so conclusions about the update operator must be validated at the real loop count.

## Proposed text (please check and improve)

**(a) Introduction, after "We hold the prompt and the fed-back window fixed so that the loop becomes an autonomous
map":**
> This autonomous map is neither ordinary autoregressive generation, whose context grows, nor a recurrent-depth
> (looped) transformer, which passes hidden states repeatedly through a shared block within each forward pass
> [Universal Transformers; Giannou et al.; Geiping et al.]. All three are iterated neural maps and the methods used
> here apply to each, but the results below concern the soft-token map.

**(b0) Discussion, a general sentence before (b), at the human's request ("we should have a sentence or paragraph
about the possibility of looping in general to introduce chaotic dynamics"):**
> More generally, any inference scheme that feeds a network's output or internal state back as its input (soft-token
> feedback, latent reasoning on hidden states, recurrent depth) defines an iterated nonlinear map. Convergence of such
> a map is guaranteed only by special structure: the readout alone, with tied embeddings, descends a convex energy
> (Sec. III), but a full transformer block has no such energy. Where a loop converges, that is a property of the
> trained weights and settings rather than of the architecture. The soft-token map shows fixed points, cycles and
> chaos in one pretrained model. In a recurrent-depth model we found convergence at default settings, native
> oscillation on some inputs and predicted loss of stability under over-relaxation, but no chaos.

Human follow-up question to address in the review: "But the internal loops like in the rumor are more like
Hopfield, so no chaos?" Our answer, to be checked:
- Hopfield-type convergence holds only for gradient (energy-descending) maps, here the tied readout alone.
- A trained recurrent-depth block is not such a map in general. Its convergence is trained, not guaranteed, and the
  Huginn data show both convergence and instabilities, without chaos so far.
- With two or three passes per token, no asymptotic regime is reached in deployment.
Is this right, and how should it be put?

**(b) Discussion, replacing the present Huginn clause:**
> In the recurrent-depth model Huginn [Geiping et al.], which iterates a shared block in depth, the loop's derivative
> is the block's Jacobian, without the readout factor of Eq. (3). At a GSM8K-style prompt the native loop converges
> to a fixed point whose leading multipliers form a pair of modulus 0.970. For the relaxed loop, the Jacobian
> predicts a Neimark-Sacker bifurcation at relaxation 1.0777, and the amplitude of the invariant circle beyond it
> gives 1.0778. A generation test fixed in advance confirmed the predicted loss of accuracy at over-relaxation 1.20,
> and its absence at 1.05. For looped and recurrent-depth language models [Universal Transformers; Giannou et al.;
> Ouro; Mixture-of-Recursions], the regime of the loop (contracting, oscillating or chaotic) can therefore matter as
> much as its number of passes. With only a few passes per token no attractor is reached, so conclusions about the
> update operator must be checked at the loop count actually used.

**(d) Discussion, appended to the paragraph "Steering and predictability" (alignment implication; the human asked
whether to add a line about alignment):**
> For safety interventions in continuous-feedback systems, such as steering toward honest or authorized behavior, the
> same reasoning suggests evaluating not only an intervention's immediate behavioral effect but also the stability,
> the reachable attractors and the observability of the dynamics it produces. An intervention that works at one
> operating point may move the loop toward a bifurcation or into another basin, and a monitor of the emitted text
> may miss state dynamics that drive the loop, a concern related to, but distinct from, the reduced visibility of
> latent reasoning [Korbak et al. 2025, arXiv:2507.11473]. We tested no safety property here: whether such changes
> reach a safety-relevant decision rather than only the wording is open.

Context for (d):
- A positive Lyapunov exponent says that some perturbations grow, not what they change. Instability and alignment are
  different properties.
- The article already reports that the loop's regime carries no detectable information about the accuracy of
  ordinary sampled answers beyond the model's uncertainty about the first answer token.
- The certified coexisting period-2 and period-3 attractors are a dynamical possibility, not "aligned" and
  "misaligned" attractors.

**(c) Data availability, added:**
> The interval-arithmetic certificates, with their verification code, exact inputs and release records, are in that
> repository: the period-8 orbit in codex/certify-llm (release record certificates/llm-p8-certificate-final-001.json),
> and the coexisting period-2 and period-3 orbits, their token loops and basin neighborhoods in
> codex/coexistence-task33.

## Questions

1. **Applicability.** Is it correct that our methods apply to each loop family, and our specific factorization only
   to expected-embedding feedback? The families are:
   - expected-embedding feedback, where DF = (1/T) Sigma_p J holds exactly;
   - hidden-state feedback (Coconut-style), where DF is the network Jacobian of the loop, with no readout factor;
   - recurrent-depth blocks, where DF is the block Jacobian;
   - agentic text loops with sampled discrete tokens and tool outputs, where no derivative exists: only our
     zero-temperature limit and sensitivity results bear on them.
   Anything wrong or missing?
2. **What may the paper claim** about looped transformers from the Huginn result and the soft-token results? Where is
   the line between a fair "so what" and an overreach? Is the "operating regime vs number of passes" sentence
   defensible, and in what form?
3. **Wording.** Please correct or tighten (a), (b0), (b), (c) and (d), to at most about 360 words in total. Is (b0)
   accurate as stated: no architectural guarantee of convergence, and all three regimes shown here? Say whether naming
   any press report in the paper would be a mistake (we plan not to).
4. **Novelty.** Is the positioning against Kim (2026) and the looped-transformer literature fair? Is anything we
   should cite missing?
5. **Next experiment.** The single most informative follow-up for the journal version (for example, Lyapunov spectra
   and multipliers of a pretrained looped LM's update operator against task accuracy and reproducibility, at the
   deployed loop count and beyond). What should we watch for?
6. **Alignment line (d).** Should the paper include it at all? If yes, in what form? Does it overreach given that no
   safety property was tested?
