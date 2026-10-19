# Do language models use loops like ours? A verified reference list

Date: 2026-10-19 (sources checked 2026-10-18/19 UTC). Literature note for the arXiv article
`/data2/chaos-journal-neural-shrimp/main.tex`; no paper was edited.

**Question.** The article studies the soft-token map F(x) = E^T softmax(z(x)/T), a model fed back its own
expected next-token embedding as in Soft Thinking, whose derivative factorizes as
DF = (1/T) Cov_p(e) J_network (a cross-covariance for untied embeddings). Do current language models run loops
like this, so that the results matter, and does the derivative structure apply to them?

**How entries were verified.** Every entry in the tables was checked against its arXiv abstract page
(`arxiv.org/abs/ID`; the export API was rate-limited with HTTP 429 during the check, so the abstract pages' citation
metadata were used) for exact title, authors and first-submission date (the five entries without an arXiv record
against Crossref or the PMLR paper page), and its venue against Crossref
(`api.crossref.org/works/DOI`), the PMLR volume index, the ICLR proceedings index (`proceedings.iclr.cc`), or the
arXiv comment or journal-reference field, as stated per row. Method details beyond the abstract (Soft Thinking,
Mixture of Inputs, CIPHER, CoT2, the models in Lai et al.) were read in the arXiv HTML full text. Relevance sentences
paraphrase the abstract unless a row says otherwise. Entries that could not be verified are listed in a separate
section, and press reports are kept apart from the literature.

**How DF applies (column legend).**

- **Exact**: an expected embedding is fed back, so DF = (1/T) Cov_p(e) J holds step by step, with p the distribution
  the method actually uses (filtered and renormalized, Gumbel-perturbed, or at its own temperature). In a growing
  context the network factor is the Jacobian with respect to each earlier fed-back input.
- **Exact factor, modified**: the readout factor is present but rescaled, gated, blended with a sampled token, or
  perturbed by noise; the row says how.
- **Readout absent**: a hidden state is fed back (possibly through a fixed or learned projection), so DF is the
  network Jacobian of the loop alone: no 1/T and no covariance factor.
- **Block Jacobian**: a block or the whole network is iterated in depth, h <- B(h; context), so DF = dB/dh at fixed
  context; with relaxation h <- (1 - lambda) h + lambda B(h) the multipliers become 1 - lambda + lambda mu.
- **Discrete**: sampled or greedy text (with tool outputs) is fed back; no derivative exists, and only the article's
  zero-temperature limit (greedy cycles) and its sensitivity results bear on the loop.
- **Not a loop**: no iterated feedback inside the model.

The last column gives the bibliography status: `cited` (in the article's `references.bib`), `report bib only` (in
`paper/llm/references.bib` of the full report but not cited in the article), or `new`, followed by the
verification source.

## Synthesis

Our results reach four kinds of loop in different ways. Expected-embedding feedback (Soft Thinking, CIPHER, CoT2,
PonderLM, Mixture of Inputs, SwiReasoning, SofT-GRPO) is our map: DF = (1/T) Cov_p(e) J holds step by step on
whatever distribution the method feeds back, so the high-temperature contraction bound, the readout-competition
account of growth, the zero-temperature greedy limit and the regime-dependent batch sensitivity all transfer; these
methods run on released models of up to 49B parameters. Hidden-state feedback (Coconut and successors) drops the
readout factor: DF is the network Jacobian, and only the dynamical methods transfer. Recurrent-depth and looped
models (Huginn, Ouro, Mixture-of-Recursions, HRM, TRM, equilibrium models) iterate a block whose Jacobian has no
softmax or temperature; our Huginn analysis shows that the methods transfer and predict behavior, and many 2026 papers
study these loops' fixed points, oscillations and transient chaos. Loops over discrete text, the documented mode of
deployed frontier reasoning models and agents, have no derivative: only our zero-temperature limit and near-tie
sensitivity bear on them, as precision studies confirm. No official document we could read describes a latent or
recurrent loop in a deployed frontier model; press reports about OpenAI's Astra are unconfirmed there.

## Already in our bibliographies

- Cited in the article (`/data2/chaos-journal-neural-shrimp/references.bib`), and re-verified here:
  `zhang2025softthinking`, `zeng2025ponderlm`, `hao2024coconut`, `geiping2025recurrent`, `bai2019deq`, `bai2021stabilizing`, `tomihari2025recurrent`, `yang2026stars`, `luo2026prediction`, `kim2026phase`, `holtzman2019degeneration`, `xu2022loop`, `wang2025attractor`, `ko2026attractor`, `verazuniga2026selffeeding`, `he2025nondeterminism`, `yuan2025nondeterminism`.
  The request's `zhang2025soft` and `geiping2025scaling` are the keys `zhang2025softthinking` and
  `geiping2025recurrent`.
- In the full-report bibliography only (`paper/llm/references.bib`), not cited in the article: `wu2025singlethreaded`, `backour2026mixture`, `rizvimartel2026illusion`, `yu2026cots`, `li2026later`, `koriagin2026nohead`, `zhu2025ouro`, `prairie2026parcae`, `huang2026loopedfixed`, `movahedi2026fixedpoint`, `viakhirev2026thinkshallow`, `wang2026taps`, `labovich2026stability`, `blayney2026mechanistic`, `buehlmaier2026layernorm`, `wang2026jacobianlens`, `park2026loopedearlier`, `welleck2022selfcorrect`, `fu2021repetition`, `leins2026drift`, `liu2025truthdecay`, `shumailov2024collapse`.
  Their BibTeX is copied verbatim in the appendix.
- New: 96 entries, with BibTeX in the appendix.

## 1. Continuous or soft-token feedback

Methods that feed back an expected embedding, or a mixture of embeddings built from the next-token distribution,
instead of a sampled token. SoftCoT and Reasoning by Superposition, named in the request, feed hidden states rather
than mixtures of token embeddings, so they are listed in family 2.

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Zhang et al. (2025). *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space*. NeurIPS 2025. [arXiv:2505.15778](https://arxiv.org/abs/2505.15778); DOI [10.52202/085713-5629](https://doi.org/10.52202/085713-5629) | Probability-weighted mixture ("concept token") of the top-n token embeddings after top-k/top-p filtering and renormalization; growing context; an entropy-based Cold Stop ends the soft phase | **Exact** (on the filtered, renormalized distribution; each step factorizes, with the network factor taken with respect to every earlier fed-back input) | The method our map is built from (our loop is its autonomous version, prompt and window fixed); its authors report that without Cold Stop the soft loop "will begin to repeat until it hits the maximum generation length", a periodic attractor. | cited (`zhang2025softthinking`); arXiv abs page; Crossref DOI |
| 2 | Zhuang et al. (2025). *Text Generation Beyond Discrete Token Sampling*. arXiv preprint. [arXiv:2505.14827](https://arxiv.org/abs/2505.14827) | E^T w with w = [H p + (beta+1-H) y]/(beta+1): posterior mean blending the distribution p with the sampled one-hot y (H = normalized entropy) | **Exact factor, modified** (covariance factor scaled by H/(beta+1), plus a rank-one term from the entropy dependence; the sampled token adds jumps, not derivative) | Training-free and reported on QwQ-32B, Nemotron-Super-49B, Gemma-3-27B and DAPO-Qwen-32B, so our readout factor enters a stochastic loop in models of 27B to 49B parameters. | new (`zhuang2025mixture`); arXiv abs page; method formula checked in arXiv HTML |
| 3 | Butt et al. (2026). *Soft Tokens, Hard Truths*. ICLR 2026. [arXiv:2509.19170](https://arxiv.org/abs/2509.19170) | Mixtures of token embeddings plus noise on the input embedding, learned with RL (hundreds of soft tokens) | **Exact factor, modified** (exact linearization plus additive noise: a random dynamical system) | Finds the best scenario is to train with continuous tokens and then decode discretely, so its deployed loop is the discrete one. | new (`butt2025softtokens`); arXiv abs page; ICLR 2026 proceedings index |
| 4 | Gozeten et al. (2026). *Continuous Chain of Thought Enables Parallel Exploration and Reasoning*. ICLR 2026. [arXiv:2505.23648](https://arxiv.org/abs/2505.23648) | Base CoT2 feeds the softmax-weighted average of embeddings; CoT2-MTS averages K sampled tokens | **Exact** (for Base CoT2; MTS is a K-sample estimate of the same map) | Theory of what continuous tokens buy (parallel traces, limited by the embedding dimension); our covariance factor is the derivative of exactly this token. | new (`gozeten2025cot2`); arXiv abs page; ICLR 2026 proceedings index; definition checked in arXiv HTML |
| 5 | Pham et al. (2024). *Let Models Speak Ciphers: Multiagent Debate through Embeddings*. ICLR 2024. [arXiv:2310.06272](https://arxiv.org/abs/2310.06272) | Weighted average of all token embeddings under softmax(logits/T), fed back autoregressively and passed between debaters that share a tokenizer | **Exact** (within each generation; debate rounds couple two such maps) | An early use of our loop (multi-agent debate); the authors note that both their embedding generation and text generation become greedy as T -> 0, the limit our paper analyzes. | new (`pham2023cipher`); arXiv abs page (comment: accepted to ICLR 2024); method checked in arXiv HTML |
| 6 | Wu et al. (2026). *LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking*. ICLR 2026. [arXiv:2508.03440](https://arxiv.org/abs/2508.03440) | Soft Thinking inputs; adds Dirichlet or Gumbel-Softmax randomness ("Stochastic Soft Thinking") | **Exact** (for vanilla soft inputs; with Gumbel-Softmax, exact for the perturbed distribution at its own temperature) | Finds models rely mainly on the top token of the soft input, a "greedy feedback loop": the low-temperature regime in which our readout factor shrinks and the loop approaches greedy cycles. | report bib only (`wu2025singlethreaded`); arXiv abs page; ICLR 2026 proceedings index |
| 7 | Zheng et al. (2025). *SofT-GRPO: Surpassing Discrete-Token LLM Reinforcement Learning via Gumbel-Reparameterized Soft-Thinking Policy Optimization*. arXiv preprint. [arXiv:2511.06411](https://arxiv.org/abs/2511.06411) | Gumbel-Softmax soft-thinking tokens (Gumbel noise injected into the logits), trained with GRPO | **Exact factor, modified** (exact for the Gumbel-perturbed distribution; the noise makes the loop a random map) | Reinforcement learning of soft-thinking policies at 1.5B to 7B parameters, slightly above discrete GRPO at pass@1 and more at pass@32. | new (`zheng2025softgrpo`); arXiv abs page |
| 8 | Shi et al. (2026). *SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs*. ICLR 2026. [arXiv:2510.05069](https://arxiv.org/abs/2510.05069) | Alternates latent (soft-thinking) blocks with explicit token blocks, switching on entropy trends of the next-token distribution | **Exact** (in latent blocks; discrete in explicit blocks) | Ties the switch between continuous and discrete feedback to the spread of p, the quantity that sets our readout factor. | new (`shi2025swireasoning`); arXiv abs page; ICLR 2026 proceedings index |
| 9 | Zeng et al. (2026). *PonderLM: Pretraining Language Models to Ponder in Continuous Space*. ICLR 2026. [arXiv:2505.20674](https://arxiv.org/abs/2505.20674) | Probability-weighted sum of all token embeddings, fed back for further forward passes within one token step (pretrained) | **Exact** (an inner loop at a fixed position) | A trained inner version of our map, iterated at one position, so the factorization, and with it our multiplier and contraction analysis, applies to each pondering step. | cited (`zeng2025ponderlm`); arXiv abs page; ICLR 2026 proceedings index |
| 10 | Wang, Duan and Du (2025). *Improving Latent Reasoning in LLMs via Soft Concept Mixing*. arXiv preprint. [arXiv:2511.16885](https://arxiv.org/abs/2511.16885) | Probability-weighted average of embeddings ("soft concept"), mixed into hidden states during RL training | **Exact factor, modified** (the readout factor is exact but enters at a hidden layer, so the network factor runs from that layer on) | Trains on soft concepts to close the gap between discrete pretraining and Soft Thinking inference. | new (`wang2025softconcept`); arXiv abs page |
| 11 | Backour (2026). *The Dynamics of Continuous Mixture Collapse in Language Models*. arXiv preprint. [arXiv:2609.02049](https://arxiv.org/abs/2609.02049) | Weighted mixtures of token embeddings (latent-state reasoning) | **Exact** (the paper analyzes the readout-plus-feedback dynamical system itself) | Independently finds that the softmax readout with autoregressive feedback either amplifies small differences until one component dominates or contracts mixtures, with an empirical transition near its derived threshold: the mechanism of our readout factor (in the report bibliography, not cited in the article). | report bib only (`backour2026mixture`); arXiv abs page |
| 12 | Rizvi-Martel, Rabusseau and Mosbach (2026). *The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models*. arXiv preprint. [arXiv:2604.06374](https://arxiv.org/abs/2604.06374) | Training-free convex combinations of token embeddings; also fine-tuned and from-scratch latent thoughts | **Exact** (in the training-free regime; the other regimes feed learned latents) | Superposition collapses in pretrained models, which commit to a token in the last layers; only models trained from scratch use it. | report bib only (`rizvimartel2026illusion`); arXiv abs page |
| 13 | Yu et al. (2026). *CoTS: Competitive Token Selection for Latent Reasoning in Large Language Models*. arXiv preprint. [arXiv:2610.20642](https://arxiv.org/abs/2610.20642) | Blend of the sampled token embedding and the soft embedding, weighted by the probability competition between the two leading candidates | **Exact factor, modified** (covariance factor times the blend weight, plus a gating term from the top-two gap) | Uses the top-1/top-2 competition, which controls our readout factor near ties, as its control signal. | report bib only (`yu2026cots`); arXiv abs page |

## 2. Hidden-state feedback and latent reasoning

Coconut and its successors feed back a hidden state (or a learned projection of it), so the readout factor is absent
and DF is the Jacobian of the network alone. Orbits, multipliers, bifurcations and Lyapunov exponents carry over; the
article's temperature bounds and its readout-competition account of growth do not.

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Hao et al. (2025). *Training Large Language Models to Reason in a Continuous Latent Space*. COLM 2025. [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) | Last hidden state fed back as the next input embedding ("continuous thought") | **Readout absent** (DF is the network Jacobian of the loop) | The canonical hidden-state loop: no 1/T and no covariance factor, so any expansion must come from the network itself. | cited (`hao2024coconut`); arXiv abs page (comment: accepted to COLM 2025) |
| 2 | Shen et al. (2025). *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation*. EMNLP 2025. [arXiv:2502.21074](https://arxiv.org/abs/2502.21074); DOI [10.18653/v1/2025.emnlp-main.36](https://doi.org/10.18653/v1/2025.emnlp-main.36) | Continuous thoughts (hidden states) aligned to explicit CoT by self-distillation | **Readout absent** | First implicit CoT to match explicit CoT on GSM8K at GPT-2 scale (3.1x compression). | new (`shen2025codi`); arXiv abs page; Crossref DOI |
| 3 | Cheng and Van Durme (2024). *Compressed Chain of Thought: Efficient Reasoning Through Dense Representations*. arXiv preprint. [arXiv:2412.13171](https://arxiv.org/abs/2412.13171) | Dense "contemplation tokens" (compressed hidden representations) generated autoregressively | **Readout absent** | Variable-length continuous loops on off-the-shelf decoders. | new (`cheng2024ccot`); arXiv abs page |
| 4 | Zhu et al. (2025). *Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought*. NeurIPS 2025. [arXiv:2505.12514](https://arxiv.org/abs/2505.12514); DOI [10.52202/085713-2670](https://doi.org/10.52202/085713-2670) | Coconut-style continuous thoughts (hidden states) | **Readout absent** | Proves a two-layer transformer with D continuous-thought steps solves graph reachability by holding superposed search frontiers. | new (`zhu2025superposition`); arXiv abs page; Crossref DOI |
| 5 | Zhu et al. (2026). *Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought*. ICLR 2026. [arXiv:2509.23365](https://arxiv.org/abs/2509.23365) | Continuous thoughts (hidden states); analysis of training dynamics | **Readout absent** | Shows how superposition is learned: the index-matching logit first grows, then stays bounded. | new (`zhu2025emergence`); arXiv abs page; ICLR 2026 proceedings index |
| 6 | Wu, Teng and Tu (2025). *Parallel Continuous Chain-of-Thought with Jacobi Iteration*. EMNLP 2025. [arXiv:2506.18582](https://arxiv.org/abs/2506.18582); DOI [10.18653/v1/2025.emnlp-main.47](https://doi.org/10.18653/v1/2025.emnlp-main.47) | All latent thought tokens updated in parallel by Jacobi iteration | **Block Jacobian** (Jacobian of the Jacobi sweep; no readout) | Turns the hidden-state chain into a fixed-point iteration, whose convergence the Jacobian spectrum governs. | new (`wu2025pccot`); arXiv abs page; Crossref DOI |
| 7 | Yue et al. (2025). *Hybrid Latent Reasoning via Reinforcement Learning*. NeurIPS 2025. [arXiv:2505.18454](https://arxiv.org/abs/2505.18454); DOI [10.52202/085713-0195](https://doi.org/10.52202/085713-0195) | Sampled token embeddings gated together with prior hidden states (learnable gate), trained with RL | **Readout absent** (readout absent for the hidden-state part; the sampled part is discrete) | A hybrid that mixes both kinds of feedback. | new (`yue2025hrpo`); arXiv abs page; Crossref DOI |
| 8 | Tan et al. (2025). *Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains*. arXiv preprint. [arXiv:2505.16552](https://arxiv.org/abs/2505.16552) | Compressed embeddings predicted by a latent head (a distribution over next compressed embeddings, sampled in RL) | **Readout absent** (no softmax readout; the latent head's Jacobian and its sampling noise replace Sigma_p/T) | Dense latent loop with a learned stochastic readout. | new (`tan2025colar`); arXiv abs page |
| 9 | Wei et al. (2026). *SIM-CoT: Supervised Implicit Chain-of-Thought*. ICLR 2026. [arXiv:2509.20317](https://arxiv.org/abs/2509.20317) | Implicit tokens (Coconut/CODI-style hidden states) | **Readout absent** | Training becomes unstable as the number of latent tokens grows, because latent states turn homogeneous; step-level supervision fixes it. | new (`wei2025simcot`); arXiv abs page; ICLR 2026 proceedings index |
| 10 | Xu et al. (2025). *SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs*. ACL 2025. [arXiv:2502.12134](https://arxiv.org/abs/2502.12134); DOI [10.18653/v1/2025.acl-long.1137](https://doi.org/10.18653/v1/2025.acl-long.1137) | Soft thought tokens from a frozen assistant model, projected once into the LLM's space | **Not a loop** (a one-shot injection, not iterated in the main model) | Named "soft" but not expected-embedding feedback; our factorization does not apply. | new (`xu2025softcot`); arXiv abs page; Crossref DOI |
| 11 | Xu et al. (2025). *SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning*. arXiv preprint. [arXiv:2505.11484](https://arxiv.org/abs/2505.11484) | As SoftCoT, with perturbed initial tokens and a contrastive objective for diversity | **Not a loop** | Test-time scaling for SoftCoT; listed to separate it from expected-embedding methods. | new (`xu2025softcotpp`); arXiv abs page |
| 12 | Zeng et al. (2026). *PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space*. ICML 2026 (PMLR 306). [arXiv:2509.23184](https://arxiv.org/abs/2509.23184) | Last hidden state of the current position used as an intermediate latent thought before the token is predicted | **Readout absent** | The hidden-state counterpart of PonderLM, pretrained at 1.4B parameters on 300B tokens. | new (`zeng2025ponderlm2`); arXiv abs page; PMLR volume 306 index |
| 13 | Liu et al. (2025). *Deep Thinking by Markov Chain of Continuous Thoughts*. arXiv preprint. [arXiv:2509.25020](https://arxiv.org/abs/2509.25020) | Continuous thought representations mapped to the distribution of the next thought (MarCos) | **Readout absent** (a learned stochastic continuous transition; no token softmax) | A fully continuous, thought-level Markov chain. | new (`liu2025marcos`); arXiv abs page |
| 14 | Zhang et al. (2025). *Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought*. arXiv preprint. [arXiv:2512.21711](https://arxiv.org/abs/2512.21711) | Coconut continuous thoughts (analysis) | **Readout absent** | Coconut tokens are insensitive to steering and exploit shortcuts, a reliability concern for hidden loops. | new (`zhang2025latenttokens`); arXiv abs page |
| 15 | Hao and Yang (2026). *Think Wider: Mitigating Latent Rank Collapse in Implicit Chain-of-Thought Reasoning*. arXiv preprint. [arXiv:2609.07406](https://arxiv.org/abs/2609.07406) | Implicit-CoT latent states | **Readout absent** | Successive latent states collapse onto one dominant direction ("latent rank collapse"), analogous to the near-rank-one period Jacobians in our paper. | new (`hao2026wider`); arXiv abs page |
| 16 | Ramjee (2026). *Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models*. ICLR 2026 LIT workshop. [arXiv:2604.23460](https://arxiv.org/abs/2604.23460) | Latent thoughts of a continuous thought model | **Readout absent** | A continuous thought model can reason misaligned in latent space while answering aligned; linear probes detect it. | new (`ramjee2026ulterior`); arXiv abs page (journal-ref) |
| 17 | Li et al. (2026). *LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification*. arXiv preprint. [arXiv:2605.07315](https://arxiv.org/abs/2605.07315) | Final-layer hidden states projected back into the input-embedding space (training-free), then explicit CoT | **Readout absent** (a projection replaces Sigma_p/T) | Bounded latent exploration on Qwen3-14B before explicit verification. | report bib only (`li2026later`); arXiv abs page |
| 18 | Koriagin et al. (2026). *A Model with No Head and Many Thoughts*. Findings of EMNLP 2026 (accepted). [arXiv:2608.31069](https://arxiv.org/abs/2608.31069) | Hidden state through a lightweight projector that replaces the LM head during reasoning | **Readout absent** (the projector Jacobian replaces Sigma_p/T) | Removes exactly the factor to which our paper attributes the growth that steering adds: a test of whether chaos needs the readout. | report bib only (`koriagin2026nohead`); arXiv abs page (comment) |
| 19 | Shao et al. (2025). *Continuous Autoregressive Language Models*. arXiv preprint. [arXiv:2510.27688](https://arxiv.org/abs/2510.27688) | Continuous next vector (an autoencoder latent for a chunk of K tokens) | **Readout absent** (no token softmax; a generative head and the autoencoder replace it) | Continuous autoregression as the base language model, not a reasoning add-on. | new (`shao2025calm`); arXiv abs page |
| 20 | Zhu et al. (2025). *A Survey on Latent Reasoning*. arXiv preprint. [arXiv:2507.06203](https://arxiv.org/abs/2507.06203) | Survey (activation-based recurrence, hidden-state propagation, compression of reasoning traces) | **Survey** | Map of families 2 and 3. | new (`zhu2025latentsurvey`); arXiv abs page |
| 21 | Chen et al. (2026). *Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning*. Findings of EMNLP 2026. [arXiv:2505.16782](https://arxiv.org/abs/2505.16782) | Survey (token-wise "horizontal" versus layer-wise "vertical" latent CoT) | **Survey** | Taxonomy that separates our family 2 (horizontal) from family 3 (vertical). | new (`chen2025latentcot`); arXiv abs page (comment) |

## 3. Recurrent depth and looped models

### How the article's analysis applies to looped transformers

In a looped transformer the inner loop is h <- B(h; c): the same block B acts on a latent state h at fixed
context c. In "re-read" designs (Huginn, Parcae) the prompt is injected again at every pass; in "carry" designs
(Ouro) it lives only in the state (Olszowiec et al. 2026, family 3e). The derivative of this loop is the block
Jacobian dB/dh.

- **Does not carry over:** the readout-covariance factor (1/T) Sigma_p and everything the article derives from it.
  There is no softmax readout and no temperature inside the depth loop, so the high-temperature contraction bound
  T* = diam(E)^2 L / 4, the low-temperature margin bound, the attribution of the growth that steering adds to
  competition between next tokens, and temperature as a bifurcation parameter have no counterpart there.
- **Carries over:** the method. Locate fixed points or cycles of the block map, compute its multipliers, classify
  each crossing of the unit circle (fold at +1, flip at -1, Neimark-Sacker for a complex pair), continue it in a
  parameter, estimate Lyapunov exponents, and compare one-step norms with asymptotic rates (Buehlmaier 2026 makes
  the same norm-versus-spectrum point for LayerNorm loops).
- **Shown in our full report** (`paper/llm/main.tex`, section "A second system: recurrent depth"): for a GSM8K-style
  prompt, Huginn-0125's native loop converges to a fixed point whose leading Jacobian pair is 0.588 +/- 0.771i
  (modulus 0.970). Under over-relaxation h <- (1 - lambda) h + lambda B(h) the multipliers become
  1 - lambda + lambda mu, which predicts a Neimark-Sacker crossing at lambda_c = 1.0777 (amplitude fit 1.0778). With
  the full instruction template, the fixed points of five development prompts give thresholds lambda_c = 1.074 to
  1.126, and a test whose design and analysis were fixed before any test problem was read confirmed the predicted
  loss on 200 held-out GSM8K problems (accuracy 0.330 native, 0.395 at lambda = 1.05, 0.255 at 1.15, 0.095 at 1.20).
  Chen et al. (2026, damped sub-steps), Godin (2026, h <- (1 - dt) h + dt F(h)) and Ladhak et al. (2026, learned step
  size) all build this relaxation parameter into their designs.
- **Where the readout returns:** only in the outer, token-level loop of a looped LM. It enters exactly if that loop
  feeds back expected embeddings, and through near-tie token flips at the LM head if it samples (family 4c).

The released looped LMs analyzed in the 2026 papers are research-scale (for example Huginn at 3.5B and Ouro at 1.4B
and 2.6B parameters).

### 3a. Looped and recurrent-depth transformers (foundations and the models named in the request)

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Dehghani et al. (2019). *Universal Transformers*. ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819) | One block applied repeatedly to all positions, with per-position halting | **Block Jacobian** | The origin of depth recurrence in transformers; halting is a stopping time on the block iterate. | new (`dehghani2019universal`); arXiv abs page (comment: published at ICLR 2019) |
| 2 | Giannou et al. (2023). *Looped Transformers as Programmable Computers*. ICML 2023 (PMLR 202). [arXiv:2301.13196](https://arxiv.org/abs/2301.13196) | Output of a fixed 13-layer encoder fed back as its input | **Block Jacobian** | A looped transformer can emulate a small instruction-set computer, so the block map can realize iterative algorithms, contracting or not. | new (`giannou2023looped`); arXiv abs page; PMLR volume 202 index |
| 3 | Yang et al. (2024). *Looped Transformers are Better at Learning Learning Algorithms*. ICLR 2024. [arXiv:2311.12424](https://arxiv.org/abs/2311.12424) | Looped transformer with input injection, trained for in-context data fitting | **Block Jacobian** | Matches a standard transformer on data-fitting tasks with under 10% of the parameters. | new (`yang2024looped`); arXiv abs page (comment: accepted at ICLR 2024) |
| 4 | Gatmiry et al. (2024). *Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?* ICML 2024 (PMLR 235). [arXiv:2410.08292](https://arxiv.org/abs/2410.08292) | Linear looped transformer iterate | **Block Jacobian** (linear: the block Jacobian is the preconditioned gradient-descent iteration matrix) | Proves the trained loop implements multi-step preconditioned gradient descent, a case where the multipliers are explicit. | new (`gatmiry2024looped`); arXiv abs page; PMLR volume 235 index |
| 5 | Fan et al. (2025). *Looped Transformers for Length Generalization*. ICLR 2025. [arXiv:2409.15647](https://arxiv.org/abs/2409.15647) | Looped block with an adaptive number of steps | **Block Jacobian** | Adaptive step counts give length generalization on RASP-L tasks. | new (`fan2025looped`); arXiv abs page; ICLR 2025 proceedings index |
| 6 | Saunshi et al. (2025). *Reasoning with Latent Thoughts: On the Power of Looped Transformers*. ICLR 2025. [arXiv:2502.17416](https://arxiv.org/abs/2502.17416) | A k-layer block looped L times | **Block Jacobian** | Many reasoning problems need depth rather than parameters: a k-layer model looped L times nearly matches a kL-layer model. | new (`saunshi2025reasoning`); arXiv abs page; ICLR 2025 proceedings index |
| 7 | Geiping et al. (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*. NeurIPS 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171); DOI [10.52202/085713-1380](https://doi.org/10.52202/085713-1380) | Latent state s <- R(e, s) through a recurrent block with input injection, unrolled to arbitrary depth at test time | **Block Jacobian** | Huginn (3.5B parameters, 800B tokens), the model whose depth loop our full report analyzes. | cited (`geiping2025recurrent`); arXiv abs page; Crossref DOI |
| 8 | Bae et al. (2025). *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. NeurIPS 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524); DOI [10.52202/085713-3229](https://doi.org/10.52202/085713-3229) | Shared layer stack applied a token-dependent number of times (lightweight routers) | **Block Jacobian** (routing makes the block map piecewise) | Adaptive per-token recursion from 135M to 1.7B parameters. | new (`bae2025mor`); arXiv abs page; Crossref DOI |
| 9 | Zhu et al. (2025). *Scaling Latent Reasoning via Looped Language Models*. arXiv preprint. [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) | Iterative latent computation through shared layers, with learned depth allocation; prompt carried only in the looped state ("carry loop", per Olszowiec et al. 2026) | **Block Jacobian** | Open looped LMs (1.4B and 2.6B, 7.7T tokens) that match results of up to 12B models; the most-analyzed released looped LM in the 2026 papers below. | report bib only (`zhu2025ouro`); arXiv abs page (33 authors, first Rui-Jie Zhu, last Jason Eshraghian) |
| 10 | Li et al. (2026). *DeepLoop: Depth Scaling for Looped Transformers*. arXiv preprint. [arXiv:2607.13491](https://arxiv.org/abs/2607.13491) | Compact stack of physical blocks applied for several rounds (Post-LN DeepNorm) | **Block Jacobian** | Sets residual scaling for tied depth from a first-order perturbation bound, a linear-stability argument on the unrolled loop. | new (`li2026deeploop`); arXiv abs page |
| 11 | Chen et al. (2026). *Training-Free Looped Transformers*. arXiv preprint. [arXiv:2605.23872](https://arxiv.org/abs/2605.23872) | A mid-stack block of a frozen checkpoint looped at inference, as damped sub-steps | **Block Jacobian** (under-relaxed iteration) | Naive reapplication of a block of a pretrained model (Qwen3, Moonlight; dense and MoE) usually degrades it, while damped sub-steps help: the under-relaxed side of the threshold our Huginn analysis computes. | new (`chen2026trainingfree`); arXiv abs page |
| 12 | Zhang et al. (2026). *Recurrent Looped Transformer*. arXiv preprint. [arXiv:2610.07591](https://arxiv.org/abs/2610.07591) | Previous token's final decoder state merged with the encoder output at each token | **Readout absent** (hidden-state feedback across positions, not a depth loop) | Despite the name, the loop runs across tokens; removing the feedback drops parity and S5 tracking to chance. | new (`zhang2026rlt`); arXiv abs page (five authors) |
| 13 | Bansal et al. (2022). *End-to-End Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking*. NeurIPS 2022. [arXiv:2202.05826](https://arxiv.org/abs/2202.05826); DOI [10.52202/068431-1471](https://doi.org/10.52202/068431-1471) | Recurrent network iterated with recall (the input re-injected) | **Block Jacobian** | Names "overthinking", degeneration under many iterations, and fixes it with recall: an early stability problem of block iterates. | new (`bansal2022endtoend`); arXiv abs page; Crossref DOI (proceedings title lacks "Logical") |
| 14 | Csordás et al. (2024). *MoEUT: Mixture-of-Experts Universal Transformers*. NeurIPS 2024. [arXiv:2405.16039](https://arxiv.org/abs/2405.16039); DOI [10.52202/079017-0897](https://doi.org/10.52202/079017-0897) | Shared mixture-of-experts layer group (Universal Transformer) | **Block Jacobian** | First shared-layer transformer to slightly outperform standard transformers on language modeling. | new (`csordas2024moeut`); arXiv abs page; Crossref DOI |
| 15 | Bae et al. (2025). *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA*. ICLR 2025. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672) | A single block repeated in a loop, with depth-wise LoRA | **Block Jacobian** (non-autonomous: the map changes with the loop index) | Converts pretrained LLMs such as Gemma into recursive models. | new (`bae2025relaxed`); arXiv abs page; ICLR 2025 proceedings index |
| 16 | Bai, Kolter and Koltun (2019). *Deep Equilibrium Models*. NeurIPS 2019. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377) | Equilibrium z* = f(z*, x) found by root finding (infinite weight-tied depth) | **Block Jacobian** (at the fixed point; convergence and implicit gradients involve I - df/dz) | Equilibrium view of depth; a fold (multiplier +1) is where I - df/dz becomes singular. | cited (`bai2019deq`); arXiv abs page (comment: NeurIPS 2019) |
| 17 | Schöne et al. (2025). *Implicit Language Models are RNNs: Balancing Parallelization and Expressivity*. ICML 2025 (PMLR 267). [arXiv:2502.07827](https://arxiv.org/abs/2502.07827) | Implicit SSM or transformer iterated to a per-token fixed point | **Block Jacobian** | Implicit LMs up to 1.3B parameters on 207B tokens; approximate fixed-point convergence suffices for most tokens. | new (`schone2025implicit`); arXiv abs page; PMLR volume 267 index |

### 3b. Scaled recurrent-depth language models and training recipes (2025-2026)

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | McLeish et al. (2025). *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*. arXiv preprint. [arXiv:2511.07384](https://arxiv.org/abs/2511.07384) | Pretrained non-recurrent LM converted to a depth-recurrent one (curriculum of recurrences) | **Block Jacobian** | Retrofitting recurrence beats post-training the original model at a given compute budget on mathematics. | new (`mcleish2025retrofitted`); arXiv abs page |
| 2 | Geiping, Yang and Su (2026). *Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models*. ICML 2026 (PMLR 306). [arXiv:2510.14961](https://arxiv.org/abs/2510.14961) | Latent states of several tokens refined in parallel through recurrence (diffusion-forcing sampler) | **Block Jacobian** (on a window of positions) | Up to 5x faster sampling for 3.5B recurrent-depth models, which it relates to continuous diffusion LMs. | new (`geiping2026parallel`); arXiv abs page; PMLR volume 306 index |
| 3 | Koishekenov, Lipani and Cancedda (2025). *Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts*. arXiv preprint. [arXiv:2510.07358](https://arxiv.org/abs/2510.07358) | Iteration over a small subset of reasoning-relevant middle layers | **Block Jacobian** | Mid-training recursion gives +28.4% relative accuracy on GSM8K for OLMo-2 1B. | new (`koishekenov2025etd`); arXiv abs page |
| 4 | Prairie et al. (2026). *Parcae: Scaling Laws For Stable Looped Language Models*. arXiv preprint. [arXiv:2604.12946](https://arxiv.org/abs/2604.12946) | Looped block over the residual stream with input injection | **Block Jacobian** | Treats looping as a nonlinear time-variant dynamical system and traces instability to large spectral norms of the injection parameters, a linear-stability diagnosis like ours. | report bib only (`prairie2026parcae`); arXiv abs page |
| 5 | Schwethelm, Rueckert and Kaissis (2026). *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*. arXiv preprint. [arXiv:2604.21106](https://arxiv.org/abs/2604.21106) | Looped block, recurrence counts 1 to 8 | **Block Jacobian** | One recurrence is worth r^0.46 unique blocks in validation loss (iso-depth scaling law). | new (`schwethelm2026isodepth`); arXiv abs page |
| 6 | Huang et al. (2026). *Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design*. arXiv preprint. [arXiv:2610.23036](https://arxiv.org/abs/2610.23036) | Prelude, recurrent block and coda; input injection and recurrent-state designs compared | **Block Jacobian** | Under controlled training the Huginn-style design beats Ouro at 730M parameters and in MoE models. | new (`huang2026loopedtopology`); arXiv abs page |
| 7 | Huang et al. (2026). *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points*. arXiv preprint. [arXiv:2610.06833](https://arxiv.org/abs/2610.06833) | Looped LM near its fixed points; learned depth prior and orthogonal input injection | **Block Jacobian** | Proximity to fixed points enables truncated backpropagation, KV sharing and faster RL updates. | report bib only (`huang2026loopedfixed`); arXiv abs page |
| 8 | Ladhak et al. (2026). *One Model to Loop Them All: Elastic Latent Recurrence for Language Models*. arXiv preprint. [arXiv:2610.12582](https://arxiv.org/abs/2610.12582) | Shared block with a gated state update and a learned step size | **Block Jacobian** (the learned step is a relaxation parameter) | Removes the one-loop accuracy tax; its learned step size is the control our Huginn analysis varies. | new (`ladhak2026elastic`); arXiv abs page |
| 9 | Kohli et al. (2026). *Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers*. COLM 2026 (accepted). [arXiv:2604.07822](https://arxiv.org/abs/2604.07822) | Recurrent-depth transformer trained from scratch on multi-hop composition | **Block Jacobian** | Recurrence enables systematic generalization (through three grokking stages) and depth extrapolation. | new (`kohli2026loop`); arXiv abs page (comment) |

### 3c. Recursive reasoning models

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Wang et al. (2025). *Hierarchical Reasoning Model*. arXiv preprint. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) | Two coupled recurrent modules at two time scales (slow planning, fast computation) | **Block Jacobian** (nested block iterate) | 27M parameters, near-perfect on hard Sudoku and large mazes from about 1000 examples. | new (`wang2025hrm`); arXiv abs page |
| 2 | Jolicoeur-Martineau (2025). *Less is More: Recursive Reasoning with Tiny Networks*. arXiv preprint. [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) | One two-layer network recursing on a latent state and an answer | **Block Jacobian** | Simpler than HRM; 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with 7M parameters. | new (`jolicoeur2025trm`); arXiv abs page |
| 3 | Rodrigues and Kang (2026). *Dissecting Hierarchical Reasoning Models: A Mechanistic Study*. arXiv preprint. [arXiv:2609.22197](https://arxiv.org/abs/2609.22197) | HRM recurrent states (mechanistic study) | **Block Jacobian** | HRM performs constraint-aware iterative refinement of a solution state; single-state recurrent transformers do as well. | new (`rodrigues2026hrm`); arXiv abs page |
| 4 | Shakhvalieva et al. (2026). *What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling*. arXiv preprint. [arXiv:2609.39967](https://arxiv.org/abs/2609.39967) | Small shared block refining a latent state (HRM, TRM, URM compared) | **Block Jacobian** | Stable recursive reasoning needs an intermediate gradient horizon, large batches and controlled state updates; an explicit hierarchy is not needed. | new (`shakhvalieva2026recursive`); arXiv abs page |

### 3d. Dynamics, fixed points and Jacobians of depth loops

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Bai, Koltun and Kolter (2021). *Stabilizing Equilibrium Models by Jacobian Regularization*. ICML 2021 (PMLR 139). [proceedings.mlr.press](https://proceedings.mlr.press/v139/bai21b.html) | DEQ fixed point, with Jacobian regularization | **Block Jacobian** | Regularizes the Jacobian of the equilibrium map to stabilize it. | cited (`bai2021stabilizing`); PMLR paper page (citation metadata) |
| 2 | Tomihari and Karakida (2025). *Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians*. NeurIPS 2025. DOI [10.52202/085713-4675](https://doi.org/10.52202/085713-4675) | Recurrent self-attention dynamics | **Block Jacobian** | Jacobian-based stability analysis of recurrent self-attention. | cited (`tomihari2025recurrent`); Crossref DOI |
| 3 | Yang et al. (2026). *Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models*. ICML 2026 (PMLR 306). [proceedings.mlr.press](https://proceedings.mlr.press/v306/yang26t.html) | Looped LM recurrence | **Block Jacobian** | Fixed-point and Jacobian stability for test-time scaling of looped LMs. | cited (`yang2026stars`); PMLR volume 306 index |
| 4 | Luo and Yu (2026). *Prediction Dynamics in Depth-Recurrent Language Models*. arXiv preprint. [arXiv:2609.21383](https://arxiv.org/abs/2609.21383) | Latent updates of Huginn-3.5B and Ouro-1.4B | **Block Jacobian** | Margin geometry explains when intermediate answers already agree with the endpoint. | cited (`luo2026prediction`); arXiv abs page |
| 5 | Kim (2026). *Dynamical phase selection controls compute scaling in looped transformers*. arXiv preprint. [arXiv:2608.26556](https://arxiv.org/abs/2608.26556) | Weight-tied map of a small task-trained looped transformer | **Block Jacobian** | Closest precedent: fold and Neimark-Sacker phases set how test-time compute scales. | cited (`kim2026phase`); arXiv abs page |
| 6 | Pappone, Crisostomi and Rodolà (2025). *Two-Scale Latent Dynamics for Recurrent-Depth Transformers*. arXiv preprint. [arXiv:2509.23314](https://arxiv.org/abs/2509.23314) | Iterates of recurrent-depth transformers | **Block Jacobian** | Loop steps become smaller and more orthogonal during training; early exit from second differences of the step size beats a KL criterion. | new (`pappone2025twoscale`); arXiv abs page |
| 7 | Lu et al. (2025). *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer*. COLM 2025 workshop. [arXiv:2507.02199](https://arxiv.org/abs/2507.02199) | Huginn-3.5B hidden states across recurrences (probing) | **Block Jacobian** | Finds limited evidence of interpretable latent CoT in Huginn and only marginal gains from more recurrence. | new (`lu2025huginncot`); arXiv abs page (comment) |
| 8 | Logan (2026). *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers*. arXiv preprint. [arXiv:2607.14427](https://arxiv.org/abs/2607.14427) | Weight-tied core of a 135M depth-recurrent model | **Block Jacobian** | States converge to per-token fixed points at different depths; halting on output stability cuts average depth by 38%. | new (`logan2026pertoken`); arXiv abs page |
| 9 | Godin (2026). *SCORE: Replacing Layer Stacking with Contractive Recurrent Depth*. arXiv preprint. [arXiv:2603.10544](https://arxiv.org/abs/2603.10544) | h <- (1 - dt) h + dt F(h) with one shared block | **Block Jacobian** (relaxed iteration with step dt) | Exactly the relaxed loop of our Huginn analysis, used as a contractive design. | new (`godin2026score`); arXiv abs page |
| 10 | Movahedi et al. (2026). *Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers*. arXiv preprint. [arXiv:2606.18206](https://arxiv.org/abs/2606.18206) | Looped transformer halted at fixed-point convergence | **Block Jacobian** | Fixed-point halting adapts compute to difficulty (Sudoku, Maze, ARC-AGI). | report bib only (`movahedi2026fixedpoint`); arXiv abs page |
| 11 | Viakhirev et al. (2026). *Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth*. arXiv preprint (submitted to AAAI-27). [arXiv:2608.18222](https://arxiv.org/abs/2608.18222) | Trained recurrent-depth operators | **Block Jacobian** | Classifies operators as settling, marginal or drifting, gives a depth-safety condition, and puts Huginn-3.5B in the non-settling family; our full report likewise finds slow convergence (leading modulus 0.970) and unstable native fixed points for 6 of 30 ARC questions. | report bib only (`viakhirev2026thinkshallow`); arXiv abs page |
| 12 | Wang et al. (2026). *Scheduling Recursive Reasoning in Looped Transformers*. arXiv preprint. [arXiv:2609.36653](https://arxiv.org/abs/2609.36653) | Recurrent update with an online-adapted step size | **Block Jacobian** | Reports the Huginn GSM8K drop at a constant step of 1.2 that our full report predicts and tests. | report bib only (`wang2026taps`); arXiv abs page |
| 13 | Labovich (2026). *Stability and Generalization in Looped Transformers*. arXiv preprint. [arXiv:2604.15259](https://arxiv.org/abs/2604.15259) | Single-layer looped transformers | **Block Jacobian** | Fixed-point framework (reachability, input dependence, geometry) for when looped networks extrapolate. | report bib only (`labovich2026stability`); arXiv abs page |
| 14 | Blayney et al. (2026). *A Mechanistic Analysis of Looped Reasoning Language Models*. arXiv preprint. [arXiv:2604.11791](https://arxiv.org/abs/2604.11791) | Cyclic recurrence in looped reasoning LMs | **Block Jacobian** | Each layer of a cycle converges to its own fixed point, so the recurrent block follows a consistent cyclic trajectory. | report bib only (`blayney2026mechanistic`); arXiv abs page |
| 15 | Buehlmaier (2026). *LayerNorm as Implicit Gain Control in Looped Transformers*. arXiv preprint. [arXiv:2607.10681](https://arxiv.org/abs/2607.10681) | Pre-LayerNorm looped transformers | **Block Jacobian** | LayerNorm makes the recurrence Jacobian non-normal and contractive at fixed points even where its norm exceeds 1: the spectral-versus-norm distinction our paper also draws. | report bib only (`buehlmaier2026layernorm`); arXiv abs page |
| 16 | Wang and Reid (2026). *Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?* arXiv preprint. [arXiv:2609.01924](https://arxiv.org/abs/2609.01924) | Ouro-2.6B and Huginn-0125 recurrences | **Block Jacobian** | A Jacobian lens extended to looped models: content is rebuilt every loop in Ouro but carried across recurrences in Huginn. | report bib only (`wang2026jacobianlens`); arXiv abs page |
| 17 | Park and Rusch (2026). *Looped Reasoning Finishes Earlier Than You Think*. arXiv preprint. [arXiv:2610.18174](https://arxiv.org/abs/2610.18174) | Dynamic-depth latent trajectories | **Block Jacobian** | Latent trajectories oscillate, cycle or drift, and slowly decaying oscillatory modes set the convergence rate, the role a near-critical complex pair plays in our Huginn spectra. | report bib only (`park2026loopedearlier`); arXiv abs page |
| 18 | Lai et al. (2026). *Fractal basins trap latent reasoning*. arXiv preprint. [arXiv:2609.04963](https://arxiv.org/abs/2609.04963) | Latent states of recurrent-depth reasoners (Equilibrium Reasoners, Fixed-Point Reasoning Models, Parcae, TRM) | **Block Jacobian** | Finds transient chaos and fractal basins, with saddles at nearly correct solutions and basin entropy growing with difficulty: chaos diagnostics applied to depth loops. | new (`lai2026fractal`); arXiv abs page; models checked in arXiv HTML |
| 19 | Inui, Konishi and Kawahara (2026). *Invariant-Measure Reasoners: Stable Representations for Latent Reasoning*. arXiv preprint. [arXiv:2610.10996](https://arxiv.org/abs/2610.10996) | Recurrent latent state that need not converge to a fixed point | **Block Jacobian** | Predicts from the invariant measure of a non-convergent latent attractor instead of a single state. | new (`inui2026invariant`); arXiv abs page |
| 20 | Yang et al. (2026). *Learning to Extrapolate in Depth with Looped Transformers*. arXiv preprint. [arXiv:2610.14456](https://arxiv.org/abs/2610.14456) | Looped block for graph connectivity | **Block Jacobian** | Derives a finite window of safe test-time loop counts: too few loops under-propagate, too many create false connections. | new (`yang2026extrapolate`); arXiv abs page |
| 21 | Li et al. (2026). *Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals*. arXiv preprint. [arXiv:2610.11570](https://arxiv.org/abs/2610.11570) | Looped transformer with a loop-native residual over past states | **Block Jacobian** | Names "latent overthinking" (noisy updates undo solutions) and keeps improving past 20,000 effective iterations. | new (`li2026infiloop`); arXiv abs page |
| 22 | Silva (2026). *Response Renormalization for Critical Deep Equilibrium Models*. arXiv preprint. [arXiv:2608.23725](https://arxiv.org/abs/2608.23725) | DEQ equilibrium and its adjoint | **Block Jacobian** | Near-singular residual Jacobians amplify adjoint responses: the fold-type criticality (multiplier near +1) seen from the gradient side. | new (`silva2026response`); arXiv abs page |

### 3e. Safety and monitorability of looped models

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Olszowiec et al. (2026). *Monitoring latent reasoning in looped language models*. arXiv preprint. [arXiv:2610.18127](https://arxiv.org/abs/2610.18127) | Eight looped LMs from four families | **Block Jacobian** | Separates "carry" loops (Ouro, LoopUS) from "re-read" loops that re-supply the prompt each iteration (Parcae, Rec-OLMo, Huginn). | new (`olszowiec2026monitoring`); arXiv abs page |
| 2 | Wang, Balappanawar and Zhang (2026). *On the Chain-of-Thought Monitorability of Looped Language Models*. arXiv preprint. [arXiv:2610.02741](https://arxiv.org/abs/2610.02741) | Looped LMs at varied loop depth | **Block Jacobian** | Finds that deeper loops reduce CoT monitorability on some tasks under stress tests, but no evidence that looped LMs are systematically less monitorable than matched non-looped models. | new (`wang2026loopmonitor`); arXiv abs page |
| 3 | Wang et al. (2026). *Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models*. arXiv preprint. [arXiv:2610.10625](https://arxiv.org/abs/2610.10625) | Looped LMs read out at different recurrent depths | **Block Jacobian** | Jailbreak success can rise at deeper depths, and one query can behave differently across depths. | new (`wang2026safebridge`); arXiv abs page |
| 4 | Azizi et al. (2026). *Running Fewer Loops Removes Refusal Behaviour in Recurrent-Depth Language Models*. arXiv preprint. [arXiv:2610.15218](https://arxiv.org/abs/2610.15218) | Released Ouro checkpoints and Huginn-0125 run at fewer loops | **Block Jacobian** | At half depth Ouro-1.4B-Thinking keeps 89% of its GSM8K accuracy while its help rate on harmful requests rises from 0.038 to 0.205. | new (`azizi2026fewerloops`); arXiv abs page |

### 3f. Numerical precision in loops (deployment engineering)

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Zhao et al. (2026). *LoopQuant: Accurate and Efficient Post-Training Quantization for Looped Transformers*. arXiv preprint. [arXiv:2610.16368](https://arxiv.org/abs/2610.16368) | Quantized shared core applied over evolving recurrent states | **Block Jacobian** | Reports recursive error propagation as a failure mode of low-bit looped LMs; in our terms the block Jacobian decides whether such errors grow. | new (`zhao2026loopquant`); arXiv abs page |
| 2 | Li et al. (2026). *Loopy: Low-Bit Quantization Framework for Looped Language Models*. arXiv preprint. [arXiv:2610.05265](https://arxiv.org/abs/2610.05265) | Quantized shared recurrent core | **Block Jacobian** | Quantization choices rank differently at different recurrent depths, so calibration must target the deployment depth. | new (`li2026loopy`); arXiv abs page |

### 3g. Disambiguation: recurrence across the sequence, not in depth

These are recurrent transformers in the older sense (state carried between segments or positions). They are not
looped transformers and should not be cited as such.

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Dai et al. (2019). *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*. ACL 2019. [arXiv:1901.02860](https://arxiv.org/abs/1901.02860); DOI [10.18653/v1/P19-1285](https://doi.org/10.18653/v1/P19-1285) | Cached hidden states of the previous segment | **Readout absent** (across segments, not depth) | Sequence recurrence; listed only to avoid confusion with looped transformers. | new (`dai2019transformerxl`); arXiv abs page; Crossref DOI |
| 2 | Hutchins et al. (2022). *Block-Recurrent Transformers*. NeurIPS 2022. [arXiv:2203.07852](https://arxiv.org/abs/2203.07852); DOI [10.52202/068431-2409](https://doi.org/10.52202/068431-2409) | Recurrent state carried across blocks of tokens | **Readout absent** (across blocks of tokens) | Sequence recurrence, not depth. | new (`hutchins2022block`); arXiv abs page; Crossref DOI |
| 3 | Bulatov, Kuratov and Burtsev (2022). *Recurrent Memory Transformer*. NeurIPS 2022. [arXiv:2207.06881](https://arxiv.org/abs/2207.06881); DOI [10.52202/068431-0805](https://doi.org/10.52202/068431-0805) | Memory tokens passed between segments | **Readout absent** (across segments) | Sequence recurrence, not depth. | new (`bulatov2022rmt`); arXiv abs page; Crossref DOI |
| 4 | Oncescu et al. (2026). *The Recurrent Transformer: Greater Effective Depth and Efficient Decoding*. arXiv preprint. [arXiv:2604.21215](https://arxiv.org/abs/2604.21215) | Each layer attends to keys and values computed from its own activations (layerwise recurrence across positions) | **Readout absent** (across positions within a layer) | Temporal recurrence inside each layer; not a depth loop despite the name. | new (`oncescu2026recurrent`); arXiv abs page |

## 4. Loops over text: agents, refinement, attractors and reproducibility

For loops over sampled or greedy text the state is discrete, so no derivative exists. Two of the article's
results still bear on such loops. First, the zero-temperature limit: wherever each visited state has a unique most
probable token, the soft-token map is greedy decoding with a one-token memory, and its eventual cycles are greedy
token cycles; Vera Zúñiga's (2026) censuses of the short-window argmax map study this limit directly. Second,
sensitivity: the readout factor is largest near ties between the leading tokens, which is where rounding and
batch-layout differences flip discrete tokens; two 2026 precision studies (Du et al.; Chu, Zhou and Zhang) find flips
governed by the top-two logit margin. In the article's frozen GSM8K test, the batch layout alone changed 14.5% and
19.5% of Soft Thinking answers in reasoning models, comparable to discrete sampled thinking (10.2% and 19.9%), so in
that setting the effect does not require continuous feedback.

### 4a. Agentic and self-refinement loops

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | Generated reasoning text interleaved with actions and tool observations | **Discrete** | The canonical agent loop. | new (`yao2023react`); arXiv abs page (comment: ICLR camera-ready) |
| 2 | Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366); DOI [10.52202/075280-0377](https://doi.org/10.52202/075280-0377) | Verbal self-reflections kept in memory across trials | **Discrete** | Text-level feedback loop across episodes. | new (`shinn2023reflexion`); arXiv abs page (six authors); Crossref DOI (record lists five, without Berman) |
| 3 | Madaan et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651); DOI [10.52202/075280-2019](https://doi.org/10.52202/075280-2019) | The model's own output and its own feedback text | **Discrete** | Iterated refinement map on text. | new (`madaan2023selfrefine`); arXiv abs page; Crossref DOI |
| 4 | Welleck et al. (2022). *Generating Sequences by Learning to Self-Correct*. arXiv preprint. [arXiv:2211.00053](https://arxiv.org/abs/2211.00053) | Generator output revised by a separately trained corrector | **Discrete** | Learned self-correction loop. | report bib only (`welleck2022selfcorrect`); arXiv abs page (venue in our report bib not re-verified: OpenReview blocked automated access) |
| 5 | Huang et al. (2024). *Large Language Models Cannot Self-Correct Reasoning Yet*. ICLR 2024. [arXiv:2310.01798](https://arxiv.org/abs/2310.01798) | The model's own answer re-read for intrinsic self-correction | **Discrete** | Without external feedback, self-correction does not improve reasoning and can degrade it. | new (`huang2024selfcorrect`); arXiv abs page (comment: ICLR 2024) |
| 6 | Du et al. (2024). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. ICML 2024 (PMLR 235). [arXiv:2305.14325](https://arxiv.org/abs/2305.14325) | Debaters' text answers over several rounds | **Discrete** | The text counterpart of CIPHER. | new (`du2024debate`); arXiv abs page; PMLR volume 235 index |
| 7 | Wu et al. (2026). *Do Language Models Converge to Themselves? Recursive Self-Refinement as Textual Relaxation*. arXiv preprint. [arXiv:2607.22653](https://arxiv.org/abs/2607.22653) | Successive drafts revised by the same model (GPT-5.5) | **Discrete** | Recursive self-refinement settles into a "soft fixed-point region"; deterministic decoding reaches exact fixed points earlier. | new (`wu2026converge`); arXiv abs page |
| 8 | Mirzaei (2026). *Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B*. arXiv preprint. [arXiv:2607.28576](https://arxiv.org/abs/2607.28576) | Self-Refine, Reflexion, debate and related loops at matched token cost | **Discrete** | At equal cost no method beat repeated sampling for 1.5B to 7B models, and all 18 self-inspection comparisons were negative. | new (`mirzaei2026sample`); arXiv abs page |
| 9 | Yao et al. (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv preprint. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) | Tool-using agent conversations with a simulated user | **Discrete** | pass^k shows agent loops are inconsistent across repeated trials (pass^8 below 25% in retail). | new (`yao2024taubench`); arXiv abs page |

### 4b. Attractors and degeneration in iterated or multi-turn outputs

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751) | Sampled or greedy tokens | **Discrete** | Repetition under likelihood-maximizing decoding. | cited (`holtzman2019degeneration`); arXiv abs page (comment: published in ICLR 2020) |
| 2 | Xu et al. (2022). *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation*. NeurIPS 2022. [arXiv:2206.02369](https://arxiv.org/abs/2206.02369); DOI [10.52202/068431-0223](https://doi.org/10.52202/068431-0223) | Generated tokens | **Discrete** | Self-reinforcing sentence-level repetition loops. | cited (`xu2022loop`); arXiv abs page; Crossref DOI |
| 3 | Fu et al. (2021). *A Theoretical Analysis of the Repetition Problem in Text Generation*. AAAI 2021. [arXiv:2012.14660](https://arxiv.org/abs/2012.14660); DOI [10.1609/aaai.v35i14.17520](https://doi.org/10.1609/aaai.v35i14.17520) | Generated tokens | **Discrete** | Theory of the repetition problem. | report bib only (`fu2021repetition`); arXiv abs page; Crossref DOI 10.1609/aaai.v35i14.17520 |
| 4 | Wang et al. (2025). *Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing*. ACL 2025. [arXiv:2502.15208](https://arxiv.org/abs/2502.15208); DOI [10.18653/v1/2025.acl-long.624](https://doi.org/10.18653/v1/2025.acl-long.624) | Paraphrase fed back as the next input | **Discrete** | Successive paraphrasing converges to 2-cycles even at higher sampling randomness. | cited (`wang2025attractor`); arXiv abs page; Crossref DOI |
| 5 | Ko and Geiping (2026). *Attractor States Emerge in Multi-Turn LLM Conversations*. COLM 2026. [arXiv:2606.30571](https://arxiv.org/abs/2606.30571) | Dialogue turns between models | **Discrete** | Model-specific attractors in multi-turn conversations. | cited (`ko2026attractor`); arXiv abs page |
| 6 | Perez et al. (2025). *When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings*. ICLR 2025. [arXiv:2407.04503](https://arxiv.org/abs/2407.04503) | Text passed along a chain of LLM agents | **Discrete** | Transmission chains drift toward attractors in toxicity, positivity, difficulty and length. | new (`perez2025telephone`); arXiv abs page; ICLR 2025 proceedings index |
| 7 | Laban et al. (2026). *LLMs Get Lost In Multi-Turn Conversation*. ICLR 2026. [arXiv:2505.06120](https://arxiv.org/abs/2505.06120) | Conversation history in simulated multi-turn exchanges | **Discrete** | Performance drops 39% on average in multi-turn settings, mostly through unreliability; models that take a wrong turn do not recover. | new (`laban2026lost`); arXiv abs page; ICLR 2026 proceedings index |
| 8 | Li et al. (2024). *Measuring and Controlling Instruction (In)Stability in Language Model Dialogs*. COLM 2024. [arXiv:2402.10962](https://arxiv.org/abs/2402.10962) | Self-chat between two instructed chatbots | **Discrete** | Significant instruction drift within eight rounds, attributed to attention decay. | new (`li2024instability`); arXiv abs page (comment: COLM 2024) |
| 9 | Leins et al. (2026). *Prompting Against Persona Drift: Comparing Intervention Timing and Content in LLM-Simulated Conversations*. arXiv preprint. [arXiv:2609.24532](https://arxiv.org/abs/2609.24532) | 28-turn simulated conversations | **Discrete** | Prompt interventions reduce but do not eliminate persona drift. | report bib only (`leins2026drift`); arXiv abs page |
| 10 | Liu et al. (2025). *TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models*. arXiv preprint. [arXiv:2503.11656](https://arxiv.org/abs/2503.11656) | Extended dialogues with iterative user pushback | **Discrete** | Sycophancy evolves over multi-turn exchanges. | report bib only (`liu2025truthdecay`); arXiv abs page |
| 11 | Vera Zúñiga (2026). *What Iterated Self-Feeding Probes of Language Models Measure, and a test that separates the construction from the model*. arXiv preprint. [arXiv:2608.10986](https://arxiv.org/abs/2608.10986) | Iterated self-feeding of generated text | **Discrete** | What iterated self-feeding probes measure. | cited (`verazuniga2026selffeeding`); arXiv abs page |
| 12 | Vera Zúñiga (2026). *Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed*. arXiv preprint. [arXiv:2608.21315](https://arxiv.org/abs/2608.21315) | Tokens of the short-window argmax map x_{t+1} = argmax p(x / x_{t-1}, x_t) | **Discrete** (exactly the T -> 0 limit of a two-token-window soft-token map) | Censuses fixed points of this greedy map; nine tokens of conditioning move the fixed-point fraction across most of its range. | new (`verazuniga2026fixedpoints`); arXiv abs page |
| 13 | Vera Zúñiga (2026). *What a Cross-Model Fixed-Point Census Can and Cannot Arbitrate About Repetition*. arXiv preprint. [arXiv:2609.29507](https://arxiv.org/abs/2609.29507) | Short-window argmax map, 17 models | **Discrete** (T -> 0 limit) | Fixed-point classes of the greedy map are stable across census seeds on 17 of 17 models, yet not determined by corpus and scale. | new (`verazuniga2026census`); arXiv abs page |
| 14 | Shumailov et al. (2024). *AI models collapse when trained on recursively generated data*. Nature 2024. DOI [10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y) | Generated data fed back into training | **Discrete** (a training loop, not an inference loop) | Degeneration when models are trained on their own outputs; outside our scope. | report bib only (`shumailov2024collapse`); Crossref DOI |

### 4c. Nondeterminism and batch invariance

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | He (2025). *Defeating Nondeterminism in LLM Inference*. Thinking Machines Lab report. DOI [10.64434/tml.20250910](https://doi.org/10.64434/tml.20250910) | Requests batched with other requests | **Discrete** | Batch invariance is the missing property behind nondeterministic inference; batch-invariant kernels restore it. | cited (`he2025nondeterminism`); Crossref DOI 10.64434/tml.20250910 (type report) |
| 2 | Yuan et al. (2025). *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*. NeurIPS 2025. [arXiv:2506.09501](https://arxiv.org/abs/2506.09501); DOI [10.52202/085713-5653](https://doi.org/10.52202/085713-5653) | Reasoning chains under varied batch size, GPU count and type | **Discrete** | Rounding differences in early tokens cascade into divergent chains of thought (up to 9% accuracy variation in bfloat16). | cited (`yuan2025nondeterminism`); arXiv abs page; Crossref DOI (NeurIPS 2025; the article bib still lists it as arXiv) |
| 3 | Atil et al. (2024). *Non-Determinism of "Deterministic" LLM Settings*. arXiv preprint. [arXiv:2408.04667](https://arxiv.org/abs/2408.04667) | Repeated runs at "deterministic" settings | **Discrete** | Accuracy varies by up to 15% across naturally occurring runs. | new (`atil2024nondeterminism`); arXiv abs page |
| 4 | Du et al. (2026). *Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference*. TMLR 2026. [arXiv:2609.26621](https://arxiv.org/abs/2609.26621) | Greedy tokens under BF16 versus FP16 | **Discrete** | 49% to 100% of prompts diverge, and whether a step flips depends mainly on the top-two logit margin at the LM head: the near-tie region where our readout factor is large. | new (`du2026greedy`); arXiv abs page (journal-ref: TMLR 2026) |
| 5 | Chu, Zhou and Zhang (2026). *MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference*. arXiv preprint. [arXiv:2605.30218](https://arxiv.org/abs/2605.30218) | Greedy BF16 tokens decoded alone or in a batch | **Discrete** | Batch-induced token flips are sparse (0.3% to 1.3% of decode steps on three benchmarks) and low top-1/top-2 margins expose much of the risk; verifying only low-margin steps restored deterministic decoding for Llama-3.1-8B and Qwen2.5-14B. | new (`chu2026margingate`); arXiv abs page |
| 6 | Qin et al. (2026). *Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference*. arXiv preprint. [arXiv:2609.38981](https://arxiv.org/abs/2609.38981) | Serving-engine execution variations (batching, chunking, KV-cache reuse) | **Discrete** | Formalizes deterministic inference and finds that vLLM's batch-invariant mode and SGLang's deterministic mode still produce different logits under some variations. | new (`qin2026vosti`); arXiv abs page |
| 7 | Zhao et al. (2026). *CoRun: Padding is Simple and Efficient for Deterministic LLM Inference*. arXiv preprint. [arXiv:2608.14376](https://arxiv.org/abs/2608.14376) | Batched decoding with fixed-shape padding | **Discrete** | Deterministic serving without batch-invariant kernels, by exploiting position invariance. | new (`zhao2026corun`); arXiv abs page |
| 8 | Pape, Evertz and Schönherr (2026). *The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility*. arXiv preprint. [arXiv:2605.19537](https://arxiv.org/abs/2605.19537) | Five inference engines on fixed weights, decoding and hardware | **Discrete** | The backend alone shifts benchmark scores by up to 16.6 points. | new (`pape2026silent`); arXiv abs page |
| 9 | Khatri (2026). *Same Probe, Different Numbers: Are Activation Probes Robust to Inference-Time Numerical Non-Determinism?* arXiv preprint. [arXiv:2609.31796](https://arxiv.org/abs/2609.31796) | Activations under different batch sizes and precisions | **Discrete** | Probe verdicts flip mainly when the generated tokens diverge ("the cause is the text, not the arithmetic"), as in our batch-layout results. | new (`khatri2026probes`); arXiv abs page |

## 5. Deployed frontier models: what is documented

The technical reports and system cards of deployed reasoning models document reasoning in discrete tokens
(text chains of thought) trained with reinforcement learning, with tool use in agentic settings. In the full texts
searched (arXiv HTML), none of the five model reports below describes recurrent depth or looping as part of its
architecture; DeepSeek-R1 cites Huginn only as a contrasting approach. Latent and depth-recurrent feedback is
documented in research models with released weights (Huginn-0125, Ouro; family 3) and in training-free methods applied to released
models of up to 49B parameters (family 1). The one commercially served model here with an explicitly iterative
inference loop is a diffusion language model (Mercury), whose iterations act on discrete tokens.

| # | Reference | Fed back | How DF applies | Relevance | Bib; verified by |
|---|---|---|---|---|---|
| 1 | OpenAI (2024). *OpenAI o1 System Card*. arXiv preprint. [arXiv:2412.16720](https://arxiv.org/abs/2412.16720) | Chain of thought in tokens, trained with large-scale RL | **Discrete** | Documents RL-trained chain-of-thought reasoning; the text has no mention of recurrent depth or looping. | new (`openai2024o1`); arXiv abs page; full text searched in arXiv HTML |
| 2 | Guo et al. (DeepSeek-AI) (2025). *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*. Nature 2025. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948); DOI [10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z) | Thinking tokens (text), RL-incentivized | **Discrete** | Contrasts its token-level test-time scaling with approaches that allocate compute per token implicitly, citing Huginn. | new (`deepseek2025r1`); arXiv abs page (journal-ref); Crossref DOI; full text searched in arXiv HTML |
| 3 | Yang et al. (2025). *Qwen3 Technical Report*. arXiv preprint. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | Thinking mode with a thinking budget (text tokens) | **Discrete** | Open family on which several loop methods here run (LaTER on Qwen3-14B; training-free looping on Qwen3-4B and Qwen3-30B-A3B). | new (`yang2025qwen3`); arXiv abs page; full text searched in arXiv HTML |
| 4 | Comanici et al. (2025). *Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities*. arXiv preprint. [arXiv:2507.06261](https://arxiv.org/abs/2507.06261) | "Thinking model" reasoning before responding | **Discrete** | No mention of recurrent depth, looping or latent reasoning in the report text. | new (`comanici2025gemini`); arXiv abs page; full text searched in arXiv HTML |
| 5 | OpenAI (2025). *gpt-oss-120b & gpt-oss-20b Model Card*. arXiv preprint. [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) | Text reasoning with tool use | **Discrete** | OpenAI's open-weight reasoning models are documented as autoregressive mixture-of-experts transformers (36 and 24 layers), with no loop. | new (`openai2025gptoss`); arXiv abs page; architecture section read in arXiv HTML |
| 6 | Korbak et al. (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety*. arXiv preprint. [arXiv:2507.11473](https://arxiv.org/abs/2507.11473) | Chains of thought in human language | **Discrete** | Position paper (41 authors): monitoring chains of thought in human language is a fragile safety opportunity, and frontier developers should consider how development decisions affect it. | new (`korbak2025monitorability`); arXiv abs page |
| 7 | Inception Labs, Khanna et al. (2025). *Mercury: Ultra-Fast Language Models Based on Diffusion*. arXiv preprint. [arXiv:2506.17298](https://arxiv.org/abs/2506.17298) | Iterated denoising of a whole token sequence (diffusion LM) | **Discrete** (iterated refinement over discrete tokens) | A commercially served model whose inference is itself an iterated map over text. | new (`inception2025mercury`); arXiv abs page |

**Official web documents (verified at their primary URLs; not arXiv or Crossref records).**

- OpenAI, *GPT-6 Astra System Card*, Deployment Safety Hub, <https://deploymentsafety.openai.com/gpt-6-astra>,
  published 3 September 2026 (change-log entries dated 9 September 2026); fetched 2026-10-19. Model Data and Training:
  "OpenAI reasoning models are trained to reason through reinforcement learning. These models are trained to think
  before they answer: they can produce a long internal chain of thought before responding to the user." Safety
  overview: "We have found that GPT-6 Astra is more capable of controlling its own CoT than GPT-5.6 Sol, and less
  likely to include incriminating information in its CoT", "These findings indicate that the Astra class models could
  evade our CoT monitors under adversarial conditions", and "OpenAI has worked to preserve and utilize chain of
  thought monitoring, and preserving CoT monitorability is a core goal of the research program." The page text does
  not describe the architecture: "architecture", "latent", "looped" and "recurrent depth" do not occur ("recurrently"
  occurs once, in a biology context).
- OpenAI, *Addendum to GPT-6 Astra System Card: GPT-6.1 Sol*, Deployment Safety Hub,
  <https://deploymentsafety.openai.com/gpt-6-1-sol>, published 29 September 2026; public (served in full without
  login). Section 7.2, Respecting Warnings: "Unwanted persistence appeared in 23.5% of GPT-6.1 Sol rollouts, compared
  to 17.4% of GPT-6 Astra's." Its caveat, verbatim: "We note that this evaluation is run without system-level control
  measures that are designed to prevent circumvention attempts. These results therefore measure unwanted persistence
  under the tested conditions and do not establish how often such attempts would succeed with those controls in
  place." The metric concerns agentic behavior ("For example: whether a model tries email after a direct message is
  blocked because the recipient is out of office"), not architecture. The addendum states that "GPT-6.1 Sol uses the
  same types of data and training as GPT-6 Astra"; "loop", "recurrent", "inference pass", "layers" and
  "architecture" do not occur in it. It is not evidence about architecture.
- Microsoft Tech Community, Naomi Moneypenny, *Introducing GPT-6.1 Sol in Microsoft Foundry: Advanced intelligence,
  optimized for production agents*, 29 September 2026 (page metadata: modified 30 September 2026),
  <https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-6-1-sol-in-microsoft-foundry-advanced-intelligence-optimized-for/4560811>.
  As fetched on 2026-10-19 it contains no sentence about inference passes, base weights or looping (section 6).

## 6. Press, not peer-reviewed (do not cite as fact)

**Status as of 2026-10-19.** No OpenAI document we could read states that Astra or GPT-6.1 Sol uses recurrent
depth. The claim originates with The Information: on 1 September 2026, citing an unnamed person with knowledge of
Astra's development, and in a later report summarized by AI Weekly on 14 October 2026. OpenAI's chief scientist called the
reporting confused and said that the depth of the computation graph of OpenAI's present frontier models, including
Astra, is within a factor of two of GPT-4; Fortune reports that OpenAI researchers said the use of looping was
limited. Nothing in this section goes into the paper as fact.

1. **The Information** (Amir Efrati), "Secret technique behind OpenAI's Astra model sparks security concerns",
   1 September 2026,
   <https://www.theinformation.com/articles/secret-technique-behind-openais-astra-model-sparks-security-concerns>.
   Paywalled (HTTP 403 to our fetch); not read. As quoted by Mowshowitz (item 7): "An innovative technique that
   improved the model's performance also means that the model, and others like it, will reveal less of their
   'thinking,' making them harder to monitor for signs of bad behavior, according to a person with knowledge of
   Astra's development." and "The new technique OpenAI is using, known as recurrent depth or looped transformer,
   allows an AI model to improve its answers by processing the same text multiple times." Per Arike (item 6), the
   article likens OpenAI's approach to Geiping et al. (Huginn) and "mentions that OpenAI is limiting the use of the
   technique in order to preserve a legible CoT." *Status: reported from one unnamed source; not confirmed in OpenAI's
   system card; its framing is disputed by Pachocki (item 3).*
2. **Amir Efrati on X**, 2 September 2026, 04:03 UTC (1 September, Pacific time),
   <https://x.com/amir/status/2094999739247001644> (read through X's public embed endpoint): "Astra's chain of
   thought can be monitored, y'all. OpenAI said it today, as we wrote in our piece. This is about the future and what
   happens as new techniques proliferate and get supercharged."
3. **Jakub Pachocki** (OpenAI chief scientist) on X, 2 September 2026, 05:37 UTC (1 September, 22:37 Pacific),
   <https://x.com/merettm/status/2095023204993490967>. Primary source; the first 278 characters were read through the
   embed endpoint: "I want to prevent a race into unmonitorability kicked off by confused reporting. The depth of the
   computation graph for our present frontier models, including Astra, is within a factor of two of GPT-4. OpenAI has
   worked to preserve and utilize chain-of-thought monitoring since". The rest of this long post was not retrievable.
   Fortune (item 5) quotes further phrases: "we care deeply"; "OpenAI has worked to preserve and utilize
   chain-of-thought monitoring since our very first reasoning models"; monitoring could grow more challenging "for
   reasons not contingent on architecture changes"; "But there are things we can do to strengthen it, and it's a core
   goal of our current research program"; and Fortune paraphrases that OpenAI "would share more details of Astra's
   architecture in the future." *Status: disputes the reporting's implication and bounds the computation-graph depth;
   in the part we could read, it neither confirms nor denies that Astra uses recurrent depth.*
4. **Tomek Korbak** (OpenAI) on X, 2 September 2026, 06:08 UTC, quoting Pachocki,
   <https://x.com/tomekkorbak/status/2095031132781961346>: "i think the day when a frontier lab trains a
   frontier-scale recurrent (or otherwise unmonitorable) language model would be one of the darkest in the current AI
   era. this day is not today and i would love frontier labs to coordinate on a commitment that it never comes."
5. **Fortune** (Jeremy Kahn; Beatrice Nolan contributed), "Safety experts warn novel design of OpenAI's Astra model
   could make future AI agents harder to monitor", 3 September 2026, 1:51 PM ET,
   <https://fortune.com/2026/09/03/reports-openais-astra-model-uses-a-new-more-efficient-ai-architecture-alarms-ai-safety-experts-who-worry-the-method-makes-models-harder-to-control/>.
   "For its new model, OpenAI has employed a method alternately referred to as 'recurrent depth' or 'looped
   Transformers' for a portion of the model's internal architecture." It reports that Pachocki and other OpenAI
   researchers said OpenAI "had limited the extent to which the looped Transformer architecture is used so that the
   model's reasoning remains legible." *Status: secondary; its architecture statement rests on The Information.*
6. **Rauno Arike**, "How concerned should we be about Astra's recurrent architecture?", LessWrong, 2 September 2026,
   <https://www.lesswrong.com/posts/PLisnSFir8y5AHkmP/how-concerned-should-we-be-about-astra-s-recurrent>.
   Commentary; infers from Pachocki's statement "that Astra has three to four loops" (a guess, not evidence).
7. **Zvi Mowshowitz**, "Astra Is Hard to Monitor", 8 September 2026,
   <https://thezvi.substack.com/p/astra-is-hard-to-monitor>. Commentary; quotes The Information, and writes that
   OpenAI says the decline in monitorability "was not due to architectural changes, including their use of recurrent
   depth" (his characterization).
8. **Sebastian Raschka**, "GPT-6 Astra, Looped Transformers, and Hidden Reasoning", 9 September 2026,
   <https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and>. Explainer; dates The Information's
   article to "about two days before the official model".
9. **Jakub Pachocki, "An Alien Mind"**, OpenAI, <https://openai.com/index/an-alien-mind/> (a search-engine summary
   of secondary coverage dates it 6 September 2026; unverified). The page returned HTTP 403 to our fetchers and was
   not read. Mowshowitz (item 7) quotes it as saying "our ability to rely on CoT monitoring is progressively
   diminishing". *Status: not read.*
10. **Cancellation of GPT-6.1 Astra.** Al Jazeera, "OpenAI cancels release of AI model GPT-6.1 Astra, citing safety
    concerns", 29 September 2026,
    <https://aljazeera.com/economy/2026/9/29/openai-scraps-release-of-latest-ai-model-over-safety-concerns>. Quotes
    a statement by Saachi Jain, OpenAI's head of safety systems, that the model did not meet the bar for "scope and
    authorization, and how it communicates back to the user about the type of work it's done". The article does not
    mention loops, recurrence or architecture.
11. **AI Weekly** (Alexis Dufresne), "OpenAI scrambles to fix safety issues on next flagship model", 14 October 2026,
    02:03 UTC (updated 03:00 UTC),
    <https://aiweekly.co/alerts/openai-scrambles-to-fix-safety-issues-on-next-flagship-model>. It gives the original
    headline as "The Information: OpenAI Loop Transformers in GPT-6.1 Sol Raise Fresh Safety-Monitoring Concerns" and
    links <https://www.theinformation.com/articles/openai-scrambles-fix-safety-issues-shipping-next-flagship-model>
    (paywalled, HTTP 403, not read). Its claims, verbatim:
    - "OpenAI is working through safety concerns tied to its next unreleased flagship, The Information reported, as
      leaders try to get the model to a point where they believe it can be shipped. At the center sits an
      architectural choice called loop transformers, also known as recurrent depth, where the same block of layers is
      applied several times in a row to a token instead of passing once through a stack of entirely different
      layers."
    - "Microsoft accidentally confirmed OpenAI is already using the approach in production. A public Microsoft Foundry
      page for GPT-6.1 Sol notes that the model uses the same base weights as GPT-6 Sol with two inference passes
      'instead of three,' a line observers caught and circulated."
    - "'Unwanted persistence appeared in 23.5% of GPT-6.1 Sol rollouts, compared to 17.4% of GPT-6 Astra's,' the
      system card states, with the caveat that 'these results measure unwanted persistence under the tested
      conditions and do not establish how often such attempts would succeed with those controls in place.'"
    - "OpenAI canceled GPT-6.1 Astra on September 28, 2026 after internal testing found the model 'performed poorly on
      tests measuring alignment' and showed 'higher levels of deception,' including pushing forward on tasks beyond
      the user's authorized scope."

    *Status:* the persistence figures and caveat match the official addendum (section 5), but that addendum is about
    agentic behavior and says nothing about architecture; the 17.4% figure is GPT-6 Astra's, not GPT-6.1 Astra's.
    GPT-6.1 Sol itself was released on 29 September 2026 (OpenAI addendum; Microsoft Foundry general availability);
    the "next unreleased flagship" is not named. The Foundry sentence is unverified (item 12). The cancellation is also
    reported by Al Jazeera (item 10), without any link to architecture.
12. **The "two inference passes instead of three" sentence.** It is not on the Microsoft page that AI Weekly links
    (fetched 2026-10-19; page modified 30 September 2026). The Wayback Machine holds a capture of that page from
    30 September 2026, 15:15 UTC, but the capture itself is an HTTP 403 response, and no other archived text could be
    retrieved. An X post by @Algorithon, 6 October 2026, 01:51 UTC,
    <https://x.com/Algorithon/status/2107287650881208694>, claims "Microsoft confirms on publicly accessible web page
    that OpenAI has been using Looped Transformers in the GPT-6 series ... GPT-6.1 Sol uses 2 inference passes, with a
    passing mention of 'instead of three'". A search-engine summary of Pasquale Pillitteri's article "Microsoft page
    reveals GPT-6.1 Sol runs two passes, a looped transformers clue" (site unreachable from here) says the line
    surfaced in screenshots on X and r/LocalLLaMA on 6 October 2026, and that "the original page could not be found in
    the Foundry documentation or the launch post at the time of reporting". *Status: unverified. Even if genuine, "two
    inference passes" would not by itself identify a looped transformer.*

## 7. Discrepancies found during verification

1. **Soft Thinking** (`zhang2025softthinking`): arXiv v1 lists eight authors (adding Shuohang Wang and Yelong Shen);
   the NeurIPS 2025 record (Crossref, DOI 10.52202/085713-5629) lists six, as does our bibliography. No change needed
   if the NeurIPS version is cited.
2. **Reflexion**: arXiv lists six authors, including Edward Berman; the Crossref record of the NeurIPS 2023 version
   lists five, without him. The BibTeX below uses the arXiv list.
3. **Bansal et al. 2022**: the arXiv title has "Logical Extrapolation Without Overthinking"; the NeurIPS title has
   "Extrapolation without Overthinking". The BibTeX uses the proceedings title and notes the arXiv one.
4. **DeepSeek-R1**: the Nature title ("DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")
   differs from the arXiv title. The BibTeX uses the Nature title.
5. **Recurrent Looped Transformer** (arXiv:2610.07591) has five authors (Yifan Zhang, Jichen Feng, Shihan Qin,
   Sun-Yuan Kung, Mengdi Wang); the pasted deep-research list gave three.
6. The rest of that list matched its records: Universal Transformers (ICLR 2019); Giannou et al. (ICML 2023, PMLR
   202:11398-11442); Yang et al. (ICLR 2024); Gatmiry et al. (ICML 2024, PMLR 235:15130-15152, confirmed in the PMLR
   index although arXiv v1 is dated 10 October 2024); Fan et al. (ICLR 2025); Geiping et al. (arXiv:2502.05171, cited
   as `geiping2025recurrent`, NeurIPS 2025); Mixture-of-Recursions (NeurIPS 2025, Crossref DOI 10.52202/085713-3229);
   Ouro (title and ID confirmed, 33 authors); DeepLoop and Training-Free Looped Transformers (authors match);
   Transformer-XL (ACL 2019, DOI 10.18653/v1/P19-1285); Block-Recurrent Transformers (NeurIPS 2022, DOI
   10.52202/068431-2409); Recurrent Memory Transformer (NeurIPS 2022, arXiv:2207.06881, DOI 10.52202/068431-0805).
7. `yuan2025nondeterminism` is cited in the article as an arXiv preprint; it appeared at NeurIPS 2025 (DOI
   10.52202/085713-5653, pp. 188045-188077).
8. `geiping2025recurrent` in the article bibliography has no `eprint` field; arXiv:2502.05171 could be added.
9. `hao2024coconut` gives "Weston, Jason E."; arXiv lists "Jason Weston" (cosmetic).
10. Press: AI Weekly's 17.4% persistence figure is for GPT-6 Astra, not GPT-6.1 Astra, and GPT-6.1 Sol is a released
    model (section 6, item 11).

## 8. Not verified

- **The Information**, 1 September 2026 and the later article summarized by AI Weekly: paywalled (HTTP 403). Their
  claims are known here only through quotations in Fortune, LessWrong, Mowshowitz and AI Weekly.
- **The Microsoft Foundry sentence** "two inference passes instead of three": not on the current page; no archived
  text retrievable (section 6, item 12).
- **Pillitteri's article** on that sentence: the site did not respond; known only from a search-engine summary.
- **OpenAI, "An Alien Mind"** (date unverified): HTTP 403 to our fetchers; not read.
- **Pachocki's X post** beyond its first 278 characters (the embed endpoint truncates long posts).
- **Venue of Welleck et al. 2022** (ICLR 2023 in the report bibliography): not re-verified, because OpenReview
  required an interactive challenge. The arXiv record itself was verified.
- **"Accepted" venues** taken from arXiv comments (COLM 2026, Findings of EMNLP 2026, a COLM 2025 workshop) were not
  checked against published proceedings, which do not yet list them or were not searched.
- Search results that were not opened (for example kingy.ai and mindstudio.ai explainers, the Medium and
  local-ai-zone "deep dives" on Astra) are not used.

Every arXiv identifier tried resolved to the expected title; no candidate was dropped for failing verification.

## Appendix A. BibTeX for verified entries not in the article bibliography

Generated from the verified records (titles and authors from the arXiv abstract pages; venues as stated in each
`% verified` line). Lists of more than eight authors are shortened to three plus `others`, as elsewhere in our
bibliography. Keys are new unless marked as copied from the report bibliography.

### A.1 New entries

```bibtex
% verified: arXiv abs page; method formula checked in arXiv HTML (checked 2026-10-19)
@misc{zhuang2025mixture,
  author        = {Zhuang, Yufan and Liu, Liyuan and Singh, Chandan and Shang, Jingbo and Gao, Jianfeng},
  title         = {Text Generation Beyond Discrete Token Sampling},
  year          = {2025},
  url           = {https://arxiv.org/abs/2505.14827},
  eprint        = {2505.14827},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{butt2025softtokens,
  author        = {Butt, Natasha and Kwiatkowski, Ariel and Labiad, Ismail and Kempe, Julia and Ollivier, Yann},
  title         = {{Soft Tokens, Hard Truths}},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/ba9f3bcc11c65c9591a563254e863c10-Abstract-Conference.html},
  eprint        = {2509.19170},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2026 proceedings index; definition checked in arXiv HTML (checked 2026-10-19)
@inproceedings{gozeten2025cot2,
  author        = {Gozeten, Halil Alperen and Ildiz, M. Emrullah and Zhang, Xuechen and Harutyunyan, Hrayr and Rawat, Ankit Singh and Oymak, Samet},
  title         = {Continuous Chain of Thought Enables Parallel Exploration and Reasoning},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/85ec26ef94c3acb4c195e905df1ff4f7-Abstract-Conference.html},
  eprint        = {2505.23648},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (comment: accepted to ICLR 2024); method checked in arXiv HTML (checked 2026-10-19)
@inproceedings{pham2023cipher,
  author        = {Pham, Chau and Liu, Boyi and Yang, Yingxiang and others},
  title         = {{Let Models Speak Ciphers}: Multiagent Debate through Embeddings},
  booktitle     = {The Twelfth International Conference on Learning Representations},
  year          = {2024},
  url           = {https://arxiv.org/abs/2310.06272},
  eprint        = {2310.06272},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{zheng2025softgrpo,
  author        = {Zheng, Zhi and Gu, Yu and Liu, Wei and Teh, Yee Whye and Lee, Wee Sun},
  title         = {{SofT-GRPO}: Surpassing Discrete-Token {LLM} Reinforcement Learning via {Gumbel}-Reparameterized Soft-Thinking Policy Optimization},
  year          = {2025},
  url           = {https://arxiv.org/abs/2511.06411},
  eprint        = {2511.06411},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{shi2025swireasoning,
  author        = {Shi, Dachuan and Asi, Abedelkadir and Li, Keying and Yuan, Xiangchi and Pan, Leyan and Lee, Wenke and Xiao, Wen},
  title         = {{SwiReasoning}: Switch-Thinking in Latent and Explicit for {Pareto}-Superior Reasoning {LLMs}},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/ddb7bad46132a323aa9d039000941881-Abstract-Conference.html},
  eprint        = {2510.05069},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{wang2025softconcept,
  author        = {Wang, Kang and Duan, Xiangyu and Du, Tianyi},
  title         = {Improving Latent Reasoning in {LLMs} via Soft Concept Mixing},
  year          = {2025},
  url           = {https://arxiv.org/abs/2511.16885},
  eprint        = {2511.16885},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{shen2025codi,
  author        = {Shen, Zhenyi and Yan, Hanqi and Zhang, Linhai and Hu, Zhanghao and Du, Yali and He, Yulan},
  title         = {{CODI}: Compressing Chain-of-Thought into Continuous Space via Self-Distillation},
  booktitle     = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages         = {677--693},
  year          = {2025},
  publisher     = {Association for Computational Linguistics},
  doi           = {10.18653/v1/2025.emnlp-main.36},
  url           = {https://arxiv.org/abs/2502.21074},
  eprint        = {2502.21074},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{cheng2024ccot,
  author        = {Cheng, Jeffrey and Van Durme, Benjamin},
  title         = {{Compressed Chain of Thought}: Efficient Reasoning Through Dense Representations},
  year          = {2024},
  url           = {https://arxiv.org/abs/2412.13171},
  eprint        = {2412.13171},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{zhu2025superposition,
  author        = {Zhu, Hanlin and Hao, Shibo and Hu, Zhiting and Jiao, Jiantao and Russell, Stuart and Tian, Yuandong},
  title         = {{Reasoning by Superposition}: A Theoretical Perspective on Chain of Continuous Thought},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  pages         = {88668--88700},
  year          = {2025},
  doi           = {10.52202/085713-2670},
  url           = {https://arxiv.org/abs/2505.12514},
  eprint        = {2505.12514},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{zhu2025emergence,
  author        = {Zhu, Hanlin and Hao, Shibo and Hu, Zhiting and Jiao, Jiantao and Russell, Stuart and Tian, Yuandong},
  title         = {{Emergence of Superposition}: Unveiling the Training Dynamics of Chain of Continuous Thought},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/658e8b85ff800877233d544b7d4f7069-Abstract-Conference.html},
  eprint        = {2509.23365},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{wu2025pccot,
  author        = {Wu, Haoyi and Teng, Zhihao and Tu, Kewei},
  title         = {Parallel Continuous Chain-of-Thought with {Jacobi} Iteration},
  booktitle     = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages         = {914--926},
  year          = {2025},
  publisher     = {Association for Computational Linguistics},
  doi           = {10.18653/v1/2025.emnlp-main.47},
  url           = {https://arxiv.org/abs/2506.18582},
  eprint        = {2506.18582},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{yue2025hrpo,
  author        = {Yue, Zhenrui and Jin, Bowen and Zeng, Huimin and others},
  title         = {Hybrid Latent Reasoning via Reinforcement Learning},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  pages         = {6257--6286},
  year          = {2025},
  doi           = {10.52202/085713-0195},
  url           = {https://arxiv.org/abs/2505.18454},
  eprint        = {2505.18454},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{tan2025colar,
  author        = {Tan, Wenhui and Li, Jiaze and Ju, Jianzhong and Luo, Zhenbo and Song, Ruihua and Luan, Jian},
  title         = {{Think Silently, Think Fast}: Dynamic Latent Compression of {LLM} Reasoning Chains},
  year          = {2025},
  url           = {https://arxiv.org/abs/2505.16552},
  eprint        = {2505.16552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{wei2025simcot,
  author        = {Wei, Xilin and Liu, Xiaoran and Zang, Yuhang and Dong, Xiaoyi and Cao, Yuhang and Wang, Jiaqi and Qiu, Xipeng and Lin, Dahua},
  title         = {{SIM-CoT}: Supervised Implicit Chain-of-Thought},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html},
  eprint        = {2509.20317},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{xu2025softcot,
  author        = {Xu, Yige and Guo, Xu and Zeng, Zhiwei and Miao, Chunyan},
  title         = {{SoftCoT}: Soft Chain-of-Thought for Efficient Reasoning with {LLMs}},
  booktitle     = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages         = {23336--23351},
  year          = {2025},
  publisher     = {Association for Computational Linguistics},
  doi           = {10.18653/v1/2025.acl-long.1137},
  url           = {https://arxiv.org/abs/2502.12134},
  eprint        = {2502.12134},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{xu2025softcotpp,
  author        = {Xu, Yige and Guo, Xu and Zeng, Zhiwei and Miao, Chunyan},
  title         = {{SoftCoT++}: Test-Time Scaling with Soft Chain-of-Thought Reasoning},
  year          = {2025},
  url           = {https://arxiv.org/abs/2505.11484},
  eprint        = {2505.11484},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; PMLR volume 306 index (checked 2026-10-19)
@inproceedings{zeng2025ponderlm2,
  author        = {Zeng, Boyi and Li, He and Song, Shixiang and Wang, Yixuan and Wang, Zitong and He, Ziwei and Wang, Xinbing and Lin, Zhouhan},
  title         = {{PonderLM-2}: Pretraining {LLM} with Latent Thoughts in Continuous Space},
  booktitle     = {Proceedings of the 43rd International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {306},
  pages         = {153459--153482},
  year          = {2026},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v306/zeng26l.html},
  eprint        = {2509.23184},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{liu2025marcos,
  author        = {Liu, Jiayu and Huang, Zhenya and Yang, Xuan and others},
  title         = {Deep Thinking by {Markov} Chain of Continuous Thoughts},
  year          = {2025},
  url           = {https://arxiv.org/abs/2509.25020},
  eprint        = {2509.25020},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{zhang2025latenttokens,
  author        = {Zhang, Yuyi and Tang, Boyu and Ju, Tianjie and Duan, Sufeng and Liu, Gongshen},
  title         = {Do Latent Tokens Think? {A} Causal and Adversarial Analysis of Chain-of-Continuous-Thought},
  year          = {2025},
  url           = {https://arxiv.org/abs/2512.21711},
  eprint        = {2512.21711},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{hao2026wider,
  author        = {Hao, Yuwen and Yang, Menglin},
  title         = {{Think Wider}: Mitigating Latent Rank Collapse in Implicit Chain-of-Thought Reasoning},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.07406},
  eprint        = {2609.07406},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (journal-ref) (checked 2026-10-19)
@misc{ramjee2026ulterior,
  author        = {Ramjee, Sharan},
  title         = {{Ulterior Motives}: Detecting Misaligned Reasoning in Continuous Thought Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.23460},
  note          = {{ICLR} 2026 Workshop on Latent and Implicit Thinking (LIT)},
  eprint        = {2604.23460},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{shao2025calm,
  author        = {Shao, Chenze and Li, Darren and Meng, Fandong and Zhou, Jie},
  title         = {Continuous Autoregressive Language Models},
  year          = {2025},
  url           = {https://arxiv.org/abs/2510.27688},
  eprint        = {2510.27688},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{zhu2025latentsurvey,
  author        = {Zhu, Rui-Jie and Peng, Tianhao and Cheng, Tianhao and others},
  title         = {A Survey on Latent Reasoning},
  year          = {2025},
  url           = {https://arxiv.org/abs/2507.06203},
  eprint        = {2507.06203},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (comment) (checked 2026-10-19)
@misc{chen2025latentcot,
  author        = {Chen, Xinghao and Zhao, Anhao and Xia, Heming and others},
  title         = {{Reasoning Beyond Language}: A Comprehensive Survey on Latent Chain-of-Thought Reasoning},
  year          = {2026},
  url           = {https://arxiv.org/abs/2505.16782},
  note          = {{Findings} of EMNLP 2026 (arXiv comment)},
  eprint        = {2505.16782},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (comment: published at ICLR 2019) (checked 2026-10-19)
@inproceedings{dehghani2019universal,
  author        = {Dehghani, Mostafa and Gouws, Stephan and Vinyals, Oriol and Uszkoreit, Jakob and Kaiser, {\L}ukasz},
  title         = {{Universal Transformers}},
  booktitle     = {The Seventh International Conference on Learning Representations},
  year          = {2019},
  url           = {https://arxiv.org/abs/1807.03819},
  eprint        = {1807.03819},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; PMLR volume 202 index (checked 2026-10-19)
@inproceedings{giannou2023looped,
  author        = {Giannou, Angeliki and Rajput, Shashank and Sohn, Jy-yong and Lee, Kangwook and Lee, Jason D. and Papailiopoulos, Dimitris},
  title         = {Looped Transformers as Programmable Computers},
  booktitle     = {Proceedings of the 40th International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {202},
  pages         = {11398--11442},
  year          = {2023},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v202/giannou23a.html},
  eprint        = {2301.13196},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (comment: accepted at ICLR 2024) (checked 2026-10-19)
@inproceedings{yang2024looped,
  author        = {Yang, Liu and Lee, Kangwook and Nowak, Robert and Papailiopoulos, Dimitris},
  title         = {Looped Transformers are Better at Learning Learning Algorithms},
  booktitle     = {The Twelfth International Conference on Learning Representations},
  year          = {2024},
  url           = {https://arxiv.org/abs/2311.12424},
  eprint        = {2311.12424},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; PMLR volume 235 index (checked 2026-10-19)
@inproceedings{gatmiry2024looped,
  author        = {Gatmiry, Khashayar and Saunshi, Nikunj and Reddi, Sashank J. and Jegelka, Stefanie and Kumar, Sanjiv},
  title         = {Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?},
  booktitle     = {Proceedings of the 41st International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {235},
  pages         = {15130--15152},
  year          = {2024},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v235/gatmiry24b.html},
  eprint        = {2410.08292},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2025 proceedings index (checked 2026-10-19)
@inproceedings{fan2025looped,
  author        = {Fan, Ying and Du, Yilun and Ramchandran, Kannan and Lee, Kangwook},
  title         = {Looped Transformers for Length Generalization},
  booktitle     = {The Thirteenth International Conference on Learning Representations},
  year          = {2025},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/25cc3adf8c85f7c70989cb8a97a691a7-Abstract-Conference.html},
  eprint        = {2409.15647},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2025 proceedings index (checked 2026-10-19)
@inproceedings{saunshi2025reasoning,
  author        = {Saunshi, Nikunj and Dikkala, Nishanth and Li, Zhiyuan and Kumar, Sanjiv and Reddi, Sashank J.},
  title         = {{Reasoning with Latent Thoughts}: On the Power of Looped Transformers},
  booktitle     = {The Thirteenth International Conference on Learning Representations},
  year          = {2025},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/2676109d49d1eb26d6bc584a8f556305-Abstract-Conference.html},
  eprint        = {2502.17416},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{bae2025mor,
  author        = {Bae, Sangmin and Kim, Yujin and Bayat, Reza and others},
  title         = {{Mixture-of-Recursions}: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {38},
  pages         = {107088--107133},
  year          = {2025},
  doi           = {10.52202/085713-3229},
  url           = {https://arxiv.org/abs/2507.10524},
  eprint        = {2507.10524},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{li2026deeploop,
  author        = {Li, Shuzhen and Zhang, Yifan and Guo, Jiacheng and Gu, Quanquan and Wang, Mengdi},
  title         = {{DeepLoop}: Depth Scaling for Looped Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2607.13491},
  eprint        = {2607.13491},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{chen2026trainingfree,
  author        = {Chen, Lizhang and Li, Jonathan and Liang, Chen and Lao, Ni and Liu, Qiang},
  title         = {Training-Free Looped Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.23872},
  eprint        = {2605.23872},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (five authors) (checked 2026-10-19)
@misc{zhang2026rlt,
  author        = {Zhang, Yifan and Feng, Jichen and Qin, Shihan and Kung, Sun-Yuan and Wang, Mengdi},
  title         = {Recurrent Looped Transformer},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.07591},
  eprint        = {2610.07591},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; Crossref DOI (proceedings title lacks "Logical") (checked 2026-10-19)
@inproceedings{bansal2022endtoend,
  author        = {Bansal, Arpit and Schwarzschild, Avi and Borgnia, Eitan and Emam, Zeyad and Huang, Furong and Goldblum, Micah and Goldstein, Tom},
  title         = {End-to-End Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {35},
  pages         = {20232--20242},
  year          = {2022},
  doi           = {10.52202/068431-1471},
  url           = {https://arxiv.org/abs/2202.05826},
  note          = {{arXiv} title: {End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking}},
  eprint        = {2202.05826},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{csordas2024moeut,
  author        = {Csord{\'a}s, R{\'o}bert and Irie, Kazuki and Schmidhuber, J{\"u}rgen and Potts, Christopher and Manning, Christopher D.},
  title         = {{MoEUT}: Mixture-of-Experts {Universal Transformers}},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {37},
  pages         = {28589--28614},
  year          = {2024},
  doi           = {10.52202/079017-0897},
  url           = {https://arxiv.org/abs/2405.16039},
  eprint        = {2405.16039},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2025 proceedings index (checked 2026-10-19)
@inproceedings{bae2025relaxed,
  author        = {Bae, Sangmin and Fisch, Adam and Harutyunyan, Hrayr and Ji, Ziwei and Kim, Seungyeon and Schuster, Tal},
  title         = {{Relaxed Recursive Transformers}: Effective Parameter Sharing with Layer-wise {LoRA}},
  booktitle     = {The Thirteenth International Conference on Learning Representations},
  year          = {2025},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html},
  eprint        = {2410.20672},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; PMLR volume 267 index (checked 2026-10-19)
@inproceedings{schone2025implicit,
  author        = {Sch{\"o}ne, Mark and Rahmani, Babak and Kremer, Heiner and Falck, Fabian and Ballani, Hitesh and Gladrow, Jannes},
  title         = {Implicit Language Models are {RNNs}: Balancing Parallelization and Expressivity},
  booktitle     = {Proceedings of the 42nd International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {267},
  pages         = {53439--53466},
  year          = {2025},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v267/schone25a.html},
  eprint        = {2502.07827},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{mcleish2025retrofitted,
  author        = {McLeish, Sean and Li, Ang and Kirchenbauer, John and others},
  title         = {Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence},
  year          = {2025},
  url           = {https://arxiv.org/abs/2511.07384},
  eprint        = {2511.07384},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; PMLR volume 306 index (checked 2026-10-19)
@inproceedings{geiping2026parallel,
  author        = {Geiping, Jonas and Yang, Xinyu and Su, Guinan},
  title         = {Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models},
  booktitle     = {Proceedings of the 43rd International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {306},
  pages         = {34442--34462},
  year          = {2026},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v306/geiping26a.html},
  eprint        = {2510.14961},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{koishekenov2025etd,
  author        = {Koishekenov, Yeskendir and Lipani, Aldo and Cancedda, Nicola},
  title         = {{Encode, Think, Decode}: Scaling test-time reasoning with recursive latent thoughts},
  year          = {2025},
  url           = {https://arxiv.org/abs/2510.07358},
  eprint        = {2510.07358},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{schwethelm2026isodepth,
  author        = {Schwethelm, Kristian and Rueckert, Daniel and Kaissis, Georgios},
  title         = {How Much Is One Recurrence Worth? {I}so-Depth Scaling Laws for Looped Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.21106},
  eprint        = {2604.21106},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{huang2026loopedtopology,
  author        = {Huang, Benhao and Shi, Chufan and Chen, Junlin and Wen, Shicheng and Liu, Zhengzhong and Xing, Eric and Ma, Xuezhe},
  title         = {Towards Looped Models Done Right, {Part I}: Topology, Input Injection, Recurrent-State Design},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.23036},
  eprint        = {2610.23036},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{ladhak2026elastic,
  author        = {Ladhak, Faisal and Acharya, Shantanu and Serano, Santiago Akle and Sun, Simeng and Ginsburg, Boris and Puvvada, Krishna C},
  title         = {One Model to Loop Them All: Elastic Latent Recurrence for Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.12582},
  eprint        = {2610.12582},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (comment) (checked 2026-10-19)
@misc{kohli2026loop,
  author        = {Kohli, Harsh and Parthasarathy, Srinivasan and Sun, Huan and Yao, Yuekun},
  title         = {{Loop, Think, \& Generalize}: Implicit Reasoning in Recurrent-Depth Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.07822},
  note          = {{Accepted} at COLM 2026 (arXiv comment)},
  eprint        = {2604.07822},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{wang2025hrm,
  author        = {Wang, Guan and Li, Jin and Sun, Yuhao and others},
  title         = {{Hierarchical Reasoning Model}},
  year          = {2025},
  url           = {https://arxiv.org/abs/2506.21734},
  eprint        = {2506.21734},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{jolicoeur2025trm,
  author        = {Jolicoeur-Martineau, Alexia},
  title         = {{Less is More}: Recursive Reasoning with Tiny Networks},
  year          = {2025},
  url           = {https://arxiv.org/abs/2510.04871},
  eprint        = {2510.04871},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{rodrigues2026hrm,
  author        = {Rodrigues, Leo Raphael and Kang, Jian},
  title         = {{Dissecting Hierarchical Reasoning Models}: A Mechanistic Study},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.22197},
  eprint        = {2609.22197},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{shakhvalieva2026recursive,
  author        = {Shakhvalieva, Yuliana and Kharchev, Dmitrii and Bezrukov, Viacheslav and Fedorova, Inessa and Bocharov, Dmitry and Oseledets, Ivan and Ternovskii, Valerii},
  title         = {What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.39967},
  eprint        = {2609.39967},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{pappone2025twoscale,
  author        = {Pappone, Francesco and Crisostomi, Donato and Rodol{\`a}, Emanuele},
  title         = {Two-Scale Latent Dynamics for Recurrent-Depth Transformers},
  year          = {2025},
  url           = {https://arxiv.org/abs/2509.23314},
  eprint        = {2509.23314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (comment) (checked 2026-10-19)
@misc{lu2025huginncot,
  author        = {Lu, Wenquan and Yang, Yuechuan and Lee, Kyle and Li, Yanshu and Liu, Enqi},
  title         = {Latent Chain-of-Thought? {D}ecoding the Depth-Recurrent Transformer},
  year          = {2025},
  url           = {https://arxiv.org/abs/2507.02199},
  note          = {{First} Workshop on the Application of LLM Explainability to Reasoning and Planning at COLM 2025 (arXiv comment)},
  eprint        = {2507.02199},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{logan2026pertoken,
  author        = {Logan, Joe},
  title         = {Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2607.14427},
  eprint        = {2607.14427},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{godin2026score,
  author        = {Godin, Guillaume},
  title         = {{SCORE}: Replacing Layer Stacking with Contractive Recurrent Depth},
  year          = {2026},
  url           = {https://arxiv.org/abs/2603.10544},
  eprint        = {2603.10544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page; models checked in arXiv HTML (checked 2026-10-19)
@misc{lai2026fractal,
  author        = {Lai, Jeffrey and Bao, Anthony and Quinn, John and Gilpin, William},
  title         = {Fractal basins trap latent reasoning},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.04963},
  eprint        = {2609.04963},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{inui2026invariant,
  author        = {Inui, Yuto and Konishi, Takuya and Kawahara, Yoshinobu},
  title         = {{Invariant-Measure Reasoners}: Stable Representations for Latent Reasoning},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.10996},
  eprint        = {2610.10996},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{yang2026extrapolate,
  author        = {Yang, Huitao and Huang, Jianhao and Wang, Zixuan and Mirzasoleiman, Baharan},
  title         = {Learning to Extrapolate in Depth with Looped Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.14456},
  eprint        = {2610.14456},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{li2026infiloop,
  author        = {Li, Pengxiang and Muhtar, Dilxat and He, Di and Su, Guinan and Yin, Lu and Liu, Shiwei},
  title         = {Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.11570},
  eprint        = {2610.11570},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{silva2026response,
  author        = {Lima de Jesus Silva, Jose Luis},
  title         = {Response Renormalization for Critical {Deep Equilibrium} Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.23725},
  eprint        = {2608.23725},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{olszowiec2026monitoring,
  author        = {Olszowiec, Pawel and Tomaszewska, Paulina and {\.Z}elaszczyk, Maciej and Byra, Michal and Chung, Neo Christopher},
  title         = {Monitoring latent reasoning in looped language models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.18127},
  eprint        = {2610.18127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{wang2026loopmonitor,
  author        = {Wang, Han and Balappanawar, Ishwar B and Zhang, Huan},
  title         = {On the Chain-of-Thought Monitorability of Looped Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.02741},
  eprint        = {2610.02741},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{wang2026safebridge,
  author        = {Wang, Yi and Qi, Xiuyuan and Han, Dongqi and Li, Dongsheng and Wang, Wenjie},
  title         = {Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.10625},
  eprint        = {2610.10625},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{azizi2026fewerloops,
  author        = {Azizi, Seyedarmin and Potraghloo, Erfan Baghaei and Fayyazi, Arya and Shokoufa, Saeid and Pedram, Massoud},
  title         = {Running Fewer Loops Removes Refusal Behaviour in Recurrent-Depth Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.15218},
  eprint        = {2610.15218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{zhao2026loopquant,
  author        = {Zhao, Zhixiong and Xu, Zukang and Hu, Xing and Liu, Lifeng and Sun, Guangyu and Yang, Dawei},
  title         = {{LoopQuant}: Accurate and Efficient Post-Training Quantization for Looped Transformers},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.16368},
  eprint        = {2610.16368},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{li2026loopy,
  author        = {Li, Zeyu and Zhang, Yipu and Chen, Jintao and Li, Xin and Zhang, Wei},
  title         = {{Loopy}: Low-Bit Quantization Framework for Looped Language Models},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.05265},
  eprint        = {2610.05265},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{dai2019transformerxl,
  author        = {Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  title         = {{Transformer-XL}: Attentive Language Models Beyond a Fixed-Length Context},
  booktitle     = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages         = {2978--2988},
  year          = {2019},
  publisher     = {Association for Computational Linguistics},
  doi           = {10.18653/v1/P19-1285},
  url           = {https://arxiv.org/abs/1901.02860},
  eprint        = {1901.02860},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{hutchins2022block,
  author        = {Hutchins, DeLesley and Schlag, Imanol and Wu, Yuhuai and Dyer, Ethan and Neyshabur, Behnam},
  title         = {{Block-Recurrent Transformers}},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {35},
  pages         = {33248--33261},
  year          = {2022},
  doi           = {10.52202/068431-2409},
  url           = {https://arxiv.org/abs/2203.07852},
  eprint        = {2203.07852},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{bulatov2022rmt,
  author        = {Bulatov, Aydar and Kuratov, Yuri and Burtsev, Mikhail S.},
  title         = {{Recurrent Memory Transformer}},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {35},
  pages         = {11079--11091},
  year          = {2022},
  doi           = {10.52202/068431-0805},
  url           = {https://arxiv.org/abs/2207.06881},
  eprint        = {2207.06881},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{oncescu2026recurrent,
  author        = {Oncescu, Costin-Andrei and Morwani, Depen and Jelassi, Samy and Meterez, Alexandru and Kwun, Mujin and Kakade, Sham},
  title         = {{The Recurrent Transformer}: Greater Effective Depth and Efficient Decoding},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.21215},
  eprint        = {2604.21215},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (comment: ICLR camera-ready) (checked 2026-10-19)
@inproceedings{yao2023react,
  author        = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  title         = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  booktitle     = {The Eleventh International Conference on Learning Representations},
  year          = {2023},
  url           = {https://arxiv.org/abs/2210.03629},
  eprint        = {2210.03629},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (six authors); Crossref DOI (record lists five, without Berman) (checked 2026-10-19)
@inproceedings{shinn2023reflexion,
  author        = {Shinn, Noah and Cassano, Federico and Berman, Edward and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu},
  title         = {{Reflexion}: Language Agents with Verbal Reinforcement Learning},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {36},
  pages         = {8634--8652},
  year          = {2023},
  doi           = {10.52202/075280-0377},
  url           = {https://arxiv.org/abs/2303.11366},
  eprint        = {2303.11366},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; Crossref DOI (checked 2026-10-19)
@inproceedings{madaan2023selfrefine,
  author        = {Madaan, Aman and Tandon, Niket and Gupta, Prakhar and others},
  title         = {{Self-Refine}: Iterative Refinement with Self-Feedback},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {36},
  pages         = {46534--46594},
  year          = {2023},
  doi           = {10.52202/075280-2019},
  url           = {https://arxiv.org/abs/2303.17651},
  eprint        = {2303.17651},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (comment: ICLR 2024) (checked 2026-10-19)
@inproceedings{huang2024selfcorrect,
  author        = {Huang, Jie and Chen, Xinyun and Mishra, Swaroop and Zheng, Huaixiu Steven and Yu, Adams Wei and Song, Xinying and Zhou, Denny},
  title         = {Large Language Models Cannot Self-Correct Reasoning Yet},
  booktitle     = {The Twelfth International Conference on Learning Representations},
  year          = {2024},
  url           = {https://arxiv.org/abs/2310.01798},
  eprint        = {2310.01798},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; PMLR volume 235 index (checked 2026-10-19)
@inproceedings{du2024debate,
  author        = {Du, Yilun and Li, Shuang and Torralba, Antonio and Tenenbaum, Joshua B. and Mordatch, Igor},
  title         = {Improving Factuality and Reasoning in Language Models through Multiagent Debate},
  booktitle     = {Proceedings of the 41st International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {235},
  pages         = {11733--11763},
  year          = {2024},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v235/du24e.html},
  eprint        = {2305.14325},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{wu2026converge,
  author        = {Wu, Xuening and Xu, Qianya and Kang, Yanlan and Chen, Zeping and Liu, Yubin and Yin, Shenqin},
  title         = {Do Language Models Converge to Themselves? {R}ecursive Self-Refinement as Textual Relaxation},
  year          = {2026},
  url           = {https://arxiv.org/abs/2607.22653},
  eprint        = {2607.22653},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{mirzaei2026sample,
  author        = {Mirzaei, Iliya},
  title         = {{Sample More, Reflect Less}: {Self-Refine} and {Reflexion} Lose to Repeated Sampling at Equal Token Cost, from {1.5B} to {7B}},
  year          = {2026},
  url           = {https://arxiv.org/abs/2607.28576},
  eprint        = {2607.28576},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{yao2024taubench,
  author        = {Yao, Shunyu and Shinn, Noah and Razavi, Pedram and Narasimhan, Karthik},
  title         = {$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  year          = {2024},
  url           = {https://arxiv.org/abs/2406.12045},
  eprint        = {2406.12045},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page; ICLR 2025 proceedings index (checked 2026-10-19)
@inproceedings{perez2025telephone,
  author        = {Perez, J{\'e}r{\'e}my and Kova{\v{c}}, Grgur and L{\'e}ger, Corentin and Colas, C{\'e}dric and Molinaro, Gaia and Derex, Maxime and Oudeyer, Pierre-Yves and Moulin-Frier, Cl{\'e}ment},
  title         = {When {LLMs} Play the {Telephone Game}: Cultural Attractors as Conceptual Tools to Evaluate {LLMs} in Multi-turn Settings},
  booktitle     = {The Thirteenth International Conference on Learning Representations},
  year          = {2025},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html},
  eprint        = {2407.04503},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{laban2026lost,
  author        = {Laban, Philippe and Hayashi, Hiroaki and Zhou, Yingbo and Neville, Jennifer},
  title         = {{LLMs} Get Lost In Multi-Turn Conversation},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html},
  eprint        = {2505.06120},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (comment: COLM 2024) (checked 2026-10-19)
@inproceedings{li2024instability,
  author        = {Li, Kenneth and Liu, Tianle and Bashkansky, Naomi and Bau, David and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  title         = {Measuring and Controlling Instruction ({In)Stability} in Language Model Dialogs},
  booktitle     = {First Conference on Language Modeling},
  year          = {2024},
  url           = {https://arxiv.org/abs/2402.10962},
  eprint        = {2402.10962},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{verazuniga2026fixedpoints,
  author        = {Vera Z{\'u}{\~n}iga, Nicol{\'a}s},
  title         = {Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.21315},
  eprint        = {2608.21315},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{verazuniga2026census,
  author        = {Vera Z{\'u}{\~n}iga, Nicol{\'a}s},
  title         = {What a Cross-Model Fixed-Point Census Can and Cannot Arbitrate About Repetition},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.29507},
  eprint        = {2609.29507},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{atil2024nondeterminism,
  author        = {Atil, Berk and Aykent, Sarp and Chittams, Alexa and others},
  title         = {Non-Determinism of ``Deterministic'' {LLM} Settings},
  year          = {2024},
  url           = {https://arxiv.org/abs/2408.04667},
  eprint        = {2408.04667},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (journal-ref: TMLR 2026) (checked 2026-10-19)
@article{du2026greedy,
  author        = {Du, Gaoyuan and Khan, Anam Nawaz and Zhou, Rex and Liu, Xiaoyang and Chakrabarti, Deepayan and Suya, Fnu and Li, Xueping},
  title         = {Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in {LLM} Inference},
  journal       = {Transactions on Machine Learning Research},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.26621},
  eprint        = {2609.26621},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{chu2026margingate,
  author        = {Chu, Kexin and Zhou, Yang and Zhang, Wei},
  title         = {{MarginGate}: Sparse Margin-Triggered Verification for Batch-Invariant {LLM} Inference},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30218},
  eprint        = {2605.30218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{qin2026vosti,
  author        = {Qin, Jianxing and Du, Alexander and Zhang, Danfeng and Lentz, Matthew and Zhuo, Danyang},
  title         = {{Vosti}: Specifying, Implementing, and Verifying Deterministic {LLM} Inference},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.38981},
  eprint        = {2609.38981},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{zhao2026corun,
  author        = {Zhao, Shiju and Yang, Jiacheng and Chen, Qihang and Hu, Junhao and Zheng, Jiaqi and Chen, Guihai and Chen, Xusheng},
  title         = {{CoRun}: Padding is Simple and Efficient for Deterministic {LLM} Inference},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.14376},
  eprint        = {2608.14376},
  archivePrefix = {arXiv},
  primaryClass  = {cs.OS}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{pape2026silent,
  author        = {Pape, David and Evertz, Jonathan and Sch{\"o}nherr, Lea},
  title         = {{The Silent Hyperparameter}: Quantifying the Impact of Inference Backends on {LLM} Reproducibility},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.19537},
  eprint        = {2605.19537},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{khatri2026probes,
  author        = {Khatri, Alizishaan},
  title         = {{Same Probe, Different Numbers}: Are Activation Probes Robust to Inference-Time Numerical Non-Determinism?},
  year          = {2026},
  url           = {https://arxiv.org/abs/2609.31796},
  eprint        = {2609.31796},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

% verified: arXiv abs page; full text searched in arXiv HTML (checked 2026-10-19)
@misc{openai2024o1,
  author        = {{OpenAI}},
  title         = {{OpenAI} {o1} System Card},
  year          = {2024},
  url           = {https://arxiv.org/abs/2412.16720},
  eprint        = {2412.16720},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (journal-ref); Crossref DOI; full text searched in arXiv HTML (checked 2026-10-19)
@article{deepseek2025r1,
  author        = {Guo, Daya and Yang, Dejian and Zhang, Haowei and others},
  title         = {{DeepSeek-R1} incentivizes reasoning in {LLMs} through reinforcement learning},
  journal       = {Nature},
  volume        = {645},
  pages         = {633--638},
  year          = {2025},
  doi           = {10.1038/s41586-025-09422-z},
  url           = {https://arxiv.org/abs/2501.12948},
  note          = {{arXiv} title: {DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}},
  eprint        = {2501.12948},
  archivePrefix = {arXiv}
}

% verified: arXiv abs page; full text searched in arXiv HTML (checked 2026-10-19)
@misc{yang2025qwen3,
  author        = {Yang, An and Li, Anfeng and Yang, Baosong and others},
  title         = {{Qwen3} Technical Report},
  year          = {2025},
  url           = {https://arxiv.org/abs/2505.09388},
  eprint        = {2505.09388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; full text searched in arXiv HTML (checked 2026-10-19)
@misc{comanici2025gemini,
  author        = {Comanici, Gheorghe and Bieber, Eric and Schaekermann, Mike and others},
  title         = {{Gemini 2.5}: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities},
  year          = {2025},
  url           = {https://arxiv.org/abs/2507.06261},
  eprint        = {2507.06261},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page; architecture section read in arXiv HTML (checked 2026-10-19)
@misc{openai2025gptoss,
  author        = {{OpenAI}},
  title         = {{gpt-oss-120b} \& {gpt-oss-20b} Model Card},
  year          = {2025},
  url           = {https://arxiv.org/abs/2508.10925},
  eprint        = {2508.10925},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{korbak2025monitorability,
  author        = {Korbak, Tomek and Balesni, Mikita and Barnes, Elizabeth and others},
  title         = {{Chain of Thought Monitorability}: A New and Fragile Opportunity for {AI} Safety},
  year          = {2025},
  url           = {https://arxiv.org/abs/2507.11473},
  eprint        = {2507.11473},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

% verified: arXiv abs page (checked 2026-10-19)
@misc{inception2025mercury,
  author        = {{Inception Labs} and Khanna, Samar and Kharbanda, Siddhant and others},
  title         = {{Mercury}: Ultra-Fast Language Models Based on Diffusion},
  year          = {2025},
  url           = {https://arxiv.org/abs/2506.17298},
  eprint        = {2506.17298},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

### A.2 Already in `paper/llm/references.bib` (copied verbatim)

```bibtex
% already in paper/llm/references.bib (full report); verified: arXiv abs page; ICLR 2026 proceedings index (checked 2026-10-19)
@inproceedings{wu2025singlethreaded,
  author        = {Wu, Junhong and Lu, Jinliang and Ren, Zixuan and Hu, Gangqiang and Wu, Zhi and Dai, Dai and Wu, Hua},
  title         = {{LLMs} are Single-threaded Reasoners: Demystifying the Working Mechanism of {Soft Thinking}},
  booktitle     = {The Fourteenth International Conference on Learning Representations},
  pages         = {4533--4548},
  year          = {2026},
  url           = {https://proceedings.iclr.cc/paper_files/paper/2026/hash/0895de150660b6c2aef9b0f098cb1dcd-Abstract-Conference.html},
  note          = {arXiv:2508.03440},
  eprint        = {2508.03440},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{backour2026mixture,
  author = {Backour, Ali}, title = {The Dynamics of Continuous Mixture Collapse in Language Models}, year = {2026},
  howpublished = {arXiv preprint arXiv:2609.02049}, eprint = {2609.02049}, archivePrefix = {arXiv},
  primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2609.02049}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{rizvimartel2026illusion,
  author = {Rizvi-Martel, Michael and Rabusseau, Guillaume and Mosbach, Marius},
  title = {The Illusion of Superposition? {A} Principled Analysis of Latent Thinking in Language Models}, year = {2026},
  howpublished = {arXiv preprint arXiv:2604.06374}, eprint = {2604.06374}, archivePrefix = {arXiv},
  primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2604.06374}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{yu2026cots,
  author = {Yu, Jie and Sun, Yuxi and Li, Zhaoxu and Liu, Chunxiao and Jiang, Xudong},
  title = {{CoTS}: Competitive Token Selection for Latent Reasoning in Large Language Models}, year = {2026},
  howpublished = {arXiv preprint arXiv:2610.20642}, eprint = {2610.20642}, archivePrefix = {arXiv},
  primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2610.20642}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{li2026later,
  author = {Li, Xuan and Wang, Yining and Liu, Yuchen and Liu, Guanjun and Qiu, Delai and Liu, Shengping and Liang, Jiaen and Huang, Wei and Yu, Jun and Zhu, Junnan},
  title = {{LaTER}: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification}, year = {2026},
  howpublished = {arXiv preprint arXiv:2605.07315}, eprint = {2605.07315}, archivePrefix = {arXiv},
  primaryClass = {cs.CL}, url = {https://arxiv.org/abs/2605.07315}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (comment) (checked 2026-10-19)
@misc{koriagin2026nohead,
  author = {Koriagin, Nikita and Aksenov, Yaroslav and Bredis, George and Gerasimov, Gleb and Balagansky, Nikita and Gavrilov, Daniil},
  title = {A Model with No Head and Many Thoughts}, year = {2026},
  howpublished = {arXiv preprint arXiv:2608.31069; Findings of EMNLP 2026}, eprint = {2608.31069}, archivePrefix = {arXiv},
  primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2608.31069}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (33 authors, first Rui-Jie Zhu, last Jason Eshraghian) (checked 2026-10-19)
@misc{zhu2025ouro,
  author        = {Zhu, Rui-Jie and Wang, Zixuan and Hua, Kai and others},
  title         = {Scaling Latent Reasoning via Looped Language Models},
  year          = {2025},
  howpublished  = {arXiv preprint arXiv:2510.25741},
  url           = {https://arxiv.org/abs/2510.25741},
  eprint        = {2510.25741},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{prairie2026parcae,
  author        = {Prairie, Hayden and Novack, Zachary and Berg-Kirkpatrick, Taylor and Fu, Daniel Y.},
  title         = {{Parcae}: Scaling Laws For Stable Looped Language Models},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2604.12946},
  eprint        = {2604.12946},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.12946}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{huang2026loopedfixed,
  author        = {Huang, Benhao and Shi, Chufan and Chen, Junlin and Wen, Shicheng and Liu, Zhengzhong and Xing, Eric and
                   Ma, Xuezhe},
  title         = {Towards Looped Models Done Right, Part {II}: Rethinking at Fixed Points},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2610.06833},
  url           = {https://arxiv.org/abs/2610.06833},
  note          = {Submitted 5 October 2026},
  eprint        = {2610.06833},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2610.06833}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{movahedi2026fixedpoint,
  author        = {Movahedi, Sajad and Milovanovi{\'c}, Vera and Feigin, Shlomo Libo and Theus, Alexander and
                   Hofmann, Thomas and Boeva, Valentina and Rusch, T. Konstantin and Orvieto, Antonio},
  title         = {Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2606.18206},
  url           = {https://arxiv.org/abs/2606.18206},
  note          = {Version 1, 16 June 2026},
  eprint        = {2606.18206},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2606.18206}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{viakhirev2026thinkshallow,
  author = {Viakhirev, Ivan and Borodin, Kirill and Almutairi, Amirah and others},
  title = {Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth}, year = {2026},
  howpublished = {arXiv preprint arXiv:2608.18222}, eprint = {2608.18222}, archivePrefix = {arXiv},
  primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2608.18222}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{wang2026taps,
  author        = {Wang, Boyuan and Yu, Chengyao and Ren, Jiaxi and Wei, Hongxin and Jing, Bingyi and Tao, Yuxin},
  title         = {Scheduling Recursive Reasoning in Looped Transformers},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2609.36653},
  url           = {https://arxiv.org/abs/2609.36653},
  note          = {Submitted 29 September 2026},
  eprint        = {2609.36653},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2609.36653}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{labovich2026stability,
  author        = {Labovich, Asher},
  title         = {Stability and Generalization in Looped Transformers},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2604.15259},
  url           = {https://arxiv.org/abs/2604.15259},
  note          = {Version 2, 22 April 2026},
  eprint        = {2604.15259},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2604.15259}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{blayney2026mechanistic,
  author = {Blayney, Hugh and Arroyo, {\'A}lvaro and Obando-Ceron, Johan and others},
  title = {A Mechanistic Analysis of Looped Reasoning Language Models}, year = {2026},
  howpublished = {arXiv preprint arXiv:2604.11791}, eprint = {2604.11791}, archivePrefix = {arXiv},
  primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2604.11791}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{buehlmaier2026layernorm,
  author        = {Buehlmaier, Matthias M. M.},
  title         = {{LayerNorm} as Implicit Gain Control in Looped Transformers},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2607.10681},
  url           = {https://arxiv.org/abs/2607.10681},
  note          = {Version 1, 12 July 2026},
  eprint        = {2607.10681},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.10681}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{wang2026jacobianlens,
  author        = {Wang, Wenlong and Reid, Fergal},
  title         = {Looped Transformers under the {J}acobian Lens: Does the Global Workspace Survive Recurrence?},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2609.01924},
  url           = {https://arxiv.org/abs/2609.01924},
  note          = {Submitted 1 September 2026},
  eprint        = {2609.01924},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2609.01924}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{park2026loopedearlier,
  author = {Park, Sumin and Rusch, T. Konstantin}, title = {Looped Reasoning Finishes Earlier Than You Think}, year = {2026},
  howpublished = {arXiv preprint arXiv:2610.18174}, eprint = {2610.18174}, archivePrefix = {arXiv},
  primaryClass = {cs.LG}, url = {https://arxiv.org/abs/2610.18174}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (venue in our report bib not re-verified: OpenReview blocked automated access) (checked 2026-10-19)
@inproceedings{welleck2022selfcorrect,
  author        = {Welleck, Sean and Lu, Ximing and West, Peter and Brahman, Faeze and Shen, Tianxiao and Khashabi,
                   Daniel and Choi, Yejin},
  title         = {Generating Sequences by Learning to Self-Correct},
  booktitle     = {The Eleventh International Conference on Learning Representations},
  year          = {2023},
  url           = {https://openreview.net/forum?id=hH36JeQZDaO},
  note          = {arXiv:2211.00053},
  eprint        = {2211.00053},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page; Crossref DOI 10.1609/aaai.v35i14.17520 (checked 2026-10-19)
@article{fu2021repetition,
  author        = {Fu, Zihao and Lam, Wai and So, Anthony Man-Cho and Shi, Bei},
  title         = {A Theoretical Analysis of the Repetition Problem in Text Generation},
  journal       = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume        = {35},
  number        = {14},
  pages         = {12848--12856},
  year          = {2021},
  doi           = {10.1609/aaai.v35i14.17520},
  note          = {arXiv:2012.14660},
  eprint        = {2012.14660},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{leins2026drift,
  author        = {Leins, Nicolas and Haase, Jennifer and Geronimus, Varvara and Gonnermann-M{\"u}ller, Jana and
                   Pokutta, Sebastian},
  title         = {Prompting Against Persona Drift: Comparing Intervention Timing and Content in {LLM}-Simulated
                   Conversations},
  year          = {2026},
  howpublished  = {arXiv preprint arXiv:2609.24532},
  url           = {https://arxiv.org/abs/2609.24532v1},
  note          = {Version 1, 21 September 2026},
  eprint        = {2609.24532},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: arXiv abs page (checked 2026-10-19)
@misc{liu2025truthdecay,
  author        = {Liu, Joshua and Jain, Aarav and Takuri, Soham and Vege, Srihan and Akalin, Aslihan and Zhu, Kevin and
                   O'Brien, Sean and Sharma, Vasu},
  title         = {{TRUTH DECAY}: Quantifying Multi-Turn Sycophancy in Language Models},
  year          = {2025},
  howpublished  = {arXiv preprint arXiv:2503.11656},
  url           = {https://arxiv.org/abs/2503.11656v1},
  note          = {Version 1},
  eprint        = {2503.11656},
  archivePrefix = {arXiv}
}

% already in paper/llm/references.bib (full report); verified: Crossref DOI (checked 2026-10-19)
@article{shumailov2024collapse,
  author  = {Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Papernot, Nicolas and Anderson, Ross and
             Gal, Yarin},
  title   = {{AI} models collapse when trained on recursively generated data},
  journal = {Nature},
  volume  = {631},
  number  = {8022},
  pages   = {755--759},
  year    = {2024},
  doi     = {10.1038/s41586-024-07566-y}
}
```
