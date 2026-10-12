# Adjudication: LLM-dynamics novelty and research-plan consult (12 October 2026)

One `research-strategy` request (gpt-6-astra, Pro mode, xhigh), sent with `brief.md` at commit f6569a5d.
Raw request, response and manifest: `../neural-shrimp-data/consults/2026-10-12-llm-dynamics-novelty/execute-001/`.
Advisory only; every point below was checked or decided locally.

## References (checked at the primary record before use)

| Reference | Check |
| --- | --- |
| Bai, Kolter, Koltun, "Deep Equilibrium Models", arXiv:1909.01377 (2019) | arXiv record: title, authors, 3 Sep 2019 |
| Bai, Koltun, Kolter, "Stabilizing Equilibrium Models by Jacobian Regularization", arXiv:2106.14342 (2021) | arXiv record |
| Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space", arXiv:2412.06769 (2024) | arXiv record |
| Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", arXiv:2502.05171 (2025) | arXiv record |
| Holtzman et al., "The Curious Case of Neural Text Degeneration", arXiv:1904.09751 (2019) | arXiv record |
| Xu et al., "Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation", arXiv:2206.02369 (2022) | arXiv record |
| Chodera, Noé, "Markov state models of biomolecular conformational dynamics", Curr. Opin. Struct. Biol. 25:135-144 (2014), doi:10.1016/j.sbi.2014.04.002 | Crossref record |
| Gallas, "Structure of the parameter space of the Hénon map", Phys. Rev. Lett. 70:2714-2717 (1993), doi:10.1103/PhysRevLett.70.2714 | Crossref record |
| Pomeau, Manneville, "Intermittent transition to turbulence in dissipative dynamical systems", Commun. Math. Phys. 74(2):189-197 (1980), doi:10.1007/BF01197757 | Crossref record (checked earlier today) |

Venues stated in the response (NeurIPS 2019, ICML 2021, ICLR 2020, NeurIPS 2022) were not checked; cite the
arXiv records until they are. The Claude 4 system card (Anthropic, 2025) has no DOI or arXiv identifier;
cite it by title and URL only after re-reading the relevant section.

## Novelty

The reviewer knows no earlier language-model study that combines continued fold and flip curves, primitive
periodic windows, period adding and calibrated intermittency, and none that applies Markov state models to
self-dialogue attractors, while stating that this is not an exhaustive search. Accepted consequence: claims
stay narrow ("a demonstrated mechanism in a precisely specified inference system"), never "the first
dynamical analysis of language-model inference" (fixed-point and Jacobian stability analyses of deep
equilibrium models exist). A systematic literature search remains our job before any manuscript.

## Theory (accepted, with one check)

For the two-step map G = f^2 at the 2-cycle's fold, with |r| = 1 and l^T r = 1, the normal form is
u' = u + a epsilon + b u^2 with a = l^T d_alpha G and b = (1/2) l^T D^2 G[r, r]. The stable-branch multiplier
gives (1 - mu)^2 / (4 (alpha_c - alpha)) -> ab (the product), and passage between fixed entrance and exit
points straddling the bottleneck takes about pi / sqrt(ab epsilon) G-iterates, so windows selected by a fixed
traversal count sit at epsilon_N = pi^2 / (ab (N + delta)^2). A persistent fractional prefactor needs an
entrance or exit that approaches the bottleneck on the sqrt(epsilon) scale. Our clock is already in
G-iterates (one lap = one "is X" pair = two steps). With delta free, the stripe centres demand ab = 374 and
delta near 0; ab = 130 would require a drifting delta (2.8 at N = 4, 4.9 at N = 7), so the float64 normal
form decides between "the window law holds with the local coefficient" and "the windows are not set by the
local passage alone".

## Plan decisions

1. **Accepted, next:** float64 fold normal-form passage test at log10 T -0.115: refine the fold, compute
   r, l, a, b (b from finite differences of float64 JVPs over a step-size plateau, a including the
   steering's effect on the prefix), compare ab with the stable-branch multiplier extrapolation (float64
   branch running on CPU), then predict held-out passage counts (epsilon = 3e-6 .. 1e-4) from controlled
   entrances (u_in = -U, u_in = 0, natural) with the discrete quadratic map and the arctangent formula; an
   analytic saddle-node map as positive control, norm-matched transverse perturbations as negative control.
2. **Accepted:** Huginn causal clamp: above onset, clamp the first unstable position (the '.') to its fixed
   point and measure the downstream oscillation, with sham clamps of earlier and later positions; held-out
   prompts as independent units.
3. **Accepted:** bridge to sampling: replace the expected embedding by the mean of m sampled token embeddings
   (m = infinity, 64, 8, 1; m = 1 is sampled decoding in the same truncated-context system) and measure
   passage times near the fold.
4. **Accepted, reframing:** self-dialogue results are reported as finite-horizon occupancy and censored dwell
   times with dialogue-level bootstrap and frozen state definitions; no stationary-weight interpretation of
   40-turn chains. The running gratitude ladder is kept as development data.
5. **Accepted:** region E (two-dimensional 4-cycles): continue one component's fold and flip boundaries and
   certify any cusp by normal-form conditions (vanishing quadratic coefficient, nonzero cubic, transversal
   unfolding); a smooth turn of a fold curve (as on the period-8 stripe so far) is not a cusp.
6. **Partly declined:** "drop undirected high-resolution grids". The human asked for higher-resolution maps;
   the two 256x256 float64 maps already running also verify periods in float64 and locate targets for (5).
   No further undirected grids after them.
7. **Accepted wording fixes:** T -> 0 recovers greedy decoding only for the fixed-prefix, one-token-window
   system with a unique argmax; expected-embedding feedback is a constructed dynamical system, not yet a
   validated proxy for sampled decoding (until (3)).
