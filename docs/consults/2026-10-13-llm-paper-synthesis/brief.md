# Brief: a language model's own inference dynamics, synthesis before arXiv

13 October 2026. A private manuscript of about 45 pages, aimed at cs.LG with cross-list nlin.CD, now
summarizes the work below. This brief asks for a director-level judgement of the claims, the literature, the
last experiment before posting, and the paper's shape. Consult 5 (12 October) found no prior language-model
study that combines continued fold and flip curves, primitive windows, period adding and calibrated
intermittency. It advised narrow claims and a prospective test of the fold mechanism; both were done.

## Systems

1. **Soft-token map.** x_(n+1) = E^T softmax(z(x_n)/T): the expected next-token embedding of a frozen model fed
   back at a fixed position after a fixed chat prefix. Steering alpha v is added at one layer (contrastive concept
   directions: philosophy, gratitude, hostility-vs-kindness). This is a smooth map; T -> 0 recovers greedy
   decoding. Float64 throughout, with Hugging Face's float32 RMSNorm and rotary tables patched.
2. **Huginn-0125 latent loop.** At fixed tokens, h <- (1 - lambda) h + lambda F(h).
3. **Sampled self-dialogue.** Two copies converse (T 0.7, 128 replicas x 40 turns), steered along a hostility
   direction; the outcome is "hostile" when a pinned toxicity classifier exceeds 0.5. A second, independently
   trained classifier was used for robustness.

Evidence rungs: 1 = numerically validated in float64; 2 = reproduced by an independent implementation or on other
hardware; "frozen" = design and analysis pushed before any outcome was read.

## Principal findings

**A. Organization of the soft-token map (Qwen2.5-0.5B).**
- Chaos with embedded periodic windows.
- A period-adding family of primitive orbits (". This" + k "is X" pairs), with window law d_N N^2 = const.
- Mechanism: type-I intermittency at the fold of the 2-cycle "is it". The fold's normal form u' = u + a eps + b u^2,
  from derivatives without fitting, predicts the controlled bottleneck passage times (rung 2).
- Prospective frozen test at two new folds: 23 of 24 forecasts were within one iterate, so the frozen all-row
  criterion failed by one row.
- Period doubling: flips located through 48 -> 96, and superstable points through period 384. The finite-order
  ratios rise to 4.625, within 1 % of Feigenbaum's delta.
- Shrimp organization (rung 1):
  - A period-6 window is bounded by fold curves that end in two cusps, with a flip curve inside.
  - At each cusp, b passes through zero with nonzero slope along the curve, and the cubic coefficient is nonzero:
    c = l^T C(r,r,r)/6 + l^T B(r,h2)/2, with the second-order centre-manifold term, from float64 JVP differences
    on a step plateau.
  - A period-4 fold curve carries two more such cusps.

**B. Hostility-steered map (Qwen2.5-0.5B).**
- A hostile period-adding cascade made of laps around the 2-cycle "!Yes". The fold normal form predicts the
  spacing per lap within 3 %; the fold was reproduced by an independent implementation (rung 2).
- A period-7 body ("I!Yes heels / defiance") whose fold curves end in two nondegenerate cusps (c < 0; three
  coexisting period-7 orbits).
- Inside the body, a full doubling cascade to chaos and back: 7 -> 14 -> 28 -> chaos with satellite windows 21,
  35 and 42 -> 28 -> 14 -> 7. Primitive period-14, -28 and -21 orbits are verified, and a flip bubble is
  located on the period-14 branch.

**C. Scale and other models.**
- Qwen2.5-1.5B: a doubling route with float64 flips 4 -> 8 -> 16 -> 32 -> 64.
- Qwen2.5-7B: coarse float64 maps (8-24 % chaotic, windows). Two verified orbits (period 6, nearly superstable;
  period 3). Locating a bifurcation at 7B is in progress; at 7B one continuation step takes about 2 minutes.
- OLMo-2 7B: coarse maps 6-13 % chaotic, with windows, including a lap-adding family of the 2-cycle
  "!<|endoftext|>".
- Running now:
  - a same-architecture comparison (Qwen2-0.5B against Qwen2.5-0.5B, gratitude and philosophy planes);
  - Llama-3.1-8B-Instruct maps, with attractors read out in two sparse autoencoders (Goodfire layer 19; a TopK
    layer-24 SAE). Both SAEs' hook points were verified.

**D. Huginn.**
- A Neimark-Sacker bifurcation predicted from the fixed-point spectrum (lambda_c = 1.0777) and observed
  (amplitude fit 1.0778, supercritical, rotation 0.156 per step).
- The GSM8K prompts' spectral thresholds (1.074-1.126) predict where over-relaxation destabilizes the loop.
- Frozen test, reviewed twice by an independent agent before the freeze, on 200 held-out GSM8K problems:
  - Accuracy 0.330 / 0.395 / 0.255 / 0.095 at lambda 1.0 / 1.05 / 1.15 / 1.20.
  - Pre-stated decline at 1.20: -23.5 points, McNemar p = 3e-12. Non-inferiority at 1.05: lower bound +2.5.
    Both pass, and the result was replayed exactly by an independent implementation.
- Item-level part (does an estimated Ritz feature of each held-out ARC question rank its own relaxed residual?
  Spearman, cohort of at least 48 of 64): all 64 questions admitted. Mean within-arm Spearman S = 0.701, 95 % CI
  [0.576, 0.779], over the four arms 1.0, 1.05, 1.08, 1.10. The pre-stated lower-bound criterion (> 0.3) passes.
  The independent replay is pending.

**E. Misaligned behaviour as an attractor (sampled dialogue).**
- Capture, at 0.5B:
  - The ladder is steep and specific to the direction: 10 -> 87 % hostile turns for alpha 0.04 -> 0.07; random
    and cooking directions stay near 0.
  - The stay probability rises 0.36 -> 0.94.
  - Hysteresis: ramp minus fresh is +0.29 [0.23, 0.35].
- Control:
  - A counter-steer triggered after a hostile reply is less dose-efficient than constant prevention.
  - Detect-and-replace (regenerate a flagged reply with a counter-steer before it enters the context) cuts the
    model's own hostility from 59 % to 12 %, at a quarter of the dose that constant prevention needs.
  - At 1.5B the dose response is not monotone (coherent hostility, then token soup, then hostility again).
    Detect-and-replace replicates on the rising side (43 % -> 10 %; 24 points better than constant at equal dose),
    but beyond the peak every counter-steer backfires.
- Detection:
  - The pre-specified lead-time test of dynamical estimates was negative.
  - A probe of the model's in-context state adds a small warning, confirmed frozen on 1024 fresh dialogues
    (+0.033 AUC [0.018, 0.048]).
  - A Monte Carlo committor (32 continuations per state) does not beat the probe; the attainable ceiling is
    unresolved.
- Second classifier: it confirms most contrasts; one hysteresis contrast and the reading of a kindness control do
  not survive.

## Known weaknesses

- One prompt and one steering construction for the maps.
- The behaviour is measured by automated classifiers only, with no human labels.
- Two small models for the behavioural study.
- No computer-assisted proof yet; an interval certificate for one orbit is in progress.
- The constructed map is not ordinary sampled decoding. A bridge section shows that noise shortens the laminar
  phases.

## Questions (please answer each, concisely, with precise references where asked)

Q1. Claims against evidence. For an expert nonlinear-dynamics reader and an ML reader:
  - Which headline claims are overstated or understated?
  - Is "the codimension-two organization of a shrimp" justified by numerically nondegenerate cusps, flip curves
    and, for the hostile body, a cascade inside? What minimal further evidence would a specialist in shrimps
    (Gallas; Facanha et al. 2013) require: superstable "endoskeleton" curves, the full unfolding, or
    self-similar inner shrimps?
  - Is "misaligned behaviour as an attractor" defensible with a classifier proxy and two small models?

Q2. Literature. Is there any published work up to October 2026 that preempts or must be cited for:
  - shrimps or codimension-two bifurcation structure in maps defined by neural networks or transformers;
  - steering strength treated as a bifurcation parameter;
  - committor or early-warning monitors for language-model behaviour;
  - detect-and-regenerate control during generation?
  Give exact citations only; we check every one at the primary record.

Q3. The single most valuable addition before posting, given about [amount redacted] of rented GPU time plus local CPU. The
options we see:
  - complete the interval proof (running);
  - human labels for a few hundred replies;
  - a frozen confirmation of detect-and-replace at 1.5B on fresh seeds;
  - float64 shrimp curves in a second model;
  - the same-architecture comparison (running);
  - a behavioural replication at 7B.
  Which one, and why?

Q4. Shape. Should the deterministic-map dynamics and the sampled-behaviour study be one paper or two? If one,
what should be cut or moved to appendices so that the central claim is clear?
