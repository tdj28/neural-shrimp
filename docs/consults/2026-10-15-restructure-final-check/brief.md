# Consult 10 brief: final check of the restructured paper, and the anatomy of its shrimp-like windows

15 October 2026. Private draft: "Chaos and bifurcations in continuous language-model feedback" (arXiv cs.CL, with
cross-lists to nlin.CD and cs.LG). The author wants the results rock solid and of real interest to frontier-lab
researchers in interpretability, alignment and inference. Today the paper was restructured, and one more
continuation run (FM5, below) finished. A second AI agent (Codex) has reviewed every line of the paper against saved
data, and its scope corrections are applied. The author edits after this consult.
Resources: about [amount redacted] of GPU credit (H100 at about [amount redacted]-3.2 per hour), and a few days. A 0.5B model is fast, a 7B
model is feasible, and a 70B model is slow.

## System

- **Loop.** A fixed chat prefix is cached, then one soft token x in R^d. One step: z = logits(prefix, x),
  p = softmax(z/T), x' = E^T p, with E the input embedding.
- **Steering.** Steering adds alpha v to the residual stream after one block. "Two-block" planes steer two concepts
  at two depths: (alpha1, alpha2) at fixed T.
- **Precision.** Float64 throughout, including the norms and rotary tables.
- **Chaos label.** Twin exponent above 1e-3 and no period up to 64. Exact tangents agree with the twins.
- **Derivative.** DF = (1/T) Sigma_p J: the embedding covariance under p times the network Jacobian. Without the
  network this is a modern Hopfield update, which cannot cycle.

## The paper as restructured today

- **Title.** It was "Bifurcations and chaos in language models fed their own output".
- **Abstract** (unchanged today): about 1900 characters, opening with Soft Thinking. It covers:
  - chaos in 0.5B-72B Qwen2.5, OLMo-2 and Llama-3 models;
  - batch-size-only reruns, where chaotic float64 loops separate until their readout tokens differ, after the number
    of steps the Lyapunov exponent sets, and periodic loops stay together (four models, up to 70B);
  - frozen arithmetic tests, where batch layout changed 56 % and 55 % of bf16 soft-thinking answers (0.5B, 7B),
    against never and 10 % for a forced greedy baseline;
  - repeating text that hides chaos;
  - a steering mechanism: tokens compete, so the readout expands; a logit bias alone suffices;
  - depth effects, and two depths creating chaos neither creates alone;
  - two loops locking each other into periodic orbits;
  - no accuracy marker and no edge-of-chaos advantage;
  - Feigenbaum ratios to 4.6688, a 1/N^2 window law, 23 of 24 frozen passage forecasts within one iterate, and an
    interval proof of an attracting period-8 orbit;
  - a confirmed Huginn spectral prediction.
- **Contributions,** now four groups:
  1. chaos in released models and what it does to inference;
  2. why the loop is chaotic and how steering makes it so;
  3. low-dimensional organization predicted from derivatives;
  4. beyond the one-token loop (longer memory, Soft Thinking's growing context, 15 models from 0.5B to 72B, two
     agents writing one text, Huginn).
- **Main text, 37 pages:**
  - introduction (about 5 pages, with related work);
  - map and precision;
  - theory (the factorization, temperature limits, the Hopfield limit, "stability belongs to the loop", the gain
    split and crossover, Singer's constraint);
  - methods;
  - phase maps;
  - consequences (reproducibility, depth and two blocks, behaviour nulls);
  - period adding;
  - fold mechanism with the prospective test;
  - two new summary sections: "Cusps, cascades and shrimp-like windows" and "Beyond the one-token loop";
  - two loops in conversation;
  - discussion (about 5 pages).
- **Appendices A-J, 47 pages:** theory background and Kaplan-Yorke dimensions; curves and cusps; cascade; hostility
  and shrimp details; bridge to sampling; replication; size; Huginn; unsteered chaos and controls; evidence and
  reproducibility.

## Shrimp-like windows: the evidence

1. **Fold and flip curves** (float64, minimally augmented systems).
   - On the seven fold curves we continued, all nine tangent reversals are located cusps: the oriented quadratic
     coefficient b sign a changes sign there, c != 0, and |det J| is between 14 and 2416.
   - Near cusps with c < 0, three same-period orbits coexist, two of them stable.
   - Frozen local cubic models matched the orbits recovered by Newton at 12 offsets, and their stabilities at 11.
2. **Gallas centre test, hostile period-7 body.**
   - The superstable curves of one continued orbit approach each other and turn away.
   - On a finer grid (1155 cells) at the closest approach, |mu1| is about 1e-4 to 1e-3 and comparable to |mu2|.
     221 cells have a complex leading pair; the minimum modulus is 1.66e-4, at a complex pair.
   - The sign of a real leading multiplier never alternates in a 2x2 block of cells.
3. **Two-block planes.**
   - In 3 of 1584 sampled chaotic cells, band return maps have two or more turning points (one N-shaped in every
     phase).
   - Near one of them lies a period-5 window: body, legs, a period-10/20 border, and attractors of periods 4 and 5
     coexisting at grid scale (61 of 447 attracting cells).
   - Its continued orbit (608 cells): |mu1| >= 0.0139 everywhere. The leading pair is complex in 91 cells, with
     negative product in 41. Per-step median moduli are 0.76 and 0.44. The trace-zero curves do not cross.
4. **FM5 today: fold and flip curves of that period-5 orbit in (alpha1, alpha2).**
   - **Gates.** Every point has residual below 1e-10 and |g| below 1e-9. The critical multiplier stays real
     at +1 or -1 to 5e-8. No other multiplier exceeds 0.12 in modulus, so no codimension-two point lies on the
     traced curves.
   - **One fold curve** bounds the right edge of the body, its top, the right side of the lower leg and one side of
     the upper leg. Branch folds found along alpha1 and alpha2 lie on it to 1e-7.
   - **The junction with the upper leg.** Where the upper leg leaves the body, the fold curve's parameter tangent
     reverses twice, about 1.2e-3 apart in alpha2, and its two outer arms cross in projection. Two reversals and a
     self-crossing form a swallowtail-shaped section. The reversals are being located and checked as cusps today
     (b sign a = 0, c != 0, det J != 0).
   - **Flip curves** bound the body's upper-left and lower sides and the other side of each leg. Each leg therefore
     lies between the fold curve and a flip curve.
   - **Beyond the flips** (going left or down), the orbit restabilizes at a second flip (a doubling bubble) and
     then dies at a fold. The bottom one is on the same fold curve; the left one is a separate fold, also being
     traced today.
   - **DSS5's anomalies explained.**
     - The isolated repelling cell (mu1 = +1.65) lies 1.2e-5 from the fold curve: it is the saddle partner.
     - The failed Newton column runs straight into the reversal pair.
   - **Coexistence.** Coexistence with period 4 occurs where the orbit is still attracting between the left flip
     and the edge of the fixed start's period-5 region.
5. **Return maps next to the hostile body.** One interior extremum per phase at 16 bins, more at finer bins. Two
   phases show doubled strands in the PCA projection.

Paper wording now (before FM5):
- "On the tracked branches the windows therefore have no one-dimensional shrimp centre: near their superstable strips
  the orbit's linearization has two comparable leading directions."
- "They are shrimp-like in body, legs and internal cascade, with located cusps on the fold curves of the period-6 and
  period-7 windows."
- "Whether they are the shrimps of an effectively two-dimensional map, as in the Henon family, is open."
- The hostile body does have superstable arcs (the leading multiplier crosses zero). They do not cross; where they
  approach each other, the leading multiplier meets the second one.

## Questions (please be adversarial about construct validity)

1. **What will a skeptical referee, or a frontier-lab reader, hit first in the restructured paper?** Is the
   practical consequence (layout-dependent answers in low-precision continuous feedback) framed at the right
   strength? Is anything in the four contribution groups overclaimed, or underclaimed?
2. **Length.** The main text is 37 pages. What would you move to the appendices or cut to reach about 25 pages
   without losing what makes the paper matter to this audience?
3. **Shrimps.** Is our claim stated at the right strength?
   - What object replaces the doubly superstable centre in invertible two-dimensional and higher-dimensional maps
     (Henon-type shrimps; endoskeletons in the sense of Facanha, Oldeman and Glass; crossroad and spring areas;
     anything else)?
   - Is a fold curve with two cusps and a self-crossing at the junction of a leg with the body (FM5) the expected
     anatomy of a shrimp in a two-dimensional map? Or does it belong to another named organization, such as
     crossroad or spring areas, or swallowtails? What in our data would tell these apart?
   - Which single computation on our continued orbits would settle whether these windows are the shrimps of an
     effectively two-dimensional map, rather than something else? For example: the curves where the leading pair
     turns complex; the product mu1 mu2 as an effective Jacobian determinant; the multiplicity of coexisting
     period-5 orbits inside the swallowtail pocket.
   - Name primary references. We verify every one before use.
4. **The last experiment.** Which experiment, in two days with this budget, would most raise the paper's value to
   the audience? Or should we stop and polish?
