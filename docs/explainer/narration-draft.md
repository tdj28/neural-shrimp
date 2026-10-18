# Explainer narration: full script in reading order

Draft of 18 October 2026, for the explainer of *Chaos and bifurcations in continuous language-model feedback* (the Chaos article on arXiv; the full report is expected as its supplementary material but may not be public at release). Planning document only: nothing here has been synthesized. The production plan, shot list and accuracy checklist are in [plan.md](plan.md).

Conventions:

- Seconds are estimated at 150 words per minute, as requested. The narration voice of the existing clips (OpenAI `gpt-4o-mini-tts`, voice `sage`) ran at 128 to 150 words per minute in their timelines, so allow 5 to 15 % more; the measured durations from the first synthesis pass decide the cut (plan.md, section 5).
- Text in braces `{...}` is optional: drop it first if the assembled cut runs long. The braces are not spoken.
- Each paragraph lists its visual cue, its sources (A = the Chaos article, by section, figure or table; R = the full report, which may not be public at release; L = the research log), and TTS respellings. The TTS input must use the respellings, not the display text.
- A displayed or spoken "word the loop writes" always means the most probable next token; the loop feeds back the probability-weighted average embedding.

| Chapter | Starts near | Words (all) | Seconds (all) | Words (core) | Seconds (core) |
|---|---|---|---|---|---|
| C0 Cold open | 0:00 | 102 | 41 | 102 | 41 |
| C1 The loop | 0:43 | 119 | 48 | 107 | 43 |
| C2 Repeating text and chaos | 1:28 | 152 | 61 | 139 | 56 |
| C3 The attractor | 2:26 | 151 | 60 | 101 | 40 |
| C4 Why steering makes chaos | 3:15 | 148 | 59 | 130 | 52 |
| C5 Order inside chaos | 4:09 | 211 | 84 | 211 | 84 |
| C6 What we can prove | 5:37 | 78 | 31 | 78 | 31 |
| C7 Why it matters for inference | 6:10 | 159 | 64 | 151 | 60 |
| C8 What it means, and what is open | 7:13 | 119 | 48 | 119 | 48 |
| **Total narration** | | **1239** | **496** | **1138** | **455** |

Narration alone: 7.6 min for the core text and 8.3 min with every optional sentence, at 150 words per minute (8.2 and 9.0 min at 138, the slowest typical rate of the voice). Unnarrated beats add 36 s (the trace's spoken tokens, the title hold, the end card, and short holds where a figure lands), and the eight 0.5 s crossfades remove 4 s. Planned runtime: **8:07 for the default cut (core text)**, 8:48 with every optional sentence. The 'Starts near' column assumes the default cut at 150 words per minute.

## C0. Cold open (starts near 0:00; 102 core words, about 41 s; 102 with optional text)

**P0.1** (62 words, about 25 s). *Cue: Shot 0.1, thread-split clip (new clip N3a).*

> Here is a language model with half a billion parameters, feeding its own output back to itself. We ran the same loop twice, with the same prompt and settings, changing only the number of CPU threads, six or four, which only reorders the additions inside the computation. For twenty-seven steps, both runs write the same words. At step twenty-eight, they part ways.

Sources: A Sec. V ('Chaos writes text too': six and four threads, same text for 27 steps); R 'Chaos with periodic windows' (float64, same processor, after the 1000-step transient); receipts docs/chaos-texts/qwen0.5b-philosophy.json and -long.json (first difference at step 28).
TTS respelling: Say 'C P U'.

**P0.2** (40 words, about 16 s). *Cue: Shot 0.2, title card over the dimmed attractor tour.*

> Nothing is sampled: the loop is deterministic, and at this setting it is chaotic. Fed its own soft output, a language model becomes a dynamical system that can be mapped, partly predicted from its own derivatives, and in places proved.

Sources: A Sec. II (deterministic map); A Table I (this cell, alpha 0.0652, log10 T -0.156, lambda 0.10); A abstract (thesis).

## C1. The loop (starts near 0:43; 107 core words, about 43 s; 119 with optional text)

**P1.1** (44 words, about 18 s). *Cue: Shot 1.1, soft-token-loop diagram, staged reveal.*

> This is the core step of Soft Thinking. After a fixed chat prompt, the model computes its next-token distribution at temperature T, and instead of sampling, it reads back the probability-weighted average of all token embeddings, a soft token, in the same input slot.

Sources: A Sec. I and Sec. II (expected-embedding map of Soft Thinking; fixed prefix; one input slot).

**P1.2** (43 words, about 17 s). *Cue: Shot 1.2, equation card.*

> The prompt and window never change, so one step is one forward pass, and the state is one vector of 896 numbers: a deterministic map, moved by two knobs, the temperature and the strength of a steering vector added to the residual stream.

Sources: A Sec. II (autonomous map of R^896; d = 896; steering added to the residual stream at layer 12).
TTS respelling: 896 -> 'eight hundred ninety-six'.

**P1.3** (32 words, about 13 s). *Cue: Shot 1.2, lower third.*

> A word shown on screen is the most probable next token; the loop itself always feeds back the full average. {At zero temperature, it reduces to greedy decoding with a one-token memory.}

Sources: A Sec. V ('what it feeds back at each step ... is the probability-weighted average of all token embeddings, not that token'); A Sec. II and Sec. VIII (T -> 0: greedy decoding with a one-token memory); consult 16, item 6c.

## C2. Repeating text and chaos (starts near 1:28; 139 core words, about 56 s; 152 with optional text)

**P2.1** (70 words, about 28 s). *Cue: Shot 2.1, article Fig. 1 with a slow push-in.*

> This scan is a phase diagram: each pixel is one setting of steering strength and temperature. In gray regions the loop settles into a cycle and repeats a fixed text. In colored regions, two copies started a hair apart separate exponentially, at a rate given by the largest Lyapunov exponent, the average log-growth of a tiny difference per step. We call that chaotic, as a numerical label, not a proof.

Sources: A Fig. 1 and caption (gray: repeating loop; 4.4 % of cells gray without a period or a positive exponent, shown on screen); A Sec. IV 'Scans and chaos labels' (twin at offset 1e-6; chaotic: exponent above 1e-3 and no period); A abstract (finite-horizon numerical label).

**P2.2** (36 words, about 14 s). *Cue: Shot 2.2, article Fig. 2 (a and b side by side), then Table II text.*

> Colored by what the loop writes, the windows turn out to be texts: 'is it', repeated; a blank line and 'The existence'; and stripes in which each window adds one more pair, such as 'is it'.

Sources: A Sec. IV 'Chaos with periodic windows', Fig. 2b; A Sec. V and Table II (each stripe adds one 'is X' pair).

**P2.3** (46 words, about 18 s). *Cue: Shot 2.3, Yes/No twin clip (new clip N3b).*

> {Between the windows, chaotic loops reuse the same words in no fixed order.} Steered toward hostility, a loop and a copy started one part in ten billion away agree for forty-four steps; at step forty-five, where 'Yes' and 'No' are almost tied, the copy writes 'No'.

Sources: A Sec. V and Table I (hostility cell, lambda 0.35, texts split at step 45; splits occur 'where no token dominates'); 'almost tied' = 0.303 against 0.298: R 'Chaos with periodic windows' and receipt docs/chaos-texts/qwen0.5b-hostility.json [R-only number, shown on screen].

## C3. The attractor (starts near 2:26; 101 core words, about 40 s; 151 with optional text)

**P3.1** (59 words, about 24 s). *Cue: Shot 3.1, 3D tour of the chaotic attractor.*

> Now collect many trajectories at one chaotic setting and project their states onto three principal components. Most settle onto a numerically observed chaotic attractor, made of thin, folded filaments and colored by the most probable next token. Its Lyapunov dimension, an estimate from the exponents, is about 1.26: by that estimate, little more than a curve inside 896 dimensions.

Sources: A Sec. VI (attractor paragraph) and Fig. 11: 498 of 512 trajectories, 14 to a coexisting period-4 orbit; exponents 0.274, -1.052, -1.486, -1.856; Kaplan-Yorke estimate 1.26; no correlation scaling range; consult 16 (chaotic, not strange; Lyapunov-dimension estimate, not a measured fractal dimension).
TTS respelling: 1.26 -> 'one point two six'; 896 -> 'eight hundred ninety-six'.

**P3.2** (18 words, about 7 s). *Cue: End of shot 3.1, then shot 3.2 (trace with spoken tokens, about 7 s without narration).*

> Follow one trajectory, and the dot jumps: each step is a full forward pass, with nothing in between.

Sources: A Sec. II (one iteration of F per step); R 'What the attractors look like' (a step moves the state by 0.30, more than the 0.21 mean distance from the center); log 2026-10-18 20:05 UTC.

*[About 7 s: the trace plays with its own audio, each most probable token spoken as the dot lands. No narration.]*

**P3.3** (24 words, about 10 s). *Cue: Shot 3.3, periodic tour with the eight orbit states marked.*

> Slightly further along the steering axis, the same loop is periodic: almost every start pours into the same eight states, and the text repeats.

Sources: Log 2026-10-18 20:05 UTC (alpha 0.0734, log10 T -0.148, float32: 509 of 512 starts reach one period-8 orbit) [L-only; number shown on screen].

**P3.4** (24 words, about 10 s). *Cue: Shot 3.4, attractor sweep along one row, with captions.*

> {Along one row of the scan, the attractor changes shape: from a short loop, through a doubling cascade into chaos, and back to order.}

Sources: Log 2026-10-18 19:15 and 19:35 UTC (ATT2 and ATT3 sweeps, float64); A Fig. 2 and Sec. VI (the crescent's cascade); visual evidence in the clip.

**P3.5** (26 words, about 10 s). *Cue: Optional: article Fig. 11 (d) or a delay-reconstruction card.*

> {The geometry can even be recovered from outside: three delays of one output number, the top token's log-probability, already give a reconstruction with few false neighbors.}

Sources: A Sec. VI (3.4 % false nearest neighbors at three delays); R (a faithful reconstruction from the output alone, not a verified Takens embedding).

## C4. Why steering makes chaos (starts near 3:15; 130 core words, about 52 s; 148 with optional text)

**P4.1** (41 words, about 16 s). *Cue: Shot 4.1, math version of the loop diagram (tikz-loop).*

> The loop's Jacobian shows where chaos can occur. It factorizes exactly: one over T, times the covariance of token embeddings under the model's own next-token distribution, times the network's Jacobian. That covariance is large only when tokens with different embeddings compete.

Sources: A Sec. II, Eq. (3) (tied embeddings, as in Qwen2.5-0.5B; untied models: a cross-covariance), Fig. 3 (checked to 1.35e-13); A Sec. II 'Where steering and temperature enter' (large near ties, vanishes where one token dominates).

**P4.2** (18 words, about 7 s). *Cue: Shot 4.1, Hopfield box revealed; then shot 4.2 (four two-token cases).*

> {Without the network, the loop would be a modern Hopfield update, which can neither cycle nor become chaotic.}

Sources: A Sec. III 'A Hopfield network with a transformer inside' (tied embeddings).

**P4.3** (40 words, about 16 s). *Cue: Shot 4.3, article Fig. 5, panels a, b, c.*

> In the half-billion model, along recorded orbits, steering into chaos raises the readout factor by 2.58 nats per step, while the network factor barely moves; a logit bias alone, with the network untouched, makes 3.5 percent of the plane chaotic.

Sources: A Sec. III 'Steering makes tokens compete' and Fig. 5 (+2.58 [1.78, 3.59]; network -0.07 [-0.23, 0.12]; logit bias 241 of 6837 cells, 3.5 %).
TTS respelling: 2.58 -> 'two point five eight'; 3.5 -> 'three point five'.

**P4.4** (49 words, about 20 s). *Cue: Shot 4.4 (steering-rotation clip), then shot 4.5 (card).*

> In the smaller models we tested, random directions, or one neuron's output direction, are about as chaotic as concept directions; the direction selects which tokens compete, and so what the attractors write. In such feedback loops, steering strength is a bifurcation parameter: do not assume a smooth dose response.

Sources: A Sec. IV 'Generality' (0.5B: 7.6, 8.3, 12.5 % for random directions against 6.8 % for philosophy; one neuron 9.9 %; Llama-3.1-8B: 27 to 39 %; 'the direction selects the texts of the attractors'); A Sec. VIII 'Steering and predictability' (not tested for ordinary generation). On screen, R: rotation keeps the chaotic share at 13 to 24 % over 103 angles; at 70B the directions differ.

## C5. Order inside chaos (starts near 4:09; 211 core words, about 84 s; 211 with optional text)

**P5.1** (56 words, about 22 s). *Cue: Shot 5.1, primer clip: fold, then flip.*

> The windows are organized as in low-dimensional maps. A periodic loop attracts while its multipliers, the eigenvalues of its Jacobian over one cycle, stay below one in size. At a fold, a multiplier reaches plus one and the loop vanishes with an unstable twin; at a flip, a multiplier passes minus one and the period doubles.

Sources: A abstract and Sec. II 'Numerics' (multipliers: eigenvalues of the monodromy matrix, the ordered product of the step Jacobians); A Sec. I and Sec. V (fold, flip). The primer itself uses classical one-dimensional maps.

**P5.2** (35 words, about 14 s). *Cue: Shot 5.2, article Fig. 10 (Feigenbaum tree), nested zoom a, b, c.*

> In the crescent, a three-step loop doubles again and again into chaos, and the ratios of successive doubling intervals approach Feigenbaum's universal constant: our last is 4.6688, against 4.6692, at the precision limit of float64.

Sources: A abstract, Sec. VI and Fig. 10 (nine flips to the flip of the period-768 orbit; ratios 3.124 ... 4.6688; at the float64 floor; the early ratios depend on the cut).
TTS respelling: 4.6688 -> 'four point six six eight eight'; 4.6692 -> 'four point six six nine two'; float64 -> 'float sixty-four'.

**P5.3** (34 words, about 14 s). *Cue: Shot 5.3, article Fig. 8 (fold), panels a-b, then c-d, then Table III card.*

> Past a fold, the vanished loop's ghost delays trajectories; forecasts of that delay from the network's own derivatives, fixed in advance at four new folds, came within one cycle in 42 of 48 cases.

Sources: A abstract, Sec. V and Table III (one iterate of G = F^p, i.e. one cycle; the six misses, all at a period-6 fold, 3.1 to 5.2 % short; both all-forecast criteria failed, shown on screen).

**P5.4** (34 words, about 14 s). *Cue: Shot 5.4, quartic cusp sweep (classical model), then article Fig. 12a.*

> In planes of two parameters, windows take the shape that names this project: shrimps, periodic islands with a body and thin legs, outlined by fold and flip curves, here first in a classical map.

Sources: A Sec. VII 'What a shrimp is' and Fig. 12; the quartic sweep illustrates R Fig. 'cusp kinds' (a classical model, not the language model; labeled on screen). The project name is the repository name.

**P5.5** (52 words, about 21 s). *Cue: Shot 5.5, data-example slides (the period-5 shrimp and its junction).*

> In the model's own map, steered at two depths, a period-5 window followed cell by cell fills a body, legs and doubled shells within its continued fold and flip curves: a numerically verified shrimp. At one junction, its orbit counts at thirteen points fixed in advance matched a swallowtail's prediction at nine.

Sources: A Sec. VII 'A shrimp in a plane steered at two depths' and Fig. 13 (gratitude at layer 4, hostility at layer 16, log10 T -0.1; 9 of 13 points; numerically verified, nothing proved by interval arithmetic).
TTS respelling: period-5 -> 'period five'.

## C6. What we can prove (starts near 5:37; 78 core words, about 31 s; 78 with optional text)

**P6.1** (43 words, about 17 s). *Cue: Shot 6.1, interval-arithmetic card (new).*

> Some of this is proved. Interval arithmetic carries every number as a range guaranteed to contain the true value. Applied to the full model, it proves an attracting period-8 orbit, to our knowledge the first such proof for a full transformer language-model map.

Sources: A Sec. VI 'Computer-assisted proofs' (alpha 0.071674, log10 T -0.115; all 24 layers, the full 151,936-token readout, the 37-token prefix; boxes of radius 2^-640; Krawczyk-type inclusion); A Sec. I (priority statement, 'to our knowledge').
TTS respelling: period-8 -> 'period eight'.

**P6.2** (35 words, about 14 s). *Cue: Shot 6.2, coexistence card over the basin slice.*

> A second proof shows two attracting orbits coexisting at one setting, one writing 'is it', the other 'The existence': same model, same settings, two stable behaviors. Which one a given start reaches is not proved.

Sources: A Sec. VI (alpha 0.06468, log10 T -0.148; certified token loops; basin neighborhoods of radius 2^-681 and 2^-700; reachability of a start outside them, the fixed start included, not proved). Basin slice: R Fig. 'pair basins' (float32, numerical).

## C7. Why it matters for inference (starts near 6:10; 151 core words, about 60 s; 159 with optional text)

**P7.1** (39 words, about 16 s). *Cue: Shot 7.1, batch-difference schematic (labeled: schematic, not data).*

> {Now a consequence that needs no dynamics vocabulary.} GPU kernels order their sums by matrix shape, so the same request in another batch differs in its last digits. A periodic loop damps that difference; a chaotic loop amplifies it.

Sources: A Sec. IV 'Reproducibility' (reduction order depends on matrix shape; first difference about 6e-15).

**P7.2** (47 words, about 19 s). *Cue: Shot 7.2, batch-divergence clip from the measured runs (new clip N4).*

> Run twice with only the batch grouping changed, 190 of 192 chaotic float64 loops separated after a median of 102 steps, on the schedule their Lyapunov exponents set, while 182 of 192 periodic loops stayed together. With the matrix shapes held fixed, the runs were bitwise identical.

Sources: A Sec. IV 'Reproducibility' (separated = passed a relative distance of 1e-2; median ratio 0.91; fixed shapes bitwise identical in all 448 cells, a control fixed in advance).
TTS respelling: float64 -> 'float sixty-four'.

**P7.3** (51 words, about 20 s). *Cue: Shot 7.3, answers card (new chart N7).*

> In bfloat16, feedback carries such differences into answers even where the loop contracts. In tests fixed in advance, reasoning models that solve 86 to 90 percent of GSM8K problems changed 14.5 and 19.5 percent of their Soft Thinking answers with the batch layout alone. Ordinary sampled reasoning changed about as many.

Sources: A abstract and Sec. IV 'Answers of a task'; the last sentence rests on R only (discrete sampled thinking changed 10.2 and 19.9 % in the same two models; R abstract) [R-only: drop it if the full report is not public at release].
TTS respelling: bfloat16 -> 'b-float sixteen'; GSM8K -> 'G S M eight K'; 14.5 and 19.5 -> 'fourteen and a half and nineteen and a half'.

**P7.4** (22 words, about 9 s). *Cue: Shot 7.4, kernels card.*

> Batch-invariant kernels remove one source of difference; in a chaotic regime, any other, from hardware, software or precision, grows the same way.

Sources: A Sec. IV 'Reproducibility'.

## C8. What it means, and what is open (starts near 7:13; 119 core words, about 48 s; 119 with optional text)

**P8.1** (51 words, about 20 s). *Cue: Shot 8.1, lessons card with figure insets.*

> Three lessons for interpretability. Single-step summaries can mislead: every step of the proved period-8 orbit stretches some direction, yet the full cycle contracts. Text can hide the dynamics: in one cascade, 65 chaotic settings wrote an exactly repeating text. And what stays predictable are the attractors and the boundaries between them.

Sources: A Sec. VIII 'Steering and predictability' (largest singular value 1.2 to 21 per step; cycle norm 0.059 to 0.644; invariant sets and boundaries remain predictable); A Sec. VI 'Chaos behind a repeating text' (65 of 182 chaotic cells, 2000 steps).
TTS respelling: period-8 -> 'period eight'.

**P8.2** (58 words, about 23 s). *Cue: Shot 8.2, limits card; article Fig. 7 under the last line.*

> The limits: chaos here is a finite-time label; our bounds locate it only by measuring along an orbit, not by forecasting from the weights; the loop's regime did not predict the accuracy of ordinary sampled answers beyond first-token uncertainty; crossroad areas are still verified only in classical maps; and larger models do not simply leave the chaos behind.

Sources: A abstract (finite-horizon label); A Sec. III (bounds close roughly as 1/k; a measurement along the orbit, not a forecast from the weights); A Sec. IV end (no accuracy information beyond first-token uncertainty); crossroad search: log 2026-10-18 20:05 UTC and the data-example slides [L-only; on-screen definition of a crossroad area needed]; A Sec. IV 'Generality' and Fig. 7 (14B against 7B; steering drives both 70B-class models into chaos).

**P8.3** (10 words, about 4 s). *Cue: Shot 8.3, end card.*

> The paper, the code and the data are linked below.

Sources: End card: arXiv:XXXX.XXXXX and github.com/tdj28/neural-shrimp (public at announcement).

## Not to be said

These phrasings were ruled out by the consult of 18 October (docs/consults/2026-10-18-attractor-naming/) or by the evidence ladder. Check the final TTS text against this list.

- "strange attractor" (say "chaotic attractor" or "numerically observed chaotic attractor");
- "fractal dimension 1.26" (it is a Lyapunov-dimension estimate from the exponents);
- "proved chaotic" or "chaos is proved" (chaos is a finite-time numerical label; only three periodic orbits are proved);
- "return map" for the loop itself (it is a discrete-time map; s(n+1) against s(n) is a lag-one plot);
- "the model writes the word X" without the convention that a shown word is the most probable next token;
- "a verified Takens embedding" (the delay reconstruction is faithful by a false-neighbor test, not verified);
- "a crossroad area in the language model" (verified only in classical maps; the search is under way);
- "the fixed start provably reaches ..." (reachability is not proved; only tiny basin neighborhoods are certified);
- "chaos causes the bfloat16 answer changes" (in Qwen2.5-0.5B and 7B the float64 loop does not need to be chaotic for them; in two reasoning models the float64 loop grows slowly);
- "edge of chaos helps" (no advantage detected; full report);
- "Feigenbaum universality is proved here" (late finite-order ratios approach the constant numerically);
- "ordinary sampled generation is chaotic" (not claimed);
- "steering creates all the chaos" (temperature alone gives chaos in eleven of fourteen dense models);
- any number of parameters for future runs (for example "405B"): no plan of that kind is in the article, the report or the log.

