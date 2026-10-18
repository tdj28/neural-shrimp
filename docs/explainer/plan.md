# Explainer video plan: chaos and bifurcations in continuous language-model feedback

Planning document of 18 October 2026. Nothing has been rendered, synthesized or committed for it. The full narration
script, with word counts, cues, sources and TTS respellings, is [narration-draft.md](narration-draft.md); it is the
canonical text for synthesis.

**Primary source.** The arXiv paper is the Chaos article, `/data2/chaos-journal-neural-shrimp/main.tex` (title
*Chaos and bifurcations in continuous language-model feedback*; 13 figure environments built from 14 PNG files plus one
TikZ figure). The full report, `paper/llm/main.tex`, is very probably its supplementary material, but the article
itself says the report "will accompany the journal version and a later version of this preprint", so it may not be
public when the video goes out. Every claim below is tagged:

- **[A]** in the Chaos article (section, figure or table numbers as in its current build, `main.aux`);
- **[R]** only in the full report;
- **[L]** only in the research log (`docs/research-log.md`) or in a receipt.

The video is to be released when the arXiv listing goes live (Monday evening ET at the earliest) and links that paper
and the repository, `github.com/tdj28/neural-shrimp`, which goes public at announcement.

## 0. Summary

An 8-minute explainer for mechanistic-interpretability researchers, built from the existing demos (bifurcation primer,
quartic cusp sweep, data-example slides, attractor sweep, 3D tours, spoken traces, steering-rotation video), the
article's figures, four blog schematics, and the output of five small new tools (cards, figure shots, two text-split
clips, a batch-divergence clip, an answers chart), all run on the CPU from saved data. Version 1 needs no
GPU and no paid compute except the speech synthesis of about 8 minutes of narration. A list of nice-to-have additions
(section 4) can follow in a version 2.

| Ch. | Title | Default cut | Key message |
|---|---|---|---|
| C0 | Cold open | 0:00-0:43 | The same deterministic loop, run with six or four CPU threads, writes the same words for 27 steps and then different ones. |
| C1 | The loop | 0:43-1:28 | Feeding back the expected embedding with a fixed prompt turns inference into a deterministic map of one 896-dimensional vector, moved by temperature and steering. |
| C2 | Repeating text and chaos | 1:28-2:26 | A scan of steering against temperature shows windows that repeat a text, separated by chaos in which copies a hair apart diverge. |
| C3 | The attractor | 2:26-3:15 | The chaos has a shape: thin folded filaments, little more than a curve in 896 dimensions, next to periodic attractors of eight states. |
| C4 | Why steering makes chaos | 3:15-4:09 | The loop's Jacobian factorizes; steering creates chaos mainly by making next tokens compete in the softmax readout, whatever the direction means. |
| C5 | Order inside chaos | 4:09-5:37 | Folds, flips, a Feigenbaum cascade and a shrimp-shaped window organize the chaos, and derivatives of the model predicted parts of it in advance. |
| C6 | What we can prove | 5:37-6:10 | Interval arithmetic proves an attracting period-8 orbit and two coexisting attracting orbits of the full model. |
| C7 | Why it matters for inference | 6:10-7:13 | Batch layout changes the rounding; chaos amplifies it on the Lyapunov schedule, and in bfloat16 feedback carries it into answers. |
| C8 | What it means, and what is open | 7:13-8:07 | Single-step summaries and emitted text can hide the dynamics; chaos labels are finite-time; bounds are measurements, not forecasts. |

The default cut uses the core narration (1,138 words, 455 s at 150 words per minute) plus 36 s of unnarrated beats;
every optional sentence adds 41 s (8:48). The voice ran at 128 to 150 words per minute in the existing timelines, so
the first synthesis pass (section 5.3, step 4) decides the final timing; the trim list is in section 5.6.

## 1. Audience and narrative arc

**Audience.** Researchers who know transformers, the residual stream, logits, softmax and temperature, steering
vectors, Jacobians, activation space and inference nondeterminism (batch-invariant kernels), and who do not know
bifurcation theory, Lyapunov exponents, shrimps, cusps or interval arithmetic. Every dynamics term is defined in one
clause at first use, in the vocabulary they have: a Lyapunov exponent as the average log-growth per step of a tiny
difference, a multiplier as an eigenvalue of the Jacobian over one cycle, a fold and a flip as a multiplier reaching
+1 or passing -1.

**Hook.** Inference nondeterminism is familiar to this audience; that it can change *what a model writes* through the
model's own dynamics is not. The cold open shows the article's thread-count result with its actual tokens: the same
cell recorded with six and with four CPU threads, which only reorders floating-point sums, writes the same text for 27
steps and then a different one [A Sec. V]. No sampling, no prompt change. The thesis follows over the title.

**Spine.** Hook (C0); the loop as a map (C1); the phase diagram, its texts and its chaos (C2); what the chaos looks
like (C3); why steering makes it (C4); the order inside it, predicted and verified (C5); what is proved (C6); what it
does to inference (C7); lessons for interpretability, limits and links (C8). The thesis has three verbs, and the
video gives each its chapter: mapped (C2, C3), predicted from derivatives at selected transitions (C5), and in places
proved (C6); C7 carries the clause on inference.

**Register.** Plain, precise journal register in American English; no rhetorical questions; no insider shorthand
("the shrimp literature"); every number from the article where possible, with [R] and [L] numbers kept on screen and
flagged in section 6. The colors in every attractor clip mark the most probable next token, and the narration says
once (P1.3) that the loop feeds back the probability-weighted embedding, not a word.

**Not in the video.** No numbers for future runs (the "405B only after the theory" idea of the brief is not in the
article, the report or the log, so nothing is said about it); no edge-of-chaos claim beyond "none detected" [R]; no
size claim beyond the article's (14B against 7B on eight held-out prompts; steering drives both 70B-class models into
chaos).

## 2. Chapters

Shot durations are planned values; section 5.3 sets them from the measured narration. "Plate" means a white-background
source (paper figure, 1280x720 clip, blog SVG) placed on the dark frame (section 5.1). Box coordinates are fractions of
the source image, `[x0, y0, x1, y1]` from the top left; check each with a still before rendering. The narration of
each chapter is quoted from narration-draft.md (braces mark optional text).

### C0. Cold open (0:00-0:43)

**Key message.** A deterministic language-model loop is chaotic: reordering its additions changes the words it writes
after 27 steps.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 0.1 | **New N3a**, thread-split clip (`tools/render_text_split.py --mode pair`, section 3.3): two token tickers, "6 CPU threads" and "4 CPU threads", steps 1 to 40; at step 28 the differing tokens turn red and the clip holds 3 s. | 27 s | Title: "The same loop, run twice". At step 28: "step 28: different words". Footer: "Qwen2.5-0.5B-Instruct · soft-token loop · philosophy steering α = 0.0652, log₁₀T = −0.156 · float64 · recorded after 1,000 warm-up steps · shown: the most probable next token". |
| 0.2 | Title over the dimmed 3D tour: `attractor-tour-f64-3x.mp4` 40.0-56.0 s, brightness −0.12, saturation 0.8, with a transparent title PNG from `render_cards.py`; fade in 0.6 s. | 16 s | "Chaos and bifurcations in continuous language-model feedback" · "Timothy D. Jones · Praxagent" · "An explainer of arXiv:XXXX.XXXXX". |

**Narration.**
P0.1: "Here is a language model with half a billion parameters, feeding its own output back to itself. We ran the same
loop twice, with the same prompt and settings, changing only the number of CPU threads, six or four, which only
reorders the additions inside the computation. For twenty-seven steps, both runs write the same words. At step
twenty-eight, they part ways."
P0.2: "Nothing is sampled: the loop is deterministic, and at this setting it is chaotic. Fed its own soft output, a
language model becomes a dynamical system that can be mapped, partly predicted from its own derivatives, and in places
proved."

**Evidence.** A Sec. V ("Run with six and with four threads, which only reorders floating-point sums, the cell next to
the period-3 crescent wrote the same text for 27 steps"); A Table I (this cell: α = 0.0652, log₁₀T = −0.156,
λ = 0.10); A abstract (thesis). Float64, same processor and the 1000-step warm-up: [R]. Tokens and step 28: receipts
`docs/chaos-texts/qwen0.5b-philosophy.json` (six threads) and `-long.json` (four threads), checked on 18 October:
first difference at step 28, " question" 0.188 against " existence" 0.186 in one run and " existence" 0.197 against
" question" 0.171 in the other (on screen only).

### C1. The loop (0:43-1:28)

**Key message.** With the prompt and window fixed, Soft Thinking's step is a deterministic map of one 896-dimensional
vector, with two knobs: temperature and steering strength.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 1.1 | Blog SVG `soft-token-loop.svg` (rendered at 3200 px), top panel only (crop y 0-0.56), plate; staged reveal with `render_figure_shot.py`: input box [0.03, 0.24, 0.19, 0.43] at 0 s; transformer with the steering plus [0.21, 0.24, 0.42, 0.53] at 2.5 s; logits and softmax [0.44, 0.24, 0.75, 0.53] at 5.5 s; expected embedding [0.76, 0.24, 0.98, 0.53] at 9 s; feedback arrow [0.08, 0.14, 0.90, 0.25] at 12 s. | 19 s | Tag "1 The loop". Caption: "One fixed prompt, one fed-back soft token at a fixed position: the step of Soft Thinking". |
| 1.2 | Equation card (`render_cards.py`), dark. | 24 s | "x₍ₙ₊₁₎ = F(xₙ),  F(x) = Eᵀ softmax(z_α(x)/T)" · "x ∈ ℝ⁸⁹⁶ (Qwen2.5-0.5B) · one step = one forward pass · prompt and window fixed" · "knobs: temperature T; steering strength α (a contrastive vector added to the residual stream at layer 12, at every position)". Lower third with P1.3: "Shown word = most probable next token · fed back = the probability-weighted average embedding". Optional line with P1.3's optional sentence: "T → 0: greedy decoding with a one-token memory". |

**Narration.** P1.1: "This is the core step of Soft Thinking. After a fixed chat prompt, the model computes its
next-token distribution at temperature T, and instead of sampling, it reads back the probability-weighted average of
all token embeddings, a soft token, in the same input slot." P1.2: "The prompt and window never change, so one step is
one forward pass, and the state is one vector of 896 numbers: a deterministic map, moved by two knobs, the temperature
and the strength of a steering vector added to the residual stream." P1.3: "A word shown on screen is the most
probable next token; the loop itself always feeds back the full average. {At zero temperature, it reduces to greedy
decoding with a one-token memory.}"

**Evidence.** A Sec. I and II (Eq. 1; d = 896; steering at layer 12, every position; T → 0 limit); A Sec. V (what is
fed back); consult 16, item 6c.

### C2. Repeating text and chaos (1:28-2:26)

**Key message.** In the steering-temperature plane, windows that repeat a fixed text alternate with chaos, where copies
started a hair apart diverge exponentially; the windows are texts.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 2.1 | **Article Fig. 1** (`fig-cross512-lyapunov.png`), plate; `render_figure_shot.py` push-in from the full figure (0-4 s) to [0.22, 0.18, 0.86, 0.66] (the crescent and the lower ends of the stripes), arriving at 24 s. | 30 s | Tag "2 Repeating text and chaos". Caption: "512 × 512 settings · Qwen2.5-0.5B-Instruct · philosophy steering at layer 12 · float64 · article Fig. 1". Lower third at 8 s: "Lyapunov exponent λ: average log-growth per step of a tiny difference between two copies". Lower third at 19 s: "chaotic = λ > 10⁻³ and no period found: a finite-time numerical label · 74 to 92 % of sampled labels persist to 10⁵ steps · gray includes 4.4 % of cells with neither a period nor a positive exponent". |
| 2.2 | **Article Fig. 2**, (a) `fig-stripes-lyapunov-curves.png` and (b) `fig-cross512-families.png` joined side by side (`magick a.png b.png -resize x1400 +append fig2-pair.png`), plate; (b) masked until 2 s; slow push 1.00 to 1.04. From 8 s, a monospace overlay of article Table II. | 14 s | Caption: "The same cells, colored by the text the loop writes · article Fig. 2". Overlay: "6   . This is a is a" / "8   . This is there is a is a" / "10  . This is there is it is a is a" / "12  . This is there (is it)² is a is a" with the header "period · repeating text (most probable tokens), article Table II". |
| 2.3 | **New N3b**, Yes/No twin clip (`render_text_split.py --mode twin`, section 3.3): loop and copy tickers, steps 1-56, with the log-distance panel; hold 3 s at step 45. | 18 s | Rows: "loop" / "copy, started 10⁻¹⁰ away". Panel: "log₁₀ distance between loop and copy". At step 45: "'Yes' 0.303 · 'No' 0.298: the copy writes 'No'" with a small "(probabilities: full report)". Footer: "hostility steering at layer 12 · α = 0.08946, log₁₀T = −0.05566 · λ = 0.35 · float64 · article Table I". |

**Narration.** P2.1: "This scan is a phase diagram: each pixel is one setting of steering strength and temperature. In
gray regions the loop settles into a cycle and repeats a fixed text. In colored regions, two copies started a hair
apart separate exponentially, at a rate given by the largest Lyapunov exponent, the average log-growth of a tiny
difference per step. We call that chaotic, as a numerical label, not a proof." P2.2: "Colored by what the loop
writes, the windows turn out to be texts: 'is it', repeated; a blank line and 'The existence'; and stripes in which
each window adds one more pair, such as 'is it'." P2.3: "{Between the windows, chaotic loops reuse the same words in
no fixed order.} Steered toward hostility, a loop and a copy started one part in ten billion away agree for forty-four
steps; at step forty-five, where 'Yes' and 'No' are almost tied, the copy writes 'No'."

**Evidence.** A Fig. 1 and caption; A Sec. I (30 % of Fig. 1 chaotic; 74 % persist) and Sec. IV (92 % for Fig. 2a;
label definition; twin at 10⁻⁶); A Fig. 2b, Sec. V and Table II (each stripe adds one pair); A Table I and Sec. V
(split at step 45 "where no token dominates"). The probabilities 0.303 and 0.298 are [R] and in the receipt
`docs/chaos-texts/qwen0.5b-hostility.json` (step index 44: "Yes" 0.303, "No" 0.298, copy "No"; log₁₀ distance −1.47).

### C3. The attractor (2:26-3:15; up to 3:25 with the sweep unnarrated)

**Key message.** A chaotic setting has a numerically observed attractor of thin folded filaments, little more than a
curve in 896 dimensions by its Lyapunov dimension, while a nearby setting pulls almost every start onto eight states.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 3.1 | `attractor-tour-f64-3x.mp4` (the brighter re-render now running; use it once its file time is later than 14:00), 0.0-29.0 s, full frame. | 29 s | Tag "3 The attractor" (top right). The clip's own title ("A chaotic attractor of a language model's feedback loop (numerically observed)"), legend and footer. Lower third at 14 s: "Lyapunov dimension ≈ 1.26: an estimate from the exponents 0.274, −1.052, −1.486, −1.856 per step, not a measured fractal dimension · article Fig. 11". |
| 3.2 | `attractor-trace-spoken-v2.mp4` (rendering now), from its first token minus 0.3 s (read `steps_detail[0].start` in the sidecar `.mp4.json`) for 8 s, **with its own audio** (spoken tokens). Do not use the old `attractor-trace-spoken.mp4`: its title says "strange attractor". | 8 s | The clip's own text. |
| 3.3 | Prefer `attractor-tour-p8-fade.mp4` if the attractor-video task finishes it (its narration script, `docs/narration/attractor-tour-p8.json`, was just changed to that source: the incoming streams fade as each trajectory's first 100 steps are dropped, leaving the eight states); otherwise `attractor-tour-p8.mp4` (eight orbit states marked, rendering now), 0.0-10.0 s. | 10 s | Lower third: "α = 0.0734: 509 of 512 starts reach one period-8 orbit, '. This is there is a is a' (float32; research log)". |
| 3.4 | `attractor-sweep-extended-12fps.mp4`, 2.0-31.6 s at 2.5x (11.8 s), fps to 30. | 12 s | Captions at source times: 2.0 "'is it' (period 2)"; 6.8 "period 3 → 6 → 12 → chaos"; 11.0 "the cascade runs backward"; 15.9 "windows and chaos alternate"; 23.6 "the strongest chaos"; 27.0 "a period-8 window". |
| 3.5 (opt.) | `paper/llm/figures/fig-delay-embedding.png` (new, uncommitted and not yet in either paper when checked; use it only once it is in the report), panels (a) then (c) then (d); otherwise a card: "delay reconstruction from log p_max alone: 70.8 → 10.4 → 3.4 % false nearest neighbors at 1, 2, 3 delays". | 8 s | Small: "a faithful reconstruction, not a verified Takens embedding". |

**Narration.** P3.1: "Now collect many trajectories at one chaotic setting and project their states onto three
principal components. Most settle onto a numerically observed chaotic attractor, made of thin, folded filaments and
colored by the most probable next token. Its Lyapunov dimension, an estimate from the exponents, is about 1.26: by that
estimate, little more than a curve inside 896 dimensions." P3.2 (last 7 s of shot 3.1): "Follow one trajectory, and the
dot jumps: each step is a full forward pass, with nothing in between." Shot 3.2 then plays about 7 s of spoken tokens
without narration. P3.3: "Slightly further along the steering axis, the same loop is periodic: almost every start pours
into the same eight states, and the text repeats." P3.4 (optional): "{Along one row of the scan, the attractor changes
shape: from a short loop, through a doubling cascade into chaos, and back to order.}" P3.5 (optional, with shot 3.5):
"{The geometry can even be recovered from outside: three delays of one output number, the top token's log-probability,
already give a reconstruction with few false neighbors.}"

**Evidence.** A Sec. VI (final paragraph) and Fig. 11 (498 of 512 trajectories; 14 reach a coexisting period-4 orbit;
exponents; Kaplan-Yorke estimate; no correlation scaling range; 3.4 % false nearest neighbors at three delays);
consult 16 (chaotic, not strange; Lyapunov-dimension estimate; projection on first mention). The jump explanation:
log 2026-10-18 20:05 UTC and [R] (a step moves 0.30, more than the 0.21 mean distance from the center). Periodic
census: [L] (log 20:05 UTC). Sweep: [L] (ATT2 and ATT3, log 19:15 and 19:35 UTC; reviewed narration
`docs/narration/attractor-sweep.json`).

### C4. Why steering makes chaos (3:15-4:09)

**Key message.** The loop's Jacobian is (1/T) × the covariance of token embeddings under the model's own distribution
× the network's Jacobian; steering creates chaos mainly by making next tokens compete, and the direction's meaning
mostly selects what the attractors write.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 4.1 | Blog SVG `tikz-loop.svg` (3200 px), plate; reveals: diagram (top 0-0.55) at 0 s; the two braces and labels (0.55-0.68) at 5 s; the equation DF(x) = (1/T) Σ_p J (0.68-0.80) at 9 s; the Hopfield box (0.82-0.97) at 14 s, only if P4.2 is kept. | 20 s | Tag "4 Why steering makes chaos". Caption: "Exact for tied embeddings (Qwen2.5-0.5B); untied models: a cross-covariance · checked against forward-mode derivatives to 1.35 × 10⁻¹³ (article Fig. 3)". |
| 4.2 | Blog SVG `spread-readout-gain.svg`, plate; reveal cases (a) to (d) at 0, 1.5, 3, 4.5 s, the scale at 6 s. | 8 s | The SVG's own label "Toy numbers". Caption: "The readout stretches a push only when tokens compete". |
| 4.3 | **Article Fig. 5** (`fig-mechanism-summary.png`), plate; full view, panels (b) [0.335, 0, 0.665, 1] and (c) [0.665, 0, 1, 1] masked until 6 and 11 s. | 16 s | Caption: "Qwen2.5-0.5B-Instruct (7B where marked) · float64 · article Fig. 5". Key numbers beside each panel as it appears: "(a) readout +2.58 nats/step [1.78, 3.59]; network −0.07 [−0.23, 0.12]"; "(b) logit bias only: 3.5 % chaotic (241 of 6837 cells); residual steering 6.2 %"; "(c) steer only the readout: +0.81; only the network: −0.49 nats/step". |
| 4.4 | `morph-steering-rotation.mp4`, the whole 19.9 s at 1.5x (13.3 s), plate at 1728x972. | 13 s | Caption: "A random steering direction rotated at constant norm: the chaotic regions move; the chaotic share stays at 13 to 24 % over 103 angles (float32, short-horizon labels; full report)". Second line: "0.5B: random directions 7.6, 8.3, 12.5 % chaotic; philosophy 6.8 %; one neuron 9.9 % · Llama-3.1-8B: 27 to 39 % (article Sec. IV)". |
| 4.5 | Card (`render_cards.py`). | 7 s | "In feedback loops, steering strength is a bifurcation parameter" · "Do not assume a smooth dose response" · "Two vectors at two depths: 78 % of the chaotic cells lie where neither push alone is chaotic (article Sec. VII)" · small: "not tested for ordinary generation". |

**Narration.** P4.1: "The loop's Jacobian shows where chaos can occur. It factorizes exactly: one over T, times the
covariance of token embeddings under the model's own next-token distribution, times the network's Jacobian. That
covariance is large only when tokens with different embeddings compete." P4.2 (optional): "{Without the network, the
loop would be a modern Hopfield update, which can neither cycle nor become chaotic.}" P4.3: "In the half-billion
model, along recorded orbits, steering into chaos raises the readout factor by 2.58 nats per step, while the network
factor barely moves; a logit bias alone, with the network untouched, makes 3.5 percent of the plane chaotic." P4.4:
"In the smaller models we tested, random directions, or one neuron's output direction, are about as chaotic as
concept directions; the direction selects which tokens compete, and so what the attractors write. In such feedback
loops, steering strength is a bifurcation parameter: do not assume a smooth dose response."

**Evidence.** A Sec. II Eq. (3) and Fig. 3; A Sec. III ("A Hopfield network with a transformer inside"; "Steering
makes tokens compete"; Fig. 5); A Sec. IV "Generality"; A Sec. VII (two depths, 21.7 % chaotic, 78 % where neither push
alone is); A Sec. VIII "Steering and predictability". Rotation share: [R] (Fig. "morph"). At 70B the directions differ
[R]; the narration's "in the smaller models we tested" covers this.

### C5. Order inside chaos (4:09-5:37)

**Key message.** The windows are organized as in low-dimensional maps (folds, flips, a period-doubling cascade with
Feigenbaum ratios, shrimp-shaped windows), and normal forms computed from the network's derivatives predicted passage
times in advance.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 5.1 | `bifurcation-primer.mp4` 12.2-36.3 s (the fold, then the flip) at 1.25x, plate at 1600x900. | 19 s | Tag "5 Order inside chaos". Caption: "Classical one-dimensional maps (illustration)". Lower third: "multipliers: eigenvalues of the Jacobian over one cycle (the monodromy matrix)". |
| 5.2 | **Article Fig. 10** (`fig-feigenbaum-tree.png`), plate; Ken Burns from panel (a) [0, 0, 1, 0.29] (0-4.5 s) to (b) [0, 0.33, 1, 0.62] (5.5-9.5 s) to (c) [0, 0.66, 1, 1] (10.5-15 s). | 15 s | Caption: "Qwen2.5-0.5B · the crescent's cascade at log₁₀T = −0.156 · float64 · article Fig. 10". Ticker: "3.124 · 4.4988 · 4.6208 · 4.6618 · 4.6673 · 4.6688 → δ = 4.6692…". Small: "finite-order ratios, the last at the float64 floor; the early ratios depend on the cut". |
| 5.3 | **Article Fig. 8** (`fig-fold.png`), plate; top row [0, 0, 1, 0.48] (0-6 s), bottom row [0, 0.52, 1, 1] (7-12 s); then a card over the dimmed figure (12-16 s). | 16 s | Caption: "The fold of the 2-cycle 'is it': passages of 144, 71, 34 and 14 iterates, as predicted · article Fig. 8". Card: "Forecasts fixed in advance at four new folds (article Table III): 42 of 48 within one iterate · the 6 misses, all at a period-6 fold, 3.1 to 5.2 % short · both all-forecast criteria failed". |
| 5.4 | `cusp-sweep-quartic-48s.mp4` 8.7-31.0 s at 2x (11.2 s), plate; then **article Fig. 12** panel (a) [0, 0, 0.5, 0.93] (6 s). | 17 s | On the quartic: "Classical model x ↦ c + (1+a)x + bx² + x⁴, not the language model". On Fig. 12a: "body (main cell) · legs · kite · cusp C · Gallas's canonical cell, classical · article Fig. 12"; definition: "cusp: where two fold curves meet in a sharp tip". |
| 5.5 | `cusp-data-examples.mp4` 0.0-20.0 s (slides 1 and 2), plate. | 20 s | The slides' own titles. Caption: "Qwen2.5-0.5B · gratitude at layer 4 × hostility at layer 16 · log₁₀T = −0.1 · float64 · article Fig. 13". Definition: "swallowtail: the fold set of a quartic equation; it predicts how many orbits each region holds". |

**Narration.** P5.1: "The windows are organized as in low-dimensional maps. A periodic loop attracts while its
multipliers, the eigenvalues of its Jacobian over one cycle, stay below one in size. At a fold, a multiplier reaches
plus one and the loop vanishes with an unstable twin; at a flip, a multiplier passes minus one and the period doubles."
P5.2: "In the crescent, a three-step loop doubles again and again into chaos, and the ratios of successive doubling
intervals approach Feigenbaum's universal constant: our last is 4.6688, against 4.6692, at the precision limit of
float64." P5.3: "Past a fold, the vanished loop's ghost delays trajectories; forecasts of that delay from the
network's own derivatives, fixed in advance at four new folds, came within one cycle in 42 of 48 cases." P5.4: "In
planes of two parameters, windows take the shape that names this project: shrimps, periodic islands with a body and
thin legs, outlined by fold and flip curves, here first in a classical map." P5.5: "In the model's own map, steered at
two depths, a period-5 window followed cell by cell fills a body, legs and doubled shells within its continued fold and
flip curves: a numerically verified shrimp. At one junction, its orbit counts at thirteen points fixed in advance
matched a swallowtail's prediction at nine."

**Evidence.** A abstract; A Sec. II "Numerics" (multipliers of the monodromy); A Sec. V, Fig. 8, Table III (one
iterate of G = F^p is one cycle of the vanished orbit); A Sec. VI and Fig. 10; A Sec. VII, Figs. 12 and 13. The quartic
sweep illustrates [R] Fig. "cusp kinds". "Names this project": the repository name, not a scientific claim.

### C6. What we can prove (5:37-6:10)

**Key message.** Interval arithmetic proves an attracting period-8 orbit of the full model and, at one setting, two
coexisting attracting orbits; it does not prove which one a given start reaches.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 6.1 | **New** interval card in three stages (`render_cards.py`, section 3.1). | 18 s | Stage 1: "Floating point: one rounded number per quantity · Interval arithmetic: a range [lo, hi] that is guaranteed to contain the true value, carried through every operation". Stage 2: "8 boxes of radius 2⁻⁶⁴⁰ in all 896 coordinates, around the 8 states of the orbit '. This is there is a is a'". Stage 3: "Krawczyk test: exactly one orbit in the boxes, and it attracts · all 24 layers, the full 151,936-token readout, the 37-token prompt · α = 0.071674, log₁₀T = −0.115 · article Sec. VI". Tag "6 What we can prove". |
| 6.2 | **New** coexistence card (transparent) over `fig-pair-basins.png` panel (a), dimmed to 35 %. | 15 s | "period 2: ' is it'   ·   period 3: '⏎⏎', 'The', ' existence'" · "both proved attracting at α = 0.06468, log₁₀T = −0.148, token loops certified · certified basin neighborhoods: radius 2⁻⁶⁸¹ and 2⁻⁷⁰⁰" · "which orbit a given start reaches: not proved" · small: "background: numerical basin slice, float32 (full report)". |

**Narration.** P6.1: "Some of this is proved. Interval arithmetic carries every number as a range guaranteed to contain
the true value. Applied to the full model, it proves an attracting period-8 orbit, to our knowledge the first such
proof for a full transformer language-model map." P6.2: "A second proof shows two attracting orbits coexisting at one
setting, one writing 'is it', the other 'The existence': same model, same settings, two stable behaviors. Which one a
given start reaches is not proved."

**Evidence.** A Sec. VI "Computer-assisted proofs" (both proofs, token loops, basin radii, reachability not proved);
A Sec. I ("we know of no earlier computer-assisted proof of a periodic orbit, or of coexisting attractors, of a full
transformer language-model map"). The orbit's text on screen is its leading-token loop from A Table II. Basin slice:
[R] Fig. "pair basins".

### C7. Why it matters for inference (6:10-7:13)

**Key message.** The batch changes only the order of floating-point sums; a periodic loop damps the difference, a
chaotic loop amplifies it on its Lyapunov schedule, and in bfloat16 feedback carries it into answers even where the
loop contracts.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 7.1 | Blog SVG `batch-difference-schematic.svg`, plate; reveal: axes and the periodic line (0 s), the chaotic line (3 s), "the texts split" (6 s), the bfloat16 dashed line (9 s), the bottom box (12 s). | 15 s | Tag "7 Why it matters for inference". The SVG's own "Schematic, not data". |
| 7.2 | **New N4**, batch-divergence clip from the measured runs (`tools/render_batch_divergence.py`, section 3.4). | 20 s | As specified in section 3.4. |
| 7.3 | **New N5**, answers chart (`tools/plot_explainer_answers.py`, section 3.5), with a slow reveal of its three groups. | 20 s | Title: "Answers changed by the batch layout alone (tests fixed in advance; bfloat16)". Footnote: "reset of the fed-back state before every step: 66 → 1.6 % (0.5B) and 64 → 1.6 % (7B), exploratory · discrete sampled thinking: 10.2 and 19.9 % (full report)". |
| 7.4 | Card. | 9 s | "Batch-invariant kernels remove one source of difference." · "In a chaotic regime, any other (hardware, software version, precision) grows the same way." · small: "article Sec. IV". |

**Narration.** P7.1: "{Now a consequence that needs no dynamics vocabulary.} GPU kernels order their sums by matrix
shape, so the same request in another batch differs in its last digits. A periodic loop damps that difference; a
chaotic loop amplifies it." P7.2: "Run twice with only the batch grouping changed, 190 of 192 chaotic float64 loops
separated after a median of 102 steps, on the schedule their Lyapunov exponents set, while 182 of 192 periodic loops
stayed together. With the matrix shapes held fixed, the runs were bitwise identical." P7.3: "In bfloat16, feedback
carries such differences into answers even where the loop contracts. In tests fixed in advance, reasoning models that
solve 86 to 90 percent of GSM8K problems changed 14.5 and 19.5 percent of their Soft Thinking answers with the batch
layout alone. Ordinary sampled reasoning changed about as many." P7.4: "Batch-invariant kernels remove one source of
difference; in a chaotic regime, any other, from hardware, software or precision, grows the same way."

**Evidence.** A Sec. IV "Reproducibility" and "Answers of a task"; A abstract (reasoning models, 86 to 90 %, 14.5 and
19.5 %; QwQ-32B and DeepSeek-R1-Distill-Llama-8B grow slowly in float64). "Ordinary sampled reasoning changed about as
many" is [R] only (R abstract; `docs/pi/pm2-summary.json`: plain arms 10.2 and 19.9 %). The gate numbers of shot 7.2
were recomputed from the raw arrays on 18 October (section 3.4).

### C8. What it means, and what is open (7:13-8:07)

**Key message.** For interpretability: summaries of single steps and the emitted text can both hide the dynamics, and
what stays predictable are attractors and their boundaries; the limits are stated plainly.

| Shot | Visual | Dur. | On-screen text |
|---|---|---|---|
| 8.1 | Lessons card, three bullets revealed with the narration; inset 1 (text): "every step: largest singular value 1.2 to 21 · the whole cycle: norm 0.059 to 0.644 (article Sec. VIII)"; inset 2: `fig-hidden-bands.png` panels (a) [0, 0, 0.34, 1] and (c) [0.66, 0, 1, 1]. | 21 s | Title: "Three lessons for interpretability". Bullets: "Single-step summaries can mislead" · "Text can hide the dynamics: 65 of 182 chaotic settings on one line wrote an exactly repeating text for 2000 steps (article Sec. VI)" · "What stays predictable: the attractors and the boundaries between them". Small under inset 2: "full report". Tag "8 What it means, and what is open". |
| 8.2 | Limits card, five bullets; under the second, an optional 4 s inset of `fig-bound-hierarchy.png` panel (a) [0, 0, 0.34, 1] [R]; under the fourth, a 5 s cutaway to `cusp-data-examples.mp4` 20.3-28.0 s (slide 3: "What is verified, and what is open"); under the fifth, **article Fig. 7** (`fig-zoom-llama70b-z128.png`) slides in. | 24 s | Bullets: "chaos: a finite-time numerical label" · "bounds: they close on the exponent roughly as 1/k, and over windows of 17 to 29 steps their sign matches the long-run exponent in 23 of 24 cells; a measurement along the orbit, not a forecast from the weights (article Sec. III)" · "the loop's regime: no accuracy information beyond first-token uncertainty" · "crossroad areas (a cusp whose two flip curves cross, where two attracting orbits coexist): verified only in classical maps; search under way" · "larger models: Qwen2.5-14B more chaotic than 7B on 8 test prompts; Llama-3.3-70B steered: 20.8 % of cells chaotic (article Fig. 7)". |
| 8.3 | End card (`render_cards.py`), 8 s with a 1 s fade to black. | 8 s | "Chaos and bifurcations in continuous language-model feedback" · "arXiv:XXXX.XXXXX" · "Code, frozen test designs and records: github.com/tdj28/neural-shrimp" · "Full report: supplementary material" · "Also in the article: coupled copies of the loop synchronize; a memory of 8 or 16 soft tokens is more chaotic than one (Qwen2.5-7B); Huginn: a predicted Neimark-Sacker bifurcation confirmed in a frozen test" · "Narration: synthetic voice". |

**Narration.** P8.1: "Three lessons for interpretability. Single-step summaries can mislead: every step of the proved
period-8 orbit stretches some direction, yet the full cycle contracts. Text can hide the dynamics: in one cascade, 65
chaotic settings wrote an exactly repeating text. And what stays predictable are the attractors and the boundaries
between them." P8.2: "The limits: chaos here is a finite-time label; our bounds locate it only by measuring along an
orbit, not by forecasting from the weights; the loop's regime did not predict the accuracy of ordinary sampled answers
beyond first-token uncertainty; crossroad areas are still verified only in classical maps; and larger models do not
simply leave the chaos behind." P8.3: "The paper, the code and the data are linked below."

**Evidence.** A Sec. VIII ("Steering and predictability"; "Beyond the one-token loop"; "Coupled copies"); A Sec. VI
("Chaos behind a repeating text"); A Sec. III (bounds: a measurement along the orbit, not a forecast from the
weights); A Sec. IV (accuracy; Generality; Fig. 7). Crossroad search: [L] (log 2026-10-18 20:05 UTC; the data-example
slide 3). Hidden-bands figure: [R].

## 3. New clips and assets for version 1

All new pieces run on the CPU from files already on disk, with `uv run --no-sync python` (project convention). Each
new tool refuses to render when a gate fails; none falls back to a default value. Fonts and colors follow section 5.1.

### 3.0 Preflight: the renders now running

Four renders started at about 13:41 (they belong to the attractor-video task; do not stop them):
`attractor-tour-f64.mp4` (900 frames, brighter tone mapping; the 90 s version used here, `attractor-tour-f64-3x.mp4`,
is a 2700-frame render of the same tour; if it has not been remade after this render, make it with
`uv run --no-sync python tools/render_attractor_tour.py --run llm-soft-loop/pod64/portraits/f64-strong --frames 2700
--out ../neural-shrimp-data/videos/attractor-tour-f64-3x.mp4`, CPU),
`attractor-tour-p8.mp4` (orbit states marked; a `-fade` version is planned), `attractor-trace-spoken-v2.mp4` and
`attractor-trace-p8-spoken.mp4`
(both with spoken intros and a `.json` sidecar). Before cutting, check that each finished (file time after 14:00, a
`.json` sidecar for the traces), extract a frame at 1 s and confirm the titles: no "strange attractor", American
spelling, "color: the most probable next token". Fallbacks: the existing `attractor-tour-f64-3x.mp4` and
`attractor-tour-p8.mp4` (13:37); if the trace v2 fails, replace shot 3.2 by panels (a)-(c) of article Fig. 11 with a
push-in (8 s).

### 3.1 `tools/render_cards.py` (new; matplotlib; about 100 lines)

Renders title, equation, interval, coexistence, steering, kernels, lessons, limits and end cards, and the transparent
title overlay, from a JSON list. Each card: 1920x1080, background `#0B0B0F` (or transparent), title in DejaVu Sans Bold
52 px white, body in DejaVu Sans 32 px `#C8C8D0`, gold bullets `#FFD93F`, math by mathtext, token strings in DejaVu
Sans Mono, footnotes 24 px `#9A9AA6`. A card may have stages; each stage is written as its own PNG
(`card-name-stageK.png`), and `render_figure_shot.py` (or an ffmpeg crossfade) turns the stages into a shot with the
reveal times of section 2. Card texts are those of the shot tables, verbatim.

### 3.2 `tools/render_figure_shot.py` (new; PIL and ffmpeg; about 120 lines)

The workhorse for figures and SVGs. Inputs: `--image` (PNG; SVGs first rendered with
`inkscape X.svg --export-type=png --export-width=3200`), `--seconds`, `--keys` (JSON list of `{"t": s, "box": [x0, y0,
x1, y1]}` in fractions of the uncropped image, as are all boxes in this plan; each box expanded to the plate's 16:9 aspect about its center, padded with the plate
color where it leaves the image; smoothstep easing; hold after the last key), `--reveal` (JSON list of
`{"t": s, "box": [...]}`: the region is covered by the plate color until t, then fades in over 0.4 s), `--crop`
(restrict the source first), `--plate 1760x990` on `--bg 0b0b0f` with `--plate-bg white` (or the SVG's own beige),
`--caption` (bottom strip), `--lower-third` (JSON list of `{"t", "until", "text"}`), `--tag` (chapter tag, first 3.5 s,
top right), `--fps 30`, `--still T` (write one PNG at time T for checking), `--out`. Implementation: load the image once
at full resolution, compute a float crop box per frame, resample with `Image.transform(..., EXTENT, ...,
BICUBIC)` for subpixel-smooth motion, composite, draw text with PIL and the DejaVu fonts of section 5.1, and pipe RGB
frames to `ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - -c:v libx264 -crf 16 -preset slow -pix_fmt
yuv420p`. Example (shot 2.1):

```bash
uv run --no-sync python tools/render_figure_shot.py \
  --image ../chaos-journal-neural-shrimp/figures/fig-cross512-lyapunov.png --seconds 30 \
  --keys '[{"t":0,"box":[0,0,1,1]},{"t":4,"box":[0,0,1,1]},{"t":24,"box":[0.22,0.18,0.86,0.66]}]' \
  --caption "512 × 512 settings · Qwen2.5-0.5B-Instruct · philosophy steering at layer 12 · float64 · article Fig. 1" \
  --lower-third '[{"t":8,"until":18,"text":"Lyapunov exponent λ: average log-growth per step of a tiny difference between two copies"}]' \
  --tag "2  Repeating text and chaos" --out $W/shots/C2-01-fig1.mp4
```

Use the report's copy of `fig-real-shrimp.png` (11:50) for any new shot: the article's copy (03:00) predates the
lower-left leg's own curves that the article's text describes (appendix C).

### 3.3 `tools/render_text_split.py` (new; matplotlib; about 200 lines): clips N3a and N3b

Dark frame; two monospace tickers (DejaVu Sans Mono 34 px) showing the last 28 tokens of each run ("⏎" for line
breaks), the current token in white, tokens after the first difference in red `#FF6B4D`; a step counter; a small panel
with the top-two probabilities of the current step (top two only, which also keeps low-probability profanity of the
hostility cell off screen).

- **N3a, `--mode pair`**: run A = `docs/chaos-texts/qwen0.5b-philosophy.json` (six threads, cell 0, 60 steps), run B =
  `docs/chaos-texts/qwen0.5b-philosophy-long.json` (four threads, cell 0, 500 steps); both at α = 0.0652,
  log₁₀T = −0.156. Tokens: `steps[k].top[0].text` of each file. Steps 1-40; 0.55 s per step up to 27, a 3 s hold at 28,
  0.35 s per step after; about 27 s with a 2 s title hold. Gate: the first differing step must be 28 (checked on 18
  October: " question" 0.1884 against " existence" 0.1860 in A, " existence" 0.1968 against " question" 0.1714 in B).
- **N3b, `--mode twin`**: `docs/chaos-texts/qwen0.5b-hostility.json`, cell 0: loop tokens `steps[k].top[0].text`, copy
  tokens `steps[k].twin`, distance `steps[k].log10_distance`; steps 1-56; a lower panel draws log₁₀ distance against
  the step with the dotted line −10 + 0.352·t/ln 10 (λ = 0.352, `lyapunov[0]`). Gates: first difference at step 45
  (index 44), loop "Yes" 0.303 and "No" 0.298 there, copy "No"; refuse otherwise. About 18 s.

### 3.4 `tools/render_batch_divergence.py` (new; matplotlib; about 160 lines): clip N4

Real data of the article's reproducibility test. Inputs: `../neural-shrimp-data/explore/llm-soft-loop/pod64/repro/
h2cells/repro.npz` (`batch64_dist`, `batch64_agree`: 1500 steps × 448 cells), `docs/h2-horizon-cells.json` (`label`,
`map_exponent`), `.../repro/fs/repro.npz` (`repeat_dist`, `fixedperm_dist`), and `docs/repro-summary.json` for the
gates.

- **Gates** (recomputed on 18 October from these files, all exact): chaotic cells passing 10⁻² = 190 of 192, median
  first step 102; periodic cells staying below 10⁻² = 182 of 192; `repeat_dist` and `fixedperm_dist` maximum = 0.0 in
  all 448 cells. Refuse to render if any differs.
- **Cells shown**: the chaotic cell with 0.25 ≤ λ ≤ 0.40 whose crossing step lies in 70-140 and whose ratio of measured
  to predicted step, ln(10⁻²/d₁)/λ with d₁ its smallest nonzero distance, is closest to the median ratio 0.91; the
  periodic cell whose largest distance is closest to the median over the 182 periodic cells that stay below 10⁻²
  (2.30 × 10⁻¹², the report's "median largest distance"). Print the chosen indices, λ, d₁, crossing and predicted
  steps, and the first step with a different most probable token.
- **Timeline** (about 20 s): 0-8 s, the two curves drawn step by step over steps 0-300 on a log axis from 10⁻¹⁶ to 10,
  with the dashed 10⁻² line, the predicted crossing (dotted, labeled "from λ") and the first different token (dot); 8-14
  s, the x axis widens to 1500 steps and all 192 chaotic (red, alpha 0.08) and 192 periodic (blue, alpha 0.08) curves
  fade in, with the counters "190 of 192 chaotic loops pass 10⁻² (median step 102)" and "182 of 192 periodic loops stay
  below"; 14-20 s, a third row "same matrix shapes (rows repeated or permuted): distance exactly 0 in all 448 cells, 1500
  steps". Footer: "Qwen2.5-0.5B-Instruct · philosophy steering at layer 12 · float64 · only the batch grouping differs ·
  article Sec. IV".

Fallback if the tool is not ready: `fig-repro.png` panel (a) then `fig-fixed-shape.png` panel (a) [R], with
`render_figure_shot.py`.

### 3.5 `tools/plot_explainer_answers.py` (new; matplotlib; about 80 lines): chart N5

Load the `dataviz` skill before designing it. Horizontal bars of "answers changed by the batch layout alone", read
from the receipts (never typed in), each bar labeled with its value and its comparison arm:

| Group | Bar (receipt, field) | Value | Comparison bar | Value |
|---|---|---|---|---|
| Soft-thinking procedure, 96 additions, bfloat16 [A] | `docs/repro-summary.json`: `answers` → `qwen0.5b (REPRO-T)` → `conditions` → `bf16 soft 0.0` → `answers_differ` | 56 % | `bf16 greedy None` | 0 % |
| | `answers` → `qwen7b (REPRO-T7)` → `conditions` → `bf16 soft 0.2` | 55 % | `bf16 greedy None` | 10 % (10 of 96) |
| Soft Thinking, published settings, Qwen2.5-7B, 256 GSM8K [A] | `docs/pi/pm-summary.json`, `arms.top15-s64.flip_rate` | 21.1 % | `arms.replay` | 5.9 % |
| Reasoning models, GSM8K [A] | `docs/pi/pm2-summary.json`, `cells.A-gsm8k.soft` (DeepSeek-R1-Distill-Qwen-7B) | 14.5 % | `.plain` [R] | 10.2 % |
| | `cells.C-gsm8k.soft` (QwQ-32B) | 19.5 % | `.plain` [R] | 19.9 % |

Gate: each value must round to the number printed in the article (or the report, for the [R] bars). Dark theme to match
the cards; colors from section 5.1.

### 3.6 Existing clips: excerpts and normalization

Every shot is encoded the same way: 1920x1080, 30 fps, yuv420p, SAR 1, H.264 CRF 16, no audio except shot 3.2.

```bash
cd /data2/neural-shrimp
V=../neural-shrimp-data/videos; W=$V/explainer; BG=0x0b0b0f
mkdir -p $W/{assets,shots,chapters,final,tts-measure}
ENC="-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 -an"
PLATE="scale=1600:900:flags=lanczos,pad=1920:1080:160:110:color=$BG"

# dark, full-frame clips
ffmpeg -y -ss 0 -t 29 -i $V/attractor-tour-f64-3x.mp4 -vf "fps=30,setsar=1,format=yuv420p" $ENC $W/shots/C3-01-tour.mp4
ffmpeg -y -ss 0 -t 10 -i $V/attractor-tour-p8.mp4     -vf "fps=30,setsar=1,format=yuv420p" $ENC $W/shots/C3-03-p8.mp4
ffmpeg -y -ss 2.0 -t 29.6 -i $V/attractor-sweep-extended-12fps.mp4 \
       -vf "setpts=PTS/2.5,fps=30,setsar=1,format=yuv420p" $ENC $W/shots/C3-04-sweep.mp4
# the trace keeps its audio (spoken tokens)
T0=$(python3 -c "import json;print(round(json.load(open('$V/attractor-trace-spoken-v2.mp4.json'))['steps_detail'][0]['start']-0.3,2))")
ffmpeg -y -ss $T0 -t 8 -i $V/attractor-trace-spoken-v2.mp4 -vf "fps=30,setsar=1,format=yuv420p" \
       -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -r 30 \
       -af "aresample=48000,aformat=channel_layouts=stereo" -c:a aac -b:a 192k $W/shots/C3-02-trace.mp4

# white 1280x720 clips as plates
ffmpeg -y -ss 12.2 -t 24.1 -i $V/bifurcation-primer.mp4 -vf "setpts=PTS/1.25,fps=30,$PLATE,setsar=1,format=yuv420p" $ENC $W/shots/C5-01-primer.mp4
ffmpeg -y -ss 8.7 -t 22.3 -i $V/cusp-sweep-quartic-48s.mp4 -vf "setpts=PTS/2,fps=30,$PLATE,setsar=1,format=yuv420p" $ENC $W/shots/C5-04a-quartic.mp4
ffmpeg -y -ss 0 -t 20 -i $V/cusp-data-examples.mp4 -vf "fps=30,$PLATE,setsar=1,format=yuv420p" $ENC $W/shots/C5-05-shrimp.mp4
ffmpeg -y -ss 20.3 -t 7.7 -i $V/cusp-data-examples.mp4 -vf "fps=30,$PLATE,setsar=1,format=yuv420p" $ENC $W/shots/C8-02b-open.mp4
# white 1920x1080 clip, 24 fps, sped up 1.5x
ffmpeg -y -ss 0 -t 19.875 -i $V/morph-steering-rotation.mp4 \
       -vf "setpts=PTS/1.5,fps=30,scale=1728:972:flags=lanczos,pad=1920:1080:96:72:color=$BG,setsar=1,format=yuv420p" $ENC $W/shots/C4-04-morph.mp4
# title over the dimmed tour (title-overlay.png from render_cards.py, transparent)
ffmpeg -y -ss 40 -t 16 -i $V/attractor-tour-f64-3x.mp4 -loop 1 -i $W/assets/title-overlay.png -filter_complex \
  "[0:v]fps=30,eq=brightness=-0.12:saturation=0.8[b];[b][1:v]overlay=0:0:shortest=1,fade=t=in:st=0:d=0.6,setsar=1,format=yuv420p" \
  $ENC $W/shots/C0-02-title.mp4
```

Burned-in captions and chapter tags on these clips use `drawtext` with the DejaVu font files of section 5.1, for
example the tag on the first shot of a chapter:

```bash
FONTB=.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf
TAG="drawtext=fontfile=$FONTB:text='3  The attractor':x=w-tw-60:y=40:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=14:alpha='if(lt(t,0.4),t/0.4,if(lt(t,3.1),1,max(0,(3.5-t)/0.4)))'"
```

Captions with timed text (shot 3.4, the lower thirds of shots 3.1 and 3.3) use `drawtext=...:enable='between(t,A,B)'`
with times converted to the sped-up clip (divide source times by the speed factor).

## 4. Nice-to-have additions (version 2)

None of these is needed for the first release. GPU items run on the local RTX 2000 Ada (idle when checked during planning) and each needs a
run card in the research log before it runs (adaptive protocol).

1. **One token palette for all attractor clips.** The tours, traces and sweep assign colors by frequency per clip, so
   ' existence', '.', ' there' and ' question' change colors between clips (the first four tokens agree). Add a
   `--palette FILE` option to the three renderers and re-render (CPU).
2. **Native 1080p primer, cusp sweep and data slides.** Add `--dpi` (default 100) to `render_bifurcation_primer.py`,
   `render_cusp_sweep.py` and `render_data_examples.py` and render at 150 dpi; same timing, so the cuts of section 3.6
   stay valid (CPU, minutes).
3. **An animated loop diagram** (`render_loop_diagram.py`, matplotlib): the prompt strip, the transformer with the
   steering plus, a live bar chart of the next-token distribution from the saved top-3 probabilities of
   `docs/chaos-texts/qwen0.5b-philosophy-long.json` (for example ' existence' 0.203 against ' question' 0.156), a
   weighted vector sum forming the soft token, and the arrow back; replaces shot 1.1.
4. **Two attractors from one setting**: a portrait recorded from the first step at the certified pair cell (α = 0.06468,
   log₁₀T = −0.148), 512 starts, float32, with `attractor_portrait.py` (recording from step 0, as `p8-approach-f32`),
   then a tour showing streams pouring into the 2-cycle and the 3-cycle; `--mark-orbit` would need two orbits (small
   change). Local GPU, minutes. For C6.
5. **Banded chaos past the cascade**: `render_attractor_tour.py --run llm-soft-loop/pod64/portraits/f64-cascade --frames
   600 --title "Banded chaos past the cascade (numerically observed)"` (data on disk; CPU). For C5 after shot 5.2.
6. **Reconstruction from the output alone**: a tour of the three-delay reconstruction (`portraits/takens-strong`) beside
   the principal-axis projection; needs a small adapter that writes delay coordinates as portrait coordinates (CPU).
7. **Batch layout with tokens**: rerun one chaotic cell twice in float64 with two batch groupings on the local GPU, saving
   the most probable tokens, for a "batch size changed, text changed" ticker (run card needed).
8. **Hidden chaos, traced**: a portrait and trace at the hostility band-chaos cell (α = 0.0655, log₁₀T = 0.18995) where a
   seven-token text repeats while the state moves (local GPU; run card).
9. **The bound hierarchy drawn progressively** from `docs/bh/` receipts (CPU), for C8.
10. **Article Fig. 4, the neuron-steering diagram** (blog `tikz-neuron-steering.svg`) as a 6 s shot in C4, with "one
    neuron's output direction: 9.9 % of cells chaotic".
11. **`narrate_video.py` option to keep the source audio** under the narration with ducking, so the trace can be narrated
    over (small code change).
12. **Burned-in captions** as a second deliverable; YouTube chapter markers from the assembly offsets.
13. **A quiet music bed** (only with a clear license; below −30 LUFS under the voice).

## 5. Production pipeline

### 5.1 Format and style

- **Video**: 1920x1080, 30 fps (the sweep's 12 fps and the rotation's 24 fps are converted by frame repetition, no
  interpolation), yuv420p, H.264 High, CRF 18 for the master (CRF 16 for intermediate shots), keyframe every 2 s,
  `+faststart`.
- **Frame**: background `#0B0B0F`. Dark-native clips (tours, traces, sweep) play full frame. White-background material
  (paper figures, the 1280x720 clips, the rotation video) and the beige blog SVGs sit on plates: 1760x990 for figures
  (margins 80/45 px), 1600x900 for the 1280x720 clips (scale 1.25, offsets 160/110 px), with the top margin for the
  chapter tag and the bottom margin for captions. This avoids white flashes between dark clips and avoids upscaling the
  720p clips to full frame.
- **Type**: DejaVu Sans and DejaVu Sans Mono, the fonts of every existing clip (matplotlib's default): files in
  `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/` (`DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`,
  `DejaVuSansMono.ttf`). Titles 52 px bold white; body 32 px `#C8C8D0`; captions 26 px `#9A9AA6`; lower thirds 30 px
  white on black at 55 % opacity, 24 px padding, bottom 90 px; chapter tag 34 px bold, top right, first 3.5 s of each
  chapter.
- **Accent colors** (from the existing clips): chaos orange `#FF8C1A`, divergence red `#FF6B4D`, periodic blue
  `#4DB8FF`, highlight gold `#FFD93F`.
- **Transitions**: hard cuts within a chapter; 0.5 s crossfades between chapters; 0.6 s fade-in on the title, 1 s
  fade-out at the end.
- **Text rules**: American spelling; "most probable next token" wherever a color or a word is shown; "numerically
  observed chaotic attractor"; "lag-one plot"; every classical-model clip labeled "classical model, not the language
  model"; [R] and [L] numbers carry a small source note.

### 5.2 Voice and audio

- **Voice**: OpenAI `gpt-4o-mini-tts`, voice `sage`, as in every existing clip, through `tools/narrate_video.py` (key in
  `.env`, never printed; lines cached in `../neural-shrimp-data/videos/narration-cache/`). Instructions, one string for
  all chapters: "Calm, clear science narrator speaking to machine-learning researchers. Natural, engaging pace; precise
  articulation of technical terms and numbers; a brief pause between sentences." (Avoid "unhurried": the tour narration
  with it ran at 128 words per minute.)
- **TTS text**: the respellings of narration-draft.md ("float sixty-four", "b-float sixteen", "G S M eight K", "four
  point six six eight eight", "eight hundred ninety-six", "period eight"); no symbols such as ⏎; avoid model names in
  speech (they are on screen). Keep the display text in a `display` field of each line for the captions.
- **Music**: none in version 1.
- **Loudness**: narrate_video.py normalizes each chapter to −16 LUFS; the master gets a two-pass linear loudnorm to −16
  LUFS integrated, −1.5 dBTP true peak, LRA ≤ 11; 48 kHz stereo (dual mono), AAC 192 kb/s. The trace's spoken tokens
  (already at −16 LUFS) stay as they are.

### 5.3 Steps

1. **Preflight** (section 3.0); create `$W` folders (section 3.6).
2. **Build the five new tools** (sections 3.1-3.5) and render their pieces: cards, figure shots, N3a, N3b, N4, N5.
   Check one still of every figure shot against its box coordinates.
3. **Cut the existing clips** (section 3.6).
4. **First TTS pass, to measure** (needs the human's go-ahead for speech synthesis). Make a black 600 s dummy,
   `ffmpeg -y -f lavfi -i color=c=black:s=320x180:r=30 -t 600 -c:v libx264 -pix_fmt yuv420p $W/tts-measure/black600.mp4`,
   and for each chapter a script `docs/explainer/narration/CN.measure.json` with `"video"` the dummy, `"pack": true` and
   the chapter's lines; run `uv run --no-sync python tools/narrate_video.py docs/explainer/narration/CN.measure.json`.
   The timelines give each line's duration; the cache makes the second pass free.
5. **Fit the pictures to the voice**: figure and card shots get the duration of their lines plus 0.35 s per gap plus 1 s
   of landing; video clips keep their cuts, and a short shortfall is filled by holding the clip's last frame
   (`tpad=stop_mode=clone:stop_duration=D`). Concatenate each chapter's shots (re-encode, not stream copy, to avoid
   timebase mismatches): `ffmpeg -f concat -safe 0 -i $W/chapters/CN.list $ENC $W/chapters/CN-picture.mp4`. C3 is cut in
   three parts: C3a (shot 3.1), C3b (shot 3.2, with its audio, not narrated) and C3c (shots 3.3-3.5).
6. **Second TTS pass, final**: `docs/explainer/narration/CN.json` with `"video": "../neural-shrimp-data/videos/explainer/
   chapters/CN-picture.mp4"`, `"out"` the narrated chapter, `"gap": 0.35`, `"tail": 0.8`, lines at their shot starts plus
   0.3 s, no holds. Example for C2 (shots 2.1 at 0 s, 2.2 at 30 s, 2.3 at 44 s):

   ```json
   {"video": "../neural-shrimp-data/videos/explainer/chapters/C2-picture.mp4",
    "out": "../neural-shrimp-data/videos/explainer/chapters/C2-narrated.mp4",
    "model": "gpt-4o-mini-tts", "voice": "sage",
    "instructions": "Calm, clear science narrator speaking to machine-learning researchers. Natural, engaging pace; precise articulation of technical terms and numbers; a brief pause between sentences.",
    "gap": 0.35, "tail": 0.8,
    "lines": [
     {"at": 0.3, "text": "This scan is a phase diagram: ...", "display": "This scan is a phase diagram: ..."},
     {"at": 30.3, "text": "Colored by what the loop writes, ...", "display": "Colored by what the loop writes, ..."},
     {"at": 44.3, "text": "Steered toward hostility, ...", "display": "Steered toward hostility, ..."}]}
   ```

7. **Assemble** with crossfades (0.5 s between chapters, a near-cut between C3a and C3b, 0.25 s between C3b and C3c):

   ```python
   import json, subprocess
   W = "../neural-shrimp-data/videos/explainer"
   parts = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6", "C7", "C8"]
   join = {("C3a", "C3b"): 0.04, ("C3b", "C3c"): 0.25}
   files = [f"{W}/chapters/{p}-narrated.mp4" if p != "C3b" else f"{W}/shots/C3-02-trace.mp4" for p in parts]
   dur = lambda f: float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                  "-of", "csv=p=0", f]))
   D = [dur(f) for f in files]
   fc = [f"[{i}:v]fps=30,format=yuv420p,setsar=1,settb=AVTB[v{i}];[{i}:a]aresample=48000,"
         f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]" for i in range(len(files))]
   v, a, t, offsets = "[v0]", "[a0]", D[0], [0.0]
   for i in range(1, len(files)):
       x = join.get((parts[i - 1], parts[i]), 0.5)
       off = t - x
       fc.append(f"{v}[v{i}]xfade=transition=fade:duration={x}:offset={off:.3f}[vx{i}];"
                 f"{a}[a{i}]acrossfade=d={x}[ax{i}]")
       v, a, t = f"[vx{i}]", f"[ax{i}]", off + D[i]
       offsets.append(off)
   cmd = ["ffmpeg", "-y"] + sum([["-i", f] for f in files], []) + [
       "-filter_complex", ";".join(fc), "-map", v, "-map", a, "-c:v", "libx264", "-preset", "slow", "-crf", "18",
       "-pix_fmt", "yuv420p", "-r", "30", "-g", "60", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
       "-movflags", "+faststart", f"{W}/final/assembled.mp4"]
   subprocess.run(cmd, check=True)
   json.dump({"parts": parts, "files": files, "durations": D, "offsets": offsets},
             open(f"{W}/final/assembled.json", "w"), indent=1)
   ```

8. **Master loudness** (two passes):

   ```bash
   ffmpeg -hide_banner -i $W/final/assembled.mp4 -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null - 2> $W/final/loudnorm1.txt
   # read measured_I, measured_TP, measured_LRA, measured_thresh and target_offset from the JSON block at the end
   ffmpeg -y -i $W/final/assembled.mp4 -c:v copy \
     -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=$MI:measured_TP=$MTP:measured_LRA=$MLRA:measured_thresh=$MT:offset=$OFF:linear=true,aresample=48000" \
     -ac 2 -c:a aac -b:a 192k -movflags +faststart $W/final/neural-shrimp-explainer-v1.mp4
   ffmpeg -hide_banner -i $W/final/neural-shrimp-explainer-v1.mp4 -af ebur128=peak=true -f null - 2>&1 | tail -12
   ```

9. **Captions**: an SRT built from each chapter's timeline (`CN-narrated.mp4.timeline.json`: `starts`, `seconds`) and the
   `display` text of its script, shifted by the chapter offsets of `assembled.json`, split into cues of at most two lines
   of 42 characters and timed in proportion to their length; write `final/neural-shrimp-explainer-v1.en.srt`. Upload as
   a sidecar. YouTube chapters: the chapter offsets, rounded down to the second.
10. **QA and accuracy gate** (section 5.4), then the human's review.
11. **Provenance**: `final/manifest.json` with the SHA-256 of the master, the SRT and every input (clips, figures,
    receipts, card PNGs), the git commit of the tools, and the TTS settings and line texts; a compact copy as
    `docs/explainer/receipt.json`. Commit the new tools, the narration scripts and the receipt (not the videos), staging
    files explicitly and sweeping them for secrets; no dollar amounts in committed files (cost goes in the private
    ledger).

### 5.4 QA gates

- Duration 7:30-8:50; no frozen frame longer than 2 s except the planned holds; no black frames except the end fade
  (`ffmpeg -vf blackdetect`).
- Loudness −16 ± 1 LUFS integrated, true peak ≤ −1.5 dBTP; every spoken token of shot 3.2 audible.
- Every frame text: American spelling; no "strange attractor"; no "return map" for the loop; colors explained as "most
  probable next token"; classical-model clips labeled.
- Every spoken and on-screen number checked against section 6, from the article or the receipt, not from this plan.
- Each figure panel named in the narration is on screen when it is named.
- The end card shows the placeholder "arXiv:XXXX.XXXXX" until the listing exists; replace it and re-render only the end
  card and the final assembly.

### 5.5 Effort and order for a same-day version 1

About four to five hours of agent time on the CPU: tools and pieces (about two hours, the four tools in parallel), clip
cuts (30 min), two TTS passes (30 min including fitting), assembly, loudness and captions (30 min), QA (45 min). The
critical path is `render_figure_shot.py`, which most shots use.

### 5.6 If the cut runs long

Drop, in this order: P3.5 and shot 3.5 (already optional), P4.2, P1.3's last sentence, P3.4 (the sweep then plays
under captions only), the first sentences of P2.3 and P7.1 (all optional, about 40 s), then shorten shot 3.4 to 8 s and
the end card to 6 s. If it still runs past 8:50, cut shot 4.2 (8 s) and shorten shot 5.4 to the quartic clip alone.

## 6. Accuracy checklist

Status column: OK = matches the source; Caveat = state as written, with the given qualification. Tags as in the
header. Line references are to `chaos-journal-neural-shrimp/main.tex` (A) and `paper/llm/main.tex` (R).

| ID | Claim (spoken unless marked "screen") | Source | Status |
|---|---|---|---|
| 0.1 | Half a billion parameters | A Sec. II (Qwen2.5-0.5B-Instruct) | OK |
| 0.2 | Same loop twice, six or four CPU threads, which only reorders the additions | A Sec. V, l. 526-528 | OK |
| 0.3 | Same words for 27 steps; they part at step 28 | A Sec. V (27 steps); receipts docs/chaos-texts (step 28) | OK |
| 0.4 | Screen: float64, same processor, after 1,000 warm-up steps | R l. 809-812 | Caveat [R] |
| 0.5 | Nothing sampled; deterministic; chaotic at this setting | A Sec. II; A Table I (λ = 0.10, labeled chaotic) | OK (finite-time label, defined in C2) |
| 0.6 | Thesis: mapped, partly predicted from derivatives, in places proved | A abstract | OK |
| 1.1 | Core step of Soft Thinking: feed back the expected embedding | A Sec. I, II | OK |
| 1.2 | Prompt and window fixed; one step = one forward pass; state = 896 numbers | A Sec. II (d = 896, autonomous map) | OK |
| 1.3 | Two knobs: temperature and steering strength in the residual stream | A Sec. II | OK |
| 1.4 | Shown word = most probable next token; the loop feeds back the full average | A Sec. V l. 514-515; consult 16 item 6c | OK |
| 1.5 | (opt.) T → 0: greedy decoding with a one-token memory | A Sec. II l. 181-182; Sec. VIII | OK |
| 2.1 | Gray: settles into a cycle and repeats a fixed text | A Fig. 1 caption | Caveat: 4.4 % of cells gray with neither label (screen) |
| 2.2 | Colored: copies a hair apart separate exponentially; rate = largest Lyapunov exponent | A Sec. IV "Scans and chaos labels" | OK |
| 2.3 | "Chaotic" is a numerical label, not a proof | A abstract, Sec. IV | OK |
| 2.4 | Screen: 74 to 92 % of sampled labels persist to 10⁵ steps | A Sec. I (74 %), Sec. IV (92 %) | OK (true extremes) |
| 2.5 | Windows are texts: "is it"; blank line + "The existence"; each stripe adds one pair | A Sec. IV, Fig. 2b, Table II | OK |
| 2.6 | (opt.) Chaotic loops reuse the same words in no fixed order | A Sec. V | OK |
| 2.7 | Hostility cell: copy 10⁻¹⁰ away agrees 44 steps, writes "No" at step 45 | A Table I, Sec. V | OK |
| 2.8 | "Yes" and "No" almost tied (screen: 0.303 and 0.298) | A Sec. V ("where no token dominates"); R l. 791-793; receipt | Caveat [R] numbers |
| 3.1 | Many trajectories, three principal components; most settle on a numerically observed chaotic attractor | A Sec. VI l. 781-786 (498 of 512; 14 to period 4) | OK |
| 3.2 | Thin, folded filaments | A Sec. VI ("thin folded filaments in layers") | OK (descriptive) |
| 3.3 | Lyapunov dimension ≈ 1.26, an estimate from the exponents; little more than a curve | A Sec. VI; consult 16 | Caveat: "by that estimate" is spoken; not a measured fractal dimension |
| 3.4 | The dot jumps because each step is a full forward pass | A Sec. II; R l. 860-862; log 20:05 UTC | OK |
| 3.5 | Nearby setting periodic; almost every start reaches the same eight states (screen: 509 of 512) | Log 2026-10-18 20:05 UTC | Caveat [L] |
| 3.6 | (opt.) Along one row: short loop, doubling cascade, chaos, order again | Log 19:15, 19:35 UTC; clip | Caveat [L] |
| 3.7 | (opt.) Three delays of log p_max give a reconstruction with few false neighbors | A Sec. VI (3.4 %); R ("not a verified Takens embedding") | OK |
| 4.1 | The loop's Jacobian factorizes exactly: (1/T) × embedding covariance × network Jacobian | A Eq. (3), Fig. 3 | Caveat: exact form for tied embeddings (screen) |
| 4.2 | The covariance is large only when tokens with different embeddings compete | A Sec. II "Where steering and temperature enter" | OK |
| 4.3 | (opt.) Without the network: a modern Hopfield update, no cycles, no chaos | A Sec. III (tied embeddings) | OK |
| 4.4 | Steering into chaos raises the readout factor by 2.58 nats per step; the network factor barely moves | A Sec. III, Fig. 5a (−0.07 [−0.23, 0.12]) | OK (recorded orbits, 0.5B) |
| 4.5 | A logit bias alone makes 3.5 % of the plane chaotic | A Sec. III (241 of 6837) | OK |
| 4.6 | Random directions, or one neuron's output direction, about as chaotic as concept directions | A Sec. IV "Generality" (0.5B and 8B) | Caveat: "in the smaller models we tested" spoken; at 70B the directions differ [R] |
| 4.7 | The direction selects which tokens compete, and so what the attractors write | A Sec. IV ("selects the texts"); R l. 622-623 | OK |
| 4.8 | Steering strength is a bifurcation parameter; no smooth dose response | A Sec. VIII | Caveat: continuous-feedback loops; untested for ordinary generation (screen) |
| 4.9 | Screen: rotation keeps the chaotic share at 13 to 24 % over 103 angles | R l. 621, 4653-4655 | Caveat [R] |
| 4.10 | Screen: two depths, 78 % of chaotic cells where neither push alone is chaotic | A Sec. VII l. 860 | OK |
| 5.1 | The windows are organized as in low-dimensional maps | A abstract, lay summary | OK |
| 5.2 | Multipliers = eigenvalues of the Jacobian over one cycle; attract while below one in size | A Sec. II "Numerics" | OK |
| 5.3 | Fold: multiplier +1, the loop vanishes with an unstable twin; flip: −1, the period doubles | A Sec. I l. 108-109, Sec. II, Sec. V | OK |
| 5.4 | Three-step loop doubles into chaos; last ratio 4.6688 against 4.6692, at the float64 limit | A abstract, Sec. VI, Fig. 10 | OK (finite-order; universality not established) |
| 5.5 | Past a fold, the ghost delays trajectories | A Sec. V, Fig. 8c-d | OK |
| 5.6 | Forecasts from the network's derivatives, fixed in advance at four new folds: 42 of 48 within one cycle | A abstract, Sec. V, Table III | OK; screen adds the 6 misses (3.1 to 5.2 % short) and the failed all-forecast criteria |
| 5.7 | Shrimps: periodic islands with a body and thin legs, outlined by fold and flip curves; classical map shown | A Sec. VII, Fig. 12; quartic clip = R Fig. "cusp kinds" | OK (classical model labeled) |
| 5.8 | Period-5 window steered at two depths, followed cell by cell: a numerically verified shrimp | A Sec. VII, Fig. 13 | OK (not proved) |
| 5.9 | Orbit counts at 13 points fixed in advance matched a swallowtail's prediction at nine | A Sec. VII l. 879-882 | OK |
| 6.1 | Interval arithmetic: every number a range guaranteed to contain the true value | A Sec. VI (ball arithmetic, Krawczyk) | OK |
| 6.2 | Proves an attracting period-8 orbit of the full model | A Sec. VI l. 756-767 | OK |
| 6.3 | To our knowledge the first such proof for a full transformer language-model map | A Sec. I l. 144-145 | OK ("to our knowledge" spoken) |
| 6.4 | Two attracting orbits coexist at one setting, writing "is it" and "The existence" | A Sec. VI l. 768-777 | OK |
| 6.5 | Which one a given start reaches is not proved | A Sec. VI l. 778-779 | OK |
| 6.6 | Screen: basin slice, float32 | R Fig. "pair basins" | Caveat [R], numerical |
| 7.1 | Kernels order sums by matrix shape; another batch differs in the last digits | A Sec. IV "Reproducibility" (about 6 × 10⁻¹⁵) | OK |
| 7.2 | A periodic loop damps the difference; a chaotic loop amplifies it | A Sec. IV l. 479-480 | OK |
| 7.3 | 190 of 192 chaotic float64 loops separated after a median of 102 steps, on the Lyapunov schedule; 182 of 192 periodic stayed | A Sec. IV; recomputed from raw arrays on 18 October | OK |
| 7.4 | Fixed matrix shapes: bitwise identical | A Sec. IV (448 cells; control fixed in advance); FS arrays all zero | OK |
| 7.5 | In bfloat16, feedback carries differences into answers even where the loop contracts | A abstract, Sec. IV "Answers of a task" | OK |
| 7.6 | Reasoning models solving 86 to 90 % of GSM8K changed 14.5 and 19.5 % of Soft Thinking answers with the batch layout alone (tests fixed in advance) | A abstract | OK |
| 7.7 | Ordinary sampled reasoning changed about as many | R abstract; docs/pi/pm2-summary.json (10.2, 19.9 %) | Caveat [R]: drop if the report is not public |
| 7.8 | Batch-invariant kernels remove one source; others grow the same way in a chaotic regime | A Sec. IV l. 483-485 | OK |
| 7.9 | Screen: 56 % and 55 % of 96 additions; greedy 0 and 10 of 96; reset 66 → 1.6 % and 64 → 1.6 % (exploratory); Soft Thinking 21 %, replay 6 % | A Sec. IV l. 489-500 | OK |
| 8.1 | Every step of the proved period-8 orbit stretches some direction, yet the cycle contracts | A Sec. VIII l. 961-964 | OK |
| 8.2 | In one cascade, 65 chaotic settings wrote an exactly repeating text | A Sec. VI l. 743-751 (65 of 182; 2000 steps) | OK |
| 8.3 | What stays predictable: attractors and the boundaries between them | A Sec. VIII l. 964 | OK |
| 8.4 | Chaos: a finite-time label | A abstract | OK |
| 8.5 | Bounds locate chaos only by measuring along an orbit, not by forecasting from the weights | A Sec. III l. 311-315 | OK |
| 8.6 | No accuracy information beyond first-token uncertainty | A Sec. IV l. 501-502 | OK |
| 8.7 | Crossroad areas verified only in classical maps; search under way | Log 2026-10-18 20:05 UTC; data-example slide 3 | Caveat [L] |
| 8.8 | Larger models do not simply leave the chaos behind | A Sec. IV "Generality", Fig. 7 | OK |
| 8.9 | Screen (end card): coupled copies synchronize; 8 or 16 soft tokens more chaotic; Huginn bifurcation confirmed in a frozen test | A Sec. VIII | OK |

If the full report is not public at release, the [R]-only items are 0.4 (drop "same processor"), 2.8 (drop the two
probabilities), 4.9 (drop the rotation share), 6.6 (keep the image, label it "numerical slice, float32"), 7.7 (drop the
sentence), the hidden-bands inset of shot 8.1 (keep the article's numbers only) and the optional bound-hierarchy inset
of shot 8.2.

## 7. Decisions for the human

1. **Speech synthesis**: approve OpenAI TTS for about 8.5 minutes of narration plus the measuring pass (cached, so
   billed once); cost recorded in the private ledger.
2. **Voice**: keep `sage`, as in all existing clips (recommended), or choose another.
3. **Music**: none (recommended) or a licensed quiet bed.
4. **Hook**: the thread-count split (recommended; the article's own example of reordered sums) or the Yes/No twin (more
   dramatic, but an injected 10⁻¹⁰ displacement, not a computation change).
5. **The full report at release**: if it is not posted with the arXiv v1, apply the [R] list of section 6.
6. **End card**: placeholder "arXiv:XXXX.XXXXX" and `github.com/tdj28/neural-shrimp`; also disclose the synthetic
   narration voice (recommended) and, if wished, that the video was assembled with AI coding agents under the author's
   direction.
7. **Before submission (outside this plan)**: the article's `fig-real-shrimp.png` predates the curves its text
   describes, and two places in the full report still say only the period-8 orbit is proved (appendix C).
8. **Captions**: sidecar SRT (recommended) or burned in. **Title** of the video: the article's title (recommended, for
   search) or a shorter one.

## Appendix A. Figure inventory

All 82 figure files of the full report (78 PNG figures and four TikZ diagrams), with the article's figure number where
it uses them and the explainer's use: "v1" = in the first version, "alt" = an alternative, "fallback" = if a new clip is
not ready, "v2" = nice-to-have. All of them also appear in the blog post, which adds the schematics listed after the
table. The article's own figures are the 14 PNG files marked with a figure number, plus Fig. 4 (TikZ, neuron steering).

| # | File (paper/llm/figures) | Article | Content (first sentence of the caption) | Explainer use |
|---|---|---|---|---|
| 1 | `fig-cross512-lyapunov` | Fig. 1 | Chaos and repeating text in a language model fed its own output: the soft-token map of Qwen2.5-0.5B-Instruct, ... | v1 C2 shot 2.1 (Fig. 1, push-in) |
| 2 | `fig-derivative-check` | Fig. 3 | The factorized derivative checked against the network's own derivatives at a state of the period-8 orbit of th... | v1 C4 shot 4.1 inset (optional) |
| 3 | `fig-norm-bounds` |  | Norm bounds of the factorized derivative along the model's own orbits (Qwen2.5-0.5B-Instruct, philosophy direc... |  |
| 4 | `fig-bound-hierarchy` |  | The bound hierarchy with exact norms (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, float64; 24 cel... | alt C8 shot 8.2 inset [R] |
| 5 | `fig-mechanism-summary` | Fig. 5 | Why steering makes the loop chaotic: the three measurements of this section (Qwen2.5-0.5B-Instruct unless mark... | v1 C4 shot 4.3 (panels a, b, c) |
| 6 | `fig-mechanism` |  | Why steering makes the loop chaotic (Qwen2.5-0.5B-Instruct, philosophy direction, float64) |  |
| 7 | `fig-morph-samples` |  | The soft-token map of Qwen2.5-0.5B-Instruct (steering at layer 12) while the steering direction rotates at con... | fallback C4 shot 4.4 [R] |
| 8 | `fig-stripes-lyapunov-curves` | Fig. 2 | Soft-token map of Qwen2.5-0.5B-Instruct steered along the philosophy direction at layer 12, 256^2 float64 cells | v1 C2 shot 2.2 (left half) |
| 9 | `fig-stripes-period-fold` |  | Soft-token map of Qwen2.5-0.5B-Instruct steered along the philosophy direction at layer 12, 256^2 float64 cells |  |
| 10 | `fig-cross512-families` | Fig. 2 | The cells of Figure (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12; 512^2 float64 cells over α∈[0.06... | v1 C2 shot 2.2 (right half) |
| 11 | `fig-zoom-z2-lyapunov` |  | Zooms of 256^2 float64 cells of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12), with the large... |  |
| 12 | `fig-zoom-z2-families` |  | Zooms of 256^2 float64 cells of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12), with the large... |  |
| 13 | `fig-zoom-z3-lyapunov` |  | Zooms of 256^2 float64 cells of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12), with the large... |  |
| 14 | `fig-zoom-z3-families` |  | Zooms of 256^2 float64 cells of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12), with the large... |  |
| 15 | `fig-strange-attractor` | Fig. 11 | Projected geometry of a numerically observed chaotic attractor of the soft-token loop (Qwen2.5-0.5B-Instruct, ... | v1 C3 optional (P3.5); fallback for the tour |
| 16 | `fig-pair-basins` |  | Basins of the certified 2-cycle (blue, " is it") and 3-cycle (orange) at α=0.06468, log10 T=-0.148, on the pla... | v1 C6 shot 6.2 background [R] |
| 17 | `fig-chaos-twins` |  | How a difference of 10^-10 becomes a different text | fallback C2 shot 2.3 [R] (low resolution, 876 px) |
| 18 | `fig-fixed-shape` |  | Fixed-shape control and dose test (Qwen2.5-0.5B-Instruct, the 448 labeled cells of two float64 maps, 1500 lock... | fallback C7 shot 7.2 [R] |
| 19 | `fig-repro` |  | Only the batch layout differs | fallback C7 shot 7.2 [R] |
| 20 | `fig-layer-sweep` |  | The philosophy direction of Qwen2.5-0.5B-Instruct, built with the same recipe at each of five blocks of 24: la... | v2 C4 (injection depth) [R] |
| 21 | `fig-two-block` |  | Two steering directions at two blocks of Qwen2.5-0.5B-Instruct: largest Lyapunov exponent on 96×96 float64 cel... | alt C4 shot 4.5 background [R] |
| 22 | `fig-fold` | Fig. 8 | The fold behind the stripes: the 2-cycle "is it" at log10 T=-0.115 (Qwen2.5-0.5B-Instruct, philosophy directio... | v1 C5 shot 5.3 (a-b, then c-d) |
| 23 | `fig-passages` |  | Bottleneck passages and the window law at the fold of the 2-cycle "is it" (Qwen2.5-0.5B-Instruct, philosophy d... |  |
| 24 | `fig-window-law` | Fig. 9 | Bottleneck passages and the window law at the fold of the 2-cycle "is it" (Qwen2.5-0.5B-Instruct, philosophy d... | alt C5 shot 5.3 (1/N^2 law) |
| 25 | `fig-return-maps` |  | Two-step return maps of the probability of "it" near the fold of the 2-cycle "is it" (Qwen2.5-0.5B-Instruct, p... |  |
| 26 | `fig-feigenbaum-ratios` |  | Finite-order interval ratios of the doubling sequences: the crescent of Qwen2.5-0.5B-Instruct (philosophy dire... | alt C5 shot 5.2 tail [R] |
| 27 | `fig-feigenbaum-tree` | Fig. 10 | The Feigenbaum tree of the crescent cascade (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, log10 T=... | v1 C5 shot 5.2 (nested zoom) |
| 28 | `fig-hostile-cascade` |  | The hostility-steered soft-token map of Qwen2.5-0.5B-Instruct (layer 12) |  |
| 29 | `fig-hub-cascade` |  | Inside the hostile period-7 body (Qwen2.5-0.5B-Instruct, hostility direction at layer 12): float64 bifurcation... |  |
| 30 | `fig-superstable-lines` |  | Superstable structure of the hostile period-7 body (Qwen2.5-0.5B-Instruct, hostility direction at layer 12, fl... |  |
| 31 | `fig-hub-shrimp` |  | The period-7 body of the spiral region (Qwen2.5-0.5B-Instruct, hostility direction at layer 12): attractor per... |  |
| 32 | `fig-hidden-bands` |  | Chaos behind a repeating text: the soft-token map of Qwen2.5-0.5B-Instruct steered along hostility at layer 12... | v1 C8 shot 8.1 inset [R] |
| 33 | `fig-cusp-kinds` |  | The three kinds of organization around a cusp of a fold curve, computed from the quartic map of Carcass`es et ... | v1 C5 via the quartic clip [R] |
| 34 | `fig-period5-folds` |  | The skeleton of the period-5 window in the plane steered at two depths (Qwen2.5-0.5B-Instruct, gratitude at bl... |  |
| 35 | `fig-real-shrimp` | Fig. 13 | The period-5 shrimp (Qwen2.5-0.5B-Instruct, gratitude at block 4 and hostility at block 16, log10 T=-0.1, floa... | v1 C5 shot 5.5 (inside the data-example clip; use the report copy) |
| 36 | `fig-shrimp-anatomy` | Fig. 12 | The anatomy of a shrimp's cell in two classical one-dimensional models, computed (gray: one attracting fixed p... | v1 C5 shot 5.4 (panel a) |
| 37 | `fig-foliation` |  | The dovetail in Mira's foliated representation: parameters and one state coordinate |  |
| 38 | `fig-sheets` |  | What the fixed-start map shows, and what it hides (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, fl... |  |
| 39 | `fig-sheets-3d` |  | The two families as sheets | v2 C2 (two overlapping families) [R] |
| 40 | `fig-hidden-strip` |  | The crescent's left edge is a switch, not its fold |  |
| 41 | `fig-e2e` |  | One picture, in float64 (Qwen2.5-0.5B, philosophy direction, layer 12) | v2 C5 (compromise orbits, panel a) [R] |
| 42 | `fig-qwen15-zoom-lyapunov` | Fig. 6 | Qwen2.5-1.5B soft-token map steered along the philosophy direction at layer 14, 256^2 float64 cells over α∈[0.... | alt C8 shot 8.2 (generality) |
| 43 | `fig-qwen15-zoom-families` | Fig. 6 | Qwen2.5-1.5B soft-token map steered along the philosophy direction at layer 14, 256^2 float64 cells over α∈[0.... | alt C8 shot 8.2 (generality) |
| 44 | `fig-scale` |  | How much of the temperature axis is chaotic, for the fifteen unsteered maps of Table : against parameter count... | alt C8 shot 8.2 [R] |
| 45 | `fig-damping` |  | Where the network damps a change of the fed-back token |  |
| 46 | `fig-zoom-llama70b-z128` | Fig. 7 | Chaos at 70B parameters: the soft-token map of Llama-3.3-70B-Instruct steered along its hostility concept vect... | v1 C8 shot 8.2 (70B, 20.8 % chaotic) |
| 47 | `fig-huginn-gsm8k` |  | Huginn-0125 chain-of-thought accuracy on GSM8K against the relaxation strength λ (eight examples, greedy decod... |  |
| 48 | `fig-dialogue` |  | Two loops writing one text |  |
| 49 | `fig-sync` |  | Synchronization of two coupled copies of the soft-token loop (Qwen2.5-0.5B-Instruct, philosophy direction, lay... | v2 end card (coupled copies) [R] |
| 50 | `fig-reservoir` |  | Memory capacity of the soft-token loop used as a reservoir (Qwen2.5-0.5B-Instruct, philosophy direction at lay... |  |
| 51 | `fig-false-chaos` |  | False chaos and real chaos in a twin test (Qwen2.5-0.5B-Instruct, float64, philosophy direction at block 12; 4... |  |
| 52 | `fig-zoom-z3-curves` |  | The period-6 window of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12) with its float64 fold cu... |  |
| 53 | `fig-p4-curves` |  | Period map of the high-temperature region of the Qwen2.5-0.5B-Instruct map (philosophy direction, layer 12; 19... |  |
| 54 | `fig-head-temperature` |  | The head of the period-5 shrimp continued in temperature: the near-nilpotent point, where the sum of the leadi... |  |
| 55 | `fig-head-rank` |  | The rank of the period-5 shrimp's return map near its head (float64; , ) |  |
| 56 | `fig-shrimp-temperature` |  | The period-5 shrimp at two temperatures (Qwen2.5-0.5B-Instruct, gratitude at block 4 and hostility at block 16... |  |
| 57 | `fig-lower-leg` |  | The lower-left leg of the period-5 shrimp and its own curves (Qwen2.5-0.5B-Instruct, gratitude at block 4 and ... |  |
| 58 | `fig-bimodal` |  | Band return maps with two or more turning points, from two-block planes of Qwen2.5-0.5B-Instruct (float64, log... |  |
| 59 | `fig-period5-window` |  | A period-5 window in the plane of gratitude at block 4 and hostility at block 16 (Qwen2.5-0.5B-Instruct, float... |  |
| 60 | `fig-return-maps-body` |  | Seven-step return maps s_n+7 against s_n, for each phase of the 7-periodic text, of chaotic cells next to the ... |  |
| 61 | `fig-cobwebs` |  | Cobweb diagrams of the 7-step return map of the hostility-steered loop (Qwen2.5-0.5B-Instruct, layer 12, float64) |  |
| 62 | `fig-repeating-text-reduction` |  | Scalar return-map approximation across the hostile cascade (Qwen2.5-0.5B-Instruct, hostility at layer 12, α=0.... |  |
| 63 | `fig-hub-zoom` |  | The lower part of the period-7 body of the hostility-steered Qwen2.5-0.5B-Instruct map (layer 12) at higher re... |  |
| 64 | `fig-bridge` |  | Mean laminar length near the fold of the 2-cycle "is it" (Qwen2.5-0.5B-Instruct, philosophy direction at layer... |  |
| 65 | `fig-qwen15-route` |  | Qwen2.5-1.5B soft-token map steered along the philosophy direction at layer 14, along log10 T at α=0.100: peri... |  |
| 66 | `fig-qwen7b-hostility` |  | Qwen2.5-7B soft-token map steered along the hostility direction (layer 14), 64^2 float64 cells: largest Lyapun... |  |
| 67 | `fig-olmo2-maps` |  | OLMo-2-1124-7B-Instruct soft-token maps, 64^2 float64 cells each, steered at layer 16 along the philosophy dir... |  |
| 68 | `fig-llama-maps` |  | Llama-3.1-8B-Instruct soft-token maps (64^2 float64 cells, steering at layer 16) along gratitude, hostility an... |  |
| 69 | `fig-llama-hires` |  | Llama-3.1-8B-Instruct soft-token maps steered at layer 16, at 128^2 float64 cells (each map merged from four 6... |  |
| 70 | `fig-same-architecture` |  | Same architecture, other weights: 128^2 float64 maps of Qwen2.5-0.5B-Instruct (left) and Qwen2-0.5B-Instruct (... |  |
| 71 | `fig-moe` |  | Mixtures of experts against dense models of the same family |  |
| 72 | `fig-prefix-scale` |  | Prompt-blocked size test (frozen before any run), Qwen2.5-7B and -14B on ten prompts (gray: development) |  |
| 73 | `fig-gain-scale` |  | Gain split of the unsteered loop (Section ) at 24 temperatures per model (500 transient and 1000 measured step... |  |
| 74 | `fig-controls-llama70b` |  | Steering controls at 70B: Llama-3.3-70B-Instruct soft-token maps (32^2 float64 cells, 300 transient and 300 me... |  |
| 75 | `fig-alpha0` |  | The unsteered soft-token map (α=0) along temperature, top to bottom: Llama-3.3-70B-Instruct, Llama-3.1-8B-Inst... | v2 C4 (chaos without steering) [R] |
| 76 | `fig-controls-q05` |  | Steering controls in Qwen2.5-0.5B-Instruct (128^2 float64 cells, steering at layer 12, one grid and one set of... |  |
| 77 | `tikz-neuron-steering` | Fig. 4 | Steering along one MLP neuron, in decoder layer 12 of Qwen2.5-0.5B-Instruct | v2 C4 (blog SVG of article Fig. 4) |
| 78 | `fig-controls-neurons-q05` |  | Steering along single MLP neurons in Qwen2.5-0.5B-Instruct (same grid, settings and color scale as Figure ): t... |  |
| 79 | `tikz-loop` |  | The soft-token loop and the factorized derivative (TikZ) | v1 C4 shot 4.1 (blog SVG) |
| 80 | `fig-readout-gain` |  | How the spread of the next-token distribution sets the readout gain (TikZ) | v1 C4 shot 4.2 via the blog SVG spread-readout-gain |
| 81 | `tikz-atlas` |  | A map of the bifurcation objects (TikZ) |  |
| 82 | `tikz-reduction` |  | The reduction to a one-dimensional return relation (TikZ) |  |

Blog-only schematics (in `/data2/praxagent/blog-source/content/posts/2026/10/chaos-bifurcations-language-model-feedback/`;
conceptual, no data): `soft-token-loop.svg` (v1 shot 1.1), `tikz-loop.svg` (v1 shot 4.1), `spread-readout-gain.svg`
(v1 shot 4.2, toy numbers), `batch-difference-schematic.svg` (v1 shot 7.1, "schematic, not data"),
`autoregressive-vs-greedy.svg` (alt C1, T → 0), `one-picture-schematic.svg` (v2 C5, compromise orbits),
`tikz-neuron-steering.svg` (v2 C4), `benettin-vs-free-twin.svg` (not used), `fig-toy-cobwebs.png` and
`fig-toy-route.png` (v2, a two-token toy loop; label as an analogy, not the language model).

## Appendix B. Existing video assets (probed 18 October, 13:49)

| File (`../neural-shrimp-data/videos/`) | Duration | Format | Audio | Use |
|---|---|---|---|---|
| `cusp-educational.mp4` | 277.5 s | 1280x720, 30 fps | AAC 48 kHz | Not cut directly; its three parts are used from their silent sources |
| `bifurcation-primer.mp4` (silent) / `-narrated.mp4` | 50.0 / 94.5 s | 1280x720, 30 fps | - / AAC | Shot 5.1: 12.2-36.3 s (fold 12.2-24.2, flip 24.2-36.3) |
| `cusp-sweep-quartic-48s.mp4` (silent) / `-narrated-v2.mp4` | 48.0 / 125.0 s | 1280x720, 30 fps | - / AAC | Shot 5.4: 8.7-31.0 s (the crossroad area forms) |
| `cusp-data-examples.mp4` (silent) / `-narrated.mp4` | 28.0 / 58.1 s | 1280x720, 30 fps | - / AAC | Shots 5.5 (0-20 s) and 8.2 (20.3-28.0 s) |
| `attractor-sweep-extended-12fps.mp4` (silent) / `-narrated.mp4` | 70.75 / 124.5 s | 1920x1080, 12 fps | - / AAC | Shot 3.4: 2.0-31.6 s at 2.5x (the row at log₁₀T = −0.148) |
| `attractor-tour-f64.mp4` / `-3x.mp4` / `attractor-tour-narrated.mp4` | 30 / 90 / 90.1 s | 1920x1080, 30 fps | - / - / AAC | Shots 0.2 (40-56 s) and 3.1 (0-29 s) of the 3x version; re-render running |
| `attractor-tour-p8.mp4` (and `attractor-tour-p8-fade.mp4`, planned by the attractor-video task) | 30 s | 1920x1080, 30 fps | - | Shot 3.3: 0-10 s; re-render with marked orbit running; prefer the fade version if it exists |
| `attractor-trace-spoken.mp4` | 52.6 s | 1920x1080, 30 fps | AAC | **Do not use** (title says "strange attractor"); v2 rendering |
| `attractor-trace-spoken-v2.mp4`, `attractor-trace-p8-spoken.mp4` | (rendering) | 1920x1080, 30 fps | AAC | Shot 3.2 (v2); the p8 trace is an alternative for shot 3.3 |
| `morph-steering-rotation.mp4` | 19.9 s | 1920x1080, 24 fps | - | Shot 4.4 at 1.5x |
| `archive/*` | - | - | - | Superseded; not used |

Narration scripts of the existing clips are in `docs/narration/*.json`; their timelines sit next to each narrated video.
All used `gpt-4o-mini-tts`, voice `sage`.

## Appendix C. Discrepancies found while planning (not fixed here)

1. **Article figure older than its text.** `chaos-journal-neural-shrimp/figures/fig-real-shrimp.png` (03:00) lacks the
   lower-left leg's own fold and flip curves; the article's text (l. 869-874: 39 of 39 leg cells, 717 of 754 pairs)
   describes them, and the report's copy (`paper/llm/figures/fig-real-shrimp.png`, 11:50) shows them. The other 13
   article figures are byte-identical to the report's.
2. **Full report, number of proofs.** The status note under the title (l. 43-44) and the caption and row of Table
   "evidence" (l. 2998-2999, 3027) say only the period-8 orbit is proved; the claims table, the abstract, the
   contributions, the limitations paragraph and the article say three periodic orbits are certified (the period-8 orbit
   and the coexisting period-2 and period-3 pair).
3. **Supplementary material.** The article says the full report will accompany "the journal version and a later
   version of this preprint"; the brief expects it with v1. The [R] list of section 6 covers both cases.
4. **Old trace title.** `attractor-trace-spoken.mp4` (12:41) is titled "One chaotic trajectory through the strange
   attractor"; the renderer's default title is now "...through the attractor", and the v2 render replaces it.
5. **Token colors.** Each attractor clip assigns colors by token frequency, so some tokens change color between clips;
   each clip has its own legend (version 2, item 1).
6. **Low-resolution figure.** `fig-chaos-twins.png` is 876 px wide, too small for a full-frame plate; it is only a
   fallback for shot 2.3.
