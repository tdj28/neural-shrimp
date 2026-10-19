# Consult 22: are the batch-layout answer effects real, and are they stated fairly?

## The decision this review informs

The article (arXiv v1, to be posted imminently) studies the soft-token feedback loop of language models as a
dynamical system. Its central numerical result is untouched here: in float64, chaotic loops of Qwen2.5-0.5B separate
on the schedule of their Lyapunov exponent when only the batch layout changes, and periodic loops mostly do not. One
paragraph then asks what this means for answers in soft-thinking pipelines, mostly in bfloat16. The human author
judged the reported answer-change rates high and asked for an audit. A local audit found a comparison bug in one
exploratory test (fixed) and showed that most changes occur where answers are fragile; the paragraph was rewritten.
Please judge the corrected paragraph: is it accurate, fairly qualified and free of artifacts, and must anything be
checked or changed before posting? You see no raw data; the counts below were recomputed locally from the raw records
(receipts named in the text). Judge whether they would support the claims if true, and say what you would need.

Terms: the batch layout is the set of requests the GPU processes together with one request. Changing it changes the
matrix shapes and hence the floating-point reduction order (He et al. 2025, Thinking Machines; Yuan et al. 2025,
reproducibility of reasoning models in bfloat16), not the intended computation. "Soft thinking" feeds back the
probability-weighted average of token embeddings instead of a sampled token.

## The corrected paragraph (article, LaTeX, verbatim)

\paragraph{Reproducibility and answers.} GPU kernels choose different reduction orders for different matrix shapes, so
a request's result depends on its batch layout, the set of requests that the GPU processes together with it
\cite{he2025nondeterminism,yuan2025nondeterminism}. The loop's regime decides what this rounding difference does
(Fig.~\ref{fig:repro}). Run twice from the same start with only the
grouping of cells into batches changed, 190 of 192 chaotic float64 cells of the 0.5B model separate past a relative
distance of $10^{-2}$, close to the step their Lyapunov exponent gives in a retrospective comparison (median ratio
0.91), while 182 of 192 periodic cells
stay within $10^{-2}$ for all 1500 steps; the same holds in Qwen2.5-7B, Llama-3.1-8B and Llama-3.3-70B. In the 0.5B
model the most probable tokens of the two runs come to differ, within the last 500 steps, in 173 of the chaotic cells
but in only 7 of the periodic ones. The feedback
contracts the rounding difference of non-associative floating-point sums \cite{goldberg1991floating} on a periodic
attractor and amplifies it at the rate $\lambda$ on a chaotic one. A control fixed in advance supports this account:
with the matrix shapes held fixed the runs are bitwise identical in all 448 cells, and injected displacements grow on
the Lyapunov schedule (median ratio 0.87 to 1.18), although, against the design's limit, more than 10\,\% of the
periodic cells also separated at displacements of $10^{-9}$ and larger, presumably by crossing basin boundaries (the
attractors reached were not classified). Batch-invariant
kernels \cite{he2025nondeterminism} remove this one source of difference, but a chaotic loop amplifies any other in
the same way. In bfloat16 the runs part within one or two steps at almost every cell, periodic ones included.

The feedback can carry such differences into answers, where those answers are fragile. When each of 96 two-digit
additions ran once alone and once beside three other additions, a short Soft Thinking procedure gave different answers
to 56\,\% of them in Qwen2.5-0.5B at $T=1$ and 55\,\% in Qwen2.5-7B at $T=10^{0.2}$ (greedy decoding: none and 10 of
96). There the procedure answers poorly: the 0.5B model answered 23 of the 96 correctly when alone, and in 24 of the 7B
model's 53 changes one of the two runs gave no number at all. At $T=10^{0.4}$, where the two models answered 90 and 96
of the 96 correctly, the layout changed no answer. In float64 no answer changed in the 0.5B model and 5\,\% changed in
the 7B model, whose loop expands at $T=10^{0.2}$. In an exploratory test on 128 newly drawn additions in bfloat16 on an
H100, resetting the batched loop to a batch-1 reference before every soft step, so that no rounding difference is
carried forward, cut the share of changed answers from 47 to 0.8\,\% in the 0.5B model and from 41\,\% to none in the
7B model. In a test whose design was fixed before its data were collected, Soft Thinking with its published settings
in Qwen2.5-7B gave different answers to 21\,\% of 256 GSM8K problems in batches of 16 prompts and of 8 (6\,\% when both
runs replayed the fed-back tokens of a reference run, so that no rounding difference was fed back); through this
pipeline the model answers only 22 to 30\,\% of the problems correctly, and whether an answer was correct changed for
6.2\,\% of them. In three reasoning models that answer 74 to 90\,\% correctly, DeepSeek-R1-Distill-Qwen-7B,
DeepSeek-R1-Distill-Llama-8B and QwQ-32B, the same change of batch layout changed 14.5, 28.9 and 19.5\,\% of the Soft
Thinking answers and 10.2, 21.5 and 19.9\,\% of those of discrete sampled thinking. Nearly all of these changes, 158 of
161 with soft thinking and all 132 with sampled thinking, came from long runs, which reached the thinking or the answer
budget in at least one layout; among runs that finished within both budgets in both layouts, the layout changed 3 of
480 soft answers and none of 477 sampled ones. The changes do not vanish with a larger budget: with 4096 thinking steps
(an exploratory rerun), the layout still changed 8.6\,\% of the soft and 17.2\,\% of the sampled answers of
DeepSeek-R1-Distill-Qwen-7B on 128 GSM8K problems. Beyond the model's uncertainty about its first answer token, the loop's
regime carries no detectable information about the accuracy of ordinary sampled answers, and when slight steering
moved a prompt's loop into chaos, no change of that accuracy was detected ($+0.36$ points, 95\,\% interval
$[-0.36,1.08]$)\smref{Behavioral tests in
detail}.

## Local audit (read-only, 19 October 2026; every item rechecked from raw records)

Construction and controls:
- Layouts. Arithmetic and reset tests: the problem alone (batch 1) or in row 0 of a batch of four with three fixed
  distractors of identical token length (unequal lengths raise an error). GSM8K tests: prompts grouped by exact token
  length, chunked into batches of 8 or 16 and filled with equal-length fillers. No call passes padding, an attention
  mask or position ids; batched and single runs use one code path (KV cache, one appended position per soft step).
- Randomness (sampled answers and sampled thinking): each row has its own generator, seeded by seed + problem index,
  and draws one uniform per global step whatever its state, so a problem's draw at step t does not depend on the
  layout. Caveat: draws are aligned by step, not by answer token, so after different switch steps the answer phase
  uses offset draws (rows where both layouts hit the cap share aligned draws and still change 10/11, 44/60, 19/26).
- Same-layout reruns are bitwise identical wherever measured: arithmetic batch-1 reruns 96/96 in all conditions; reset
  reference 128/128; the GSM8K replay arm reproduces the top-15 arm's layout-16 run bit for bit (256/256); a later
  rerun on the same GPU type reproduced every reasoning row that finished within 1536 thinking steps.
- Extraction: arithmetic, the first integer in 8 greedy tokens (none = INVALID, counted as a value by the frozen
  design); GSM8K soft thinking (Qwen2.5-7B-Instruct), the number after the last "answer is"; reasoning models, the last
  \boxed{}, else the last number, also in budget-capped rows.

Bug and correction:
- The exploratory reset test compared the decoded 8-token answer texts, not the parsed integers its docstring and the
  frozen arithmetic test use. With integers: H100 0.5B bf16 free 60/128 (was 85), reset 1 (was 2); 7B bf16 free 52
  (was 82), reset 0 (was 2); workstation 0.5B free 88 (was 102), reset 7 (was 14). Every free-against-reset McNemar
  p <= 5e-16 on the H100. Fixed in code with a test; originals kept.

Qualifiers found (receipt docs/pi/layout-answer-audit.json):
- Arithmetic, bf16 soft, 96 problems. 0.5B: log10 T = 0, 0.2, 0.4 -> 54, 8, 0 changed; correct alone/batch 23/22,
  48/51, 90/90. 7B: log10 T = 0.2 -> 53 changed (24 a number against no number, 29 two different numbers), correct
  49/61; log10 T = 0.4 -> 0 changed, 96/96 correct.
- GSM8K Soft Thinking as published (Qwen2.5-7B-Instruct, top 15, T = 0.6, 64 soft steps): 54/256 changed; accuracy
  through this pipeline 22-30 % (greedy chain of thought 89 %); 16 problems ran in identical batches in both layouts
  (short length groups filled from one pool) and none changed; 54/240 among the others; short batches 11/78 and 26/77.
- Reasoning models (thinking budget 1536 steps, answer budget 256 tokens), GSM8K, 256 problems per cell: soft changed
  37, 74, 50 (R1-Qwen-7B, R1-Llama-8B, QwQ-32B); sampled 26, 55, 51. Changes with a budget reached in either layout:
  soft 158 of 161, sampled 132 of 132. Finished within both budgets in both layouts: soft 3 of 480 changed, sampled
  0 of 477. An exploratory rerun with 4096 thinking steps still changed 8.6 % (soft) and 17.2 % (sampled) of
  R1-Qwen-7B's answers on 128 problems; rows that had finished within 1536 steps were reproduced exactly.

## The full report's account of the answer tests (LaTeX, corrected, verbatim)

The same test at the level of answers, with a design frozen before the run, shows what this means for a task. We
asked Qwen2.5-0.5B 96 held-out two-digit additions through a short soft-thinking pipeline: for 32 steps the model is
fed the probability-weighted average of its token embeddings (the input of $F$), each new one appended to a growing
context, and a fixed answer cue then reads the number off greedily. Each problem ran alone and in a batch with three
fixed distractor problems of the same length. Nothing is padded, so the two runs compute the same quantities and differ
only in how the GPU groups its sums for the two matrix shapes, that is, by rounding. In bfloat16 the extracted answer changed with the batch layout in 56\,\% of the problems at $T=1$ and in
8\,\% at $T=10^{0.2}$. At $T=1$ the pipeline answers 23 of the 96 problems correctly alone and 22 in the batch, so
most changes are between wrong answers; at $T=10^{0.4}$, where it answers 90 of the 96 correctly in both layouts, no
answer changed (\repo{docs/pi/layout-answer-audit.json}). A forced greedy baseline with the same budget (32 greedy tokens, then the same cue) produced
identical token sequences in both layouts for every problem. The excess disagreement of soft over greedy decoding,
averaged over the two temperatures, is 32.3 percentage points (95\,\% interval $[26.5,38.1]$; sign-flip $p=10^{-4}$,
the smallest attainable). In float64 no answer changed within the 32 steps. There, the median growth of an injected
kick to the first soft token is $0.033$ nats per step at $T=1$ (range $-0.13$ to $0.24$ over six problems and three
kick sizes), and even a relative kick of $10^{-4}$ changes no answer. The bfloat16 discrepancies therefore do not
require the float64 map's chaos; they are consistent with bfloat16's rounding differences accumulating through
continuous feedback. A discrete argmax absorbs differences smaller than its margin, but not over a long feedback run
whose answers the model holds weakly (``Soft Thinking as published'', below). (The forced greedy baseline's accuracy is zero and not
informative: forced through 32 tokens, the model writes past its answer, and the cue then extracts another number.)
The same frozen test in Qwen2.5-7B, whose soft-thinking pipeline answers 51 to 64\,\% of these problems, replicates
the result. In bfloat16 the batch layout changed the soft-thinking answer in 53 of the 96 problems (55\,\%) at
$T=10^{0.2}$, the temperature at which the soft trace still matters. In 24 of these 53 problems one layout gave no
number (an invalid answer, which the frozen design counts as a value); in the other 29 (30\,\% of the problems) both
gave numbers, and they differ. The pipeline answered 49 of the 96 correctly alone and 61 in the batch; at
$T=10^{0.4}$ it answered all 96 correctly in both layouts, and no answer changed. At this precision even the forced greedy baseline
changed in 10 of the 96, the known batch-variance effect. The excess over greedy decoding is 17.7 points
($[9.3,25.6]$, $p=2\times10^{-4}$). In float64, at that temperature, the 7B's loop amplifies an injected perturbation
(median $+0.034$ nats per step), injected kicks change answers (2 of 18 kicks: six problems and three kick sizes), and
the rounding-level difference between the layouts changed 5\,\% of the answers within the 32 steps.
Two horizons thus matter. Within a short budget, continuous feedback in low precision makes answers depend on the
batch, and where the loop expands, even float64 rounding does. Over longer runs, a chaotic regime amplifies float64
rounding into different readout tokens, on the schedule its Lyapunov exponent sets.

An exploratory causal test locates the low-precision effect in the loop. Following a suggestion of an external review, we reran
the 0.5B pipeline at $T=1$ on 128 newly drawn additions, after a batch-1 reference run of each (exploratory; first on a
workstation GPU, where the native pipeline's answers changed with the layout in 74\,\% of the problems, against
56\,\% on the H100 of the frozen test). In the batch of four, two arms start from the reference's prompt state and
are answered by the same batch-1 decoder. In the free arm the soft phase runs on its own; in the reset arm the
target's cache and logits are replaced by the reference's before every soft step, so that each step runs in the batch
but only one at a time. In bfloat16 the answer changed in 88 of the 128 problems (69\,\%) in the free arm and in 7
(5.5\,\%) in the reset arm, a difference of 63 points (95\,\% interval $[55,72]$; 82 discordant pairs against 1, exact
McNemar $p=2\times10^{-23}$), while accuracy stayed between 23 and 25\,\% in every arm. One batched step moves the
target's logits by a median $1.5\,\%$ of their norm; in the free arm the median difference grows to $0.79$ of it by
step 32, while the reset arm's median per step stays between $1.3$ and $3.1\,\%$. In float32 the free arm's drift grows about 140-fold over the
32 steps but stays near $10^{-4}$, and 3 of the 128 answers changed (none with reset). The test replicates on an H100
and in Qwen2.5-7B. On the H100, in bfloat16, the answer changed in 60 of the 128 problems (47\,\%) in the free arm and
in 1 (0.8\,\%) with reset in the 0.5B model, and in 52 (41\,\%) against none in Qwen2.5-7B at $T=10^{0.2}$
(exact McNemar $p\le5\times10^{-16}$); in the 7B model even float32 answers changed in 33 (26\,\%) against none. In bfloat16 on the H100 most batched steps reproduce the reference bitwise (the reset arm's median step distance is
0), and the free loop amplifies the rare step that does not. No arm's
accuracy differs from the reference's by more than 5 points. Within this budget, the batch layout
changes low-precision answers because the loop carries each step's rounding forward and amplifies it, not because the
last steps alone are fragile; 128 problems per arm do not show that accuracy is unchanged. These counts compare the
extracted numbers, as the frozen test does. Until 19 October 2026 the test compared the decoded answer texts, which
also counts the same number followed by different text, and reported higher counts (102 and 14 on the workstation; 85
and 2, and 82 and 2, on the H100); the corrected and the original summaries are in \repo{docs/rr}.

\paragraph{Soft Thinking as published.} A test frozen before its data (\repo{docs/pi-matrix-confirmatory.md}) ran
Soft Thinking with its published settings on Qwen2.5-7B-Instruct in bfloat16. The settings are concept tokens
restricted to the 15 most probable, $T=0.6$ and 64 soft steps. The test used 256 GSM8K training problems that no
earlier run had scored, on an H100 GPU. Only the batch layout changed, between batches of 8 and of 16 prompts of equal
length without padding, and the answer changed for 21.1\,\% of the problems (54 of 256; 95\,\% interval clustered by
batch $[16.3,26.1]$). This confirms the prediction of more than 10\,\%. The batches deviated from the design in one
respect: the harness filled each group of equal-length prompts from one pool of fillers, and where that pool ran
short, both layouts received the same batch. Rebuilt from the tokenizer, 16 of the 256 problems ran in identical
batches in both layouts, and none of them changed; among the other 240 the rate is 54 of 240 (22.5\,\%). Some batches
were also short (11 of the 78 batches of 8 and 26 of the 77 of 16; \repo{docs/pi/layout-answer-audit.json}). Each secondary prediction compares another arm with this one on
the same 256 problems (paired differences of flip rates, intervals clustered by batch, exact McNemar tests):
\begin{itemize}
  \item replaying the reference run's fed-back tokens in both layouts: 5.9\,\% (difference 15.2 points,
  $[10.4,20.2]$, $p=8\times10^{-9}$); predicted lower, and it is;
  \item 16 soft steps instead of 64: 6.6\,\% (difference 14.5 points, $[10.0,19.0]$, $p=4\times10^{-7}$); predicted
  lower, and it is;
  \item ordinary greedy chain-of-thought decoding: 2.3\,\%, at 89\,\% accuracy (difference 18.8 points,
  $[13.7,23.6]$, $p=10^{-11}$); predicted lower, and it is;
  \item feeding back only the most probable token: 17.6\,\% (difference 3.5 points, $[-2.9,10.3]$, $p=0.35$);
  predicted lower, and the test cannot tell the two apart.
\end{itemize}
The three confirmed differences would survive a correction for the four tests by a wide margin, and the fourth is a
clear null: discrete feedback through the same pipeline flips nearly as many answers as continuous feedback does.
The full vocabulary gave 22.3\,\% (no prediction). Most changes are between wrong answers: through the soft pipeline
this model answers only 22 to 30\,\% of the problems correctly, and the layout changes whether the answer is correct
for 6.2\,\% of the problems. In float64 the same pipeline contracts. On 1312 and 1314 of the 1319 GSM8K test problems
(full vocabulary and top-15), a perturbation of $10^{-10}$ ended below its initial size, and none changed an answer.
These flips are therefore not chaos of the float64 map: the feedback carries bfloat16 rounding differences forward
until they change answers that the model holds only weakly, and it does so for discrete feedback as well.

\paragraph{How the dependence builds up.} An exploratory study read the answers of the same pipeline after every
doubling of the number of soft steps, from 2 to 128. It used 200 GSM8K test problems, two temperatures (0.6 and 1.2),
and six steering strengths along the hostility direction ($\alpha$ from $-0.02$ to $0.05$). The two layouts agree on
90 to 96\,\% of the answers after 2 steps and on 51 to 70\,\% after 128, in every configuration. Among the problems
whose layouts disagree, the median first disagreement comes after 8 steps. Float64 copies of the same configurations,
started $10^{-10}$ apart, separate the two sources of disagreement.
\begin{itemize}
  \item Unsteered or weakly steered ($|\alpha|\le0.02$), the copies change the answer text of at most 1.5\,\% of the problems. The loop
  contracts, so the bfloat16 disagreements are rounding carried forward by the feedback.
  \item At $\alpha=0.05$ they change the answer text of 13.5\,\% of the problems at $T=0.6$ and 27.5\,\% at $T=1.2$, and
  the extracted number of 4.5 and 9.0\,\%. The loop itself is
  then sensitive, but accuracy has fallen to between 0.5 and 2.5\,\% in either layout.
\end{itemize}
In this model, then, the regime in which Soft Thinking is useful is the regime of accumulated rounding, not of chaos
(\repo{docs/pi/h-horizon-summary.json}).

\paragraph{Reasoning models that answer well.} Qwen2.5-7B-Instruct answers few of these problems through the soft
pipeline, so most of its flips are between wrong answers. A second test, also frozen before its data
(\repo{docs/pi-matrix2-confirmatory.md}), used reasoning models, as the Soft Thinking paper did:
DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Llama-8B and QwQ-32B. A row thinks in concept tokens (top 15,
$T=0.6$) until its most probable next token ends the thinking block, or until 1536 steps. It then answers by sampling
discrete tokens, with random numbers shared between the two layouts. The comparison arm thinks with discrete tokens
sampled the same way. On GSM8K problems not used before in this study the batches held 8 or 16 prompts of equal length. On the integer-answer
problems of MATH-500, each problem ran alone or as eight copies of itself. In the first run the tokenizer of
DeepSeek-R1-Distill-Llama-8B loaded wrongly: it dropped the spaces and newlines of every prompt, and the model thought
far less and answered 45\,\% of a check set correctly, against 83\,\% with the correct tokenizer in bfloat16 and
81\,\% in float64 (\repo{docs/pi/pm2x-llama-diag.json}). Both of its cells were rerun with the correct tokenizer,
an amendment after the data. Over the five primary cells, the batch layout changed 27.1\,\% of the soft-thinking
answers (243 of 896; 95\,\% interval $[24.7,29.5]$), confirming the prediction of more than 5\,\%. The prediction
holds in every version of the pooling, including the original run (29.6\,\%) and the run without both cells of this
model (21.9\,\%) (\repo{docs/pi/pm2fix-summary.json}, \repo{docs/pi/pm2fix-math-summary.json}).
\begin{itemize}
  \item All three models changed their GSM8K answers with the layout alone: DeepSeek-R1-Distill-Qwen-7B
  for 14.5\,\% of the problems ($[10.9,18.0]$) at 88 to 90\,\% accuracy, and QwQ-32B for 19.5\,\% ($[16.4,22.3]$) at
  86 to 87\,\%, and DeepSeek-R1-Distill-Llama-8B for 28.9\,\% ($[23.4,34.4]$) at 74 to 77\,\%. The two layouts disagreed
  on correctness for 11.7, 14.5 and 18.0\,\% of the problems.
  \item Discrete sampled thinking also changed many answers. On GSM8K the rates were 10.2, 19.9 and 21.5\,\% for the
  same three models. Only in DeepSeek-R1-Distill-Llama-8B did soft thinking change significantly more, by 7.4 points
  ($[1.9,12.9]$, McNemar $p=0.029$), and it answered correctly 4.7 points less often
  ($[-8.6,-0.8]$, McNemar $p=0.096$). Elsewhere no soft-minus-plain
  difference is significant, which does not show that the two are equal. In these pipelines the effect did not
  require continuous thinking: long sampled feedback in bfloat16 showed it too.
  \item On an A100 instead of an H100, the first model's soft rate was 11.7\,\% (difference 2.7 points,
  $[-1.6,6.6]$), a separate replication on a second GPU type; the difference is not significant, which again does not
  establish equality.
  \item Where the budget cut the reasoning short, accuracy was lower and the layout changed more answers. On MATH-500,
  where half or more of the problems reached the thinking cap, it changed 61\,\% of the answers of
  DeepSeek-R1-Distill-Qwen-7B (44 to 48\,\% correct) and 67\,\% of those of DeepSeek-R1-Distill-Llama-8B (39 to
  42\,\% correct).
  \item Nearly all GSM8K changes came from long runs. Of the 161 answers that the layout changed with soft thinking,
  158 came from problems in which the thinking budget (1536 steps) or the answer budget (256 tokens) was reached in at
  least one layout, and so did all 132 changes with sampled thinking. Among problems that finished within both budgets
  in both layouts, the layout changed 3 of 480 soft answers and none of 477 sampled ones
  (\repo{docs/pi/layout-answer-audit.json}).
\end{itemize}
These runs did not use the Cold Stop rule of the Soft Thinking paper, which ends thinking once the entropy has stayed
below 0.01 nats for 256 steps. An exploratory rerun used it, with a thinking budget of 4096 steps
(\repo{docs/pi/pm2cs-design.md}, \repo{docs/pi/pm2cs-summary.json}). Cold Stop ended thinking in only 6 of the 512
soft rows and in none of the 512 sampled rows; the models nearly always ended it themselves. The layout still changed the answers: 8.6\,\% (soft) and
17.2\,\% (sampled) for DeepSeek-R1-Distill-Qwen-7B on 128 GSM8K problems, 50.0 and 54.7\,\% for it on MATH-500, and
17.2 and 9.4\,\% for QwQ-32B on 64 GSM8K problems. The soft-minus-sampled difference changes sign between these
cells. On the same type of GPU, every row that ended thinking on its own within the original 1536 steps was reproduced
exactly in its saved switch step and answer, so the changes come from the rows the original budget cut short; the DeepSeek-R1-Distill-Qwen-7B GSM8K rerun
ran on a different H100 variant.

(The report's next paragraph, float64 copies of the reasoning pipelines, is omitted here: in R1-Qwen-7B the
copies contract and 2.3 % of answers change; in QwQ-32B and R1-Llama-8B a weak finite-time instability over about a
thousand thinking steps changes 13.5 and 18.4 % of answers on top of the rounding.)

## Questions

1. Magnitudes. Are the corrected rates plausible for bfloat16 batch variance carried forward by feedback, given the
   cited literature on batch invariance and bfloat16 reasoning reproducibility? Which still look suspicious, and why?
2. Artifacts. Is anything missed that could inflate the rates (extraction rules, INVALID counted as a value, budgets,
   offset sampling draws, top-k concept tokens, temperature regimes)? For each serious candidate, name the cheapest
   decisive check and its expected outcome under artifact against mechanism.
3. Construct. Given that changes concentrate where answers are fragile (low accuracy; truncated reasoning) and vanish
   where answers are solid, what claim does the evidence support? Does the corrected paragraph claim more or less than
   that, and should it lead with a different framing?
4. Reasoning models. Is reporting the frozen rates (14.5-28.9 %) with the budget split as written fair? What is the
   right one-sentence summary, given that a 4096-step rerun still changed 8.6 and 17.2 %?
5. Wording. Flag any sentence of the paragraph that overstates, understates or would confuse a dynamics reader.
6. Triage. What must change before posting, and what can wait for the journal version?
