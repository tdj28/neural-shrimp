# Run PM2: Soft Thinking with reasoning models under two batch layouts (frozen before any data)

16 October 2026, written before any run. Run PM (docs/pi-matrix-confirmatory.md) confirmed that Soft Thinking with
its published settings changes 21 % of answers with the batch layout alone in Qwen2.5-7B-Instruct. That model,
however, answers only 22 to 30 % of the problems correctly through the pipeline. The human asked to fix the test's
weaknesses: one model, one GPU type, one task, and a model that answers poorly. The Soft Thinking paper used reasoning
models, and this run uses three of them, on two tasks and two GPU types.

## Design

- **Models (bfloat16), with pinned revisions:**
  - DeepSeek-R1-Distill-Qwen-7B, 916b56a4;
  - DeepSeek-R1-Distill-Llama-8B, 6a6f4aa4 (a second family);
  - QwQ-32B, 976055f8 (the Soft Thinking paper's main model).
- **Pipeline:** tools/pi_reason.py at the commit of this document.
  - The user message is the problem followed by "Please reason step by step, and put your final answer within
    \boxed{}.", under the model's chat template, which opens the thinking block.
  - A row thinks until its most probable next token is the end-of-thinking token, or until 1536 thinking steps, when
    that token is inserted.
  - It then answers with discrete tokens, sampled from the top-p 0.95 nucleus at T = 0.6 with common random numbers,
    until the end-of-sequence token or 256 answer tokens.
  - The answer is the last \boxed{} content (integers compared as integers), else the last number.
  - Rows run in lockstep without padding, so a row's inputs are identical in both layouts.
- **Arms:**
  - **soft:** thinking with concept tokens, the probability-weighted embedding of the 15 most probable tokens of
    softmax(z/0.6), renormalized, as published.
  - **plain:** thinking with discrete tokens sampled as in the answer phase.
  - The answer phase is identical in both arms.
- **Tasks:**
  - **GSM8K:** training problems after the first 256 (those were PM's), in dataset order, taken into complete groups
    of 16 prompts of identical token length, until 256 problems. Layouts are batches of 8 and 16; each 16-problem group
    is one cluster.
  - **MATH-500:** the first 64 problems, in dataset order, whose reference answer is an integer. Layouts are alone (1)
    and as 8 copies of itself (8); each problem is one cluster. MATH prompts rarely share a token length, so copies
    make the batches.
- **Hardware:** RunPod. One pod per model, each running every cell of that model; a fourth pod repeats one cell on a
  second GPU type.

| pod | model | GPU | tasks |
|---|---|---|---|
| A | DeepSeek-R1-Distill-Qwen-7B | H100 | GSM8K, MATH-500 |
| B | DeepSeek-R1-Distill-Llama-8B | H100 | GSM8K, MATH-500 |
| C | QwQ-32B | H200 | GSM8K |
| D | DeepSeek-R1-Distill-Qwen-7B | A100 80GB | GSM8K (replication on a second GPU type) |

Order on each pod: the soft arms of its tasks first (in the table's order), then the plain arms, because the primary
endpoint needs only the soft arms. A pod whose time budget runs out skips the remaining arms and logs that. An arm cut
midway keeps the problems completed in both layouts.

## Endpoints and predictions

**Answer flip:** the extracted answers of the two layouts differ; a missing answer counts as its own value.

**Primary:** the soft arms' flip rate pooled over the five primary cells, which are A-GSM8K, A-MATH, B-GSM8K, B-MATH
and C-GSM8K. Problems are the units, and the bootstrap resamples clusters within each cell.
- **Prediction:** above 5 %, with the lower end of the 95 % interval above 2 %, i.e. a practically relevant rate in
  reasoning models that answer well.
- A cell that cannot be completed is reported as missing. The pooled estimate then uses the completed cells and says
  so.

**Reported for every cell, without predictions:**
- the soft and plain flip rates with intervals;
- accuracy in each layout;
- the share of problems with exactly one layout correct, i.e. how often the layout changes correctness;
- soft minus plain flip rate (paired, exact McNemar);
- soft minus plain accuracy in the larger layout (paired, exact McNemar);
- missing answers and forced ends of thinking.

**Second GPU type:** the A100 replication (D) against A-GSM8K. The soft and plain flip-rate differences are reported
with intervals, without a prediction.

## Analysis and scoring

- **Tool:** tools/pi_matrix2_analyze.py at the commit of this document. Cluster bootstrap with 10,000 draws and seed
  20261016.
- **Scoring:** results are used as computed.
  - A failed pod is relaunched once with the same settings.
  - The outcome is reported in the paper whether the prediction holds or fails.
  - Every per-cell estimate is reported with its interval, without a count of cells that pass.

## Cost

About $30 expected and a hard ceiling of $45 over the four pods, with each pod's limit set by the orchestrator guard.
Amounts and approvals are in the private ledger. Human approval: "Full plan now", 16 October 2026.
