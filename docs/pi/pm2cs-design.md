# PM2CS design note (exploratory; fixed 17 Oct 2026 before any data)

**Question.** Do PM2's layout flips persist when thinking ends with the published Cold Stop rule and a longer
thinking budget?

**Status.** Exploratory: no prediction is tested. It follows PM2 (docs/pi-matrix2-confirmatory.md) and changes only
the end of thinking.

**What changes from PM2.**
- **Cold Stop.** The rule of Soft Thinking (Zhang et al. 2025, arXiv 2505.15778), as in its code (eric-ai-lab/Soft-Thinking at b97407c6, sglang `schedule_batch.update_topk_info`).
  - During thinking, count consecutive steps whose entropy of softmax(z / T) over the full vocabulary is below 0.01 nats. The counter resets on any step at or above it.
  - When the count reaches 256, that step's token becomes the end-of-thinking token.
  - These are the published settings (`early_stopping_entropy_threshold` 0.01, `early_stopping_length_threshold` 256).
  - Applied in both arms, so the arms differ only in the fed-back thinking input.
- **Thinking budget.** 4096 (PM2: 1536). The answer budget stays at 256.
- **Option.** `--cold-stop-entropy 0.01 --cold-stop-length 256 --think-budget 4096` in tools/pi_reason.py. The defaults (Cold Stop off, 1536) reproduce PM2.

**What stays as in PM2.**
- Soft arm: top-15 concept tokens, T 0.6.
- Plain arm: top-p 0.95 sampling.
- Answer phase: common random numbers.
- Prompts, layouts, problem selection, seed and bfloat16.

**Cells.**
| model | task | problems | layouts | GPU |
|---|---|---|---|---|
| R1-Distill-Qwen-7B | GSM8K train after 256, full groups of 16 | 128 (the first 128 of PM2's 256) | 8 / 16 | H100 |
| R1-Distill-Qwen-7B | MATH-500 integers | the first 64 (as PM2) | 1 / 8 copies | H100 |
| QwQ-32B | GSM8K, as above | 64 (the first 64 of PM2's 256) | 8 / 16 | H200 |

**Endpoints (as PM2, from tools/pi_matrix2_analyze.py).** Per cell and arm:
- the flip rate (extracted answer differs between layouts), with a cluster bootstrap 95 % interval (clusters as in PM2);
- accuracy in each layout;
- the share of problems with exactly one layout correct;
- soft minus plain flips, paired, with McNemar p.

**Also reported.**
- How thinking ended: the model's own end-of-thinking token, Cold Stop, or the budget.
- The switch step.
- A comparison with the PM2 rows on the same problems.

**Reading.**
- If the flip rates stay near PM2's (R1-Qwen-7B GSM8K 14.5 %, QwQ 19.5 %, R1-Qwen-7B MATH 60.9 %), the flips do not depend on how thinking ends or on PM2's budget.
- A large drop on MATH would mean that PM2's budget-limited rows drove the MATH flips there.
- Nothing here is confirmatory. Intervals are descriptive.
