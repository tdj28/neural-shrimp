## 2026-10-15 07:05 UTC: run (frozen design): REPRO-T, answers of continuous-feedback inference under batch-layout changes (consult 9, direction 1)

Design frozen in docs/repro-task-test.md (pushed with this entry, before any held-out run); problems in docs/repro-task-problems.json; tool tools/repro_task.py.
- **Local CPU smoke test** (2 development problems and 1 held-out problem, 4 soft steps): the pipeline and scoring work, and every answer is correct.
  - Batch-layout distances after 4 steps: up to 5e-3 in bf16 and 6e-15 in float64; no answer changed.
  - Reruns are identical.
- **Pod.** One H100 class, job set reprotask: all 8 development and 96 held-out problems, both precisions, log10 T 0.0, 0.2 (primary) and 0.4 (secondary), greedy baseline, mechanism subset.


## 2026-10-15 07:32 UTC: REPRO-T result (frozen design): in bfloat16 the batch layout changes the answer of soft-thinking inference in 56 % of problems at T = 1, against none for greedy decoding (Delta = 32.3 points); in float64 no answer changes within the 32-step budget, and the bfloat16 effect is not the float64 map's chaos

Pod 9oreujzv1m3s7r (SECURE H100 NVL; runner rc 0, hash verified, terminated; 1548 s; commit a9c411ff, the commit that froze docs/repro-task-test.md). Output repro/t05/runs.jsonl: 832 pipeline records and 54 mechanism records. Inline analysis in this session, following the frozen endpoints.
- **Primary endpoint, bfloat16:** Delta = 100 x mean_i [ (d_soft(0.0) + d_soft(0.2)) / 2 - d_greedy ] = **32.3 percentage points**, 95 % bootstrap interval [26.6, 38.0], sign-flip p = 1e-4 (10,000; the smallest attainable), n = 96 held-out problems. **Positive.**
- **Per condition** (held-out; canonical answers differing between layouts; accuracy single / batch):

| precision | mode | log10 T | answers differ | accuracy | soft-token distance at step 1 / step 32 (median) |
|---|---|---|---|---|---|
| bfloat16 | soft | 0.0 | 56.2 % | 0.240 / 0.229 | 6.1e-7 / 0.22 |
| bfloat16 | soft | 0.2 | 8.3 % | 0.500 / 0.531 | 2.5e-6 / 0.013 |
| bfloat16 | soft | 0.4 | 0 % | 0.938 / 0.938 | 7.9e-7 / 0.004 |
| bfloat16 | greedy | | 0 % (token sequences identical in 96 / 96) | 0 / 0 | |
| float64 | soft | 0.0 | 0 % | 0.312 / 0.312 | 4.7e-15 / 1.3e-13 (max 1.7e-9) |
| float64 | soft | 0.2 | 0 % | 0.510 / 0.510 | 6.4e-15 / 5.0e-15 |
| float64 | soft | 0.4 | 0 % | 0.958 / 0.958 | 3.5e-15 / 1.7e-15 |
| float64 | greedy | | 0 % | 0 / 0 | |

- **Determinism.** Every single-layout rerun is identical.
- **INVALID rates:** 0-1 % for soft decoding; 8.3 % for bfloat16 greedy.
- **Greedy baseline, disclosed.** Its accuracy is 0 because the frozen budget forces 32 greedy tokens. The model ends its answer within a few tokens and then writes an invented new turn, so the cue extracts another number. Its reproducibility comparison stays valid (identical 40-token sequences in both layouts at every problem); its accuracy does not measure ordinary greedy inference.
- **Mechanism subset** (float64, single layout; the first soft token displaced by delta |x|; 6 problems; median growth of the distance over 31 steps):
  - log10 T 0.0: +0.03 nats per step;
  - log10 T 0.2: -0.14;
  - log10 T 0.4: -0.10.
  - No answer changed at any delta (1e-12, 1e-8, 1e-4).
- **Interpretation** (frozen rules):
  - bfloat16: positive with growth of the layout distance (6e-7 to 0.22 at T = 1), so operational sensitivity for this implementation and budget is established.
  - Attribution: the float64 map amplifies perturbations only mildly over the same 32 steps, and a 1e-4 kick changes no answer. The bfloat16 discrepancies therefore are not the float64 map's chaos. They arise from bfloat16 rounding differences accumulating through the continuous feedback, which greedy decoding's discrete argmax absorbs.
  - float64: null at this budget (short-horizon robustness). This does not contradict REPRO1, where chaotic float64 loops diverge after a median of 102 steps.
- **Caveats.**
  - 0.5B model; one task family; a soft-thinking pipeline without top-k (not the published method's exact settings).
  - Accuracy rises with temperature here (soft tokens at T = 2.5 are nearly uninformative, and the model then answers from the question). Answer changes are therefore largest where the soft trace carries the most information.


