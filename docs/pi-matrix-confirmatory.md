# Confirmatory soft-thinking matrix (run PM): frozen before any data

16 October 2026, 11:09 UTC. This design comes from consult 11 (docs/consults/2026-10-16-practical-utility), as
adjudicated there. It was written after the exploratory runs E1 (development: the first 400 GSM8K test problems), E4,
PIF64 and the first two configurations of H. At this commit, no run has scored any problem used below.

## Question

Soft Thinking as published uses concept-token top-15 at T = 0.6. Run on Qwen2.5-7B-Instruct in bfloat16, does it give
answers that depend on the batch layout at a practically concerning rate on fresh problems? And how do the vocabulary
filter, the number of soft steps, discrete feedback, ordinary decoding and a clamped feedback pathway compare?

## Design

- **Model:** Qwen2.5-7B-Instruct, snapshot a09a3545 (the harness's pinned path), bfloat16.
- **Hardware and software:** the local NVIDIA RTX 2000 Ada (16 GiB) with the software of E1 (torch 2.14.1+cu130,
  CUDA 13.0), so that development and test share the kernels.
- **Code:** tools/pi_harness.py at commit 65f2bfba, and tools/pi_matrix_analyze.py at the commit of this document.
  The harness's default soft and hard paths compute as in E1 and E4; the commits since then only added options
  (saved and replayed concept tokens, mid-run readouts, steering, device and dtype).
- **Problems:** the first 256 problems of the GSM8K train split, in the dataset's order (`--split train --limit 256`).
  - None has been scored in any run.
  - Some train problems served as unscored batch fillers in E1, E4 and H.
  - Here, fillers come from the test split and are not scored.
- **Layouts:** batches of 8 and of 16 prompts with identical token length and no padding, as in E1. A row's inputs are
  identical in both layouts; only the matrix shapes differ.
- **Arms.** Every arm reads the answer after the cue with 12 greedy answer tokens, and every arm uses the harness's
  default seed.

| arm | options |
|---|---|
| top15-s64 (primary) | `--mode soft --top-n 15 --temperature 0.6 --soft-steps 64 --save-embeddings` |
| replay | as top15-s64, with `--replay-from` the top15-s64 run |
| greedy | `--mode hard --temperature 0 --max-new 320` (ordinary chain of thought) |
| top1-s64 | `--mode soft --top-n 1 --temperature 0.6 --soft-steps 64` (discrete feedback) |
| full-s64 | `--mode soft --top-n 0 --temperature 0.6 --soft-steps 64` |
| top15-s16 | `--mode soft --top-n 15 --temperature 0.6 --soft-steps 16` |
| full-s16 | `--mode soft --top-n 0 --temperature 0.6 --soft-steps 16` |

- **Replay arm.** Both layouts are fed the concept tokens that the reference run saved, instead of computing them. The
  reference is the top15-s64 run's 16-problem layout, which is the first file in the harness's sorted order. This
  clamps the fed-back pathway. The transformer's batch-dependent arithmetic, the growing cache and the answer decoding
  still run in each layout.
- **Order:** the arms run in the order of the table, on the local GPU, after run H finishes (tools/pi_matrix_run.sh at
  the commit of this document).

## Endpoints and predictions

**Answer flip.** The extracted answers of the two layouts differ. A missing number counts as its own value.

**Clusters.** The batches of the 16-problem layout, which share one layout intervention.

**Primary.** The flip rate of top15-s64 over the 256 problems.
- Prediction: above 10 %, with the lower end of the cluster-bootstrap 95 % interval above 5 %.
- Development: 25.75 % (103 of 400, E1).

**Secondary.** Each contrast is against top15-s64, paired by problem. Development values are given where they exist.
- full-s64: no prediction of a difference (development: +0.5 points).
- top1-s64: top15-s64 flips more (development: +9.5 points).
- greedy: top15-s64 flips more (development: +19.75 points, E4).
- replay: top15-s64 flips more than its replay, because the fed-back pathway carries the difference. This is
  predicted from the causal test of the short addition pipeline.
- top15-s16: top15-s64 flips more than with 16 steps. In E1 the median first divergence of the two layouts'
  trajectories came at step 19.
- full-s16: reported with the other arms, without a prediction.

A secondary prediction holds when its 95 % cluster-bootstrap interval excludes zero in the predicted direction. The
secondaries are not corrected for multiplicity, and they are reported as such.

**Descriptive.** For each arm and layout: accuracy, the share of problems with exactly one layout correct, missing
answers and the first step at which the trajectories diverge. E1's flips fell mostly between wrong answers: accuracy
21 %, with exactly one layout correct in 5.8 % of the problems. This test reports how many flips change correctness.

## Analysis and scoring

- **Tool:** tools/pi_matrix_analyze.py, with arms named as above. Cluster bootstrap with 10,000 draws and seed
  20261016; exact McNemar tests for the paired contrasts.
- **Scoring:** results are used as computed.
  - A run that fails is rerun once with the same settings.
  - An arm that cannot be completed is reported as missing; if it is the primary arm, it counts as a failed prediction.
  - The outcome is reported in the paper whether the predictions pass or fail.

## Not part of this test

- A float64 check on these problems. PIF64 measured float64 sensitivity on the test split: top-15 and the full
  vocabulary at 64 steps, at T = 0.6.
- Steering and other temperatures. Run H covers them on the test split.

## Amendment 1 (16 October 2026, 16:33 UTC, before any data): hardware

The human asked to run the test on RunPod, because the local GPU and CPU are busy with other runs. No run of this test
had started: the local queue was waiting for run H and was stopped before PM began, and no PM output exists.

- **Hardware:** one NVIDIA H100 (any form factor) or H200 on RunPod, recorded in the run's hardware log, instead of the
  local RTX 2000 Ada.
- **Software:** torch 2.14.1 built for CUDA 12.6 (the pods' build) instead of CUDA 13.0; transformers 5.19.0 as before;
  pandas 3.0.6 and pyarrow 26.0.0 as locally.
- **Code changes, which do not change what is computed:**
  - tools/pi_matrix_run.sh takes the Python interpreter from $PY, with the old default.
  - The runner job set `pm` downloads the pinned Qwen2.5-7B-Instruct and the pinned GSM8K snapshot (740312ad, as
    used locally), and links the harness's fixed cache path to the pod's data directory.
- **Consequence:** batch-layout differences depend on the kernels, so the development flip rate (25.75 %, E1, RTX 2000
  Ada) is a reference, not a calibration, for this hardware. Problems, arms, order, endpoints, predictions and analysis
  are unchanged.
