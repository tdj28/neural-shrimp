# Saved-checkpoint inference precision pilot — 10 October 2026

The user renewed every research track. This is an exploratory inference unit,
separate from the failed training-forecast gate. All six existing final800-step
calibration checkpoints (28151/28152 x .8/1.6/2.5) are included. No new training,
protected seeds, assessment or validation outcomes; no controller. Use the eight
already consumed32-character training prefixes, saved batch ef17075eafaab531914245efadf43024aa51f135ed97bde6f3f4ecf7ac68d92c.
Exact metadata equals calibration data; full calibration manifest
a8a4166cb0b03c5f8b72b0a8039dfb99acc0e4ad1fc12a53d4d3371f0b0d4bff is verified.
The source corpus is not opened. The unchanged trained TinyTransformer15776
source hash is27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8.

Freeze each float32 parameter vector; cast those same rounded values to float64.
For32 greedy steps per prefix, sliding context32 and reset positional indices,
measure float32 twice (exact repeat control), float64 on the same float32 history,
and float64 on its own history. Fixed CPU1, deterministic algorithms, no dropout,
cache or sampling. Store every last-position logit vector, token choice and
same-history mask. The measured top-two float32 margin>twice max absolute pairwise
logit difference is an empirical sufficient screen for this pair, not an interval
rounding bound. Record changed matched-history decisions, free-rollout mismatch
counts and first divergences, even if all are zero. This answers numerical
inference sensitivity only, not chaos, training causation, generalization/task
quality, or shrimp-specific utility. Training precision and checkpoint weights
are not confounded; both forward computations use the same rounded weights.

Commit source/card/tests and peer skim before one run at
/data2/neural-shrimp-data/explore/saved-lm-inference/attempt-001. Full-shape synthetic
profile before opening targets:6 forwards per dtype, initialized seed28155;
2*max(timing)*768+5<90 seconds required. Exactly768 target forwards,780 total cap.
CPU1/nice10;90s worker/105+5 launcher,2GiB cooperative RSS,16MiB worker+1MiB
launcher, no GPU or paid calls. Owned process group cleanup; all raw sources,
failed attempts and manifests retained read-only, SHA-bound receipt, independent
saved-array result audit. Do not rerun a failed profile or alter precision/length
without a dated amendment. No headline confirmation or heldout access authorized.

Peer source skim required immediate durable storage of every returned forward
(including nonfinite outputs) before qualification and per-call case/step/mode
attempt tags. Adopted before target; profile timing includes retention. A partial
case remains reconstructable from prefix and previously returned logits even
if a repeat/nonfinite/deadline gate fails. Bind all transitive package imports
(__init__, controls, models). Original source e80b1a9d acquired no target data.
