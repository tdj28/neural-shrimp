# EXP-007 new-start waveform policy test

`EXP-007/attempt-001` retains 7,149 receipt-listed files plus its receipt:
374,438,288 bytes of recorded artifacts. Source commit
`ac098c3e3833daaf27c7dc2869d45e8b7085f948` was pushed before outcomes and is
preserved as `archive/EXP-007-source-v1`. Manifest SHA256:
`573d817836d6c3271e685a45988be660743c104b47d96a5036414db4e6893ac7`.
Receipt SHA256:
`5a944722af26b9406bf2ba881067305491a115c41fe44a282ad2e9cb72282d23`.

All 12 baseline attempts qualified; 4,032 previews and 2,784 target records
completed. The primary interval is 75.1–300; previews use 0–75. Collection
and summary took 137.509 seconds, excluding verification. The complete attempt
passed frozen replay and an independent raw-data audit. All five pooled
primary contrasts remain unresolved. No paid compute or paid review.

```sh
python tools/run_hopfield_holdout.py verify --output evidence/EXP-007/attempt-001
python tools/plot_hopfield_holdout.py evidence/EXP-007/attempt-001 --destination paper/figures --check
```

# Numerical evidence

The primary current result is an actual neural shrimp-shaped parameter window,
not the earlier exploratory training pilot. All raw attempts remain immutable.

EXP-006 now adds a complete, explicitly defined waveform-preservation task and
local periodic-orbit continuation. It supports a dose-specific comparison,
not a general geometry advantage or trained-model result.

- `EXP-004/attempt-001`: 39,523 grid/transect settings of the published three-neuron
  recurrent tanh network, selected step refinements, three independently verified
  attracting cycles, and complete neuron/activation/perturbation traces.
- `hopfield-summary.json`: derived counts and representative measurements,
  checked by `python tools/summarize_hopfield.py`.
- `EXP-005/attempt-001`: eight independent longer-horizon runs at four fixed
  controls, with two steps each. Both distant controls remain finite-horizon
  irregular; the core and failed near-edge comparison are certified nine-peak
  cycles. All 21 receipt-bound artifacts and the receipt are retained exactly.
  Source `01209371d541aaa9d072a1addb3960aa3a74efd3` was pushed before execution.
  This separate follow-up does not replace the EXP-004 exterior failure.
- `EXP-006/attempt-001`: all 4,320 planned trial-step records at two numerical
  steps, all 12 eligible attempted anchors, certified baselines, local
  sensitivity arrays, short activation waveforms, independent long assays,
  and four continuation branches. All 4,472 receipt-bound artifacts are retained.
  Source `ddb568e6cd67ef2cfbdb45e05710cd86025ef548` was pushed before execution.
  Receipt SHA-256 is
  `cdcfc91b9ce8a444a5007e1eda35a1ed34d492810ca1a152598754df27467ac2`.

At primary step 0.005 and radius 0.016, atlas RMSE is 0.301672 versus 0.466653
for the declared symmetric local comparator; the descriptive paired-benefit
interval is [0.013853, 0.326023]. Both smallest radii favor local sensitivity
at every anchor, and the prespecified pooled comparison is unresolved:
0.025738 [−0.027862, 0.083081]. The `tangent_plus` diagnostic has lower mean error
than the atlas at every dose, including 0.260795 at the largest radius. These
limits accompany the positive comparison rather than disappearing from summaries.

The short error window is the first 300 time units after a weight change;
nine-peak retention is measured independently after 10,000 settling plus 2,000
measurement units. Eventual retention does not guarantee immediate waveform
preservation. The frozen `anchor-08` illustration remains unchanged and favors
both local signs, although every shown arm eventually retains nine peaks.

Continuation retains 196 node records, including 122 unstable records and five
failed correctors; repeated seeds and overlapping branches are included.
Geometric checkpoint agreement is 27/28. One unchanged 1e-8 closure threshold
fails at 1.0591e-8. The failed checkpoint, candidate crossings, reversals,
and all stopping reasons remain in the raw branch files. They are not an exact
two-dimensional boundary certificate.

See [utility audit](../docs/EXP-006-utility-results-audit.md),
[continuation audit](../docs/EXP-006-continuation-results-audit.md),
[protocol](../docs/EXP-006-protocol.md), and
[all four new figure receipts](../paper/figures/hopfield-drift-figure-provenance.json).
The 423.866-second collection-and-analysis clock excludes final verification.
Local sensitivity took 26.627 seconds within that total; the old atlas's
140.54 seconds is separate. No paid resources or paid Pro review were used.

EXP-004 source `b670086775bbc07e115de730b8fcb30ca41cce6b` was pushed before
execution. The archived attempt retains all 475 receipt-bound artifacts and
its original receipt byte-for-byte. The untracked platform-specific compiled
library is reproducible from the frozen source; its build metadata and hash
are retained in `build.json`. The receipt's `manifest_sha256` binds the source
plan file; `manifest.json` is the same plan reserialized and has its own artifact
hash. No numerical values were changed while packaging evidence.

The scan ran locally on four CPU workers. No paid compute or paid review was
used. Parameter maps are finite-horizon candidate classifications; selected
numerical orbit certificates do not certify every cell or exact boundaries.

# Initial private exploratory pilots

These raw receipts preserve every planned trial, including divergent runs. The
full executable code and plans were pushed at
`bd3d733b51e9f1ae9247ba712633aca023e7afac` before all three pilots. That source
commit is retained by implementation pull request #1. The implementation was
subsequently squash-merged; its main-branch SHA is therefore different.
The archival tag `archive/pilot-source-v1` was created after outcomes to retain
the original source commit; the tag's creation is not the prospective freeze.

This is a private prospective record, not a formal preregistration or public
confirmation. Raw artifacts are immutable. Results are small local-CPU pilots,
not a completed search for neural shrimps or a demonstration of their utility.

- `EXP-001/attempt-001`: complete Hénon control parameter scan.
- `EXP-002/attempt-001`: complete tiny neural training parameter scan.
- `EXP-003/attempt-001`: complete four-policy utility pilot.
- `reports/`: descriptive reports generated by the frozen implementation.
- `pilot-summary.json`: cross-run manuscript counts, source and receipt hashes.

Recompute integrity and every empirical manuscript count:

```sh
uv run --frozen python tools/summarize_pilots.py
```

The utility summary's per-seed lists follow each policy's trial-ID order. They
are **not** positionally matched between policies. Pair raw records by
`(seed, lr_hidden, lr_output)`; the verifier above does so and checks the full
fixed/recurrence state arrays. No successful-only utility mean is reported when
divergent runs are present.

No RunPod resource or paid review was used. The total new paid spend was $0.
