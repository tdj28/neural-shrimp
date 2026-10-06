# EXP-008 learned-representation and action-selection test

**Release status: private working diagnostic draft; PR #7 is unmerged.**
Release candidate `1d12ebeda0400c179e54dd36e647a835f7e0d91d` is not a fully
reproduced release. Local frozen verification passed, but Linux PR run
34803148459 fails coefficient `[1]` (absolute difference 3.68269315e-8 at
combined `rtol=1e-7, atol=1e-9`), while push run 34803133930 stops earlier at
shared feature `[34]` (1.39784018e-14 at `rtol=1e-12, atol=1e-14`). Exact
[PR](../docs/EXP-008-linux-pr-failure.txt) and
[push](../docs/EXP-008-linux-push-failure.txt) logs remain evidence. The PR
reaches fitting after metric checks; the push does not complete the
feature/waveform loop.

Later [PR 34804593644](../docs/EXP-008-platform-linux-pr.json) and
[push 34804589708](../docs/EXP-008-platform-linux-push.json) diagnostics at
`bb4ecd25cf5c025064e7bd200b6acd898953d0e5` pass exact raw bindings and preserve
all three penalties and all 288 learned actions using **saved-feature refits**.
Maximum prediction differences are 1.3874261184e-9 and 1.9615957192e-9,
versus minimum saved action-score gap 3.7010270022e-6. Saved-model score replay
passes `rtol=1e-12, atol=1e-14`. The PR retains two shared-feature failures
and 8 baseline plus 9 shuffled coefficient failures; the push retains zero
shared-feature failures and 10 plus 10 coefficient failures. Geometry
coefficients pass in both. See the [platform status](../docs/EXP-008-platform-status.md)
for scope and retained report bindings. These diagnostics do not repeat
waveform metric checks, refit on completely regenerated inputs or integrate
trajectories. General portability remains unresolved; the strict gate has
not passed and is not replaced or bypassed by diagnostic completion.

`EXP-008/attempt-001` retains **10,650 receipt-listed artifacts plus the
receipt**, totaling 728,224,123 artifact bytes. Source
`3c5604497ec57b19d62fbb595da7b190102f4b33` was pushed before outcomes and is
preserved as `archive/EXP-008-source-v1`. Manifest SHA-256:
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`.
Receipt SHA-256:
`ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb`.
The recorded start is 2026-09-14 at 02:57:26 UTC (13 September locally).

All 24 origins qualify, all 3,456 previews complete, and all 7,008 target
records complete: 6,912 nonzero candidate evaluations and 96 shams, including
both numerical steps. The model packet uses development labels only. Its
input seal, test-feature and decision seals, and target-start record preserve
the execution order. Every test policy has the same eighteen candidate actions
and previews; the task uses all activations over `75<t<=300` with one constant
phase shift.

The secondary observed prediction MSE is 17.17% lower with geometry
(0.018753462 versus 0.022640656). The primary action benefit remains
unresolved: 0.000857628 [-0.007701467, 0.006545434]. Minimum-preview RMSE is
0.009958855 versus geometry 0.016966044, 41.30% lower in this sample. These
results and all twelve blocks remain in
[information-paper-summary.json](information-paper-summary.json).
The first declared example selects candidate 4 under every policy; its
coincident curves are retained.

The frozen verifier and independent raw/feature/outcome checks passed locally.
The independent local SVD model audit retains a failed coefficient comparison: eight
baseline and eight shuffled coefficients exceed the unchanged tolerance,
although chosen penalties and all actions agree. All-candidate numerical-step
differences are also not uniformly small (maximum RMSE change 0.476851).
Neither issue is erased by the steadier selected actions or successful source
local replay. See the [raw audit](../docs/EXP-008-results-audit.md),
[model audit](../docs/EXP-008-model-results-audit.md), and
[figure receipts](../paper/figures/hopfield-information-figure-provenance.json).

```sh
python tools/run_hopfield_information.py verify --output evidence/EXP-008/attempt-001
python tools/audit_hopfield_information.py --out /tmp/EXP-008-independent-check.json
python tools/audit_information_models.py evidence/EXP-008/attempt-001 \
  --out /tmp/EXP-008-model-independent.json
python tools/plot_hopfield_information.py evidence/EXP-008/attempt-001 \
  --destination paper/figures --check
```

Use the repository's locked environment for these commands. Audit outputs
must stay outside the immutable attempt. Figure checks verify existing output
hashes and re-derived arrays, not cross-platform render-byte identity. The
model audit's retained coefficient failures must be read alongside its other
passing checks; a completed report does not mean every comparison passed.

Collection and summary took 162.352 seconds before final verification.
Sensitivity used 23.770 seconds and fitting/seal writing 0.079 seconds within
that clock; overlapping worker-call sums are separate measures. The original
map's 140.54 seconds is an additional historical cost. No paid compute or
review was used. Numerical refinements or new shortlist policies require new
prospective packets and cannot overwrite this attempt.

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
