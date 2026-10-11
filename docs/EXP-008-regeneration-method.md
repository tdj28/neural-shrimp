# EXP-008 saved-trajectory regeneration diagnostic

14 September 2026. This is a new, post-run development diagnostic of already
observed EXP-008 evidence. Its implementation and this method must be reviewed,
tested, committed and pushed before its first research execution. No research
execution result is claimed by this document. The original consumed attempt,
source, numerical tolerances and strict verifier remain unchanged.

The earlier Linux diagnostic refitted models on saved feature matrices and
rebuilt shared features using saved preview metrics. The new auditor
`tools/audit_information_regeneration.py` instead re-evaluates the complete
saved-trajectory chain: sensitivity readouts and Grams, preview metrics,
shared and geometry features, development labels, three development-only
ridge fits, and test choices and waveform outcomes. It reuses the
receipt-bound original numerical algorithms. It is not an independent
algorithm implementation or a new integration experiment.

## Fixed inclusion and numerical meaning

All 24 saved baseline certificates, 48 phase contexts, 3,456 previews,
7,008 target/sham records at both steps and all 288 learned decisions are
required. Missing, duplicated, malformed, nonfinite or unavailable cases
remain failures. There is no subset option, success-only fit, saved-label
fallback or replacement of a failed case. A missing development case prevents
fitting; a missing target/sham prevents a complete primary comparison.

Before numerical work, require the original receipt SHA-256
`ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb`,
manifest SHA-256
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`,
source `3c5604497ec57b19d62fbb595da7b190102f4b33`, every receipt artifact's
size/hash, every frozen source/input hash, exact census identities and the
model/decision/target seals. Hash failures are enumerated before stopping
dependent scientific work. No unbound evidence is fitted.

Sensitivity activation derivatives are recalculated from the saved neuron
states and state-sensitivity arrays. Activation velocity is recalculated by
evaluating the original vector field at those states. No ODE solver is called.
The tail remains `times > 75`; its regenerated Gram and minimum direction
are compared with both saved NPZ values and metadata using the existing
`rtol=1e-12, atol=1e-14` arithmetic rule.

The actual interventions remain those that generated the stored trajectories.
As in the frozen verifier, candidate directions are reconstructed from the
saved local direction and the original fixed candidate rule, then exactly
checked against raw deltas, weights and initial states. Fresh minimum
directions are additionally reported in full, with exact equality and maximum
difference. Replacing a saved action by a slightly changed fresh direction
would require a new trajectory; the auditor does not make that substitution.

Every preview and target metric receives a fresh, original 128-bin global
phase scan plus bounded scalar optimization. Preview samples remain `0..75`
and target samples `75 < t <= 300`; the original reference, phase epoch,
period, weighting and single constant phase are retained. The freshly
optimized metrics feed feature generation and model labels. In parallel,
the saved phase is re-evaluated under the frozen verifier's exact semantics:
all saved RMS values use `rtol=1e-12, atol=1e-14`, equivalent minimizers use
the existing interpolation-slope optimizer bound, and positive optimizer
evaluation counts are recorded without demanding exact count equality.
Thus fresh phases are not silently substituted in the original evidence.

Shared features use regenerated preview metrics and sensitivity Grams.
Geometry features are recalculated from the original receipt-bound atlas.
The same separately indexed cyclic donor shift five supplies shuffled
geometry. No test label enters fitting. Original ridge/CV algorithms,
penalties and unclipped predictions are retained. Original versus regenerated
model comparisons use the existing `rtol=1e-7, atol=1e-9` rule; changed input
and packet hashes are reported as provenance, not treated as an impossible
exact-equality requirement for changed numerical inputs.

The report distinguishes saved-model/saved-input decision replay,
saved-model/regenerated-input choices, and regenerated-model/regenerated-input
choices. It retains every candidate score and action change. Outcome panels
separate original choices with original labels, original choices with
regenerated labels, and regenerated choices with regenerated labels. The
original whole-block bootstrap and complete-target gate remain in force.
No new score-difference acceptance threshold is invented.

## Execution and preserved results

The CLI requires an explicit full `--expected-head` matching a clean checkout
with a containing remote-tracking ref. The coordinating run record must also
verify the actual remote ref with `git ls-remote` after pushing. Record its
output and command exit status. The report records audit source/tree,
auditor hash, original source bindings, Python/NumPy/SciPy versions, platform,
NumPy build information and explicit numerical-library thread settings.
The source and complete evidence bindings are checked again after computation;
a changed source or artifact invalidates completion without discarding the
already accumulated diagnostic results.

Choose a new durable report path outside the entire repository and immutable
attempt, under `/data2/neural-shrimp-data/audits`. Create a unique audit
directory before execution and preserve it alongside stdout/stderr and the
exit-status record:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python tools/audit_information_regeneration.py \
  --expected-head FULL_PUSHED_AUDIT_COMMIT_SHA \
  --out /data2/neural-shrimp-data/audits/EXP-008-regeneration-UNIQUE/report.json
```

The command refuses an existing output path and never writes into the
repository. Retain its report, stdout, stderr and exit status, including
failed runs. Exit 0 means this diagnostic completed without its comparison
failures; exit 1 means it completed while retaining arithmetic/comparison
failures; exit 2 means execution, identity, binding or complete-regeneration
requirements were unmet. No exit code replaces the original strict verifier
or authorizes merging its failed release.

This diagnostic cannot resolve integration convergence, establish a fresh
periodic orbit, certify completely regenerated ODE inputs, or supply a new
held-out geometry-utility success. Those require separately frozen work.
