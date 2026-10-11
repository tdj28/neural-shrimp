# EXP-013 refinement: outcome-selected numerical diagnosis

This is a bounded follow-up of the single candidate whose original EXP-013
attempt-002 native waveform comparison failed. It is selected after outcomes,
not a fresh-origin experiment or a revision of EXP-013. The original immutable
receipt SHA256 is `058aa80de824ecdfa00d8abc77d85ff231353a2af60e8db932f39eb07936552e`;
its source is `e5864f93ec6e822ba4fc6670640149f4a7c07b7b`.

The only cases are `fresh-11-p1350-r0.016-c07` and the matched zero-dose
`fresh-11-p1350-sham`. Each receives three new native RK4 integrations at
steps .000625, .0003125 and .00015625, and two DOP853 integrations:
(rtol=1e-12,atol=1e-14,max_step=.025) and
(rtol=1e-13,atol=1e-15,max_step=.0125). Exactly ten traces are planned.
The saved initial state, realized parameter perturbation, original certified
reference,75-unit duration,751 samples at .1-unit spacing,128-bin phase search,
full metrics/component/window definitions and fixed-reference metric remain
unchanged. The existing `fresh_preview_runtime.worker`, `score`, `execute`,
owned-entry cleanup and native compilation functions are reused. No frozen
source is edited.

The plan binds the original receipt and manifest, baseline result/certificate,
policy and winner seals, access audit, postflight, both cases' three original
native raw arrays and their terminal records. These are verified against the
original receipt inventory. Each saved original input is reconstructed from
the original certificate phase and canonical candidate direction, then compared
exactly with the saved initial state/alpha/gamma/delta. Six original complete
metrics packets are replayed without new integrations. All new raw packets are
hash/size checked, required to contain exactly six named arrays with finite
float64 dtype and exact shapes/initial/grid, and rescored with the original
strict score comparison (rtol1e-12,atol1e-14).

For each case, report ten pairwise comparisons among the five new methods,
five comparisons of the original .00125 trace against each new method, and the
two original adjacent-native pairs (.005 versus .0025; .0025 versus .00125).
These are34 distinct comparisons globally. The chain original .00125 → .000625
→ .0003125 → .00015625 supplies three named native-adjacent comparisons per
case, six globally, selected from the same comparisons. No expected convergence
order or monotonic improvement is imposed after observing these values.

Each comparison retains the absolute aggregate aligned-RMSE change, direct
same-clock activation RMS and maximum differences, per-neuron RMS, maximum
location/time, potential differences, and each waveform's score at the
**original fine trace's saved phase**. The unchanged scalar/waveform thresholds
are respectively1e-4,1e-3 and1e-2, with equality passing. The shared-phase score
is descriptive and never substitutes for the original independently aligned
objective. All required pair failures/missingness are shown. A diagnostic flag
says the original fine trace is resolved only if all new methods mutually pass
and that old fine trace passes against every new method. This flag cannot
qualify, replace or edit the original failed .0025/.00125 waveform comparison.
The matched sham is reported separately.

No policy refitting, new origins, changed shortlist winners, fitted directions,
relabeling, geometry-cohort means or deployment claims are authorized here.
The historical result remains unchanged regardless of this diagnostic.

Execution requires a complete local review and a committed, privately pushed
source/plan freeze. CLI `run` requires the expected40-character source HEAD,
clean tracked/untracked checkout, exact committed source bytes and live remote
branch equality. New source plus all original EXP-013 source paths are bound.
Only root may launch after confirming that private freeze. No target trace was
used to develop or test this code; tests use synthetic arrays/mocked workers.

Use at most four CPU workers, each numerical library thread capped at one;
require8GiB available system memory and2GiB free space on the actual attempt
filesystem. No GPU, RunPod or paid call is used. The120-second producer alarm
covers input replay, compilation, ten traces and analysis, with each trace
limited to30 seconds. Preflight source/input validation precedes that timer;
owned-worker termination and final failure-preservation/source-postflight/
receipt sealing are outside it and may extend elapsed wall time. No retry or
replacement is allowed. Every failed/unstarted task retains its planned row;
all34 comparison rows remain visible on producer failure. Available partial
native/adaptive outputs, compiler receipts, task records, PID/boot/start-time
ownership and cleanup records are retained through existing runtime behavior.
No unowned process is inspected beyond system resource availability or stopped.

Output is an exclusive attempt directory under external
`/data2/neural-shrimp-data/raw/EXP-013-refinement/`; paths resolving anywhere in
the repository are rejected. No existing attempt is resumed or overwritten.
The whole-attempt cancellation uses a BaseException-derived signal so per-record
I/O failure handlers cannot swallow its deadline. The final receipt inventories
and hashes all artifacts. `verify` binds the plan and every source file to Git
at the recorded commit, and validates complete and partial terminal identities,
producer/postflight errors, receipt conclusions and any dispatched-worker
cleanup before accepting a failed-attempt placeholder. `verify` only reads
saved inputs/arrays, checks exact inventories, ownership records and metrics,
and reconstructs all comparisons; it never integrates or fits. Review reports
belong outside immutable raw evidence. A partial attempt remains partial.

Prepare (metadata/saved replay only) after copying reviewed sources into place:

```
.venv/bin/python tools/run_fresh_failure_refinement.py prepare --out experiments/hopfield/EXP-013-refinement-plan.json
.venv/bin/python tools/run_fresh_failure_refinement.py validate --plan experiments/hopfield/EXP-013-refinement-plan.json
```

After root's complete private freeze, use `run --plan ... --data-root
/data2/neural-shrimp-data/raw --attempt attempt-001 --expected-head <frozen-head>`.
Subsequently use `verify --directory <exclusive-attempt>` with no raw writes.
Private branch backup does not substitute for authenticated PR review or
permit a main merge. No new global experiment number is assigned.
