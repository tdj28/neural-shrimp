# Task19 optional fixed stress-cell adaptation — 2026-10-14

This is source-only preparation for the two optional cells frozen before their own outcomes: original H3 index **4**, log10(T)=**0.1799135096497498**, reported symbolic period10; and index **13**, log10(T)=**0.18478127233738384**, reported symbolic period28. Both use alpha0.0655. Selection and the producer's original/amended findings were already exposed. These are exploratory independent finite-horizon stress checks, not fresh confirmatory tests. No optional target, model, profile, producer reducer or target-array decoding was executed during preparation.

## Source and scientific delta

Adopt only new `codex/return-map-crosscheck/stress_runtime.py` and focused `tests/test_codex_return_map_stress.py`. The accepted mandatory `runtime.py`, `trajectory.py`, `reduce.py`, model/assets and every prior attempt remain untouched. `runtime.diff` compares this new adapter with the accepted mandatory runtime for review; it is not a patch to apply to the mandatory file.

The adapter pins the five exact existing scientific files and adds only its own captured source. It imports the unchanged runtime's `Recording`, manifest authentication, writer and source-loader helpers; uses the unchanged `trajectory.advance`, finite-lag detector, symbolic period, SVD/candidate/fold/aggregate and sample-band reductions. A focused AST control verifies the entire model load/prefix/native step/twin advance block is identical to the mandatory block. The only physical parameter change in that block is the independently bound cell's log10(T).

Each cell starts from the same independent model's original fixed initial state, with a newly generated prefix at the same alpha. It uses1000 transient single-row calls and2000 recorded batch3 calls, offsets1e-6/1e-9, seed0, original floors, all four disjoint windows and20 running means:3000 native calls/7000 row-forwards per cell. Every native input/return, growth, floor and renormalized state remains retained by the unchanged recording and advance code. Full-vocabulary readout uses the same input/output row clock. Model/vector admission and post-run asset hashes remain mandatory.

No optional cell inherits the mandatory seven-phase/phase6 or trace-best/bands-summary comparison. The actual observed symbolic period alone chooses the saved all-phase fits and band rows. The reported original periods10/28 are checked as immutable original identity and recorded as comparison facts; disagreement does not change the observed period, select a preferred phase, stop the run or count as a computational failure. q=0 retains full trajectory/growth/finite-lag diagnostics and explicitly reports no symbolic-period phase fit. Degenerate/short/rank-deficient/undefined fits retain the unchanged reducer dispositions.

The adapter checks the full65-entry metadata shapes and exact indexed row in the already immutable input authority, including original trajectory recipe/offsets/finite comparison exponents. This admission reads the metadata but does not select from producer outcomes. The original indexed exponent and original/amended fitted-map comparisons are computed only after the new trajectory and all observed-clock fits finish. Comparison projections are copied literally from the accepted saved65 runtime; amended selection keeps the historical `producer_value_fit_qualified` rule, including separately disclosed rank warnings, and the physical-coordinate Chebyshev root count. Purged contiguous diagnostics remain unchanged. Every missing/unqualified comparison and numerical delta is retained. There is no late-state equality requirement and no exact/asymptotic chaos, invariant-band separation or conjugacy claim.

## Fixed plan and supervisor interface

Inference owns concrete cards, source/checkpoint adoption and any eventual external launcher. No card or admission change is prepared here.

- CLI remains `stress_runtime.py --plan <captured plan> --output <fixed path>`.
- Exact plan keys: schema, admission, mode, cell_index, envelope, source_sha256, input_manifest_sha256, experiment, scope.
- Schema `codex-return-map-stress-v1`; mode `stress`; admission must be `reviewed`. Prepared cards refuse before creating an output.
- `cell_index` must be the integer4 or13. Distinct exclusive output paths are `/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/stress-cell04-001/run` and `.../stress-cell13-001/run`; there is no arbitrary output or retry suffix.
- Six-source inventory: the unchanged original five plus `codex/return-map-crosscheck/stress_runtime.py`. The original five hashes are hard-pinned in `BASE_PINS`; adapter hash is the exact captured new source.
- Original input manifest remains `d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8`. The existing snapshot provenance limits remain unchanged.
- Experiment: exact alpha/log10(T), transient1000, record2000, offsets[1e-6,1e-9], seed0, `expected_producer_lead_period`10 or28, maximum_period64, threshold1e-8. There is no primary_phase field.
- Environment remains `TASK19_SOURCE_COMMIT`, `TASK19_PLAN_SHA256`; CPU thread/offline settings and NumPy2.5.3/Torch2.14.1+cu130 checks are unchanged.
- Output `status.json` uses `completed`, `computational_completed=true`, mode=`stress`, exact cell_index. Negative/unqualified scientific outcomes do not falsely become execution failures; experimental_claims/certificate_passed remain false. Full result, input binding, original plan and manifest remain separate. Exceptions retain a failed status/traceback and the external owner remains responsible for original termination/receipt.

## Measured planning basis, not launch approval

Root supplied the completed mandatory run measurement **1221.013s**, **5.093GiB peak**. Each optional cell has the same3000-call/7000-forward schedule and same model dimensions. A transparent per-cell extrapolation `2*1221.013+300 = 2742.026s` fits the unchanged prospective4200s cap; two sequential such forecasts total5484.052s, with8400s combined hard worker caps. This is an estimate, not a measured optional-cell runtime. Actual phase count and fit/band work can differ; no extra timing profile is proposed.

The adapter retains the already accepted trajectory envelope CPU2, nice10,12GiB AS/RSS-watchdog,4200s,512MiB output with4MiB closing reserve,36GiB startup/16GiB runtime MemAvailable. Inference/Root still decide exact cards, current capacity and serialization; these source constants do not authorize launch. No simultaneous optional execution is requested.

The fixed native array payload is143,706,176 bytes before NPY headers, journals, JSON fits/comparisons, manifests and captured source. It does not grow with observed q. The all-phase fit output retains every candidate/fold; the original512MiB cap remains enforced rather than assuming a tighter measured optional output bound. At the reported clocks, band rows would number45 or378; actual observed q determines the real count. No phase is omitted to reduce cost or improve agreement.

## Focused verification

Fifteen tiny controls pass in0.03s with CPU1/nice10/thread caps. They exercise the two exact identities, all other index/type refusals, prepared/changed parameter/source/cap rejection, separate output paths, missing/invalid original metadata, observed-clock-only fitting including q=0, original/amended comparison rules and preservation, and exact AST equality of native wiring. No old numerical tests, model/profile/producer tests or raw target calculations were repeated. Final control output is `controls-002.log`; the initial15 controls also passed before adding recipe/nonfinite-comparison admission coverage.
