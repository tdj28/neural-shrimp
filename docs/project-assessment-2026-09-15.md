# Project assessment, 15 September 2026

Updated through EXP-012, completed 16 September UTC. **Continue the research
on the local CPU.** We have a numerically verified small-network dynamical
result, qualified short-preview labels, and reproducible comparisons showing
why simple controls remain essential. Geometry's incremental control value
is still unresolved. The tested GPU implementation provides neither the
required numerical compatibility nor a measured pipeline speedup; RunPod is
not needed for the next bounded study.

## Completed work and its limits

| Work | Established result | Remaining limit |
|---|---|---|
| EXP-004–005 | The known three-neuron tanh system has numerically verified full-state primitive cycles and transverse stability at selected points within the mapped window. | Complete boundary organization and transfer to trained networks are unfinished. |
| EXP-006–008 | Controlled waveform experiments and complete candidate comparisons distinguish prediction improvements from action outcomes. | Pooled incremental action utility is unresolved; the original EXP-008 strict portability gate remains failed. |
| Arithmetic/regeneration audits and EXP-009 | Saved-input regeneration preserves original aggregate scores, penalties and selected actions; all 220 originally selected actions pass EXP-009's declared scalar refinement rule. | Nineteen regeneration comparisons remain failed. Many longer candidate labels remain unresolved; stable selected actions do not qualify the full 300-unit table. |
| EXP-010 | All 6,944 required traces complete; all 3,456 short-preview labels pass the declared native scalar/waveform checks, all sixteen independent sentinels agree, and all 192 rankings remain unchanged. | Qualification concerns tested 75-unit calculations, not an acceptable task-loss margin or longer-horizon utility. |
| EXP-011 | All 24 origin folds and 96 model recipes complete and pass numerical checks/replay. Mean four-preview RMSE is 0.00856899 for signed axes, 0.0107899 for geometry and 0.0162661 for shared features; the full-eighteen reference is 0.00617504. | These are historical development origins. Geometry's paired increment over shared features, 0.00547615, has a descriptive interval spanning zero: [-0.00265276, 0.01561846]. No net deployment saving is established. |

The [EXP-011 audit](EXP-011-results-audit.md),
[preview qualification](EXP-010-results-audit.md), and
[previous assessment](project-assessment-2026-09-14.md) retain the detailed
scientific and numerical scopes. The earlier training pilots did not establish
verified trained-network cycles or recurrence-control benefit.

## GPU qualification is complete; retain CPU execution

EXP-012 completes **54 batches and 64,188 repeated known-case integrations**,
covering 3,456 unique preview cases. All 25 CPU batches reproduce their saved
reference, all ten GPU repeat/layout comparisons pass, and candidate rankings
remain unchanged. However, the full-census CPU/GPU state and metric parity
gates fail at both step sizes. The frozen qualification therefore rejects
deployment of this CUDA backend. Retain the failures and unchanged thresholds.

The GPU is slower in **all 24 matched warm timing pairs**. At the declared
3,456-case, `dt=0.00125` deployment setting, median pipeline time is
**69.6893 seconds CPU versus 75.8564 seconds GPU**; median paired CPU/GPU
speedup is **0.918177**, below the required 1.25. There is no finite measured
break-even. These timers include the declared scoring, storage and verification
work on the shared server; they are not kernel-only timings or a universal
claim about GPUs.

Independent numerical auditing passes 194,275 evidence checks with zero
audit failures, and the separate CLI replay reproduces the original audit and
summary. This verifies the recorded failed qualification outcome. Complete
raw evidence is retained at
`/data2/neural-shrimp-data/raw/EXP-012/attempt-001/`, including `summary.json`;
the [frozen protocol](EXP-012-protocol.md) defines the gates. No paid compute
was required.

## What remains, in execution order

1. **Finish the complete EXP-013 package.** The
   [fresh-origin tradeoff design](fresh-origin-tradeoff-design.md) specifies
   24 new starting origins beyond the 36 distinct prior starts, conditional
   on the observed atlas. Complete its staged runner, fresh feature adapter,
   qualified-label access, cost accounting, failure ledger, machine plan and
   independent review. The proposed budgets are 2/4/8/18; preserve strong
   fixed controls and the full-eighteen reference at every radius.
2. **Freeze before fitting or dynamics.** The draft
   [final-model module](../tools/preview_final_model.py) and its 34 synthetic
   tests are preparatory. No EXP-013 final model has been fitted and no
   EXP-013 fresh dynamics have run. Commit and privately push the complete reviewed
   protocol, code and bound plan first. Fit four recipes only on the 24 old
   origins, seal them before fresh dynamics, and seal fresh predictions and
   shortlists before accessing fresh perturbed-preview labels.
3. **Run and audit the CPU study without substitutions.** The draft requires
   10,608 short traces, plus reference/certificate/sensitivity work, with
   fixed deadlines and all 24 origins retained in the denominator. Any
   required numerical or completeness failure prevents the planned qualified
   full-cohort comparison. Report loss versus charged preview budget and
   separately measured prerequisite costs; do not invent a quality margin or
   equate fewer calls with a measured controller speedup.
4. **Keep the larger claims separate.** Resolve longer-target reliability
   before renewed 300-unit utility claims. Boundary organization, a
   topology-specific mechanism and benefits in trained networks each require
   additional evidence. Finish private repository review and independent
   external-data backup verification; later experiments do not clear the
   original EXP-008 merge gate.

The latest complete test suite reports **542 passed, one optional GPU test
skipped**; the earlier separate synthetic CUDA smoke passed. Local CPU/RAM
capacity is sufficient for the bounded next step. RunPod remains a costed,
explicitly approved fallback if future capacity or contention warrants it.
