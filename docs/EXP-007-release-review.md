# EXP007 independent scientific release review

Date: 2026-09-11. Disposition: **accepted for the private working-paper update;
no unresolved scientific release blocker found**. This is a local review,
not publication approval, a paid review, a new experiment, or replication of
the separately discussed preprint. The reviewer changed only this document.

## Evidence and claims checked

Reviewed the new material in `paper/main.tex`, `paper/exp007.tex`,
`paper/generated-exp007-results.tex`, README, next steps and claim ledger,
against the frozen EXP007 attempt, `EXP-007-results-audit.md` and
`EXP-007-independent-check.json`. The frozen source is
`ac098c3e3833daaf27c7dc2869d45e8b7085f948`; existing EXP004–006 sections were
checked for contradictions introduced by the new conclusions.

- Independently reaggregated all 2,784 target records, without importing the
  experiment analysis helper. All **35 means and 30 contrast/interval values**
  in the two new numerical tables match their six-decimal presentation.
  Recomputed 10,000 paired-anchor bootstrap draws confirm that all five
  primary pooled intervals contain zero at both integration steps. All ten
  displayed largest-dose/pooled intervals contain zero. The largest-dose
  local-relative reduction is 55.653989%, correctly presented as 55.65% and
  paired with its unresolved interval and 6/12 positive-anchor count.
- Independently reread the selected raw state arrays: tangent and axes have
  identical actions in all **192 conditions**, identical selected preview
  states in 192 pairs and identical target states in **384 pairs**. Their
  actions are +gamma in 168 and −gamma in 24, with alpha fixed. The manuscript
  properly treats this as a sample-specific attribution result, not proof of
  universal coordinate optimality or proof that geometry is useless.
- The unfavorable fixed example, zero-dose floor, coarse/fine discrepancies,
  census, artifact count and reported costs match the independent audit.
  The manuscript distinguishes selected-phase RMS reconstruction from a new
  solver validation, and explicitly rejects uniform pointwise convergence.

## Interpretation and display

The abstract and discussion preserve the measured neural rhythms and unstable
continuation evidence while reporting that the stronger utility comparison is
unresolved. They do not overwrite the development-stage EXP006 result. The
hybrid's three exact local-policy copies and quarter-sized pooled contrast
are disclosed. Next steps treat both experiments as development data and
require a new freeze for any changed switch or controller.

The new sections distinguish new starting locations on an already observed
atlas from a new network or unseen map; the exclusion concerns old origins,
not every old destination. The 0–75 preview and 75.1–300 scoring samples are
correctly separated, with their dynamic dependence and cloned-simulator
assumption stated. A single phase fit is distinguished from frequency fitting.
Candidate-call budgets, sensitivity costs, map costs and concurrent wall-time
sums are not presented as equal total resources, elapsed runtime, or a speedup.

Scientific skim of `tools/plot_hopfield_holdout.py` found no serious display
error. It retains all seven policy means, all five declared contrasts, both
steps, every attempted anchor, and the predeclared waveform example. Contrast
signs, centered heatmap scale, native/normalized coordinates and exclusion
circles are consistent. Waveform errors use the saved tail-fit phase and are
checked against the recorded RMSE; the synchronous reference is explicitly
distinguished. Curve identities are disclosed rather than counted as separate
evidence. Exact raw/output hashes and portable arithmetic replay are retained.
Print sizing and final compiled-page visual QA are handled separately by the
root reviewer; no visual-layout certification is implied here.

## Separate preprint paragraph

The versioned local reading note and exact Markov-chain audit support the
bounded wording used here. The **12 synthetic identity tests pass**. The
manuscript identifies the source as an unreviewed preprint, distinguishes
extensive capacity from unlimited finite-network storage, and separates
overlap magnitude from waveform fidelity. Printed-equation inconsistencies
are not described as proven implementation failures. Finite strictly causal
nilpotence motivates a request for an explicit infinite-time operator; it is
not presented as disproving the intended asymptotic mechanism. Capacity
numerics remain explicitly unreplicated, and no shrimp/controller benefit is
inferred from this separate work. This release review did not independently
reproduce or re-review those external capacity numerics.

No target integration, paid call, runtime edit or manuscript edit was performed
for this review. No additional scientific change is requested.
