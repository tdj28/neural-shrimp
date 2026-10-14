# Task18 run002 saved-summary recovery — prospective 2026-10-14

The original process failed while writing its final status, after all scientific
events. This separate recovery never replaces that failed status or receipt.
No actual scientific events/arrays have been read during preparation: only exact
run.py/analysis.py/launcher/plan source and Inference's metadata-only closure.
The new recovery code was AST-parsed, not imported, executed or tested.

## Source-bound failure and adaptation

Original source7918af949fa60cb8bb123504e78c2dc959943d9c captures run.py
80bf95a99738cc05e4d3480994051fc684666230a4e06aebca5719af1b1d23c0
and analysis.py
a5553396985f947bb0d8e628f6f1190a683141b341b46538fdc06ceaeec9faf2.
analysis.phase computes covariance_pass with a NumPy scalar comparison. Its
phase-summary event passes through Store.convert, which calls np.generic.item().
But the returned summary is retained unconverted in completed_phases. The final
run.py write(status,result) bypasses Store.convert and fails on the first NumPy
bool. Because write uses exclusive x mode, the exception handler's second write
of status raises FileExistsError on the original truncated file. Inference
confirmed exact stderr `TypeError: Object of type bool is not JSON serializable`
and the subsequent FileExistsError; original stderr SHA
6cf6250d0e25a829223bed7c4e2759fda699338eff08ee4029b3a7d4a1a93401.

No original source is patched. source.diff is only the new recover.py file.
The adaptation reads converted immutable events; it does not rerun or change
phase arithmetic, gates, matrices, eigensolvers, SVDs or producer analyses.

| Intended final field | Exact recovery authority / limit |
| --- | --- |
| orbit_gates | Unique original orbit-gates event, with threshold/boolean consistency and original orbit metadata |
| phases | Eight original phaseN-phase-summary events, adding index from their literal names; all8 direction summaries linked to native direction events |
| phase leading spectra/ratios | Retained covariance_eigenvalues/map_singular_values references; no eigen/SVD computation |
| cyclic_monodromies | All8 original monodromy-phaseN events, leading8 signed/complex values and first2 saved singular values; scalar product of saved step norms reconciled |
| composed_controls | Original composed-controls-summary and all3 original composed-direction scalar fields |
| elapsed | Original launcher receipt elapsed_seconds2031.384801283013, explicitly launcher timing |
| monodromy seconds | Original event_seconds; intended later per-loop final-summary timestamp is not exactly recoverable |
| original peak RSS | Unavailable: accepted launcher has no peak field and truncated worker status never reached it. Startup RSS is not peak. |
| original backend | Unavailable: final version/thread fields followed the truncated phase list. No current versions substituted. |
| exit/status/receipt | Original exit1, loop_gain_failed, numerical_gain_result_available=false and truncated status preserved; no synthesized success receipt |

The original failure/closure authority is exact:
- receipt4a3153f174bec42f034e245b63d5fe48978962908cbf82478ace3498b632b4fa;
- plan98a01706d756e3271f3f566dedf792719bf088055ba084c753b199b649b198c4;
- card8f44e3255796dd2529840a32ca4a213987d5a68f3f2b31170aa7aa7f5fb1f5f5;
- run-002-closure-001/report.json
  2d3a679a6dabde4379f95c56c66b303d9a82a5e1ddad6b76f4a3d7296c66b047.
That coordinator report binds3765 artifacts/436,662,559B, seven exact original
Git blobs, session44216 reaped1 and original waiter/launcher/worker birth groups
absent. The recovery independently hashes all original files and original
supervision members, but inherits that completed Git/process attestation;
it makes no fresh Git/model/process query. Assets are not reloaded. The fact
that original control flow reached final status serialization supports execution
of its preceding source/asset checks, and is labeled as that evidence basis.

## Bounded contract, not launch permission

Prepared-card.json remains admission=prepared and refuses execution. Root must
accept exact source/card/closure; Inference owns private checkpoint, original
process lifecycle and the existing externally bounded supervisor adaptation.
Proposed envelope: CPU1+one supervisor, nice10,1GiB hard AS,120s external deadline,
8MiB output,20GiB startup and16GiB runtime available memory. Serialize with other
auxiliary work unless combined resources are explicitly admitted. No paid/GPU
resources. NumPy2.5.3 plus stdlib only; no Torch/model/producer imports.

CLI: `/data2/neural-shrimp/.venv/bin/python -I -B -u recover.py --card EXACT_CARD
--output /data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-summary-recovery-001`.
The output must be absent. Report and failure artifacts remain in this new
exclusive path. The script hashes original436.7MB, reads only small completed
JSON events and saved spectrum vectors (~sub-MB), and does no dense products.
1GiB/120s is a conservative finite proposal, not a new timing measurement.

The script requires the exact587 event names/order (all8 complete phase blocks,
all8 monodromies and final composed summary), hashes the native spectrum
references and checks their original dimensions, ordering and finite values.
Original scalar-gate/leading-summary reconciliation uses fixed
atol5e-15/rtol5e-13. That is bookkeeping consistency, not relaxed scientific
thresholds; original1e-9/1e-12/1e-8 gates remain explicit. Every phase and cyclic
start is retained, with signed/complex multipliers. Recovered gate failure is
reported separately from recovery completeness; no favorable phase is selected.

Outputs: report.json, manifest.json, byte-preserved
original-partial-status.json.bytes and original-stderr.log. report.status is
saved_summary_recovered with recovery_completed=true only for a complete saved
reconstruction; original_exit_success=false always. saved_numerical_gates_pass
is a separate field, not an interval certificate or a claim of original process
success. Actual recovery elapsed/peak/backend are separately labeled recovery_*.
An authentication/reconstruction error yields saved_summary_recovery_failed and
retained progress/traceback, return1. No automatic retry or evidence replacement.
No producer numerical artifact has been read; comparisons remain separate later
work after this own saved result is sealed and reviewed.
