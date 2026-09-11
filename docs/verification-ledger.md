# Verification ledger

| Component | Verification at initial freeze | Limit |
|---|---|---|
| Neural gradient, Hessian, training map | Local analytic-control and independent finite-difference tests | Float64 CPU only |
| Recurrence / cycle / multipliers | Exact logistic and Hénon controls; false-period and transient tests | Numerical verification, not interval proof or shrimp geometry |
| Utility intervention | Analytic quadratic-GD paired-path fixture | No initial empirical benefit claim |
| Manifest and raw receipts | Deterministic plan/control matching; actual CLI archive fixtures | Not a registry preregistration; no resume |
| RunPod archive and process watchdog | Offline integration and failure tests | Never called provider live; provisioning/teardown controller not implemented |
| Dependency environment | Locked dedicated local environment | GPU/image parity untested |
| Paper | LaTeX/bibliography compile plus six-page PDF visual inspection | Working draft; human publication review pending |
| GitHub | Private repository creation/push verified live via gh | Remains private |

At the initial freeze and after evidence packaging, all 52 local tests passed.
The implementation also passed both push and PR CI runs. The complete pilot
trial census and receipt hashes passed `tools/summarize_pilots.py`; see the
[results update](initial-results.md). No “verified-live RunPod” or GPU speedup
claim is implied.

## 2026-09-11 visual revision

- All 59 local tests pass, including seven direct algebra checks for the tied
  scalar two-cycle, its multiplier, prediction errors, and stability boundary.
- Both figure indexes and all ten per-figure receipts pass source/output hash
  checks. All 323 original trials retain their original receipt hashes and
  classifications; no new target experiment ran.
- Every finite neural/policy terminal test MSE is reconstructed from frozen
  inputs and saved final weights. All six empirical plates' plotted values
  and selections are re-derived under declared numeric tolerances.
- Same-environment regeneration is byte-identical for the empirical and
  conceptual figure families. Temporary image and derived-value corruptions
  are rejected; original artifacts are restored. See
  [the figure audit](empirical-figure-notes.md).
- Independent mathematical and design review verified the exact example and
  corrected three future-design issues: generic period information now belongs
  to the baseline; final scoring data are separate from online validation;
  finite null readouts do not establish exact invisibility.
- The revised PDF has 17 pages and ten figures. All 17 rendered pages were
  visually inspected; compilation has no LaTeX warnings or overfull/underfull
  boxes. Root review also inspected every empirical plot and representative
  final pages. Final PDF/source hashes are recorded in `paper/provenance.json`.

This extends artifact verification and the prospective research plan. It does
not establish neural shrimp geometry, effective recurrence control, net
compute savings, a GPU result, or paid-review approval.


## 2026-09-11 numerical correction: actual neural shrimp

The historical entries above describe earlier revisions. The current manuscript
is rebuilt around EXP-004 and EXP-005; its final page/figure inventory and file
hashes are in `paper/provenance.json`.

- All 85 local tests pass. The 26 Hopfield tests cover independent state and
  Jacobian formulas, integration order and event timing, variational dynamics,
  analytical Floquet controls, recurrence rejection, failure preservation,
  and the eight independently initialized control calls.
- EXP-004 retains all 475 receipt-bound artifacts, including 39,523 grid/transect
  settings and every selected refinement. All artifacts, coordinates, event
  sequences, recurrence classifications, and census were verified after copying.
- Three EXP-004 selected orbits pass independent full-state shooting, primitive
  peak counts, neutral-flow alignment, and transverse Floquet checks. The
  originally selected exterior fails its intended interpretation: 0/9/9 peaks.
  It is explicitly retained in the manuscript and original waveform plate.
- EXP-005 retains all 21 receipt-bound artifacts. Eight fresh common-initialization
  runs use two steps and a 10,000-unit settling horizon. Two controls certify
  nine-peak cycles; both distant controls have no short recurrence and positive
  finite-time exponents at both steps. Those are bounded irregularity checks,
  not proofs excluding arbitrarily long transients or high periods.
- Independent saved-data audits reconstruct neuron/activation moments and
  perturbation separations and check the monodromy eigenvalues and neutral mode.
  Very small transverse multiplier magnitudes/signs are numerically unresolved;
  the reported dominant modes are separated from unity. The original dense-output
  cycle-RMS scalar is receipt-bound, but its unsaved dense residuals cannot be
  reconstructed at 1e-9 accuracy from the coarser saved sample grid alone.
- Original EXP-001--003 raw data and all ten historical figure sets remain
  unchanged. Their verification still passes. The training/policy pilot remains
  null; it is not reinterpreted using the recurrent-network results.
- Exact pre-execution source revisions are preserved under
  `archive/EXP-004-source-v1` and `archive/EXP-005-source-v1`. Those tags were
  created after execution to retain history; the original pushed commits are
  the prospective records.
- No RunPod resource, paid review, or other paid compute was used. The new
  experiments' measured local wall times are 140.54 and 4.17 seconds.

See [EXP-004 results audit](EXP-004-results-audit.md),
[EXP-005 results audit](EXP-005-results-audit.md), and the explicitly limited
[claim ledger](claim-ledger.md). Full boundary continuation, basin-volume
estimation, phase-aware robustness, and task utility remain future experiments.
