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
