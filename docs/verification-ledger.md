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

## 2026-09-11 EXP-006 utility and branch audit

This entry updates the research state after the historical entries above.
EXP-006 executes the initial waveform-utility and local branch-continuation
experiments. It does not complete the full boundary or coupled-control agenda.

- The complete source, design, and machine plan were pushed at
  `ddb568e6cd67ef2cfbdb45e05710cd86025ef548` before outcomes. All 136 tests and
  source-freeze CI passed. Integrated result/manuscript checks are defined in
  `.github/workflows/tests.yml`; exact run status is retained in GitHub.
  No public-release approval is asserted here.
- All 4,472 receipt-listed artifact byte counts and SHA-256 values match.
  The census contains exactly 4,320 complete trial-step records and 12/12
  eligible attempted anchors. No condition or failed anchor was replaced.
- The independent utility audit recomputed every saved activation waveform
  RMS at its recorded single global phase; maximum discrepancy was zero.
  It separately reconstructed the declared contrasts without importing the
  experiment aggregation helper. Numerical-step sensitivity and zero-dose
  floors remain reported rather than used to select preferred outcomes.
- The primary result is dose dependent: the largest-dose atlas comparison
  favors the atlas versus the declared symmetric local policy, both smallest
  doses favor local at all anchors, and the pooled local comparison is
  unresolved. The `tangent_plus` diagnostic has lower mean error than the atlas
  at every dose and remains an explicitly exploratory comparison.
- The immediate waveform window and eventual nine-peak assay have different
  horizons. The frozen example remains `anchor-08`, which favors the local
  comparator despite eventual nine-peak retention for all displayed arms.
  No high-error immediate trace is thereby certified an asymptotic waveform.
- A separate continuation audit reconstructed the two-dimensional Poincaré
  spectrum from saved matrices using a cross-product orthonormal section basis,
  independent of the implementation's SVD basis. It checked neutral-flow and
  phase residuals, spectra, stability flags, crossing candidates, reversals,
  and computational limits. No new target integration was performed.
- Four branches retain 196 node records, 122 marked unstable, and five failed
  correctors. These include overlapping branches and repeated seeds.
  Independent geometric checkpoint agreement is 27/28. The near-edge initial
  positive-gamma branch's node 40 fails the original closure gate:
  1.0590609546e-8 exceeds 1e-8. Its matching state and period do not convert it
  into a pass. No threshold was weakened.
- Compact continuation nodes do not store integrated endpoints. Their closure
  and primitive-event checks were audited against the recorded gates, not
  independently reintegrated. Crossing brackets are arclength-neighbor
  candidates, not exact gamma intervals at folds or a complete shrimp boundary.
- The receipt records 423.866 seconds for collection and analysis before final
  verification. Its 48 local-sensitivity solves total 26.627 seconds within that
  clock; the prior atlas cost 140.54 seconds separately. No RunPod, other paid
  compute, or paid Pro review was used.

See [EXP-006 utility results audit](EXP-006-utility-results-audit.md),
[EXP-006 continuation results audit](EXP-006-continuation-results-audit.md),
and the [EXP-006 receipt](../evidence/EXP-006/attempt-001/receipt.json).
The current final PDF inventory and artifact hashes belong in
`paper/provenance.json` after its final render-and-verify pass.

## 2026-09-11 EXP-007 prospective freeze

The new-start selector, simulator-preview policy comparison, separate scored
window, failure accounting and analysis are covered by local synthetic tests
and two local audits. The complete plan is compiled from existing atlas data
and the EXP006 manifest; no EXP007 target outcomes exist at this entry.
Runtime and scientific outcome verification follow in a separate dated entry.
The literature note also has independent analytical checks of printed
partial-refresh identities; these do not reproduce the preprint numerics.

## 2026-09-11 EXP-007 completed evidence and release checks

- The only attempt ran from the pushed source
  `ac098c3e3833daaf27c7dc2869d45e8b7085f948`, preserved under
  `archive/EXP-007-source-v1`. All 12 baseline attempts qualified; all 4,032
  previews and 2,784 target records completed. Frozen verification exited 0.
- The copied evidence passed exact checks for all 7,149 receipt-listed
  artifacts. A separate auditor with no experiment imports reconstructed
  504,320 RMS values in 9,600 metric blocks, with maximum discrepancy 4.44e-16,
  and independently reproduced all selections and bootstrap contrasts.
- A separate scientific release review reaggregated all 35 table means and
  30 displayed contrast/interval numbers directly from records. All primary
  pooled intervals span zero at both steps. Tangent and coordinate actions
  match exactly in 192/192 conditions and 384/384 target arrays.
- The new driver, selection and analytical preprint checks pass alongside the
  previous tests. No paid review, RunPod resource or other paid compute ran.
- Three new empirical figure families retain plotted arrays and exact source
  and image hashes. The display-only amendment enlarges the utility plate;
  its numerical arrays and sample selection do not change. Final PDF inventory,
  test count, render inspection and file hashes are in paper/provenance.json.
- GitHub's required tests check remains the final merge gate. The research
  repository and its complete raw evidence remain private.

- The initial source-freeze GitHub run (34657214248) failed because the existing
  PDF provenance still bound the preceding workflow file. This was a stale
  release-metadata hash, not a numerical assertion failure. Final provenance
  was refreshed from the actual revised files; the checker and thresholds
  remain unchanged. The original failed run remains in GitHub history.
- Final visual QA inspected all 27 pages at 125 dpi. After the utility-plate
  layout correction, page 23 was inspected again and exact PNG hashes confirmed
  the other 26 pages unchanged. The final LaTeX log has no warnings or box
  overflows. All 186 tests passed locally.
