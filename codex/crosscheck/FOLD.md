# Independent fold and normal-form cross-check

The independent manual Qwen implementation recovers the two-cycle fold and its
local bottleneck coefficients. Substituting the newly computed coefficients
into the quadratic passage model leaves all 18 scalar passage predictions
unchanged; those predictions remain within one iteration of the independent
full-model passages. This supports the LLM manuscript's unfitted local
mechanism at the specified fixed-prefix soft-token map.

## Fold location and classification

At `log10(T) = -0.115`, the independent augmented spectral Newton solve starts
from the saved centre displaced by `1e-7*r` and alpha displaced by `1e-9`.
It takes two full corrections and three dense monodromy evaluations:

- `alpha_c = 0.06279827217666506`, differing from the producer by `-5.8009e-15`.
- Full-state two-step relative closure: `6.55145e-15`.
- Augmented closure/unit-multiplier residual: `5.43531e-14`.
- One-step relative separation: `1.03505`, distinguishing the primitive two-cycle.
- Critical multiplier: `0.999999999999946`; next modulus: `0.0369846745279201`.
- Both phase closures pass; critical left/right modes pass normalization and
  eigenpair residual checks. The complementary directions contract.

The fold has a **neutral +1 critical mode to numerical precision**. In the
immutable raw attempt, a generic strict-radius comparison printed an attracting
flag because the floating eigenvalue lies 5.4e-14 below one. That flag does not
establish attraction at the fold. The fold gate already used the isolated +1
mode and contracting complement; the audit classifies it accordingly. Future
fold output labels have been corrected, without changing this raw evidence.

## Independent derivative estimates

`a = l^T partial_alpha(F^2)` includes the steering dependence of the prefix.
`b = (1/2) l^T D^2(F^2)[r,r]` comes from nested automatic differentiation of
the independently implemented forward map. These are float64 numerical
estimates, not exact-arithmetic results. Sign and phase match the saved modes.

| Quantity | Independent AD | Producer | Difference |
|---|---:|---:|---:|
| a | -4.834227003689755 | -4.834227003692353 | +2.598e-12 |
| b | -25.4284561647936 | -25.42851228205931 | +5.61173e-5 |
| ab | 122.92692945398646 | 122.92720073765378 | -2.71284e-4 |

The producer's b is an average of two finite-difference estimates. Repeating
its full five-step-size table agrees to within 5.99e-10, including its averaged
b to within 4.91e-10. Its 2.21-parts-per-million difference from nested AD is
consistent with the observed quadratic finite-step bias. The independent
parameter finite-difference estimate differs from AD by 3.22e-7 (relative
6.65e-8). All tables and discrepancies are retained in the audit and raw output.

## Consequence for passage predictions

With the original fixed U, entrance locations and epsilons, new AD coefficients
preserve all 18 discrete quadratic counts. The largest change in the continuous
arctangent prediction is only `0.00013743` two-step iterations. The prediction
comparison uses the existing independent full-state runs at the original fold
parameter/entrances; it does not claim a second set of full-state trajectories
at the newly refined centre. The alpha shift is reported above.

## Evidence and reproducibility

- Source: `48e921044b3124f1151d14fb40e0f8ae3ef5f6d9`.
- Raw root: `/data2/neural-shrimp-data/codex/task2-independent-crosscheck/fold-001`.
- Complete checked inventory: `receipts/fold-001.json`.
- Independent saved-array audit: `audit_fold.py`, `audits/fold-001.json`.
- Four CPU threads, chunk 128, 700.48 seconds, no GPU or paid resources.

The saved-array audit re-solves the eigensystem, checks mode residuals and
coefficient contractions with extended-precision arithmetic, recomputes scalar
passages with Decimal60, and checks source/output hashes. The manual forward
implementation and the independent numerical solver use the same pretrained
weights and PyTorch primitives; they import no producer research tool or
Transformers model. This is rung-2 numerical cross-validation of this map and
parameter slice, not an interval proof or a global window-organization result.
