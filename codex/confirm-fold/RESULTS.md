# Task 5: prospective fold-passage results

**TLDR.** Of 24 planned forecasts, 24 passages were observed; 23 meet the frozen one-G-iterate quadratic accuracy criterion and 24 meet the separate unrounded arctangent criterion. 2/2 targets passed setup. Evidence integrity: **True**. These are forecasts of lingering and escape in the same fixed-prompt, continuous-feedback Qwen model. They do not establish a benefit for sampled text or behavior monitoring.

Plan SHA-256: `3ac03b7510060f52fda73d6e832a98b8c6ef5600095197050e9b745d96c59810`. Primary all-row success: **False**; arctangent all-row success: **True**. Individual scientific scores do not override a failed integrity check.

## Target A: G=F^2

Status: ready; result status: completed.

Independent fold: alpha=0.06705485936004332, log10(T)=-0.14. Critical multiplier modulus=1, argument=0 turns; next modulus=0.0879828927. AD coefficients in physical `alpha`: a=-7.05492494858, b=-24.9273403079; ghost direction=+1. U=0.00401165943758. Transverse seed=10501; max G=512.

Left-mode norm=28.8452; The transverse directions are defined by the left eigenvector. Transverse kick norm=0.00401166.

| epsilon | entrance | measured G | quadratic G | arctan G | quadratic error | arctan error | primary pass | arctan pass |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 3e-06 | full | 118 | 118 | 117.11547 | 0 | 0.88452877 | True | True |
| 3e-06 | midpoint | 59 | 61 | 58.557736 | -2 | 0.44226438 | False | True |
| 3e-06 | full+transverse | 117 | 118 | 117.11547 | -1 | -0.11547123 | True | True |
| 1e-05 | full | 56 | 56 | 55.976845 | 0 | 0.02315548 | True | True |
| 1e-05 | midpoint | 28 | 29 | 27.988422 | -1 | 0.01157774 | True | True |
| 1e-05 | full+transverse | 56 | 56 | 55.976845 | 0 | 0.02315548 | True | True |
| 3e-05 | full | 26 | 26 | 25.954638 | 0 | 0.045362387 | True | True |
| 3e-05 | midpoint | 13 | 14 | 12.977319 | -1 | 0.022681194 | True | True |
| 3e-05 | full+transverse | 26 | 26 | 25.954638 | 0 | 0.045362387 | True | True |
| 1e-04 | full | 10 | 10 | 9.744268 | 0 | 0.25573199 | True | True |
| 1e-04 | midpoint | 5 | 6 | 4.872134 | -1 | 0.127866 | True | True |
| 1e-04 | full+transverse | 10 | 10 | 9.744268 | 0 | 0.25573199 | True | True |

Target integrity: **True**. Failed checks: none.

## Target B: G=F^8

Status: ready; result status: completed.

Independent fold: alpha=0.0717, log10(T)=-0.1033462421844031. Critical multiplier modulus=1.00000000001, argument=0 turns; next modulus=1.10757496e-06. AD coefficients in physical `log10T`: a=-11.8997788501, b=-152.054516949; ghost direction=+1. U=0.000657658858195. Transverse seed=10502; max G=512.

Left-mode norm=8.20777; The transverse directions are defined by the left eigenvector. Transverse kick norm=0.000657659.

| epsilon | entrance | measured G | quadratic G | arctan G | quadratic error | arctan error | primary pass | arctan pass |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 3e-06 | full | 26 | 26 | 25.403409 | 0 | 0.59659107 | True | True |
| 3e-06 | midpoint | 13 | 14 | 12.701704 | -1 | 0.29829554 | True | True |
| 3e-06 | full+transverse | 25 | 26 | 25.403409 | -1 | -0.40340893 | True | True |
| 1e-05 | full | 10 | 10 | 9.5048919 | 0 | 0.49510812 | True | True |
| 1e-05 | midpoint | 5 | 5 | 4.7524459 | 0 | 0.24755406 | True | True |
| 1e-05 | full+transverse | 10 | 10 | 9.5048919 | 0 | 0.49510812 | True | True |
| 3e-05 | full | 4 | 4 | 3.4803086 | 0 | 0.5196914 | True | True |
| 3e-05 | midpoint | 2 | 2 | 1.7401543 | 0 | 0.2598457 | True | True |
| 3e-05 | full+transverse | 4 | 4 | 3.4803086 | 0 | 0.5196914 | True | True |
| 1e-04 | full | 2 | 2 | 1.0856167 | 0 | 0.91438335 | True | True |
| 1e-04 | midpoint | 1 | 1 | 0.54280833 | 0 | 0.45719167 | True | True |
| 1e-04 | full+transverse | 2 | 2 | 1.0856167 | 0 | 0.91438335 | True | True |

Target integrity: **True**. Failed checks: none.

## Scope and interpretation

The targets were selected before new passage outcomes. A is a new point on the familiar P2 fold; B is a different period at the assigned P8 fold. Fold coordinates and composed-map nested-AD coefficients were computed independently. The exact numerical design, runtime, analysis and entrances were privately pushed before trajectories. Every planned epsilon and entrance is shown, with no fitted passage coefficient or post-outcome replacement. Counts are in complete G steps; multiply by 2 or 8 to obtain model-map calls.

The quadratic map is the primary predictor. The arctangent is a separate approximation, compared without rounding. The transverse test is one fixed direction per target, of norm U in ker(l); it is not a statistical sample of perturbations. Passing these selected deterministic cases supports their finite-range local prediction. Failures constrain that accuracy and robustness claim. The evidence level is numerical validation. Generalization to other models, prompts and sampled-token behavior requires separate tests.

The external raw store contains full states at every counted G iterate, including first exit, with hash-bound receipts. The audit independently reconstructs all coordinates, crossings, coefficient contractions and scalar predictions; it does not re-run the language model. Private Git protects source and compact records; the bulky local raw store has no verified independent backup in this task.

## Pre-outcome prediction limits

The quadratic primary criterion passed 23/24 rows and failed its all-row conjunction: A midpoint at epsilon 3e-6 was predicted 61 G iterates and measured 59 (error −2). Every other error was zero or−1. The unrounded arctangent secondary criterion passed 24/24. This secondary result does not replace the failed primary endpoint. The [pre-outcome scalar audit](arctangent-scope.md) had already shown that the two one-iterate criteria cannot both pass this midpoint row; its allowed integer sets are disjoint. No predictor, tolerance or entrance was changed after outcomes.

The short P8 forecasts include one- and two-iterate cases, where one-iterate tolerance is weak relative accuracy. The test supports useful finite-range forecasts for these selected points, while falsifying universal one-iterate accuracy of the specified quadratic model over this 24-row design. It does not explain the two-iterate discrepancy; attributing it to transverse relaxation, higher-order terms or a different reduction would require new disclosed exploratory work.

## Evidence and scope

| Component | Evidence rung | Evidence and assumptions |
|---|---|---|
| Fold/orbit and coefficients |2, independent numerical implementation|Hash-bound setup attempts; full-state closure, primitive separation, composed derivatives; no interval enclosures|
| Prospective passage counts |2, numerical cross-check|Exact private freeze and independent tensor map; all 24 saved traces; one checkpoint/prompt and selected parameters|
| Outcome integrity|Saved-artifact audit|Independent code reconstructs every crossing, scalar prediction and hash binding; no independent model replay in this audit|
| Broader language-model behavior|Not tested here|No claim about sampled-token decoding, persona drift or intervention utility|
