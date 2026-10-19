# Task37 rounded-R pair: closed, numerical qualification failed

The original two-seed calculation and the first independent saved-data replay are closed. Both Newton solves converged and produced the planned evidence, but four of six composed-map finite-difference checks exceed the unchanged final relative-error threshold `1e-5`. Both orbit qualifications, scientific admission, certificate status and body-point eligibility remain false. This attempt does not establish coexistence at R and does not disprove the existence of the proposed orbits.

The exact target is binary64 `(alpha, log10 T) = (0.072912, -0.154104)`, period 6, philosophy layer 12, using the full manual Qwen map and the two pinned original seed rows. Rounded R is not a bisected flip crossing.

| Seed | FD direction | Final relative error | Gate |
|---|---|---:|---|
| 0 | random0 | 1.2609053230675582e-5 | fail |
| 0 | random1 | 6.392441542270864e-5 | fail |
| 0 | critical_real | 6.027947869868129e-5 | fail |
| 1 | random0 | 3.074552501761685e-6 | pass |
| 1 | random1 | 1.901207222740084e-5 | fail |
| 1 | critical_real | 4.840742422827987e-6 | pass |

All six directions show a quadratic interval over the four prescribed relative steps `1e-3, 5e-4, 2.5e-4, 1.25e-4`. The observed ratios are consistent with remaining truncation error; that pattern does not satisfy the endpoint threshold. No threshold was relaxed and no additional model evaluation was performed.

The saved-array checks reproduce all six phase closures and proper-divisor separation tests for each seed. The largest phase-return residuals are `1.6559905782167558e-14` and `1.8016236618532218e-13`. The independent checker recomputed all 896 column assemblies, eigenpair residuals and full spectra per seed, random/scalar comparisons, composed-FD rows and saved-logit token checks without a seed or root-summary replay error. Its failed qualification flags remain false; agreement with a failed gate is not a passed scientific validation.

For diagnostic context only, the leading real multipliers of the saved matrices are `-0.9952248297623896` and `-1.0005445296072357`, with next moduli `0.002185760770974642` and `0.004266383229002312`. The labels attracting and unstable are conditional on the unresolved FD qualification. Neither a pair of attracting orbits nor a classified crossing is admitted. Saved phase arrays have best cyclic shift 0 and relative separation `0.06398175170841057`; this remains a conditional array distinctness diagnostic. The original gated alignment remains null.

Original execution: source `8b56522882743d6c84afd2eccb6ec09616b7e9be`, session 8013 reaped 1, external elapsed at most `7763.898526` seconds. All 1,289 original outputs (96,764,991 bytes), five source blobs, original plan/card, ten inputs and the original host closure were authenticated. The intentional `SystemExit(1)` and `failure.json` remain immutable; the supervisor reports no runtime failure.

Saved replay: source `ab09687ed0b53b2cafc8d0df493185631bc001ff`, session 76531 reaped 1, external elapsed at most `2.827992` seconds. The single result, four source blobs, 24 preflight pins, original complete inventory before and after, and actual host closure are bound in `r-pair-001-audit-001-completion.json`. The replay reads 1,276 original files; complete membership is authenticated separately over all 1,289. The short replay ended before Root's separate live check; only the supervisor's recorded post-exec identity and Root's subsequent host closure are claimed. A prior private-checkpoint race refused before allocation or any checker call and is preserved in `r-pair-001-audit-preallocation-refusal-001.json`.

Source and file hashes bind the implementation, inputs and records, not a hardware instruction trace. The checker does not independently replay left-eigenvector condition estimates, decoded text/embedding readouts, full Newton/GMRES decisions, or the effective internal arithmetic. Its `scientific_verdict_admitted` and `certificate_passed` remain false. Existing closed certificates concern other exact cells and are unchanged.

No owned numerical worker remains. No automatic retry, full dense rerun or body-point launch follows. A future bounded smaller-offset composed-FD diagnostic on the exact saved centres could test the truncation-error interpretation, but requires its own disclosed source, card and capacity admission. This result is closed and must not be silently relabeled after such a prospective test.
