# Task18 final saved-output comparison review

The completed comparison supports a close independent numerical reproduction of the available producer **p8b** orbit, all eight step Jacobians, and the saved loop spectra at the declared fixed setting. I accept this finite scientific comparison on the authenticated saved-evidence basis below. The original **p8** arrays are unavailable and remain unaligned. Original independent run-002 exited 1 during final status serialization; its complete earlier numerical events were recovered separately. This review does not change that exit status, issue an interval certificate, or certify historical producer assets.

I read the sealed comparison JSON, all eight phase comparisons, all eight forward rotation scores, all 64 matrix-pair summaries, the signed loop comparison and all eight cyclic summaries. I also read the coordinator's completed source/output/closure authority. This review reads saved JSON only; it does not load target arrays or rerun the model, Jacobians, matrix products, eigensolver, SVD, or comparator. The companion extract preserves every saved signed phase and loop scalar used here, all cyclic summaries and all pair summaries. Its absence of the long column ledger is editorial only: every original column comparison remains in the original report.

The fixed setting is Qwen2.5-0.5B-Instruct, 896-dimensional float64 state, layer 12 philosophy steering, alpha 0.071674, log10(T) = -0.115 and T = 0.7673614893618189, one soft feedback slot and the original 37-token prefix. Own x0 follows exactly 1,000 steps from the fixed start. The own source binds model/config/vector bytes and exact prefix IDs. Producer JSON agrees on saved parameters, layer and concept; its missing historical asset hashes and prefix IDs are explicitly unavailable. Producer token strings are retained next to own token IDs; the comparator did not load a tokenizer to independently verify their text.

The own closure is 3.250425216758018e-15 relative to mean phase norm; minimum pairwise relative phase separation is 0.1503826910030647. Both saved gates pass. Using the producer's phase-0 denominator gives own closure 3.756127612773231e-15; the producer's saved states give exactly its reported 2.0494006356669066e-15. These two normalization conventions must not be conflated.

Only forward rotation 0 qualifies under the prospective maximum state-mismatch rule < 1e-8. Its maximum mismatch is 3.922114626768719e-13, using the producer's mean phase norm. The seven other maximum scores, shifts 1 through 7, are 1.6421271449002122, 1.70980421213921, 1.6391861672883037, 1.5984474848487356, 1.6391861672881296, 1.7098042121390669 and 1.6421271449000459. No reversal, transpose or alternative phase choice was selected. This threshold identifies alignment; it is not an accuracy proof.

Each row below covers the whole 896 by 896 Jacobian, with all 896 column errors retained in the original JSON. Matrix relative error uses the producer Frobenius norm. The signed entry difference is own minus producer at the maximum-absolute-error coordinate.

| Phase | Relative matrix error | Signed largest entry difference | Own leading step singular value | Own entropy (nats) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.870725242522342e-13 | -1.490058076925038e-13 | 11.09125166134889 | 1.417355259261518 |
| 1 | 7.684012924156003e-14 | -5.961897642237091e-14 | 16.68344010466335 | 2.286559325022172 |
| 2 | 5.728542011942032e-13 | +7.044642647002775e-13 | 20.90777695714715 | 2.466674274381015 |
| 3 | 2.01006374090449e-12 | -1.410788152966802e-12 | 14.71235964456269 | 2.243194245140629 |
| 4 | 2.711743933890581e-12 | +6.076528169529638e-13 | 5.365040033070338 | 0.618859387339078 |
| 5 | 5.581836078510957e-13 | +5.221864607385385e-14 | 2.618888285279104 | 3.513731712510633 |
| 6 | 1.476528820856691e-12 | -1.04562192237978e-13 | 1.240081953580162 | 0.1112471041693738 |
| 7 | 2.295622351359331e-14 | +7.66053886991358e-15 | 10.76885627376739 | 2.145892605205607 |

The largest matrix relative error is 2.7117439338905806e-12 (phase 4); the largest individual column relative error is 1.4523034477960833e-11 (phase 4, column 713). The largest absolute matrix-entry difference is 1.410788152966802e-12 (phase 3, signed negative). State and Jacobian arrays are not bit-identical. All 96 phase scalar comparisons are nonzero and retained. The largest absolute step-singular-value difference is -1.5274892462002754e-11 (phase 3, first singular value). The largest absolute entropy difference is -9.134915046615788e-13 (phase 4). The largest absolute covariance-eigenvalue difference is -2.6183916146393926e-14 (phase 4, leading value). The largest absolute top-three probability difference is +2.1749269052406817e-13 (phase 3, leading probability). These are numerical differences, not silently rounded equality.

The own factorization, dense-direction and covariance gates pass at every phase. Maximum dense factorization relative error is 3.442218647050999e-14; maximum factorized-direction and dense-direction errors are 8.265928151997498e-14 and 3.011501335615533e-13. Covariance minimum eigenvalues are positive in all eight saved rows, with minimum 4.863336302525681e-10. Three independently composed eight-step JVP controls pass, with derivative errors 1.1169603757423053e-13, 4.731808350561387e-13 and 1.8022422904377204e-13, and zero saved primal differences. The producer's separate eight-direction identity check has maximum error 1.3417296425324618e-13. Its random directions differ from the own directions, so those checks are not paired estimates.

At phase 0, the complete saved leading signed multiplier and singular-value comparison is:

| Quantity | Own | Producer p8b | Signed difference |
| --- | ---: | ---: | ---: |
| /monodromy_singular_values/0 | 0.19289912009484797 | 0.19289912009508223 | -2.3425705819590803e-13 |
| /monodromy_singular_values/1 | 7.7754190535851817e-06 | 7.7754190536330391e-06 | -4.7857361519867969e-17 |
| /monodromy_singular_values/2 | 8.2457601275443578e-08 | 8.2457601270877806e-08 | +4.5657722834966958e-18 |
| /monodromy_singular_values/3 | 1.4360128419670964e-09 | 1.4360128382955986e-09 | +3.6714978106885732e-18 |
| /multipliers/0/0 | 0.011040155870127638 | 0.011040155870133969 | -6.3317406873153459e-15 |
| /multipliers/1/0 | -2.9611745420202473e-07 | -2.9611745422253282e-07 | +2.0508097021125103e-17 |
| /multipliers/2/0 | -6.8494606597323608e-09 | -6.8494606334712218e-09 | -2.6261139008612023e-17 |
| /multipliers/3/0 | 6.7798363883643812e-11 | 6.7798498353655759e-11 | -1.3447001194669729e-16 |
| /prod_sigma1 | 10679877.352872368 | 10679877.352895379 | -2.3011118173599243e-05 |
| /prod_sigma1_sigma2 | 42425505562.170418 | 42425505562.374176 | -0.20375823974609375 |
| /prod_sigma2_over_sigma1 | 0.00037195850685360242 | 0.00037195850685378581 | -1.8339279747592308e-16 |

All four compared leading multipliers have saved imaginary part exactly 0 in both runs, so their imaginary differences are 0. Their real signs are +, -, -, +. The fourth multiplier has a relative difference -1.983377437731208e-6 despite its small absolute difference -1.344700119466973e-16; the fourth singular value has relative difference +2.5567304920799093e-9. These small modes should not be described as having the leading mode's relative precision. All four multiplier-modulus comparisons and the complete 16-pair signed eigenvalue comparison grid remain in the companion and original report; no matching permutation or conjugation was chosen.

All eight own cyclic starting phases are retained. The table reports their saved leading multiplier and first two singular values; each complete eight-multiplier signed spectrum is in the companion extract.

| Cyclic start phase | Leading multiplier (real; imaginary 0) | First singular value | Second singular value |
| --- | ---: | ---: | ---: |
| 0 | 0.011040155870127638 | 0.19289912009484797 | 7.7754190535851817e-06 |
| 1 | 0.011040155870127627 | 0.26037398263841083 | 1.231335632401749e-05 |
| 2 | 0.011040155870127643 | 0.059473489240608343 | 9.4451098280273525e-06 |
| 3 | 0.011040155870127648 | 0.089785230318327053 | 1.1264810160416627e-05 |
| 4 | 0.011040155870127629 | 0.12295402125198324 | 2.401375228647857e-05 |
| 5 | 0.011040155870127608 | 0.62206436713293622 | 2.442869840880477e-06 |
| 6 | 0.011040155870127681 | 0.64370229350527453 | 1.1047101495218914e-06 |
| 7 | 0.011040155870127638 | 0.32875349025318801 | 2.9768162485271244e-06 |

Every cyclic summary has the same leading +, -, -, + real pattern, followed by two complex-conjugate pairs near 6.26e-12 and 3.84e-14 in modulus. The complex real and signed imaginary components are retained; they are not replaced with zeros. Phase-dependent singular values are expected, whereas the leading multipliers of cyclic products agree closely. The comparator compares producer loop spectra only at its recorded start phase 0; it does not claim eight independently observed producer cyclic spectra.

The scientific point survives: every saved one-step operator norm is greater than 1, with their product about 1.068e7, while the phase-0 loop spectral radius is about 0.01104 and the loop operator norm is about 0.19290. All eight own loop operator norms are below 1. A product of step norms is only an upper bound and loses the orientation information in the ordered product. A positive semidefinite covariance factor, its spectrum, or a large one-step singular value alone does not establish stability or instability of the feedback orbit. Here the ordered loop and the independent composed controls supply the relevant numerical evidence. This remains a local finite-precision orbit/linearization result at the fixed parameters; it does not establish a global basin, training benefit, native-generation behavior, a universal temperature law, or an interval theorem.

The reporting failures and provenance limits remain part of the result. Original run-002 receipt 4a3153f174bec42f034e245b63d5fe48978962908cbf82478ace3498b632b4fa says exit 1 and loop_gain_failed. The separately accepted recovery report f8f42ddb1dcd25ce13cd602f2a8aa670335d79b72cc0fbd21f5aed3a46825fef reconstructs previously complete saved events; its 56 summary-to-saved-scalar reconciliations are exact. It neither changes the original receipt nor reruns the model. Original total elapsed time is 2031.384801283013 seconds. Original peak RSS and backend identity were not recoverable; the comparison/recovery backend and RSS cannot fill those fields. The monodromy event timings are distinct from unrecoverable later intended summary timings. Earlier run-001 and its BrokenPipe launch failure also remain preserved.

Producer p8 has no original Jacobian/state archive here, and therefore has no paired matrix or loop comparison in this report. Producer p8b current bytes are hash-bound and its exact declared settings are retained, but original dirty-worktree execution, historical model/vector/prefix bytes and process-birth/termination evidence are not independently attested. The accepted finite basis is reviewed source plus stable saved artifacts and the disclosed producer record. Prior source/producer-summary exposure makes this an exploratory independent numerical check, not blinded fresh-data confirmation.

Inference, the actual process owner, authenticated the comparison's four output files, nine supervision files and four captured Git blobs, and reports the original supervisor/worker groups and sessions absent and tool session 14835 reaped with exit 0. I read that completed authority; I did not independently observe execution. The comparator itself took 0.9694336759857833 seconds and reports peak RSS 160532 KiB. Its original scientific_acceptance_pending field records its pre-review stage; this separate finite review supplies the scientific disposition without rewriting it.

This review accepts the available p8b reproduction and the stated missing-evidence dispositions. No extra numerical audit or scientific rerun is requested. Whole-task administrative closure and final delivery remain with Root and Inference.

Evidence bindings:

- Sealed comparison report: /data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-comparison-001/report.json; SHA256 5d1e63c58275c5f8500c0f242f81555b03921bf63d92dcfa98833a2bec6dbd5d.
- Actual source/output/closure authority: /data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-comparison-001-authority/report.json; SHA256 398388244c818e7d341ab2c8de5ed14bbc29354d06a81083ebae9c76e3498a6b; source commit 2e1fbc3db2d5043f76dd64a5ec6478ada38f9141.
- Comparator source SHA256 9f549e77472c783586e48215591f1505fa1ce652656c9e26c8bd35bef8e90263; admitted input manifest SHA256 cdbbaaa908d30635e76e8d1135aa93a19a59203e709a5d71a880095568ceb08c; selected inputs 112572607 bytes.
- Original own scientific source commit 7918af949fa60cb8bb123504e78c2dc959943d9c; plan SHA256 98a01706d756e3271f3f566dedf792719bf088055ba084c753b199b649b198c4.

- Editorial exact-field extract: /tmp/task18-final-comparison-review-extract.json; SHA256 6a7642aeff284b7481b0f721aaefdfc4771bcde893aa5fc5fb0baf74556dc8b9.
