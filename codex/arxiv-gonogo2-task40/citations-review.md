# Task40: citation and external-review calibration lane

GO WITH FIXES, for this lane only.
Six findings: three must-fix and three should-fix.
A3 CPU/GPU agreement is outside this lane; no numerical evidence was rerun here.

This review binds article `bb42a82189da91c37f50804c077832df9e2237ae`, compared with `4ca5b5e35bc766581a5e2d326858fcd0edc312bb`, and research `c567545ca93f90324dae75be2c333c8ac452dab5`. Exact article `main.tex` is 164,521 bytes, SHA256 `b0ecb1d8b74d19bb0b1a6b87c402846679f12b50346d135f9a0beecf45d562f1`; `references.bib` is 36,000 bytes, SHA256 `d71ac43d72519f400d1dafa19db0b3de12d1358bbaa0d50c806902eb0b2b838e`. Locations below refer to that snapshot, not later edits.

The bibliography diff and independent difference of citation keys used in `main.tex` identify exactly six additions. All six primary metadata records match the entries. This does not independently reproduce the cited research or establish the article's priority claim. The first and third external reviews are available here only as accepted-point summaries in the frozen research log, not their original 27- and 18-page reports. Consult 24's full review and adjudication were read. Consequently this is a complete check of the available accepted-point lists, not a claim to have read all three original reviews.

### F1. Abstract still attributes the full-distribution operator to published Soft Thinking [must-fix]

Where: `main.tex`, Abstract, lines 43–44.

Claim: “Fed back through its own soft next-token distribution over the full vocabulary, as in Soft Thinking, a pretrained language model defines a discrete-time map”.

Evidence: The official [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/f7396d1c54d51416958d63e285377103-Paper-Conference.pdf), page 5, applies top-k/top-p filtering, retains top-n tokens and renormalizes their weights. The [publisher record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7396d1c54d51416958d63e285377103-Abstract-Conference.html) identifies the paper. The accepted mathematical-review summary at 02:16 says this distinction was fixed in the abstract and introduction. Introduction lines 84–85 distinguish the operators, but the abstract still identifies them. This matters because filtering can change smoothness and the map being studied.

Correct wording: Delete “as in Soft Thinking” from this sentence. If the attribution is retained, say: “We study a smooth full-distribution variant of Soft Thinking; the published implementation filters and renormalizes the distribution.”

### F2. Introductory synchronization claim remains broader than the revised evidence [must-fix]

Where: `main.tex`, Introduction, synchronization summary, lines 167–168.

Claim: “two copies of the loop coupled through their soft tokens synchronize near the coupling that the theory of coupled chaotic maps predicts from the loop's own Lyapunov exponent”.

Evidence: Consult 24 B01 requires distinguishing a typical-orbit transverse threshold from global synchronization. Section VIII now makes that distinction. Table VII itself gives, for the near-start 10,000-step, 1e-8 criterion, cell 3's bracket `(0.12,0.15]` versus predicted `c*=0.089`, and no synchronization through 0.19 for cell 2 versus `c*=0.177`. Far-start runs also reach other attractors. These are internal article/review consistency checks, not a fresh computation of those values. Lines 1553–1559 and 1623–1631 correctly limit the prediction; the introductory sentence loses those limits.

Correct wording: “Coupled copies have transverse growth rates that approximately follow the prediction from the single-loop Lyapunov exponent; observed synchronization thresholds additionally depend on initialization, duration, tolerance and attractor selection.”

### F3. The cited mixture threshold needs its assumptions [should-fix]

Where: `main.tex`, “Where steering and temperature enter”, lines 291–293.

Claim: “In a two-token reduction with linear transport, the same inverse temperature multiplies a scalar coupling, and the product's critical value 2 separates the washout of a mixture from its takeover by one token”.

Evidence: Backour's [primary v1](https://arxiv.org/html/2609.02049v1), Sections 3.1–3.2, reduces the imbalance to a tanh recurrence with both coupling and field terms. The balanced-state threshold assumes zero field and positive coupling; the constant-coupling case crosses at 2, while time-dependent results use uniform coupling bounds. Remark 1 discusses the nonzero-field qualification. The article's sentence omits these assumptions. The source is a preprint, not an independently replicated result here.

Correct wording: “In Backour's unbiased two-token reduction with linear logit transport and positive constant coupling, inverse temperature rescales the coupling; crossing the value 2 destabilizes the balanced mixture.” If extending this to variable coupling, also state the source's uniform-bound assumptions.

### F4. Reconcile the logged eigenpair-residual limitation with saved diagnostics [should-fix]

Where: `main.tex`, cascade precision paragraph, lines 1104–1108, and flip qualifications, lines 1123–1126.

Claim: “The precision of the multipliers limits how far this can be pushed.”

Evidence: The frozen research-log entry at 01:37 says eigenpair residuals of the longest monodromy products were not done and were to be stated as a limit, together with crossing speeds and cubic coefficients. The latter are explicit at 1123–1126. However, Root has identified an action-based `ritz_residual` in the saved P384 flip record (`crescent-t-0.156-d/cascade.json`) and a source computation using the monodromy action. The log may mean a separate dense-product residual, or may itself be inaccurate. Root is checking the longest-period records. The log alone cannot justify adding an absolute “not computed” sentence.

Correct wording / action: Reconcile exactly which residual was computed for each reported late product and which independent checks remain absent. Then disclose that precise limitation if material. Do not add the earlier proposed blanket “eigenpair residuals were not computed” wording. This item remains unresolved in this lane pending Root's raw/source check; it is not evidence that a multiplier is wrong.

### F5. Exact coupling-formula attribution remains unverified [should-fix]

Where: `main.tex`, Section VIII, transverse-stability derivation, lines 1554–1556.

Claim: “This is the transverse-stability condition for identical chaotic systems coupled in this way” followed by `fujisaka1983stability`.

Evidence: The [primary publisher record](https://academic.oup.com/ptp/article/69/1/32/1835380) verifies Fujisaka and Yamada's title, journal, volume, issue, pages and DOI, and its abstract supports a Lyapunov-stability framework for synchronized coupled systems. It does not expose enough full text to verify this exact discrete output-coupling convention. Consult 24 R01 explicitly asked for a precise original-source passage or a broader historical attribution. The manuscript's own derivation establishes its formula; this review does not assert the citation is false.

Correct wording: “The calculation above is the transverse-stability condition for our symmetric output coupling; it follows the Lyapunov-stability framework of Fujisaka and Yamada.” Alternatively supply a verified passage in that primary source establishing precisely this convention.

### F6. Data Availability describes publication that is not established [must-fix]

Where: `main.tex`, Data Availability, lines 1770–1778.

Claim: the repository materials “are openly available” and the complete certificate evidence “is archived at Zenodo” at DOI `10.5281/zenodo.23488416`.

Evidence: The current human Task40 brief explicitly describes this DOI as reserved and not yet published. The applicable repository policy requires the repository to remain private; no public-release authorization is implied by this audit. Thus the claimed submission state is not established. Root's unsuccessful public URL checks are access failures, not evidence of an independently observed 404. This finding requires no archive-content audit, which the human has deferred.

Correct wording / condition: Before submission, make both statements match the actual authorized access state. If they remain private/unpublished, say that the code, records and certificate evidence are available from the author on request and identify the Zenodo DOI as reserved for a planned deposit, rather than already archived. If separately authorized publication occurs first, verify the public landing pages and contents before retaining the present-tense availability statements. This review does not authorize repository visibility changes or publication.

## Added-citation coverage

| Key; bibliography lines | Verified primary metadata | Claim-specific coverage |
|---|---|---|
| `backour2026mixture`; 958–961 | Ali Backour, *The Dynamics of Continuous Mixture Collapse in Language Models*, 2026, arXiv:2609.02049, v1 2 September 2026, cs.LG. [Record](https://arxiv.org/abs/2609.02049), [v1](https://arxiv.org/html/2609.02049v1), [DOI](https://doi.org/10.48550/arXiv.2609.02049). | Introduction 109–112 broadly supported. Threshold sentence 291–293 needs F3's assumptions. |
| `liu2026jacobianscopes`; 963–968 | Toni J. B. Liu, Baran Zadeoğlu, Nicolas Boullé, Raphaël Sarfati, Gurbir Arora, Christopher J. Earls, *Jacobian Scopes: token-level causal attributions in LLMs*, 2026, arXiv:2601.16407; accessed v4, 15 June 2026, cs.CL. [Record](https://arxiv.org/abs/2601.16407), [v4](https://arxiv.org/html/2601.16407v4), [DOI](https://doi.org/10.48550/arXiv.2601.16407). | Lines 245–249: Section 4.2 equation 5 and Appendix A.9 equation 15 support output Fisher covariance and its token-level pullback. |
| `wang2026fishback`; 970–974 | Sihan Wang, Jiayi Zhao, Qingyan Cao, Hongbo Yao, Lin Shu, *FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers*, 2026, arXiv:2605.17231; accessed v2, 18 August 2026, cs.LG. [Record](https://arxiv.org/abs/2605.17231), [v2](https://arxiv.org/html/2605.17231v2), [DOI](https://doi.org/10.48550/arXiv.2605.17231). | Lines 245–249: Sections 3 and 4.1, Theorem 1, equation 5 support covariance/pullback geometry for intermediate-layer steering. |
| `wang2026logitfixed`; 976–979 | Tongxi Wang, *Sharp Spectral Thresholds for Logit Fixed Points*, 2026, arXiv:2605.15651, v1 15 May 2026, cs.LG. [Record](https://arxiv.org/abs/2605.15651), [v1](https://arxiv.org/html/2605.15651v1), [DOI](https://doi.org/10.48550/arXiv.2605.15651). | Lines 368–369 correctly restrict the claim to affine softmax feedback. Section 3.1, Theorem 2 gives contraction when beta times the tangent spectral norm divided by 2 is below 1; no full-transformer theorem is inferred. |
| `zheng2023deqnar`; 981–986 | Zaixiang Zheng, Yi Zhou, Hao Zhou, *Deep Equilibrium Non-Autoregressive Sequence Learning*, Findings of ACL 2023, July 2023, pp. 11763–11781, DOI 10.18653/v1/2023.findings-acl.747. [Publisher](https://aclanthology.org/2023.findings-acl.747/), [paper](https://aclanthology.org/2023.findings-acl.747.pdf). | Lines 108–109: Section 3.2 supports expected-embedding feedback and equilibrium/root-finding in non-autoregressive decoding. |
| `hofmann2026map`; 988–992 | Thomas Hofmann, *The Map Behind the Flow: Finite-Step Gradient Descent as a Dynamical System*, 2026, arXiv:2607.04993, v1 6 July 2026, cs.LG. [Record](https://arxiv.org/abs/2607.04993), [v1](https://arxiv.org/html/2607.04993v1), [DOI](https://doi.org/10.48550/arXiv.2607.04993). | Lines 121–123 and 1110–1111: solvable training maps, period doubling and Feigenbaum ratios are supported by the introduction, Section 2.8 and appendix. This is training dynamics, not fixed-weight transformer inference. |

The five arXiv entries have no journal volume/pages on the accessed primary records. Their preprint status is appropriate; absent venue fields were not guessed. Existing-entry metadata edits are distinguished from new keys in the evidence JSON. This is not a repeat of Task39's full bibliography audit.

## External-review coverage and remaining limits

**Bifurcation review, logged 01:37:** quadratic-universality assumptions/higher-order caveat are present at 1065–1068; unestablished renormalization hypotheses at 1111–1114; simple -1 crossings/doubled orbit and the subcritical contrast at 1062–1064; uncomputed crossing speed/cubic coefficient at 1123–1126; late-product rounding at Table VI and 1112–1114; the distinction between a zero exponent and crossing classification at 1115–1116; fold genericity at 861–864; cusp coordinate convention at 1374–1382; A/B pseudo-crossings at 1376–1379; shrimp organization versus a local normal form at 1288–1292 and unresolved legs at 1447–1450. Table VIII is present at 1661–1688 and numerical/certified scope at 1693–1695. F4 is an unresolved log/raw discrepancy, not an established missing calculation. Numerical values, the trace/ditch row's categorical wording, and cusp/fold/cascade calculations are assigned to Root's raw-evidence lane.

**Mathematical review, logged 02:16:** the introduction distinguishes filtered feedback (84–85), but F1 remains in the abstract. Mixture-of-experts routing restrictions are stated at 214–216. Literature-search dates are explicit at 104–106; no new exhaustive priority search was performed. The energy argument has strict-descent and accumulation-point qualifications at 446–448; Root was separately notified that “an orbit need not converge” is stronger than “this descent estimate alone does not establish convergence.” The P8 passage at 1187–1206 specifies exact stored weights, model dimensions, prefix, binary64 parameters, real temperature, radius, release hashes and trust boundary; those are textual checks, not a new proof audit. Finite-time chaos qualification occurs in the abstract, introduction and discussion. Actual Zenodo publication remains Root's independent release check.

**Consult 24, full review and adjudication:** B01's typical-orbit limitation is incorporated in Section VIII (1507–1510, 1553–1559, 1615–1617, 1623–1631), but F2 remains in the introduction. B02's protocols, tolerances, horizons and brackets are tabulated at 1574–1593 and discussed at 1598–1604. B03's “on-off type” bursts and qualified bubbling interpretation occur at 1605–1614, including the missing unstable-orbit calculation and unisolated rounding mechanism. S02's top-1 readout limitation is at 1540–1541. B04's approximate agreement is qualified at 1519–1523 and 1598–1600. S01's fixed/varied/untested protocol dimensions are described at 1527–1530, 1561–1571 and in Table VII; no full factorial experiment is inferred. S03's packet/log range correction is not an article value independently recomputed here. F01's distinction between mean distance and point classification is explicit at 1517–1521; rendered-figure checking is a separate lane. R01's Pecora/Carroll historical role and Platt/Ashwin framework roles are separated; F5 records the remaining exact-source limitation.

Relevant historical primary records: [Pecora and Carroll, PRL 64(8), 821–824 (1990)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.64.821), [Platt, Spiegel and Tresser, PRL 70(3), 279–282 (1993)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.279), and [Ashwin, Buescu and Stewart, Physics Letters A 193(2), 126–139 (1994)](https://www.sciencedirect.com/science/article/pii/0375960194909474). The last was accessible only as a primary-publisher search-index result: title, venue, issue, pages, DOI and abstract were retrieved, but its author list was not freshly verified from that result. Full-text source-specific mechanism validation is not claimed.

Completed: six added-key metadata checks and claim-specific primary text checks; bibliography/cited-key delta agreement; available accepted-point calibration; consult24 full-review mapping; snapshot hash binding. Partly completed: exact Fujisaka coupling attribution and Ashwin full record, with the limitations above. Skipped here: unchanged Task39 full bibliography, new numerical/proof/model runs, full original human-review documents unavailable locally, raw synchronization recomputation, archive publication, and PDF placement/build (separately assigned).

Inspection failures preserved: two initial `git show` hash pipelines were accidentally invoked in the research repository instead of the article repository and failed with missing-path errors; their empty-stream SHA values were discarded and recomputed with explicit `git -C` and pipefail. A later read-only search used nonexistent `/data2/neural-shrimp/log.md`; the canonical log is `docs/research-log.md`. No files or evidence were modified by these errors. No paid review, private-manuscript upload or scientific execution occurred.
