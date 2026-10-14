# Task 17 — empirical scale and scope review

The small dense-model scans support chaos without semantic steering, but the current paper overstates what the large-model sweeps establish. The landed 70B/72B evidence is a mixture of coarse-grid negatives, sparse positive finite-horizon estimates, and explicitly failed precision checks. It supports narrower and shifted candidate regimes, not “contraction at every temperature,” a monotone law of model size, or a causal attribution to post-training. Most existing C1 percentages agree with the frozen log. Task16's independent raw tabulation remains separate.

## Authority and limits

Report-only review of commit `6fcd6ba78c641c2a4b8dcd9ab8a87f32778f2940`, export `/tmp/task17-source-20261014-6fcd6ba7`; manifest SHA-256 `e39408a60b07cc6b353b933c0b3e50465d4704d7667af87e0a3f9c44cdb636bf`. All 43 exported file hashes and byte sizes matched. Manuscript `paper/llm/main.tex` SHA `47452e4083ccb1dec00639ebd0c386588359bb7a3b51453319991d7de2d9189b`; log SHA `3e5f5befc23bc27974eb9c2d753e30c51c40060282cfbbe98bda9a88c8882268`.

The paper was last changed at **08:53 UTC** (commit6246ea60); the export includes scientific scale results through **09:23 UTC**, and later independent Task19 delivery. Below, `P` means line in exported main.tex and `L` means line in exported research-log.md. Source and compact JSON reads only: no raw arrays, model, producer analysis, pod/process access, Git operations, or paper edits. Reported run numbers are not a fresh validation of raw evidence. No new external citation was needed or added.

## Corrections to make

### S01 — replace the global 70B negative and supply its missing evidence subsection (high)

**P1721–1722:** “The 70B-class models we tested sit at the stable end: without steering, their loops contract at every temperature (Section…scale).” Section scale, P862–1019, supplies no 70B/72B results at all. Its final model is Qwen3.5-35B-A3B. Even before the paper's08:53 update, B7 at08:51 had five positive single-offset Qwen72B cells; subsequent B5/B6 add positive estimates in both Llama70B checkpoints (table below). The earlier B1/B2 zero counts apply to particular coarse grids only. Negative finite-horizon twin estimates from one start also do not prove contraction of the map or all trajectories.

**Correction:** add the completed scale evidence with explicit grid, horizon, offset and status; replace the sentence by: “The first coarse unsteered70B sweeps found no positive finite-horizon exponents, but later sweeps found sparse candidate chaotic regimes. Their widths and precision qualification remain limited by the sampled grids and offset checks.” Do not call B6/B7 robust two-offset confirmations: their1e-9 checks failed. Preserve the original zero counts rather than replacing them with adaptively targeted fine-grid shares.

### S02 — keep the SAE/random conclusion local; 70B directions differ (high)

**P1617–1618 and P1698–1700:** feature directions steer “no differently” from contrastive/random directions. **P40–41, P108–109, P149** also generalize “as readily.” The existing8B numbers support occurrence across the four tested directions, not statistical equivalence; layer16 random2 is17% versus28% concept, and layer19 spans26.5–38.9%. Completed70B controls differ more strongly: concept44/1024, SAE hostility36/1024, Gaussian11/1024, random-live SAE2/1024. The compact JSON is complete (`unfinished:0` each), but this is finite-grid descriptive evidence, not independent sampling of directions or a semantic causal effect.

**Correction:** “Chaos occurs along contrastive, random and selected SAE directions in the tested0.5B/8B controls; prevalence depends on direction and model. In the70B control grid it is rarer and is higher along the two hostility-related directions.” Restrict the small-model abstract claim by naming those controls. Do not import L23091's heading “only along the hostility directions”: the same result records11 and2 chaotic cells along the other directions. Keep the selected SAE feature as a direction selected by a frozen category-profile rule, not a verified unique hostility mechanism.

### S03 — date the search census precisely, rather than presenting it as the current total (medium)

**P1753–1756, P1773–1776, P1784, P1787–1788:**34maps/987,308cells/27float64/six models,29reported/893,868cells,8loaded/7reported. L22642–22644 identifies this as a census of raw data at **00:19 UTC14October**, using main.tex5077125e. It precedes new L14, F6, MoE and large-model results. “Before steering-control maps” does not explain all of those omissions. Current paper itself includes MoE, already beyond the original six mapped models.

**Correction:** label this table the “initial exploratory census at00:19 UTC14October,” state subsequent additions separately, or refresh the inventory once. Do not invent new exact totals by adding log entries: distinctness, merged tiles, sweeps and partials need a consistent rule. The packet contains the log's census summary, not the original complete enumeration, so I cannot independently certify the exact987,308/893,868 totals. Their difference is93,440 and34−29=5, consistent internally. Keep the six original prospective tests as a dated cohort; do not imply that later SC/PS/B7b have completed or passed.

### S04 — the unsteered figure caption collapses unresolved cells into periodic ones (medium)

**P1624–1625:** “Red: no period…positive exponent; other points are periodic.” L22692 (C1 Qwen7B entry) reports65chaotic plus **6 more without a period**; L22734 reports32chaotic plus **3 more without a period** for OLMo. P271–272 itself allows5 nonperiodic temperatures below the bands. Thus non-chaotic is not equivalent to periodic.

**Correction:** state the exponent threshold (`>1e-3`, as the analyses) and distinguish detected periodic points from unresolved/no-period non-chaotic points. Do not change a failed or missing period gate to a periodic label. I did not inspect figure pixels or raw cells.

### S05 — incompatible maximum-period caption (medium)

**P954:** Llama64² white means “no period up to48”; **P938** says windows up to32. The L13/L14 settings in the log describe maxperiod32, with L14 explicitly retaining L13 settings (L22590–22593); the current runner's L13 command at tools/runpod_soft_runner.sh544–554 adds no48 override. P965 correctly uses32 for the later128² figure.

**Correction:** reconcile the64² caption with its actual saved run metadata; the supplied source/log support32, not48. This is a source/report mismatch, not a raw verification of the figure generator.

### S06 — large-model precision status must accompany the new numbers (high, integration requirement)

**P28–30, P1655–1659**, and any new scale paragraph: float64 is not a successful effective-precision test. B7 L23180 records1e-9 giving positive exponents even at periodic cells and explicitly invalidates the two-offset classification. B6 L23222 records the same issue and a period tolerance near the numerical noise floor. B5 is a one-offset run. These findings do not automatically invalidate every1e-6 result, but do prevent claims that all large-model labels passed the stated multi-offset gate.

**Correction:** preserve the failed1e-9 checks, call B6/B7's1e-6 positives candidates, and separate period-unresolved states from unfinished computations. B7b is only a prepared follow-up in this snapshot. Do not silently reinterpret hidden_chaos.py's original multi-offset `classify()` as a successful one-offset confirmation. Its offsets are absolute state distances (lines58–75); gain_split.py uses relative offsets (lines54–73), so offset claims must retain those conventions.

### S07 — scale-summary source counts the shared72B endpoint twice (medium)

**tools/scale_summary.py27,39–67:** B1[-1,.3]128points and B4[.3,1.2]64points are concatenated; both include.3. The core predicate admits both copies. This produces **129core observations and192overall observations**, versus **128distinct core and191distinct overall temperatures**. It does not change the zero numerator in these two runs, but changes the denominator and gives the repeated temperature double weight. The source also excludes B7/B6 hidden_chaos fine lines, and still uses old narrow7–8B reference sweeps despite the later SCx wide-run plan.

**Correction:** report runs separately or explicitly reconcile the shared endpoint before claiming a count of distinct temperatures. Label the resulting figure “regular-grid survey” and show targeted fine-line evidence separately. Preserve frozen SC scoring on[-1,.3], with wide-range results beside it. Do not pool targeted fine grids into an unbiased prevalence estimate. Missing/unfinished/nonfinite rows must be reported separately before a new aggregate claim; this loader currently checks only file existence, not complete finite row validity (lines43–67). No evidence here that a current admitted scan actually contains NaNs.

### S08 — correct the unresolved SAE70 count (medium, log/compact consistency)

**L23037:** “394 cells without a period… and exponent<=1e-3” for SAE hostility. The compact `fig-controls-llama70b.json` gives630periodic,36chaotic,1024complete;394 is **all no-period cells**, leaving **358 non-chaotic no-period** cells. The four-way disjoint counts are in the table below.

**Correction:** say394without a detected period, of which36 have positive qualifying exponents;358remain unresolved by those labels. Do not count all394 as non-chaotic. The other compact counts and reported44/36/11/2 percentages agree exactly.

### S09 — avoid causal size/post-training wording when integrating G2b and B5 (medium)

**Potential additions to P862–1019 / P1735–1741:** L23208's heading says scale and post-training “both reduce” chaos; L23254 says the network factor “carries the scale effect.” Same-release Llama3.1-8B versus70B is a useful comparison, but two trained checkpoints and differently resolved temperature grids are not a controlled intervention on parameter count. Llama3.3 versus3.1 is likewise not an isolated post-training intervention established by this packet.

**Correction:** use “size-associated difference in these checkpoints/grids.” G2b's finite twin factorization is along the evolving perturbation, not a causal ablation: gain_split.py65–85 defines norm-relative factors with a boundary term; gain_compare.py45–48 compares medians within broad entropy bins. Equal entropy is not equal covariance, direction, temperature or attractor. Report per-bin sample counts and missing bins. The0.5–1.3 lower Llama3.1 network factor applies approximately to bins1–10nats; above10 the difference is only.18. The conspicuous Llama3.3 readout reduction is at1–3nats, not all entropy bins.

### S10 — retain prefix and unfinished-scaling boundaries (medium, integration requirement)

**P1653, P1735–1741:** one-prefix limitation remains appropriate for the completed scale observations. docs/prefix-scale-test.md12–35 defines2development plus8held-out prefix units,48coarse and24adaptive-transition temperatures, but no PS outcome is in the frozen log. SC has only the32B result landed here; it is not a completed within-family ordering test.

**Correction:** keep PS/remaining SC/SCx/B7b/Z70 as pending, not confirmation. When they land, report8held-out units separately from2development units, missing paired results without imputation, both stages separately, and preserve original scoring ranges. Do not infer a larger-model universal absence from a single fixed prompt. The prefix plan's “both well above” a2e-8 floor is a design assumption;1e-7 is5× that stated floor, and the72B floor is not itself a measurement for every7B/14B prefix.

### S11 — small scope wording cleanup (low)

**P1651–1653:** Llama8B has “coarse maps only,” while P957–965 explicitly presents two128² maps. Replace with “maps, including128² refinements, but no continued orbits in those families.” The numerical percentages in P935–948 and C1 agree with the supplied descriptive log; this correction concerns the scope description, not those numbers.

### S12 — hardware and result-count cutoff are stale (medium)

**P1825–1829:** “one GPU per run,” H100/A100 only, and138runs with results. The completed large-model log explicitly uses2×H200/H100NVL; both number of GPUs and H200 must be included if these results are added. The tracked manifest's supplied schema extract is already generated02:04:55UTC with165result rows, not138. That02:04 manifest is itself earlier than the later large-model runs.

**Correction:** identify the exact historical cutoff for138, or update the manifest and paper together. Say “one or two GPUs per run as recorded,” addH200, and distinguish results present from complete provenance. The Root-supplied source-only extract reports151nonempty result hashes and57nonempty code/data hashes out of165; Root's separate review addresses the overbroad “for every run” claim. I did not read raw pod receipts or initiate any hardware investigation. Supplement `/tmp/task17-cusp-source-supplement/runpod-runs-schema-and-scoped-records.json` binds tracked manifest source SHA `97d6f0fcc0d32b81d93d86863f6ba20ab9dd1e7e2ab2abc67e981f9d6a93ca23`.

## All landed large-model evidence to carry into the revision

These are **producer reports/compact summaries**, not an independent raw-data replay. Counts are tied to their own grid and horizon. All logarithmic temperature ranges below use log10T. Negative finite-time exponents are not a global contraction proof.

| Evidence and log lines | Reported result | Required interpretation |
|---|---|---|
| B1 Qwen72B,06:44,L22954–22956 |0/128 on[-1,.3],1000+2000steps; max−2.76;128detected periodic |Initial coarse-grid negative, retain unchanged. |
| B2 Llama3.3-70B,06:51,L22974–22976 |0/128 on[-1,.3]; max−.265;19no detected period |Negative estimates from fixed start, not “all periodic” or all-temperature exclusion. |
| S70P,05:26,L22910–22912 |FVE .460/.4655/.4653 at50/51/52; selected hostility38767,random55957 |Documented hook51 only narrowly best; finite selected-feature comparison, no universal SAE-semantic conclusion. |
| S70M complete,08:12,L23091–23093; compact JSON |44/36/11/2 of1024 for concept/SAE-hostility/Gaussian/random-SAE; each alpha0column0/32 |Allfour maps complete; direction-dependent positive labels, not “only hostility.”300+300steps,32²grid,periodlimit32. |
| B4 Qwen72B,07:57,L23080–23084 |0/64 on[.3,1.2],max−.447; entropy jump .357–.386 |Coarse negative over wider range; narrow transition motivated adaptive B7. |
| B3 Qwen72B steered,08:28,L23120–23124 |8/1024 (.78125%),max.354;109no-period |32²/300+300steering grid. Its8cells must not be described as robust offset-invariant chaos without an actual such check. |
| B7 Qwen72B,08:51,L23178–23189 |At1e-6,5/64 on[.355,.395],positive cells.3728–.3760,exponents.082–.192;1e-9invalid |Retain candidates and explicit failedprecision gate; B7b has plan only. |
| B5 Llama3.1-70B,08:58,L23208–23212 |3/128 on[-1,.3], at.116/.136/.198,exponents.042/.298/.414;31no-period |2.34375% versus8B18.75% (=8-fold grid-share ratio), singleoffset and different resolution; not isolated causal scale effect. |
| B6 Llama3.3-70B,09:10,L23220–23224 |At1e-6,2/64 on[.25,.60], at.2500/.3056 with.137/.126;1e-9invalid |First point is **inside** B2's nominal range; heading “above B2's range” applies only to.3056. Different grid can reveal an unsampled point inside old range. |
| SC2 Qwen32B,09:11,L23232–23236 |0/130 core[-1,.3],1/160 overall[-1,.6]; at.58 exponent.356,H10.9 |Only this new SC component has a result here; retain wide-range positive despite coreP1pass. |
| G1/G2/G2b,07:39/07:55/09:23,L23055–23057,L23074–23076,L23252–23256 |24temperature gain decompositions,500+1000; G2disk failures followed by disclosed G2b completion |Include all completed models and failed attempts; descriptive entropy-bin comparison with boundary term, not causal attribution. |

Disjoint S70M counts computed only by arithmetic on the included compact summary (not by reading cells):

| Direction | Complete | Detected periodic | Chaotic | No period, non-chaotic | Chaotic share |
|---|---:|---:|---:|---:|---:|
| Concept hostility |1024|512|44|468|4.296875%|
| SAE hostility38767 |1024|630|36|358|3.515625%|
| Gaussian random1 |1024|589|11|424|1.07421875%|
| Random-live SAE55957 |1024|799|2|223|.1953125%|

G2b's landed moderate-entropy-bin medians (network/readout, log units per step) are: Llama3.1-8B at1–3/3–6/6–10: −.05/+.19, +.24/−.10, +.28/−.20; Llama3.1-70B: −.79/−.10, −.24/−.01, −.98/−.16; Llama3.3-70B: −.59/−2.33, missing, −.19/−.24. Qwen7B: −.90/+.34, −.03/−.13, missing; Qwen72B: −1.46/−1.72, −.94/−1.04, missing. These match the log's printed values; bin counts/raw uncertainties were not supplied here. They must not be presented as inferential confidence intervals or exact AD Jacobian factors.

## Existing numbers that remain source/report-consistent

- Four long unsteered sweeps: Qwen0.5B7/512=1.3671875%, Qwen7B65/512=12.6953125%, OLMo7B32/512=6.25%, Llama8B96/512=18.75%. P1594–1600's counts and rounded ranges/maxima agree with the frozen log. Llama's41% is restricted to[-.3,.3], not its full512point range. Near-zero1.5B/0.5B columns are correctly called near-zero and must not be silently substituted for exactalpha0.
- Qwen0.5B C1 full maps:7.6/8.3/12.5% Gaussian versus6.8/7.5% concept; final16384cells per map, including preserved timed-out lower96rows plus completed upper32rows. The earlier2.5/6.5/5.7% provisional partial numbers are obsolete. `analyze_controls.py` explicitly masks unfinished rows and uses completed denominators; Task16 still must independently check actual raw rows.
- Llama8B C1 layer16 .282/.259/.173 and layer19 .278/.290/.265/.389 round as printed. L14 .2856 rounds29%. SAE readout overlaps .19/.11 and reconstruction ranges .64–.76 versus.52–.56 match the descriptive log; they establish neither unique semantics nor mechanism.
- MoE paper P1005–1019 preserves the corrected52/1024=5.078125% irregular hostility labels, no philosophy positives, thinking-prefix amendment, and routing-jump limitation. Keep that correction; do not reuse the superseded “no chaos steered” heading. “Negative exponent on each tested trajectory” is more precise than claiming map contraction.
- The exact-map certificate, Task18/19 reduction claims, sampled-dialogue statistics and theory citations are covered by the other Task17 lanes and prior accepted reports; this note does not repeat their audits or confer broader proof status.

No additional computation is requested by this review. The fixes are reporting corrections and source-scoped denominator/status checks; the separate Task16 tabulation should remain a separate task.
