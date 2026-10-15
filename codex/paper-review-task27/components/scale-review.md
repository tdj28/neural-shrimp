# Task 27 — replication and scale review

The tested maps show that sensitive soft-token feedback occurs across several model sizes, while its location depends strongly on temperature, prompt and steering direction. That finding survives the review. The current text needs narrower precision and causal claims, corrected temperature ranges, and a clearer separation of old coarse scans from later checks. This review requests reporting changes, not additional experiments.

## Scope and authority

Reviewed frozen `paper/llm/main.tex` lines **1489–1944** at private commit `66ffa102c56f6d5b22f116a1bb194f503c75562e`, paper SHA-256 `d78a22090469e1dc32e67e715c0a2f51cc538b6728496c04974d9fc16a99b05f`. The source packet is `/tmp/task27-source-001` (durable copy `/data2/neural-shrimp-data/codex/paper-review-task27/source-001`), packet manifest `b875fb110e1323b570bc7aaa450d9bcd3eb7ce6ad3b7b9cdbe5a8d0ddff9e784`.

All assigned manuscript lines were read. I read the scale/prefix/start-state scoring sources, frozen designs, all 15 scale-summary rows, all 14 start-state summary rows and their scores, all 20 prompt-test count/endpoint summaries, and the relevant frozen log entries for the completed fold-law, precision, prompt, clamp and base-model studies. I reused accepted Task 16 independent tabulation and Task 17 source/report review rather than rerunning them. The companion coverage ledger distinguishes those levels of evidence.

The new scale, gain, precision and fold-law results below are **literal producer summary/log/source checks**, not an independent replay of their arrays, AD calculations, statistical tests or figures. I did not inspect every original remote archive, reconstruct historical process births, or independently authenticate model bytes. Those limits do not negate the completed producer records, but they preclude calling this a new numerical verification. Newer producer commits, unfinished outputs and the stale compiled PDF are outside the frozen review. No model, numerical analysis, test or producer code was executed; no producer files were edited. No new citation is added here.

## Must fix

### S01 — replace the stale 7B claim and separate sampling stages (lines 1539 and 1707–1711)

“No bifurcation has yet been located at 7B” contradicts the located 7B period-2 fold at lines 1631–1638 and the completed FL7 normal-form log. It can say **“These short continuations did not locate a bifurcation; a period-2 fold is located below.”**

The prompt-0 14B sentence also mixes stages. `prefix-scale-result.json` records `C1=12`, `C2=1`. Its ordered chaotic-coordinate list contains the twelve coarse-stage values beginning at **0.5957**, followed by the fine-stage value **0.5188**. `prefix_scale_score.py` appends coarse then fine cells, so the attribution is explicit. “12 of the 48 temperatures, between 0.52 and 0.90” gives the coarse count the fine-stage lower endpoint. Say **“On the paper's prompt, 12 of 48 coarse temperatures were chaotic; the adaptive fine stage found one additional chaotic temperature near 0.519.”** If reporting stage-specific ranges, derive their outward endpoints from the original coordinates, not these four-decimal summaries.

The same sentence says Table `tab:scale` ends at 0.6; its current 14B row ends at **1.2**. Replace this by **“The initial sweep ended at 0.6; the table now also includes the upper sweep.”** The corresponding producer log retains the earlier stage history; do not rewrite that history.

### S02 — stop inward rounding of observed ranges and distinguish rounded maxima (lines 1629–1635, 1733–1747, 1794–1798)

The 1.5B interval `[0.29,1.03]` excludes saved rounded chaotic coordinates **0.2881** and **1.0314**; the 14B interval `[0.52,1.14]` excludes **0.5195** and **1.1438**. This is already demonstrable without a raw replay. `scale_summary.py` stores these coordinates after rounding to four decimals, however, so they are not authority for exact replacement extrema. Generate manuscript range endpoints from the original saved coordinates, rounding lower ends downward and upper ends upward. Do not manufacture tighter bounds from the rounded log/summary. The twelve-temperature 1.5B confirmation starts at **0.288** in its original run description, so “between 0.29 and 0.58” likewise trims its declared range.

The table's `max lambda` column uses ordinary nearest rounding (`:.2f`), e.g. 0.28426632849419375 becomes +0.28, 0.854805511814244 becomes +0.85, and 0.41432951408204777 becomes +0.41. Preserve those as explicitly **approximate maxima**, or use outward upper displays if the column is intended as a bound. The value −0.0006737156509113927 printed as `−0.00` is especially unhelpful; scientific notation keeps its sign and scale.

The fold-law log gives a 7B next modulus approximately **0.0092**, versus `0.009` at line 1631, and a five-window passage fraction approximately **0.801**, versus an apparent upper endpoint `0.80`. Label ordinary rounded point estimates as approximate; use outward displays for ranges or limits. I have not independently extracted the exact original fold-score endpoints, so this note does not supply invented replacement bounds.

### S03 — make storage and effective-precision cohorts explicit (lines 1653–1655, 1672–1679, 1750, 1787–1790, 1853–1856, 1877–1879, 1902, 1914)

“Every checkpoint is stored entirely in bfloat16” contradicts the immediately preceding MoE description that some tensors are stored in float32 and restored exactly. Replace the universal sentence by **“We evaluate the stored checkpoint values in float64; the dense checkpoints here use bfloat16 weights, while the MoE has the stated float32 tensors.”** Retain only the storage claims actually established by the metadata.

The disclosed two-device hook defect also affected more than the isolated 72B window. The frozen 14 October 16:00 defect entry explicitly lists the 70B gain splits and **the start-state pod that spread all fourteen models over two GPUs**. The gain and CP paragraphs do not make that cohort association clear, while their captions say float64 without qualification. Mark original mixed-precision results separately from fully patched reruns, including the older S70/Z70 maps. Do not silently promote old arrays when selected cells were rechecked.

The assertions “matters only ... 1e-6 and below” and “Where exponents are large, the defect does not matter” exceed the finite comparisons. PTa sampled twelve positive cells of the concept zoom; it did not retest all 4,096 zoom cells or all four S70 direction maps. State instead: **“The selected positive cells retained their labels at the tested larger offsets; this is a finite check, not a qualification of every original map cell.”** Preserve the exact QFa/QFb corrected-float64/tangent results as stronger evidence for their particular cells. This requests correct attribution, not rejection or rerunning of all old results.

### S04 — do not identify a uniform-contraction threshold from trajectory exponents (lines 1799–1800)

The claim that the contraction proved in the theory “sets in only above log10 T approximately 1.1” is not established by a finite sweep of trajectories from one start. Negative finite-time exponents do not verify the theorem's uniform operator-norm condition or exclude other attractors. Replace by **“On these sampled trajectories, the upper sweeps become non-chaotic above approximately this region; we did not locate the theorem's uniform-contraction threshold.”** Use the actual model-specific grid ranges if stating precise endpoints.

### S05 — entropy clamping does not isolate one original-map factor (lines 1823–1835)

“An intervention separates the two factors” is inconsistent with the derivative formula that follows. The state-dependent temperature changes the map and adds a rank-one derivative term. Periodic clamped trajectories are evidence about that modified map, not a controlled replacement of just the original readout factor with its network factor held fixed. The 32 tested prompt/target trajectories support: **“Entropy clamping tests whether forcing intermediate entropy is sufficient to destabilize this modified loop. It did not do so at these targets; this is consistent with, but does not identify, network damping in the original map.”** Keep the formula, finite-difference qualification and two-prompt scope.

### S06 — failed base-model prediction does not eliminate post-training as an explanation (lines 1835–1845 and 1924)

“Nor does post-training make the 72B stable” and “neither ... post-training itself explains the stability; what remains is ...” are too strong. PB72 rejects its prespecified margin on four prompts; the tested base checkpoint is also mostly non-chaotic. That does not isolate all effects of post-training or establish a unique remaining mechanism. The paper properly discloses different horizons, which further limits this causal exclusion.

Use **“The pretrained checkpoint was also mostly non-chaotic in the tested grid, so post-training is not necessary for this observed scarcity. The failed prediction does not quantify or exclude post-training's effects.”** At line 1924 say **“the two 70B Llama checkpoints with different post-training show different finite-grid results”**, consistent with the explicit descriptive qualification at line 1821. The observational gain split is useful evidence; it is not a causal mediation experiment.

### S07 — correct the entropy theorem and honor CP2's frozen outcome (lines 1860–1875 and 1939)

The theory does not confine chaos to a fixed intermediate **entropy** interval. A sufficiently high temperature can yield a contraction bound, but near-uniform output entropy alone does not guarantee it. The paper itself reports near-uniform chaos immediately above. Replace the opening by **“Motivated by the saturation and high-temperature limits, we tested intermediate-entropy breadth as a heuristic predictor.”**

The primary CP1 prediction fails (`rho=0.2953568736619875`), but the frozen secondary CP2 **passes its stated threshold** (`rho=0.5171349644228007`, `p=0.05826423877056071`); its complete-sweep sensitivity is `rho=0.44371293379038995`. The current sentence reports these numbers then declares that no tested one-pass property predicts usefully. Use **“The primary prediction failed. The secondary median-over-prompts criterion narrowly passed on the frozen outcome and weakened on the expanded sweeps, so we have not established a reliable general predictor.”** Lack of conventional significance does not retroactively change the frozen pass rule. Also preserve the effective-precision cohort limitation in S03.

### S08 — disclose the post-outcome fold-window selector correction and retain candidate-edge status (lines 1621–1638)

The 03:03 log fixes the bounded-offset criterion before the line results; the 03:27 result explicitly corrects `foldlaw_score.py` after its first scoring so the fan follows `p,p−q,p−2q,...`, skipping other periods. It says the nearest-four conclusions were unchanged, but the corrected rule added farther windows. The statement that the criterion was fixed is true yet incomplete for the selected extended fan. Add **“The scoring criterion was fixed before outcomes; a disclosed post-scoring correction to the period-progression selector retained the nearest-four conclusions and changed the farther-window selection.”** Retain original and corrected scores.

The located short-cycle folds are substantially stronger evidence than the individual distant window edges inferred from grid runs. A period doubling/tripling on the far side is compatible with a fold on the near side but does not classify that near edge by its multiplier. Change “so these edges are fold edges” to **“this organization is consistent with near-side fold edges; the fitted distances use grid estimates of those edges.”** The observed law remains a finite empirical comparison, not continuation/classification of every window edge.

## Should fix

### S09 — bound absence, coexistence and route statements by what was tested (lines 1496, 1499–1500, 1533–1539)

Say **“No period-adding fan was resolved in this map”**, rather than absolute absence. An abrupt sampled period/exponent change alone does not establish two separate attractors instead of a flip; cite the branch/coexistence evidence or say it is **consistent with an attractor switch rather than a resolved smooth doubling**. For the 7B coexistence paragraph, both reported stable continuations overlap on approximately `[0.0476,0.0507]`, not the union of their ranges. State that overlap explicitly, and describe the observed fixed-start switch as a basin-selection boundary **within the tested continuation range**. The later located fold remains separate.

### S10 — distinguish grid-weighted extent from an invariant-set measure (lines 1752–1758 and 1764–1768)

The caption already defines spacing weights, and the shared 72B endpoint is now counted once; those are improvements over Task 17. Still, `np.gradient(logT)` weighted finite cell labels are an estimate of sampled extent, not a rigorous measure of every chaotic parameter. The short scans give incomplete-domain estimates, not mathematical lower bounds on the full chaotic set. Use **“grid-weighted chaotic extent over the tested range; models stopping at 0.3 have incomplete temperature coverage.”** Keep the old coarse counts separately from adaptively targeted fine-line checks, as the current table correctly does.

### S11 — retain the contradictory period/exponent row in the 1.5B confirmation (lines 1797–1798)

The completed local 11:22 log reports nine chaotic labels and three periodic labels, but one periodic row at `log10 T approximately 0.3942` has positive twin exponents `+0.109/+0.107` at both offsets. It was retained as returned, and agrees across offsets while failing the no-period chaos condition. The stated `9/12` does not hide this numerically, but a brief note should retain the anomalous period-positive row rather than describing the whole check as unqualified confirmation. This note reads the producer disposition, not a new diagnosis of that row.

### S12 — state the actual finite classification threshold consistently (lines 1671, 1704, 1751–1752, 1906–1907)

Body “positive” can mean greater than zero, whereas the sources and caption use **greater than 1e-3 plus no detected state period**. Define that abbreviation once and keep the no-period/non-chaotic category explicit; unresolved is not automatically periodic. The four original long unsteered sweep counts and Llama direction counts are independently supported by Task 16, but a finite period test still is not an exact orbit proof.

## Optional presentation improvements and results to retain

- Replace section headings “What sets it” and “What this implies for larger models” with language that signals measured associations and tested scope. The existing caveat at 1821 is good; use it consistently in the surrounding conclusions.
- Preserve the failed P1/P2 outcomes for OLMo, failed PS1/PS2, and the release/size confounding already stated. Keep development prompts separate from eight held-out prompts.
- Table `tab:scale` agrees with all 15 literal scale-summary core counts/unresolved counts and ordinary-rounded whole-axis summaries. Its 72B denominator **191 distinct temperatures / 128 core** fixes the earlier duplicated shared endpoint. That is source/summary consistency, not a fresh array audit.
- The accepted Task 16 independent result binds Qwen0.5B `7/512`, Qwen7B `65/512`, Llama8B `96/512`, OLMo7B `32/512`, their unresolved counts `1,6,2,3`, and the Llama steering-control percentages. Do not call the new 15-model summary equally independently audited.
- The PS summary gives all eight held-out 14B coarse counts above 7B: `14,9,11,15,12,13,17,11` versus `6,3,0,0,7,3,4,5`; the largest min-offset exponent prediction succeeds on only three. Those reported outcomes are source-consistent.
- Preserve the withdrawal of the original small-offset 72B window, the two-of-three Llama70B plateau result, and the corrected QFa six-of-eight result. Current prose retains these contrary outcomes rather than overwriting old scans.
- MoE's 52/1024 routing-sensitive cells remain “irregular”; the current text correctly avoids promoting them to smooth-map chaos. The dense-Qwen3.5 architectural control remains unrun.
- The Soft Thinking cross-reference concerns finite growth of early perturbations in a growing-context system. Theory's bridge lane owns the detailed verification; this scale review does not turn that finite record into an autonomous attractor theorem.

## Coverage disposition

All assigned text has been reviewed. New literal summary/log associations were checked only at the scope stated above; new flip numerical gates at 1.5B are assigned to Theory's cascade lane, and new citations to Root/Theory. Figure pixels, raw NPZ endpoints, complete per-cell precision diagnostics and remote original evidence were not independently replayed here. Existing accepted numerical evidence remains unchanged. The manuscript can address the findings by reporting the available evidence accurately, without new model calculations.
