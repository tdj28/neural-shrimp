# Task21i — final saved BZ/BP scientific review

The original BZ/BP saved arithmetic is reproduced: **856 published JSON leaves match exactly**, with no raw-versus-stored BZ cell-summary differences; **all 120 fields from the subsequently supplied inline-adjustment transcript match at their printed precision**. Those are different comparison standards. No unknown full-precision producer OLS coefficient is inferred from printed decimals.

The substantive conclusions remain limited. BZ compares two steered conditions with near-zero correctness, while its unsteered controls perform much better; it does not establish absence of a possible regime effect. BP’s 7B negative association reproduces, but its size depends on task composition and the retrospective adjustment. The latter uses entropy measured during generation and does not identify a causal effect, a direction of overcorrection, or a range containing the true effect. This is unblinded independent saved-row arithmetic and scientific interpretation, not a fresh confirmation or model replay.

## Coverage and evidence

The sole completed own run is `/data2/neural-shrimp-data/codex/night-results-review-task21/task21i/run-001`, source `f5fa2fd151de4f500d74b83c7176ccc9525360fb`. It retains 100 selected original files, all 90 BZ cells/30 three-arm pairs/32,400 responses, and both BP models’ 400 label prompts/1,200 prompt-temperature rows/12,400 responses. All **44,800 response rows** remain. The original authority preserves 21 attempt histories, including 13 retrieved archives and eight failed/unretrieved BPq7 predecessor attempts; successful replacements do not erase those histories.

I reviewed the authenticated report through the sole literal presentation, including all BZ cell/task and pair rows, all 30 selection predicates, nine metric/diagnostic summaries, every BP model/temperature/subset and composition table, all label/response flag inventories, all three observed OLS fits, all 12 entropy bins, all 15,000 saved OLS fit-flag dispositions, every printed comparison, and the complete comparison-exception inventories. A previously truncated display was repaired with bounded selections of already generated presentation fields, without repeating calculations. Full coefficients, replies and prompt memberships remain in the literal files; the narrative does not claim human inspection of every response’s prose or every permutation coefficient.

The compact complete selection is [`task21i-compact-saved-dispositions.json`](/tmp/task21i-compact-saved-dispositions.json), SHA256 `94af8a6f69287efc70559d710386661a499e8dc28f3f769dbcf4a17b0d56400d`. It keeps all cell/pair/metric/selection rows, BP/composition summaries, label and response inventories, fit-flag counts, and every comparison leaf. The original scientific computation and subsequent presentation both completed without retry. Neither computational completion nor arithmetic agreement is a causal or native-chaos validation gate.

## BZ: reproduced floor-limited comparison

Every one of the 30 saved selection predicates is true: matched temperature, declared alpha gap at most 0.001, declared chaotic exponent above 0.05, and periodic exponent below −0.05 with positive detected period. The exact source-map nearest-row/minimum-gap selection and native label trajectories were not replayed. A small steering gap does not randomize regime or remove direct steering effects. Generation used the original float32 ordinary steered inference; the float64 map-label source does not make generation float64.

Saved correct-response counts show the floor directly:

| Task | Chaotic | Periodic | Unsteered |
|---|---:|---:|---:|
| Arithmetic | 3/4,500 | 4/4,500 | 3,499/4,500 |
| NQ | 20/4,500 | 16/4,500 | 190/4,500 |
| MBPP | 0/1,800 | 0/1,800 | 501/1,800 |

The MBPP counts inherit saved code-test flags; generated programs were not executed. All 270 cell/task groups retain their response counts, mean entropy and mean length, with zero nonfinite-entropy or negative-length flags. All raw-versus-stored mean comparison leaves match.

All following differences are chaotic minus periodic. Accuracy uses fractions correct; entropy and length use their source-defined units. Endpoints below are the literal saved values, avoiding inward-rounded bounds.

| Task/metric | Mean difference | Saved 95% bootstrap endpoints | Nonzero sign-test p |
|---|---:|---|---:|
| Arithmetic accuracy | −0.00022222222222222223 | [−0.0011111111111111111, 0.00044444444444444447] | 1 |
| Arithmetic entropy | 0.0011812115639522928 | [−0.04104078291990093, 0.04460554329853181] | 1 |
| Arithmetic length | 0.11133333333333327 | [−0.05222222222222221, 0.2735611111111112] | 0.09873714670538902 |
| NQ accuracy | 0.0008888888888888889 | [−0.0017777777777777776, 0.0035555555555555557] | 0.454498291015625 |
| NQ entropy | 0.03751272292511635 | [0.016076059565263735, 0.059496338228812416] | 0.016124801710247993 |
| NQ length | 0.07711111111111091 | [−0.3031166666666675, 0.4324444444444444] | 0.584664711728692 |
| MBPP accuracy | 0 | [0, 0] | 1 |
| MBPP entropy | −0.034993655990870356 | [−0.06701978222386422, −0.003917400885346974] | 0.09873714670538902 |
| MBPP length | −0.8194444444444495 | [−5.80556944444445, 4.671374999999989] | 0.361594608053565 |

Each metric contains all 30 pairs and 10,000 finite bootstrap draws; no input/draw is nonfinite or dropped. The sign test excludes zero differences by its original rule; accuracy has 3, 16 and 0 nonzero pairs respectively. NQ entropy and MBPP entropy must remain visible even though correctness differences are small: “no differences in any behavioral metric” would be false. Different sign-test and bootstrap conclusions are preserved, not harmonized. These are exploratory finite pair summaries; shared prompts, chosen cells and nearby regimes limit population and independence interpretations. The all-zero MBPP correctness interval is the observed empirical floor, not proof of equivalence.

## BP: all models, temperatures and subsets

Each model has 200 fixed prompts, 100 arithmetic and 100 NQ, ten sampled answers at each of three temperatures, and one greedy answer. All 1,200 saved prompt-temperature label records have finite qualified period/exponent fields and exact agreement with the source’s chaotic/periodic/other union rule. Hidden-versus-visible subclass remains inherited because lead-period evidence is absent. None of these scalar checks recalibrates native chaos classification.

The table gives the exact saved chaotic-minus-periodic fraction-correct difference and original raw permutation fraction. Each defined contrast uses 10,000 prompt permutations. The two undefined rows consume zero draws, so the actual BP workload contains ten defined batches, rather than the manufactured worst-all-groups-present allocation of twelve.

| Model | log10 T | Subset | Chaotic / periodic prompts | Difference | Raw permutation p |
|---|---:|---|---:|---:|---:|
| q05 | 0.25 | all | 94 / 104 | 0.01153846153846158 | 0.7274 |
| q05 | 0.25 | greedy-correct | 35 / 37 | 0.012277992277992311 | 0.8507 |
| q05 | 0.32 | all | 21 / 179 | −0.013886671987230644 | 0.484 |
| q05 | 0.32 | greedy-correct | 7 / 66 | −0.017316017316017313 | 0.665 |
| q05 | 0.4 | all | 0 / 198 | undefined: missing chaotic group | undefined |
| q05 | 0.4 | greedy-correct | 0 / 72 | undefined: missing chaotic group | undefined |
| q7 | 0.25 | all | 38 / 161 | −0.37371690094802235 | 0/10,000 |
| q7 | 0.25 | greedy-correct | 14 / 105 | −0.26857142857142857 | 0.0011 |
| q7 | 0.32 | all | 68 / 129 | −0.25706794345645234 | 0.0002 |
| q7 | 0.32 | greedy-correct | 33 / 84 | −0.2378787878787878 | 0.0004 |
| q7 | 0.4 | all | 90 / 109 | −0.3576146788990825 | 0/10,000 |
| q7 | 0.4 | greedy-correct | 40 / 79 | −0.3604746835443039 | 0/10,000 |

The other-label counts at the three temperatures are q05: 2, 0, 2; q7: 1, 3, 1. They remain in the coverage ledger and are excluded from the original two-label contrasts, not discarded from the report. The resampling unit is a prompt’s mean accuracy, not each of its ten answers. Raw zero exceedances are an estimate of zero in that finite Monte Carlo calculation, not a guaranteed probability below 10⁻⁴; the separately saved plus-one value is `9.999000099990002e-05`. The exact original raw convention is retained. These label-shuffle fractions are not randomized-treatment evidence.

Task composition is materially different between labels. The exact symmetric identity separates the observed aggregate into within-task and composition terms; it is not a causal decomposition or “fraction explained.” For q7 all-prompt rows:

| log10 T | Within-task term | Composition term | Saved closure difference |
|---:|---:|---:|---:|
| 0.25 | −0.09275333130964056 | −0.2809635696383816 | 2.220446049250313e−16 |
| 0.32 | −0.07751076888354597 | −0.17955717457290646 | −1.1102230246251565e−16 |
| 0.4 | −0.12733858469287507 | −0.23027609420620757 | −1.1102230246251565e−16 |

All q7 task/subset rows have common support. At 0.4, NQ contrasts are positive (`0.056334841628959284` for all, `0.12777777777777782` for greedy-correct) while arithmetic contrasts are negative; an all-task negative effect claim would be inaccurate. q05 at 0.32 has no chaotic NQ prompts, making its complete task decomposition undefined in both subsets. At 0.4 neither q05 subset has a chaotic group. These missing-support rows are retained without imputation. All eight defined decompositions’ signed floating closure differences remain in the compact artifact, including nonzero last-bit values.

## Retrospective q7 adjustment: exact transcript reproduction

The producer supplied the original inline command, printed output and invocation order before this own run. Earlier preparation documenting the missing record remains preserved, and the result explicitly records the later transcript as superseding evidence. Order is now producer-transcript attested, not independently observed historical execution. Full own outputs reproduce every one of the transcript’s 120 scalar/presence fields after original formatting.

The OLS columns are intercept, chaotic indicator, mean sampled generation entropy, NQ indicator and greedy correctness. There is one seed-1 stream across three batches of 5,000 permutations of the chaotic column, with covariates otherwise held fixed.

| log10 T | Included prompts | Chaotic coefficient | Entropy coefficient | Raw p | Exceedances |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 199 | −0.036821230845983655 | −0.06663290477980954 | 0.2038 | 1,019/5,000 |
| 0.32 | 197 | −0.04627860047036536 | −0.17678292079316957 | 0.0690 | 345/5,000 |
| 0.4 | 199 | −0.04485049743706537 | −0.41936667496393937 | 0.0862 | 431/5,000 |

All three observed fits have rank five and finite solutions; **all 15,000 saved permutation fits are defined, rank five, with no deficient or undefined fits**. Full observed coefficient vectors, singular values, residual sums and every permutation coefficient vector are retained. The rows excluded as other labels are prompt 98 at 0.25; prompts 54, 73 and 47 at 0.32 in source order; and prompt 175 at 0.4. Every retained prompt has ten sampled answers; no additional missing-generation exclusion occurred.

All four greedy-correct entropy quartile bins survive in each temperature’s output. Their boundaries, prompt identities, exact means and weights are in the compact/literal records. The following saved contrasts retain both favorable and adverse bins:

| log10 T | Bin | Chaotic / periodic | Chaotic-minus-periodic accuracy | Weight |
|---:|---:|---:|---:|---:|
| 0.25 | 0 | 1 / 29 | 0 | 1 |
| 0.25 | 1 | 0 / 29 | undefined: empty arm | 0 |
| 0.25 | 2 | 6 / 24 | −0.15000000000000002 | 6 |
| 0.25 | 3 | 7 / 23 | −0.20931677018633527 | 7 |
| 0.32 | 0 | 2 / 27 | 0.007407407407407418 | 2 |
| 0.32 | 1 | 5 / 24 | −0.14749999999999985 | 5 |
| 0.32 | 2 | 12 / 17 | −0.025490196078431393 | 12 |
| 0.32 | 3 | 14 / 16 | −0.10803571428571429 | 14 |
| 0.4 | 0 | 4 / 26 | 0.026923076923076827 | 4 |
| 0.4 | 1 | 4 / 25 | −0.029000000000000026 | 4 |
| 0.4 | 2 | 13 / 17 | −0.3171945701357467 | 13 |
| 0.4 | 3 | 19 / 11 | 0.07559808612440194 | 11 |

The min-arm-count weighted contrasts are respectively `−0.16894409937888194`, `−0.07700204660988974`, and `−0.10313306739699932`, using 3/4/4 eligible bins and total weights 14/33/32. The missing 0.25 bin was omitted from the original printed contrast lines; it is explicitly retained here. Infinite outer bin endpoints are intentional boundaries, not nonfinite observations. No finite entropy input is missing.

The adjusted coefficients reproduce the approximately −4 to −5 percentage-point description. They are conditional linear-projection coefficients, not causal effect estimates. Entropy is measured during sampling; its inclusion may change the estimand and introduce selection/conditioning issues. The source’s suggestion of “over-correction” is not itself established, nor does the arithmetic show that a true effect lies between these coefficients and the raw BP differences. The unconditional label shuffle does not supply an exact conditional-independence test when labels and covariates are associated. This review reproduces that declared statistic without replacing it.

## Complete response-scoring and finite-field dispositions

All 39,400 arithmetic/NQ saved-text score comparisons agree. Of these, 181 BP NQ replies reach the 200-character cap and remain possibly truncated prefixes: q05 has 35/38/53 at the three temperatures; q7 has 7/13/35. Prefix agreement does not establish the score of any unretained suffix. All 27,000 BZ arithmetic/NQ records are below their 400-character cap. The 5,400 MBPP scores remain inherited and unexecuted. Consequently the 5,581 presentation “exception” records consist of unavailable MBPP checks plus possible truncations, not 5,581 scoring failures.

Every present entropy field is finite and every present length is nonnegative. The 400 BP greedy records omit those fields by the original schema and are recorded as absent, not invalid. No response-scoring mismatch, finite-field failure, source-label union mismatch, invalid saved period qualification, published JSON discrepancy, or transcript printed-precision mismatch was found. This does not validate generated-code execution, hidden/visible subclasses, model precision at runtime, or full native dynamical labels that were outside the saved-data contract.

## Verdicts and suggested wording

| Assigned point | Disposition |
|---|---|
| BZ stored means, pair statistics and original summary | Confirmed as saved arithmetic; steered correctness floor and unsteered counts retained. |
| BP q05/q7 original model/temperature/subset summaries | Confirmed, including missing q05 chaotic group at 0.4. |
| Task-composition description | Confirmed finite algebra with explicit missing-support rows; no causal fraction. |
| Original q7 inline OLS/quartiles | Confirmed at all supplied printed precisions; full own values and all fit flags retained. |
| Native labels, precision/loaded model behavior, MBPP re-execution | Cannot check independently from this scalar/response audit; inherited source/receipt basis preserved. |
| Chaos causes accuracy loss, adjusted/raw values bracket the true effect, or BZ proves equivalence | Unsupported by these saved observational analyses. |
| Fresh confirmatory status for exposed BPC/BPCS | Unavailable; a genuinely new cohort/design is required. |

Suggested paper wording: “Independent saved-response reanalysis reproduced the BZ and BP summaries. In BZ, both selected steered regimes had near-zero correctness, so their small difference is floor-limited. In the 7B prompt pilot, chaotic-labelled prompts had lower aggregate sampled accuracy, with substantial task-composition differences. A retrospective linear adjustment including generation entropy yielded chaotic-label coefficients of approximately −0.037, −0.046 and −0.045 and reproduced the reported permutation fractions. These are descriptive associations; entropy is measured during sampling, and the analysis does not identify a causal effect or establish a bound between raw and adjusted estimates.”

For numerical tables, use the exact saved endpoints above or explicit outward-enclosing displays. Keep raw zero exceedances as `0/10,000`, rather than converting them into an exact probability bound.

## Future-fresh outline, preserved and not executable

The accepted [`task21i-source-scope-and-fresh-plan.md`](/tmp/task21i-source-scope-and-fresh-plan.md), SHA prefix `17488d5d`, remains a prospective outline. Its historical missing-source sections are superseded by the admitted original source packet and later transcript, while its freshness and inferential distinctions remain applicable. BP, BPC and BPCS outcomes were already exposed; their old outcomes cannot be preregistered now, and this review did not add a BPC/BPCS raw replay.

A separately approved future design would first fix one model/revision, temperature and arithmetic population; an exact unseen prompt/equivalence exclusion ledger; sample allocation; source/precision/chunk/scoring policy; and handling of undefined labels/fits. Covariates and greedy answers would be sealed before sampled answers, retaining the distinction between gold-dependent benchmark features and an answer-blind diagnostic. The proposed primary is a predeclared adjusted association with prompt/equivalence-cluster uncertainty; a frozen held-out predictive comparison is secondary. Neither is a causal chaos experiment. Exact N, resources, thresholds and undefined-replicate decision rules remain unresolved and must be specified before fresh outcomes. No new model run or budget is authorized by this outline. Exposed BPC/BPCS edge/tail findings remain development context, not newly discovered confirmatory thresholds.

## Pins, lifecycle and timing

- Original own report: `run-001/report.json`, SHA256 `3fb8d17b3976e8b7feaa4b2e520ce4b54928de279f7d98097a31cc28daad1d48`.
- Complete response ledger: `run-001/responses.jsonl`, SHA256 `3d93150198e8b7fd5fb190fded33d033ae95a5aca5d84d64080182053bb0ee6d`.
- Worker manifest: `ae31b30865bad7223e674b2992f5ae165edd5734432cbf5dddceafd272b4589d`.
- Original source/input/output/closure authority: sibling `run-001-authority/report.json`, SHA256 `2a19d9b824c91930ccebbae16ca8d74e24f3c347de28f8e0dbc46d085b3eb5b4`. It authenticates three outputs, ten supervision files, five captured Git blobs, 145 input pins, and closed original supervisor/worker groups/sessions. Tool session 26780 was reaped with exit 0.
- Input/original-path association: `/data2/neural-shrimp-data/codex/night-results-review-task21/task21i/source-001/inputs.json`, SHA256 `44dd14a95c2c1f88a862bcc06c9b9110dc22d9eea4ae76395d8f2f269edba706`; exact original files, selected replacement attempts and archive/source limitations are retained there and in its bound authority. The transcript supplement is `3b5bcecc019bfb3080e5b83a83287cfa6b9852de6f4fdb42727077b23c986b49`.
- Sole literal presentation: `/tmp/task21i-saved-read-001`; summary `d18636c274871b1dd4ee2a47601e9348794359e76a65d5e5e427d2878af19405`, manifest `f3f7253034dfef8a22735bb4d9a9b77cae688a081412cba3a58c2a72a0be3bd5`. Full OLS rows are `ols-fits.jsonl`; all BP prompt rows are `bp-prompt-fields.jsonl`.
- Presentation receipt: `/tmp/task21i-saved-read-001-supervision/receipt.json`, SHA256 `5a584f87085e85e905c4878eb8eba37351b7206455f0c63489d8bdae7d732e76`. It records reader source `7fe3bc0b7f356cea62bd68e46cc95878da0b0793b90f455dd78c5caac0ce6aea`, exit 0/no timeout, external 1.2665457599796355 seconds and original reader PID 3143946/start 202296436 absent with its group/session; session 12203 is closed.

Direct execution/reaping and closure observations belong to Inference. My work is independent source review and subsequent reading of the completed, authenticated saved results. The original worker reports 2.4191807608585805 seconds and 65,220 KiB observed peak **before final pin rehash/report write**; the separately reported supervisor window is 3.117328632 seconds. Neither worker observation is a full-lifetime wall/peak measurement. The metadata authority took 0.1446807470638305 seconds. The presentation observed 1.1490934710018337 seconds and 47,332 KiB before its summary write. These scopes are distinct, with no invented combined runtime.
