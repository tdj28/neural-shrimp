# Task21i independent source review

Disposition: accepted for the proposed finite saved-data reconstruction and manufactured resource check. No concrete arithmetic, source-contract, coverage, or fixture defect was found in the released files. This is a source/design/control review: I did not execute the controls, manufactured program, statistical routines, producer tools, or models, and did not read target outcomes. Actual measured resource evidence, the final actual run card, and operational admission remain separate.

## Exact reviewed release

All files below are under `/tmp/task21i-preparation/`. The final release manifest is `prepared-source-release.json`, SHA256 `c08e46e37012a34c6166a73fa884d523e4da194ca4180767001e9099c72d8fb8`.

| File | SHA256 |
|---|---|
| `statistics.py` | `4985313cb433a69fc51bbfbdfe06f665fa82f372487e371897038d30271363ca` |
| `run.py` | `7412d16d0432f71cab81c17fb6343a8dde35ac870f80985b65f881fa325d94ef` |
| `controls.py` | `37623683da30be4cfcc1d279cc2b303198fd8bb06ed50c931322f015ddbb280e` |
| `manufactured.py` | `473d5fcac6ce59cd8cf4150495b6ff532f0664b2cd2f65a6d76c9915f53a0be5` |
| `component-design.md` | `7543b183a8fa8066bf215fcbc8ecf5e6807d915e19201cc700cf58c928cad237` |
| `adapter-design.md` | `83d46179fc8dff1508804824cedff6293c1035ce6a7c789254f6da7adaa9098b` |
| `source-association.md` | `4d573cdc9fa9fac4bc945c9609afe0df2c2543651d0b4a2a145f59d34ff46677` |

I read the complete source, designs, and focused control definitions, including the final combined manufactured driver. Relevant original generation/scoring and later analysis call sites were checked in the exact exported source packet; this does not attest the original loaded model weights or unrecorded command order. The input contract is pinned by `inputs.json` SHA256 `44dd14a95c2c1f88a862bcc06c9b9110dc22d9eea4ae76395d8f2f269edba706`. Its durable outer copy and unchanged literal nested paths are an operational association, not a new historical attestation.

## Findings

- **Coverage and evidence retention.** The driver requires the declared 90 BZ cells, 30 complete three-arm pairs, 32,400 BZ responses, and 200 BP prompts/6,200 responses per model, totaling 44,800 responses. Duplicate keys/identities, missing expected coverage, wrong associations, and malformed required fields refuse rather than silently select a favorable subset. Original label diagnostics, nonfinite or invalid fields, undefined comparisons, and raw/stored/published differences remain explicit. Output ownership is exclusive; failures cannot overwrite an existing output directory.
- **BZ ancestry and arithmetic.** The original BZ generation source has no analyzer; the later pinned analyzer supplies the historical comparison convention. The own raw-response means, stored cell-summary comparisons, continuous seeded bootstrap order, linear percentiles, and nonzero sign-test convention are kept distinct. Tiny raw-versus-stored differences are propagated, not forced to equality. Pair resampling does not establish independence of shared tasks or a population-level causal effect.
- **BP units and order.** Prompt-level mean accuracy is the unit, not each sampled answer. The literal label mapping, temperature/subset order, inclusive two-sided permutation comparison, and raw exceedance fraction are retained; the separately reported plus-one value does not replace the historical convention. The frozen canonical shard/line order is prospective for this reproduction. Missing historical CLI order can affect Monte Carlo draws and does not justify calling canonical draws the original draws. Missing groups remain undefined and consume no draws.
- **Scoring and missing evidence.** Full saved replies and capped prefixes are distinguished. Arithmetic/NQ normalization comparisons follow the exported source; generated MBPP code is not executed, so its correctness flags remain inherited. BP hidden-class detail is not reconstructed where lead-period evidence is absent. The unavailable inline adjusted regression remains uncheckable; the explicit task-composition identity is descriptive and is not represented as that regression, causal adjustment, or a new significance test.
- **Exposure and claims.** This is an unblinded saved-data reproduction. BPC/BPCS and their already disclosed outcomes are context only. The prospective fresh-plan outline is not a preregistration of those old runs and does not authorize a new experiment. Raw zero permutation exceedances are not a guaranteed population probability bound.
- **Manufactured check.** Source inspection confirms that the initial ten focused controls are invoked once, followed by five additional refusal/diagnostic groups and the complete manufactured-shape group. The latter exercises 90,000 bootstrap draws, 120,000 permutation draws, all 44,800 emitted responses, and scientific serialization. Its timer and observed RSS have explicitly limited scopes. Fixture construction, original input hashing/decoding, actual string lengths, and final output overhead are not a proven worst case. These are prospective checks, not claimed passing results in this note.

No source changes or repeated controls are requested. Inference retains the concrete supervisor/card, private checkpoint, sole manufactured execution, and eventual actual-run lifecycle. The proposed one-CPU, 1 GiB address-space, 55 CPU-second/60 wall-second, 128 MiB-output manufactured envelope remains a bounded proposal until its own measured result is available; this note does not authorize target execution.
