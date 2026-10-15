# Task21f historical source association — narrow review

The exact historical factor schemas are compatible with the accepted independent core. No scientific formula correction is needed. The adapter must enforce the per-run field availability below and preserve incomplete breadth-first coverage. I read original source/diffs, constructor call sites and metadata only; no target scalar array, numerical progress record or grid-summary outcome was decoded, and no source/model/test was executed.

## Bound versions and fields

The source map is `/tmp/task21f-source-001/source-map.json`, SHA256 `f921bbb3e9ebaa420e9a84a3a75047ebc628105731a91ad7eb296f612e66cc75`. Metadata inputs are `bff3fbc5937f345f35ca6312e87517c9927b7a9b709c48a4ade1ba0292100aff`; original authority is `f7af712f5f6eeebb2132aeb7456d3bd01c03e45f5bd5261add5910a07e35788e`. Original operational authority is coordinated separately by Inference; I do not claim direct historical execution observation.

| Original run | Original commit | Factor-source SHA256 | Declared grid | Historical field scope |
|---|---|---|---|---|
| p7-blob | 7217f136aa447b932ad4db0b91f8ddf3c488e05d | 78aaa3ce9dda58aae33d8b13d792e8e3b36a0cf208a995a7869525ea4dd5b71b | 31×31 | fixed-reference factors only |
| p7-blob-lo / hi | 68e9d2c32c9a45f272485fd4bacb0967dda91ca3 | same78aaa3ce | each16×31 | fixed-reference factors only |
| p7-zoom-a / b / c | d89d5bc9d40f0790902bf7971bf90c64b6c7cf2f | b50adb9fd2d02478b6bcea6538060f3786e16cf6fa42d051f4596ccbc8710264 | 12×33 /12×33 /13×33 | fixed-reference plus transported factors, stretchings and carrying cosine |

The 7217→68e9 factor diff is empty. The 68e9→d89 diff (`88044c85dbe8caeac343349df9bdda3d09edf6326d5c277fb2247d65e8852d26`) adds the same `chain_dirs` and `transported` algebra already reviewed at frozen Task27 source74e2adb9, carries orientations with BFS parent state and writes the three added fields. It does not change the fixed-reference factors or the real-part/clean-leading convention. The absence of transport fields in the first three runs is legitimate unavailability, not zero factors, failed identities or evidence about the amended method.

The d89→Task27 diff (`2342fd78a8f5a16afbbf46f5f8bda22d4583aac152880a8b2a7541523c7deb78`) adds later two-block parameters, the `multipliers_only` branch, a saved three-value multiplier array and `row_axis`. None of the six historical runs has these fields/options, and their original source does not bypass the seed clean/real gate. The adapter must not require or fabricate those later fields. Conversely, every zoom must have the complete transported bundle. The same fixed alpha/log-temperature axes, seed fields, period7 phase count and original command order remain explicit.

## Numerical semantics preserved

In original78aaa3ce, `leading` lines60–67 selects the first modulus-sorted Ritz value, rotates its vector and takes the real part; `clean` means modulus separation greater than ten, not reality. The seed checks detected period7, Newton tolerance and a clean value with imaginary part at most1e−8 (lines96–118). There is no corresponding clean/real refusal at subsequent grid cells (lines148–161). Thus complex or unclean finite factor rows must remain present and unqualified for a single-real-eigenline interpretation, exactly as in the accepted core's separate masks.

The source's `ok` array requires a successful Newton branch and `split >1e−7`; `split` checks proper period divisors, which for period7 is the period1 distance. It does not include clean-leading, token, carrying-cosine or attraction requirements. Newton-success cells enqueue neighbors even when the split test fails. The metadata's processed-index counts807/477/483/396/396/429 are coverage counts, not numbers of converged, primitive, clean, real or same-token cells. Those numerical dispositions remain unread until the admitted actual pass.

Shared `llm_feigenbaum.py` SHA256 `fbb0dc468bd340084825f71403a46ab1156a2b026bedbbbea3f943176e14f944` uses `max(tol,1e−13*p)` (lines48–52), which leaves the declared period7 Newton tolerance at1e−12. The Ritz routine (102–126) sorts by modulus and returns after at most12 passes even if its watched residual test has not passed. The DSS caller does not add a residual refusal, and the grid/progress schema omits Ritz residuals/history. Do not invent an enforced successful residual gate from the routine's nominal tolerance. The saved calculation can reproduce scalar comparisons and source predicates; native residual/direction quality remains inherited. This does not demand another native run.

## Snapshot and completion scope

All original versions initialize numerical arrays toNaN and flags tofalse, append each processed BFS record, snapshot only every25 processed cells or when the queue empties, then write final gridJSON only after queue exhaustion. They do not write parent/orientation vectors. The completed low/high jobs can leave unvisited cells when the reachable queue is exhausted. Original blob tool status124 must remain distinct from the runner's later status0 and the absence of gridJSON; its later progress tail must not be filled into the last NPZ. Final summaries and current snapshots can be compared after admission without reconstructing unwritten cells.

The original runner's common helper records the inner timeout exit separately; its end-of-function logging can leave the overall runner status0 after an inner124. This supports preserving both statuses rather than relabelling the interrupted blob successful.

The actual header metadata fixes zoom row counts12/12/13, so the plot's literal drop-first-row rule gives35 joined rows and33 columns. This resolves the earlier planned33-row versus later35-row wording at the schema level; it does not establish any numerical contour or complex-cell count. Exact overlap coordinates/payloads and independently seeded phase gauges must still be retained without alignment, averaging or tolerance-based coordinate substitution.

## Loader and precision call-site closure

Historical SoftLoop versions are5ae69a0f (blob),0d91e9f0 (lo/hi) and92200c8e (zooms). The first delta changes only plotting/rendering. The second adds optional second-block state and guarded extra-steering assignments; `extra=None` for these one-block DSS calls, and the relevant SoftOrbit AD JVP routine is unchanged. Original DSS always passes explicit `float64`, window1 and the command's CUDA device into SoftLoop; SoftOrbit's default JVP method is forward AD through math attention, not the stale dense-central-difference wording in its class docstring.

The narrow original constructor supplement `/tmp/task21f-loader-supplement/manifest.json`, SHA256 `3ebedac0f035e4d1301850328acd407cf5f0e07e4dfa4128003f1e032aafff09`, closes the intervening SoftLoop→SteeredModel→Engine call association. The first two original constructors are identical (`9a68ae21a3365565b9f622416d05ca1e3a90806786615e7b984b406794bee7e0`); the zoom constructor (`d27567f3df9f20924bb48424b944898764d93dbb0467bda93fbea7fee7186cc6`) only adds empty extra-steering dictionaries. Both forward dtype/device/weights directly to Engine.

All three versions use original `explore_llm_attractors.py` SHA256 `8e4537194026fe4dc73e8788d77bd3da9390fb4ea1d65ef34fb5efbaede179c9`. Its relevant CUDA float64 branch casts the loaded model tofloat64 and calls `keep_float64_norms`; the helper patches RMSNorm arithmetic and reconstructs rotary frequencies/trigonometry in float64 with source checks. This is an exact source association, not new verification of historical backend binaries, weight bytes or composed-map precision. No device/model execution or convergence test was repeated, and no new precision premise is inferred from NPZ dtype labels alone.

## Adapter disposition

Use the accepted core with these exact optional-field/schema distinctions. Original input hashes, archive/receipt authority and per-run status remain the coordinator's finite admission basis. Preserve the previous discarded fixed-reference method, the outcome-informed transported amendment, all original runs and every nonfinite/unqualified cell. The closest-gap1.3e−3 claim still lacks a named saved analysis definition in this bounded packet; it remains unverified rather than inviting an outcome-chosen new distance metric. No actual Task21f arithmetic or native claim has been accepted by this source note.
