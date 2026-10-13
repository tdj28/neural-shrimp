# Independent evaluator obligations for a final Qwen P8 certificate

13 October 2026 UTC. Reviewer: Codex subagent `boundary_resume`, acting only as the independent evaluator-contract reviewer for Task 1. This review read source and existing reviews in the isolated `llm-p8-certificate` worktree. It performed no checkpoint load, target model call, target artifact admission, center refinement, derivative campaign, Git operation or change to existing source. Earlier boundary experiments are outside this task.

**Verdict:** no formula-level blocker was found in the current scalar, Jet, affine or streamed evaluator for the fixed exact-real state map, under the concrete input, cache and trusted-arithmetic premises below. The source supports the required analytic and enclosure arguments. Those arguments do not establish that any particular target artifact actually satisfies the premises. The final assembly and independent result audit must discharge the remaining evidence obligations before release. This is a source-level mathematical review, not a current orbit certificate or proof-assistant verification.

The conclusion concerns periodic behavior of a frozen expected-embedding feedback map. It says nothing by itself about ordinary autoregressive token generation, training dynamics, useful language behavior, or global shrimp organization.

## 1. Exact map being reviewed

Let `d=896`, `V=151936`, and let `E` be the complete `V` by `d` checkpoint embedding matrix, with each admitted stored binary coefficient interpreted as an exact real dyadic. All other admitted weight and bias arrays have the same exact-coefficient interpretation. The pinned configuration fixes 24 decoder layers, causal attention and grouped KV heads, tied embeddings, SiLU, positive RMS epsilon, default full-head rotary embedding, and zero dropout. Unsupported variants are rejected by `BallQwen.__init__`.

Fix the complete prefix token tuple, zero-based steering layer 12, stored steering vector `v`, and the exact binary64 values represented by Python literals `alpha=.071674` and `theta=-.115`. These are **not** the decimal rationals 71674/1000000 and -115/1000. The mathematical temperature is

    T = exp(log(10) * theta) > 0.

Likewise the rotary frequencies are the real expressions

    omega_j = exp((-2*j/head_dimension) * log(base)),

and rotary sine/cosine are evaluated at exact integer positions. The head scale is real `1/sqrt(head_dimension)`. The configuration's epsilon and rotary base are interpreted through the documented binary64 conversion of the pinned JSON numeric values. No rounded Torch temperature, inverse-frequency or trigonometric table enters the ball evaluator.

The prefix is recomputed in this mathematical graph, with steering added after the selected decoder block at **every** prefix/current position. Its per-layer keys and values are constants with respect to the appended state. Each application appends precisely one full-dimensional soft embedding `x` at position `len(prefix_ids)`. The prefix cache is reused as a constant; the newly processed soft token is not appended to that cache for the next feedback iteration.

Write `h(x)` for the resulting final RMS-normalized decoder output. The returned map is

    z(x) = E h(x) / T,
    p_a(x) = exp(z_a(x)) / sum_{b=0}^{V-1} exp(z_b(x)),
    F(x) = sum_{a=0}^{V-1} E[a,:]^T p_a(x).

Both scalar and streamed implementations include **every** vocabulary row in the softmax denominator and expected embedding. There is no top-k selection, discarded vocabulary tail or argmax operation in `F`. The optional label diagnostic is downstream evidence about logit margins; its discrete candidate selection does not define the dynamical map.

`codex/crosscheck/qwen_map.py::from_pinned_files` is relevant here as the admitted asset loader. Its numerical transformer and float64 proposal Jacobians are not the exact evaluator. The source hashes the config/checkpoint/vector assets before loading them, converts tensors to float64, selects the `philosophy` vector, and passes those arrays to the ball constructor. Exact file-to-array identity still requires the concrete stored dtypes/values and pinned bytes to satisfy this conversion contract. Hash equality is not an independent proof of a loader's behavior.

## 2. Analytic hypothesis: a C1 neighborhood exists

Under the fixed finite-real constants above, `F` is in fact real analytic on all of `R^d`, so it is C1 on an open neighborhood of every finite nominal phase box.

- Linear maps, biases, residual additions, fixed steering, and fixed-position rotary rotations are affine or polynomial in their variable inputs.
- For every real vector `u`, the RMS denominator satisfies `mean(u_i^2)+epsilon >= epsilon > 0`. Its reciprocal square root is analytic on that positive range. Learned RMS weights do not change this denominator argument.
- SiLU is `g/(1+exp(-g))`; its real denominator is strictly positive everywhere.
- Each attention softmax and the final all-vocabulary softmax has a finite number of positive exponential terms. Its real denominator is strictly positive. `T` is a fixed positive constant.
- Head grouping, causal masks, prefix length, rotary positions, steering placement and tensor dimensions are fixed indices. They introduce no state-dependent piecewise branch. A finite composition of these analytic operations remains analytic.

This is a mathematical argument about the exact real graph, not an inference from small observed residuals. The evaluator also has computational domain gates: its actual interval denominators must exclude zero, RMS and temperature enclosures must be strictly positive, and returned values/derivatives must be finite. A wide interval multiplication `u*u` may include negative numbers even though the exact square is nonnegative. Failure of that interval positivity gate is a conservative inability to enclose the graph by this calculation, not a nonsmoothness or nonexistence result.

The softmax midpoint shift is a conditioning constant chosen once for the supplied enclosure. For **any** finite common shift `s`, `softmax(z-s*1)=softmax(z)` identically. It is therefore valid to hold that selected `s` fixed when differentiating the real map over the entire box, even if the algorithm would choose another midpoint for another call. Neither a differentiable argmax nor a derivative of the chosen shift is required. In the scalar evaluator `key=float` only selects such a shift; it does not provide a proof bound or alter the real identity.

## 3. Enclosure and derivative reasoning

The trusted numerical premise is the documented enclosure semantics of the pinned python-flint/FLINT/Arb implementation: exact supported constructors and outward scalar/matrix arithmetic, including `exp`, `log`, `sin`, `cos`, reciprocal square root and matrix products. This review does not prove that backend, Python, NumPy, Torch asset conversion or the machine itself correct.

### Affine and scalar operations

`ball_linear.affine` copies each bounded coefficient block to binary64 and constructs its Arb matrix without a floating dot product. Widening finite binary16/32 values to binary64 is exact; binary64 values are unchanged. Supported dtypes, shape and finiteness are checked. Each product and partial sum is an Arb operation. A transposed/noncontiguous embedding view still contributes its numerical coefficients; no reinterpretation of raw layout as another matrix occurs. Bias is applied once, and the Dual/Jet callers select only the primal column of each token group. Tangents receive `W*U`, not `W*U+b`.

The weight-cache key includes exact copied block bytes, shape and precision. It caches coefficients, not input-dependent products. Reusing those exact points does not import a floating product or erase input radii. The map and its aliases must remain immutable throughout the call; cache guards are not defenses against concurrent hostile mutation.

For a Jet `(f, (D_k f)_k)`, the implemented rules are

    D_k(fg) = (D_k f)g + f(D_k g),
    D_k(1/f) = -(D_k f)/f^2,
    D_k exp(f) = exp(f) D_k f,
    D_k f^(-1/2) = -(D_k f) f^(-3/2)/2.

The signs and powers in `Jet` and scalar `Dual` agree. RMSNorm applies these rules to the full quadratic mean and epsilon. Softmax uses

    D_k p_a = p_a * (D_k z_a - sum_b p_b D_k z_b).

All quantities, including primal states and tangents, retain interval radii. By induction over the finite computational graph, each Jet value encloses its true value for every admitted state, and each tangent encloses the indicated derivative at that same state. Dependency losses may widen these sets; no unjustified cancellation or midpoint replacement is used to narrow them. Interval-valued constants can be treated as containing the fixed true constants; their repeated use without correlation remains an overestimate.

### From signed directions to the whole Jacobian

A final shard sets exactly 64 positive coordinate directions: `U[start+k,k]=1`, every other entry zero. They are not a random subspace or one interval direction cube. The live callback checks these exact coefficients after scalar construction and before decoder block zero. If the receiver box `Y` contains the nominal box `X_i`, induction gives

    for every x in X_i, DF(x)e_j is contained in the recorded column j.

The full derivative is recovered only after all 896 directions occur exactly once for each of eight phases: 14 shards per phase, 112 total, 7168 columns. `derivative_coverage.collect_derivatives` rejects missing/duplicate directions and non-basis seeds, keeps every output row, and assembles signed entries. Different shards may use different containing receiver boxes or interval widths. That does not invalidate the entrywise matrix enclosure on their common nominal box: each actual `DF(x)` has each of its columns in its corresponding admitted interval column. No low-rank assumption or bound on omitted directions is available or needed once this complete coverage succeeds.

The common primal intersection check across shards is a valid consistency check: each primal enclosure contains `F(X_i)`. It is not a substitute for authenticated point images `F(c_i)`, nor does nonempty intersection alone prove that a derivative was evaluated correctly.

## 4. Scalar points and the streamed derivative head evaluate the same map

The Jet subclass inherits the same transformer block, causal attention, grouped-head indexing, rotary formulas, residuals, SiLU, steering and prefix machinery. Its arithmetic hooks replace scalar construction, coercion, RMSNorm, softmax, affine sequence packing and finiteness checks consistently. `StreamedHeadQwenJet.forward` mirrors scalar forward through the final RMSNorm, with the same appended position and prefix. Its extra orchestration changes how the output head is accumulated, not the decoder graph.

For hidden derivative matrix `U`, let `Z=E U/T` and `mu=p^T Z`. The complete scalar-Jet readout gives

    D F = E^T [diag(p) Z - p mu]
        = E^T diag(p) Z - (E^T p) mu.

The streamed first pass forms all `V` primal logits and their common normalization. In its second pass, each token chunk contributes `E_chunk^T p_chunk` and `E_chunk^T diag(p_chunk) Z_chunk`, and contributes to the signed sum `mu`. The final subtraction of `primal[i]*mean[k]` is exactly the outer-product term above. Both passes visit rows `0..V-1` exactly once; the last partial chunk is retained. The driver records ordered completion of both passes after all decoder layers.

This identity also holds when the fixed temperature is only enclosed by a nonzero-radius positive interval. Both implementations then enclose the same expression at its actual fixed `T`; interval dependency can differ between them. Bitwise equality or equal widths between scalar and streamed images is neither expected nor a condition for correctness. Overlap of numerical outputs, by itself, would not prove this identity; the algebra and common graph do.

The point worker requires the live state to be exactly the requested nominal center and every live state tangent exactly zero, and retains the full output image balls. Its scalar forward uses the same all-vocabulary tied embedding. `point_evidence.map_bridge` reconstructs the distinct point and derivative schemas from the same externally supplied assets, prefix IDs and dyadic parameters; the vocabulary field names/digests remain distinct. That bridge authenticates a declared common map. The source argument in this section supplies the mathematical reason those implementations represent it.

## 5. Separate 1024/2048 prefix enclosures

Let `K*` denote the single exact prefix KV collection computed by the fixed graph and constants in section 1. A valid point cache `K2048` and derivative cache `K1024` satisfy, componentwise,

    K* in K2048,   K* in K1024.

There is **no** requirement that their serialized values be equal, that one interval contain the other, or that they have the same width. Both contain the same exact constant. Agreement of metadata or overlap of cache balls is not sufficient to establish that assertion; each needs its own actual construction/provenance authority.

The current prefix code starts from exact stored embedding rows with zero state tangents, applies causal blocks in order, saves each block's pre-steering KV pair, and applies steering to the residual that feeds subsequent blocks. This is the correct fixed causal-prefix dependency. The appended state cannot affect prefix activations. Installing zero state tangents on nonzero-radius cached primal enclosures is consequently correct for derivatives with respect to `x`; it would be wrong for derivatives with respect to alpha or another parameter that changes the prefix. Such parameter derivatives are outside this contract.

`cache_io.load_prefix` binds an external receipt and original source/assets, exact parameter ratios, token tuple, backend, matching precision, and real temperature/rotary conventions. It checks receiving config and the identical admitted weight/vector objects, and exact layer 12. It reconstructs all layers/tokens/KV heads/coordinates with full midpoint and radius, then wraps them in zero-tangent scalars. It does not recompute the prefix, independently prove the original source, or itself establish the file-hash-to-array relation. Attribute identity checks also do not protect against mutation through another alias.

`point_evidence.point_prefix` requires the separately completed 2048-bit point-prefix setup. `derivative_evidence.authenticate_prefix` admits the separately pinned 1024-bit successful saved-prefix stage, including its ordered layer events and cache save, even when that historical attempt later failed its unrelated box-JVP stage. That later failure cannot retroactively invalidate a completed valid prefix computation, but it must remain recorded as an overall failed attempt. Neither cache may be relabeled as the other precision. The receiving evaluator recomputes its own transcendental constants at its declared precision from the same exact inputs.

**Remaining prefix obligation:** this review inspected current construction/import/admission source and existing source reviews, not the actual historical cache payloads or captured historical source archives. The final independent evidence audit must bind and verify each historical construction chain (including any reviewed source difference), concrete loader inputs, cache completion and full retained enclosures. Any unread older capsule remains an explicitly named external audit assumption, not a new review accomplished by this document.

## 6. Original-domain and receiver containment

For a nominal exact center coordinate `c` and nominal radius `r>0`, a rounded receiver midpoint `m` needs radius at least `r+|c-m|`. `phase_domain` computes these dyadic quantities by exact rational arithmetic, then constructs outward balls. Its independent endpoint witnesses check the actual ball satisfies

    actual_lower <= c-r,    actual_upper >= c+r.

The derivative driver does not trust only the earlier constructed record: the callback saves and checks the **live Jet scalar inputs** immediately before the first decoder operation. That closes the gap between proposed encoded domains and actual evaluated domains.

The collector rechecks the original serialized receiver endpoints using exact rationals **before** decoding. A too-narrow original input is not repaired by an outward-expanded decoder. Conversely, original derivative/image outputs may be conservatively widened by decoding; their endpoint coverage must never shrink. `derivative_coverage._decode` checks this explicitly, and matrix assignment gets another check against the original endpoints. `point_evidence` decodes full original image radii at the restored assembly context and requires nominal exact center records to survive unchanged.

The original prefix importer relies directly on Arb midpoint/radius constructor enclosure semantics. This is consistent with the trusted backend premise, but it is not the same as the collector's explicit rational non-narrowing witness. No claim that all these decodes preserve literal radius bytes is warranted. The final source/backend contract must continue to distinguish exact centers/radii, original producer balls and the actual native downstream objects.

The two domain hash conventions (`phase_domain` canonical JSON and `shooting_gates.domain_sha256`) identify the same mathematical nominal data via different byte conventions. They must each be recomputed on the actual admitted data; their hash strings must not be equated. Object/content binding after admission is also required: the matrix/image objects handed to the gates must be the non-narrowed admitted objects, with their actual native hashes retained.

## 7. Remaining obligations and release blockers

There is no requested evaluator-code change from this review. The following are still mandatory **for a target certificate**, rather than optional generic assurances:

1. **Actual fixed constants and architecture:** externally authenticate the original config/checkpoint/vector/prefix and exact binary parameter values; establish supported finite stored dtypes and lossless conversions into the admitted arrays. Pinned dimensions must really be the intended integers. The generic constructors use some `int(...)`/`float(...)` coercions and are not validators for arbitrary foreign configs or wider vector types. Preserve immutable config, arrays, aliases, parameters, prefix and context throughout each evaluation. Reviewed bounded callers satisfy the intended usage structurally; no target asset audit was performed here.
2. **Both actual prefix chains:** establish the two inclusions for `K*` above through their separately pinned reviewed computations. Current-source hashing, matching metadata, cache overlap or successful import alone does not discharge this. Verify captured legacy source/backend and retain the 1024-bit containing attempt's failure scope.
3. **Actual final evidence:** externally bind the fixed final-domain definition and all eight exact point images to the same map; authenticate all 112 complete successful final derivative receipts, their complete source bundle, live inputs, basis directions, all decoder/head completion and full output records. A prototype, interrupted shard or partial derivative set cannot count. No such actual target collection was performed in this review.
4. **No enclosure narrowing in assembly:** check original input endpoints, live receivers, every output decode/matrix assignment, native object hashes, exact nominal centers/radii and phase ordering. Preserve all point-image radii. Backend correctness remains a stated trusted-computation premise rather than a fact inferred from its version string.
5. **Conditional gates and independent result audit:** the separately reviewed shooting algebra still needs successful strict root rows, proper-divisor separation and the correctly ordered full monodromy attraction bound on these exact native inputs. Root contraction is not dynamic attraction. The assembly/finalizer must retain failures and cannot promote metadata or a producer Boolean to a certificate. A separately pinned completed independent result audit must review the actual artifact chain and all gate witnesses; its degree of arithmetic replay must be described accurately.

Until these are discharged, an assembly may report conditional gate outcomes, but must leave `certificate_passed:false`. If they are discharged and the mathematical gates pass, this source argument supports the claimed computer-assisted primitive period-8 locally attracting orbit of the specified exact-real feedback map, unique only as the ordered shooting-root tuple in the admitted product box. It does not supply a proof-assistant theorem, a whole-box trapping claim or global organization.

## 8. Small independent controls performed here

The conclusions above come from the formulas and source control flow, not from test counts. As an additional check, one CPU1/nice10 in-memory calculation with a 25-second watchdog used five exact-dyadic embedding rows, three hidden coordinates, two signed directions and `h=0`. The independent oracle was the exact rational covariance formula

    F(0) = mean(E),
    DF(0) U = [mean(E_a E_a^T) - mean(E_a)mean(E_a)^T] U / T.

At 64 and 192 bits, token chunks 1, 2, 4 and 8 passed all 72 exact rational output-endpoint comparisons. Six additional prefix-decoder endpoint checks used positive/negative 2056-bit midpoint mantissas and nonzero magnitude radii at 64, 1024 and 2048 bits; all preserved the original rational endpoints. Total: **78 checks**, 0.004 seconds inside the initialized harness; no checkpoint, model or target artifact was accessed. The first harness invocation stopped at a `Fraction(flint.fmpz)` type-conversion error before comparing an enclosure. Explicit conversion to Python integers fixed the oracle; evaluator code, data and bounds were unchanged.

No existing tests or source were edited. Existing review claims were read as supporting context and scope boundaries, not substituted for this mathematical reasoning. The preserved test evidence also does not prove arbitrary backend correctness.

## 9. Exact reviewed source identities

SHA-256 below binds the inspected source snapshot; it does not assert execution or correctness merely from a digest. The seven requested evaluator/admission files were read in full. Supporting loader, point worker, domain/collector and prefix-admission sections were inspected for their stated bridge obligations; this is not a new audit of all historical capsule code.

| File | SHA-256 |
|---|---|
| `ball_qwen.py` | `fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80` |
| `ball_jet.py` | `46b14d8bd65ed6c80174758daf37d00110cfa908813917bdb46c13749f999719` |
| `ball_linear.py` | `5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2` |
| `streamed_head.py` | `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a` |
| `cache_io.py` | `f4e25b2134bef8a71c27d3156baf01be9d25c593d7b780f90a66b9c1141abdef` |
| `derivative_shard.py` | `3fe44a583176e0551bb209093fec08cd059d055fcb1a57702ac8799c7cb7cc50` |
| `point_evidence.py` | `3b8bcfc0aa3b3f2f3f9886f4f6038325f37288314e3704d371199a156d82a9e9` |
| `phase_point.py` | `e87f1efdffd17925d8ec971f7ce164cfd5c1eaa1f0ad046cbc02916994ec1b4a` |
| `phase_domain.py` | `a3856de84393068788c4cf4dc42fa6015da45b7912b56db995cc6b12922f8d8d` |
| `derivative_coverage.py` | `eb01dc001193b5e3982bcb41a582ac7c61cfe4e8fcd2a1ddf249126623fe63cf` |
| `derivative_evidence.py` | `8124c73351a0a8bd2b6c488983afd00874d2b9be517f15be34393957668d72da` |
| `center_continue.py` (referenced admission context) | `705e3c020c1a7c77f50a5e738d9d519fb428369f7f4865df00ae5f338e58f6ed` |
| `codex/crosscheck/qwen_map.py` | `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e` |
| `final-assembly-design.md` | `74320da74e4cc0fda4c6419c9c63db0eee99f2421ad5f2644ebc1cdfdd22a4f7` |

Unless fully prefixed, paths are relative to `codex/certify-llm/`. Existing source-review context read: `map-code-review.md`, `affine-code-review.md`, `derivative-cache-review.md`, `streamed-head-review.md`, `derivative-shard-review.md`, `point-evidence-review.md`, `phase-domain-review.md` and `derivative-coverage-review.md`. Earlier approvals were explicitly limited to their source/pilot/admission scopes and are not treated as a completed target proof.
