# Independent review of final conditional evidence assembly

13 October 2026. Reviewer: the separately assigned boundary reviewer, now restricted to Task 1 Qwen P8 source review. This is a review of the two requested source files and their existing adapter/gate contracts, not an evaluation of target evidence. No model, checkpoint, candidate artifact, derivative shard, paid service, or original target gate was evaluated. No existing source was edited and no Git command was used.

## Verdict and exact scope

**PASS for the reviewed conditional assembly source and its original-to-native binding contract.** I found no blocking mismatch between the requested `assembly.py`, its synthetic integration tests, and the evaluator obligations previously documented. This verdict does not assert that the actual final domain has passing evidence, that any target gate passed, or that a target certificate exists. The code correctly keeps `certificate_passed:false`, including after all conditional mathematical gates pass. A source review, an authenticated review-file hash, and a producer's declaration are not substitutes for the external mathematical hypotheses or independent final result review.

The requested hashes were confirmed before review and again after testing:

| File | SHA256 |
|---|---|
| `codex/certify-llm/assembly.py` | `46011955b92d3f849f6420e34e73b2a518cc5b709f4094a4c6280f7674eb4bc6` |
| `tests/test_codex_llm_assembly.py` | `afdf9f2948034e780f4e3c7dbe3112063ecb77b3b826cf5106eb6ceb09ff4582` |
| `codex/certify-llm/final-evaluator-obligations-review.md` | `92cf2b93036b92d18c09944da7e3073d2b358756a93ab43eb41a6e50fdc8961e` |
| `codex/certify-llm/final-assembly-design.md` | `74320da74e4cc0fda4c6419c9c63db0eee99f2421ad5f2644ebc1cdfdd22a4f7` |
| `codex/certify-llm/point_evidence.py` | `3b8bcfc0aa3b3f2f3f9886f4f6038325f37288314e3704d371199a156d82a9e9` |
| `codex/certify-llm/derivative_evidence.py` | `8124c73351a0a8bd2b6c488983afd00874d2b9be517f15be34393957668d72da` |
| `codex/certify-llm/derivative_coverage.py` | `eb01dc001193b5e3982bcb41a582ac7c61cfe4e8fcd2a1ddf249126623fe63cf` |
| `codex/certify-llm/phase_domain.py` | `a3856de84393068788c4cf4dc42fa6015da45b7912b56db995cc6b12922f8d8d` |
| `codex/certify-llm/center_refine.py` | `f24f9cdecec243abd25671a684aec03b8b105288526ff7fa4f60a963af745b02` |
| `codex/certify-llm/shooting_certificate.py` | `a4aca57acf7aaab33c77745ea6170935dd6b4dac40c9c982a063743f98727c3e` |
| `codex/certify-llm/shooting_gates.py` | `9c967c0dcb9783df53f33c05eb897050f4e8971f66767abbc8a01a8b1b85cf1f` |
| `codex/certify-llm/proof_gates.py` | `ee6678b7dd5153ef095fd73d5711fcfc605f7a228c92b1335a9541cc3d7b2c6a` |

The prior evaluator review supplies the detailed formula inspection of the fixed exact-real transformer, Jet chain rules, all-vocabulary head, scalar/derivative equivalence, both prefix precisions, and outward receiver containment. The present review checks how those premises are handed to the gate call. I read the new assembly and test files, the design, the relevant point/derivative adapter interfaces, the original proposal reader/converter, and the combined gate wrapper. I did not infer mathematical correctness from these hashes or test counts.

## Original evidence to actual native objects

The externally frozen assembly plan has fixed period 8, dimension 896, columns per derivative shard 64 and assembly precision 1024. `validate_plan` requires the exact transitive source-key set, all four ordered review scopes, explicit separate backend contexts, the exact gate policy and the complete canonical assignment of 112 distinct original shard attempts. Boolean aliases for integer assignments are refused. Original plan bytes are checked before their parsed object is used, and duplicate original JSON keys are rejected. The current source/review bytes are checked before admission and again after the conditional report is written.

This does not make a plan intrinsically authoritative: its external selection and pinning remain an obligation of the caller and subsequent independent audit. In particular, hashing a file named as a review does not establish the contents' truth. The manifest states this limitation explicitly.

The point adapter is called on the original frozen definition and the independently pinned point-admission plan. It checks original exact centers before any numeric decoding. The returned centers must repack identically to the original nominal definition. The assembly reconstructs nominal radii from the exact dyadic midpoint records and then requires their entire packed representations to equal the original radius records. Thus a radius decoder cannot silently change the nominal theorem domain. Unused historical seed centers are discarded; they are not substituted for the final centers.

The original point images are admitted at their separate 2048-bit source context and outward decoded into the assembly context. The actual returned native image hash is checked again before the gate call. Their radii are retained. The derivative collector authenticates all original assigned attempts, including actual live receiver inputs, basis directions, complete decoder/head completion, and all derivative outputs. Its rational original-input checks precede decoding, so a too-small original receiver cannot become admissible merely because an Arb reconstruction widens it. Its full-column and native assignment checks are then tied to the eight native matrix digests used by the gates.

The assembly checks both independently defined domain digests: the phase-domain JSON identity and the shooting-domain identity. It does not equate these different hashing conventions. It requires every phase to have all 896 columns and the total to be 7168. It also compares every authenticated shard's phase and ordered direction indices with the externally assigned 64-column block. This supplies full-state derivative coverage rather than an orbit-tangent or selected-column surrogate.

As an additional corruption/consistency test, each original point-image interval must intersect the original primal intersection collected from that phase's whole-box derivative evaluations. The comparison is made using exact rational endpoints from the original records, before native expansion. A contradiction stops admission. Conversely, a nonempty intersection does **not** prove either evaluator correct or prove the two formulas equivalent; those remain the reviewed evaluator premises.

The combined enclosure digest is computed from the actual native images and Jacobians, not from a receipt hash or pre-decoding payload digest. The proposal digest likewise uses the actual native A/B matrices. The saved admission manifest, its hash, and these recomputed values form the gate contract. The gate wrapper checks the actual input digests itself, and the assembly checks the returned digests again. These checks prevent an authenticated artifact from being confused with a different set of objects subsequently supplied to the theorem calculation.

## Independent exact A/B conversion check

The original seed reader authenticates the complete seed receipt and seven-array NPZ inventory, exact finite float64 shapes, seed map parameters, numerical phase records, and archived numerical source identities. Its returned source commit is compared with the external seed pin. The seed/NPZ/driver/map pins are explicitly cross-bound to the point admission's base plan. No optimizer, new inverse, or seed refinement runs in this path.

`base.exact_matrix` is not trusted solely because its output is exact. After conversion, `verify_exact_proposals` independently visits all eight A matrices and B. For each authenticated finite binary64 value x it computes the exact integer ratio n/d using `float(x).as_integer_ratio()`, checks that d is a positive power of two, and constructs the real value n*2^(-log2 d). At 1024-bit precision this value is exactly representable: a finite binary64 number has at most 53 significant binary digits, regardless of its exponent. It then requires each actual converted Arb entry to be finite, have zero radius, and equal that independently reconstructed value. Shapes are checked as well. Signed binary64 zero has the same exact real value in this contract.

This check is independent of the converter's algorithm, although it shares Python's binary64 interpretation and the trusted Arb representation. It is not a second arithmetic backend. A changed-but-still-exact converter is rejected by a dedicated corruption test. These matrices remain arbitrary exact preconditioner proposals; they need not be derivatives at the final centers or proven inverses. The actual root gate must prove the needed defect/inclusion inequalities.

## Same fixed map, prefixes and backend identification

The assembly obtains checkpoint/vector/config hashes and fixed prefix IDs from literal assignments in the independently reviewed, externally source-pinned `codex/crosscheck/qwen_map.py`. In the production entry point, the transitive source checks run before this extraction. The pure `assemble` function instead receives independent expectations explicitly, which its docstring correctly makes a caller premise. Artifacts do not choose their own expected map within the production path.

`map_bridge` binds the distinct point and derivative schemas to the same fixed exact-real map. The point report retains both schemas; the contract uses the derivative schema without relabelling the point schema. The evaluator-source digest covers the point and derivative source bundle and the source-bound reviews, rather than only one driver. The current arithmetic backend is compared with the separate 2048-bit point and 1024-bit derivative/assembly contracts, and both the adapter and assembly check precision/thread restoration.

The two prefixes are deliberately separate authenticated attempts. The 2048-bit point prefix and the retained 1024-bit derivative prefix enclose the same fixed constants under the earlier mathematical review. Their interval payloads need not be equal or nested. The assembly rejects an attempt-path substitution between them and delegates the detailed historical stage/receiving-array/source checks to the unchanged adapters. It does not promote the historical failed box diagnostic into a successful derivative evaluation.

The Torch version reader uses `importlib.metadata` to locate `torch/version.py`, parses its AST, and accepts exactly one top-level literal `__version__` assignment or annotated assignment. It preserves a producer version's local suffix, such as `+cu...`, which a distribution's simplified version can omit. It does not import Torch, execute the file, or construct a model. The version path/hash, distribution version, literal producer version and no-execution flags are retained. The two reads during main admission must agree on the producer version. Synthetic tests demonstrate no execution even when the version file includes a raising statement, preserve a local suffix, and reject duplicate assignments.

This reader identifies the installed version metadata used for compatibility checks. It is not a binary/library integrity proof, proof that arbitrary Python code would ultimately expose the same attribute, or proof of the historical producer's execution. The immutable installed backend and honest captured runtime remain explicit trusted-environment premises. No claim stronger than that is needed for this conditional assembly. In particular, the reader does not replace the native FLINT/Arb enclosure assumption.

## Contract handed to the mathematical gates

The external declaration requires one fixed C1 map on neighborhoods of the nominal boxes, exact-center F(c_i) enclosures, and whole-box DF(X_i) enclosures. The prior evaluator review gives the source-level reasoning for the stronger real-analytic property in this fixed model: positive RMS epsilon and softmax denominators, fixed indexing/attention pattern, fixed prefixes, and smooth elementary operations. It also explains the full-vocabulary readout and full coordinate-basis Jet derivatives. The present code neither changes those formulas nor claims metadata establishes them.

The gate call is exactly the fixed factored shooting root check plus proper-period and attraction checks, with powers (1,2) and no stability weights. The root equations are H_i(x)=F(x_i)-x_(i+1), including cyclic closure. Passing the root gate is a conditional existence/local uniqueness result for the ordered full-state shooting tuple in the product box. The primitive gate separates the same nominal phase boxes for divisors 1,2,4. It need not and does not assert forward invariance of an entire phase box.

The combined wrapper forms the interval product by `monodromy = jacobian * monodromy` for phases 0 through 7, yielding J_7 ... J_0. Each phase's whole-box derivative enclosure therefore encloses the corresponding factor at the shooting root. A strict induced-norm bound below one for this product or its square supplies local attraction. Root-solver contraction is not substituted for dynamical attraction. The retained statement correctly limits uniqueness to the ordered shooting tuple and attraction to a neighborhood of the orbit.

All three results are retained even if one fails. `conditional_assembly_completed` can therefore coexist with `conditional_gates_passed:false`; it means the calculation completed, not that an orbit was proved. The unconditional certificate flag remains false throughout. A failed sufficient bound is not evidence that no orbit exists, and no global shrimp, training, or language-model benefit follows from these gates.

## Persistence, test evidence and remaining obligations

The assembly writes original candidate/admission-plan bytes and each completed adapter/proposal report before the gate call. It then writes the full native-input admission manifest and contract. A compact gate summary precedes the potentially large full report. JSON streaming preserves a `.partial` file on serialization failure and atomically completes the destination with fsync. Main catches failed admission/calculation/serialization and records a failure stage with a false certificate flag. A pre-existing run directory is refused rather than overwritten. Final source/review/original-plan checks can still turn a late attempt into failure; an earlier summary cannot override that failure.

I ran only the tiny synthetic integration suite:

```
timeout 90s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nice -n 10 /data2/neural-shrimp/.venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_codex_llm_assembly.py
```

**39 tests passed in 8.65 seconds.** The fixtures use synthetic one-coordinate, eight-phase evidence, actual adapter/gate code, isolated copied-source imports, and deliberately corrupted artifacts. They include missing/duplicate/misassigned shards, backend/map/source/review changes, original low-bit center/domain mismatch, narrower original receivers, failed original attempts, changed native matrices/images/proposals, exact-but-wrong A/B conversion, unstable mathematical gates, original primal contradiction, duplicate JSON keys, interrupted gates and interrupted serialization. The tests do not evaluate model weights or target artifacts. Nonfatal shell environment messages saying `Failed to create stream fd: Operation not permitted` appeared; pytest still completed with exit 0. No cache/bytecode output was requested in the worktree.

This is not a target-size feasibility measurement. The 896-dimensional gate cost and all actual original evidence remain unevaluated here. Before a run, a separately reviewed source-bound launcher/card must enforce the external wall-time/ownership/resource envelope and preserve exit/timeout/partial artifacts. The CLI installs an address-space limit and a SIGTERM handler, but does not itself create the 3600-second watchdog. A bare direct invocation is not the complete bounded launch protocol. Resource completion is separate from the mathematical gate flags.

Before a certificate release, the following remain mandatory:

1. An authorized externally frozen plan with actual original artifacts, complete source/review pins, and all 112 successful final-domain shard receipts; admission must succeed for those exact files. This review has not checked their existence or outcome.
2. The evaluator premises for that same fixed map and actual domains, including original receiver coverage, correct same-constant prefix evidence, non-narrowing transfer, exact-center images, and all full-state Jacobian entries. The source reasoning is conditional on the pinned assets, actual admitted configuration and trusted arithmetic implementation.
3. All strict target root, primitive-period and attraction inequalities, including actual signed monodromy order, with complete serialized witnesses and unchanged input/context identities.
4. Verified bounded process completion and intact final artifacts. A saved intermediate success field cannot overrule a later timeout, authentication failure or incomplete write.
5. A separately authored/authorized post-run result audit, externally pinned by a later finalization plan. That audit must say exactly what it independently recomputed, preserve the explicitly inherited older-capsule trust boundary, and discharge the stated hypotheses for these artifacts. Review-file hashes alone are insufficient. The finalizer requires its own reviewed source and tests; it is not implemented by this assembly.

No source fix is requested for the reviewed conditional assembly. These remaining run/result/release obligations are intentional scope boundaries, not facts proven by the synthetic suite. Any eventual claim is a computer-assisted exact-real orbit result with documented arithmetic/evaluator/audit assumptions, not proof-assistant formal verification.
