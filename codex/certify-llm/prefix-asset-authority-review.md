# Actual asset and historical prefix authority review

13 October 2026 UTC. Independent reviewer: Codex `literature_cascades`.

**Verdict:** the actual pinned constants have the finite stored types required by the reviewed exact-real evaluator, and the two historical successful prefix computations support enclosures of **the same exact fixed prefix**. This conclusion uses the original assets, captured source and mathematical enclosure argument below; cache overlap alone is not its justification. It closes the actual-asset and historical-prefix obligations numbered 1–2 in `final-evaluator-obligations-review.md`, subject to the explicit trusted execution/arithmetic premises here. It does not close that review's final-target derivative, image, root, primitive-period, attraction or independent-result-audit obligations. `certificate_passed` remains false.

No model was constructed, loaded as tensors, or evaluated. The audit streamed the checkpoint's raw bytes in bounded chunks, read its safetensors header and the small selected vector, and decoded existing prefix enclosures. No GPU, paid resource, protected data, optimizer action, target replay, existing-source edit or Git mutation was used. Read-only `git cat-file blob` comparisons checked the captured source identities.

## Actual fixed constants

All three original file hashes match the constants in the unchanged `codex/crosscheck/qwen_map.py`, SHA-256 `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e`, and both historical input records. The exact paths and resolved filesystem paths are retained in the audit JSON.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| Checkpoint `model.safetensors` | 988,097,824 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Architecture `config.json` | 659 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| Steering archive `vectors.npz` | 14,860 | `b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41` |

The checkpoint header has 290 tensors, exactly the complete tied Qwen architecture inventory, and **494,032,768 coefficients**. Every tensor is stored as BF16. All shapes, contiguous nonoverlapping byte offsets and payload lengths were checked against the explicit architecture. A raw little-endian 16-bit scan of every coefficient found **zero exponent-255 encodings**, so no infinity or NaN is present. This is stronger than trusting the header's dtype declaration or a subset of sampled weights. The scan never materialized the approximately 4 GB binary64 model.

Each finite BF16 number is an exact binary dyadic with a significand and exponent range contained in binary64. Therefore the loader's BF16-to-binary64 widening introduces no numerical approximation, assuming the specified safetensors/Torch conversion behaves correctly. The subsequent NumPy views preserve those binary64 values. The Arb affine kernel converts their exact binary coefficients, not decimal renderings or floating matrix products. This argument does not claim that the BF16 checkpoint reproduces any hypothetical unquantized training weights: the stored coefficients define the map being certified.

The selected `philosophy` entry is **binary64, shape `(896,)`, all 896 coordinates finite**; the archive also contains `gratitude`, which the loader does not select. The selected array's raw-byte SHA-256 is `eea291094ad156ba23b9b3c5f4dd1fb1be1d028d71fe85ce04479c8e4afff3c6`. Its copy, conversion to a CPU binary64 tensor, and later NumPy view require no rounding. No wider floating dtype or object array is being silently narrowed here.

The actual JSON dimensions are integers, not booleans or coercible fractions: state dimension 896, 24 layers, 14 query heads, two KV heads, intermediate width 4864 and vocabulary size 151936. The resulting head dimension is 64 and group size is seven. The actual configuration has tied embeddings, SiLU, zero dropout and no sliding-window attention; rotary scaling is absent and the default full-head rotary convention applies. This establishes the supported configuration concretely, rather than relying on permissive generic `int(...)` conversions.

The exact scalar conventions are:

- `alpha = 5164655995074447 / 72057594037927936`;
- `log10T = -8286623314361713 / 72057594037927936`;
- the JSON epsilon parsed as binary64 is `4722366482869645 / 4722366482869645213696`;
- the rotary base parsed as binary64 is exactly 1,000,000;
- `T = exp(log(10) * log10T)` and frequencies are `exp(log(base) * (-2*j/64))`, enclosed by Arb.

These conventions distinguish the two parameter dyadics and epsilon from their short decimal text. All 37 actual prefix IDs are exact in-vocabulary integers and agree with both historical records. They are saved in full in the audit JSON. Steering is fixed at zero-based layer 12.

## Historical construction and byte authority

| Binding | 1024-bit prefix | 2048-bit prefix |
| --- | --- | --- |
| Attempt | `arb-pilot-002` | `arb-point-003` |
| Source commit | `0c650f87ad1ba20c3afeeeae443b4d5e7bbcc0a0` | `d674d2101eae58bd09dc919530892f51c56017d3` |
| Externally supplied receipt SHA-256 | `565ecc98b37b592bb1da16a58ad4109e1dd2ab5f2eac71682f2af860531f33b9` | `b286831e06d65280ddc44591ba196e40c2adbb33f125c20a676c615434dd51e5` |
| Complete artifact inventory | 22 / 22 | 32 / 32 |
| Captured source files matched to manifest and exact Git blobs | 9 / 9 | 20 / 20 |
| Prefix gzip SHA-256 | `153bca72b85f4baf06bc536c4673b86253df849e4405c9513350c2288f4d4d22` | `9d4a4193ecbb997a68c067312cae20c1f96de3e1919f7218e841715bad231053` |
| Ball evaluator SHA-256 | `3250d8b262e4a940b83de84005ad6f3155bc70635bfc623b0a000804cfc28c3e` | `fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80` |
| Pilot source SHA-256 | `09ccc111a314f68a1b721c9c8bdb6de465ea874264d0e430007f4591308eb6ea` | `354f166af4ac0d7afbe68d2b9c61ed17e54aa9f3e4ed2b448bef5e70545ccca1` |
| Overall recorded outcome | Later box-JVP setup failure, exit 1, no timeout | Prefix and point completed, exit 0, no timeout |

Both attempts live under `/data2/neural-shrimp-data/codex/llm-p8-certificate/`. Every inventoried file was rehashed and its size checked, including files outside the prefix stage. There are no unlisted files except the receipt itself, duplicate inventory entries, symlink substitutions or writable artifact files. The source manifest, receipt, input record and recorded remote-main revision agree. This checks the archived remote-revision record and local Git blobs; it is not a new network verification of the remote repository.

Each command launches its captured `pilot.py` through isolated Python `-I -B`, with one numerical thread and the recorded precision. Both use the same asset loader and affine kernel bytes. `ball_linear.py` is unchanged at SHA-256 `5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2`. Each event log contains, in order, all 24 prefix starts and all 24 prefix completions, then one prefix-save event whose digest matches the actual gzip bytes. All completed prefix progress bounds are finite, with exact-zero reported tangents. Standard output equals the event-log bytes.

The 1024-bit attempt's subsequent RMS-denominator refusal in the much wider state-box diagnostic remains a failure. Its successful, fully serialized earlier constant-prefix computation is a separate completed stage; the failure is neither erased nor treated as evidence against the real map. The 2048-bit driver adds only the explicit `--skip-box` option and associated status branch after the completed point stage; its prefix computation is unchanged apart from precision and the evaluator hooks described next.

## Why both prefixes enclose the same exact object

I read the full historical-to-current evaluator diff and the original pilot construction sites. The only evaluator changes introduce default static hooks for `Dual.make`, `dual`, `rmsnorm`, `softmax`, `linear_sequence` and the finite-value predicate, then route the previous direct calls through those hooks. Both original pilots instantiate **`BallQwen` directly**, without a subclass or method rebinding. The defaults invoke precisely the previous functions. The finite test changes only the equivalent Boolean spelling `not A or not B` to `not (A and B)`.

A separate AST control verified the **entire module** becomes identical to the historical module after explicitly inlining exactly these six defaults and the equivalent finite predicate. The current source is byte-identical to the 2048-bit captured source. This check supports, but does not replace, the mathematical reasoning: arbitrary future hook overrides would need their own review.

Define `K*` as all 24 layers' K/V tensors obtained from the 37 exact embedding rows, the fixed causal decoder graph, the exact stored weights/vector and fixed alpha. At layer zero, the admitted embedding constants are exact dyadics, enclosed by the initial scalar constructors. At each subsequent decoder operation, the reviewed Arb affine, residual, rotary, RMSNorm, softmax and SiLU operations enclose the corresponding real operation on their admitted input sets. The fixed shifts used by softmax are harmless common constants. The completed stages have passed the code's positivity and finite-value gates. Induction over this finite computation encloses each exact K/V entry of `K*` at either precision.

The causal mask admits positions 0 through the current prefix position; integer rotary positions are 0 through 36. Steering is added after decoder block 12 at every prefix position. Consequently the layer-12 K/V values are formed before that block's steering addition, and subsequent layers see the steered hidden states, consistently in both sources. The prefix is constant with respect to the appended soft state. No numerical Torch K/V cache is imported into either Arb prefix computation, and no proposal point, sampled token label or empirical recurrence is used to set its exact values. Temperature is part of the fixed map identity but is not used within prefix attention.

It follows, under the stated arithmetic/execution assumptions, that

`K* belongs to saved_prefix_1024` and `K* belongs to saved_prefix_2048`.

This is the needed same-map authority. It is not deduced merely from similar metadata, successful cache import, midpoint agreement or mutual overlap.

## Complete payload and decoder checks

Each original serialized cache has exact shape **24 × 2 × 37 × 2 × 64**, totaling **227,328 balls**. Every leaf was checked for the exact midpoint/radius schema, integer mantissa/exponent pairs, nonnegative radius, finite Arb reconstruction and valid original rational endpoints. Per-layer original midpoint mantissas fit the declared 1024/2048-bit precision. No leaf or layer was sampled or skipped.

At its own declared receiving precision, every reconstructed ball preserves its original exact midpoint and encloses both original rational endpoints. As expected for outward magnitude reconstruction, **111,887** of the 1024-bit radius encodings and **112,488** of the 2048-bit encodings expand on decoding. Expansion is allowed; no radius was dropped or replaced by a midpoint. These checks exercise the same `arb(mid_pair, rad_pair)` operation used by `cache_io.unpack`. Subsequent `Dual.make` returns an existing finite Arb value without narrowing it; the separately reviewed Jet importer likewise retains those primal enclosures and seeds zero state derivatives.

As an additional consistency check, all **227,328** original 2048-bit intervals are contained in their corresponding original 1024-bit intervals. This comparison uses exact serialized endpoints, before any decoder expansion. It is supporting evidence only; the source/induction argument above is what identifies the same enclosed exact prefix. The importer still requires the receiving arithmetic precision to equal the chosen prefix precision: the 1024-bit derivative path and 2048-bit point path must retain their separate pinned prefixes.

## Immutable usage and trusted premises

The original `pilot.py` creates NumPy views of the loaded weight tensors and marks **each weight view read-only** before constructing the ball evaluator. The vector is a binary64 NumPy view of an independently cloned Torch vector; it is not separately marked read-only. The weight tensors themselves also retain writable Torch aliases. I checked the bounded driver's source and its loader/map operations: they contain no weight/vector/config mutation after construction, no optimizer, and no callback that mutates evaluator inputs. Prefix construction and serialization are single-threaded; subsequent block calls create new lists and sums rather than modifying stored prefix entries. The config is copied, prefix IDs become a tuple and scalar parameters are constructed once.

Thus immutability is supported by the actual controlled call path, **not enforced against hostile alias mutation**. The existing identity guards do not establish that arbitrary external code cannot mutate arrays or objects. A future caller must continue to satisfy this precondition. The audit also assumes ordinary faithful execution of the captured Python source; hashes and event records are not machine-level execution attestations.

Recorded and current arithmetic versions agree: python-flint 0.9.0 and FLINT 3.6.0; both original records identify NumPy 2.5.3 and Torch 2.14.1+cu130. We explicitly trust the enclosure semantics of Arb/FLINT and the binary storage/conversion semantics of the loader stack. This review neither verifies the numerical backend formally nor reconstructs historical wheel binaries from version strings. It establishes concrete finite file inputs and reviews the source-to-enclosure chain under those standard trusted-computation premises. The generic constructor's potential coercion weaknesses do not apply to the actual integer config, finite BF16 checkpoint and binary64 vector just checked.

## Retained independent audit evidence

The complete audit succeeded in **76.044 seconds**, one CPU thread at nice 10, peak RSS **523,312 KiB** (about 0.50 GiB). It did not repeat prefix or model arithmetic. The following files are new audit outputs under `/tmp`; archive these bytes before using their hashes as a durable final authority.

| File | SHA-256 |
| --- | --- |
| `/tmp/codex-prefix-asset-authority-audit.py` | `ab6b7c1bc94c5f5a220255f2f47b94ed86315b7d8f451ddecc7830e09b0200d0` |
| `/tmp/codex-prefix-asset-authority-audit.json` | `1cf096b123bd8c35a8b2a1707846d5fc3d1669d98677dc5b3e6db394ac47854b` |
| `/tmp/codex-prefix-hook-equivalence.py` | `cede0e5405a98714a27c5bfd438f63f20e904cc10d6dc49ff1e81b0a0651b0b2` |
| `/tmp/codex-prefix-hook-equivalence.json` | `9b4ac0dee706c8fe8c4fa0b1e3007ca70f7c37ad49fd1890da01dff58441c182` |
| `/tmp/codex-prefix-map-equivalence.diff` | `6a657efa7d8a9e023a03eaa31bed84e83fc7e24c7a595ce0afbecaa40acdce24` |

The output retains every tensor's shape/dtype/offset/finite count and every prefix layer's complete checked count, maximum stored radius pair, midpoint precision and decoder expansion count. The prior `pilot-results-review.md` and `point-003-review.md` were read for context; their reported successes were not substituted for the fresh byte, dtype, inventory, source-blob and exact-endpoint checks above. No target orbit conclusion follows from these prefix obligations alone.
