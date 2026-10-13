# Independent review of the Arb affine kernel

13 October 2026 UTC. Reviewed `codex/certify-llm/ball_linear.py`, SHA-256 `5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2`, its synthetic test suite, and the primal/tangent packing caller in `ball_qwen.py`.

**Verdict: no correctness blocker found for the bounded, source-pinned enclosure pilot.** The implementation encloses affine maps with exact stored binary coefficients and Arb arithmetic. This review does not certify the nonlinear model, an orbit, a Jacobian norm, or the eventual proof gates. No checkpoint was loaded and no target model calls were made. Only this review document was written; source and tests were not changed.

## Exact coefficients and arithmetic

`exact_dyadic` admits finite Python binary64 floats and NumPy binary16/32/64 values, rejecting wider floating types and non-floating inputs. Widening binary16/32 to binary64 preserves their exact values, including subnormals. The scalar Arb float constructor represents the stored binary value exactly; it does not reinterpret a decimal string. Direct `arb_mat` construction from the flattened Python float block has the same property. Independent checks confirmed point exactness at context precision 16, 53 and 256 for ordinary binary64 values, adjacent representable values, the smallest positive binary64 subnormal, and both signs of the largest finite binary64 value.

Weight arrays admit only floating dtypes of 2, 4 or 8 bytes and two dimensions. Every current block is copied and widened into native C-order binary64 before finite-value checking and exact Arb conversion. Noncontiguous, negative-stride and nonnative-endian arrays therefore supply their mathematical stored values rather than their raw byte interpretation. There is no floating dot product: each product is `arb_mat * arb_mat`, and each partial sum and bias addition is an Arb operation. Arithmetic rounding is enclosed by that library. The final finite-value check refuses an unusable enclosure instead of returning a success-like numerical placeholder.

Matrix input balls are retained as balls rather than replaced by midpoints. The list/tuple path requires actual Arb values and wraps them in a one-column matrix. Input shape, nonempty output/input dimensions, finite input balls and finite matching biases are checked. All input columns share exact coefficient blocks but their products remain separate columns; sharing the conversion does not mix primal and tangent values.

## Bias and caller convention

By default a bias is added to every matrix column, as the generic affine API documents. For joint primal/tangent evaluation the caller must explicitly supply the primal columns. The inspected `linear_sequence` does so: column pairs are `(value_j, tangent_j)` for each position, and `bias_columns=range(0,2*count,2)` selects exactly the even primal columns. The tangent columns consequently compute Wv, not Wv+b. Bias is applied once after all input-column blocks have accumulated.

Explicit empty bias columns add no bias. Duplicate, negative, noninteger, Boolean and out-of-range selected column indices are refused. For a vector input only column zero exists. The expected-embedding multiplication uses the transposed embedding view without a bias, which this API supports.

## Cache, precision and memory contract

The optional cache key comprises the copied block's shape, current context precision and complete canonical binary64 bytes. It does not depend on array identity or a cryptographic hash. Mutating a weight between calls cannot return the previous block: the new block is recopied and its full bytes are compared. A precision change selects a different key. The cached objects contain exact coefficient points, not precision-dependent matrix products or input-dependent outputs. Matrix multiplication and addition return fresh results; the code does not write into a cached coefficient matrix.

The LRU evicts retained matrices until their scalar-entry count fits `max_elements`; a single oversized block is used without retention, including when the budget is zero. `clear()` drops retained entries. This is a bound on retained scalar entries, not resident bytes or total temporary memory. Key bytes, Python/FLINT object overhead, the current block/list conversion, input slices and output matrix remain additional allocations, as documented. In particular a large vocabulary output is not bounded by the coefficient-cache budget.

The caller must keep global Arb precision/thread settings stable during a call, use the cache in one context/thread, and prevent concurrent mutation of weights. The module deliberately does not own these global settings. The bounded driver supplies CPU1 and immutable admitted arrays. This review does not treat private attribute mutation or concurrent external changes as a supported API. Cache precision binding is conservative; it does not supply a guarantee for unrelated cached nonlinear prefix values, which have their own map-level guards.

## Independent synthetic checks and retained diagnostic

The existing affine suite contains exact-rational dot-product checks, all corners of a mixed-sign box, extreme exponents/cancellation, bias placement, noncontiguous float32 coefficients, cache mutation/precision behavior, entry limits, and invalid data rejection. The coordinating agent and map reviewer had already reported the final combined 67-test pass; this review did not redundantly rerun that suite.

Additional CPU1, nice-10 checks used no model code:

- Fifteen direct Arb-matrix constructor checks at precision 16, 53 and 256 established exact rational equality for the binary64 constants described above.
- A two-output, three-input map with big-endian binary64 weights in negative-stride order was applied to six columns representing three primal/tangent pairs. Every one of the eight corners of each actual input ball box was evaluated with exact `fmpq` arithmetic for both outputs. Tests at precision 16, 53, 256 and again 16 passed all **384 exact-rational endpoint-membership checks**. Bias was present only in columns 0, 2 and 4.
- A separately retained complete coefficient block produced the expected cache hit when only the input changed, a miss after precision changed, and a miss after changing a backing weight. Context precision/threads and the declared cache-entry bound remained unchanged by `affine`.

Two initial ad hoc harness attempts reported a failed `Arb.contains(fmpq)` assertion at precision 16. Investigation showed that the returned output ball's **exact rational** midpoint-minus-radius and midpoint-plus-radius bounds contained the exact target. The low-precision conversion of that rational argument to an Arb ball can make the containment predicate conservatively false. Raising only the comparison precision made `contains` true for the same saved output ball. The final oracle therefore compared exact `fmpq` endpoints directly, without converting the target rational through a low-precision ball. The kernel, its test data and its numerical outputs were not modified. This was a test-oracle false refusal, not evidence of a missing enclosure; the distinction is recorded rather than silently dropping the failed harness attempts.

## Limits of approval

This component proves neither that a practical ball radius will remain small through 24 decoder blocks nor that a box tangent will preserve enough correlation through eight feedback steps. Arbitrary-precision arithmetic encloses rounding; it does not remove interval dependency. The bounded pilot should retain measured widths and all failures. The later root/attraction checker still needs the exact-map contract, rigorous primal and derivative enclosures, strict full-region inclusion/contraction, proper-divisor separation and a valid monodromy power bound described in `proof-design-review.md`.
