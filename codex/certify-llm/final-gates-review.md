# Independent final conditional gate review

Reviewed 13 October 2026 UTC; source inspection and CPU1 synthetic checks only.
No model, target candidate, holdout, or derivative shard was evaluated.

| File | SHA-256 |
| --- | --- |
| `shooting_gates.py` | `9c967c0dcb9783df53f33c05eb897050f4e8971f66767abbc8a01a8b1b85cf1f` |
| `shooting_certificate.py` | `a4aca57acf7aaab33c77745ea6170935dd6b4dac40c9c982a063743f98727c3e` |
| `proof_gates.py` | `ee6678b7dd5153ef095fd73d5711fcfc605f7a228c92b1335a9541cc3d7b2c6a` |
| `test_codex_llm_shooting_gates.py` | `c3eb419697a66aae8e910cde7d84f1ec0e0675919bd46d60cf5476193052f410` |
| `test_codex_llm_shooting_certificate.py` | `70f1f7e544e33b578df41f249023c5513faae9be6b1b064eae24390595e26372` |
| `test_codex_llm_proof_gates.py` | `0ee68e10074354173f188980b3b7a78e1749fba294b44104b3b913f29271af77` |

## Mathematical verdict

No algebraic or implication blocker was found. The result remains conditional
on genuine full-box enclosures and their external authentication. The wrapper
correctly never emits `certificate_passed=true`.

For the cyclic equations `H_i(x)=F(x_i)-x_(i+1)`, exact dyadic A_i and B define
one fixed linear preconditioner R. Neither needs to be an exact Jacobian or
inverse. Put `P_i=A_(i-1)...A_0`, `Q_j=A_(p-1)...A_(j+1)`, and
`D=I-B(I-P_p)`. Direct block multiplication gives

```
R_ij = -P_i B Q_j - 1[j<i] A_(i-1)...A_(j+1)
(I-R DH)_ij = 1[j=0] P_i D - R_ij (J_j-A_j).
```

The reverse suffix and forward block recurrences at shooting_gates lines
158–190 have these signs and orders. Products are rigorous Arb enclosures of
products of the exact original proposals; rounded product midpoints are not
silently reused as exact matrices. The factored alternative, lines 194–239,
uses signed P_i, Q_j and P_i D before taking norms, then a conservative triangle
and submultiplicative bound. Losing cancellation can cause rejection, not an
invalid success.

The residual recurrence at lines 242–256 encloses `-R H(c)`. In the weighted
product norm `max_i ||v_i||_inf/r_i`, both modes bound
`q=max_i sum_j ||K_ij||_inf r_j/r_i`. The gate requires, separately in every
phase, `q_i<1` and `||(-RH(c))_i||_inf + sum_j ||K_ij||_inf r_j < r_i`.
All comparisons use outward endpoints and the exact nominal radii; equality
fails. These imply that `x-RH(x)` contracts the full closed product box strictly
into itself. The product box is nonempty and complete. At an actual point,
`||I-R DH||<1` also makes the square product R DH nonsingular, hence R is
nonsingular. The unique fixed point is therefore a zero of H, not merely a
zero of RH.

This proves exactly one ordered shooting-root tuple in the full product box.
It does not exclude another F^p fixed point in X_0 whose other phases leave the
prescribed boxes. Strict interior inclusion and continuity do imply local
uniqueness of the proved F^p fixed point. The wrapper's local-uniqueness wording
and the root gate's explicit full-product qualification are consistent.

`primitive_phase_gate` uses every proper divisor. At p=8 these are 1, 2 and 4.
A proved shooting root has x_d=F^d(x_0) in X_d, so strict coordinate separation
of X_d from X_0 excludes that divisor. The wrapper cross-binds the same exact
domain and requires root existence in the final conjunction. It never treats
X_d as an enclosure of F^d(X_0). Touching boxes fail the strict test.

Attraction uses the actual derivative enclosures J_7...J_0 in that order, not
the numerical proposals A_i. The chain rule at the root establishes membership
of its monodromy in this product despite lack of whole-box forward invariance.
Interval powers retain all entries and signed multiplication. If one declared
power has induced positive-weight infinity norm strictly below one, the root
monodromy has spectral radius below one. C1 regularity then yields local
asymptotic attraction of the primitive orbit. This does not show that the
entire product box is a dynamical basin. Root-finding contraction by itself
would not imply dynamical attraction; the wrapper keeps these tests separate.

## External obligations still required for a target certificate

The hashes and declarations bind evidence but do not establish its truth.
A final caller must authenticate exact candidate centers/radii, exact-center
images F(c_i), all full-box J_i, exact map/assets and backend, and the independently
reviewed evaluator. There must be no missing transverse directions. The new
coverage/evidence path can supply all 8×896 derivative columns; it does not
supply the point-image authentication or itself establish a root. Images must
belong to the identical exact centers and map, retain their original radii,
and be cross-bound to the assembled matrices in the final contract.

Different point and derivative arithmetic precisions are legitimate if both
rigorously enclose the same exact-real map. A separate nominal-domain identity
must be preserved across their distinct hash conventions. C1 regularity and
positive normalization/temperature denominators follow from reviewed map
formulas and constants, not a producer Boolean. No prototype box, float
Jacobian, numerical spectrum or center-selection history can replace a missing
full-box enclosure. Failed sufficient gates remain inconclusive.

## Static resource analysis at p=8, n=896

An n-square matrix has 802,816 entries; the eight Jacobians have 6,422,528.
No reviewed shooting path allocates a 7,168-square matrix. The current factored
path does retain nine prefix matrices in addition to eight J_i, eight A_i,
B and several matrix temporaries. O(p n²) is not constant memory.

Counting the current native dense matrix-multiplication calls, including calls
with identity matrices:

| Stage | Calls |
| --- | ---: |
| Factored root: p prefixes, one D, p products P_i D, p−1 suffix updates | 24 |
| Streamed root: complete signed defect blocks and recurrences | 151 |
| Ordered monodromy construction | 8 |
| Declared attraction powers (1,2), recomputed by current helper | 3 |
| Total, factored root and powers (1,2) | 35 |
| Total, streamed root and powers (1,2) | 162 |

The root residual also uses 16 dense matrix-vector products. The factored
root performs 41 full row-norm scans, about 32.9 million entry visits, before
contract validation, hashing and wrapper scans. A classical n-cubed count for
35 products is about 25.18 billion scalar product terms, but FLINT can use
other algorithms; this is not a time estimate. Using the standalone attraction
default powers (1,2,4,8,16,32,64) would perform 28 power multiplications instead
of three, because each power starts over.

For nonzero full-mantissa 1024-bit entries, midpoint bit payload alone is
0.0957 GiB per matrix, excluding native structures, radii, allocation overhead
and multiplication workspace. Twenty-nine simultaneously live input/prefix/
temporary matrices would have about 2.78 GiB of such midpoint payload; actual
live counts and entries vary. Binary64 proposal matrices can use less, whereas
2048-bit matrices can use roughly twice the full-mantissa payload. None of
these bit counts is a measured RSS bound.

A separate implementation cost is `proof_gates.attraction_gate` lines 267–271:
it retains complete nested Python matrix records for the monodromy and every
requested power. With powers (1,2), this is 2,408,448 ball dictionaries, each
with two lists and exact integers, besides native matrices. The seven-power
default stores 6,422,528 such records. JSON encoding may add another large
allocation unless the outer writer streams it. This can consume gigabytes and
substantial Python time. Existing compact shooting hashes do not eliminate
this attraction serialization cost.

The next safe gate-resource step is a separately bounded actual-size synthetic
896-square Arb multiplication/norm/hash/serialization pilot at the proposed
precision, including narrow interval operands. No model is needed for that
measurement. A separately reviewed compact attraction variant could stream
matrix artifacts and retain hashes/row bounds instead of millions of in-memory
records, while preserving all declared powers and failed attempts. That is an
engineering option, not an implemented or benchmarked optimization. The current
factored method has a credible bounded-memory structure, but neither a one-hour
runtime nor an 18 GiB peak has been established at target dimension.

## Validation

All 89 focused shooting/wrapper/proof-gate tests passed (0.35 seconds, CPU1).
The broader gate/evidence/shard/coverage/domain suite passed 287 tests. Tests
cover strict equality failures, uncertain images and derivatives, nonnormal
power attraction, unstable roots, proper-divisor failures, domain/input
mutations and preservation of interval widths.

An additional independent exact-Fraction construction formed the full R and
DH directly from explicit block products for p=2,3,8 and n=2, using signed
noncommuting dyadic proposals. All 308 entries of I−R DH exactly matched the
streamed blocks; all 77 exact block norms were bounded by the factored outputs.
This check used no floating eigensolver and no target artifacts. Script retained
at `/tmp/codex-final-gates-independent.py`; these tiny checks verify algebra,
not target-scale performance or the transformer's enclosure contract.
