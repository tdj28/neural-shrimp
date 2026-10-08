# Rigorous orbit certificates

This C1 pilot asks whether full-state neural-network training cycles already
observed numerically exist and attract nearby states in the exact mathematical
map. It does not test a training benefit or shrimp geometry. The pilot is
exploratory. Independent review of producer commit
32528cbbb4a3313bf82f63a9a812584ad754a82e found no blocking issue for the
narrow exact-map claims; see [the review](c1-independent-review.json).

The checker uses Arb real balls through python-flint. Every stored binary64
constant (dataset, scale, ridge and learning rate) is interpreted as its exact
dyadic rational. The loss is `loss_scale * mean((prediction-y)^2) +
ridge/2 * sum(theta^2)` and the real map is `theta - P gradient(loss)`, with
`tanh` in real arithmetic. This is not a theorem about finite-precision training.
The input exporter checks regenerated dataset hashes and network values against
the unchanged EXP-028 plan, then stores the dataset itself for offline checking.

For `H=F^p-I`, the checker refines a dyadic proposal, encloses the map and exact
analytic Jacobian on a box `X`, and takes an exact dyadic approximate inverse
`R`. It requires both strict Krawczyk inclusion
`x-R H(x)+(I-R H'(X))(X-x) < interior(X)` and infinity norm
`||I-R H'(X)|| < 1`. These establish that `T(z)=z-R H(z)` maps `X` strictly
inside itself and is a contraction. The derivative bound also makes `R`
nonsingular, so the unique fixed point is a zero of `H`. Proper divisors of
`p` are excluded by a coordinate separating `F^d(X)` from `X`. Attraction
requires a rigorous bound `||M^k||inf < 1` for an interval enclosure of the
monodromy at the proved periodic point. Thus every actual monodromy in the
enclosure has spectral radius below one. Float eigenvalues never decide a gate.

Direct interval iteration can fail through wrapping. A failed enclosure is
inconclusive and is retained. Multiple shooting is a possible later amendment,
not part of this initial checker. Passing artifacts carry evidence rung 3 for
the narrow mathematical statement; the research status remains an exploratory
certificate pilot. Immutable envelopes retain their historical pending-review
labels; the separate source-bound record documents the completed review. The checker and upstream Arb library
are not formalized in a proof assistant.

## Execution and replay

Use the project environment (python-flint 0.9.0 / FLINT 3.6.0 at development).
Code must be committed and a research-log run card recorded before target runs.
`tools/certify_c1.py prepare` exports inputs from an existing EXP-028 numerical
receipt; `run` writes an exclusive new attempt under the external data store.
The first selected candidate was row 121, S2-28011-zoom component 936, period
10: the best-conditioned embedded period-5/10 candidate by existing numerical
`condition(M-I)`. Scope review found it was absent from the final historical
window qualifying lists. Its orbit proof is retained with no surrounding-chaos
claim. The amended target was row 35, S2-28011 component 140, period 10: the
best-conditioned period-5/10 candidate in the intersection of the unchanged
EXP-028 summary and independent audit qualifying lists. Both selections and
the amendment are disclosed in the research log; neither is confirmatory.

A compact receipt will bind inputs, all raw files, source commit and source
hashes. Then one offline command regenerates all enclosures and compares them
with the saved result, refusing missing/corrupt inputs, modified checker code,
changed verdicts or any failed proof gate:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/verify_certificates.py \
  certificates/c1-row35-receipt.json certificates/c1-row121-receipt.json
```

The receipt's external root locates raw artifacts on zeta. A Git push backs up
source and the receipt; it does not independently back up the external raw files.

| Component | Evidence | Artifact / assumption |
|---|---|---|
| Map/Jacobian implementation | Test cross-check, not a theorem by itself | `tests/test_rigorous_orbits.py`; independent frozen NumPy implementation and enclosed interior samples |
| Existence/local uniqueness | Rung 3 gates pass; exploratory, independently reviewed | Both current receipts; Banach contraction, analytic map implementation and Arb/FLINT trusted |
| Primitive period | Rung 3 gates pass; exploratory, independently reviewed | Both receipts exclude divisors 1, 2 and 5 by disjoint coordinates |
| Attraction | Rung 3 gates pass; exploratory, independently reviewed | Both receipts enclose the monodromy and its 64th power below unit infinity norm |
| Chaos, window geometry, invariant circles, utility | Not established by this track | Existing evidence retains its own scope and labels |

## Pilot outcome (7 October 2026, user time zone)

**Behavioral TLDR:** for two selected learning-rate settings of a 13-parameter tanh network
(the same S2 dataset and architecture), the exact-real training map
has an attracting full-state period-10 orbit. Thus small perturbations near
those mathematical cycles decay while the full parameter state keeps cycling.
One orbit is in a historically qualifying EXP-028 window. This result does
not yet establish a shrimp-specific behavioral effect, a useful readout,
surrounding chaos rigorously, or robustness to finite-precision arithmetic.
The computation passes proof gates and independent mathematical/code review
found no blocking issue within this scope. This remains an exploratory pilot.

| Attempt | Target and purpose | Outcome | Compact receipt |
|---|---|---|---|
| 001 | Row 121, sequential matrix powers | Inclusion and primitive-period gates pass; attraction inconclusive through wrapping | `history/c1-attempt-001-receipt.json` |
| 002 | Row 121, disclosed repeated-squaring amendment | All gates pass; superseded only for checker source binding | `history/c1-attempt-002-receipt.json` |
| 003 | Row 35, qualifying-window target | All gates pass; authoritative for this orbit | `c1-row35-receipt.json` |
| 004 | Row 121 replay after receipt hardening | All gates pass; authoritative for this orbit; not an additional orbit | `c1-row121-receipt.json` |

At 256-bit midpoint precision with box coordinate radius `2^-80`, the
Krawczyk contraction bounds are below `9e-13` (row 35) and `8.1e-14`
(row 121). The rigorous upper bounds for `||M^64||inf` are below `0.056`
and `0.108`, respectively. These are upward-rounded summaries of the exact
dyadic bounds in the receipts, regenerated by the offline command above.
Attempt 004 exactly reproduces attempt 002's mathematical result, including
every enclosure; only provenance/runtime changes. Each actual pilot used one
CPU thread and less than one second; no paid compute was used.

The initial failed attraction enclosure is an implementation limitation, not
evidence of instability. Its raw output and exact original code commit remain
available. Historical receipts must be replayed with their bound source commit;
the offline command above checks the two authoritative current certificates.
All raw attempts are retained under the external root in each receipt, and the
code prevents reusing an output directory. The tests include independent
NumPy map/Jacobian comparisons, interior enclosure samples, exact linear
positive controls, false primitive-period and unstable controls, and corrupt
input/source-inventory/summary rejection. These tests supplement the independent review recorded below.

## Independent review (7 October 2026, user time zone)

[The review record](c1-independent-review.json) binds both authoritative
certificate hashes, producer tip and source hashes, and reviewer source.
Independent reasoning checked the Krawczyk/Banach argument on the closed
convex initial box, nonsingularity of the exact midpoint preconditioner,
whole-box chain-rule enclosure, proper-divisor exclusions, and attraction
from the rigorous 64th-power norm. No blocking finding was identified.

Both authoritative replays and all eight producer tests passed. Additional
review recomputed stored witnesses at 384 bits, independently differentiated
the scalar objective using second-order Arb jets at each center and an
interior point, and checked the map/Jacobian enclosures against those values.
Exact dyadic inputs match the saved EXP-028 orbit, dataset and unchanged
plan. Both proofs also pass at 384 bits with radius 2^-90 boxes strictly
inside the original boxes.

This uses the same Arb/FLINT library, which remains trusted, and is not
proof-assistant formalization. Sampled AD comparisons supplement mathematical
code inspection; they do not alone prove implementation equivalence. The
original raw receipts and exclusions for geometry, chaos, finite-precision
dynamics and utility remain unchanged. The additional reproducible checker
is tools/review_c1_evidence.py at the reviewer commit in the JSON record.
