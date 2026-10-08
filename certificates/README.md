# Rigorous orbit certificates

This C1 pilot asks whether full-state neural-network training cycles already
observed numerically exist and attract nearby states in the exact mathematical
map. It does not test a training benefit or shrimp geometry. The pilot is
exploratory. The independent reviews and their different scopes are recorded below.

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
certificate pilot with independent review complete. The checker and upstream Arb library
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

The compact receipts bind inputs, all raw files, source commit and source
hashes. Then one offline command regenerates all enclosures and compares them
with the saved result, refusing missing/corrupt inputs, modified checker code,
changed verdicts or any failed proof gate:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/verify_certificates.py \
  certificates/c1-row35-receipt.json certificates/c1-row121-receipt.json \
  certificates/s1r-period2-receipt.json
```

The receipt's external root locates raw artifacts on zeta. A Git push backs up
source and the receipt; it does not independently back up the external raw files.

| Component | Evidence | Artifact / assumption |
|---|---|---|
| Map/Jacobian implementation | Test cross-check, not a theorem by itself | `tests/test_rigorous_orbits.py`; independent frozen NumPy implementation and enclosed interior samples |
| Existence/local uniqueness | Rung 3 gates pass; exploratory, independently reviewed | All three current receipts; Banach contraction, analytic map implementation and Arb/FLINT trusted |
| Primitive period | Rung 3 gates pass; exploratory, independently reviewed | S2 receipts exclude divisors 1, 2 and 5; S1R excludes divisor 1, by disjoint coordinates |
| Attraction | Rung 3 gates pass; exploratory, independently reviewed | S2 monodromy powers 64 and S1R power 2048 have infinity norms below one |
| Chaos, window geometry, invariant circles, utility | Not established by this track | Existing evidence retains its own scope and labels |

## Pilot outcome (7 October 2026, user time zone)

**Behavioral TLDR:** for two selected learning-rate settings of a 13-parameter tanh network
(the same S2 dataset and architecture), the exact-real training map
has an attracting full-state period-10 orbit. Thus small perturbations near
those mathematical cycles decay while the full parameter state keeps cycling.
One orbit is in a historically qualifying EXP-028 window. This result does
not yet establish a shrimp-specific behavioral effect, a useful readout,
surrounding chaos rigorously, or robustness to finite-precision arithmetic.
The computation passes proof gates and independent mathematical/code review; target selection
remains exploratory. The review record is `c1-independent-review.json`.

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
the offline command above checks the three authoritative current certificates.
All raw attempts are retained under the external root in each receipt, and the
code prevents reusing an output directory. The tests include independent
NumPy map/Jacobian comparisons, interior enclosure samples, exact linear
positive controls, false primitive-period and unstable controls, and corrupt
input/source-inventory/summary rejection. None replaces independent review.

## Independent review (8 October 2026 UTC)

The source-bound review in `c1-independent-review.json` reports no blocking
findings for producer tip `32528cbbb4a3313bf82f63a9a812584ad754a82e`. Both S2
authoritative receipts replayed; input bindings, stored Krawczyk/divisor/power
witnesses and an independent second-order Arb differentiation of the objective
were checked. Both proofs also passed at 384 bits in smaller nested boxes.
This shares the Arb library and is not proof-assistant formalization. Immutable
receipts retain their original creation-time review-pending labels; this
companion record supplies the subsequent review status.


## Full 72D S1R certificate (8 October 2026 UTC)

`s1r-period2-receipt.json` proves local existence, uniqueness, primitive period
two and attraction in the 72-parameter S1R-28002 map in a certified box near selected saved node 0004.
The exact learning rates are `1870594557782953/140737488355328` (hidden) and
`2385682499131913/562949953421312` (output); the input file binds all dataset
values, scales and the exact dyadic interpretation of ridge 0.001.
The fixed settings were 256-bit precision, radius `2^-80`, five proposal Newton
steps and maximum monodromy power 4096. The successful witness uses power 2048
with an exact upper bound below 0.556. The unchanged C1 kernel took 3.30s on one
CPU thread; no paid compute or GPU was used. Selection remains exploratory.

The companion `../docs/reviews/s1r72-certificate-outcome-receipt.json` binds the
original selected arrays and exact-real inputs, checks the source/resource
records, replays the proof once and independently computes an exact-Fraction
whole-box proximity bound. The declared normalized threshold 1e-7 passes. This
does not assert that the original floating-point state lies inside the proof
box, prove historical branch ancestry, enclose a parameter interval, or prove
the bifurcation/window geometry. Proximity is separate from proof validity.

This review reuses the already reviewed Arb checker and theorem arguments.
The separate second-order Arb differentiation and 384-bit nested-box review
described above apply to the two S2 proofs, not this new outcome audit.
Reproduce the outcome auditor from its recorded historical checkout: later
observer additions to `gdmaps.py` changed auxiliary source hashes without
changing the three C1 proof-kernel files. The ordinary offline proof command
above still uses those unchanged kernel files. Original receipts retain their
creation-time pending labels; the companion review supplies subsequent status.
