# Rigorous orbit certificates

This C1 pilot asks whether full-state neural-network training cycles already
observed numerically exist and attract nearby states in the exact mathematical
map. It does not test a training benefit or shrimp geometry. The pilot is
exploratory and has not yet received independent mathematical/code review.

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
certificate pilot with review pending. The checker and upstream Arb library
are not formalized in a proof assistant.

## Execution and replay

Use the project environment (python-flint 0.9.0 / FLINT 3.6.0 at development).
Code must be committed and a research-log run card recorded before target runs.
`tools/certify_c1.py prepare` exports inputs from an existing EXP-028 numerical
receipt; `run` writes an exclusive new attempt under the external data store.
The first selected candidate is row 121, S2-28011-zoom component 936, period 10:
the best-conditioned embedded period-5/10 candidate by existing numerical
`condition(M-I)`. This outcome-informed selection is disclosed and exploratory.

A compact receipt will bind inputs, all raw files, source commit and source
hashes. Then one offline command regenerates all enclosures and compares them
with the saved result, refusing missing/corrupt inputs, modified checker code,
changed verdicts or any failed proof gate:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/verify_certificates.py \
  certificates/c1-receipt.json
```

The receipt's external root locates raw artifacts on zeta. A Git push backs up
source and the receipt; it does not independently back up the external raw files.

| Component | Evidence | Artifact / assumption |
|---|---|---|
| Map/Jacobian implementation | Test cross-check, not a theorem by itself | `tests/test_rigorous_orbits.py`; independent frozen NumPy implementation and enclosed interior samples |
| Existence/local uniqueness | Pending actual pilot | Banach contraction and interval enclosure correctness; Arb/FLINT trusted |
| Primitive period | Pending actual pilot | All proper divisors must have a disjoint coordinate |
| Attraction | Pending actual pilot | Interval matrix-power norm below one |
| Chaos, window geometry, invariant circles, utility | Not established by this track | Existing evidence retains its own scope and labels |
