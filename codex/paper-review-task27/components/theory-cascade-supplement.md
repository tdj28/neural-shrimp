# Task 27 compact 1.5B cascade supplement

Same frozen paper `66ffa102c56f6d5b22f116a1bb194f503c75562e`, main.tex SHA `d78a22090469e1dc32e67e715c0a2f51cc538b6728496c04974d9fc16a99b05f`. This supplements, rather than replaces, scientific note `cb9280cd8b4bc0cbdb6db262af14c842585c5741244f063adf35f7e38ddb44b9`. Inference's scale lane explicitly leaves the original scalar comparison at lines 1498–1510 to this lane.

The seven saved 1.5B flips supporting those lines have been read individually. The two compact `edges.json` files classify their edges as flips and report detected periods 16 and 32. The remaining rows come from three separately retained cascade attempts. All selected critical values are real with argument one half-turn; the values below are literal saved fields, not a fresh eigensolve or ratio recomputation.

| Parent period | log10(T) | Critical multiplier | Next modulus | Final Newton residual |
|---|---|---|---|---|
| 4 | −0.003126922235424109 | −1.0000000000024956 | 0.014057295682365469 | 5.746899120946596e−13 |
| 8 | −0.0021666499294120977 | −1.000000000003337 | 7.736264363863835e−5 | 1.0074694328369522e−13 |
| 16 | −0.00190024522705658 | −0.9999999999971524 | 4.625916868485172e−9 | 6.360853780209465e−15 |
| 32 | −0.0018380241237525106 | −1.0000000001917813 | 6.612920030539866e−15 | 2.6996280021842046e−13 |
| 64 | −0.001824434015999929 | −0.9999999998888498 | 2.3066237402704826e−14 | 8.419745514260444e−13 |
| 128 | −0.0018215102546514878 | −1.000000019765423 | 2.912795983136911e−15 | 9.61491897718704e−13 |
| 256 | −0.0018208834782723728 | −0.9999999893523165 | 2.6686305466708268e−15 | 5.403870333509553e−13 |

The saved parameters and next moduli match the rounded paper values. The five displayed ratio values are producer combinations of this stitched sequence; the original files themselves contain no complete five-ratio list. This lane checked source association and selected-row gates, not new independent cross-file arithmetic. The frozen log explicitly preserves FB15D's refusal when its proposed period-128 switching point was already beyond that flip; FB15E used a disclosed nearer bracket and located the next two flips. The earlier failed/partial attempts must remain distinct from the completed rows.

**Should; lines 1506–1507.** Replace “the floor set by float64 parameters” by “a location uncertainty estimated from the observed test-function scatter and its local slope in the float64 computation.” The cited log estimates about 3×10⁻¹⁴ from scatter about 10⁻⁷ divided by slope about 3.4×10⁶. This is a conditional numerical estimate; it is neither a rigorous enclosure nor simply the representable spacing of the log-temperature parameter. The claim that the finite ratios approach 4.669 is supported as a numerical trend, not an infinite-cascade proof.

The original records are pinned in compact manifest `96c710c0`: q15-a0.100-p4/cascade.json `f34ae48b56fc86ebb9715b474556419e87f9b5440ec1a5deabe8b13a860769f0`; q15-a0.100-p16/edges.json `21835010d1a556563b3007d095838ce0c1231d93ed99fdb5541be678492382df`; q15-a0.100-p32/edges.json `09c41b7d27d3870c1422290ba845f8df523fdca7c8752291470e0d277ccbddf8`; q15-a0.100-p64/cascade.json `a188f9524ecbfb28142ce11c6fa4c5a1237f827765e01d81e5dccd8cb97d4ff3`; q15-a0.100-p128/cascade.json `ea7720e728617bd59ca407efcade66c78ef5c360b1c8049e17afd6488bcef920`. Exact original paths and captured-byte associations remain in that manifest. Historical process/source attestation is the producer log's stated basis, not retrospectively recreated by this review.
