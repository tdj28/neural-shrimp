# EXP-006 continuation audit

Source freeze: `ddb568e6cd67ef2cfbdb45e05710cd86025ef548`.

Completed branch artifacts audited: 4/4. No new trajectory integration was performed.

Independent reconstruction uses the saved monodromy matrices, states and phase normals. A cross-product orthonormal section basis reconstructs the two-dimensional Poincare map; this differs from the implementation’s SVD basis. Neural vector fields are reevaluated algebraically from the declared weight matrix. Neutral-flow and phase residuals, multipliers, stability flags, crossings, parameter reversals, checkpoint agreement, and computational limits are checked.

Integrated endpoints are not saved in compact branch nodes. Consequently the shooting closure and primitive-event checks are audited against recorded values/gates, not independently reconstructed by integration. This distinction is intentional; the task forbids new integration.

## core-direction-+1.json

- SHA-256: `cfc68a893041ad502fcf7915cad8fb05f54ea6524b39e0e9bc4e0410f2e52faf`
- Accepted nodes including seed: 49; failed correctors: 0/48; stop: `maximum_accepted`.
- Unstable nodes: 31; indices: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48].
- Gamma range: [10.260722222222222, 10.45241426247645]; transverse spectral-radius range: [4.490535089020165e-06, 43.0661555323598].
- Independent multiplier discrepancy (maximum absolute): 1.42109e-14.
- Maximum reconstructed neutral-flow residual: 4.39625e-08; phase residual: 4.35714e-17; recorded relative closure: 9.11673e-11.
- Independent shooting checkpoints agreeing with continuation: 7/7.
- Parameter reversals: `[]`.
- Crossing brackets: `[{"nodes": [17, 18], "gamma": [10.325552924595891, 10.329636479460625], "period": [37.29230441156904, 37.26530129207536], "flags": ["plus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[4.807529181066716e-11, 0.0], [1.6205378262921803e-11, 0.0]], [[0.8996079347149627, 0.0], [1.2683076325391536, 0.0]]]}]`.
- Failed-corrector reasons: `{}`.
- Checkpoint disagreements: `[]`.

## core-direction--1.json

- SHA-256: `c7542cf0bca91a79e32ab783a31193d03668da480ab84d813edc3a2111fc2e7e`
- Accepted nodes including seed: 49; failed correctors: 2/50; stop: `maximum_accepted`.
- Unstable nodes: 31; indices: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48].
- Gamma range: [10.202365150998146, 10.319630969240295]; transverse spectral-radius range: [0.0008232286309120209, 155.15521894577344].
- Independent multiplier discrepancy (maximum absolute): 7.10543e-14.
- Maximum reconstructed neutral-flow residual: 6.68916e-08; phase residual: 2.05874e-15; recorded relative closure: 6.00817e-10.
- Independent shooting checkpoints agreeing with continuation: 7/7.
- Parameter reversals: `[{"node": 18, "gamma": [10.202942911984804, 10.202365150998146, 10.202839158199158]}]`.
- Crossing brackets: `[{"nodes": [17, 18], "gamma": [10.202365150998146, 10.202839158199158], "period": [37.45823322021997, 37.428570168200935], "flags": ["plus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[-2.957498434010921e-11, 0.0], [-1.920127272404297e-11, 0.0]], [[0.8074078047298268, 0.0], [1.985743862852453, 0.0]]]}]`.
- Failed-corrector reasons: `{"bounded line search failed": 1, "maximum Newton iterations reached": 1}`.
- Checkpoint disagreements: `[]`.

## failed-near-edge-direction-+1.json

- SHA-256: `2a2e5db6d078550e721e770890b934787dbbe87ffdd2751b3595ef6c5f611f4f`
- Accepted nodes including seed: 49; failed correctors: 2/50; stop: `maximum_accepted`.
- Unstable nodes: 41; indices: [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48].
- Gamma range: [10.290693393445766, 10.415734040365132]; transverse spectral-radius range: [0.34397295046380816, 93.87912534101876].
- Independent multiplier discrepancy (maximum absolute): 5.68434e-14.
- Maximum reconstructed neutral-flow residual: 2.74064e-08; phase residual: 4.09069e-16; recorded relative closure: 1.26627e-09.
- Independent shooting checkpoints agreeing with continuation: 6/7.
- Parameter reversals: `[{"node": 6, "gamma": [10.305678692296274, 10.306167392268229, 10.30573363260006]}, {"node": 16, "gamma": [10.290778804511234, 10.290693393445766, 10.291835686320047]}]`.
- Crossing brackets: `[{"nodes": [5, 6], "gamma": [10.306167392268229, 10.30573363260006], "period": [37.3384370054528, 37.33868850308022], "flags": ["plus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[5.4406293869055815e-12, 0.0], [4.2856619884070206e-12, 0.0]], [[0.9744083580407972, 0.0], [1.2618526761798394, 0.0]]]}, {"nodes": [14, 15], "gamma": [10.290778804511234, 10.290693393445766], "period": [37.289532576165605, 37.27098064423259], "flags": ["plus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[-2.3102828414673292e-12, 0.0], [1.0223699881203185e-11, 0.0]], [[1.361735451462366, 0.0], [0.6738185771796842, 0.0]]]}, {"nodes": [16, 17], "gamma": [10.291835686320047, 10.294041403715855], "period": [37.249200165178436, 37.22650366682532], "flags": ["minus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[-0.34397295046380816, 0.0], [-1.6496924111992861, 0.0]], [[-1.2584582876858345e-11, 0.0], [-2.0020126457866277e-12, 0.0]]]}]`.
- Failed-corrector reasons: `{"bounded line search failed": 2}`.
- Checkpoint disagreements: `[{"node": 40, "reason": "closure, transverse_stability", "verified": false, "period_agreement": true, "state_agreement": true}]`.

## failed-near-edge-direction--1.json

- SHA-256: `8bddb349169daf552cb64c61fe79e88cd9c9dc96c96bd90edc760cda4ff56c0e`
- Accepted nodes including seed: 49; failed correctors: 1/49; stop: `maximum_accepted`.
- Unstable nodes: 19; indices: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48].
- Gamma range: [10.212013126546088, 10.297166666666667]; transverse spectral-radius range: [0.0013453492123767117, 64.27014869372525].
- Independent multiplier discrepancy (maximum absolute): 1.98869e-14.
- Maximum reconstructed neutral-flow residual: 2.74064e-08; phase residual: 3.85976e-16; recorded relative closure: 1.47227e-09.
- Independent shooting checkpoints agreeing with continuation: 7/7.
- Parameter reversals: `[{"node": 31, "gamma": [10.21239159069086, 10.212013126546088, 10.21298937458674]}]`.
- Crossing brackets: `[{"nodes": [29, 30], "gamma": [10.21239159069086, 10.212013126546088], "period": [37.41831320347389, 37.39852759619641], "flags": ["plus_one_bracket", "unit_circle_bracket"], "multiplier_pairs": [[[9.620734477977391e-12, 0.0], [-1.3653009668368415e-11, 0.0]], [[0.4886399722702959, 0.0], [1.1642823435473943, 0.0]]]}]`.
- Failed-corrector reasons: `{"maximum Newton iterations reached": 1}`.
- Checkpoint disagreements: `[]`.

## Independent checkpoint disagreement

Near-edge initial +gamma branch, node 40: gamma=10.383151826863658. The PA record has relative closure 1.3771356909740124e-12, whereas the independent shooting checkpoint reports 1.0590609546229447e-08, exceeding the frozen 1e-8 closure threshold. It remains a failed checkpoint. State and period still agree at their declared thresholds, and transverse radii agree at approximately 62.6347466. No threshold was relaxed.

## Interpretation

All four branches reached their 48-successor-node cap: 196 total node records including the repeated seeds, 122 unstable node records, 5 failed correctors among 197 attempts, and 27/28 agreeing independent shooting checkpoints. These counts include overlapping continuations and are not counts of distinct cycles.

Unstable periodic orbits are retained wherever reported above. A multiplier crossing bracket and a parameter reversal are numerical evidence about the continued branch, not a rigorous bifurcation proof. These local gamma sections are not a complete two-dimensional shrimp boundary. Unsuccessful corrections are unresolved numerical steps, not evidence that an orbit or basin ceased to exist.

The gamma pairs listed above are exact coordinates of adjacent accepted nodes that bracket a multiplier crossing along continuation arclength. At a turning point, they need not bound the true crossing gamma: the branch can turn between those nodes. Do not present these pairs as rigorous parameter-location intervals or exact boundary estimates.
