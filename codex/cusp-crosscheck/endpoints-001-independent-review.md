Task14(a) final saved-evidence review — 2026-10-13

Accepted for the declared finite numerical scope. Both original period-7 endpoints, rows 13 and 14, pass the frozen orbit, primitive-period, near-real-+1, hyperbolic-complement, dense/composed directional, homological, and coefficient-precision checks. The opposite quadratic signs and nonzero negative cubic values support a candidate segment for further cusp localization. They do not establish a located cusp, an exact zero of the quadratic coefficient, a connected certified fold segment, or a rank-two unfolding. Both `cusp_qualified` and `certificate_passed` remain false.

This is a retrospective review of the sole completed saved-array audit. I read both complete row records, all 14 coefficient FD rows, all 14 independent-probe FD rows, all six physical-alpha rows, all 26 precision gates, all 226 nonzero scalar differences, and the source/admission/closure authorities. I did not run another audit, tests, model, JVP/AD evaluation, eigensolve, inverse, or coefficient calculation. The producer assignment and Root's endpoint summaries were already exposed; this is not a blinded confirmation. Inference observed and reaped the original executions, then relayed their closure to Root.

The audited map uses one full period of seven iterates, all 896 state coordinates, float64 arithmetic, physical alpha (including its effect on the steered prefix), and theta=(log10(T),10 alpha). Its mode convention is unit Euclidean right vector with largest-absolute coordinate positive and left-right pairing one. Coefficients use x(u)=x*+r u+h2 u²/2+… . The following values are the independent saved reductions in the completed report, rounded only for this table.

| Quantity | Row 13 | Row 14 |
|---|---:|---:|
| Physical-alpha coefficient a | -0.5685198728572858 | -0.5272297971182208 |
| Quadratic b | -30.840002884452204 | 59.06020295925012 |
| Orientation-invariant b sign(a) | 30.840002884452204 | -59.06020295925012 |
| Cubic c | -580377.7865190986 | -531766.3315274358 |
| Direct cubic term | -599297.6787201943 | -548960.3100806961 |
| Center-manifold term | 18919.89220109566 | 17193.97855326033 |
| Cancellation ratio | 1.0651985401252881 | 1.0646674207593123 |
| Full-period relative closure | 1.7533217304972459e-12 | 4.301143426536771e-12 |
| Period-1 separation | 0.06870231252477585 | 0.06872020951697949 |
| Critical multiplier minus one | 6.447980815948995e-9 | 1.6223444809781995e-9 |
| Complement maximum modulus | 0.00016037005396244848 | 0.0001632040623902414 |
| Nearest complementary unit-circle distance | 0.9998396299460376 | 0.9998367959376098 |
| Relative cubic change under near-mu solve | 4.203405871349972e-10 | 1.0489338706091513e-10 |

The critical multipliers are numerically close to, but not exactly, one. The complement is numerically attracting; this does not establish full-orbit attraction or exact fold existence. The retained left/right residuals are below 1.7e-15. Both bordered and unbordered homological residuals are below 1.2e-15 after the report's normalization. The fold solve and the separate mu²I−M sensitivity solve retain their border, orthogonality, h2 norm, and coefficient changes; neither replaces the other.

Every fixed relative state step [.01,.003,.001,.0003,.0001,.00003,.00001] remains in the report. Large coarse errors are real and are not suppressed: row 13's first mixed-derivative relative error is about 1.640; row 14's is about 1.767. Row 14's coarsest b estimate is -90.82518561477332, opposite to its positive AD value. Its next estimate is 1.4112750407299581. Thus coarse agreement must not be claimed. The original gate uses the fixed finest two rows plus the stated convergence requirement, not a selected plateau. Across the nine coefficient/vector FD error columns, the largest finest-two errors are the b projection errors 1.9179520713746257e-4 and 1.1936953952569295e-4. Both are below the frozen 1e-3 gate with the required reduction. The finest-two total-c errors are at most 8.566204711250092e-6 and 6.644606585612353e-6. All direct, center, total, Brr, Crrr, Brh2, near-mu mixed, and critical-map columns were inspected, not just c.

All seven separate probe rows pass their frozen ladder rule. All three physical-alpha steps [7e-6,2e-6,7e-7] preserve the prefix-sensitive derivative and agree in sign with the negative AD a at both endpoints. Their finest-two full-vector errors are at most 2.1111499613381763e-7 and 2.803472196322486e-7. Both endpoints' fixed finest-two numerical a, b, c sign flags are true. No steps, thresholds, states, or coefficients were retuned.

The sole audit authenticated 1,191 original artifacts, 11 captured source hashes, 681 finite saved arrays, 413 exact input/return array associations, and 474 scalar comparisons. It retained all 226 nonzero scalar differences. The largest listed absolute difference is 3.4924596548080444e-10 in large cubic terms; all remain within the predeclared saved-comparison tolerance 5e-12+5e-10*abs(original scalar). These comparison tolerances do not relax the original scientific gates. The complete coarse/fine rows and all discrepancies remain immutable in the report linked below; no excluded or failed row was discarded.

Independence is limited precisely. The audit independently reduces the saved arrays and rechecks retained eigenpair equations, matrix-vector actions, finite differences, projected coefficients, and homological solutions. It shares the original native float64/AD returns. Full dense eigenvectors were not retained, so the complete spectrum's saved eigenpair-residual vector remains an authenticated source-dependent diagnostic; no fresh eigensolve was performed. Row 13 reuses the previously accepted full-matrix/profile authority rather than repeating its 896-column audit. Original Git-object equality for the 11 source files is inherited from accepted metadata authority `c9fa512d…`; the new audit independently rehashes captured source bytes and reports zero fresh Git queries. Historical producer checkpoint/tokenizer/vector identity limitations in the declared-map ancestry remain unchanged.

The saved audit executed once under source commit `24c69e37e572888041da92cecb81d5aa366db960`, CPU1 plus supervisor, nice10, 1GiB AS, and external 120s. Worker elapsed was 0.6627205391414464s, peak RSS 75,696KiB; supervision took 1.0331683510448784s, exit0 with no timeout or failure. Session 28168 was reaped by Inference. Supervisor 995603/start188069013 and worker 995619/start188069151, including their groups and sessions, are confirmed absent by the accepted final metadata authority. Original endpoint execution also remains closed under receipt `0cf633c0…`; it took 2132.3018840400036s with no timeout. No new model work occurred in the saved audit.

Exact release authorities (all paths under `/data2/neural-shrimp-data/codex/llm-cusp-crosscheck/`):

| Artifact | SHA-256 |
|---|---|
| `endpoints-001-independent-review-001/report.json` | `ee4d502d84b9ea16552d0123f08a0ba665e87917c28ff5b69a09c9526fff1b31` |
| `endpoints-001-independent-review-001/executed-audit.py` | `19b7d85f9e5f18aeea82ce0f53366fd3df4e6194e05626382f7865e2cb9af678` |
| `endpoints-001-independent-review-001/card.json` | `381a62f97f05399190c11f64a5ca7895addc1957ea35da5622a51bf2d4a8d59b` |
| `endpoints-001-independent-review-001/final-manifest.json` | `0be1309417d944c61cd83e2924c98755dd007105758bfdb00b033573ce085416` |
| `endpoints-001-independent-review-001-supervision/admission.json` | `b7f774b7a8730216db23ecad763595968194df3d450ed5f6fef0a2b44a885484` |
| `endpoints-001-independent-review-001-supervision/closure.json` | `c359529c92d82b694ca34a758a3bf0b9b4297d592f73cb2850033de93180c5f8` |
| `endpoints-001-independent-review-001-supervision/final-manifest.json` | `cfb14b04920971f964caf76ba8a57390ee933c6088c13e65cfb1abc3ecdef264` |
| `final-assembly-002-witness-audit-001-authority/report.json` | `9db2eaa3fcdcfc130fec9c09ba151b93b8bceb8b23911025b34db960ef32d78f` |
| `final-assembly-002-witness-audit-001-authority/manifest.json` | `cbfa56e93d411dd10a6a25b41bcac5fc128f1112f0204ca56ada34a59a852c46` |

The last authority retains a legacy Task1 adapter directory name. Its actual contents bind the Task14 source, report, session, and identities above. The exact path is preserved; no rename or replay was performed. The original source/cards and the disclosed prelaunch amendment removing memory-incompatible fresh Git queries also remain preserved. This final note reviews the completed evidence and introduces no new computation or acceptance threshold.
