# Independent review: secant restoration and scalar reuse

Reviewed `secant_restore.py` SHA-256
`b89cb8ca176918a8178ff2bbe2516ea9f4e33ca9c2aa7d117f4c00d6d886e7ba`
and tests
`4641567e0db5374914da2095ff823d5b4b045006d8faaf6ff7aafce89f9bf23c`;
`inverse_secant_fast.py` SHA-256
`50a74a620b7744eb6ef4226664199969a2b84d64d1a607b63a1944195e59b026`
and tests
`65e3219be889d3464304e2d34b93ebdcd8542a04452d571233ff255583f80c74`.
The original kernel remains unchanged at
`01065e31a70b7b7b0b394609e2afef78ea96669455a72fd4c17546b94d8d914b`.
No target computation, completed-target kernel replay or Git mutation occurred.

No unresolved blocker found in these pure helpers. Restoration consumes every
externally authenticated decision in bound chronological order and reconstructs
the active history **after the final proposal**, including its clear. Every
rejection clears history. At four existing factors, the next update clears and
discards that pair; it does not clear and then accept it. A proposal step-cap
fallback clears even previously accepted factors. The complete parent/source/
receipt, chronological state and residual bindings remain obligations of the
future caller, not guarantees obtained merely by parsing records.

Retained factors must be canonical exact dyadic midpoints of the original saved
finite enclosures with the expected full dimension. The rank-one infinity norm
is recomputed with exact rational arithmetic and compared against its retained
bound and fixed cap. Accepted signal/noise and positive-denominator conditions
are checked. Candidate construction is checked against the saved exact centres
and full chosen correction at the calling arithmetic precision; the future
2048-bit driver must preserve that context. Installation copies factor lists
and A/B matrices into an independent mutable kernel. It admits numerical
inverse proposals only and never establishes inverse accuracy or an orbit.

One review refinement was incorporated before signoff: the original step-norm
endpoint bounds are now checked, and `accelerated` requires
`accelerated_step_upper <= 4 * baseline_step_lower`. The step-cap fallback
requires the opposite decision. Retained original intervals determine those
norm bounds; a wider decoded ball cannot repair a corrupt original norm.

The separate fast kernel differs from the original only in its explanatory
header and hoisting `_dot(w, ys)` out of the `scaled_secant_defect` coordinate
comprehension. Neither input vector changes during that expression. With fixed
arithmetic context and immutable inputs, computing the same signed sum once,
in the same order, produces exactly the same Arb ball as each repeated call.
The following multiply/add/subtract operations per coordinate are unchanged.
This preserves mathematical and serialized decisions; it does not change the
method, precision, caps, acceptance rule, candidate, or residual definition.
For 7168 coordinates, the previous defect expression redundantly visits
51,380,224 dot-product terms instead of 7168. This is an operation-count
improvement; target wall-clock savings have not been measured by this review.

Independently ran 74 combined restoration, original-kernel and fast-kernel
synthetic tests (0.46 seconds, CPU1). Full decisions/proposals/histories match
exactly across 128-, 256- and 2048-bit cases including nonnormal, collinear,
zero, noisy and capped inputs. A separate 256-coordinate case counts 257 total
calls to the original dot helper versus two for the new helper, while preserving
every saved witness exactly. Restore tests cover post-proposal clearing,
cap restart/discard, rejected factors, malformed records, step-norm corruption,
exact candidate construction and deep-copy isolation.

At this reviewed source, `secant_restore.install_factors` still constructs the
original kernel class. Future execution must explicitly choose and source-bind
the fast class if intended; reviewing equivalence does not silently change that
wiring. Both helpers set `certificate_passed=False` and leave external evidence
authentication explicit.

## Integration check, 13 October 2026

Root checked the subsequent one-line import change: `secant_restore.py` now
selects `inverse_secant_fast`, SHA256
`c1daae4ad7ccc4e633701db3b4f308465fe519050faa3c8c90f538eafecf92ba`.
No restoration or kernel arithmetic changed. The combined original/fast kernel,
restoration, phase-worker and scheduler suite passed136 tests (exit0,5.21s).
The only warning was inability to write pytest's disposable cache in the
restricted worktree. The original kernel hash remains unchanged.
