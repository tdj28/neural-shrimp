# Numerical two-parameter fold continuation, 9 October 2026

The corrected adapter carries the certified primitive18 fold seed through64
numerically qualified boundary points (32 per direction), using actual hidden
and output rates. All321 returned augmented evaluations are retained. The run
used58.862s and51292KiB; owned process group exited. No new attracting orbit
instance and no rigorous inter-point connection follows.

Initial attempts exposed a numerical-adapter constructor precision bug: exact
rational model constants were parsed before the requested Arb precision was
set. Extended coordinates alone did not remove the residual floor. Fixing the
constructor order reduced the seed residual to9.34e-19. All15 controls passed
before the final run, including a regression resetting global precision to53.
The old rigorous proof kernels and certificate drivers were unaffected. Both
failed attempts and the intermediate synthetic test failure remain disclosed.

The positive-output direction has a second real multiplier changing from
0.97208 to1.00232 between nodes12 and13, near(hidden73.5–73.4, output0.827).
The normalized fold coefficient jumps from−63.95 to+753.60 there. This is a
candidate loss of simple+1 isolation and normalization pole, **not a cusp
zero**. The first saved-only summary's tentative cusp label is superseded by
summary002, which records all critical gaps and normalized-left-vector norms.
No two-dimensional geometry or codimension-two bifurcation is certified here.

Next: continue the same fold beyond its carried endpoint, trace the known flip
branches in the two-rate plane, and refine competing-multiplier crossings
without a simple-multiplier normal form at the degenerate point. Keep the
separate validated interior-connection track running. Exact source/input/output
pins, all scope limitations and prior failures are in the JSON companion.
