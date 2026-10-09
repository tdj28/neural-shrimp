# Period-18 fold kernel: independent source review

Source review passes `f45444e54d07b19bd497ce74ef0be4683e27b350`, following
initial snapshot `8dd896d3`. The only requested correction makes the exact
historical output-rate identity a required adapter gate; two synthetic fixtures
cover matching and mismatched conversions. The author reports 16 passing
controls. This reviewer inspected their source and did not run a target or
repeat the test suite. File hashes are in the companion JSON.

The mixed jet correctly differentiates the explicit simultaneous gradient
update, including the required third loss derivatives. Its augmented Jacobian
is the derivative of `(R, R_x v, (v^T v-1)/2)` with respect to `(x, eta, v)`.
Strict whole-box Krawczyk inclusion and contraction give the local normalized
root. At that root, the validated left border supplies `w^T v=1`, a
one-dimensional kernel and an algebraically simple multiplier +1. Nonzero
`a=w^T R_eta` and `c=w^T R_xx[v,v]` are the generic fold conditions; the
quadratic scalar coefficient is `c/2`. The deflated matrix
`Q=M-v w^T` has zero plus the remaining multipliers, so a validated power
norm below one proves the attracting complement. All these enclosures and the
proper-divisor exclusions 1,2,3,6,9 use the full proved box.

The source saves its root gate before the later border/power checks, requires
all gates for a full fold result, binds the consumed inputs and runtime source,
and provides byte-identical mathematical replay. The new kernel needs its own
review; the earlier unchanged C1 checker is only a dependency. A passing target
would establish a local primitive-cycle fold in exact real arithmetic, with
neutral stability at the fold. It would not establish finite-parameter branch
identity, the whole shrimp boundary, floating-trajectory fidelity or utility.
