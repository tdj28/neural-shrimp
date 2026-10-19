# One-tube conditional infinity-operator helper

Disposition: prepared isolated source and new small controls for Root review. No target model, point evidence, raw data, Git mutation or process operation was used. This is not a target driver, launch card, trajectory enclosure or certificate. A first target evaluation remains a separately measured, bounded one-tube experiment after actual original point evidence is audited.

## Source and evidence

- `/tmp/tube_operator.py`: SHA-256 `4e837348df63dbea4a1cef752b30d3d73bce23a0a279a35696a502245d804722`,6794 bytes.
- `/tmp/tube_operator_controls.py`: `89f9ab2246c7d15c2515e4a90312e183274c052c29d967a276d5f4b3c863225e`,5248 bytes.
- `/tmp/tube_operator-controls.json`: `5bd6179bf1ee55cab6d51af48262e6f61c27cfad67d4c259a9c1526eeb026251`,1575 bytes.

Reviewed existing source semantics:

- `ball_jet.py`: `46b14d8bd65ed6c80174758daf37d00110cfa908813917bdb46c13749f999719`.
- `streamed_head.py`: `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a`.
- `ball_linear.py`: `5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2`.
- `ball_qwen.py`: `fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80`.

All these existing files were read only. The tests import the existing Jet arithmetic for tiny scalar/polynomial expressions; no BallQwen/StreamedHead model instance, target checkpoint, prefix, state or forward evaluation was constructed. Test backend was python-flint0.9.0 / FLINT3.6.0,128-bit context and one thread. No old point-loop, scalar-recurrence or target arithmetic tests were repeated.

## Exact quantifiers

Let c be an exact d-vector and r>0 an exact dyadic number. Define the nominal closed convex cube X={x: ||x-c||_infinity<=r}. Let U=[-1,1]^d. The helper constructs or admits native primal balls P_i with [c_i-r,c_i+r] contained in P_i, and a native Arb matrix of shape(d,1) with [-1,1] contained in every entry V_i. Every endpoint containment is checked from **original native midpoint/radius integer pairs** with Fraction arithmetic. There is no endpoint decoding through another rounded Arb operation.

The single matrix column denotes the Cartesian set of all real vectors v with independently variable coordinates v_i in V_i. It is not one selected vector, the all-ones vector, one basis vector, or a finite sample. The helper refuses those narrower choices. A wider interval input is accepted only as a conservative containing set.

The conditional evaluator contract is:

For every x in product(P_i), and every v in product(V_i), the returned primal ball Y_i contains F_i(x), and signed derivative ball D_i contains [DF(x)v]_i, with the same fixed mathematical map and authenticated fixed prefix/parameters/assets. All relevant denominator/domain gates must have succeeded over those containing sets.

For each native derivative ball with exact endpoints l_i,u_i, define b_i=max(|l_i|,|u_i|), and L=max_i b_i. Then for every x in X and every v in U,

    ||DF(x)v||_infinity <= L.

Taking the supremum over v in the infinity unit cube and then x yields

    sup_{x in X} ||DF(x)||_infinity <= L.

There are no omitted tangent directions: their entire cube is represented by the one set-valued column. This is precisely why one ordinary all-ones JVP cannot supply the same conclusion. This bound need not be tight and does not reconstruct Jacobian entries or a signed matrix product.

## Why existing interval-Jet semantics support the conditional contract

BallQwenJet.forward accepts a native Arb(d,K) matrix and passes each entry through exact() without deleting its radius. For K=1, the scalar Jet stores the supplied interval tangent. StreamedHeadQwenJet's _direction_rows performs the same operation. Affine packing passes the complete primal and tangent columns into the reviewed Arb matrix product. The nonlinear Jet rules are the chain/product/quotient/exp/positive-rsqrt rules with outward interval arithmetic. The streamed head retains interval directions in both complete vocabulary passes and in its signed derivative formula.

Inductively, fix any one admissible pair (x,v). Every actual intermediate primal and derivative value for this same pair is contained in the corresponding interval computed by the forward graph. When interval products or sums forget correlations between repeated occurrences, the resulting set grows: it includes choices that might not come from the same (x,v), but still contains every actual shared choice. This gives conservative overestimation, not missing directions. The same argument applies to uncertain but fixed prefix constants with zero state tangent: actual source/cache authority must establish that they enclose the intended fixed prefix.

The existing helper/kernel comments about K columns not covering omitted basis directions apply to ordinary selected directions. They do not invalidate the distinct, explicit quantifier above, where a column is admitted to contain the whole unit direction cube. Conversely this helper must not be used with an evaluator that silently replaces interval directions by their midpoints.

## Helper interface and evidence

`build(centres, radius, dimension)` creates live inputs. Centres must be a full list/tuple of finite exact Arb scalars. Radius is an integer or Fraction with positive power-of-two denominator; floats, bools, uncertain balls, nonpositive values and non-dyadic fractions are refused.

`evaluate(centres, radius, forward, dimension=..., primal=None, directions=None, save=None)` makes exactly one supplied callback call. If explicit live inputs are provided, both are required and admitted by the same exact endpoint checks. Before/after native records detect changed primal balls, directions, nominal centres and arithmetic context. A callback that broadens a direction while still containing the unit interval is refused by the snapshot equality. These are finite before/after value guards, not protection against a hostile callback that changes an input and restores it before returning; the evaluator and progress/saving callbacks remain trusted reviewed source.

If `save` is supplied, the helper emits an input record before evaluating and an operator record after successful full output admission. The operator record retains exact centres, nominal radius, all native live primal/direction balls, every full signed primal/derivative output, each exact rational endpoint-magnitude bound and exact rational maximum L. No midpoint-only output is substituted. The helper performs no filesystem I/O by itself; the future owned driver must require durable exclusive saving, preserve failures and authenticate the complete run.

Every emitted record has `conditional_operator_bound_only:true`, `trajectory_enclosure:false` and `certificate_passed:false`. The evaluator's source, backend, fixed-prefix installation, full dimensions/layers/vocabulary, completed domain checks, model assets and original execution/closure are not proved by a callback signature. They must be joined separately before any target L is used.

## New controls

27 new controls passed with exit0. The two mathematical controls are:

1. F(x,y)=(x-y,0): an ordinary all-ones tangent gives exactly zero, but the independently varying interval direction cube returns exact L=2, the correct infinity operator norm. The actual helper refuses ordinary all-ones and basis columns.
2. F(x)=x^2 on the positive-radius tube [3/4,5/4]. The exact directional derivative extrema over v in[-1,1] are +/-5/2. The live Jet output contains both extrema and produces exact rational L=671088643/268435456 =5/2+3/268435456. That tiny positive native magnitude-rounding slack is retained rather than falsely asserting exact endpoint equality. The native primal tube endpoints exactly match [3/4,5/4].

The other25 controls cover too-narrow/wrong-shape/nonfinite directions, uncertain/nonfinite/wrong-dimensional centres, unsupported/nonpositive radii, insufficient/nonfinite/wrong-dimensional live primal inputs, full output dimensions/finiteness, live primal/direction/centre/context mutation and partial input allocation. All concern the new whole-direction-cube interface; prior thin-loop/recurrence checks were not rerun.

## What remains before target use

After the original thin-point evidence is audited, freeze one actual centre and a positive exact dyadic radius containing its separately established incoming error. Admit a new one-tube driver/card with exact reviewed evaluator/source/prefix/backend/input pins and explicit resource bounds; retain all layers/head coverage and original enclosures. Measure real wall/CPU/RSS/output cost and actual positivity/L result. A point-call timing or64-column timing divided by64 is not a valid one-tube forecast.

Natural interval dependency can make L enormous or cause positivity failure, even when the exact trajectory is stable. Either is an honest finite experiment outcome; no automatic precision/radius retry is supplied. For trajectory propagation, independently authenticate the local defect and require incoming error<=r before applying e_next<=defect+L*e. Every later tube or block needs its own full-domain and intermediate-containment evidence. No periodic-root power bound, larger basin, exact-start reachability or certificate follows from this helper or its manufactured controls.

## Root review and integration, 2026-10-19T01:43Z

Root read the complete helper, exact-quantifier argument and original27-control
source/results. The whole direction cube, native primal containment and signed
output endpoint maximum establish the stated conditional infinity norm bound;
no sampled or all-ones JVP is substituted. Original mathematical source kernels
remain unchanged. Native construction slack is retained and the wrapper detects
primal/direction/centre/context changes. The fixture source is preserved verbatim
as tube-operator-controls-source.txt with its original temporary-path imports;
the completed controls were not rerun during integration. No target evidence was
read or evaluated. A required original-output/source/regularity join and a new
measured one-tube driver/card remain before target use.
