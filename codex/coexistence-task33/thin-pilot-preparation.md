# Thin fixed-start pilot preparation

Prepared source-only, separately from pinned Task34 assembly. No target model or
point was evaluated, no raw evidence was read/written, and no Git/process action
was taken. The completed five thin-loop controls and eight scalar recurrence
controls were read as existing authority and were **not repeated**. The new
wrapper was syntax-checked only; runtime admission and new wrapper-specific
interface review remain Root's work before any launch.

## Concrete isolated caller

`/tmp/thin_pilot.py` reuses the unchanged `thin_pseudo_orbit.generate` loop with
the accepted `transient_pilot` model/prefix authority setup. It is a distinct
point-defect experiment, not a retry of the closed uncertain-input pilot.

- New card must say `experiment: thin-pseudo-orbit-point-defects-v1`,
  `trajectory_enclosure:false`, `certificate_passed:false`, exactly 20 steps,
  2048 bits, and the exact binary64 Task33 cell (.06468,-.148).
- Exact state zero is checkpoint embedding row198, the last token of the
  authenticated37-token prefix. Its initial error is exactly zero relative to
  this mathematical fixed-start definition. The producer `SoftLoop.prepare`
  source is pinned only as definition authority; the producer is not executed.
  The manual map's lossless checkpoint conversion and existing exact-asset
  review remain the authority for the actual initial coefficients.
- The same completed2048 prefix is required: original source
  8c5170bce23a3cf054de8638ba047dcabc893518, receipt
  0e787d0ebd6fb505a202c96fb3cc2429803aa8cc50c5271f728dc3da59af6f12,
  admission84be8d159949dc60aa0f3909c6a16ee5a967dab0e40cd717dda2738331bcfa28,
  and original cache/metadata included in card input pins. The accepted importer
  restores every primal radius; the prefix remains fixed with zero state tangent.
- Every center is exact. Every full896-component image, selected next midpoint,
  signed local defect enclosure and its exact bound/radius are retained by the
  unchanged loop. The wrapper additionally records per-point time, exact map
  ratios, source/card/helper hashes, fixed-start/prefix authority, progress and
  failure. Every saved record explicitly carries `trajectory_enclosure:false`.
- Inputs and assets are authenticated before and after; source/card/helper/input
  pins must be in the external supervised plan. Fresh owned `thin-pilot-*` output
  refuses reuse. Failures retain all already saved steps. The wrapper makes no
  trajectory-error propagation, whole-tube derivative, basin, token or certificate
  claim. A successful local point image still encloses F(c_n); what is absent is
  a bound that the exact fixed-start trajectory remains near all later c_n.

The new card may start from the old transient card's unchanged prefix, fixed-start
and asset pins, but must replace the old experiment/driver with the new declaration,
new caller/helper source pins, a fresh output/supervision path and reviewed limits.
The failed transient attempt and its output remain untouched. Root should bind
this helper and wrapper in an exact new private source checkpoint before launch.

The old measured point-call range110–111.49seconds suggests2200–2229.8seconds for
20 point evaluations before setup/serialization. This is only point-call guidance:
new off-orbit point sizes, positivity gates and elapsed time can differ. A proposed
new pilot envelope is one CPU plus supervisor,18GiB AS,2600CPU/2700wall,128MiB
output,32GiB startup/16GiB runtime floors, with fresh capacity and a separately
reviewed card. The larger output allowance is prospective for full input+image+
next-center+defect records; it is not an extension of an old attempt. Root may
choose a measured conservative envelope instead. This has **not** been admitted
or launched and cannot budget derivative proofs from point timing alone.

## Smallest rigorous whole-tube derivative mechanism supported by current source

A single **interval tangent column** covering the entire direction cube is
supported by the existing reviewed Jet interface. This requires no new
transformer arithmetic or full896-column Jacobian assembly:

1. Freeze exact center c and nominal radius r for the actual convex tube
   X=c+[-r,r]^896. Construct and retain the live primal Arb inputs, proving with
   original rational endpoints that they contain every nominal coordinate.
2. Use `StreamedHeadQwenJet(..., direction_count=1)` and an Arb matrix V of
   shape(896,1) with every entry exactly [-1,1]. Do **not** pass a vector of ones,
   a single basis direction or a finite selection of random vectors. Each
   component of the interval column ranges independently over [-1,1]. Save
   original interval endpoints and check the entire direction cube is contained.
3. Install the authentic fixed prefix with zero state tangents and run the
   unchanged full decoder/head over the live primal tube and interval directions.
   Its interface explicitly accepts Arb interval columns (`ball_jet.forward` and
   `streamed_head._direction_rows`), and its arithmetic preserves tangent radii.
   Every coordinate of the returned one-column derivative encloses
   [DF(x)v]_i for every x in X and every ||v||inf<=1.
4. Therefore L=max_i sup|D_i| is a rigorous bound on
   sup_(x in X)||DF(x)||inf. Take actual outward endpoints/absolute upper bounds,
   retain all896 signed output derivative balls, and independently check the
   scalar maximum with exact Fraction arithmetic. Retain primal outputs,
   all24 layer events, both all-vocabulary head passes, live input/direction
   witnesses, original source/backend/assets/prefix and successful closure.

The justification is the same finite-graph interval derivative induction, now
quantified over interval directions as well as interval primal inputs. Interval
operations may forget correlations, which enlarges the set but does not exclude
any J(x)v. Since the whole infinity unit cube is included in this **one set-valued
column**, there are no omitted directions. This is not the invalid assertion
that one ordinary JVP bounds all896 directions. It supplies an operator norm
upper bound, not individual Jacobian columns or a signed monodromy matrix.

This bound can be extremely loose: all-direction dependency loss can grow through
many layers even when the true derivative is contracting. Increasing point
precision does not necessarily repair that loss. Primal interval positivity may
also fail on the proposed tube. A finite but enormous L is an honest rigorous
upper bound that may make the subsequent sufficient tube test fail; it is not
evidence that the exact trajectory diverges. No target measurement of this
mechanism has occurred. The source-level proposal needs an explicit new finite
review/controls for the direction-cube quantifier before admission; earlier
thin-point and scalar recurrence controls do not supply that review.

## Minimal measurement and what it can establish

The first meaningful positive-radius tube can be selected from the new point
pilot's first two completed original records after their audit. At exact start,
e_0=0 and e_1<=delta_0 follows directly from the first full point image, where
delta_0 bounds ||F(c_0)-c_1||inf. Choose and disclose an exact positive dyadic
r_1>=e_1 (for example twice that exact positive defect bound); do not use a
floating underflow display of the tiny defect. Pin c_1, r_1, the complete original
point image/defect and source/prefix authority in a separate **one-tube** bounded
measurement card. If the bound is zero, choose a declared positive radius rather
than silently treating a zero-radius derivative as whole-tube evidence.

Measure this one K=1 interval-direction evaluation at the declared precision,
recording wall/CPU/peak RSS/output bytes, actual positivity outcome, output L and
all retained enclosures. It is not justified to scale the64-direction shard cost
by1/64, or to call the scalar point cost a derivative measurement. Admission can
use a conservative finite ceiling informed by prior model calls, but only the
actual one-tube result supports a forecast for subsequent tube bounds. A failed
or unhelpfully loose result is preserved; no automatic radius/precision retry.

Given the already enclosed local defect delta_1 at c_1, the mean-value theorem
then gives e_2<=delta_1+L_1*e_1 because the complete segment from c_1 to the true
x_1 lies in the certified convex input tube. The existing exact scalar recurrence
can recombine this only after the original whole-tube authority is joined. Each
later step needs its own proved radius and whole-tube L, and e_n<=r_n before use.
A point Jacobian or point residual at c_n is insufficient when e_n>0.

A first implementation may stop at this one-step bound and measurement. A later
explicit finite card can declare at most20 tube bounds, input-radius selection
rules and hard stops, based on actual timing/width results. If interval-cube bounds
are too loose, the next alternative is complete signed whole-tube Jacobian
columns (14x64 directions per tube), or a separately reviewed second-order /
Taylor-remainder method. Neither is supplied by the present point pilot. Point
Jacobians become valid only with an independently established derivative-Lipschitz
or remainder bound over the entire tube, not by numerical closeness alone.

For blocks, a signed product of derivative enclosures over independently proved
intermediate tubes can exploit cancellations, but every intermediate exact
trajectory inclusion must hold before the final block derivative is used.
`block L` from a periodic root or unconstrained center chain cannot replace
those obligations. Existing recurrence intermediate(a,b,r) rows are scalar
assertions to be proved and authenticated separately, not self-certifying data.

## Connection to the released tiny basin

Even successful20-step point generation is not a reachability result. After a
validated error chain, entry into a released Task33 phase0 center ball requires
an exact check ||c_N-c_target||inf+e_N<=rho, with the same exact map and the
original certified saved center. P2rho=2^-681 and P3rho=2^-700 are separate known
inner-neighborhood authorities; a larger1e-6 radius requires new whole-domain
proof. The P2 M^8 bound is DF^16 at the periodic root, not an F^8 transient
bound. No initial-point or tube statement here extends those old boxes.

## Prepared source and inspected source hashes

- `/tmp/thin_pilot.py`: `d83f2f2e36d76228889c6dfc53bf6260770d51732340b3ddd709ba799502c4d1`
- `codex/coexistence-task33/tube-preparation.md`: `94129772cfbfa0a26c638d021936825869f13ca99792d35e7cd8672c3827353e`
- `codex/coexistence-task33/thin_pseudo_orbit.py`: `1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e`
- `codex/coexistence-task33/tube_recurrence.py`: `f03cfbb475cc54f788cfa6900a7e92963cc682b00995ff21deddf3a039b74fe5`
- `codex/coexistence-task33/transient_pilot.py`: `e71c16dbfed2d32b2fd654747743fb3568852efcc5b95651fcaf912715473780`
- `codex/coexistence-task33/transient-pilot-001-card.json`: `798146ab93aeb7d0d451c30f12b14fdeaa67b6cb7b63ab77b025393ee46f8ab0`
- `codex/coexistence-task33/transient-pilot-review.md`: `00be17f254f1ec6b010b3ecb3c94c9d66cbd28fc540aa3c22780e544ae5e63d9`
- `codex/coexistence-task33/prefix_import.py`: `b4f82802a9bf77b8d90e399c5c7e9b2635106d39532079852539ee227f1211e8`
- `codex/crosscheck/qwen_map.py`: `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e`
- `tools/explore_llm_soft_loop.py`: `ead3b44a82c556c1a376f3e58c389cd3fb2f7329786ef0a712a3d92e7a920843`
- `codex/certify-llm/ball_jet.py`: `46b14d8bd65ed6c80174758daf37d00110cfa908813917bdb46c13749f999719`
- `codex/certify-llm/streamed_head.py`: `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a`

## 2026-10-19T01:37Z wrapper and card review completed

The initial source-only wrapper now has18 actual new interface/authority refusal
controls and an independent full source review. See thin-pilot-001-source-review.md,
thin-pilot-001-independent-review.md and thin-pilot-001-controls.json. The concurrent
producer file change is explicitly crossbound to the unchanged original complete
SoftLoop class, while the new current full file hash is pinned separately. Earlier
source hashes above describe the earlier prepared snapshot and are not relabeled.
The new card and serial-held plan are prepared; no model call, prefix loading or
new numerical launch has occurred. Original loop/recurrence controls were not
repeated. Current P6 producer and saved audit must close before fresh serial
admission. Whole-tube derivative measurement remains separately required.
