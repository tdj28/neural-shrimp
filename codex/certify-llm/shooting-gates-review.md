# Independent review of streamed multiple-shooting gates

Date: 13 October 2026 UTC. Reviewer: Codex inference subagent, independent of
the gate implementation. Reviewed source:
`codex/certify-llm/shooting_gates.py`, SHA-256
`7908f88bf6b0e5134619d2247f516d64a29e0f4d54b9aaf5b0ee6d45a37a0c44`.

No conditional algebraic defect was found. The verdict is conditional on the
explicit full-phase-box enclosure contract, and does not certify a target model.
This review used source inspection and CPU1 synthetic calculations only; no
checkpoint, target model or target artifact evaluation occurred.

## Root operator and streamed blocks

Write the shooting residual as `H_i(x)=F(x_i)-x_(i+1)` with cyclic indices.
Let `A_i` and `B` be the exact dyadic proposals, and let
`P=A_(p-1)...A_0`. Define `Phi_i=A_(i-1)...A_0`, with `Phi_0=I`.
The implemented linear preconditioner R is determined by

```
z_0 = -B * sum_j [A_(p-1)...A_(j+1)] b_j
z_(i+1) = A_i z_i - b_i.
```

These equations specify a fixed linear operator even if B is only an
approximate inverse. They give

```
R_(0,j) = -B [A_(p-1)...A_(j+1)]
R_(i+1,j) = A_i R_(i,j) - delta_(i,j) I.
```

For a full-box Jacobian enclosure `J_i`, let `E_i=J_i-A_i` and
`D=I-B(I-P)`. The defect blocks obey

```
(I-R DH)_(i,j) = delta_(j,0) Phi_i D - R_(i,j) E_j.
```

The reverse-column suffix recurrence, forward phase recurrence, column-zero
`Phi_i D` term and their signs all match this identity. The interval products
preserve signed cancellation until the induced norm is taken. Rounded Arb
products remain enclosures of the exact products of the original dyadic
proposals; their midpoints are not silently promoted to exact products.

The preconditioned residual recurrence uses the opposite sign to R: starting
with `s=0`, it updates `s <- A_i s+h_i`, obtains `delta_0=B s`, and then
`delta_(i+1)=A_i delta_i+h_i`. Thus the stored correction encloses `-R H(c)`,
which is the constant part of the fixed-point displacement for `x-RH(x)`.
This matches the separate `shooting_correction.py` sign convention.

## Strict gates and exact scope

For exact positive scalar phase radii r_i, the gate sums
`||K_(i,j)||_inf r_j` before dividing by r_i. Taking each block's maximum row
sum separately is conservative: it can reject a valid contraction, but cannot
understate the required product-space norm. Every multiplication, division,
absolute value and endpoint comparison is performed in Arb with outward bounds.
The strict inclusion comparison uses the supplied exact nominal radius, not a
larger radius introduced by an Arb box constructor.

The required conditions are, for every phase,

```
q_i = sum_j ||K_(i,j)||_inf r_j/r_i < 1
||(-R H(c))_i||_inf + sum_j ||K_(i,j)||_inf r_j < r_i.
```

Conditional on the declared C1 and full-box derivative enclosures, these give
a contraction of the full nominal product box into itself. They also establish
that R is nonsingular: at an actual point in the box, `||I-R DH||<1` makes
`R DH` invertible. Consequently a fixed point of `x-RH(x)` is a zero of H,
rather than merely a point with `R H=0`. The result is one unique root tuple
of the cyclic shooting equations in that full product box.

This does **not** assert uniqueness of every `F^p` fixed point in `X_0`:
another such point could have intermediate phases outside the other prescribed
boxes. The implementation states this distinction correctly. It also correctly
sets `attraction_checked` to false; invertibility and contraction of the
root-finding operator do not establish dynamical attraction of the orbit.

The enclosure contract hashes the exact domain, all supplied primal/Jacobian
balls and all A/B proposals. It explicitly requires C1 regularity on a
neighborhood of each phase box and bounds for all state coordinates and
transverse directions throughout each box. Hash syntax and declarations do not
establish that requirement. `conditional_gates_passed` and
`enclosure_validity_checked_by_this_module: false` retain that distinction.

## Primitive period is conditional on the same root domain

For each proper divisor d of p, `primitive_phase_gate` asks for a strict
coordinate separation of `X_d` and `X_0`. A root tuple in those boxes obeys
`x_d=F^d(x_0)`, so this separation excludes period d. Testing every proper
divisor excludes all smaller possible minimal periods of a period-p point.
Equality of box boundaries correctly fails the strict-separation test.

The phase box `X_d` is not asserted to enclose `F^d(X_0)` for every initial
point in `X_0`. The implementation correctly states the dependency on a shooting
root and returns the domain hash. A later aggregator must require successful
root existence for **that same domain** before treating this separation as a
primitive-period conclusion. This gate alone is not an existence theorem.

Attraction would still require separately checked rigorous derivative products
at the proved root, using the full-box J_i enclosures in the correct phase order
and a rigorous norm/power bound. The numerical A_i product is only a proposal
and must not substitute for those enclosures.

## Independent validation and limits

The existing **32 synthetic tests** pass on CPU1. They cover noncommuting dense
comparison, signed cancellation, inverse defects, derivative uncertainty,
strict-boundary refusals, exact nominal radii, unstable-map root versus
attraction, contract/input mutations and primitive-phase dependencies.

An additional independent check constructed R by explicit exact-rational block
products rather than the implementation's suffix/forward recurrence. For
periods 2, 3 and 5, state dimension 2, signed noncommuting dyadic A/B proposals
and separately perturbed Jacobians, all **152 matrix-entry comparisons** of
`I-R DH` with the streamed Arb blocks were exact. No floating eigensolver or
norm approximation entered these checks.

The implementation streams n-by-n blocks and does not allocate the full
(pn)-by-(pn) matrix. This is a memory property, not a measured target-scale speed
claim. Dense block products and per-entry hashing still require substantial
work; a target execution needs its own measured, bounded resource plan. These
tiny tests do not establish 896-dimensional gate runtime or supply the missing
full-box derivative enclosures.

## Addendum: optional factored norm bounds

Reviewed revised source SHA-256
`9c967c0dcb9783df53f33c05eb897050f4e8971f66767abbc8a01a8b1b85cf1f`.
The earlier source hash identifies the original streamed-only review; this
addendum covers the current two-method version. `bound_method='streamed'`
remains the default and its signed block calculation is unchanged.

The optional `factored` mode is a sound conservative alternative. With
`Q_j=A_(p-1)...A_(j+1)` and `S_ij=A_(i-1)...A_(j+1)` for `j<i`, the exact
preconditioner block is

```
R_ij = -Phi_i B Q_j - 1_(j<i) S_ij.
```

Here `S_(j+1,j)=I`; for `i<=j` the second term is absent. The implementation
therefore bounds its norm by

```
||R_ij|| <= ||Phi_i|| ||B|| ||Q_j||
             + 1_(j<i) product_(k=j+1..i-1) ||A_k||.
```

It then bounds `K_ij` by the column-zero `||Phi_i D||` term plus that
preconditioner bound times `||J_j-A_j||`. Signed prefixes, suffixes and
`Phi_i D` are formed with Arb before their norms are taken. The remaining
triangle and submultiplicative estimates are outward scalar bounds; they may
lose cancellation but cannot understate the exact block norm. The empty segment
product is correctly one when `i=j+1`, and the absent term is zero when `i<=j`.
All later phase-weighted strict inclusion/contraction comparisons use these
bounds in the same conservative way as the original mode.

Factored outputs identify their method and retain all scalar factors. They do
not fabricate a matrix enclosure hash or row-by-row matrix norm witnesses when
no K block was constructed. The same complete enclosure/proposal/domain
contract, exact nominal radii, primitive-phase dependency and absence of an
attraction conclusion remain in force. A failed factored bound is inconclusive;
it is not evidence against a contraction that a less conservative method might
establish.

Independent validation: all **37 synthetic tests** pass on CPU1. An additional
check assembled the full exact-rational `I-R DH` independently for signed,
noncommuting dyadic examples of periods 2, 3 and 5 in dimension 2. Every one of
**38 block norms** was no larger than the corresponding factored bound. The
previous 152 exact-entry checks apply to the preserved streamed method. No target
run was performed. Reduced matrix-multiplication count is an algorithmic
advantage, not a measured 896-dimensional runtime result; this mode retains a
list of signed prefixes, using O(p n^2) storage besides inputs rather than only
O(n^2) streamed temporaries.
