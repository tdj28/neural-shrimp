# Period-18 augmented-fold interval pilot — 9 October 2026

Behavioral question: does the observed end of the small model's period-18
window contain an actual local saddle-node of primitive cycles? The proposed
certificate concerns neutral stability at one fold; it is not a fifth attracting
orbit certificate or a proof of the whole shrimp boundary.

This is a new exploratory kernel. Its mathematical review must precede the
single target proof. The existing C1 checker is unchanged and its prior review
does not qualify these new derivatives or fold gates. No target computation has
occurred while preparing this card.

## Exact inputs and unknowns

Use only consumed C1 inputs from attempt-003, SHA256
`a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81`.
The qualified numerical ray0 fold outcome is
`/data2/neural-shrimp-data/explore/s2-period18-fold/attempt-001/outcome.json`,
SHA256 `ddd90afa59622d1bc3da59bfc1eb424afcb16570abbcf8dee8f3eefeadd4ec13`;
its complete manifest is pinned in the adapter. Preserve these artifacts.

The 27 unknowns are `(theta[13], eta_hidden, v[13])`. Set the initial eta
proposal by the original producer's scalar Python operation
`10.0 ** float(log10_hidden)` and then interpret its binary64 value exactly.
The variable itself is the real rate eta, so the proof needs no log/exponential
rate derivative. The fixed output rate is the exact dyadic corresponding to
`0x1.a461eed55764cp-1`, identical to historical row111. The source binds this
conversion and the numerical log-coordinate proposal. Proximity or common
constant identity does not prove connection to the earlier interior orbit.

Let `R(theta,eta)=F_eta^18(theta)-theta`, `A=R_theta`, and
`G=(R,A v,(v^T v-1)/2)`. The mixed forward jet carries a scalar value, its 14
base derivatives in `(theta,eta)`, its theta-direction derivative along the
interval vector v, and that directional derivative's 14 base derivatives with
v held fixed. Apply product/tanh rules to the explicit analytic loss-gradient
formula. This is second-order AD of the gradient, including third loss
partials. No floating derivative is used in the proof. Every update uses the
same pre-update state. The augmented Jacobian is
`[[A,R_eta,0],[D_theta(A v),D_eta(A v),A],[0,0,v^T]]`.

## Gates and theorem scope

At 256 bits, perform five Arb Newton proposal steps, then use a common requested
radius `2^-80`. Serialized Arb radii are the actual outward radii. A dyadic
midpoint inverse preconditions G. Strict Krawczyk inclusion and a whole-box
infinity-norm contraction below one establish existence and local uniqueness
of a normalized augmented root in this 27D box. The hidden-rate box must be
strictly positive. Other phases and the opposite sign of v are distinct local
representations; no uniqueness across them is claimed.

Exclude proper divisors 1,2,3,6,9 by coordinate separation of the interval
forward orbit from its initial state box. On the full proved box, validate
invertibility of `B=[[A^T,v],[v^T,0]]` with a dyadic inverse defect bound below
one; solve `B(w,alpha)=(0,1)` using Arb. At the exact root, `Av=0` and
`v^T v=1` force alpha=0 and w^T v=1. Invertible B implies rank(A)=12 and
nonzero left/right overlap, hence the multiplier +1 is algebraically simple.

Enclose `a=w^T R_eta` and `c=w^T D_theta^2 R[v,v]`; both must exclude zero.
The scalar quadratic coefficient is c/2 and eta''=-c/a. Finally form
`Q=M-v w^T` over the same box and require `||Q^(2^j)||inf<1` for some power
at most4096. At the exact root its spectrum is zero plus the other12
multipliers. This proves the complement is strictly attracting, while the
fold itself has a neutral +1 multiplier.

All gates give a local nondegenerate saddle-node of primitive18 cycles in the
exact-real map. No finite-detuning orbit/circle enclosure, global branch
identity, cusp, resonance ancestry, whole-window theorem, floating-trajectory
claim, or training benefit follows. A root-only or otherwise partial result
must retain its actual passed gates and remain below the full-fold claim.

## Bounded execution and replay

The committed driver `tools/certify_s2_period18_fold.py` supports three modes:
`profile` (synthetic only), `target` (requires subsequent root authorization),
and `replay` (same source and libraries, pinned saved bundle, byte-identical
mathematical result). Source inventories include all transitive package imports,
tests, and this card. Original C1 source is unchanged. The source snapshot,
exact inputs, five returned Newton evaluations, whole-box evaluation, root gate,
final certificate or failure, runtime/PID/command metadata and manifests are
retained. Save returned evaluations before checking the deadline.

Profile: exactly two synthetic 13D,24-sample,18-step mixed-jet evaluations
(center and radius2^-80 box), with nonunit scales and biases. It makes no fold
claim and opens no target arrays. Report the measured time and a seven-evaluation
projection; this is a cost screen, not a guarantee for the target.

Per invocation: CPU/BLAS1, nice>=10,120s cooperative internal wall deadline,
135s external TERM plus5s KILL,160 ticks (each composed map step, retained
evaluation, and matrix-power gate counted),16MiB raw plus at most1MiB launcher,
2GiB monitored peak RSS. RSS is a monitored limit, not a hard allocation cap.
The ordinary-write reserve is14MiB. The target uses one fixed setting only;
no radius/precision retry or alternative target is authorized by this card.
All exclusive directories and failed/partial attempts are retained read-only.

Planned roots:
`/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-001-profile`
and `attempt-001-target`. Use the project Python. Example target command after
review/authorization:

```
/data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_fold.py --mode target --out /data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-001-target
```

Replay takes `--mode replay --bundle <target>/run --manifest-sha256 <pinned
run manifest> --out <new exclusive attempt>`. It never modifies original proof
bytes. It also reconstructs the exact target inputs from the pinned consumed
artifacts, preventing a rehashed input/metadata substitution from being treated
as the original target. Trust remains the analytic formulas, Python/Arb/FLINT,
and their directed enclosures; this is not a proof-assistant verification.

Synthetic controls cover exact mixed polynomial/tanh rules, all augmented
blocks, bias/scaling/ridge conventions against the independent existing analytic
map Jacobian, high-precision directional differences of that Jacobian (tests
only), a nonnormal left-vector normalization, fold/degenerate/defective cases,
other unstable modes, period-divisor rejection, and interruption/settings gates.
