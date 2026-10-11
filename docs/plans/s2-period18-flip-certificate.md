# Exact-ray primitive18 flip certificate — 9 October 2026

The behavioral question is whether the selected ray13 boundary is a genuine
period-doubling of the full18-update training cycle. A −1 multiplier is not
sufficient: prove its simplicity, nonzero total crossing and nonzero cubic
coefficient. A supercritical result predicts nearby stable doubled cycles
qualitatively; a finite-detuning36-cycle and its connection need separate work.
This is exploratory source development, not a launched proof or confirmation.

## Exact family and data

Use consumed S2-28011 data from the previous C1 input file (SHA256
`a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81`), historical
EXP028 row111/S2-28011-zoom:587 and the lowest-index flip ray13. Historical
certificates and rays have hashes `414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8`
and `a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4`.
The numerical author retains a full theta13,s,q13 proposal and its exact source
and input hashes. The forthcoming adapter must reconstruct the exact family
from these historical files and bind the complete numerical bundle.

For both layers use eta_i(s)=exp(log(10)*(ell_i+d_i*s)), where ell_i is the exact
rational interpretation of the original binary64 center log and d_i is the
exact rational interpretation of the original binary64 product
scale_i*ray13.direction_i. This multiplication convention is shared with the
numerical producer; it does not silently substitute an exact product of the
unrounded factors. Arb encloses log(10), exp and their derivatives. Both rates
vary, and both are differentiated. The exact exponential s=0 rates generally
differ from the earlier dyadic center-certificate constants. No connection to
that certificate follows from this choice. The positive rates in inherited
map_inputs are placeholders unused by the exact-ray update.

## New proof kernel

The old C1/fold kernels remain unchanged. New `rigorous_flips.py` differentiates
the explicit simultaneous loss-gradient update, with biases and ridge as in the
old model. A jet carries f, its gradient in (theta,s), D_q f, the base gradient
of D_q f, D_q²f and D_q³f; q is held constant during differentiation. Third-order
AD of the gradient supplies the required fourth loss derivatives without a full
tensor. Derivative fields contain no factorial factors. No finite differences
or floating derivatives enter proof gates.

For G=F_s^18, M=G_theta, R=G-theta and A=M-I, solve
H=(R,(M+I)q,(q^Tq-1)/2)=0 in27 unknowns. Whole-box Krawczyk strict inclusion and
contraction prove the normalized augmented root. Exclude proper divisors
1,2,3,6,9. An interval inverse witness for [[(M+I)^T,q],[q^T,0]] and its Arb solve
give a one-dimensional nullspace and p^Tq=1 at the root, hence algebraically
simple−1. Enclose A^-1 with an explicit inverse defect; theta_s=−A^-1 G_s.
The total crossing is alpha=p^T(G_theta,s q+B(theta_s,q)); the derivative at a
frozen state would be wrong when the orbit moves.

With u=A^-1 B(q,q), enclose
b=p^T C(q,q,q)/6−p^T B(q,u)/2, requiring b and alpha to exclude zero.
This is the cubic coefficient after eliminating the quadratic term; its
convention is derived in [Kuznetsov's primary lecture, pp.11–12](https://webspace.science.uu.nl/~kouzn101/NBA/UvA_lect3.pdf).
The formula was checked against that full-text derivation during implementation.
Finally prove ||(M+q p^T)^(2^j)||inf<1 for some power<=4096. This establishes an
attracting complement; the cycle at the flip itself has neutral multiplier−1.
If all gates pass, b>0 is supercritical with stable primitive36 cycles locally
on −alpha*(s−s*)>0. A negative b is subcritical and the nearby doubled cycles
are unstable on the opposite side. No numerical detuning size is certified.
Computed coefficient signs remain provisional until all gates pass.

## Source-only checkpoint and planned execution

The first checkpoint contains the kernel and synthetic controls only. The
adapter, full source/input inventory, exact profile gate and exclusive raw
paths will be added before launch. Root source review is required by this
unit's coordination. No target evaluation is authorized by this document.

Planned initial proof:256 bits, radius2^-100, five Newton proposals, power<=4096.
A synthetic13D/24-example/18-step center-plus-box profile measures cost first.
CPU/BLAS1, nice>=10,120s internal and135s external TERM plus5s KILL,160 map and
observer ticks,16MiB raw plus1MiB launcher, monitored2GiB peak RSS. One target
only after reviewed source and the measured resource gate; preserve all failures.
A later change requires a dated disclosed amendment, source checkpoint and updated
cost bound, with root coordination for target changes; existing user authorization
for bounded local exploration does not require another human approval.
No GPU, paid run, protected seed or holdout is used.

Synthetic controls cover exact third mixed products and tanh/exp rules; old
analytic map/Jacobian agreement with bias, scaling, ridge and both variable
rates; high-precision differences against that independent Jacobian (tests
only); a moving branch with partial crossing−13 but total crossing−1; nonnormal
left vectors; quadratic correction reversing the raw cubic sign; known
super/subcritical, zero cubic, zero crossing, defective and unstable-complement
cases; divisor exclusion; and linear phase-frame covariance alpha'=alpha,
b'=b/k². These are controls of code and conventions, not additional neural data.

## Adapter and executable card — source freeze before proof outcomes

The adapter `tools/certify_s2_period18_flip.py` is now implemented. It has a
pure independently testable historical family/seed reconstruction and binds
all consumed file hashes above. It also verifies the externally pinned complete
numerical outer and run manifests, all eight committed/retained numerical
runtime sources, the source/input hash carried by the seed, the selected full
state/eigenvector and original scale/unit-direction factors. Proof inputs
contain exact rational ell,d and theta,s,q; only real Arb log/exp enters the
mathematical map. Neither a floating eigenvalue nor numerical success passes
a proof gate. Source inventories include the new kernel, inherited exact-map
helpers and transitive imports, adapter, tests and this card.

The selected numerical proposal is now available from
`/data2/neural-shrimp-data/explore/s2-period18-flip/attempt-002-target`, execution
`9cc21b7f52b7b2edadb02cd1098d4843c4f1a3aa`, outer manifest
`bd34d93e8430ed17b0705725ef27c384c7e8ae4853e20f8363042cfcc493123c`.
Its `run/flip-proof-seed.json` hash is
`370bfc601a991f2e9186894287e7e9a141e2c5164dbc0424f1be474b7b9a7185`.
These are metadata supplied by the numerical producer before the proof run.
The numerical author reports an earlier complex-pair instability and re-entry
along the33-node carried branch, followed by this qualified flip and three
unstable primitive36 cycles. The first attempt stopped on the earlier loss of
attraction and remains retained. Thus the proposed flip is not asserted to be
the first stability boundary, nor is supercriticality presumed. These related
outcome summaries were known before this executable card; all proof gates and
the one selected target remain unchanged.

The resource-only profile makes exactly two finite synthetic18-step evaluations
(center and box) using13 parameters/24 examples and both variable rates. Its
externally SHA-pinned complete bundle must match the current eight bound source
files, libraries and CPU1/nice contract. The conservative gate is
7*max(measured evaluation seconds)+10 <=60s, within the120s internal ceiling.
Recorded elapsed/RSS/exit/inventory constraints must also pass. Reuse that exact
profile only while bound executable/card bytes are unchanged; adding an outcome
receipt or research-log entry does not alter those bytes. No favorable learning
or proof result controls this resource gate.

The target then makes one256-bit/radius2^-100/five-Newton/maxpower4096 attempt.
Retain each returned Newton and whole-box evaluation before the deadline check,
and the root gate before later border/coefficient/power failures. The raw bundle
contains the whole orbit, augmented Jacobian, M, mixed B(q,.) columns, B(q,q),
C(q,q,q), validated solves, full action vectors, power matrices, source/runtime,
exact inputs, and complete failure or certificate output. The launcher retains
its command, owned group PID, log, exit and verified final cleanup; it does not
interfere with other jobs. All outputs become read-only.

Separately record exact-Fraction proximity of the entire27D proof box to the
selected dyadic proposal, normalized by max(1,proposal infinity norm), with
cap1e-7. A mathematical proof and selected-proposal association are distinct
fields; a failed association does not erase the proof, but prevents a combined
selected-candidate pass. This is proximity, not inclusion or branch ancestry.
Replay requires an externally pinned proof manifest, reconstruction from the
same pinned numerical/historical inputs, matching source/libraries, byte-identical
mathematical result and exact association arithmetic.

Exclusive planned roots are
`/data2/neural-shrimp-data/explore/certificates/s2-period18-flip-2026-10-09/attempt-001-profile`
and `attempt-001-target`. After root source review, run the profile with the
project Python and `--mode profile --out <profile-root>`. The target command is:

```
/data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_flip.py --mode target --numerical-root /data2/neural-shrimp-data/explore/s2-period18-flip/attempt-002-target --numerical-manifest-sha256 bd34d93e8430ed17b0705725ef27c384c7e8ae4853e20f8363042cfcc493123c --profile-root <same-profile-root> --profile-manifest-sha256 <reviewed-profile-pin> --out <exclusive-target-root>
```

Replay uses `--mode replay --bundle <target-root>/run --manifest-sha256 <run-pin>
--out <new-exclusive-root>`. This card coordinates one measured target; routine
bounded adaptations remain permitted under the user's exploratory mandate,
with a dated amendment, source checkpoint and updated cost bound. No protected
holdout, broad parameter search, paid call or GPU run is included.
