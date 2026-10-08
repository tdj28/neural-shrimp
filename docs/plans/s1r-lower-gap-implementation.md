# Six-state lower-gap derivative diagnostic: implementation card

8 October 2026 UTC. Implements design0f28cc5a5f3ea73e5270f8c0033419b09cf42467.
Root authorized source and synthetic controls only. **No target/profile or
real-model evaluation, correction, continuation or launch is authorized by
this card.** Existing producer and independent analytic source files remain
unchanged. An independent source review and separate root launch instruction
must precede the single prospective run below.

## Exact arithmetic scope

The six roles, historical input hashes and exact binary64 controls/rates are
those in docs/plans/s1r-lower-gap-inputs.json. Full collection inventory,
selected payload/array identities, historical execution/reviewed source bytes,
model/data identities, and the previously audited independent analytic helper
are checked. Source inventory covers all indirect package imports and the
unaltered analytic helper. A clean committed checkout is required before
creating output. No target arrays were evaluated while writing this source.

For each theta, the adapter calls the analytic loss/gradient/Hessian helper
once at theta and once at F(theta), reusing each returned gradient/Hessian to
assemble its map and Jacobian. Thus all six states require exactly12 map
values and12 state Jacobians, with no duplicate derivative, finite-difference,
HVP, eigensolver, Newton correction or trajectory. Retained outputs include
both raw analytic returns, both phase maps/Jacobians and partial_h F, followed
by R=F²(theta)−theta, M=J1 J0, A=M−I and
b=J1 partial_h F(theta)+partial_h F(F(theta)). Hidden-log-rate derivatives use
−ln(10) P_hidden gradient with the saved rates; output rate stays fixed. This
is the underlying smooth rate family through the stored controls, not a
claim about differentiating rounded pow output.

The four qualified endpoint phases retain the existing normalized closure/
transition denominator1e-9+1e-8*max(|x|,|y|) and primitive separation1e-6.
Their computed monodromy is compared with the saved native-phase matrix at
maximum absolute error2e-11, the previously used independent analytic
agreement tolerance. The exact F(theta) and the stored opposite phase differ
within closure tolerance; their tiny numerical difference is not silently
identified as exact equality. A consistency failure stops the interpretation.
The final failed iterate's reconstructed R must agree with stored R within
2e-11 absolute per component. Returned matrices are retained before checks.

## Explicit scaling, SVDs and abstention rules

Use one fixed D=diag(max(1,|theta_last_native_phase0|)) for every role and
sigma_h=0.000125. No outcome-dependent scaling choice is permitted. Raw
matrices are A and B=[A,b]. Scaled matrices are

    A_s = D^(-1) A D,
    B_s = D^(-1) B C,     C=diag(D,sigma_h),
    R_s = D^(-1) R.

The row scaling is D^(-1); the state-column scaling is D; the augmented
column scaling is C. A unit right null vector v_s of B_s converts to original
units as C v_s, then is normalized in original Euclidean norm. The original
and scaled residuals of that null relation are both reported. The same
transformation applies to the carried secant for orientation: s_s=C^(-1)s.
Left singular vectors/couplings are reported in their declared coordinates;
a scaled left covector would convert by D^(-T), not by D. We do not call a
converted scaled singular pair an original-coordinate singular pair.

Every matrix uses np.linalg.svd(full_matrices=True). Retain matrix,U,s,Vh,
including the complete73x73 right basis V=Vh.T for B and B_s. Four SVDs per
role give exactly24; no spectral-norm/condition routine causes a hidden SVD.
Residual and reconstruction denominators use max(1,||matrix||_F), with a
fixed1e-10 bound on both singular equations, reconstruction and orthogonality.
For each singular spectrum set scale=max(1,s_max): low=1e-12*scale and
high=1e-10*scale. A singular value in(low,high] gives ambiguous numerical
rank; full row rank is resolved only when every singular value exceeds high.
Otherwise report numerical rank deficiency, not exact rank proof.

A unique null tangent is reported only for full-row-rank B/B_s with residuals
<=1e-10. Ambiguous rank or larger numerical nullity causes abstention. The
softest A direction is isolated for descriptive coupling only when the next
singular value exceeds high and s_min<=0.1*s_next. No soft mode, rank or
coupling produces a fold verdict. Coupling without isolation remains
explicitly unqualified. Competing modes are not tracked by nearest angle.

Secants are the full predecessor-to-last state/control differences, matched
using the accepted edge's native/carried phase bits. Each phase receives its
corresponding secant; both failed states inherit the failed edge's carried
phase. Choose tangent sign by positive dot product with that secant in the
declared coordinates. Zero secant or absolute cosine<=1e-8 causes orientation
abstention. Endpoint phase transport uses already saved J0 and partial_h F;
no extra model call. The failed states' tangents describe a **local residual
level set at a nonclosed state**, never an orbit branch. Qualified endpoints
support only a local numerical branch-tangent interpretation, not a certified
branch or fold.

## Saved linear solve and line-search diagnostics

At the final stalled state retain Ap_saved+R and Ap_saved+R_saved. Report the
absolute linear residual, q=||Ap+R||_2/||R||_2, and Frobenius normwise backward
error ||Ap+R||_2/(||A||_F||p||_2+||R||_2) separately. Abstain from q when
||R||_2<=1e-12 max(1,||theta||_2), and from a zero-denominator backward error.
The prospective material-forcing threshold is1e-6 and backward threshold1e-12.
If ||R−R_saved||_2 exceeds1e-6||R||_2, label the relative comparison arithmetic-
resolution-limited before attributing any failure to the stored solve. These
are diagnostic interpretation thresholds; they do not relax the original
Newton stopping or acceptance gates and authorize no update.

Reduce the13 saved residuals using exactly the original unscaled infinity-
norm strict-decrease rule; verify byte-exact saved trial state=theta+2^-k p,
call473/iteration3/order and saved acceptance bits. No trial is reevaluated.
An accurate local equation with failed finite-step trials leaves predictor,
nonlinearity and globalization explanations open. A discrepancy identifies
a question about the equation/arithmetic, not its unique cause.

## Resource/durability contract and exact prospective command

CPU/BLAS1,nice>=10; internal15s, GNU timeout20s TERM+5s KILL, supervisor25s.
A one-second process reap allowance is bookkeeping only. All operation
attempts consume fixed12/12/24 caps; failed calls are not retried. SIGTERM is
sticky/cooperative at durable boundaries. The separate launcher copies the
independently reviewed owned-group wrapper from CIFAR source8aa146fd, changes
only its target/schema/time/storage settings and completion fields, and never
changes that frozen source. It preserves inherited-pipe/closed-pipe guards,
stdout128KiB, source/argv/resources/PIDs/exit/full inventories and read-only
finalization. A supplementary2GiB child-group RSS monitor is not kernel-hard;
launcher interpreter memory is additional. Foreign processes are untouched.

Raw aggregate8MiB =7MiB scientific Store +1MiB launcher. Store's existing
quarter reserve is unavailable until finalization. Raw arrays and SVD returns
are fsynced before downstream checks. Incomplete attempts retain completed
roles, raw returns, operation attempts/returns, pending stage and terminal
manifest where possible. Missing or malformed scientific manifests cannot
produce launcher success. SIGKILL may leave partial files; preserve them.
No retry, alternate state, changed scale/tolerance or expanded role set.

Root must check current resources, record launch authorization and resolve
both full SHA placeholders before one invocation from the clean reviewed tree:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_s1r_lower_gap.py \
  --out /data2/neural-shrimp-data/explore/s1r-lower-gap/attempt-001 \
  --launcher /data2/neural-shrimp-data/explore/s1r-lower-gap/attempt-001-launcher \
  --source-commit EXECUTION_HEAD_FULL_SHA \
  --review-commit INDEPENDENT_REVIEW_FULL_SHA
```

Both directories must be absent; no target directory was created here. Bind
actual execution HEAD and review separately. A distinct prospective saved-
outcome audit must be frozen before viewing new outcomes. It can recompute
matrix products, decomposition residuals, scaling, null conversion,
orientation, saved-step forcing and all fixed decisions from the retained
arrays; it must label reuse of the independent analytic model primitive.
No current result proves a fold, absence of a branch, historical family
association, or training benefit. Root owns manuscript decisions.
