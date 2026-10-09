# Ray13 numerical flip/period36 pilot — 9 October 2026

Question: does the full period18 orbit carried from EXP028 row111 reach the
selected historical ray13 flip, and is there a nearby primitive36 branch?
This is local floating-point exploration, not a branch-connection or
bifurcation proof. Only previously consumed S2 data are used. No holdout,
new data, GPU or paid service. The period18 fold proof stays unchanged.

Use the hash-pinned consumed C1 network/x/y, historical certificate row111,
and A2 ray13. Retain the original log center, scales, unit direction, and
binary64-rounded products d=scale*direction. The interval-proof worker and
this proposal use the exact family
`eta_i(s)=exp(log(10)*(ell_i+d_i*s))`, with ell,d exact binary64 dyadics and
both coordinates active. The numerical implementation uses scalar powers
`10.0**float(ell_i+d_i*s)`. Those are proposals for the exact-real family;
its s=0 exponential constants need not equal the earlier dyadic interior
certificate rates. A later interval connection must bridge that difference.

Carry all18 phases over33 parameters from s=0 to0.999 times the saved ray13
boundary. Each point uses damped Newton with the analytic return-map Jacobian,
final relative correction<=1e-10, normalized physical closure<=1e-8, every
proper-divisor separation>=1e-6, and complete-spectrum residuals<1e-9.
Require attraction along this prefix. Every edge carries the same native
phase and retains a reverse corrector (error<1e-7); full native orbit movement
must be<0.1. These are numerical witnesses, not validated interval tubes.

Refine `(F^18(x)-x,(M+I)q,(q^Tq-1)/2)=0` by one bounded augmented solve.
Require residual<1e-8, primitive18, simple−1 (error<1e-6, next−1 gap>1e-4),
qualified left/right residuals and other modes strictly attracting with1e-7
margin. Save the exact dyadic theta/s/q proposal and exact family/model for
the separate interval proof. Source and data hashes accompany every result.

The joint analytic evaluator carries `R_s` through BOTH rate derivatives.
Compute `x'=-(M-I)^-1 R_s` and compare the total tangent multiplier derivative
with corrected-branch central differences at2^-12,2^-14,2^-16,2^-18. Last
slopes must be nonzero/same-sign and agree within relative1e-3, including the
tangent-vs-corrected-branch check. Also retain an optional actual adjacent
native-phase check using `q1=Jq/||Jq||` and `p1=||Jq|| J^-T p`; failure of that
extra diagnostic is explicit. The numerical pilot does not estimate the cubic
coefficient; the interval worker is implementing its higher derivatives.

Locate period36 using three fixed small positive amplitudes
`a=2^-12,2^-11,2^-10` times `max(1,||x_flip||inf)`. Solve the14D system
`(F^36(x)-x, p^T(x-F^18(x))/2-a)=0` with its analytic Jacobian, so collapse
onto the period18 parent cannot satisfy the amplitude constraint. Subsequent
seeds scale the last qualified displacement linearly and parameter displacement
quadratically. Retain all three attempts, including failures. Qualify physical
closure, all proper divisors1,2,3,4,6,9,12,18, both18-separated native starts,
noncollapse>=1e-6, nearby state<0.05 and |s-s_flip|<0.01, and the corrected18
parent at the same parameter. Report stability separately: primitive periodic
qualification does not imply attraction. No doubled branch is guaranteed.

Budget: CPU/BLAS1, nice10;100s internal,115s external TERM+5s KILL;
40,000 joint map/Jacobian/parameter-derivative calls (80,000 direct gradient
and Hessian calls),32MiB raw plus1MiB launcher; monitored2GiB RSS. Each returned
evaluation is durable before the next budget check. Aggregate writes include
all logs and duplicate summaries, with28MiB ordinary allowance and4MiB failure/
finalization reserve. An oversized duplicate is marked unwritten with its hash;
an overflowing returned journal record is retained in the reserve. No foreign
process is touched. All attempts become read-only with full source snapshots,
command/PID/time/exit metadata and SHA256 manifests.

The preceding numerical fold completed in0.73s; this heavier pilot still has
large wall headroom. Optional tiny resource profile: exactly five synthetic
13D/24-sample/18-step evaluations, no target arrays, same supervisor. Execute
from committed code/card in exclusive
`/data2/neural-shrimp-data/explore/s2-period18-flip/attempt-001-profile`
(`--profile`) and `attempt-001-target` with
`/data2/neural-shrimp/.venv/bin/python tools/explore_s2_period18_flip.py --out …`.
If a retained failure warrants a smaller continuation step or changed branch
switching method, disclose the finding and commit an amendment before a new
attempt; do not erase failures or quietly loosen qualification gates.
