# A validated interval through the period-18 center — 9 October 2026

Behavioral question: can the already proved attracting 18-step training pattern
be followed continuously for a nonzero learning-rate interval, rather than just
certifying isolated points? This first exploratory pilot tests one small interval
through the existing center. Passing would be a local anchored branch, not a
connection to the fold, flip, entire window or useful training outcome.

## Exact object and theorem

Use the exact consumed C1 data/model and native phase order from
`certificates/s2-period18-receipt.json` (SHA256
`f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277`).
Its input SHA is `9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f`;
all four raw anchor files and the unchanged C1 kernel are checked before use.
The actual hidden-rate center is exactly `1254985102288567/17592186044416`;
output rate stays exactly `1848862009810323/2251799813685248`. This differs from
the separately proposed flip-ray exponential family's tiny rate-rounding shift:
a later join to that family needs a proved two-rate bridge or a matching anchor.

For the 18 phase states U=(u0,...,u17), H_j(U,h)=F_h(u_j)-u_(j+1 mod18).
The square shooting system has234 variables. Two bounded Newton steps prepare
a dyadic center; their convergence confers no proof. A midpoint inverse C of
H_U and a dyadic slope s=-C H_h define p(h)=U+s(h-h0). Set exact halfwidth
2^-20 and exact correction radius2^-28; precision256 bits, power cap4096.
On X=p([h0-delta,h0+delta])+[-r,r]^234 evaluate the sparse block Jacobian A
and parameter partial b with Arb, reusing the unchanged analytic C1 objective
derivatives. The actual hidden-rate partial is -gradient in hidden coordinates
and zero in output coordinates. No binary64 derivative decides a gate.

The mean-value enclosure H(p(h),h) subset H(U,h0)+(A*s+b)*[-delta,delta]
keeps the predictor cancellation. With D=I-C*A, require ||D||infty<1 and every
coordinate of -C*[H0+(A*s+b)*[-delta,delta]]+D*[-r,r]^234 strictly inside
[-r,r]. This gives a uniform contraction, unique phase-labelled root in every
parameter slice, nonsingular H_U and continuous (indeed locally smooth) branch.
All proper divisors1,2,3,6,9 must have a separated phase coordinate throughout
the tube. Native monodromy is DF(u17)...DF(u0), with balanced parenthesization
only to reduce interval wrapping; a rigorous infinity-norm power below1 proves
uniform attraction. Existence, primitive period, attraction and anchor identity
are separately retained gates.

At exact h0, all18 phase enclosures of the old certified orbit must lie inside
the new shooting uniqueness domain U+[-r,r]. This identifies the same root
using common-parameter uniqueness, not overlapping boxes. The driver binds
identical model, data, rates, phase ordering and period to the old certificate.
Endpoint Krawczyk images are saved for possible later exact-parameter joins;
they do not themselves prove another segment. No subsequent segment is in this
pilot. Offline replay reconstructs the fixed original inputs, replays the C1
anchor proof and reproduces the full tube result with unchanged source/library
identities. Analytic implementation and Arb remain trusted, not formalized.

## Measurement and resource card

First run only a synthetic resource profile using the same234D shooting algebra,
13D biased tanh map and24 artificial samples. All phase proposals are zero at
rates1/128. It is a fixed point, so primitive18 and anchor gates are expected
false and confer no target claim. Save the full result regardless. A target
requires the exact profile manifest, unchanged bound source files, completed
profile under20 seconds, 4*profile elapsed+2<100 seconds and three times its
saved size below60MiB. This is a conservative engineering estimate; data-dependent
interval/matrix cost and wrapping can still stop the target. Existing center
single-shooting proof took0.407s for a comparable number of one-step derivatives;
the new dense234D algebra is precisely what the synthetic profile measures.

Profile: CPU/BLAS1, nice10, internal20s, external25s+5s, tick cap240, RSS2GiB.
Target: CPU/BLAS1, nice10, internal120s, external135s+5s, same tick/RSS limits.
Worker raw cap64MiB (60MiB ordinary plus4MiB finalization); launcher at most1MiB
additional. No GPU, paid work, new dataset, holdout or global continuation.
Exclusive output roots under
`/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-2026-10-09/`:
`attempt-001-profile` followed, only after the measured gate and source review,
by `attempt-001-target`. No automatic width/radius/precision retry. Any adaptation
gets a dated card and separate immutable attempt; all failures remain available.

```
nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_connection.py \
  --mode profile --out <base>/attempt-001-profile
nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/certify_s2_period18_connection.py \
  --mode target --profile <base>/attempt-001-profile/run \
  --profile-manifest <exact run manifest hash> --out <base>/attempt-001-target
```

Runtime/source/input records precede science. Each Newton proposal and root gate
is fsynced before another stage. The final result, all partial stages, failure,
launcher command/PID/exit, source archive and exact complete inventory are
retained and made read-only. Deadlines are cooperative inside C calls with an
external process-group timeout; kill can prevent final worker output, but outer
inventory retains all files. The launcher cleans only its owned group. No claim
is made from an incomplete or failed result. Synthetic controls cover primitive
and unstable negatives, wrong/missing anchor, actual-rate derivatives, native
product order, invalid controls and corrupt inventories/pins.


## Disclosed storage amendment after profile001

The original synthetic profile took1.992s but saved31,055,672 worker bytes of
verbose matrix JSON, failing the declared3x disk projection. It remains retained;
no target ran. The full certificate is now `certificate.json.gz`, deterministic
gzip level6/mtime0; parsing preserves the exact JSON payload and byte-identical
mathematical replay. A separately hash-bound payload record also gives exact decompressed SHA256 and byte count, checked on every certificate read. Maximum decompressed content256MiB; the64MiB worker disk
cap and60MiB ordinary allowance remain. All model/proof/settings are unchanged.
One amended profile uses `attempt-002-profile`; if the unchanged timing/storage
gates pass, the first target remains `attempt-001-target`, binding profile002.
