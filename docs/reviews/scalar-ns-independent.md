# Independent scalar Neimark–Sacker review — 8 October 2026 UTC

TLDR: the selected exact scalar factorization model has a rigorously enclosed
local transition from an attracting period-two state to nearby attracting
oscillatory curves. This establishes a small-model mechanism. It establishes
neither a trained-network benefit nor membership in a particular global shrimp
structure. The curve conclusion is qualitative and local, without a certified
numerical detuning interval or an irrational-rotation claim.

**Verdict:** no mathematical blocker for the stated local exact-real theorem.
The actual original artifact and seed binding pass independent checks. One
nonblocking artifact limitation is documented below: the frozen verifier does
not authenticate arbitrary rewritten provenance metadata. Original source and
raw evidence remain unchanged; this review is a separate source-bound record.

## Object and evidence

Producer execution source `0652d49529265565078b63360ec76bd9dd5f21a8`;
result-document commit `edc79ac2276627820c1e3df25479c303d5c6658f`.
Independent reviewer execution source
`47fd3bd97e8c35272e2eb8b92ce0c41003c82162`, on
`codex/scalar-ns-independent-review`.

| Component | Evidence rung | Artifact | Assumptions and limits |
|---|---:|---|---|
| Producer augmented root and NS gates | 3 | Original `certificate.json` and `receipt.json`, hashes in the review receipt | Exact dyadic coefficients; unknown real critical eta; trusted Arb/FLINT and theorems |
| Independent proof recomputation | 3 | Review `review.json`, `independent` object | Complete rational polynomial expansion, formal derivatives, 512-bit Arb/Acb; same arithmetic library |
| Byte-identical official replay | Audit of rung 3 artifact | `official_replay` | Uses frozen producer implementation; not independent mathematical evidence |
| Synthetic tests and tamper controls | Software checks | Review manifest and preserved mutation directories | Twelve synthetic tests pass; provenance-only mutation is accepted as documented |

Compact authoritative review receipt:
`certificates/scalar-ns-independent-review.json`.
Original raw directory:
`/data2/neural-shrimp-data/explore/certificates/scalar-ns-2026-10-07/attempt-001`.
Review raw directory:
`/data2/neural-shrimp-data/audits/scalar-ns-independent/attempt-001`.
All recursive review artifact sizes and hashes were checked against its manifest.
The source-committed run card precedes the target audit; one CPU thread,
nice 10, PID 4086622, 0.230019 seconds, maximum RSS 49,320 KiB, no GPU or paid
compute. The process exited. Raw evidence is retained read-only.

The exact map is gradient descent for
`L(u,v)=(uv-d)^2/2 + l*(u^2+v^2)/2`, with coordinate rates `eta` and `4*eta`.
Here `d=5404319552844595/4503599627370496` and
`l=5404319552844595/18014398509481984`: the exact dyadic encodings of the
selected binary64 coefficients, not the decimal rationals 6/5 and 3/10.
The independently checked selected seed SHA256 is
`cbef5ea46ee1a0253dccbe93fbc4e36523a47f9363533861c8bce09e4381215e`.
All three exact proposal coordinates and all fixed coefficients were compared
with that seed's binary64 exports. Outcome-informed selection remains explicit.

## Mathematical audit

1. **Augmented existence and uniqueness.** The system is
   `H(u,v,eta)=(F_eta^2(u,v)-(u,v), det D(F_eta^2)(u,v)-1)`.
   Producer `rigorous_scalar_ns.py:119–133` bounds its entire Jacobian over
   the box, using jet derivatives of the complete polynomial composition.
   With the exact midpoint preconditioner R, the map `z-R*H(z)` maps the box
   strictly into itself and has infinity-norm derivative bound below one.
   Banach therefore gives one fixed point. The same norm bound implies R is
   nonsingular (otherwise `R*DH` could not be invertible), so fixed points
   are exactly roots of H. This proves uniqueness within this box, not global
   uniqueness. Smoothness and convexity hold for this polynomial and box.

   The actual producer coordinate radius is **2^-40 + 2^-69**, not exactly
   2^-40. The outward-rounded displacement balls in the inclusion use the
   same radius. This rounding is conservative. The independent box shares
   the exact center and has radius **2^-48 + 2^-77**, exactly 1/256 of that
   radius, so its root is the same locally unique root.

2. **Independent derivatives and inclusion.** The reviewer expands each
   component of F squared into 33 rational polynomial terms and formally
   differentiates them, without calling producer Jet, augmented, inclusion,
   or coefficient routines. At 512 bits, a separately evaluated whole-box
   Jacobian gives a contraction bound below `8.565e-11` and strict inclusion
   in every coordinate. The full polynomial representation has digest
   `abe11ce94b10789abd27692de6a8e09d9f05040fb0b2b7ce8da310e1e442c118`.
   This shares the producer center as a proposal, not its proof assertions.

3. **Primitivity and complete spectrum.** Both independent coordinate
   differences `F(x)-x` exclude zero (near 0.12523 and 0.93632). A root of
   `F^2(x)-x` therefore has primitive period two. At this root the augmented
   equation gives determinant exactly one. The enclosed trace is near
   -0.79587625252, and the discriminant gives a simple nonreal pair. Since
   this is a real planar map, these are the entire spectrum. Trace intervals
   exclude -2, -1, 0 and 2, excluding the resonances of orders one through
   four. No incomplete eigensolver coverage is involved.

4. **Continuation and transversality.** `A-I` is invertible, so the implicit
   function theorem supplies a local smooth period-two branch. Both paths
   compute `x'=-(A-I)^(-1)*partial_eta G` and then the total derivative of
   `D=det A` along that branch. The independent determinant slope enclosure
   is `[6.705972714 +/- 9.40e-10]`; the **modulus** slope is half of it at
   D=1, `[3.352986357 +/- 4.70e-10]`. These printed balls conservatively
   summarize the exact saved enclosures. The sign is strictly positive.
   This is a pointwise derivative plus an implicit local branch, not a
   constructed tube over any specified parameter interval.

5. **Eigenvectors and nonlinear coefficient.** Producer lines 173–181 set
   `q=(A01,mu-A00)` and normalize `q* q=1`; the first row of
   `[q,conj(q)]^(-1)` is `p*`, not the column p. Thus `p* q=1`,
   `p* conj(q)=0`, and the pairing convention is correct. The reviewer uses
   a separate transpose eigenvector `(A10,conj(mu)-A00)` and normalizes the
   Hermitian pairing directly. The analytic eigenvector identities and all
   denominators were checked; residual balls contain zero as a diagnostic.

   Producer lines 188–193 implement the tensor/resolvent formula and the
   planar scalar formula, with Taylor factors 1/2 and 1/6 treated consistently.
   The reviewer instead eliminates quadratic monomials by
   `H(z)=z+h20*z^2+h11*z*conj(z)+h02*conj(z)^2`, where
   `h20=(g20/2)/(mu^2-mu)`, `h11=g11/(mu*conj(mu)-mu)` and
   `h02=(g02/2)/(conj(mu)^2-mu)`. Expanding `f o H = H o N`, with N having
   no quadratic term, gives resonant cubic coefficient
   `c=g21/2+g20*h11+g11*(conj(h11)+h20)+g02*conj(h02)`.
   There is no derivative-of-H times a quadratic term of N, since that term
   is zero. The independently enclosed `Re(conj(mu)*c)` is
   `[-1.7002020 +/- 5.08e-8]`, strictly negative and overlapping both
   producer formulas. Formula equivalence is needed only at the enclosed
   exact root, where D=1; evaluating enclosing expressions off that root
   does not assume the whole box has unit determinant.

## Local theorem and exact scope

The verified smooth planar map, simple unit pair, exclusion of resonances
through order four, nonzero radial slope, and negative cubic radial coefficient
meet the standard supercritical Neimark–Sacker hypotheses. Therefore, for all
sufficiently small positive `eta-eta*`, the two-step map has a nearby attracting
invariant closed curve and the corresponding fixed point loses stability.
This applies the local theorem, not a directly computed curve enclosure.
See Kuznetsov and Sacker,
[Neimark–Sacker bifurcation, Scholarpedia](https://www.scholarpedia.org/article/Neimark-Sacker_bifurcation).

For the one-step map F, primitivity separates the two phases. Since
`det D(F^2)=1` at the cycle, both one-step derivatives there are nonsingular.
F locally maps the curve near one phase to a second, disjoint curve near its
companion phase; F exchanges them and their union is invariant. This is the
local two-component consequence. It supplies neither an explicit positive
parameter range nor a guarantee of irrational rotation or quasiperiodicity.
No global ancestry, global uniqueness, shrimp association, higher-dimensional
network claim, implementation-rounding claim, or formal proof-assistant
verification follows. Arb/Acb/FLINT and the cited mathematical theorems remain
trusted components.

## Artifact finding and reproduction

At frozen `tools/certify_scalar_ns.py:95–122`, offline verification checks the
inventory, hashes, source contents, internal runtime bindings, byte-identical
recomputed result and quoted summary. Independent tamper controls confirm that
rehashed changes to a proof gate or coefficient are rejected, as is replacing
a dyadic coefficient by 6/5 while retaining the old proof. An unchanged relocated
bundle passes. The preserved review mutation `rehash_changed_provenance` sets
`inputs.provenance.seed_sha256` to 64 zeros and consistently rehashes the input,
certificate and receipt; **it passes**. `compute` does not inspect provenance,
and `verify` does not call the seed-export validation used by `run`.

Consequently the offline proof checker authenticates the recomputed mathematics
for supplied exact inputs and internally consistent artifacts, not the truth
of all arbitrary metadata after complete bundle rewriting. This finding does
not invalidate the original artifact: this independent review separately binds
its actual receipt and files to the selected seed and exact exported inputs.
The pinned review receipt is the authority for those checks. Frozen producer
source, raw evidence and historical `independent review pending` fields were
preserved; this separate review records the completed audit.

To repeat the official mathematical replay, use the frozen producer sources
and run `python tools/certify_scalar_ns.py verify <original-receipt-path>`.
To repeat the complete independent audit, archive reviewer execution commit
`47fd3bd97e8c35272e2eb8b92ce0c41003c82162` into an isolated checkout and run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  timeout --signal=TERM --kill-after=5s 120s nice -n 10 \
  /data2/neural-shrimp/.venv/bin/python tools/review_scalar_ns.py \
  --receipt /data2/neural-shrimp-data/explore/certificates/scalar-ns-2026-10-07/attempt-001/receipt.json \
  --output /absolute/path/to/a/new/exclusive/review-attempt
```

Read `independent.passed`, the gate list and each mutation outcome; the review
runner reports findings rather than treating every expected mutation failure
as an overall process failure. Local tests were
`pytest -q tests/test_rigorous_scalar_ns.py tests/test_scalar_ns_independent_review.py`:
12 passed in 0.13 seconds. A cache-write warning in the restricted review shell
did not affect test execution. No new target, trajectory or continuation was run.
