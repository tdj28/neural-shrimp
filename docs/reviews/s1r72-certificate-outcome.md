# S1R72 saved-certificate outcome audit — 8 October 2026 UTC

**Pass:** the single authorized replay verifies an attracting primitive period-two
orbit of the frozen **72-dimensional exact-real GD map**, with existence and local
uniqueness in the returned phase box. The independently computed whole-box
proximity to the selected numerical node also passes. These are separate claims;
neither establishes branch identity or boundary geometry.

The authority is the [compact outcome receipt](s1r72-certificate-outcome-receipt.json).
Original certificate and creation-time receipts retain their pending-review text
and unchanged hashes. This companion records the completed review without
rewriting historical evidence.

## Mathematical result

At fixed settings 256 bits, radius 2^-80, five Newton proposal steps and maximum
power 4096, all Krawczyk inclusions pass and the contraction upper bound is
below 2.148e-19. Proper divisor one is excluded, proving primitive period two.
For the interval monodromy M, the stored exact dyadic upper bound gives
`||M^2048||_infinity <= 0.55545265800661285858... < 1`, proving strict local
attraction. The receipt retains the exact mantissa/exponent and Fraction values;
these decimal displays are summaries of those bounds.

This is an independent audit using the same reviewed C1 mathematical checker,
not a second proof kernel. It relies on the Krawczyk/Banach argument, correct
model derivatives, interval divisor exclusion, the matrix-power attraction
criterion and python-flint 0.9.0 / FLINT 3.6.0 outward interval arithmetic.
The map uses the supplied binary64 constants as exact dyadic real numbers;
the theorem does not certify floating-point training trajectories.

## Selected-state proximity

For node0004 phase0, independent exact-Fraction decoding gives
`D = max_i(|m_i-s_i|+r_i) / max(1,||s||_infinity)`
`= 94208937568474953061232063439100508234989438849623364813720369/103365902932817117988242767686465981634214315413410022400748520248600032182272`,
approximately **9.114121281338416e-16**, below the prospectively fixed **1e-7** cap.
All 72 coordinate calculations and exact normalization are saved in stage03.
This bounds the entire certified box relative to the selected binary64 phase;
it does not assert that the seed lies in the box, prove unique branch ancestry,
or provide interval continuation. The dataset, model, both rates and phase were
bound independently to the original branch artifact and frozen EXP-028 generator.

## Audit evidence and scope

Audit source `be4d36cae1c74760f3c5ed5ee75ce4d2de014686` is a docs-only authorization successor
to root-reviewed auditor `b773e64b06d66f9c65dfcaf871221a0637b42b48`.
All bound executable/card bytes remain unchanged. Root reported 60 focused
controls passing before authorizing this one audit; author controls had 68 passes.
Profile, collection, nested C1 and original selected-node inventories, source
commits, exact inputs, settings, launcher PIDs/exits, resource gates and enclosure
replay all pass. There was exactly one checker replay and no retry, new trajectory,
alternate setting, GPU use, paid call or protected-data use.

The audit finished in 3.798106s internally and
4.208545s externally, peak RSS 201,688 KiB,
CPU/BLAS 1, nice 10. Owned audit PID 128103 and timeout PID 128102
exited with status 0. Fresh host inspection recorded an unrelated busy GPU
process, which was untouched. The internal 100s / external 120s + 5s limits and
32MiB new-output cap were respected; all audit and launcher evidence totals
73,985 bytes, now read-only.

Raw audit: `/data2/neural-shrimp-data/audits/s1r-period2/attempt-001`. Manifest SHA-256:
`81c708ca8fab815483136d72b13438fa7b6846ae34335e6808a334bad4f67281`.
Collection manifest: `7a5f43508b4a24dc22b809eec8b22834dfa40a603a3f7b28af20d0d1988ad21e`.
Certificate SHA-256: `f01fb87f3516dec265f3e29ec5e3a5b0a9d71dd716eddf0ec036aa3ae5f5c329`.
The receipt binds every stage and all four launcher companion files, including
the exact command, source, review authorization, host inventory, PID and exit.

**Historical checkout required:** later integration ab06b511 adds an observer to
gdmaps after proof execution. Replay must use the audit execution commit above
(or the complete identical bound source inventory). It is not claimed that a
future/current integration checkout has identical gdmaps bytes; do not relax the
source guard. Repository pushes protect these compact records and source, while
the data-root evidence requires its separately tracked storage backup.

No parameter-neighborhood, branch-ancestry, shrimp boundary, invariant-circle,
Neimark-Sacker, chaos, training/inference utility or geometry-specific utility
claim follows from this result. It extends the orbit certificate dimension;
it does not settle those scientific questions.
