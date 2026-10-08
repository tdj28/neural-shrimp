# Prospective S1R72 certificate outcome audit card

8 October 2026 UTC; written before any profile or target outcome. Producer source
`2c00544ca35a5cf4edb28bacc0a80e285e284073`; selected inputs SHA256
`b05c7456ef9db17aec5bd1f72e76b0f5b3be3f5d6050e8bd950c1067888ff443`.
This card authorizes no execution. Root will authorize any saved-outcome audit
separately after the profile/collection evidence exists.

1. Check complete wrapper/profile/nested-C1 inventories, safe paths, hashes,
   source/settings/input identities, launcher PIDs/exits and actual resource
   bounds. Recompute the resource gate; never infer a proof from profile success.
   Preserve failed/incomplete attempts and all original settings.
2. Independently decode original selected record arrays and compare every exact
   input rational with its binary64 value using `Fraction`, including 72D phase 0,
   72x8 data, targets, scales, ridge and both saved rates. Regenerate the frozen
   EXP-028 dataset/initialization identity; do not evaluate protected data.
3. If collection passes, replay the unchanged fail-closed C1 verifier once and
   compare every regenerated enclosure/verdict. This reuses the reviewed Arb
   implementation, not an independent proof kernel. Check primitive period 2,
   dimension 72, full-state uniqueness/attraction flags and fixed settings
   256 bits, radius 2^-80, Newton 5, maximum power 4096. No correction/search outside
   the verifier's exact replay, alternate phase, relaxed threshold or retry.
4. Before calling this an upgrade of the selected numerical state, compute an
   independent exact-rational proximity bound. For each returned box coordinate,
   decode serialized Arb midpoint m and radius r from their integer mantissa/
   exponent pairs into Fractions. Let s be the exact saved binary64 phase 0 and
   calculate `D=max_i(abs(m_i-s_i)+r_i)/max(1,max_i(abs(s_i)))`.
   Require finite valid dyadic endpoints, nonnegative radii and **D<=1e-7**.
   This bounds the farthest coordinate of the entire box from s. A failure
   preserves any valid exact-real map certificate but blocks a selected-state
   proximity/association claim. A pass establishes proximity only, not unique
   branch ancestry, interval continuation or inclusion of s itself in the box.
5. Report proof outcome, proximity and resources separately, with evidence rung
   and theorem/library assumptions. No success claim for geometry, circles,
   chaos, floating-point training or usefulness; a failed proof is not disproof.

Proposed audit envelope: CPU/BLAS 1, nice 10, no GPU, internal 100s, external 120s
plus 5s kill grace, 32MiB new raw audit cap. Confirm against the measured profile
and completed collection cost before root authorization; this is not a measured
runtime guarantee. Use an exclusive new audit directory under
`/data2/neural-shrimp-data/audits/s1r-period2/`, retain source/input/log hashes,
all exact proximity fractions and failure/termination records. Commit any audit
implementation and focused synthetic controls before its one target replay.
