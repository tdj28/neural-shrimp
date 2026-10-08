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

## Bound audit implementation and exact run card — 8 October2026 UTC

Implementation prepared without opening any collection target artifact. Only
the resource-only profile's launcher schema/receipt was read for its metadata
contract; root supplied the collection source/path/manifest identity and elapsed
time. This is disclosure of timing, not an inspected proof or proximity outcome.

`tools/audit_s1r72_certificate.py` now binds the following immutable identities:
profile execution750233fc03b0a24dbdba32a2e49435931495fdab, manifest
850a055d40541e1ebbaa118e773b41f036d8ae3b5f6a4691044ec19312ea073e;
collection executiondc2732da3b76dcf85c4c86051d4e096f5a3c903e, manifest
7a5f43508b4a24dc22b809eec8b22834dfa40a603a3f7b28af20d0d1988ad21e.
Both source snapshots must match all11 files reviewed at2c00544c; both launcher
receipts must identify source review426ce4fe. The profile/collection directories
are respectively `profile-001` and `attempt-001` below
`/data2/neural-shrimp-data/certificates/s1r-period2`; sibling suffixes are
`.launch-start.json`, `.launcher.json`, `.launcher.log`. No future document HEAD
supplies numerical inputs. Original saved-node/data decoding is independent;
the frozen dataset generator and the existing mathematical checker are shared
primitives, explicitly recorded. All exact proof enclosures are replayed once.

The paired resource gate is recomputed, every nested C1/outer input and receipt
cross-bound, and all actual source bytes checked against recorded Git commits.
The exact whole-box1e-7 proximity gate remains unchanged. Invalid/failed proof
replay halts; proximity failure retains a valid map proof while blocking the
selected-state proximity claim. SIGTERM/SIGALRM exceptions retain completed
fsynced stages, failure metadata and a manifest when the filesystem remains
writable; SIGKILL or filesystem failure cannot promise finalization.

Exclusive audit output after code review and root authorization:
`/data2/neural-shrimp-data/audits/s1r-period2/attempt-001`.
Launcher prefix:
`/data2/neural-shrimp-data/logs/s1r-period2-audit-attempt-001`.
Intended command, no authorization implied:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src timeout --signal=TERM --kill-after=5s 120s
nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/audit_s1r72_certificate.py
--out /data2/neural-shrimp-data/audits/s1r-period2/attempt-001`.
Record exact audit execution commit, review, owned PID and exit status, and hash
all logs. Internal100s,32MiB cap; no retry, extra proof setting, continuation,
trajectory or paid resource. The quoted producer process elapsed3.7092s leaves
headroom within this proposed audit cap; authorization still follows source review.
