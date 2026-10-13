# Independent review of mixed-precision phase-domain helper

Date: 2026-10-13. **Approved for its pure domain-containment scope.** No model,
checkpoint or target artifact evaluation was performed by the reviewer. This
review does not approve or report a derivative or orbit certificate.

## Reviewed sources and independent tests

| File | SHA-256 |
| --- | --- |
| `phase_domain.py` | `a3856de84393068788c4cf4dc42fa6015da45b7912b56db995cc6b12922f8d8d` |
| `tests/test_codex_llm_phase_domain.py` | `b57be2e63f858c06872079690f5c40e2d5a2f5e20618093579a8a8fb0eb85f3c` |
| `mixed-precision-domain.md` | `0343aa0ec8b7795aac9a12fe3e74cfc1bcac46d105096d5540208556753ecf15` |

Independent CPU1 execution of
`python -m pytest -q -p no:cacheprovider tests/test_codex_llm_phase_domain.py`
passed **35 tests in 0.13 seconds**, exit status zero. OpenBLAS, OMP and MKL
were restricted to one thread and bytecode writes disabled.

An additional deterministic synthetic audit used 8 phases and 12 coordinates,
with positive/negative dyadic centers of 65, 151, 1031 and 2050 significant bits
and phase radii `2^(-2500-i)`. At target precisions 16, 64 and 1024 bits, **288**
independent exact-`fmpq` endpoint checks passed. This uses a different exact
rational implementation from the helper's `Fraction` comparisons. The audit
also confirmed midpoint-only input rejection, actual-ball hash tamper refusal,
precision-context restoration and the distinct shooting/domain hash formats.

## Mathematical correctness

For the original exact center `c`, chosen lower-precision midpoint `m`, and
positive nominal radius `r`, a radius at least `r+|c-m|` contains every point
of `[c-r,c+r]`. The implementation computes the center error and required
radius as exact dyadic rationals, then bounds the radius outward in Arb.
It does not silently replace the high-precision nominal center by `m`.

The receiver is checked independently of that construction: its saved exact
midpoint/radius endpoints must satisfy both endpoint inequalities and the
radius/error inequality. These non-strict containment comparisons are correct;
this is domain coverage, not the later strict shooting inclusion. Every phase
and every coordinate is retained through explicit expected period/dimension
checks. Nominal centers/radii reject uncertainty, floats and booleans; actual
receiver balls must be finite Arb objects.

Decoding is permitted to widen a magnitude radius. `_decode` proves exact
endpoint non-narrowing and the final containment calculation uses the actual
receiver ball. This correctly handles the one-ulp outward-radius behavior
observed in the point003 audit. It never depends on byte equality between
encoded and reconstructed radii.

## Binding and integration requirements

The externally supplied expected artifact hash prevents an altered artifact
from authorizing itself. Verification recomputes domain, encoded-ball,
actual-ball and witness bindings and every endpoint witness. It checks the
Arb backend version and declared target precision. The helper-source hash is
provenance inside the artifact; the verifier's fresh containment calculation,
not that declaration alone, establishes the domain fact.

The following remain explicit integration requirements:

- Use the returned live balls, including their radii. If they are reconstructed
  again or replaced by wider balls, recheck and bind the **actual** receiver
  balls with `containment_witnesses`; do not keep a stale input hash.
- Bind the identical exact nominal center/radius data to both this helper and
  `shooting_gates`. Their domain hashes use different serializations and are
  **not interchangeable**. Verify the exact data and record both hashes.
- The decoder restores its caller's precision. A later evaluator must select
  its declared precision explicitly and use the corresponding same-map,
  same-parameter prefix cache. A 1024-bit evaluation uses the properly bound
  1024-bit prefix, not a cache relabeled by dtype or precision metadata.
- Establish full-box derivative coverage for all actual input balls and every
  state direction. This helper supplies input-domain containment only; point
  JVPs, metadata or narrower point uncertainty do not discharge that obligation.

The accompanying note states these boundaries accurately, including that
mixed 1024-bit derivative evaluation and 2048-bit nominal centers/residuals can
be mathematically consistent only when the domain and evaluator obligations
are met. No correctness blocker was found, and no performance or certificate
claim follows from this helper review.
