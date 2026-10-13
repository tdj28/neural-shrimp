# Independent review of next certificate setup pilots

Date: 13 October 2026 UTC. Reviewer: Codex inference subagent. No checkpoint,
target model or target-evaluation call was used in this review. No job was
launched. This review is about bounded exploratory setup, not a certificate.

## 2048-bit prefix / point setup and launcher

The new `pilot.py --skip-box` path preserves the original numerical proposal,
label checks, full fresh Arb prefix, and full one-step Arb point computation. It
records `skip_box` in the input artifact, then returns with status
`point_setup_completed`, `certificate_passed: false`, and an explicit reason
that the state-box diagnostic was omitted. The return is before box construction
and box evaluation. This is a disclosed point-only experiment; it does not turn
the pilot002 box failure into a success or remove that retained failure.

The 2048-bit path constructs its own `BallQwen` and calls its prefix computation
at that precision. It imports no prefix from pilot002. Existing numerical
proposals remain binary64; this path measures the accuracy/cost of enclosing
the exact-real map at one such dyadic input. It does not yet refine the centre
at 2048-bit accuracy. A subsequent refinement must retain newly computed centres
as exact high-precision dyadics rather than write them back to binary64.

The launcher now admits drivers `pilot`, `signed_pilot`, and `center_seed`.
Thread counts must lie in 1–4, and only `center_seed` may request more than one.
`--skip-box` is exclusive to `pilot`; an explicit committed JSON plan is
exclusive to `signed_pilot`. Each command dispatch passes only its applicable
arguments. Manifest thread count and BLAS/OpenMP/NumExpr environment agree with
the admitted count. The original and signed Arb drivers remain single-threaded.
The 18 GiB address-space limit, maximum 3600-second watchdog, niceness, source
snapshot, private-origin ancestry check, exclusive attempt directory, owned
process-group termination and final artifact receipt remain intact.

Synthetic admission tests intercepted the very first Git call, so no repository
operation, file creation or process launch occurred. Four valid cases reached
that boundary: CPU1 2048-bit pilot with skip-box, CPU4 centre seed, CPU1 centre
seed, and the committed-plan signed-pilot form. Thirteen invalid combinations
were refused before it, covering illegal thread counts, wrong-driver flags,
missing/wrong-driver plans, path traversal, excess precision, and invalid
watchdog limits. AST inspection confirmed that skip-box returns before the box
stage; syntax compilation passed.

No blocker was found for a separately committed and resource-admitted
**CPU1, 2048-bit, point-only** pilot. Numerical centre-seed calculations require
the separate source review below. A launcher bound does not establish that
2048-bit execution will fit the watchdog; an incomplete attempt must be retained.

| Reviewed source | SHA-256 |
|---|---|
| `pilot.py` | `354f166af4ac0d7afbe68d2b9c61ed17e54aa9f3e4ed2b448bef5e70545ccca1` |
| `launch.py` | `bc2dcefe460a538184e40311fcf52637eb3d1aff36d6483e7678ee379c6151a5` |

## Numerical centre-seed source

Pending the implementing agent's stable source. No approval of that
implementation is implied by the launcher admission test above.
