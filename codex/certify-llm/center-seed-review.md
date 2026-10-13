# Independent audit of centre-seed attempt 001

Date: 2026-10-13 03:43 UTC. **Accepted as a completed numerical proposal setup.**
This is not a certificate or a rigorous derivative/inverse enclosure. The audit
loaded saved arrays only; it did not load model weights or evaluate the model.
No attempt files or Git state were changed.

## Immutable evidence and provenance

Attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/center-seed-001`.

| Binding | Value |
| --- | --- |
| Source commit | `4324234a5c1703e5e9589c9bb6b21bc4710b4df5` |
| Complete receipt SHA-256 | `ee952c34b442a5947c09471d6d40129b150965ee1f2abf69760c7d304952b6c3` |
| Manifest SHA-256 | `da7c8bb26aa0380962677955c8538d7d208dea86b76c63505e9099bf6f019587` |
| Final preconditioner NPZ SHA-256 | `c853fceb10036ff6229b6435a01951eccc2e629c6d70bc967da8cde1fb008b43` |
| Seed driver SHA-256 | `87cf27a05999641995d441704cd7a1ec3ea5aa0d2c0d200b0c05662e584572e5` |
| Independent map source SHA-256 | `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e` |
| Original nine-state proposal SHA-256 | `d8c57bff1d8a986c7c6a0f7d083caa77b4f6373c560d1f60ce0c57461caf668b` |

All **56** receipt-inventoried artifacts passed independent size and SHA-256
checks. The actual file inventory matched those entries plus the receipt
itself; every saved artifact and receipt was read-only. All **22** captured
source files matched both manifest hashes and the exact Git blobs at the
recorded commit. The source, manifest, input record and recorded private-main
revision agree. This audit did not fetch or mutate Git.

The run exited zero, did not time out, and completed in 677.271 seconds of
launcher wall time (676.593 driver seconds). The resource record binds the
owned PID 174682, four CPU threads, nice 10, an 18-GiB address-space cap and a
3600-second ceiling. Reported peak RSS was 5,863,288 KiB. The assigned exact
binary64 alpha/log-temperature ratios, period eight, dimension 896, 37 prefix
IDs and pinned asset hashes match the captured independent map source. The
original proposal was independently rehashed. No claim is made that this
array-only audit independently recomputed the original model evaluations.

## All eight phase gates

Each JSON gate was read individually and compared with both aggregate records.
Every phase is finite binary64, contains an 896-by-896 Jacobian, has zero
nonfinite entries, and passes the declared relative closure threshold `1e-10`.
The relative residuals below were independently recomputed from saved
`image-next_center` arrays.

| Phase | Recomputed relative L2 residual | Jacobian seconds | Gate |
| ---: | ---: | ---: | --- |
| 0 | `6.72977800798229e-14` | 80.601 | Pass |
| 1 | `1.87496009199965e-14` | 77.649 | Pass |
| 2 | `2.49835696175702e-14` | 83.096 | Pass |
| 3 | `1.74715686269744e-14` | 82.650 | Pass |
| 4 | `1.70544030242959e-14` | 84.505 | Pass |
| 5 | `1.80196793646389e-14` | 85.735 | Pass |
| 6 | `1.57743835323042e-14` | 84.196 | Pass |
| 7 | `2.24488125304375e-14` | 94.063 | Pass |

Every phase NPZ hash matches its gate and complete receipt. Its stored center
and next center equal the corresponding cyclic entries in the aggregate
centers; its Jacobian equals the matching `A[i]`; its residual equals the
stored image minus the next center. The earlier residual-only checkpoints
match the corresponding later phase arrays exactly. All phase-save events
are present, ordered 0 through 7, and passing. These are numerical consistency
gates, not interval proof gates.

## Aggregate arrays and numerical preconditioner check

The final NPZ contains exactly seven unique keys, all finite float64:

| Key | Shape |
| --- | --- |
| `A` | `(8,896,896)` |
| `B` | `(896,896)` |
| `centers` | `(8,896)` |
| `parameters` | `(2,)` |
| `product` | `(896,896)` |
| `left_inverse_defect` | `(896,896)` |
| `right_inverse_defect` | `(896,896)` |

Parameter order is exactly `[alpha, log10_temperature]`, with values
`[0.071674, -0.115]`. The eight centers equal the first eight states of the
hash-pinned nine-state proposal. The separate center and phase-matrix
checkpoints match the aggregate arrays exactly.

On one CPU thread, the reviewer recomputed the ordered binary64 product
`P=A[7] @ ... @ A[0]`, then `I-B@(I-P)` and `I-(I-P)@B`. In this environment
all three recomputed arrays were entrywise equal to the saved binary64 arrays. Their
numerical diagnostics are:

| Quantity | Recomputed value |
| --- | ---: |
| `||P||_inf` | `1.0395393398685395` |
| `||I-B(I-P)||_inf` | `3.0221396927105457e-15` |
| `||I-(I-P)B||_inf` | `2.739038248894002e-15` |
| `||B||_inf` | `2.045126150247826` |
| Estimated `cond_2(I-P)` | `1.8898299640663134` |

The saved condition estimate was `1.889829964066314`; the difference is ordinary
floating rounding. These results support the numerical plausibility and order
of the proposed cyclic inverse. They do not prove invertibility in exact-real
arithmetic, enclose a full-box Jacobian, or establish dynamical attraction.
The saved entries may be interpreted as exact dyadic **chosen constants** for
later rigorous computations; the numerical calculations that proposed them do
not become rigorous by that interpretation.

## Audit qualification and downstream use

The only target stderr content is a Torch `jit.script` deprecation warning;
there is no traceback. An initial audit assertion expected empty stderr and
therefore failed on that warning. The text was inspected, that audit-only
assumption was corrected, and the entire audit was repeated successfully.
No target data or computation changed. The complete repeat took 0.957 seconds
on CPU1 and exited zero.

This attempt is suitable as the fixed `A_i,B` proposal input to the separately
reviewed, bounded high-precision center-refinement setup. All saved setup
records retain `certificate_passed=false`. A later certificate still requires
rigorous residuals, whole-phase-box derivative enclosures, strict shooting-root
inclusion/contraction, proper-period separation and a separate attraction
proof. None of those follow merely from the successful numerical seed.
