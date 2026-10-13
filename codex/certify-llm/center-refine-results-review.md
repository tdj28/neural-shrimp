# Independent audit of center-refine-001

Completed-attempt review, 2026-10-13. Read-only retained-data and exact arithmetic audit; no checkpoint load, map evaluation or target replay.

The attempt passed the specified setup audit. Its full multiple-shooting residual norm strictly decreased: the new rigorous upper bound is below the old rigorous lower bound. Rounded display values are:

- Before: lower and upper both display as 9.6226572500e-15.
- After: lower and upper both display as 2.4361464264e-27.
- Ratio of new to previous upper bounds: approximately 2.5316774391e-13.

These displays conceal tiny nonzero enclosure widths. The audit used exact dyadic endpoints for inequalities, not these rounded decimal strings. A residual reduction is established; existence, uniqueness, primitive period and attraction of an exact orbit are not established by this setup result.

## Provenance and row gates

- Attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/center-refine-001`.
- Receipt SHA-256: `faa6f920253c17eda0639280fd4ff76c3da579676bd31af932be508223ba89a0`.
- Source commit: `ba6c4d77fd033fe9699c246e48af1a50bd6445d1`.
- Driver SHA-256: `f24f9cdecec243abd25671a684aec03b8b105288526ff7fa4f60a963af745b02`.

All 77 receipt artifacts matched sizes and hashes, with no unaccounted files apart from the receipt itself. All 31 source-manifest blobs matched both their retained files and the Git objects at the exact source commit. The manifest records one CPU thread, 18 GiB and a 3600-second watchdog. The launcher exited zero without timeout.

All 16 ordered phase rows completed. The audit checked all 14,336 phase-input/primal-output coordinates, exact zero tangent fields, full vector dimensions, input/output hashes, exact cyclic successor centres and saved maximum radii. Original before-centre dyadics equal the pinned seed's binary64 values exactly. All before/after residual intervals enclose the original saved image minus the correct exact successor centre. Saved correction-image intervals enclose their actual source images, all correction residual records retain full dimensions and bounds, and every new centre equals the recorded 2048-bit midpoint update. No centre was reconstructed through binary64.

The retained upper and lower infinity-norm summaries were checked using exact rational endpoints. Their comparison booleans agree with those exact bounds. All checks passed on the first completed-attempt audit execution; no producer output or criterion was amended.

## Every measured phase

Times are seconds; radii are rounded displays of the maximum saved primal radius for the phase. Each row passed all applicable checks above.

| Phase | Before time | After time | Before max radius | After max radius |
|---:|---:|---:|---:|---:|
| 0 | 109.514618 | 139.116046 | 6.7202402470e-451 | 6.7212784654e-451 |
| 1 | 109.510435 | 139.656524 | 4.4147927708e-451 | 4.4170389189e-451 |
| 2 | 109.586398 | 139.645844 | 6.2553646571e-451 | 6.2729667793e-451 |
| 3 | 109.635283 | 161.874056 | 4.3099707523e-451 | 4.3135503410e-451 |
| 4 | 109.547888 | 153.101412 | 5.9295424182e-451 | 5.9467769083e-451 |
| 5 | 110.026539 | 156.920708 | 5.7759056967e-451 | 5.7913683356e-451 |
| 6 | 115.327553 | 190.798209 | 1.8679545953e-450 | 1.8706795539e-450 |
| 7 | 140.067010 | 164.633312 | 7.5864620176e-451 | 7.5999900827e-451 |

Before-pass mean/max: 114.151965 / 140.067010 seconds. After-pass mean/max: 155.718264 / 190.798209 seconds. Sum of all phase times: 2158.961834 seconds. Driver time: 2176.182979 seconds; launcher time: 2176.975077 seconds. Peak RSS: 5,763,316 KiB (about 5.50 GiB).

For the next costed plan, two corrections require sixteen fresh phases when the complete after images are reused. At this attempt's after-pass mean that is about 2491.49 seconds, and at its slowest after-phase rate about 3052.77 seconds, before setup overhead. Three corrections would take about 4579.16 seconds at that slowest observed rate, exceeding the one-hour watchdog. These are workload estimates, not guarantees: contention and phase-dependent costs may change. Two corrections are therefore the defensible next bounded scope under the unchanged one-hour cap.
