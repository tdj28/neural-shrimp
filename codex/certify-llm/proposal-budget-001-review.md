# Independent audit: proposal-budget-001

Reviewed 13 October 2026. This was a read-only CPU1 audit of authorized saved
proposal arrays, with an 18 GiB address-space limit and 120-second alarm. It made
no model calls. The outcome is accepted as a **floating planning calculation**,
not a rigorous derivative bound, inclusion gate, or certificate.

## Evidence and admission

- Receipt SHA256: `16745bba2341309a50592f827bd764331ae01e5d1892e2f0897d8672b20417ee`.
- Source commit: `95693c8cb73e6ed182f418f9e18c3bf4e69d1b26`.
- Driver SHA256: `63893ec0c91739b0cd5668e1257163110c1846a3ea0d40389dce92e246deb8c2`.
- Plan SHA256: `b5f68f9573e0098f6557d0bd20fe4a8720cfca6b9126a7d9bfa0f4ad2ec690ce`.
- Estimates SHA256: `38886e8e4f8ec840fde1b4df1781cf59ba4f84a3a694a0cfed0c38edc5a0acf1`.

All 63 inventoried artifacts and all 57 captured source files match their hashes
and sizes. Every source file also matches `git show` at the stated source commit.
The committed plan, saved inputs, source bindings, resource envelope and summary
hashes agree. The saved threadpool records each use one thread. The receipt is
exit 0 without timeout; 3.328648 seconds total, 2.210670 seconds in the driver,
686,524 KiB peak RSS.

The reviewed loader was independently re-executed on the authorized saved data.
It checked the complete 32-artifact prefix receipt and 56-artifact seed receipt,
the seven-array preconditioner inventory, pinned assets and prefix IDs, and
returned exactly the saved source-binding metadata. This does not construct a
model. The independent audit then recomputed signed prefix products, the inverse
defect, each suffix product from scratch, all eight rows, and all 64 phase-pair
factors without calling the production estimator. Every scalar agreed within
1e-12 relative tolerance. Its numerical calculation remains ordinary binary64.

## All rows

| Phase | Propagated inverse-defect estimate | Discrepancy coefficient | Strict threshold estimate |
|---:|---:|---:|---:|
| 0 | 3.0221396927e-15 | 238.703052392 | 4.1893054571e-3 |
| 1 | 7.8056360130e-16 | 428.765611809 | 2.3322765923e-3 |
| 2 | 5.7185356180e-16 | 336.860065682 | 2.9685917147e-3 |
| 3 | 6.8014916862e-16 | 1009.99596266 | 9.9010296771e-4 |
| 4 | 3.6388667247e-16 | 23828.5389464 | 4.1966484066e-5 |
| 5 | 1.4171965802e-15 | 1232343.39923 | 8.1146213030e-7 |
| 6 | 3.0691191908e-15 | 45470811.7324 | 2.1992129938e-8 |
| 7 | 1.4574609679e-15 | 386013110.541 | 2.5905855855e-9 |

No failed or omitted row was found. Phase 7 limits the common discrepancy to the
floating planning threshold `epsilon < 2.5905855855480174e-9` for equal phase
radii. Epsilon means the **induced infinity matrix norm** of every `J_j - A_j`,
not an individual entry's radius. Entrywise errors must be summed across each
row before comparison. The estimate concerns contraction only: it does not
establish a residual-versus-radius inclusion, primitive period, attraction, or
an orbit. None of these binary64 estimates is an outward rigorous bound.

The independent audit completed in 1.879 seconds after imports, peak 674,004 KiB.
An initial audit-script import omitted the `codex/crosscheck` search path and
failed before reading the arrays; adding that path resolved it. No evidence or
producer source was changed.
