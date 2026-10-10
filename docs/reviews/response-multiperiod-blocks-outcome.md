# Fixed two- and four-period response blocks

We can now recover the chosen regularized response column accurately for both saved training cycles. This is a numerical linear-solver improvement in two 13-parameter models, not evidence that shrimp geometry improves training. Both fixed methods still fail the predeclared efficiency screen.

| Saved cycle | Block size | Iterations | JVP+VJP calls | Readout error | Accuracy endpoint | Half-CG-call screen |
|---|---:|---:|---:|---:|---|---|
| p10 row35 | 2 periods | 144 | 46,560 | 4.44e-10 | pass | pass |
| p10 row35 | 4 periods | 79 | 25,760 | 9.08e-11 | pass | pass |
| p18 row111 | 2 periods | 253 | 146,592 | 1.51e-9 | pass | fail |
| p18 row111 | 4 periods | 151 | 87,840 | 1.23e-10 | pass | fail |

The p18 four-period run exceeds its 86,832-call cutoff by 1,008 calls. The threshold stays unchanged. Every method and the earlier CG/one-period outcomes remain in the raw result. Factor construction and triangular solves are counted separately; dense exact blocks up to 936 coordinates are a small-model oracle, not an established LLM implementation. Historical wall times are not a controlled speed comparison.

A separate saved-arithmetic checker reconstructed K/KT actions, true residuals, readouts, callback totals and retained Cholesky identities without any new factorization, solve, derivative or model call. It passed in 1.23 seconds; maximum relative factor residual was 2.23e-16. This check was written post-outcome by the producer agent. Independent personnel outcome review remains pending.

The bounded acquisition took 4.66 seconds internally (5.03 seconds externally), with 114,168 KiB peak RSS; all 1,876 inventoried files (201,516,392 bytes) are immutable, and owned PIDs exited. Exact source, raw and audit hashes are in [the companion receipt](response-multiperiod-blocks-outcome.json).

The next useful test must replace the dense-block oracle with an operator-only approximation and count its inner phase actions. A fixed number of inner CG iterations is not a fixed linear SPD preconditioner; either use an explicitly positive stationary polynomial or an outer method designed for variable preconditioning. Merely increasing the dense block length would not establish scalability.
