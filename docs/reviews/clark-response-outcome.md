# Finite-window response benchmark — 9 October 2026

The response identity is useful on these small saved training cocycles, but the
finite-window implementation is not automatically a spectral counter. All216
regularized linear solves have true relative residual≤1.84e−12, yet boundary bias
is still visible; one converged finite-window readout even exceeds the ambient
count13. Small linear-system residual does not establish the infinite-time limit.

The fixed panel used binary64 midpoints of already-saved C1 phase Jacobian
intervals:13D p10 row35 and p18 row111. No new model/derivative, trajectory or
certificate call was made. Floquet and512-period QR counts agree at all four fixed
thresholds in both slots. Signed multipliers, complex angles and individual
exponents remain separate in the raw references. These floating midpoint cocycles
are numerical surrogates; the response analysis is not an interval certificate.

At64 periods and eta1e−6, counts should be13 at threshold s=0. The free response
returns12.996770 (p10) and12.999649 (p18); periodic returns13.003077 and13.000259.
Across all four thresholds at this longest window, maximum absolute count errors
are0.071916/0.088262 for free and0.083277/0.089916 for periodic. The largest errors
occur at s=−.005, close to the slowest contraction exponent; changing eta alone
does not remove this finite-window effect. Every window, eta and boundary result
is retained and summarized, including the intentional causal wrong-limit control.
For example the longest p10 causal count at s=−.05 is9.994054 versus reference9.
A causal finite-window solve becoming nearly integer is not evidence it found the
correct two-sided stable count. Synthetic exact-neutral and singular examples
explicitly abstain from the identity assumptions, and the fixed finite causal
eta→0 limit is demonstrated to return1 for an unstable scalar.

The matrix-free callback prototype is a negative scaling result in its present
unpreconditioned form. Both single-e0-impulse LSMR probes hit the fixed300-iteration
limit, with true normal residuals4.39e−4/5.70e−4 and readout-column errors
0.07675/0.09644 against direct solves. They used96,480/173,664 individual phase
JVP/VJP callbacks; here these are cheap13D matrix products, not measured large-model
HVPs. This does not justify a larger-LM acquisition. A next method unit should first
show materially lower operator counts with a justified time preconditioner or a
boundary treatment on the same saved operators. No cavity closure or training
utility is inferred, and signs/angles/isolated modes remain indispensable.

The complete panel took6.193s at124908KiB RSS, CPU/BLAS1 nice10, under the60s/512MiB
allocation.31 synthetic controls pass. Preserved earlier evidence includes a
73-row address-space failure and a dense-profile cost gate failure. The disclosed
allocation adaptation replaced Python scalar sparse assembly, raised only the
virtual-address bound to1GiB (physical groupRSS480MiB unchanged), and measured the
full synthetic window mix with a factor-two safety allowance. The passing profile
projected19.366s; the complete rerun is exploratory continuation, not confirmation.
All five launch bundles and the saved-only summary are immutable and hash-bound,
and their owned processes exited. Companion receipt records exact source/pins;
independent outcome review remains pending.
