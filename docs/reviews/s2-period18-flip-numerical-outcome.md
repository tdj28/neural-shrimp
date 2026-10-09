# Ray13 reaches an unstable doubled branch, after an earlier stability pocket

The carried period18 training pattern first loses stability through a complex
pair, later regains stability, and then reaches a genuine numerical−1 crossing.
The three nearby primitive36 cycles found there are **unstable**, while their
period18 parents are attracting. Thus this ray does not support a simple
stable18-to-stable36 transition. The numerical behavior is consistent with a
subcritical flip; the higher-order coefficient and interval theorem are being
checked separately.

The source-bound pilot used the previously consumed13-parameter S2 model and
historical ray13, with both learning rates varying. It carried all18 phases,
checked native-phase movement and reverse corrections, and refined a simple−1
point at `s=0.25405443918025816`. The full-state closure is4.56e-14, augmented
residual7.29e-13, and the largest other multiplier modulus0.891418. Corrected
branch differences at four decreasing steps and the analytic branch tangent
agree on `d mu/ds≈−51.23463`. The adjacent native phase gives the same crossing
slope and qualifies independently in floating point. This is evidence rung1,
not an interval or continuous-branch proof.

Amplitude-constrained36-cycle solves cannot silently collapse onto the18-cycle
parent. All proper divisors1,2,3,4,6,9,12,18 were excluded numerically; both
18-separated native starting phases pass closure and spectral-residual gates.
Their closures are below2.2e-15 and minimum divisor separations exceed3.14e-4.

| Requested amplitude | `s-s_flip` | `(s-s_flip)/amplitude²` |36-cycle radius|18 parent|
|---:|---:|---:|---:|:---|
|0.0003807053|−2.208965e-7|−1.524091|1.00004518|attracting|
|0.0007614106|−8.836376e-7|−1.524180|1.00017963|attracting|
|0.0015228212|−3.535371e-6|−1.524534|1.00070171|attracting|

The near-quadratic parameter offsets and parent-stable side support the local
subcritical interpretation. These are direct ratios of saved outcomes, not a
fitted coefficient or extrapolated proof. No attracting36 cycle was found in
these three attempts; no absence theorem or conclusion about other rays follows.

An important earlier finding changes this ray's anatomy interpretation. The
first run stopped at the attraction gate despite qualified primitive/full-state
and reverse checks. A complex pair crossed outward between
`[0.15862523928269184,0.16655650124682642]`; the amended continuation retained
unstable states and found re-entry between
`[0.17448776321096102,0.18241902517509562]`. The historical coarse ray sampled
across this pocket. Its later `s_boundary≈0.254054437` is not the first observed
loss of attraction along the carried branch. Preserve the original ray census;
do not silently relabel it as a census of first boundaries. Neither complex
crossing has yet received a nondegeneracy proof in this pilot.

The A2 aggregate “10/15 flip-adjacent period36 labels” is not a full-state
association to these local cycles. A stable finite-amplitude36 attractor or a
different basin/family could still underlie a coarse cell label. The discrepancy
motivates checking other rays and direct branch associations, not declaring
all historical geometry wrong.

Attempt001 is retained unchanged. Attempt002 resumed its last qualified state,
with an explicitly amended stability interpretation and unchanged closure,
primitivity and phase gates; it completed in2.441s,6301 joint map/Jacobian calls,
under100s/40k calls/32MiB/CPU1 caps. Across the retained prefix and continuation,
33 phase-resolved points were sampled, but no continuous tube was proved.
Both attempts, the tiny synthetic timing profile, exact log-ray family,
flip-proof seed, raw events/evaluations and all source/PID/hash receipts are
bound by `s2-period18-flip-numerical-receipt.json`. All raw files are read-only,
hashes verified and owned PIDs absent. No holdout, new data, GPU or paid work.
