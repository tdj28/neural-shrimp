# Fixed temporal-mean curvature sufficiency pilot — 11 October 2026

Behavioral question: can temporal mean curvature recover the observed small
perturbation response where initial frozen curvature overpredicts it? This is
consumed-development mechanism work. The two-snapshot candidate failed its
accuracy/persistence gate; causal-horizon drift warnings added no catch beyond
gradient change; the small reversal-odd contribution does not exclude other
ordering effects. We test a different, explicitly retrospective approximation.

For seeds 28161/28162, arm 256, update 600, retain the exact original float64
reference, training tokens, ridge .001, rate and normalized gradient direction.
Compute Jbar = I − eta/16 sum H(theta_k), k=0..15, and Jbar^h u at h=1,4,16.
Each of 16 powers uses 16 native HVPs on the same current vector. Compare the
qualified two-amplitude finite difference, ordered response, frozen J0,
latest-J1, two-snapshot and persistence, including gain, signed alignment and
training-CE slope. No fitted weights and no causal/utility/commutator claim.

Two matched control candidates use the exact same checkpoint, tokens, objective,
ridge and observed direction at eta/4 for 16 updates. Acquire the reference,
ordered/frozen response and four perturbed branches at the original two
amplitudes. Retain all 16 objective increments; label a control small-step
monotone only if all are nonpositive. Never retune or replace a failed control.
The original 8-window cases remain contextual, not matched-data controls.
All data/checkpoints are consumed development; no fresh-seed confirmation,
protected assessment, extra network, paid resource or GPU is involved.

Keep the existing 1% two-amplitude and ordered-transport qualification, including
h1. CE qualification uses the same 1% gate and objective gradient minus ridge
term; the 17 already counted reference gradients supply all CE slopes.
Fixed descriptive gate: BOTH original expanded-gradient h16 mean errors <.20
and strictly tenfold smaller than frozen errors, conditional on those unchanged
qualifications. Report every horizon/control regardless of the gate. A positive
result supports temporal-mean sufficiency, not prediction or all ordering
causality; a negative result preserves the failed approximation.

Target counts: 4×256 mean HVP plus 2×32 control HVP = 1088 HVP;
2×(17 reference +64 perturbed) =162 gradients; 2×(17 reference +12 perturbed)
=58 CE. Profile: consumed seed28160/update50/arm256, exactly two each of these
three operations at full 15776-dimensional size. Fresh measured forecast
2*(1088 maxHVP +162 maxgradient +58 maxCE)+30 seconds must fit600 seconds.
The source records every attempted input before acquisition and every complete
returned value before validation. Exact-byte deduplication of vectors has
explicit references, so each complete event remains auditable. At most352
auxiliary vectors cover204 state rows,136 response/input rows and12 CE-gradient
rows; twice all native return bytes plus those vectors and64MiB metadata must
fit512MiB. This is a conservative uncompressed bound, not a compression forecast.

CPU1nice10/BLAS1,4GiB RSS/8GiB AS. Profile60s/80+5 owned; target600s/620+5 owned.
512MiB worker with4MiB closing reserve;514MiB outer. A failure stops this fixed
attempt, with complete failed receipt; no cap adaptation or implicit retry.
Source/tests/card and independent review precede the profile; measured literal
profile hashes and a separately committed target card must precede target.

Source-only helper controls cover noncommuting matrices (power of mean differs
from ordered product), identical input vectors within each mean action, exact
combined counts including readouts, literal direction, unchanged qualification,
failed monotonicity, partial-readout/nonfinite retention and resource pricing.
The full binding/ownership caller is being prepared; this helper card authorizes
no native acquisition by itself.
