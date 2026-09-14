# Arithmetic diagnostic design audit

14 September 2026. Reviewed before executing the new diagnostic against
EXP-008 research inputs. This is a post-run numerical investigation of known
failures; it cannot repair or supersede the original frozen replay gate.

The implementation preserves the frozen projected columns for its binary64
comparison, then separately measures directional quadratic-form evaluation,
Gram formation, and projection error using a wider-mantissa reference.
The direct residual norm uses the same flattened sample denominator and
original saved action delta. The reference cannot recover accuracy already
lost in stored trajectories or sensitivities.

The ridge comparison holds centering, scaling, group weights, intercept,
penalty grid and development rows fixed across normal equations, weighted
SVD and augmented pivoted QR. Test labels do not select a solver. Independent
stationarity calculations use the weighted design without forming another
normal matrix. Their finite-precision error estimates are explicitly
diagnostic rather than certified bounds.

The CLI binds the expected source commit, original receipt and all listed
artifacts, validates complete row censuses and seals, requires a clean source
checkout and an origin tracking ref containing the commit, and reserves a new
external report before computation. Existing output paths are rejected.
The operator additionally checks the actual server-side branch SHA after
push. Library, precision and thread settings accompany the result.

Local review found no blocking issue. The implementation's synthetic tests
cover an analytic ridge solution, deliberate cancellation, correlated designs,
invalid input, evidence-binding rejection, pushed-source gating and retention
of failed output. Initially 17 tests and 30 adjacent tests passed together
in 0.55 seconds. A final additional cross-validation-path test brought the
standalone arithmetic suite to 18 passing tests in 0.33 seconds; the adjacent
tests were not repeated. No research-data diagnostic or new integration was
run during this design audit.

Execution will use the server CPU with one numerical-library thread, preserve
all comparison failures and write full output under
`/data2/neural-shrimp-data/audits/EXP-008-arithmetic/attempt-001/`.
