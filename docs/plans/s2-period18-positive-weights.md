# 10 October 2026 UTC — saved-defect anisotropic-weight diagnosis

Root authorizes one bounded exploratory saved-data calculation after the
quadratic width1/64 pilot. No model, gradient, Hessian, trajectory or new root
solve. Pin parent execution02ab12123a3a3f5bedba6284f3c984f3e3f85c90, worker
manifest60bddca8270784a7264bd0d9d6650e5efd46ce5eca18b9ff1f30f515c863fbc3,
under `s2-period18-quadratic-connection-2026-10-10/attempt-001-target/run`.
Complete inventories and original source bytes must match before reductions.

For each of the two retained234D charts, separately reduce the forcing
`-C*predictor_residual` and propagated error `D*E`, D=I-C*A. For a single
deterministic proposal form binary64 B from the saved absolute defect bounds,
compute its numerical spectral radius rho, choose q=min(.95,max(.05,2*rho)),
and solve `(I-B/q)w=1`. Require finite positive entries and normalize max(w)=1,
with min(w)>=1e-6. Convert every entry to its exact binary64 dyadic. Floating
eigenvalues and solve only propose weights; they prove nothing.

At384-bit Arb precision independently recompute
`max_i sum_j |Dij|*wj/wi`, all weighted coordinate inclusions using radii r*wi,
and the separate normalized forcing/propagation contributions. Strict positive
exact weights and wi<=1 ensure the anisotropic error box is a subdomain of the
already checked original box. The saved Jacobian is valid on this smaller
domain. Any future wider parameter or state domain MUST reacquire its true
Jacobian; do not extrapolate the saved bounds as a proof. New root containment
inside the old uniqueness domain identifies the same root if these sufficient
weighted gates pass. This diagnostic does not enlarge the proved interval.

Use `tools/analyze_s2_period18_quadratic_weights.py`, CPU/BLAS1,nice10,
15s internal,20s external plus5s cleanup,1GiB at checkpoints,1MiB raw. One
exclusive output `/data2/neural-shrimp-data/explore/certificates/s2-period18-quadratic-weights-2026-10-10/attempt-001`.
Use `tools/launch_s2_period18_quadratic_weights.py --out <exclusive output>`.
The outer launch command uses timeout and archives its exact argv, source,
start/end/PID/exit/log hashes alongside the worker; no automatic retry. Source,
tests and this card are committed/pushed before saved target arithmetic. Preserve
all failures, source/library metadata and immutable exact weights/results.

Compare residual and q*r contributions before choosing one wider pilot setting.
The observed cubic residual motivates testing a higher width only if weighted
contraction and inclusion leave margin. Otherwise diagnose a different basis
or error enclosure; do not claim the root advances toward the fold from these
saved-only reductions. Independent outcome review remains separate.
