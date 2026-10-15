# Source-only fixture amendment, 15 October 2026

Root's full review of release399fc236 identified one fixture wiring error: check_frozen_vs_sensitivity constructed `current` but did not pass it to `reduce_cp`. The subsequent expected sensitivity rho−1 therefore could not hold. The corrected call is `reduce_cp(documents, current)`.

Original controls1aac5c1e2e70609d74bfb7a4c14691792afa3b0793705308cb0dda45d19efe3a and release399fc2367b289f9558e77ee146e036be5da99027dc684fcb37cb453f0453891d are preserved under preparation-001. No controls, target calculations or model operations were executed. The scientific reducers, source policy, fixture data and expected assertions are unchanged. This is a preparation correction, not a numerical retry.
