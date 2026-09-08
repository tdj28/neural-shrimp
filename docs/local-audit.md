# Local audit before initial target outcomes

2026-09-08. Source and complete manifests are committed and pushed after this
audit and before pilot execution. The studies remain private exploratory pilots.

Independent reviews of the numerical core, compute package, and paper/protocol
identified an integration error (`classification` versus `status`), missing
push/manifest binding, and on-run generation of synthetic inputs. All were
corrected before target outcomes. Frozen input files now precede the freeze;
runtime verifies their exact bytes and shapes. Adaptive policy history resets
to the current map after intervention and observes a minimum dwell time.

Numerical validation compares analytic gradients against scalar-loss central
differences and analytic Hessians against gradient differences; exact logistic
and Hénon controls pin period, primitive-period rejection, stability, neutral
directions, and Lyapunov handling. An analytic quadratic-GD fixture checks the
paired pre-intervention path, cycle rescue, and non-triggering at fixed points.

Archive integration tests execute the actual CLI from source exports without
`.git`, check full input-copy hashes, and exercise corruption, deadline,
no-progress, signal cleanup and exclusive-attempt failures. They are instrument
fixtures, not target experiment results.

Scope decisions: the initial neural network includes biases and ridge and is
inspired by the Sohl-Dickstein setup, not an exact reproduction. Hénon seeds
are identifiers for the one fixed initial condition, not basin replicates.
Four utility task seeds are separate from the two mapping seeds; a seed controls
both task and one initialization. Pilots have no inferential p-values. We do not
classify nonrecurrent bounded states as chaos. No boundary continuation or
geometry-specific controller is implemented. Tests cannot establish such claims.

Paid review: `not_requested` under canonical integrity §7A. Paid resources: none.
The initial runs fit local CPU resources. Remote GPU parity and actual provider
lifecycle control remain unverified and are not required for these local pilots.
