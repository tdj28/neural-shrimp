# Independent outcome-auditor source review — 8 October 2026 UTC

Root reviewed the outcome-blind auditor implementations and launchers at utility
fba1c46caf3df06b9639ff2a4cb3e00b2c62f6d7 and CIFAR
b8f074b8539d5a1ae7e3d4f629b8dcb6fb659bdd. Numerical producer outcomes were already
known to root, but were not supplied to the auditor authors before their source
and cards were committed and pushed. This is exploratory verification, not a
preregistered hypothesis test or a new utility/solver experiment.

The utility auditor independently implements the analytic gradient using explicit
sample/coordinate loops and reconstructs dose decisions at all four original
states. The CIFAR auditor reuses saved Q/R, independently reconstructs projections
and norms, and checks all saved identities without a new QR, model evaluation or
linear solve. Both bind historical sources and externally supplied full manifests,
retain failures, enforce exclusive output directories and bounded owned processes,
and separate comparison tolerances from the original scientific gates. Shared
binary64 libraries and saved producer map images remain explicit limitations.

Root reran their source-only synthetic suites: utility30 passed in0.46s and CIFAR34
passed in0.09s. These fixtures do not open real target outcomes. No blocking source
issue was found. This review authorizes exactly one attempt per committed card,
with no retry or source mutation after launch:

- Utility: CPU/BLAS1,nice>=10;5s internal,10s TERM plus5s KILL;1MiB total new raw.
  Source fba1c46caf3df06b9639ff2a4cb3e00b2c62f6d7; producer execution
  977514e4f53025f485bcd47ce947c9adef044843; raw manifest
  d0b6a4fd5a0c96c3a1ac95bbfb33cb3661b82202062e37fac238f00c92bf67d2;
  producer launcher manifest
  b3092fb6fee21f53fde9a09b7fa87665b618543e7225d5bf69969799d79de9e2.
  Outputs /data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001
  and its attempt-001-launcher sibling. This commit is the source-review identity.
- CIFAR: CPU/BLAS1,nice10;15s internal,20s TERM plus5s KILL;4MiB total new raw.
  Source b8f074b8539d5a1ae7e3d4f629b8dcb6fb659bdd; pins-file SHA256
  0ba6972bc719ac178a0028a50246627ee3a6573fc8ba8b7c949826675e19f382.
  Outputs /data2/neural-shrimp-data/audits/cifar-input-span-independent/attempt-001
  and its attempt-001-launcher sibling.

No GPU, paid call, protected data or additional experiment is authorized by this
review. Retrieve and hash complete outputs, verify owned processes exited, and
publish compact independent companions privately even if an audit fails. Producer
companions retain their historical pending-at-creation audit labels.
