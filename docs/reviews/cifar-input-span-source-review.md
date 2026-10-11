# Independent CIFAR input-span source review

8 October 2026 UTC. **Pass for amended source17453ecc08b1d0b08c159f506b9447b1b6061968**,
with the inspected external wrapper bound in the [receipt](cifar-input-span-source-review.json).
This approves source for root's separate launch decisions; no actual profile,
target QR/projection, target gradient/derivative or new outcome was run/read.

The full Euclidean-coordinate derivation is correct for this fixed model:
W=QB+N, orthonormal Q containing range(X transpose), Q transpose N=0.
Data loss is independent of N; isotropic ridge supplies lambda*N alone, so
GD is a direct product of the active map and qN. Period-p elimination requires
q^p !=1, and full attraction additionally requires |q^p|<1. The omitted
monodromy block is q^p I. Both layers and biases can train; no frozen-layer or
linearized-tanh assumption is needed. The independent nonlinear toy gradient
and complex-step residual JVP reproduce these identities. A nonzero invisible
primitive period-two example at q=-1 checks the exceptional case.

Packing matches the actual CIFAR model: row-major3072x32 first-layer weights,
then32 hidden biases,32x10 output weights and10 output biases. The rest remains
active. Retaining all512 reduced-QR columns, including orthonormal completion
under deficient rank, gives98666 full/16746 active/81920 omitted coordinates.
It neither asserts exact data rank nor performs singular-value truncation.
Independent rank-zero/rank-one fixtures preserve every thin-QR column. The
implicit projection avoids a3072x3072 matrix. Required basis, identity,
Pythagorean and complete-vector checks are fail-closed numerical diagnostics.

Input loading binds complete fixed reference/linear manifests and historical
source bytes, exact model/data/rate identities, five saved state/residual/map2
associations and the original p/Ap/e direction. No trial Jacobian is invented.
The exact-real Fraction q is distinguished from the rounded factor used in
floating checks. A residual share above0.5 is triage only; it cannot explain a
GMRES stall, discover an orbit or support a utility claim. Removing N need not
preserve predictions on unseen inputs.

Two findings in originale6485c67 were fixed before any profile/target:
collection had self-hashed the current profile manifest instead of requiring
an externally approved pin; and `np.array_equal` did not implement the named
bitwise saved identities. Original controls reproduced six expected failures.
The amendment requires the externally pinned64-hex profile SHA, validates full
profile inventory/source/environment/settings/actual metrics and records the
pin; all exact relationships now compare bytes, including signed zero. A
consistently rewritten/rehashed fake profile is rejected under its old pin.

All41 combined controls pass in1.82s on CPU1nice10, including11 independent
cases. The only warning is an upstream TorchScript deprecation. All six
execution-bound files were compared byte-for-byte with17453ecc. The external
wrapper was read, not executed: SHA256
3c6c82bf15c55bb078981c62546bd796a19d9884d2f82a73e8960da5e12a8cc5.
Its fixed8+2s profile and45+5s target supervisors kill the owned group even with
an inherited open pipe; each log is capped128KiB and each launcher256KiB.
Target Store deducts completed profile bytes and leaves512KiB for both
launchers, preserving the64MiB aggregate contract. The wrapper checks actual
aggregate bytes, retains exact source/commands/PIDs/exits/manifests, and freezes
outputs. Caught partial writes are manifested; SIGKILL or disk failure can
leave explicitly incomplete evidence. These resource and provenance checks
are not proof of future runtime or target identity acceptance.
