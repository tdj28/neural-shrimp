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

## Completed fixed caller

The source-prepared JSON card is deliberately unlaunchable. The wrapper binds
all four original response/two-snapshot producer and independent-audit immutable
inventories and historical source identities. It checks current native-model
bytes and runtime equality, exact float32 checkpoint promotion, original
reference updates, literal observed direction, readout gradients and unchanged
perturbation amplitudes. The original comparator archive stores h1..h16; the
wrapper authenticates h1 against the original ordered response and prepends
only the exact original h0 direction. New control g0 must equal the saved
original g0 byte for byte. No original comparator is recomputed with extra HVPs.

Every native input and return has an exclusive indexed event. Vector objects
are losslessly deduplicated by their complete NPY bytes; trajectory aliases are
row-reference lists, not duplicate dense arrays. Nonfinite native returns remain
saved. Full immutable inventories and absent owned process IDs precede reuse.
The only writable historical containers permitted are the two literal older
broader-data roots; every file and manifest remains immutable and hash-bound.
The source archive passes an isolated import-closure control. The owned launcher
uses a512MiB parent softAS and8GiB inherited hardAS, so the worker can apply its
actual8GiB limit without an inherited contradiction. Source is CPU float64 only.

Before native acquisition, actual historical schema preflight caught and fixed
the comparator's h1-origin indexing. The initial import-closure control counted
Torch pseudo-file names as real paths; its fixed file-existence filter now
checks every real loaded local module. Twelve source/binding/synthetic controls
pass. These were preparatory checks, not native profile or target failures.

## Profile001 packaging failure and pre-import repair

Execution a717415c stopped before creating the worker directory or acquiring any
native value: the inherited supervisor512MiB softAS prevented Torch mapping
before main() could raise it. Immutable outer4760072e1ecaf3487b215656a98b80dc54a37cad2b87fbdcc7270173b7af8c0b,
0.290s, both owned PIDs gone. Its complete source/inventory/ownership is now a
mandatory input. The same declared8GiB workerAS is installed at script entry
before NumPy/Torch imports. No mathematical or resource ceiling changes. A real
worker --help subprocess starting with the actual512MiB/8GiB inherited limits
passes;14 combined controls pass3.49s. Fresh exclusive profile002 only, original
target001 remains unopened. The card returns to unlaunchable source_prepared
until narrow peer review and current host admission. No failure is overwritten.
