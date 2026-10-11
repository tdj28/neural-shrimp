# Saved eigenpair-locus forcing/defect diagnosis — 10 October 2026

Root authorizes this diagnosis after retained failed target001, not a new
acquisition. Pin outer35c8aeb88105a5836d47de7b6747b74da789bc266ab6084731b1f33688798369,
run6e21d614a08c336e3d7cde3127e0a6f8d6d4af461e261282df56df95106dbe26,
source463e55775ec89d9f45c00ebf594f616a09b278f3. Verify all immutable inventory
and source bytes. No model, new derivative, orbit, or proof replay call.

At384bits restore exact saved balls and compute C*r and absolute-defect row
bounds relative to recorded coordinate radii. Retain exact forcing vectors,
nominal-radius propagation ratios and separate saved image/radius ratios;
nominal Arb radius inflation must not be silently identified with exact E.
Compare predictor-only and whole-correction-domain saved Jacobian defect
bounds, without attributing their difference to an independent derivative.

Compute eigenvalues/right vectors of the floating upper-entry matrix |D|;
retain the largest-modulus mode and residual. These are enclosure-matrix
engineering diagnostics, not orbit Floquet multipliers or interval proofs.
Propose abs(Perron-vector) weights with a fixed2^-40 floor and, if positive,
e=(I-|D|)^-1*|C*r| expanded by17/16 plus a fixed relative2^-40 floor. Save
weighted ratios and every failed solve. Any changed domain/radius requires
new whole-domain derivative acquisition and unchanged rigorous gates; no
automatic target rerun or gate relaxation follows.

Exclusive `explore/eigenpair-locus-prototype/attempt-001-diagnosis`.
CPU/BLAS1 nice10, internal45/external60+5s,32MiBworker+1MiBlauncher,
1.5GiB sampled owned-groupRSS,4GiB hard virtual-address ceiling. Preserve
partial/failure outputs and immutable manifests. Commit source/tests/card
before one invocation of tools/launch_eigenpair_diagnosis.py. Source-only
controls cover nonnormal all-ones inflation, neutral/expanding matrices and
bad inputs. This is adapted explanatory arithmetic after the negative target.

## Dated saved-only repair after diagnosis001

The first saved reduction completed and is retained. Its positive-resolvent
proposal checked e>0 before adding the declared positive floor, rejecting the
natural possibility of an exactly zero output-gauge coordinate. It also did
not retain that solve before rejection. This is an implementation limitation,
not a failed mathematical contraction criterion. Diagnosis002 keeps the same
input, decomposition, precision and all resource caps; retain every finite
solve coordinate first, require e>=0, then add the original positive floor.
Negative/nonfinite solves still fail without clipping. The new zero-gauge and
negative-solve synthetic control passes. This only repairs a floating proposal;
no rigorous proof gate, target domain or native acquisition changes. New
exclusive attempt-002-diagnosis; preserve attempt001. One rerun only.
