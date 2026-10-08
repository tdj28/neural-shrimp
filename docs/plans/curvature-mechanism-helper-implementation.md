# Curvature mechanism helper: source and synthetic preparation

Date: 2026-10-08 UTC. This bounded source-only unit began at 13:51:38 UTC
with a 20-minute limit. Base integration is `186ee656`; corrected design is
`403aff7e76e747b7968c402aeaac4e425f21e59b`,
`docs/plans/curvature-subspace-mechanism.md`. No empirical input arrays, target
files, model, profile, acquisition, trajectory or selected-panel evaluation
were opened or run. Independent source review and empirical acquisition are
pending; this card does not authorize either acquisition or a larger-model run.

## Implemented scope

`src/neural_shrimp/curvature_mechanism.py` is a supplied-array helper, with no
I/O or model imports. `analyze_curvature(gradients, hessians, rates,
actual_monodromies=None, observer=None)` accepts finite float64 arrays with
2 <= dimension <= 72 and positive diagonal rates. Gradients are retained as
inputs, not used to infer orbit closure. Closed-orbit qualification, original
model/control identity and exact input provenance belong to a future caller.

Two explicit symmetric eigendecompositions fit largest algebraic curvature
and trace-matched isotropic bulk, after the frozen symmetry and top-gap checks.
The helper retains the full modeled, aligned and orthogonal spectra, including
bulk multiplicities. Fixed leading-two selection rejects mixed/nonconjugate
pairs rather than substituting a favorable complex pair. Near-real complex
pairs and a tie across the second/third magnitude cutoff also abstain. The
aligned threshold in the design is a limiting-model convention: within
`1-c*c <= 1e-10` the reported spectrum is the one-dimensional aligned limit,
not a claim of exact rank-one collinearity for arbitrary floating vectors.
Such a case cannot receive complex-pair attribution.

One explicit general eigendecomposition includes left and right vectors. One
full right-eigenbasis SVD retains U, singular values and Vh. All decomposition
inputs, starts, returns and counts are exposed; caps are 2 eigh, 1 eig and 1
SVD. No hidden operator-norm or condition-number decomposition is used.
Residuals use explicit Frobenius normalization, and the complex-left pairing
uses the adjoint and conjugate eigenvalue. The diagonalization residual enters
`delta_diag + kappa2 * delta_op_total`; the returned expression is only a
floating diagnostic, not a rigorous spectral enclosure, validated
invertibility test or eigenvalue-count theorem. Per-reference diagnostic
completion and positive complex-pair attribution are separate outputs.

Without supplied native monodromies the target is explicitly the supplied
Hessian pair product `S_b S_a`. With both native matrices supplied in original
coordinates, the first whitened matrix becomes the target. Its Frobenius
formation mismatch from `S_b S_a` is added to the fit bound; opposite-phase
mismatch is separately exposed. Disagreement with either pair product fails
the declared elementwise absolute-plus-relative gate. This preserves the
distinction between approximate pair closure and the actual native derivative.
Exact-real Floquet interpretations remain conditional on an actual orbit.

Controls retain strengths while changing only orientation. The better of the
two commuting errors is labeled an oracle, not a deployable predictor. Complex
modulus is strength-determined; orientation receives no modulus credit.

## Synthetic validation and limits

The focused command was:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src timeout --signal=TERM --kill-after=5s 30s nice -n 10 /data2/neural-shrimp/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_curvature_mechanism.py
```

Final result: **13 passed in 0.13 s**, exit 0. Earlier preparation passed the
same 13 controls in 0.15 s before tightening the attribution abstention logic.
All inputs were deterministic synthetic arrays. Controls cover explicit
operation counts/no hidden norm or cond, aligned/orthogonal multiplicities,
bulk-leading selection, mixed/nonconjugate abstention, nonsymmetry and top-gap
rejection, perturbed numerical eigenvectors, external-matrix mismatch,
nonuniform-rate whitening, return retention, unordered pair assignment, and
orientation-invariant complex modulus.

This is not the six-slot empirical runner. It does not implement acquisition,
trajectory/orbit qualification, file-level durability, wall/RSS supervision,
aggregate empirical engineering gates, a runtime cost profile or paper
integration. Those require separate preparation/review and authorization. The
observer exposes operation returns to a future durable caller; the helper alone
does not persist them to disk. No empirical effectiveness, novelty, training
utility or controller benefit is claimed. A later independent reviewer should
especially check the near-aligned convention, finite precision basis
qualification, complete multiplicities and external-native formation term.

Exact helper/test/card/design hashes are in the companion
`curvature-mechanism-helper-source-receipt.json`; it deliberately omits its own
hash to avoid a self-reference. The containing Git commit binds the receipt.
