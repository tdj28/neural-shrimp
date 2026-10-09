# Fixed curvature mechanism pilot: exploratory outcome

All six old period-two references requalify, but the fixed top-curvature plus
isotropic-bulk approximation fails the engineering screen. This is a direct
prediction failure, not merely an inconclusive conservative error bound.

| Reference | Actual leading pair | Predicted pair | Normalized pair error |
|---|---|---|---:|
| S2-28011-stable-real | real_pair | complex_conjugate_pair | 0.353639 |
| S2-28011-stable-complex | complex_conjugate_pair | complex_conjugate_pair | 0.080197 |
| S1R-28002-stable-real | real_pair | complex_conjugate_pair | 0.728292 |
| S1R-28002-stable-complex | complex_conjugate_pair | real_pair | 1.934852 |
| S2-28011-crossing-neighborhood | complex_conjugate_pair | complex_conjugate_pair | 0.297182 |
| S1R-28002-crossing-neighborhood | complex_conjugate_pair | complex_conjugate_pair | 1.757615 |

Every error exceeds the unchanged0.02 gate. Both real controls are falsely
classified as complex; the S1R stable-complex slot is falsely classified as real.
Mean error0.858630 improves on the stronger commuting comparator1.202295,
but both are poor in absolute terms; mean improvement cannot rescue failed
accuracy, character, angle and floating-attribution gates. None of the four
complex slots gains positive floating spectral attribution.

The panel completes in0.885s (wrapper1.336s), with23 counted fused analytic
returns including one reused from the profile:22 new panel calls plus the
profile call. The single-phase S2 slot saves one exact cache call. The panel
uses12 Hermitian eigensystems,6 general eigensystems and6 basis SVDs; the
synthetic resource profile adds2/1/1 engineering decompositions. No trajectory,
corrector, new reference, GPU, paid compute or holdout. Complete raw/source
inventories, resource observations and process exits are in the companion
receipt; all four directories are immutable and full manifests verify.

Source36d97ec6; root reviewfbd0bce0; profilefbd0bce0; panelf0d556d4.
Independent saved-arithmetic interpretation is pending at this record.
The next useful question is which information the single-direction model
discarded. This result alone does not reject richer curvature mechanisms or
establish anything about LLM training or inference.
