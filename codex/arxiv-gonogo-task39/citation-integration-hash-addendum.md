# Task39 integration provenance addendum

This additive note corrects the timing limitation in `integration-review.md`; the original note is preserved unchanged.

The initial integration read returned the earlier “released certificates” wording. The later hash computation read the file again after root's concurrent correction. Consequently, the original note attached the final hash to a review comment describing earlier wording. It should not be read as evidence that the final hashed bytes still contain that wording. No hash of the earlier read was captured at that moment.

A new bounded read of the opening, M1 and the closed-certificates paragraph confirms that `/tmp/task39-final-review.md`, SHA256 **c9771ad429076302a800b20fe0599bfd60b194a23fbacf710ee667063472d7db**, contains the corrections:

- The opening says “separate previously audited interval certificates.”
- The later paragraph says “The previously audited periodic certificates remain closed.” The public-release ambiguity identified by this lane is resolved.
- M1 now explicitly says that the square root describes the side where its radicand is positive and that `ab>0` is not a generic requirement of the expression. It identifies omitted transverse stability as the generic-sentence issue and restricts the proposed attracting-branch formula to the studied fold, whose stated signs and transverse stability supply the required scope.

The original integration pass otherwise stands, with its timing limitation disclosed above. This was only a targeted text/hash check; no full-review repeat, web access, numerical work, model call, Git operation or paper edit occurred.
