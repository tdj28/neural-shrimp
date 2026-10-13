# Independent review of the conditional shooting wrapper

Reviewed `shooting_certificate.py` SHA-256
`a4aca57acf7aaab33c77745ea6170935dd6b4dac40c9c982a063743f98727c3e`
and its tests SHA-256
`70f1f7e544e33b578df41f249023c5513faae9be6b1b064eae24390595e26372`.
The delegated review was read-only; no evaluator/model/receipt call or Git
mutation occurred. Dependencies reviewed here are `shooting_gates.py`
`9c967c0dcb9783df53f33c05eb897050f4e8971f66767abbc8a01a8b1b85cf1f`
and `proof_gates.py`
`ee6678b7dd5153ef095fd73d5711fcfc605f7a228c92b1335a9541cc3d7b2c6a`.

No mathematical or evidence-boundary blocker found in this conditional wrapper.
It first validates the complete shooting contract, exact nominal domain,
point-image/whole-box-Jacobian enclosures and dyadic proposals through the
reviewed root gate. The primitive check must reproduce the same exact domain
hash. The same admitted, ordered Jacobians form the signed interval product
`J_(p-1) ... J_1 J_0`; every partial product and its phase matrix are hash-bound.
The wrapper checks input/context immutability by contract, and checks context
precision/threads again before returning. It does not authenticate external
files or independently prove that the supplied enclosure contract is true.

The mathematical implication is correctly limited. A shooting root has its
corresponding phases inside the declared boxes, so this product encloses its
return-map derivative without requiring each entire box to map into the next.
The outward norm of an actual matrix power below one proves local attraction
of that root. Proper-divisor phase-box separation establishes primitivity only
conditional on existence of that same root. The root gate establishes a unique
tuple in the full product box, hence local uniqueness; it does not exclude
other return-map fixed points whose intermediate phases leave those boxes.
Neither primitive separation nor the attraction norm can replace the root gate.

Every ordinary failed sufficient condition remains visible, and the aggregate
requires all three flags. Contract/shape/nonfinite input errors raise. The
return explicitly sets `certificate_passed=False`,
`external_authentication_performed=False`, and
`enclosure_validity_checked_by_this_module=False`, including when every
conditional gate passes. Numerical proposals or metadata therefore cannot be
promoted to a final certificate by this API.

Validation: independently ran 89 combined wrapper, shooting and proof-gate
synthetic tests (0.40 seconds, CPU1, 256-bit synthetic cases). They cover a
repeated fixed point, an unstable isolated shooting root, failed inclusion,
noncommuting phase order, a nilpotent nonnormal product requiring power two,
proper divisors of eight, interval width, contract mismatch and invalid power
plans. An additional independent exact-Fraction calculation generated 64
signed two-by-two phase products of periods two through four, matching all
retained product hashes and 192 reported matrix powers exactly. All retained
aggregate flags matched the conjunction and remained non-certificates. This is
algebra/integration validation, not evidence about the target Qwen orbit.
