# Mixed-precision phase domains

This pure helper constructs and verifies lower-precision Arb input boxes containing exact nominal phase boxes. It does not load a model or certify a derivative, root, period, attraction, or token readout.

For each exact saved dyadic centre coordinate c and exact positive phase radius r, the target-precision centre m is an arithmetic-rounded midpoint. The construction encloses r + |c-m| as its radius. After serializing and decoding the ball, it checks the actual receiver endpoints using exact Python Fraction arithmetic:

    actual_lower <= c-r,   actual_upper >= c+r,
    actual_radius >= r + |actual_midpoint-c|.

The comparison never discards centre bits, substitutes a binary64 value, or assumes a nominal radius dominates the rounding error. Coordinates and phases must match explicit expected dimensions. Each phase has one exact scalar radius, matching the mathematical phase-box convention of the shooting gates.

`build_phase_domains` returns the encoded balls, separately packed actual decoded balls, exact coordinate witnesses, backend versions, helper source hash, and canonical artifact hash. `decode_verified_domains` requires an independently pinned artifact hash and the original exact nominal data, recomputes every endpoint witness, and returns the live balls that must actually be evaluated. The helper source hash is provenance inside the pinned artifact; the receiver independently recomputes containment rather than trusting that declaration. Its nominal-domain hash convention is specific to this helper: later integration must explicitly bind the same exact centres and radii to the shooting gate's distinct domain-hash convention, not assume those hash strings coincide.

Arb may widen a stored magnitude on reconstruction. Encoded and actual decoded ball hashes are therefore distinct. Further reconstruction or use of conservatively wider receiver balls is valid only after `containment_witnesses` checks and records those actual balls and their new hash. Equality of serialized radii is not required. Midpoints alone are never an admitted receiver domain. The decoder restores the caller's precision context; the subsequent evaluator must explicitly run at its declared precision and use the returned full balls.

Using 1024-bit derivative evaluation on such a containing domain while retaining 2048-bit nominal centres, residuals, and gate calculations can be mathematically sound. It still requires a rigorous derivative enclosure over every actual input box, all state directions, the same exact-real map and parameters, and the correctly bound 1024-bit prefix cache. This helper proves only domain containment; measured point-JVP uncertainty does not establish a whole-box enclosure or a computation-cost saving.

Validation: 35 synthetic tests passed without a checkpoint/model call. These include a 2048-significant-bit centre evaluated at 1024 bits with nominal radius below the rounding ULP, both rounding directions and signs, extreme dyadic exponents, outward serialization roundtrips, omitted coordinates, nonexact/nonfinite inputs, externally pinned artifact tampering, and independently rejected narrowed balls. A separate reviewer additionally checked 288 exact-fmpq endpoint cases at 16, 64, and 1024 bits.
