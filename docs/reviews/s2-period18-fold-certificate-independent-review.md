# Independent review of the primitive-period18 fold certificate

**Pass.** One bounded review from `f2284ebb5d5aa8cfa816da65ad15d04a2ec38224`
reproduced the successful certificate byte-for-byte, independently checked saved
interval arithmetic and exact input/source bindings, then proved the same root
at384 bits with radius2^-110 and five Newton proposals. Its entire27D box is
strictly inside the producer's256-bit/radius2^-100 box by exact-Fraction tests.
The complete manifests and exact interval endpoints are in the companion JSON.

The stronger proof has Krawczyk contraction below7.326e-15 and left-border
inverse defect below2.796e-16. It excludes proper divisors1,2,3,6,9;
`a` is approximately−0.015615973354 and `c` approximately−8.0001160068, both
strictly negative. The scalar quadratic coefficient is `c/2`. The actual hidden
rate is approximately75.764292031115, with the fixed exact output rate
1848862009810323/2251799813685248. The validated deflated bound is
`||Q^32||inf < 0.035627`, proving all12 complementary multipliers lie strictly
inside the unit disk. The entire producer augmented box lies within a normalized
2.138e-11 of the selected dyadic numerical proposal (declared cap1e-7).

This establishes a local nondegenerate fold of primitive18 cycles for the
exact-real GD family. The fold itself has a neutral +1 multiplier. Proximity
does not prove connection to the earlier attracting interior orbit, and this
result does not certify the whole shrimp boundary, finite-detuning branches,
chaos, floating-trajectory fidelity or training utility.

The original radius2^-80 attempt remains immutable and explicitly bound. Its
root/simplicity/primitivity/coefficient gates passed but the complement norm
gate did not; one disclosed same-target radius amendment to2^-100 was needed.
That failure concerns conservative interval enclosures and does not establish
instability. The independent source review preceded execution; this outcome
review was designed after the producer summary. Replay and higher precision
share the mixed-jet implementation and Arb/FLINT. The independent work is the
mathematical source review, provenance reconstruction and saved arithmetic;
no separate AD or proof-assistant check is claimed.

Execution took3.691s internally and3.954s externally, CPU/BLAS1, actualnice10,
60,472KiB peak RSS. All5,647,757 bytes, manifests and read-only modes were
verified; all three owned PIDs exited. Four synthetic review controls passed
before the single target review. No retry or further proof run occurred.

Raw review: `/data2/neural-shrimp-data/audits/s2-period18-rigorous-fold/attempt-001`.
Outer manifest: `95909be677227d553581e82c8de6932eedd62720ec016b9251b8a6d65776b666`.
