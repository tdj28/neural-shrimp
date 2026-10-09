# Period-18 interval fold attempt001 — partial result

The exact small-network map has a unique normalized augmented root in the
reported27D box, a primitive18 orbit there, an algebraically simple multiplier
+1, and nonzero parameter/quadratic fold contractions. The certificate is
**not fully passing**: the remaining multipliers were not enclosed strictly
inside the unit circle by the conservative deflated-power test. This does not
show they are unstable.

The requested radius was2^-80 at256bits, with five Newton proposals and powers
through4096. Krawczyk contraction is below7.866e-6; the transpose-left-border
inverse defect is below4.152e-7. The saved a and c intervals exclude zero.
The smallest deflated-power bound was12.114 at power8; subsequent interval
bounds grew. The full partial mathematical result, all returned evaluations,
inputs, source snapshot, and unsuccessful stability bounds are retained.
Target execution took1.721s internally/1.944s externally, peakRSS53,988KiB,
147 ticks. The preceding synthetic profile projected1.744s for seven jet
passes. All files are hash-verified/read-only and owned processes have exited.

See `s2-period18-fold-attempt001-receipt.json` for exact artifact identities and
serialized bounds. This is evidence rung0 under the all-gates rule, despite
the individually successful interval gates. Independent outcome review is
pending. No radius retry occurred within this attempt. No whole-window,
center-to-boundary branch connection, or practical-training claim follows.
