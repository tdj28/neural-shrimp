## 2026-10-15 06:05 UTC: FBS156-192 result: P-F2 holds at log10 T -0.156 (last ratios 4.572 and 4.643); DSS3 refused at its seeds (the period-6 hook is not effectively one-dimensional)

- **FBS156-192.** Pod 9emvmdzf4axl8t (SECURE H100 80GB HBM3; runner rc 0; commit b5af43a7). s_192 = 0.06510338547902846 (converged, clean bracket 5e-15).
  - Superstable ratios at -0.156: 10.21, 3.713, 4.572, 4.643 (r3 4.5716 (worst case 1.02-4.64) r4 4.6429 (+- 0.0070 from the s48 bracket)).
  - P-F2 (frozen at 01:30 UTC: the last two superstable ratios computed at a new temperature within 5 % of 4.669, at least 4.436) **holds at -0.156**: 4.572 and 4.643. The first depends on the imprecise s_24 (worst-case range above); the second only on s_48, s_96 and s_192, which are precise.
  - Band 384, after FBD's 384->768 flip, will make the last two ratios both precise.
  - At -0.152 P-F2 cannot be scored: no cascade (FB152e).
- **DSS3a/b** (period-6 hook of the philosophy map; pods ran commit d89d5bc9). Both refused at the seed, as the tool requires a clean real leading multiplier:
  - seed (0.080471, -0.071235): leading multiplier complex, -0.0195 + 0.264i (next modulus 0.264);
  - seed (0.080471, -0.068588): mu = -0.261 with next modulus 0.099, below the cleanliness factor 10.
  - So the hook's period-6 orbits are not effectively one-dimensional: two multipliers of similar size, sometimes a complex pair. Superstable curves (leading multiplier 0) are not defined there, and Singer's theorem does not apply. This is the regime where two-dimensional shrimp organization (as in Figure fig:zooms, the period-5 x period-6 crossing) can occur. A two-dimensional test is needed instead.

