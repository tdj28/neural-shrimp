# Independent audit of box-derivative-001

Completed-attempt review, 2026-10-13. Read-only retained artifacts and exact rational arithmetic; no checkpoint load or target-map replay.

The engineering pilot passed its declared gates. It produced finite whole-box enclosures for all 896 output coordinates and the 32 specified state-derivative columns at phase zero. Every primal coordinate and every entry of the eight shared derivative columns overlaps the earlier point enclosure using the original saved exact dyadic endpoints. The result does not enclose the remaining 864 derivative columns or supply an orbit certificate.

## Provenance and domain admission

- Raw attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/box-derivative-001`.
- Receipt SHA-256: `1a5ba00856078b78de611c46f9045c5229970e190f0d63b8e1fca0f7d1d37c3f`.
- Source commit: `10a945d589c417dc4d5085d6a6902dd17578937e`.
- Exact driver, plan and launcher source review: `boxed-pilot-review.md`.

All 51 artifact sizes/hashes and all 38 source-manifest blobs were checked; the latter also matched Git objects at the recorded commit. The two upstream inventories were independently rechecked in full: 22 files for arb-pilot-002 and 27 for signed-pilot-001. The expected receipt, proposal and shared-point output hashes match. Successful exit, no timeout, CPU1, 18 GiB and the 3600-second cap are retained in the launcher evidence.

The domain audit verified the internal construction, witness, source and file hashes; all 896 nominal centres equal the pinned binary64 phase-zero proposal as exact dyadics. The radius is exactly 2^-800. Every actual receiver coordinate encloses the entire nominal interval, with its centre-rounding error and radius checked by exact rational endpoint inequalities. The audit separately binds the actual receiver balls after the final reconstruction; no equality of pre- and post-reconstruction radii was assumed. Every saved witness field and margin agrees with those exact endpoints.

The 32-by-896 saved direction inventory contains exactly the basis vectors e0,...,e31, with zero radius and no omitted state coordinates. Its hash and the actual input-file hash are tied to the saved output. Every one of the 28,672 derivative entries and 896 primal entries has a valid finite dyadic midpoint and nonnegative finite radius. Progress records include all 24 ordered decoder-layer starts and ends, followed by the saved complete step and completion marker. Full-vocabulary execution is supplied by the reviewed inherited architecture and pinned model dimensions (151936), not inferred from output dimension alone.

## Every derivative row

Each row retains all 896 finite coordinates. The first eight rows also passed 896 exact point/box overlaps each. Remaining rows have no prior point comparison. Values below are rounded displays of exact saved maximum radii.

| Direction | Maximum radius | Shared point overlap |
|---:|---:|---:|
| 0 | 1.3919002723E-76 | 896/896 |
| 1 | 7.6747453875E-76 | 896/896 |
| 2 | 1.6841382147E-76 | 896/896 |
| 3 | 7.8019428842E-76 | 896/896 |
| 4 | 2.6229400030E-76 | 896/896 |
| 5 | 4.4025898877E-76 | 896/896 |
| 6 | 4.3086539675E-76 | 896/896 |
| 7 | 1.1412046175E-76 | 896/896 |
| 8 | 6.6262595193E-76 | Not compared |
| 9 | 4.1406003114E-76 | Not compared |
| 10 | 3.5453822252E-76 | Not compared |
| 11 | 2.0495455303E-76 | Not compared |
| 12 | 2.4697512320E-76 | Not compared |
| 13 | 7.1499130889E-76 | Not compared |
| 14 | 3.7115383732E-76 | Not compared |
| 15 | 6.7981142607E-76 | Not compared |
| 16 | 8.6461555684E-76 | Not compared |
| 17 | 1.7544974281E-76 | Not compared |
| 18 | 1.4234361188E-75 | Not compared |
| 19 | 9.5950863035E-76 | Not compared |
| 20 | 2.9161649536E-76 | Not compared |
| 21 | 3.3472349282E-76 | Not compared |
| 22 | 2.8001174785E-76 | Not compared |
| 23 | 1.3424447472E-76 | Not compared |
| 24 | 3.9941038234E-76 | Not compared |
| 25 | 3.6004699349E-76 | Not compared |
| 26 | 1.1011046116E-75 | Not compared |
| 27 | 2.5982159931E-76 | Not compared |
| 28 | 2.1720926180E-76 | Not compared |
| 29 | 1.2448820870E-76 | Not compared |
| 30 | 7.9659536125E-76 | Not compared |
| 31 | 3.5027340517E-76 | Not compared |

The maximum primal radius is approximately 5.2556341201e-79. The largest derivative radius is approximately 1.4234361188e-75, in column 18. These bounds are much wider than the earlier point-only enclosures; this reflects the enclosing computation on a box and does not measure actual dynamical sensitivity by itself. The original saved endpoints, rather than additionally widened decoded balls, were used for all compatibility comparisons. All 896 primal overlaps and all 7168 shared derivative overlaps passed.

## Cost and scope

The Arb forward call took 988.569604 seconds; driver elapsed time was 999.271292 seconds and launcher elapsed time 1000.089299 seconds. Peak RSS was 8,315,588 KiB (about 7.93 GiB). These measurements support a separately bounded next engineering plan; they do not establish a speed or memory guarantee for more columns, another phase or another algebraic readout.

The tiny box is centred on the original floating prototype and is not an isolating orbit-box proposal: its radius is vastly below that prototype's measured exact-real residual. It remains separate from the newly refined centre tuple. No Krawczyk inclusion, complete Jacobian, primitive-period separation, attracting monodromy bound or rigorous text-label cycle was evaluated. Every status retains `certificate_passed: false`.

All checks passed on the first completed-attempt audit execution. No raw output or criterion was changed.
