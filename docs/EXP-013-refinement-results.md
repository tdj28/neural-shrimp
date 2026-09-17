# EXP-013 numerical follow-up: the original fine trace agrees with finer solvers

All ten planned traces complete, and all thirty newly required comparisons
pass the frozen scalar and waveform thresholds. For both the selected failed
candidate and its matching sham, all five new methods agree mutually and
with the original `dt=0.00125` trace. This resolves the original fine trace
under the follow-up's declared diagnostic rule. It does not change the failed
`dt=0.0025` versus `0.00125` comparison or qualify the original EXP-013 cohort.

The two cases were selected after seeing EXP-013's numerical failure. New
methods are RK4 at .000625, .0003125 and .00015625, plus DOP853 at
(1e-12,1e-14,.025) and (1e-13,1e-15,.0125). Initial states, parameter changes,
reference, duration, saved sample grid and scoring definitions are unchanged.
All original labels, policy lists and winning actions remain unchanged.

| New method, failed candidate | RMSE change from original fine | Activation RMS difference | Maximum activation difference |
|---|---:|---:|---:|
| RK4 .000625 | 4.206e-7 | 6.447e-5 | .001630 |
| RK4 .0003125 | 4.798e-7 | 6.874e-5 | .001738 |
| RK4 .00015625 | 4.798e-7 | 6.875e-5 | .001738 |
| DOP853 1e-12 | 4.840e-7 | 6.905e-5 | .001745 |
| DOP853 1e-13 | 4.768e-7 | 6.853e-5 | .001732 |
| Declared maximum | 1e-4 | .001 | .01 |

The sham's maximum old-fine/new activation difference is below 3.26e-10.
Of all 34 saved comparisons, 32 pass. The two retained failures are the
candidate's original adjacent native pairs: .005/.0025 and .0025/.00125.
Their maximum activation differences are .53675 and .02665 respectively.
The first is an additional saved-case diagnostic; the second is the original
cohort-blocking waveform failure. No convergence order is imposed or inferred
from the observed ratios, and shared-phase scores remain descriptive.

Acquisition uses privately pushed commit
`406b89553b956bac1b97ef05413c799e383a9103`, plan SHA-256
`884e04f11efcf57e0607bbec97f87a8cfc8716ac1436c35bfd90ea3335b28800`.
The producer/finalization timer is 1.685 seconds, excluding preflight input
and source validation. All ten owned workers terminate cleanly. The frozen
saved-array verifier reproduces ten new and six original full score packets
and all 34 comparisons. A separate 323-check audit verifies source/inventory,
raw identities and independently recomputes waveform comparisons and flags;
it reads hash-bound aligned scalar scores, while the frozen verifier replays
their scoring primitive. No new fits or hidden outcome substitutions occur.

Raw evidence is retained at
`/data2/neural-shrimp-data/raw/EXP-013-refinement/attempt-001/`: 33 artifacts,
495,511 bytes. Receipt SHA-256:
`6463eb18784d6024f1036a0f6e6ef27e0bebbce8cc801478da008927db3f8bce`.
[The compact receipt](EXP-013-refinement-summary.json) binds the summary and
audits. The [original result](EXP-013-results-audit.md) remains unqualified,
with no full-cohort means or bootstrap intervals. Any new cohort analysis
requires its own explicit, outcome-aware follow-up plan; this diagnostic
alone authorizes no retrospective qualification or geometry-utility claim.
