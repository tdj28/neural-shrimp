# Independent completed-evidence audit: center-secant-001

**Passed**, as a centre-refinement setup result. No model evaluation or dense
inverse-secant kernel replay was performed by this audit. The final receipt was
present before any result was admitted; no partial result was used.

- Source commit: `ac376355a37f6dc1368128ed8c9519a7bd2afb2a`.
- Receipt SHA-256: `de7470e715c2988fc1674030d69bb67685dd1571989d5b03bad98ef7ed23f956`.
- Driver SHA-256: `c828ebc744eb95795e9051d5dbe0b8361828ebdfb13688afae5c1f60f7694132`.
- Immutable attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/center-secant-001`.
- Audit script: `/tmp/neural_shrimp_center_secant_audit.py`, SHA-256
  `5d824967da2e9c4f9625b36ea33493c033dcb000ce4d206da755438b1000be31`.
- Machine-readable audit: `/tmp/neural_shrimp_center_secant_audit.json`, SHA-256
  `9378283dc53265b6ad6ce987f121ad58c61cb08cdde8eb049e7e68ada6bb8bab`.

The independent CPU1/nice10 audit completed in 94.732
seconds under its 180-second watchdog. It checked all 107 retained artifact
sizes/hashes, the complete inventory without extra files, and all 69 captured
source files against their exact Git blobs. The one-candidate plan, source,
2048-bit backend, pinned assets, exact map parameters/prefix/steering and
receiver cache are bound. The reviewed complete-history admission was invoked,
without kernel replay: all 77 original refinement artifacts, 98 resume
artifacts, 32 prefix artifacts and 56 numerical-seed artifacts were admitted.
All four genuinely evaluated prior tuples appear once, in chronological order;
no aliased before-pass was counted as a new evaluation.

## Numerical result and every decision

The four incoming infinity-norm residual upper bounds were
`9.622657250049e-15`, `2.436146426368e-27`, `6.582941947286e-40`, and
`7.157202373766e-52`. Every original image radius was retained or conservatively
widened, and full residuals were recomputed against the correct successor.

All three chronological secants were accepted as `accepted_midpoint_rank_one`.
Their retained induced-infinity rank-one update bounds were respectively
`2.733001689534e-12`, `1.522918967825e-12`, and `5.495959627196e-12`, below the
fixed 1024 cap. Signal exceeded 128 times the retained noise in every row.
Active history advanced 0→1→2→3. The single proposal selected the
`accelerated` correction, passed its step-cap decision, and retained three
factors after the proposal. No rejection, restart or fallback occurred; all
three decision records and the complete chosen/baseline corrections remain
in the immutable attempt.

The one freshly evaluated candidate reduced the residual upper bound from
`7.157202373766E-52` to `2.155327583370E-64`. Both
`upper_bound_decreased` and the stronger exact comparison
`after_upper < before_lower` are true. The rigorous ratio bound
`after_upper / before_lower` is approximately
`3.011410703252E-13`. These displayed decimals are rounded
summaries of exact dyadic bounds; the saved lower and upper bounds are distinct
and were compared exactly, not rounded to these displays.

The audit checked all 8×896 exact candidate coordinates and every full image
and zero tangent, original phase-input/output hashes, cyclic successors,
residual endpoints and norm summaries. It independently checked each centre
difference, scaled denominator enclosure, every `w` enclosure, retained `u/w`
midpoints, exact rank-one norm bounds, and low-rank/defect interval consistency.
The final exact midpoint construction was reconstructed at the recorded
2048-bit precision. The expensive baseline A/B actions and the complete
`u = scaled_s − R scaled_y` computation were **not replayed**; those parts rely
on the immutable reviewed kernel and its saved witnesses. This distinction is
intentional and is not presented as independent recomputation of all arithmetic.

## Complete fresh-phase and resource record

| Phase | Seconds | Maximum primal radius |
|---:|---:|---:|
| 0 | 165.089874895 | 6.721279374548E-451 |
| 1 | 130.547507139 | 4.417040614042E-451 |
| 2 | 142.845998720 | 6.272923844829E-451 |
| 3 | 159.784101338 | 4.313546466439E-451 |
| 4 | 159.684565125 | 5.946800971321E-451 |
| 5 | 159.736237407 | 5.791383544909E-451 |
| 6 | 143.581651380 | 1.870677025219E-450 |
| 7 | 120.751105219 | 7.599952603207E-451 |

All eight phases completed in their original order. All 24 decoder start/end
stages per phase were retained and checked. New evaluations: 8; cumulative
authenticated phase evaluations: 40. Mean phase time was
147.752630153 seconds; maximum was
165.089874895 seconds; phase total was
1182.021041223 seconds. Driver elapsed time was
1421.306978936 seconds and launcher elapsed time
1422.103681121 seconds, with exit 0 and no timeout under the
1800-second cap. Peak RSS was 5,954,212 KiB
(5.6784 GiB).

This establishes a substantially more accurate candidate centre at the same
assigned map and parameters. It does not establish a root, primitive period,
attraction, phase-box derivative enclosure, or shrimp geometry. The decision
name `accelerated` describes the chosen numerical formula; without a matched
new fixed-R comparison it does not demonstrate that this method accelerates
convergence or wall-clock time. Every certificate flag remains false.
