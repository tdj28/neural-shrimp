# Task38 addendum: figure/caption and PV static review

This adds item 9 and a static review of the parent's saved PV/steering audit. No numerical audit was rerun; no model, array computation, rendering, producer edit or Git mutation was performed. PNGs were viewed directly. The only executed new script hashes files, reads JSON and converts saved agreement fractions to counts. The complete earlier geometry findings remain in `geometry-review.md`.

Exact text snapshots: article main `b8413881e7ce97a9e58adcf1bf7d78a8cebb0514`, p6-only `d49718d3294f180e67f94de8194536458655e918`; report `651a50059a145e2bb0526f9675bcc41e9cb895a8`; blog SHA-256 `4c6646cae55b08d5d3d85dd19ef9d0ad11348a82bef504fc20f3be92ecd21926`. `figure-pv-manifest-001.json` pins the exact sources/results/images inspected. All ten current PNG files (five article and five report, including the additionally inspected fold figure) match their assigned-commit blobs. The four requested figure identities are `fig-repro`, `fig-real-shrimp`, `fig-p6-shrimp`, `fig-p6-kite`; counting the assigned article's figure environments gives 12, 20, 22, 23. The brief's 19/21/22 references are stale.

## Material findings

### F1 — must-fix classification mismatch in the PV audit/statistics

**Where/quote:** `/tmp/task38-root/pv_steering_audit.py:22,25`, `bc=base['lyapunov']>.001` and `chaos=lam>.001`. Article 529–538 and report 5211–5220 call the resulting shares, Jaccards and frequencies “chaotic”. Report 165–166 defines this label as positive finite-time exponent **and no detected period**.

**Evidence:** The saved result explicitly contains positive-exponent cells with detected periods: base 12, starts s1/s2/s3 respectively 10/9/14. The base's `chaotic_count=1152` therefore differs from `chaos_without_detected_period_count=1140`; starts are 1130/1151/1130 versus 1120/1142/1116. Frequency, agreement and Jaccard are all calculated from the same exponent-only masks. The code correctly computes those masks, but their scientific names do not match the declared classification. The result's own overlap fields expose this; no fresh array audit is needed to establish the mismatch.

**Replacement/action:** Preserve these values as “exponent-threshold mask” statistics, or separately calculate and name `(period == 0) & (lyapunov > 1e-3)` statistics before using the paper's chaos label. The existing `all_condition_gates_pass` checks configuration/grid/finite/prefix fields, not classification consistency; it must not be used to close this issue. This review does not supply revised Jaccard/agreement/frequency values from a fresh run.

### F2 — should-fix: “The start does not matter” is broader than the result

**Where/quote:** article 533–534, report 5215–5216, blog 2027: “The start does not matter ... maps agree ... 98.9 to 99.0%”.

**Evidence:** The saved s1/s2/s3 threshold-mask agreements are 16228/16384 (99.0478515625%), 16211/16384 (98.944091796875%), 16216/16384 (98.974609375%). But detected-period label agreements are 14570/16384 (88.92822265625%), 14168/16384 (86.474609375%), 15260/16384 (93.1396484375%). An equality of coarse positive-exponent masks says little about which periodic orbit is selected. Even matching period labels do not establish matching orbit identity. These are independent facts visible in the saved result, not a claim that every disagreement represents a distinct converged attractor.

**Replacement:** “For these three random starts, the positive-exponent mask agrees with the base at 98.9–99.0% of cells, while detected-period labels agree at 86.5–93.1%; the coarse region of positive finite-time expansion is comparatively insensitive to these starts.” If F1 is fixed by recalculation, substitute the resulting chaos-mask values and retain the period comparison.

### F3 — should-fix extension of geometry G7: HP5's image uses the mismatched rule too

**Where/quote:** article HP5 caption 1015–1020; report corresponding `fig:realshrimp` caption; `tools/plot_real_shrimp.py:142–144`: `chaos = clam > 1e-3`, then family and other-periodic masks exclude `chaos`.

**Evidence:** The displayed HP5 panels agree visually with the stated 512x512 layout, axes, cyan followed-domain edge, four cusp markers, magenta fold self-crossings and blue fourth-leg samples. But the plotting source assigns positive-exponent/detected-period cells the chaotic color rather than the stated period-family/other-periodic colors. The already-completed original-array audit established 33 such cells inside the nearest-cell FA mask, 32 in the P5 family; the global stored HP5 `chaotic` flag uses the no-period criterion. This extends G7 from the brief's percentages to a paper figure. No new array reduction was performed here.

**Replacement/action:** Use the declared period-aware chaos mask for panels (a,b), or explicitly caption the colors as positive finite-time exponents that override detected periods. A period-aware redraw should accompany corrected G7 percentages. The separate trace/Type-A and census panels were visually checked only; their older numerical audits were not repeated.

### F4 — should-fix caption scope: the periodic reproducibility curves have exceptions

**Where/quote:** article 621 and p6-only 577, Fig. `fig:repro`: “the periodic ones stay at the rounding level”.

**Evidence:** The PNG shows individual periodic/chaotic curves, with medians and quartiles; its source loops over all 192 cells of each label, adding only the median of the 64 other cells. Thus the requested “one line per cell” redraw is present. The blanket caption sentence conflicts with article 632–633, which already reports that only 182 of 192 periodic cells stay within 1e-2. The image has exceptional blue traces above the rounding band. No old REPRO numerical audit was repeated.

**Replacement:** “The chaotic runs mostly separate to order one; the median periodic distance stays at the rounding level, with exceptions.” Keep the exact 190/192 and 182/192 results in the adjacent prose. Panel (c)'s retrospective minimum-nonzero-distance definition matches implementation; the source module docstring's “first nonzero” description is stale, whereas the figure caption correctly says “smallest nonzero”.

### F5 — note: the unstable kite branch looks solid despite the dashed encoding

**Where/quote:** article 1213, report 2484, kite caption: “solid where the orbit is attracting and dashed where not”.

**Evidence:** In the actual PNG, the middle unstable S branch between the two folds is a visually continuous gray line; the upper unstable tail is visibly dashed. `tools/plot_p6_kite.py:145–148` requests dashed styling independently for every adjacent pair. Restarting the dash phase on each very short segment can make the dense section appear solid. Color still distinguishes it from the blue stable portions, so this does not reverse the coexistence inference. The green interval starts at the flip as intended, and its bounds, folds, A/B/R intersections and located C marker match the already-audited JSON.

**Replacement/action:** Draw each contiguous unstable segment as a single polyline so the dash pattern remains visible, or caption/legend the stable branch as blue and unstable branch as gray. This is a visual consistency fix, not a new dynamical finding.

## Checked, partial and skipped

- **P6 main figure checked:** panel (a,b) shows HR10's 640x640 window and the correct axes; panel (b) has the cyan pseudo-crossing and C star; panel (c) contains the six cusp entries, with the lower pair visually overlapping as captioned. Its JSON identifies HR10, lists all 18 curves and all six located records, and preserves four rejected rows. The earlier G3 gate exceptions still apply to four accepted extended-curve points. No new claim that the full boundary validates follows from this visual pass.
- **Kite checked/partial:** actual image, JSON corners, branch temperature -0.15395151820007266 and coexistence interval [0.07290699285213686,0.0729915446873536] are consistent with the completed raw audit. The caption's blanket “not a bifurcation” reading should retain the earlier checked-cells limitation. The figure JSON does not record `--scan` or resolution, so it alone cannot bind the background specifically to HR11; the separately audited HR11 does have 512x512 cells. The image's visual appearance is not an independent grid-count check.
- **HP5 metadata limitation:** no adjacent `fig-real-shrimp.json` exists. The supplied `docs/fa/fa-100-summary-closeup.json` records FA and its boundary gates, but not the HP5 closeup argument/hash. The figure title gives 512x512, and the earlier original HP5 audit confirms that resolution. This is a provenance limitation, not evidence of a wrong background.
- **Repro metadata limitation:** no adjacent `fig-repro.json` exists. Source and PNG agree on the redraw and displayed 192/192/64 split. Source reads the original repro JSON/NPZ; those old arrays were not rerun in this addendum.
- **PV static calculations checked:** fraction/Jaccard construction, the 19-prefix frequency denominator, text/vector split (16/3), row-mean profile and p14-to-p14b selection are internally consistent for the exponent-only mask. Start comparisons use exact stored period labels. Prefix/argument gates are not trajectory convergence, precision or chaos-certification gates. `args_common` is a one-sided mismatch dictionary and would not detect a missing base key in a variant; no actual missing-key discrepancy is established here.
- **Steering checked:** source `layer_states`/`build_vectors` uses final-token states after zero-based decoder layer 12, averages the eight concept minus eight neutral states, and normalizes by the equal-text mean of per-text non-padding-token mean norms. Saved norms 8.095698612656449 and 188.99483498183588 give 0.042835554809921225 and factors 1.459068296823466–1.809244688061098 over alpha .0625–.0775. The stated 8.1,189,.043,1.5–1.8 are ordinary rounded values, not upper-bound claims. Activations were not recomputed.
- **Skipped by design:** model/derivative/certificate recomputation, basin-pair2 missing record, any fresh raw-array numerical audit, rendering, and edits to producer or paper assets. P6-only's older numerical/caption state remains G4; this pass does not synchronize it.

Evidence: `figure-pv-pins.py`, `figure-pv-manifest-001.json` (SHA-256 `c260ce4e9178d9ec82bbbea4f1e9267c3f4b2d271d80e883e44318f8546307c2`) and the prior hash-bound geometry package. A final addendum manifest pins this report and the additional inspected steering source without altering earlier evidence.
