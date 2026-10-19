GO WITH FIXES

This is the numerical, geometry, steering and reproducibility subreview of the **exact article commit `4ca5b5e35bc766581a5e2d326858fcd0edc312bb`**, not the final integrated build/citation verdict. I found no remaining numerical error that overturns the article's main conclusions. The new close-up boundary counts pass an independent saved-data check, and the period- 5 removal is internally consistent. There are six corrections or qualifications below. The main review should retain the distinction between an original-record check and correspondence with the full report; this subreview did not newly validate every historical experiment.

All line numbers refer to `/tmp/task39-raw/article-main.tex`, SHA- 256 `30c1f059ee225dbe0c2a3462d50ca501a580d9245e3209974eb17b98f403ceec`. The complete source was read. `claim-coverage.md` records every numerical/geometry claim group, including captions and tables, its evidence class, and what was delegated or not independently replayed. No article, producer, shared log or Git file was changed. No target model, GPU or matrix computation was run.

## Must-fix numerical blockers

None established in this subreview. In particular, the newly introduced close-up boundary counts are no longer an unverified gate: the one admitted check reproduces both counts and medians. Root owns the integrated public-availability, source-bundle, compilation, figure and reference verdict. The separately reported stale README/form counts are not adjudicated here.

## Should-fix

### R1. One Table III forecast is rounded incorrectly

**Article:** line 843, last row, full-channel forecast `154.75`.

**Evidence:** The frozen `codex/confirm-fold-2/design.json` stores `arctan_G: 154.74482095152212` for the full and full-plus-transverse rows of both folds (lines 251,279,470,498 in the current owned checkout). The original result receipt retains `max_abs_arctan_error: 5.744820951522115`. Thus the prose's maximum deficit `5.74` at line 808 is correct as an ordinary rounded range end; the table's two-decimal forecast should be `154.74`. The intermediate `154.745` in the old RESULTS table appears to have been rounded a second time. This is a display error, not a changed forecast, failed gate or new passage result.

**Replacement:** `154.74` in the last-row full-channel forecast. Leave the measured counts, primary failures and 42/48 conclusion unchanged. No new numerical run was used: the exact frozen forecast and result receipt were read.

### R2. Scope the appendix's denial of computed flip coefficients

**Article:** lines 1300–1302: “We did not compute flip coefficients, so these flips are not classified as super- or subcritical.” Lines 875–876 instead report “the flip normal form computed from the network's derivatives” and amplitude ratios 1.005 and 1.003.

**Evidence:** The full report at the already-frozen snapshot `651a5005`, lines 1929–1933, explicitly computes the cubic flip coefficient for the separate doubling-bubble test. Its table lists those amplitude ratios. Context permits interpreting the appendix sentence as referring only to the preceding shrimp-boundary and cascade flips, but the unqualified wording reads as a study-wide denial.

**Replacement:** “For the shrimp-boundary and cascade flips discussed here we did not compute normal-form coefficients, so we do not classify them as super- or subcritical. The separate doubling-bubble test in Sec.VI computes its local flip normal form.” This is a scope clarification, not a claim that the bubble result is absent or invalid.

### R3. A Huginn complex-pair crossing is consistent with a Neimark–Sacker bifurcation

**Article:** lines 1264–1266: “its multipliers predict that ... leading complex pair crosses the unit circle near relaxation 1.0775, a Neimark–Sacker bifurcation ... against 1.0778”.

**Evidence:** Task 37's original-record audit correctly obtains 1.0775400117776854 from the saved pair and 1.0778010471978443 from the squared-amplitude fit. The crossing is complex, transverse and away from low-order resonances in the saved partial spectrum. But the first Lyapunov coefficient was not computed; the early float32 spectrum also lacks Arnoldi convergence flags or Ritz-vector residuals. The finite amplitude/rotation observations support the interpretation without establishing all nondegeneracy conditions.

**Replacement:** “its estimated multipliers predict a complex-pair unit-circle crossing near relaxation 1.0775, consistent with a Neimark–Sacker bifurcation, against 1.0778 from a fit of the squared amplitude of the observed oscillation.” The 1.20 accuracy loss,1.05 noninferiority and six unstable ARC fixed points remain supported by the complete prior row audit.

### R4. The cause of the non-unit multipliers at R is not isolated

**Article:** lines 1053–1056: multipliers `−0.9952` and `−1.0005` “miss−1 only because the point is the crossing of the two traced curves”.

**Evidence:** Task 38 independently reproduced the saved Newton solutions, cyclic-alignment gap 6.4%, both multipliers, and nearest-curve-point distances 1.7 and 0.3 teaser-grid steps. The point is an approximate geometric intersection, not a simultaneous solution of both flip conditions. Those diagnostics do not isolate interpolation, parameter rounding, orbit refinement and spectral approximation into an exclusive cause. This does not challenge distinctness of the two loops or the pseudo-crossing interpretation.

**Replacement:** “At this approximate crossing their leading multipliers are−0.9952 and−1.0005. The nearest saved points of the traced flip curves lie 1.7 and 0.3 steps of Fig.1's grid away.” Alternatively say the deviations are consistent with the tracing resolution; omit “only because”.

### R5. Restrict the kite-edge basin interpretation to the checked adjacent cells

**Article:** lines 1122–1124, kite caption: “the straight edge ... is not a bifurcation: on both sides of it both attracting period- 6 orbits exist”.

**Evidence:** The original pair 1 `explore/llm-soft-loop/kite/basin-check/basin-check.json`, independently gated in Task 38, verifies both fixed-start solutions and cross-initialized Newton solves at the two selected neighboring cells. The leading multipliers round to0.18 and 0.60–0.61. It supports a change of selected basin at those cells. It does not trace the whole visible edge or exclude an intervening bifurcation at arbitrary finer resolution. The missing pair 2 record from Task 38 is not needed for this local conclusion and was not newly rerun.

**Replacement:** “At the two neighboring cells checked across the straight gray edge near log10T=−0.15353, both attracting period- 6 orbits persist, with leading multipliers 0.18 and 0.60–0.61; the fixed start selects different orbits. This supports a basin-selection interpretation of the edge.”

### R6. State the transverse stability and orientation assumptions for the fold branch

**Article:** lines 1287–1290: a generic fold needs nonzero a,b and no other multiplier on the unit circle; “then the stable branch has mu=1−2sqrt(ab(alpha_c−alpha))”.

**Evidence:** The listed generic-fold conditions allow other multipliers outside the unit circle; then neither full-state branch is attracting. The displayed alpha-direction formula also has the orientation/sign of the particular fold studied in Sec.V, rather than every generic fold. The coefficients and tests in the body are not contradicted: its a,b are both negative, and its next multiplier lies well inside the unit circle. This is a condition/scope issue in the generic appendix sentence.

**Replacement:** “For the fold of Sec.V, ab>0 and every noncritical multiplier lies inside the unit circle; its attracting branch has mu=1−2sqrt(ab(alpha_c−alpha)) to leading order.” Keep the generic fold conditions in the preceding sentence.

## Notes and verified corrections

- **The new boundary gate passes.** The source/plan/metadata were privately checkpointed by root at `24909dbf650a2d08b214748cc1d738e3d96bc340` before execution. The new script checks only the newly specified close-up window, using eight-neighbor pairs whose midpoint lies inside it, axis-specific followed-grid steps and nearest fold/flip polylines. It verifies that all original input hashes equal the previously audited ones. Flip:232 pairs,220 within one step,232 within two, median 0.18810811607177516, maximum 1.5144595945312982. Fold:459 pairs,all 459 within one step, median 0.19079212512463897, maximum 0.6843470659438482. This supports article 1068–1071. The old full-domain counts 388/916 and 1720/1888 are reused from Task 38, not rerun.
- **Task 38 geometry fixes are present.** Lines 1046–1047 explicitly disclose four extended-curve points reaching 2.9e−7; cuspC's bound is now 7.7e−9; the six-cusp table uses the valid overall 8.2e−9/0.74 bounds. The conditional two-dimensional model remains conditional, and the comparable third multiplier is disclosed. The unsupported DSS 6 determinant count and 7/38 classification are not in this article.
- **Prompt and reproducibility fixes are present.** The PV paragraph now defines its exponent-only indicator and discloses 3–18 detected-period overlaps per map; it gives both 98.9–99.0% threshold agreement and 86.5–93.1% detected-period agreement. It no longer says the start does not matter. The reproducibility caption now correctly refers to the median periodic distance and mentions exceptions. For independently readable PV captions, naming “positive-exponent indicator” would be clearer than redefining the article's standard chaos label locally, but the exception is now disclosed and is not a concealed count error.
- **Huginn's Task 37 numerical fixes are present.** The analytic crossing is1.0775, noninferiority at1.05 is explicitly within four percentage points, and the energy-descent limitation names the actual feedback operators. “Found no chaotic regime” remains a finite observational statement, not a proof that Huginn cannot be chaotic.
- **Period- 5 removal is coherent.** No dangling references to the P5 shrimp, its four-cusp table, two-block figure, “second shrimp”, or the old P5 head interpretation remain. The overview, P6 section, Heads paragraph, anatomy caption, appendix and six-row table agree on a single body-qualified P6 example. The hostileP 7 head comparison remains explicitly a candidate contour geometry. Generic references to period 5 in other models' period lists are not stale P5-shrimp material.
- **Readouts of finite studies retain their limits.** Most old numbers were checked against the full report and existing source/math audits, not independently recomputed now. The article's sampled tangent/precision/horizon checks should not be read as validating every cell; its operational definitions and observed label failures are the relevant scope. No general asymptotic chaos proof is claimed. Missing saved states/eigenvectors for the symbolic Newton summaries remain a limitation handled by the companion review.

## Execution and evidence

One authorized new saved-data reduction ran, exactly once, with `/data2/neural-shrimp/.venv/bin/python`, HF offline and OMP/OPENBLAS/MKL/NUMEXPR thread counts 1. The script sets nice 10,1 GiB address-space and 30 CPU-second limits,45-second alarm, and checks the 16 GiB shared available-memory floor. Exit 0; tool wall time 0.231261379s, reduction timer 0.1412727888673544s; stderr empty. No retry, model call or producer import.

- Script: `boundary-closeup.py`, SHA- 256 `de52018c4dce36fb4262caf07ecfdd8b723fbbc156f11677aab864d4ee9f62fc`.
- Result: `boundary-closeup-result-001.json`, SHA- 256 `a0f4e93720e2b18fd80fd054757e8f25f020e3d9db6e0b8fa187c98e1cd11d8c`; includes every selected pair and input hashes.
- Stdout: `boundary-closeup-001.stdout`, SHA- 256 `86ca20272a3ad4c2fde075bb9e4b5f71253b59ad3e99811bd48109dbe9fd43fd`; empty stderr retained separately.
- Scope/evidence ledger: `claim-coverage.md`. Final manifest binds the article snapshot, this review, ledger, plan, script, result and reused compact audit authorities. It does not claim to rehash every original historical array in Task 39.
