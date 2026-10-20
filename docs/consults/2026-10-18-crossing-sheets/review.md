# Research opportunity and what the evidence leaves open

**The scientific opportunity is to separate attractor existence from attractor selection—and then explain selection dynamically. I recommend a small, paired-start continuation experiment across two band–stripe switches, using the already characterized crescent switch as a calibration.** This can distinguish “one attractor disappears” from “both survive, but \(x_0\) changes basin” without another large parameter map.

Your reported coexistence results establish that the fixed-start image hides overlapping attractors. They do **not yet establish that every apparent crossing is a basin switch**, or identify the invariant set organizing those switches.

Three refinements matter:

- **“Sheets” are an illustration, not a mechanism.** Periodic branches fold and double; chaotic attractors need not form smooth sheets. There is no intrinsic vertical ordering of attractors in \(\mathbb R^{896}\). What appears “on top” is whichever basin contains \(x_0\).
- **The proximity explanation is currently unsupported.** The reported nearest-phase classifier performed much worse than the majority baseline. Proximity of \(x_0\) to the paragraph phase may describe its trajectory, but does not explain capture. Replace that claim with interventions on the initial state.
- **A newborn sink does not necessarily have a globally tiny basin.** A saddle-node gives a local square-root separation between node and saddle. Global returns can feed a substantial basin into that local neighborhood. Basin-fraction and switch-offset laws require additional geometry.

The most economical explanatory figure is therefore not a 3D rendering. It is a **one-dimensional branch diagram with a separate fate ribbon**: all continued attractors, their stability, and the destination of each fixed start. A second panel should show transient residence near a candidate boundary invariant set.

Fractal boundaries, crises, and transient chaos remain plausible in particular regions, not necessary explanations everywhere. Multiple switches along a segment establish neither fractality nor Wada structure. Wada requires at least three distinct basins; verified asymptotically stable periodic sinks have open basins, excluding riddling of those basins in its usual sense.

# Ranked experimental directions

## Shared measurement rules

Use full-state recurrence and continuation to identify cycles; use token fingerprints only as convenient observables. A period-6 label alone does not establish membership in the crescent family.

A workable initial cycle criterion is relative recurrence error below \(10^{-8}\), maintained over four repetitions, with the minimal period tested up to 128. Calibrate this threshold on known cycles and measured numerical repeatability. Anything exceeding the period or horizon limit remains unresolved.

At decisive points, polish the orbit and estimate dominant Floquet multipliers with matrix-free Jacobian-vector products and multiple Arnoldi starts. Require convergence checks; unavailable or inconclusive stability estimates are a limitation, not evidence of instability.

Parameter cells, sweep directions, and successive iterates are **not independent replicates**. For deterministic branch measurements report numerical brackets and tolerance sensitivity, not sampling confidence intervals. For randomized initial states, retain one independent seed as the sampling unit and pair it across parameters. Never discard unresolved outcomes: report destination fractions as intervals, with unresolved starts allocated to neither or potentially either destination.

## 1. Separate basin switches from intrinsic branch changes

**Question and hypothesis.** At band–stripe visibility reversals, do both observed periodic attractors remain stable across the fixed-start switch? The hypothesis is that at least some reversals occur entirely inside stable overlap. The strongest alternative is a local bifurcation, crisis, or finite-transient labeling error.

**New knowledge.** This determines whether the crescent’s demonstrated coexistence mechanism also explains the crossing pattern, rather than merely occurring elsewhere.

**Minimum experiment.** Use three short transects: the crescent calibration and two held-out band–stripe switches. At 17 points per transect, compare \(x_0\), one frozen alternative token embedding, and continuation of both families from both parameter directions. Detailed selection appears below. Calibrate thresholds on the crescent only.

**Comparisons and controls.**

- Strong baseline: bidirectional branch continuation with selected stability calculations.
- Cheap baseline: the current \(x_0\)-only map.
- Positive control: at the reported coexistence point, starts on the band-6 and stripe-10 orbits must recover their respective attractors.
- Negative control: different phases of the same cycle must receive the same orbit identity; zero initial-state perturbation must reproduce the result.

**Endpoint.** Classify each held-out switch as stable-overlap selection, intrinsic branch change, mixed, or unresolved. The primary denominator is **two held-out switch neighborhoods**, not the number of cells. Also report the confirmed overlap width in \(\alpha\) or \(\log_{10}T\), and the switch bracket width.

Expand if overlap survives stability checks on both sides. Revise to a mixed explanation if continuation encounters a flip or other event; a flip is not extinction of the whole family. Abandon the “pure basin switch” explanation for that neighborhood if the selected attractor genuinely ceases to exist or attract before the switch.

**Interpretation.** Positive results establish selection within overlap, not a saddle-boundary mechanism. Null results favor intrinsic changes only if continuation and stability measurements are valid. Mixed results imply different visible edges need different explanations. Failed controls invalidate classification.

**Resources.** Cheapest pilot: nine points, three starts, one band–stripe switch. Budget approximately 50,000 map evaluations including extensions. At an assumed, **to-be-measured** 20–200 ms per evaluation, that is roughly 0.3–3 GPU-hours. The expanded design is bounded below.

## 2. Identify the boundary organizer and test initial-state causality

**Question and hypothesis.** Does a visible switch arise because \(x_0\) approaches the stable manifold of a saddle periodic orbit? A strong competitor is a chaotic boundary set or a complicated global return geometry without one dominant saddle.

**Minimum experiment.** At a parameter with two verified attractors, bisect a segment whose endpoints reach different basins. Use several phase choices and two distinct boundary neighborhoods. Because the reported segments have multiple crossings, refine **individual adjacent opposite-fate brackets**; do not assume one global threshold.

Integrate bracket endpoints, periodically re-bracketing when they separate, to obtain an edge trajectory. If it becomes nearly periodic, attempt Newton–Krylov refinement. Separately perturb \(x_0\) along a frozen direction near its parameter switch and measure the resulting switch displacement. Fit displacement using two development perturbation amplitudes; predict two held-out amplitudes.

**Quantitative prediction.** For a hyperbolic saddle with one relevant unstable multiplier \(\Lambda_u\) over period \(p\), transverse escape should satisfy approximately
\[
\tau(s)=-\frac{p}{\log|\Lambda_u|}\log|s-s_*|+C,
\]
with \(\tau\) measured in map steps. Measure residence near the saddle, not merely total settling time.

**Controls and baselines.** Compare against nearest-phase distance and a simple initial-state classifier. Use a synthetic saddle map with known multiplier as a positive control for bisection and slope recovery; known in-basin segments and zero perturbations are negative controls.

**Endpoint and precision.** Primary endpoint: agreement between measured escape slope and the independently estimated unstable multiplier. Use approximately 12–20 geometric distances, excluding the roundoff plateau. These are deterministic scale measurements, not independent samples. Report fit sensitivity to the scale range. Intervention magnitude is measured in embedding-space norm, preferably normalized by inter-orbit separation.

**Branches.** A recovered saddle plus matching slope supports a local manifold account. A reliable basin intervention without saddle recovery establishes selection causality but not the organizer. An irregular edge trajectory motivates chaotic-boundary tests; failed edge tracking alone does not establish fractality. Abandon a single-saddle model if reproducible residence dynamics and slopes contradict it.

**Resources.** Pilot one bracket first; roughly 20,000–100,000 map evaluations plus derivative work, approximately 0.1–6 GPU-hours under the same timing assumptions. This depends on direction 1 finding a clean stable overlap, and should reuse the boundary search already running rather than duplicate it.

## 3. Distinguish attracting chaos from long-lived chaotic transients

**Question and hypothesis.** Are some apparent chaotic interruptions actually finite-lived excursions into a chaotic saddle? The competitor is a genuine chaotic attractor coexisting with periodic windows; slow passage near a fold is another alternative.

**Minimum experiment.** Only launch this where direction 1 or the running maps reveal horizon-sensitive labels. Generate 24 independent initial-state draws that enter a specified pre-transition chaotic regime, recording the selection rate and burn-in rule. Pair those states across five nearby parameter values. Follow for 2,048 steps, extending survivors selectively.

The seed-generation distribution is part of the estimand. Snapshots from one trajectory are not 24 independent samples.

**Controls and baselines.**

- Strong baseline: survival curves at multiple horizons, with direct state-space escape detection.
- Cheap baseline: fixed-horizon twin exponent and token fingerprint.
- Positive control: an open synthetic chaotic map with known escape behavior.
- Negative control: a stable periodic orbit must not be falsely classified as escaping chaos.

**Endpoint.** Estimate \(S_\theta(H)\), the fraction of the defined initial ensemble still in the chaotic regime after \(H\) map steps, and restricted mean residence time. Treat unfinished runs as right-censored. With 24 independent seeds, worst-case survival-fraction uncertainty is roughly ±0.2; this detects large changes, not precise crisis exponents.

**Branches.** Reproducible eventual escape into stable cycles supports transient chaos. A sharp parameter-localized loss of survival plus continued boundary-set evidence supports a crisis interpretation. Persistent survival gives only a lower bound on lifetime, not proof of an attractor. Mixed destinations require a multibasin account. Invalid escape detection requires revised state-space criteria.

Do not fit a crisis exponent merely because lifetimes increase: exponents depend on the collision geometry, and fold ghosts can also produce long transients. Expand only after several parameter values provide sufficient uncensored exits.

**Resources.** A 12-seed, three-parameter pilot costs about 37,000 evaluations. The basic 24-by-five experiment costs about 246,000 evaluations before extensions: approximately 1.4–14 GPU-hours. It is lower priority unless horizon sensitivity is actually observed.

## 4. Test local fold geometry—not a universal basin-volume law

**Question and hypothesis.** Can local saddle-node geometry predict where a basin switch sits relative to the fold? The alternative is that global returns dominate selection, making local fold distance a poor predictor.

Near a nondegenerate fold, a return-map coordinate can have the form
\[
u_{n+1}-u_n=a\delta-bu_n^2+\cdots .
\]
Node–saddle separation then scales as \(\sqrt{\delta}\). A quadratic switch-offset law along the fold follows only under an additional smooth, transverse entry-map assumption.

**Minimum experiment.** Continue both stable and saddle branches at seven geometrically spaced distances from the known period-3 fold, spanning at least a factor of 16 if numerically feasible. Align orbit phases and project separation onto the critical direction with fixed normalization. Fit coefficients on three distances and predict four held-out distances. Then measure the location of one basin crossing on a frozen initial-state segment.

**Controls and baselines.** Compare the normal-form prediction against linear separation and a purely empirical fit. Validate the pipeline on a synthetic saddle-node; repeat phase alignment and tolerances as negative implementation checks. Pseudo-arclength continuation is the strong method; natural continuation is the cheap baseline but cannot resolve the unstable branch through a fold reliably.

**Endpoint.** Primary metric: held-out node–saddle separation in normalized embedding-coordinate units. Secondary metric: boundary position along the specified segment. A measured segment fraction is not basin volume in \(\mathbb R^{896}\).

**Branches.** Correct square-root branch separation but failed switch prediction isolates global basin geometry as the missing ingredient. Agreement on both supports the local-entry model only in that neighborhood. A null scaling result may mean the sampled region is outside the asymptotic regime; expand toward the fold only while numerical resolution permits. Failed saddle continuation leaves the hypothesis untested.

**Resources.** Pilot three distances before seven; approximately 50,000–150,000 map evaluations plus Newton/JVP work, roughly 0.3–8 GPU-hours before derivative overhead.

A fractal-boundary program should wait. If three destinations occur, a two-dimensional slice with multiscale boundary tests becomes worthwhile. Finite-scale uncertainty scaling can measure roughness, but neither a single segment nor a projected slice proves Wada structure in the full state space.

# Recommended next experiment

## One experiment: a paired-start, bidirectional microtransect test

This outranks saddle hunting because it first establishes **what actually needs explaining**. It also outranks further basin-fraction sampling: fractions depend on the start distribution and need not predict the fate of this particular \(x_0\).

### Experimental matrix

| Component | Specification |
|---|---|
| Development transect | \(\log_{10}T=-0.148,\ \alpha\in[0.06460,0.06480]\), using the reported crescent switch |
| Held-out transect B | At \(\alpha=0.0708\), the nearest \(x_0\) family switch below the known overlap temperature \(-0.1396\) |
| Held-out transect S | At \(\log_{10}T=-0.1396\), the nearest \(x_0\) stripe–band switch in \(\alpha\) |
| Selection rule | Select B and S from the \(x_0\) map alone, before inspecting their seeded-map outcomes |
| Resolution | 17 points across each switch neighborhood, then adaptive refinement |
| Starts | \(x_0\); one frozen non-special token embedding; each family continued in both directions |
| Horizon | 512 steps initially; extend unresolved runs and a fixed 10% sample of settled runs to 4,096 |

Choose neighborhoods spanning the adjacent coarse cells around each switch. If the specified switch is absent, record that fact and use the next nearest switch under a frozen ordering rule. Do not select only switches already known to exhibit coexistence.

The two held-out neighborhoods are distinct test cases, **not IID draws from all boundaries**.

### Primary endpoint and decisive checks

The endpoint is the number of held-out switches, out of two, for which:

1. \(x_0\) changes family inside a refined parameter bracket;
2. both corresponding periodic attractors persist on both sides;
3. their dominant multipliers remain numerically inside the unit circle.

Report unresolved cases explicitly. Refine to at least 1/64 of the original coarse-cell spacing, then check whether further refinement changes the classification.

At each decisive switch, show:

- branch observable—such as mean \(P(\text{“}\backslash n\backslash n\text{”})\)—versus parameter;
- stability markers and separate fate ribbons for \(x_0\) and the alternative start;
- settling time versus parameter, with censored runs marked.

These are experimental readouts, not substitutes for the experiment.

### Implementation sequence and budget

1. **Local CPUs:** implement recurrence classification, synthetic controls, plotting, and branch bookkeeping.
2. Benchmark exact-map throughput and memory. Float64 weights alone are approximately 4 GB; actual memory requires measurement.
3. Check cached versus uncached evaluations. Prefix caches must respect \(\alpha\)-dependent steering, and the replacement slot must not accumulate history.
4. Run the nine-point pilot before launching all transects.
5. Reuse converged states from the six-GPU jobs; do not repeat their broad maps.
6. Spend derivative calculations only on decisive anchors and switch brackets.

The full matrix has 306 trajectory arms before refinement. Approximately 0.3–1 million map evaluations is a planning range, not a guarantee. At 20–200 ms/evaluation, that spans roughly 2–56 GPU-hours. **Cap incremental work at 18 GPU-hours** and reduce to nine-point held-out transects plus targeted refinement if throughput is slow. At an assumed rental rate of [amount redacted]–3/GPU-hour, that cap is [amount redacted]–54, excluding the existing jobs. No API calls are needed.

**Decision branches:** two clean overlaps justify a basin-selection account for those crossings and launch edge tracking. One clean overlap establishes heterogeneous mechanisms. Neither clean overlap redirects effort toward branch bifurcations or chaotic survival. Failed controls trigger a narrower rerun, not a mechanistic conclusion.

# First 48 hours

**Hours 0–6**
- Freeze transect selection, recurrence rules, and horizon-extension criteria.
- Run synthetic saddle/fold controls and the reported coexistence control.
- Benchmark throughput; verify autonomous replacement-slot behavior.

**Hours 6–16**
- Execute the nine-point band–stripe pilot.
- Continue both attractors from both directions.
- Produce the first branch-plus-fate cross-section.

**Hours 16–24**
- Refine the strongest switch and calculate its selected Floquet multipliers.
- Deliver one experimentally classified crossing before the submission deadline: selection, intrinsic change, mixed, or unresolved.

**Hours 24–48**
- Complete the second held-out crossing within the compute cap.
- Extend horizon-sensitive cases.
- If stable overlap is confirmed, integrate the running boundary-search result and begin one edge-tracking bracket. Otherwise launch the small chaotic-survival pilot only where indicated.

# Two-week research roadmap

**Days 3–5: Establish local mechanism.**  
For a clean overlap, recover an edge trajectory, test periodic recurrence, and compare escape slope with an independently measured unstable multiplier. Deliver a quantitative residence-time test, not merely a saddle visualization.

**Days 6–8: Intervene on selection.**  
Freeze an initial-state direction and predict held-out switch displacements. If perturbations move the selection boundary while the attractor branches remain unchanged, retain that causal result even if saddle recovery fails.

**Days 9–11: Follow the evidence-dependent branch.**
- **Periodic edge:** test local fold/entry geometry.
- **Chaotic or horizon-sensitive edge:** run paired survival experiments.
- **Several intertwined destinations:** test finite-scale boundary roughness on a slice, without claiming fractality or Wada prematurely.

**Days 12–14: Test transfer.**  
Apply the successful measurement unchanged to one new crossing neighborhood selected without seeded outcomes. Deliver a mechanism comparison across neighborhoods, including explicit counterexamples.

The strongest eventual explanation may be simple but not uniform: **overlapping attractors explain why either family can be visible; local basin geometry explains particular switches; bifurcations and crises explain the exceptions.** The proposed experiments determine which parts of that statement are actually earned.
