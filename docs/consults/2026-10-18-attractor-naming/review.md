# Research opportunity and what the evidence leaves open

**Discrete time is not an obstacle to calling a set an attractor.** The substantive uncertainty is whether the observed chaotic sets attract nearby states indefinitely, rather than representing long chaotic transients—and whether the proposed unstable manifold actually explains their geometry.

My recommendation is **a full-state, phase-resolved unstable-manifold and transverse-perturbation experiment at the banded-chaos cell**. It has a sharper mechanistic target than another dimension estimate: it can distinguish genuine manifold tracking and finite-time transverse attraction from a close match visible only in projection.

First, concrete answers to the six terminology questions:

1. **“Attractor” is appropriate, with an explicit numerical qualification.** Name the neighborhood-attraction convention: a compact invariant set approached by trajectories starting in an attracting neighborhood. Do not claim that the chaotic sets satisfy this definition rigorously.

   Suggested wording:
   > “We use ‘numerically observed attractor’ for a recurrent set approached from the tested initial conditions. For the chaotic cases, invariance and attraction are supported by finite-time numerical evidence, not certified.”

   Calling the set a **Milnor attractor** would not eliminate the evidential gap: finite sampling does not establish a positive-volume basin or the relevant minimality condition. Your fixed-radius directional starts sample a shell, not a volume. A **physical measure** requires a positive-volume basin of convergent time averages; an **SRB measure** involves additional unstable-direction structure. Neither is established here.

   Coexistence changes “the attractor” to **“a coexisting attractor”** or “the chaotic attractor examined here.” The 14/512 outcome describes this particular sampling distribution, not global basin volume.

   There is also a useful exact observation: \(F(x)\) lies in the compact convex hull of the token embeddings. Continuity therefore gives a compact invariant globally attracting set in a broad, nonminimal sense. That set can contain multiple attractors and connecting invariant structures; its existence does **not** identify the recorded chaotic component as an attractor.

2. **Prefer “chaotic” over unqualified “strange.”** The reported positive finite-time exponent supports numerical chaos. The Kaplan–Yorke estimate is a **Lyapunov dimension estimate**, not a measured fractal dimension; equality with a geometric dimension requires additional hypotheses. Layered projections are suggestive, not decisive, and you correctly decline to report a correlation dimension.

   Paper:
   > “A numerically observed chaotic attractor, with a positive leading Lyapunov-exponent estimate and thin, folded projected structure.”

   Figure:
   > “Projected geometry of a numerically observed chaotic attractor.”

   Public video:
   > “Numerical chaos in a language model’s soft-token feedback loop.”

   “Apparently strange attractor” is defensible if immediately explained, but adds ambiguity without adding scientific content.

3. **Call \(F\) the “iteration map” or “soft-token feedback map.”** It is not a Poincaré return map. Return maps also exist for discrete systems, but require a specified return set and return time.

   For panel (d), use:
   > “Lag-one plot of the first principal coordinate: \(s_{n+1}\) versus \(s_n\).”

   “Scalar return plot” has established informal usage, so it is not intrinsically wrong; **“first-return plot” is misleading here**. Nor is the plot necessarily a scalar map: different full states can share \(s_n\) but have different successors. “Stretching and folding” should be described as a visual interpretation unless derivatives or material segments demonstrate it.

   For the banded cell, \(F^3\) is the **three-step band map**; it is a first-return map to a band only insofar as the specified band is visited first after three steps. \(F^6\) is its second iterate.

4. **Specify projection in the caption.**
   > “Projection of the numerically observed attractor onto its first three principal coordinates.”

   Subsequent shorthand “the attractor” is fine once the viewing convention is clear. Projection can superpose distinct filaments.

5. **The manifold sentence overclaims.** Your reported comparison supports closeness of recorded states to a sampled manifold representation in twelve coordinates. It does not establish equality of closures, reverse coverage, or full-state agreement.

   Replacement:
   > “In twelve principal coordinates, every recorded state in this band lies within \(9.8\times10^{-6}\) of the sampled unstable-manifold iterates (median \(3.9\times10^{-7}\)). This supports an unstable-manifold description at the sampled resolution, but does not establish equality with its closure.”

   Specify whether distances are to vertices, interpolated segments, or another representation.

6. **Other likely referee objections:** failure to detect periods through 64 is not proof of aperiodicity; finite-time exponents do not exclude chaotic saddles; variance explained is not a uniform reconstruction-error bound; the delay-coordinate evidence supports useful reconstruction, not a verified Takens embedding; and jump length does not establish disconnectedness or stretching/folding. Distinguish the smooth real-arithmetic map from its rounded implementation.

   Accepting the disclosed interval certificate, the certified attracting cycles can be called attractors without the chaotic cases’ numerical qualifier. The tiny certified neighborhoods and float32 basin plane do not certify the whole observed basin.

   Finally, replace the video legend:
   > “Each color identifies the highest-probability next token.”

   The loop writes a probability-weighted embedding, not a discrete word.

# Ranked experimental directions

The ranking favors mechanistic discrimination and executable measurements over attempts to strengthen terminology indirectly. No human judgments or model judges are needed.

No hardware specification or rental budget is supplied. The plans therefore start on the machine already running this model, with **zero API expenditure**, and express costs primarily as forward-pass equivalents. Wall times below are estimates conditional on measured throughput.

## 1. Does the unstable manifold organize the band in the full state space?

**Question and hypothesis.** Does the phase-resolved unstable manifold approximate the recorded band in all 896 coordinates, and do small transverse perturbations return toward it? The falsifiable hypothesis is **full-state geometric agreement plus finite-time transverse attraction**. The strongest alternative is a projection coincidence or an incomplete manifold sample near a nonattracting chaotic saddle.

This would add a mechanism, not merely another picture.

**Minimum experiment.** Develop an adaptive manifold representation using both local unstable branches of the period-3 saddle. Because the unstable multiplier of \(F^3\) is negative, do not assume that iterating one branch under \(F^6\) covers everything needed. Include its \(F^3\) image and construct the other band phases with \(F\) and \(F^2\).

Freeze the representation before testing independently initialized trajectories. At one phase-matched state per trajectory, apply small displacements normal to the local manifold tangent, tangent displacements, and sham displacements. Follow forty band returns. Repeat geometry calculations in both twelve coordinates and the full state.

**Comparisons and controls.**

- Strong baseline: a phase-matched trajectory point cloud with comparable storage and development data.
- Cheap baseline: the existing twelve-coordinate manifold representation.
- Negative controls: uniterated displaced states and wrong-phase manifold matching.
- Positive control: near the saddle, perturbations along its computed unstable and stable eigenvectors should reproduce local expansion and contraction under \(F^3\), including the unstable sign reversal.
- Refine the manifold mesh and halve the initial seed-segment length. Failure to stabilize distances is a representation failure, not evidence against the manifold hypothesis.

**Endpoint and independence.** Primary endpoint: the fraction of independent trajectory roots whose prescribed normal perturbations recover to the frozen manifold tube. Each root contributes one binary result; its perturbations and time samples are repeated observations. Report an exact binomial interval. Numerical failures count as nonrecovery and are separately identified; inadequate geometric resolution invalidates the intended test.

Expand after high recovery and stable full-state geometry; revise if projection-only agreement disappears; abandon this manifold-attraction account at the tested scales if well-resolved perturbations consistently move away.

**Interpretation.** Agreement plus recovery supports finite-scale manifold organization. Agreement without recovery supports geometry but not attraction. Recovery with substantial reverse-coverage gaps suggests an overextended manifold representation. A null does not rule out a different attracting set. None establishes equality of closures or perpetual chaotic attraction.

**Resources.** Cheapest useful pilot: four roots and one displacement scale. The recommended sixteen-root version below is approximately **40,000–80,000 forward-pass equivalents**, including a bounded manifold-construction allowance. Full-state storage for 100,000 points is approximately 0.72 GB.

## 2. Is the strongly chaotic component persistent, or a long transient toward periodic motion?

**Question and hypothesis.** Does the component at \(\alpha=0.07248\) retain positive finite-time expansion and avoid detected periodic capture as observation horizons increase? The competing explanation is a chaotic saddle with rare escape, potentially into the coexisting period-4 attractor.

The existing 600-step records do not settle this.

**Minimum experiment.** Draw 32 independent starts from a declared full-dimensional ball around the original start. Use eight separate development starts to freeze classification thresholds. Record outcomes at nested horizons through 8,192 steps. Apply the existing recurrence criterion, extend the search through period 256, and require repeated recurrence plus local contraction before declaring periodic capture.

Estimate the leading exponent on predefined blocks for a preselected subset of roots, rather than selecting trajectories because they look chaotic. Compare float32 and float64 on a smaller subset using distributions and classifications—not pointwise trajectory agreement.

**Comparisons and controls.** The original horizon is the strongest existing comparison. A recurrence-only classifier is the cheap baseline. The known period-8 cell is a positive control for capture detection; the certified cycles provide another control where initialization inside practical numerical basins is available. Shuffled records should fail genuine repeated-orbit tests. Injected NaNs, truncated runs, and intentionally underresolved recurrence calculations should trigger invalid-result handling rather than “chaos.”

**Endpoint and precision.** Primary estimand: probability, under the declared starting distribution, of no detected periodic capture by each horizon; denominator 32 roots. Use root-level survival summaries and binomial uncertainty at the final horizon. Nested horizons are not independent replicates. Incomplete runs are censored, not counted as persistent chaos. Even zero escapes among 32 roots permits a nontrivial escape probability; it says nothing directly about infinite-time survival.

Expand to longer horizons if classifications and exponent blocks stabilize. Revise the description toward “long-lived chaotic transients” if convincing captures accumulate. If results depend materially on precision, prioritize a numerical-resolution study before interpreting dynamical mechanisms.

**Interpretation.** Persistence strengthens a finite-horizon attracting-chaos account but cannot prove it. Captures can reveal basin coexistence or transient chaos; identifying which requires examining the pre-capture histories. Mixed exponent blocks may indicate intermittency rather than measurement failure.

**Resources.** Cheapest pilot: eight roots × 2,048 iterations. Full version: roughly **300,000–500,000 forward-pass equivalents**, depending on JVP cost. This is scientifically useful but less decisive per unit compute than direction 1.

## 3. Is the scalar “return plot” approximately closed, or does it hide essential state?

**Question and hypothesis.** Can \(s_{n+1}\) be predicted adequately from \(s_n\), or do delays contain substantial additional information? The alternative to an approximately one-dimensional map is a many-to-one projection with hidden memory.

**Minimum experiment.** Use separate trajectory roots for training, development, and held-out testing—e.g. eight in each group, with 1,024 recorded states each. Freeze the principal coordinate using training data only. Compare predictors using one coordinate, 3/6/12 delays, and twelve current principal coordinates. Fit local-linear predictors with development-selected neighborhood sizes. Also test \(F^3\) within a band.

**Comparisons and controls.** Persistence and nearest-neighbor prediction are cheap baselines. The exact full-state \(F\) evaluation is the deterministic oracle. A synthetic one-dimensional logistic map is a closure control; a two-coordinate recurrence
\[
u_{n+1}=1-1.4u_n^2+0.3u_{n-1}
\]
is a positive control for detecting delay information. Shuffled successors should destroy predictability.

**Endpoint and precision.** Primary metric: held-out squared prediction error divided by held-out variance, averaged first within each test root, then across roots. Bootstrap roots, not individual transitions. Solver failures receive a predefined baseline prediction and are reported. Invalid oracle checks invalidate interpretation.

Expand reconstruction work if delays substantially outperform the scalar predictor with root-level uncertainty excluding a negligible improvement. Abandon the scalar-map approximation if its persistent error is much larger than both numerical error and delay-model error. If all predictors perform poorly, revise sampling or modeling before asserting irreducible complexity.

**Interpretation.** Delay improvement establishes useful predictive state beyond \(s_n\), not a Takens theorem or intrinsic dimension. No improvement supports a practical scalar approximation only over the sampled distribution and resolution.

**Resources.** About **25,000–35,000 forward passes**, followed by CPU-scale regression. A four-root exploratory pilot is useful for implementation, but not for the final comparison.

# Recommended next experiment

## Full-state manifold tracking with controlled transverse kicks

This outranks the alternatives because the supplied saddle, multiplier, and geometric match already specify a testable organizing structure. It also directly repairs the strongest mechanistic overclaim without requiring an unattainable proof of infinite-time behavior.

### Experimental matrix

Use the banded cell at \(\alpha=0.06626\), with temperature fixed as supplied.

| Component | Prespecified design |
|---|---|
| Development | Separate roots; build phase-specific manifold representations and freeze distance thresholds |
| Independent test units | 16 independently sampled initial states from a declared ball around the original start |
| Initialization | 1,024 transient iterations; then take the first state in the designated band phase |
| Normal interventions | Two seeded normal directions, each with both signs, at each of two amplitudes |
| Amplitudes | \(10^{-3}L\) and \(10^{-2}L\), where \(L\) is development-set median within-band pair distance in full state space |
| Other arms | Two signed tangent perturbations at the larger amplitude; one sham |
| Follow-up | 120 iterations per arm: forty \(F^3\) returns |
| Primary amplitude | \(10^{-2}L\); smaller amplitude is a resolution/linearity check |

A root that does not reach the designated band is reported explicitly, not silently replaced. The perturbation estimand is conditional on reaching that band; the experiment must also report the fraction reaching it.

### Frozen measurement and endpoint

Construct a full-state, phase-specific distance-to-manifold function. Define the recovery tube from development unperturbed residuals and observed mesh-refinement error. These are empirical resolution checks, not rigorous enclosure bounds.

Before interpreting recovery, require:

- primary normal kicks initially lie at least four tube widths away;
- almost all held-out shams remain within the tube;
- refinement changes distances negligibly relative to the intervention;
- the saddle expansion/contraction positive control passes.

A root succeeds if **all four primary-amplitude normal kicks** lie within the tube for the final eight band returns. Report successes/eligible roots, the eligibility fraction, and an exact confidence interval. Sixteen roots provide a discriminating pilot, not a precise basin estimate.

Also report two secondary geometric quantities in original embedding-distance units:

1. held-out trajectory-to-manifold distances;
2. manifold-to-held-out-trajectory distances, with a matched-density trajectory-cloud comparison.

The second measurement checks unsupported extensions of the manifold representation; finite records cannot turn either direction into a closure theorem.

### Implementation sequence and branches

1. Benchmark 100 forward passes and a small JVP batch; measure memory.
2. Reproduce the saddle residual and eigenvector residuals sufficiently for the local intervention.
3. Construct both branches, phase images, and adaptive refinement.
4. Run four pilot roots; freeze the protocol only after resolution and controls pass.
5. Run sixteen fresh roots; do not tune thresholds on these results.

**Expand** if at least 14/16 eligible roots recover and full-state geometry is stable.  
**Revise** if twelve-coordinate agreement survives but full-state agreement does not.  
**Reject finite-scale transverse attraction at these doses** if the measurement is resolved but recovery is consistently absent.  
**Declare the test unresolved**, rather than negative, if the manifold cannot be resolved within the compute cap.

Estimated cost is **40,000–80,000 forward-pass equivalents**. At an assumed 0.1–2 seconds per equivalent, this spans roughly **1–44 hours**, before unusually expensive derivative or nearest-segment calculations. Benchmark rather than assume this range. Float64 weights alone are roughly 4 GB; actual peak memory must be measured. No API is needed. Rental only becomes relevant if the benchmark prevents completion within two weeks.

# First 48 hours

**Hours 0–6:** Implement the fixed map wrapper, phase bookkeeping, distance metric, deterministic seeds, and invalid-result flags. Benchmark throughput. Verify that “color” means argmax token, separately from the soft-token update.

**Hours 6–18:** Reconstruct the saddle’s local branches and phase images. Run the signed unstable-direction and stable-direction controls. Deliverable: local response curves and eigenvector/residual checks.

**Hours 18–30:** Refine the manifold representation; compare full-state and twelve-coordinate distances. Deliverable: distance stability versus mesh resolution and seed-segment size.

**Hours 30–48:** Run the four-root perturbation pilot. Deliverable: sham, tangent, and normal distance histories with a compute-accounting table.

If geometry is unresolved, spend the remaining time improving adaptive sampling—not producing more chaotic trajectories. If full-state agreement already fails robustly, retain that result and pivot to the trajectory-cloud comparison.

# Two-week research roadmap

**Days 3–5:** Freeze the valid protocol and execute the sixteen-root held-out experiment. Produce root-level outcomes, uncertainty, geometry residuals, and explicit eligibility counts.

**Days 6–8, conditional branch:**

- **Geometry and recovery succeed:** test the other two phases and a wider perturbation scale; investigate whether apparent exceptions concentrate near folds.
- **Geometry succeeds but recovery fails:** extend selected trajectories and examine escape destinations. This directly motivates direction 2.
- **Projection-only agreement:** quantify omitted-coordinate residuals and test whether the trajectory-cloud baseline predicts them better.
- **Resolution fails:** improve the manifold parameterization within a fixed cap; do not reinterpret unresolved distances as attraction.

**Days 9–12:** Run direction 3’s scalar-versus-delay comparison. This supplies a concrete interpretation of the lag-one panel regardless of the manifold outcome.

**Days 13–14:** Replicate the decisive finding with fresh roots. If manifold tracking succeeds, scale toward 64 independent roots or test one neighboring parameter cell—not both simultaneously. If escape evidence emerges, launch the persistence experiment instead.

The resulting program can establish **finite-scale attraction, full-state manifold organization, predictive closure, or their failure**. Those are substantive dynamical findings even if “strange attractor” remains an unproved—and unnecessary—label.
