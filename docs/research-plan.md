# Do periodic-window geometries improve neural-network control?

**23 September 2026 priority update:** The user's primary question is what
changes in neural behavior across shrimp-associated regimes. The historical
control-utility roadmap below is retained, but it no longer defines the sole
research objective. [EXP-025](EXP-025-results.md) completed a direct comparison
across two transects: repeating regimes showed substantially greater
repeatability after a tiny disturbance, while the frozen long-delay memory
test found no consistent advantage for either regime. See its
[prospective behavioral protocol](EXP-025-protocol.md) for the amended scope.

**Status:** private exploratory research program, 2026-09-08. The initial pilots are complete; the main geometry-utility study remains a design roadmap, not a completed study or preregistration. Pilot outcomes remain exploratory permanently. The repository is private by human request. No public release, registry submission, paid Pro review, or GPU rental is authorized by this document.

## Initial outcome and the next decision

The first pilots were executed from source `bd3d733b51e9f1ae9247ba712633aca023e7afac` after its private push. Their raw attempts, hashes, and descriptive summaries are retained under `evidence/`; see the manuscript's **Initial pilot results** section and the [dated claim-ledger update](claim-ledger.md#2026-09-08-initial-pilot-evidence-update). The prospective design below is retained as the record of what was planned. This opening update reports what the outcomes change.

The Hénon control produced numerically verified attracting periodic orbits, while the tiny neural scan did not. The recurrence policy never triggered; independently compared fixed/recurrence pairs have identical saved trajectories and predictive outcomes. All utility policies have divergent runs, so the predeclared all-finite mean test-loss summaries remain undefined. There is no demonstrated neural shrimp, diagnostic intervention benefit, or geometry-specific advantage.

The appropriate continuation has two stages. First, document a new bounded **local exploratory** search for a reproducible neural periodic regime, using a published small-model precedent as an additional positive control if useful, longer-horizon diagnostics, and targeted parameter refinement. Second, if a regime is verified, build controlled interventions that measure task benefits and harms, then graduate to the held-out incremental-geometry test described below. The present pilot must not be silently recycled as a confirmatory evaluation or turned into a global absence claim. Paid computation and paid Pro review were not used; a paid scale-up is not justified by these pilots alone.

## The question that would make this worth a paper

Finding an attractive picture is insufficient. The central question is whether a verified organization of periodic windows helps choose an intervention that improves task quality, reliability, or total computation **beyond information already supplied by loss, local stability, and recurrence**.

A shrimp is a structure in a plane of **control parameters**. A state trajectory can approach the periodic attractor supported inside that region; the state does not itself enter a learning-rate plane. During training, weights are the state and learning rates are controls. During latent inference, model weights are fixed controls and the evolving hidden state is the state. An initial-condition basin map is a different object from a control-parameter map.

Three claims must remain separate:

1. **Existence:** stable periodic attractors occur in this specified neural map.
2. **Organization:** the associated parameter windows have a verified shrimp-like bifurcation organization.
3. **Utility:** measuring that organization improves a decision on held-out cases after charging its cost.

The literature already contains neural-network shrimps and neural-training period doubling. Neither their existence nor oscillating loss would be a defensible novelty claim. See [literature provenance](literature.md).

## What could they tell us, and should we avoid them?

| Situation | Interpretation | Candidate action | Evidence needed |
|---|---|---|---|
| Stable nontrivial weight cycle with poor validation error | Training has stopped making asymptotic progress under the fixed update | Reduce a learning rate, change its layer ratio, or leave the window | Better validation error or faster target attainment than ordinary scheduling |
| Correct, predictable latent cycle | Internal repetition need not harm a task whose decoder is already correct | Stop early or use a frozen phase/readout rule | Correctness and calibration under perturbations; cost of verification counted |
| Incorrect latent cycle | Repetition can consume the reasoning budget without correction | Change relaxation or restart from a controlled checkpoint | Correct solutions recovered at equal total cost, including lost solves |
| Transient sensitivity without an attracting cycle | The trajectory may still be doing useful search | Continue or intervene only if a predictive rule warrants it | A controller must avoid cutting off productive difficult cases |
| Broad useful window with substantial attraction basin | Robustness to control perturbation may be useful | Choose an interior setting with adequate performance margin | Robust task utility, not merely a negative stability exponent |
| Thin window or narrow attraction basin | A setting found by a scan may be fragile or inaccessible | Avoid relying on that setting without a robustness test | Repeatability across starts, precisions, and small parameter perturbations |

These are hypotheses and decision rules to test, not recommendations established by results. Negative Lyapunov exponents measure attraction, not correctness. A stable wrong answer can be worse than a temporarily sensitive route to a correct answer. Likewise, a cycle can have good predictions even though its weights never converge. The relevant objective is operational utility, with dynamical labels as potential predictors.

## The strongest falsifiable novelty boundary

**Working hypothesis:** for a fixed distribution of small deterministic neural training problems, previously measured periodic-window geometry predicts an effective direction and size of a learning-rate intervention better than equally tuned controllers using only trajectory diagnostics; the advantage survives held-out tasks and initializations after charging measurement and map-building costs.

This hypothesis is stronger than “detecting loops helps.” A recurrence-triggered scheduler with no parameter geometry can support only a cycle-diagnostic claim. It cannot show that shrimp shape, its limbs, or its bifurcation organization matters.

An informative null would be that geometry adds no held-out predictive or control value beyond loss, recurrence, and local stability at the tested measurement budget. That would favor the cheaper diagnostics for this population. It would not prove that no neural system contains shrimps or that all geometric control is useless.

The strongest permitted eventual conclusion is conditional on the frozen task generator, architectures, controls, measurement access, horizon, precision, and cost model. No result here automatically extends to stochastic minibatch training, large language models, deployed APIs, or arbitrary parameter slices.

## Stage A: executable local pilot

The initial code targets a tiny tanh network trained by deterministic full-batch gradient descent with two positive layer learning rates. It is inspired by Sohl-Dickstein's construction, rather than an exact reproduction of its architecture and complete settings. The network has two inputs, three hidden units, a scalar output, and biases in both layers, for thirteen evolving parameters. Training minimizes half mean squared error plus `0.001 / 2 * ||theta||²`, including biases in the ridge penalty. Reported predictive MSE excludes the penalty. A Hénon map supplies a classical positive control for period and stability diagnostics. Analytical contractive and periodic fixtures test the instrument without claiming neural outcomes.

The current pilot manifests specify the following result-free design. These counts are planned workloads, not completed trials or statistical sample-size claims.

| Manifest | Planned cases | Horizon | Readout and scope |
|---|---|---|---|
| `EXP-001-henon-pilot.json` | 13 `a` values × 5 `b` values = 65 cells; common initial state `(0.1, 0.1)` | 3,000 updates | Tail classification and numerical orbit certificate; the seed field is only a panel label for this deterministic map |
| `EXP-002-training-pilot.json` | Task/initialization seeds 101–102 × 9 hidden-rate values × 9 output-rate values = 162 cells | 2,000 gradient updates | Tail classification, numerical orbit certificate, and terminal test MSE for finite runs |
| `EXP-003-utility-pilot.json` | Task/initialization seeds 201–204 × 6 rate pairs × 4 policies = 96 runs | At most 1,200 gradient updates per run | Terminal test MSE and divergence fraction; update counts, wall time, and diagnostic time |

Neural tasks use a fixed random tanh teacher with independent standard-normal input splits of 24 training, 48 validation, and 96 test examples; initial weights use the task seed plus 100,000. A CPU input-preparation command generates result-free `.npz` artifacts **before** the code/plan freeze, retaining teacher parameters, all three input/target splits, and initialization. Each manifest binds these files by SHA-256. Execution loads and verifies frozen inputs rather than regenerating data, and checks input finiteness and model-compatible shapes. The current pilot therefore has one initialization per task, not a factorial separation of task and initialization variation. All pilots use float64, retain up to 512 tail states for scans, search periods up to 16, and use absolute/relative screening tolerances `1e-8`/`1e-6`. Utility retains every weight state, pre-update regularized training loss, realized rate pair, and intervention event. Code-bound defaults, including teacher width, label noise, and numerical root tolerances, form part of the recorded source; the manifest is not the entire specification.

The discovery scan records all requested cells, including divergence, unresolved transients, and candidates rejected by verification. The machine configuration must bind the map, data generator, seeds, parameter ranges, precision, burn-in, retained horizon, maximum searched period, tolerances, escape criterion, and stopping budget. Save full-state tail evidence sufficient to reproduce candidate checks. Never classify unverified bounded motion as chaos merely because no short period was detected.

For an alleged period `p > 1`, require recurrence over multiple repeats, reject shorter primitive periods, and check that the cycle's amplitude is distinguishable from tolerance. Independently evaluate closure of `F^p(theta) - theta` and the eigenvalues of the period Jacobian product. A spectral radius safely below one establishes local attraction of a verified isolated orbit; it does not establish a large basin. Near-unit eigenvalues and non-isolated orbit families need a distinct nonhyperbolic/unresolved label. Symmetry or flat directions must not be removed merely to pass the stability threshold.

The pilot utility comparison uses identical problem instances and initial states for fixed learning rates, predetermined scheduled decay, scalar stagnation-triggered decay, and recurrence-triggered decay. Arms follow identical trajectories until their first differing intervention, although each is executed separately. Each decay halves both layer rates and each policy permits at most two decays. Scheduled decay acts at updates 400 and 800; the adaptive rules inspect every 40 updates from update 200. Stagnation compares the minimum regularized training loss in successive 40-step windows with a relative-improvement threshold of 0.001. Recurrence uses full-weight tail screening for nontrivial periods, without online orbit certification. These are the same action magnitudes and adaptive opportunities, but their triggers need not occur together.

The pilot's six initial rate pairs are `(0.3, 0.3)`, `(1.2, 1.2)`, `(1.8, 1.8)`, `(2.5, 0.3)`, `(0.3, 2.5)`, and `(2.5, 2.5)`. Its predeclared primary readouts are terminal **test MSE** and divergence fraction, not the later proposed fixed-resource validation endpoint. No test or validation outcomes trigger the policy. The descriptive report shows policy-level means only if all constituent runs are finite, plus divergence counts, triggers, costs, and seed-level outcomes. It computes no inferential p-values, confidence intervals, or population-wide benefit claim. This answers a preliminary **diagnostic** question; it does not yet test geometry-specific utility.

Both adaptive detectors wait at least two check intervals after each rate change. Recurrence screening uses only states from the current constant-rate segment; scalar stagnation uses two complete loss windows from that segment. This prospective implementation choice avoids comparing states generated by different update maps. The online trigger is still a finite-history heuristic, not an independently verified orbit certificate.

Equal maximum optimizer-update budgets are an implementation check, not a total-cost match; divergent runs stop early and must retain their actual update counts. Measure detection overhead, extra state storage, validation access, and wall time separately. A performance difference at equal steps alone cannot be advertised as a speedup. The current timing includes common pilot instrumentation, such as recording every state and evaluating training loss each step. It is not a benchmark of an optimized production implementation. A shared simulated prefix would be charged to each hypothetical deployed policy even if a future runner saved computation by reusing it.

The complete code, prepared inputs, manifests, and analysis must be committed and pushed before pilot execution. The local source guard requires a clean worktree and a containing `origin` remote-tracking ref; retain the successful push record because a local ref is not independently a live remote receipt. No target outcomes are needed for this scope audit. Raw scan files retain the bounded tail (including operational divergence evidence when available), utility files retain all generated states through termination, and already written trials remain present after a failed run. Receipt verification and trial census precede descriptive analysis. Infrastructure exceptions still require a corrected successor attempt with the original failure record preserved.

## Stage B: verify structure before naming a shrimp

This stage requires new implementation beyond the initial pilot.

1. Refine candidate regions on nested parameter grids with the original map and data held fixed. Retain the coarse map and every failed refinement.
2. Continue periodic orbits across parameters, rather than independently recoloring only the attractor reached from one start.
3. Locate saddle-node and period-doubling boundaries and their arrangement around connected isoperiodic regions. A smooth visual silhouette alone is insufficient.
4. Repeat with longer horizons, tighter tolerances, perturbed initializations, and sufficient precision. Test multistability by sampling basins.
5. Separate appearance in the chosen coordinates from structural facts about the branches. Distances, widths, and apparent shape depend on the scaling of the control axes; freeze the physically meaningful coordinates, such as log learning rates, before using geometric margins.

Do not transplant a Rössler reinjection mechanism merely because a silhouette resembles its windows. A Hénon success validates part of the measurement pipeline; it does not validate the neural map or guarantee that the selected neural slice intersects a shrimp.

## Stage C: the actual utility experiment

This stage is proposed, not implemented or frozen. A separate result-free machine plan must specify exact counts and thresholds before held-out outcomes are generated.

**Development and holdout.** Separate (i) exploratory mapping tasks and seeds, (ii) controller development tasks and seeds, and (iii) untouched evaluation tasks and initialization seeds. For synthetic regression, the task unit is a newly generated teacher/data-generating problem, not a neighboring pixel or an additional time step. Specify architecture-transfer evaluation separately. Do not generate test labels until thresholds, features, controller rules, and costs are frozen.

**Predictive experiment.** At common checkpoints, predict whether a specified intervention will improve future validation utility. Compare a fixed baseline feature set (recent loss level/slope/variance, gradient and update norms, recurrence residual, finite-time stability estimate) with the same model and fitting budget plus geometry features. Candidate geometry features include signed distance to a verified boundary, boundary normal, period label, and local window width. Hold out spatial blocks or complete windows as well as tasks so interpolation between adjacent pixels cannot masquerade as generalization. Label the outcome using future performance, not the same recurrence statistic used as the predictor.

**Control experiment.** Branch identical checkpoints into a frozen intervention set: continue; predetermined schedule; scalar stagnation trigger; recurrence trigger; local-stability trigger; map-informed trigger and direction. The baseline also gets the map-informed controller's action magnitudes and decision opportunities. Compare shared intervention doses to identify the information advantage; subsequently compare full optimized policies under equal tuning budgets. Include random directions with the same norm and an ordinary locally sampled performance map with equal scan cost. The latter tests whether the apparent gain is simply from extra hyperparameter trials.

**Primary utility endpoint.** Proposed endpoint is paired held-out validation error at a fixed total resource budget. Freeze the exact aggregation, normalization, divergence penalty, and practical-effect threshold after the pilot but before held-out evaluation. A complementary endpoint is compute to a predeclared quality threshold, treating failure to reach it as budget-censored rather than dropping it. For latent tasks, use exact task correctness at a fixed total inference budget and incorrect-termination rate, with all timeouts included. Evaluation test labels cannot trigger a live controller.

**Cost accounting.** Report both cold-start cost and explicitly amortized cost over a specified number of future runs. Charge offline scans, parameter probes, failed runs, controller tuning, Jacobian or Hessian calculations, trajectory storage and transfers, and any extra validation evaluations. If an atlas is useful only after many repeated tasks, report the measured break-even count; do not assume unlimited reuse or transfer.

**Inference.** Use paired estimates at the task level, retaining seed-level variation within each task. Freeze a hierarchical or task-cluster bootstrap and its draw count, primary contrast, multiplicity handling, failure categories, and practical effect before target generation. Report complete task strata and the prevalence of identified cycles in the unselected population. Conditional rescue performance among selected traps cannot establish population-wide savings by itself.

**Falsification outcomes.** If ordinary recurrence ties geometry, report cycle detection as sufficient. If map construction erases a per-run gain, report no demonstrated net benefit. If intervention improves training loss but harms held-out quality, report that tradeoff rather than optimization success. If no verified neural shrimp is located, continue only as an explicitly relabeled periodic-diagnostics project or design a separately documented search extension.

## Stage D: latent reasoning extension

Do not conflate long transient settling times with stable cycles. An exact decoded answer that repeats need not imply full hidden-state recurrence. Freeze a deterministic recurrent update and all evolving state, including secondary latents, caches, momentum, or schedule phase where relevant. Log residuals, decoded answers, correctness, and failure to terminate separately.

A useful intervention family is relaxation:

\[
G_\alpha(h)=(1-\alpha)h+\alpha F(h),\quad \alpha>0.
\]

Its fixed points are exactly those of `F`. At a fixed point, an eigenvalue `lambda` of `DF` becomes `1 - alpha + alpha lambda`. Thus positive under-relaxation cannot stabilize a real `lambda > 1` direction, while it may suppress oscillatory instability. This local algebra is not a global guarantee of correctness or faster reasoning. It does not preserve nontrivial cycles or attraction basins.

For two controls that preserve the fixed-point equation, a later experiment can use relaxation and momentum, augmenting the state by the previous iterate. A feedback-gain sweep is also possible, but it changes the fixed-point equation and must be labeled as a changed solver. Do not attribute its gain solely to escaping the same dynamical trap. Match restarts, damping, and geometry-based interventions against simple residual/stagnation policies at equal inference cost; use disjoint puzzle instances and multiple checkpoints before claiming transfer.

## Integrity and release conditions

Follow the canonical `EXPERIMENT_INTEGRITY_SKILLS.md` and `RESEARCH_NOTE_WRITEUP.md` in the sibling `agent-skill-documents` repository. Preserve plan hashes, code revision, environment, exact machine configuration, failed attempts, raw outcomes, and analysis provenance. A private commit does not supply public prospective verification. Local scientific and zero-context audits are the default. Paid Pro review was not requested and was not run.

The initial manuscript is an AI-assisted working draft for human shaping. It contains no invented results and makes no claim that the human has inspected generated artifacts. Before public release, complete the appropriate fresh confirmatory design and review the canonical disclosure so that every assertion is true. Public release and repository visibility remain separate human decisions.
