# Research opportunity and what the evidence leaves open

**The strongest next question is whether instability gives an actor *selective control*, rather than merely causing unpredictable divergence.** I recommend a small, white-box experiment asking whether one bounded perturbation early in a soft-feedback loop can steer an objectively correct answer toward a *specified wrong answer*, more effectively than compute-matched random perturbations.

Your disclosed results establish amplification, bifurcations, coexistence, and some downstream consequences. They do **not** establish that amplified directions are behaviorally important, that particular outcomes are reachable, or that trajectory divergence changes long-run behavior. Those distinctions open useful research programs for security, compression, agent coordination, steering, and interpretability.

Below are **ten concrete consequences grouped into five ranked experimental directions**. Each direction has a minimum experiment and a conditional extension—not five projects to run simultaneously.

Two paths deserve deprioritization:

- **Generic “chaos predicts quality/hallucination.”** Your accuracy null already challenges that inference. Replace it with controlled questions about reachability, dependence, and objective error.
- **“Compressed and original trajectories diverge, therefore compression is harmful.”** Chaos makes that almost inevitable. Test changes in basin occupancy or task outcomes instead.

All proposed tasks have executable ground truth. No human reviewers or model judges are necessary.

# Ranked experimental directions

## 1. Selective controllability: does chaos help choose the failure?

**Expected value/day: highest.** This directly separates a consequential security mechanism from generic numerical sensitivity.

**1A — Claim:** A small, single early-state perturbation can increase the probability of a *specified* incorrect answer, not just change the answer.

Positive Lyapunov exponents predict amplification, but not alignment with the desired readout or access to another answer basin. The strongest competing explanation is ordinary adversarial susceptibility at the final readout, unrelated to feedback instability.

**Minimum experiment.** Use a 0.5B model, short arithmetic problems, and a fixed-prompt feedback map followed by a common greedy answer suffix. Compare:

- no perturbation;
- one random perturbation;
- compute-matched random search;
- projected gradient search;
- the same targeted attack applied immediately before readout.

Vary feedback horizon and perturbation radius. Select two temperature settings on development problems; measure prompt-specific finite-time expansion rather than assuming temperature determines regime.

Use 16 development problems and 64 held-out problems from disjoint expression templates and operand tuples. Problems—not attack restarts, temperatures, or horizons—are the independent units.

**Endpoint and controls.** Primary estimand: targeted-conversion rate for optimized minus compute-matched random attacks, in **percentage points among baseline-correct problem–setting pairs**. Report that denominator explicitly, plus unconditional results over every problem. A large-radius late attack is the optimization positive control; a forced final-logit bias checks scoring; zero perturbation is the negative control. Greedy feedback with derivative-free search is a strong non-soft comparison.

**Interpretation and decisions.** A headline would be substantial targeted control at small realized perturbations, with greater leverage early than late. Unstructured answer changes without targeted advantage support instability but not selective control. Comparable early and late success suggests ordinary readout vulnerability. Failed large-radius controls invalidate conclusions about controllability. Expand for a reproducible advantage around 15 percentage points or greater; retire this attack method if a sufficiently precise interval excludes a 10-point advantage.

**1B — Claim:** The perturbation needed for targeted control decreases with feedback horizon until basin barriers or saturation intervene. Test a radius staircase at three horizons. An approximately exponential threshold decline would be striking; threshold independence despite measured state expansion rejects the simple “Lyapunov security budget” hypothesis.

**Resources, estimated:** cheapest pilot, 8 problems and one radius, 1–3 local GPU-hours. Full experiment: roughly 6–24 GPU-hours, depending on differentiation throughput. No API expenditure.

---

## 2. Compression fidelity: can excellent one-step imitation preserve the wrong dynamics?

**Expected value/day: high.** This offers a concrete deployment criterion beyond conventional logit agreement.

**2A — Claim:** Quantization can change the probability of entering persistent repetitive behavior even when one-step predictive distributions remain close.

Near bifurcations, a small structured change can move a boundary. The competing explanation is merely the already-known separation of trajectories while behavior distributions remain unchanged.

**Minimum experiment.** Compare BF16 and a supported 4-bit implementation of the same 0.5–1.5B model. On 16 development prompts, choose a temperature grid spanning—not exclusively hugging—known transitions. Evaluate 48 held-out prompts, each with 16 standardized initial states.

Measure one-step KL on common states separately from closed-loop behavior. The primary endpoint is the difference in the **fraction of starts entering a verified period-≤8 orbit within 256 steps**, averaged equally across prompts. Verify recurrence in state space; repeating text alone does not qualify. Longer confirmation runs on a subset test transient misclassification.

Strong baselines are BF16 versus FP32 and precision-matched logit perturbations with comparable one-step KL. A cheaper baseline is greedy-output agreement. A temperature intervention known on development cases to cross a transition is the positive control; two identical BF16 executions are the negative control.

**Interpretation and decisions.** Low KL plus a ≥10-point occupancy shift would motivate feedback-specific acceptance tests. Divergent trajectories with unchanged occupancy are a scientifically valuable null. Changes confined to narrowly selected transition points establish local fragility, not widespread deployment harm. Excess unresolved trajectories require longer measurement, not classification as chaos. Expand only if shifts survive longer runs and appear beyond cherry-picked boundary cases.

**2B — Claim:** Short-rollout distillation preserves feedback behavior better than one-step distillation at matched predictive fidelity. Conditional on 2A, train two small LoRA students using one-step versus eight-step state losses on identical examples and compute. A held-out occupancy improvement is the headline; no improvement at matched KL rejects that particular training remedy.

**Resources, estimated:** pilot of 8 prompts × 8 starts, under 2 GPU-hours; main comparison, approximately 4–16 hours. The training extension costs an additional local day and is not needed to establish 2A.

---

## 3. Coupled agents: does synchronization manufacture misleading consensus?

**Expected value/day: medium-high.** Your observed locking motivates a practical consequence, but locking itself does not establish harmful agreement.

**3A — Claim:** Reciprocal communication can increase agreement on the *same wrong answer* beyond the dependence already induced by a shared model and task.

Use exact modular-arithmetic or small constraint-solving tasks. The strongest competitor is shared-model error correlation: agents may agree incorrectly without any dynamical synchronization.

**Minimum experiment.** For 64 held-out task instances and eight starting-state pairs, compare equal-compute:

1. independent loops;
2. live bidirectional communication;
3. replayed messages from another run on the **same task**, breaking reciprocal dependence while retaining plausible message content.

Include an ordinary text-only communication arm as a transfer test, not as an assumed instance of the same dynamics.

The primary metric is **excess same-wrong-answer agreement per task**, in percentage points, relative to independently paired runs from that task. Average starts within task before inference. Also report each agent’s accuracy: more agreement at unchanged or improved correctness is not the proposed harm.

Independent solving is the cheap baseline; a single solver given the same total compute is a strong practical baseline. Forced echoing of a known wrong answer validates the agreement detector; communication from unrelated tasks is a negative control for task-specific informational effects.

**Interpretation and decisions.** Live coupling exceeding both independent and replayed wrong agreement would implicate reciprocity. Agreement without excess errors supports synchronization but kills the harmful-consensus claim. Live and replayed effects of equal size favor message content over reciprocal dynamics. Accuracy floors invalidate the task choice. Expand for a reproducible ≥10-point excess; otherwise narrow the claim to coordination rather than reliability.

**3B — Claim:** Sparse or delayed communication can recover complementary errors without adding inference compute. Compare every-step exchange with exchange every fourth step, holding total model calls fixed. Reduced wrong agreement *without reduced accuracy* is the headline; merely suppressing all communication benefits kills the proposed remedy.

**Resources, estimated:** 8-task pilot, 1–2 hours; full small-model matrix, 3–12 local GPU-hours.

---

## 4. Steering history: can the same present setting produce different policy behavior?

**Expected value/day: medium.** Coexisting attractors make history dependence plausible; whether it matters for controlled behavior remains open.

**4A — Claim:** A steering coefficient’s present value is insufficient to characterize its behavioral effect because previous steering can select a persistent basin.

Use synthetic access-control tasks: a supplied permission table determines whether a requested operation should return `ALLOW` or `DENY`. This measures formal policy compliance—not general alignment.

**Minimum experiment.** On 48 independent permission-table/request instances, compare direct initialization at a final steering value with upward and downward excursions returning to that value. Equalize total iteration count and assess after a fixed 32-step endpoint dwell. Nest eight initial states within each instance.

Primary endpoint: the history-induced difference in **false-ALLOW probability among prohibited requests**, in percentage points. Also measure permitted-request false denials.

Strong baselines are cold restart at the endpoint and an endpoint-only logit bias; the cheap baseline is no excursion. Initialize a known contrasting attractor as a positive control for history detection. Zero-amplitude excursions are negative controls.

**Interpretation and decisions.** A persistent ≥10-point compliance difference after return would show a practical limitation of static steering calibration. Differences that vanish during endpoint dwell indicate relaxation delay, not persistent hysteresis. Identical endpoint behavior bounds the consequence for these tasks. If the steering never changes the measured policy output, the intervention is invalid rather than reassuring.

**4B — Claim:** Steering can exhibit *rate-induced tipping*, even without crossing a static stability boundary. Compare ramp speeds only after verifying stable frozen-parameter attractors throughout the path and survival under a quasi-static ramp. A speed-dependent transition then becomes interesting. Without those checks, call it hysteresis or bifurcation delay—not rate-induced tipping.

**Resources, estimated:** 8-instance pilot, about 1 hour; main experiment, 3–10 GPU-hours. High-precision confirmation is useful only for a small number of candidate mechanisms.

---

## 5. Hidden chaos: when is a visible trace an inadequate state description?

**Expected value/day: medium; very cheap if hidden-chaos cases are already reproducible.**

**5A — Claim:** Two states with identical visible loop traces can respond differently to the same subsequent objective query.

Hidden chaos establishes latent motion, but not downstream distinguishability. The competing explanation is that the unstable directions remain invisible to relevant readouts.

**Minimum experiment.** Find exact visible-trace collisions among eight starts for each of 32 held-out task prompts. Require matching the entire recorded observation window, not merely the last token. From each collision pair, apply identical answer readouts using the same fixed prompt and suffix; only the continuous state differs.

Primary endpoint: **answer-disagreement fraction among qualifying prompt-level collision pairs**. Select at most one primary pair per prompt and report how many prompts qualify. Exact-state restoration is the strong baseline; visible-trace-keyed state substitution is the cheap comparison. Identical-state replays are negative controls. A deterministic scalar state readout demonstrates that the pipeline can detect an intentionally observable latent difference.

**Interpretation and decisions.** Frequent downstream disagreement establishes that the visible trace is not a sufficient record for this soft-state interface. Latent separation without readout disagreement establishes an observability limit, not a hidden behavioral failure. Fewer than roughly 20 qualifying prompts makes the planned estimate too imprecise; broaden acquisition rather than interpreting missing collisions as robustness. This does **not** indict ordinary token-prefix KV caching, which retains different information.

**5B — Claim:** A small causal state summary can preserve downstream behavior better than text alone. Fit low-rank summaries on development states, reconstruct held-out states, and rerun the readouts. Compare learned, random-projection, and exact-state checkpoints. Successful reconstruction—not merely probe prediction—is the headline; prediction without intervention fidelity rejects the proposed summary as sufficient.

**Resources, estimated:** 8-prompt pilot under 1 hour; full acquisition and readouts, 2–8 GPU-hours.

### Shared measurement discipline

Use disjoint task-template families where possible, not just new random seeds. Analyze task/prompt blocks with paired intervals or a block bootstrap; never count temperatures, starts, or optimization steps as independent samples. With 64 independent binary units, an individual rate has roughly ±12-point worst-case 95% precision; paired contrasts can be wider.

Malformed answers count as incorrect but not as successful targeted conversions. Execution failures remain visible, with sensitivity bounds—not silent exclusion. Preserve “unresolved dynamics” as a separate category. These are objective task measurements; no model-judge agreement is being substituted for semantic accuracy.

# Recommended next experiment

## One-shot targeted control of a soft-feedback answer

This outranks the alternatives because it asks a sharp new question with a strong null: **exponential expansion may be real while useful directional control is absent.** It also avoids the already-planned stabilizer, voting, and repetition studies.

### Experimental matrix

Start with a 0.5B released model and the existing fixed-prompt loop implementation.

| Factor | Levels |
|---|---|
| Task blocks | 16 development; 64 untouched held-out arithmetic problems |
| Temperature | Two development-selected settings with contrasting measured expansion |
| Feedback horizon | 8 and 32 steps; immediate-readout diagnostic |
| Perturbation radius | Two development-fixed doses |
| Intervention | None; single random; budget-matched random search; targeted optimization |
| Timing diagnostic | Early injection versus injection immediately before readout |

The attacker chooses **one perturbation**, not repeated interventions. Choose a specified incorrect integer answer deterministically from each problem, preferably matching the correct answer’s token length.

Set radii relative to the initial feedback-state norm, while also reporting their scale against measured batch-layout discrepancies. Do not label an arbitrary small norm “rounding-sized.”

**Important implementation constraint:** evaluate the perturbation actually surviving conversion to the model’s input dtype. If a proposed dose rounds away, report it as unrealizable; do not claim a sub-ULP attack.

### Primary endpoint

At the longer horizon and smaller dose, estimate

\[
\Delta =
P(\text{specified wrong answer}\mid\text{optimized attack})
-
P(\text{specified wrong answer}\mid\text{budget-matched random search}),
\]

among baseline-correct problem–temperature pairs, weighting problems equally. Report eligibility, invalid outputs, realized perturbation norms, and unconditional results alongside it.

Temperature and finite-time expansion are moderators, not randomized proofs that chaos causes vulnerability.

### Implementation sequence and checks

1. **Establish task viability.** On development problems, verify sufficient baseline correctness—approximately 70% or better—to avoid studying an already-failing system.
2. **Implement attack scoring.** Differentiate the teacher-forced log probability of the specified target answer through the feedback rollout. Use checkpointing and frozen model weights.
3. **Validate numerics.** Check directional derivatives on a small FP32 configuration. Evaluate BF16 attacks on their actual discrete input lattice; analytic gradients there are optimization heuristics.
4. **Validate controls.** Require large-radius late attacks to work on some development cases. Forced target-logit bias must pass the exact-answer scorer.
5. **Match budgets.** Compare five optimization steps against random shooting with matched measured compute, including backward-pass cost.
6. **Lock the matrix, then run held-out tasks.**

The most useful mechanism intervention is **early versus late injection on paired trajectories**. On a small subset, also compare leading finite-time tangent directions with equal-norm orthogonal directions. Growth without answer control would locate the missing link between instability and behavioral reachability.

### Resource envelope and branches

**Estimates:** approximately \(2\text{–}6\times10^5\) forward-equivalent feedback steps, or 6–24 local GPU-hours. The supplied 190-token/s batching rate is not an appropriate assumption for differentiated rollouts; benchmark the pilot.

Use no API calls. Reserve rental spending for a handful of precision-sensitive cases on the small model, not a 70B expansion. Assuming [amount redacted]–6/hour, 4–10 H100 hours would cost approximately [amount redacted]–60; verify pricing and float64 operator support first.

- **Targeted advantage, greater early leverage:** expand radius–horizon mapping.
- **Targeted advantage equally strong late:** study readout susceptibility; do not attribute it to chaos.
- **Random and optimized attacks both succeed:** investigate broad basin accessibility, not special gradient control.
- **Expansion without targeted success:** retain the meaningful null; probe whether unstable directions are readout-invisible.
- **Positive-control failure:** repair the optimizer or measurement before drawing scientific conclusions.

# First 48 hours

**Hours 0–8:** Build the objective task generator, exact scorer, target-selection rule, dtype-aware perturbation operator, and paired replay harness. Deliverable: a reproducible eight-problem control table.

**Hours 8–20:** Run the cheap pilot. Measure actual throughput, realized perturbation sizes, baseline correctness, and early/late attack success. Check gradients and positive controls.

**Hours 20–30:** Use only development tasks to select temperatures, doses, and attack budget. If task accuracy is inadequate, simplify arithmetic before proceeding; do not select held-out cases by attack success.

**Hours 30–48:** Launch the held-out matrix. Deliverable: paired outcomes with denominators and compute consumption, plus the first early-versus-late comparison. If controls fail, use this window for a corrected pilot rather than a misleading large run.

# Two-week research roadmap

**Days 3–5 — Establish the main result.** Complete held-out runs, quantify uncertainty, and rerun a small set of successes and failures under fixed execution conditions.

**Days 6–8 — Follow the discriminating branch.**

- Selective early control: estimate radius thresholds across horizons.
- Generic late susceptibility: test margin-matched examples and greedy feedback.
- Large latent growth without control: compare unstable-direction growth with downstream target-score changes.

**Days 9–11 — Test one generalization.** Use a second small model family or a second objective task family, not both initially. Transfer attack settings unchanged where possible. Failure to transfer narrows the claim rather than invalidating the original controlled result.

**Days 12–14 — Choose the substantive continuation.** If directional control survives, test one-shot logit-space perturbations, which keep feedback states inside the embedding convex hull and offer a more deployment-relevant intervention. If control is absent despite validated amplification, move to direction 2: structural changes from compression may matter more than small transient perturbations.

This yields a bounded program whose successful outcome can be either **a controllability mechanism** or **a demonstrated separation between chaos and actionable control**—both materially beyond another observation of divergence.
