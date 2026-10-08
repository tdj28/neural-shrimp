# Proposed tiny-LM follow-up: useful diagnostics during drift

Status: **design only; no launch authorized by this document**. Drafted 2026-10-08 UTC after the exploratory pilot at `021e96b6d5d4b1e9c3f550c2dce8bcfed5b30432`. Prior results and raw attempts remain immutable. This is a development experiment, not a confirmatory test of a headline claim. Commit implementation, tests, this plan's final hash and a research-log run card before execution; disclose adaptations in dated amendments.

Question: can cheap recent-update geometry improve causal loss-burst prediction or learning-rate control after accounting for drift and strong ordinary baselines? Exact recurrence is unnecessary for this question. Predictive benefit and closed-loop benefit are separate endpoints. No rotation-number, Floquet, NS, chaos or geometry-specific claim follows from success.

Cohen et al. already report transformer EoS ([Appendices I.4/L.1](https://arxiv.org/pdf/2103.00065v3)); existing manuscript comparisons cover gradient reversal and warmup. Neither observation would be a new result here. See the [reference audit](../reviews/core-edge-references.md).

## Fixed task and development split

Reuse the 15,776-parameter transformer, tied-parameter accounting, context 32, float32/TF32-off arithmetic, ridge 0.001, no dropout, and corpus SHA `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. The next-character input/target convention and existing first-90%-prefix vocabulary remain unchanged. Offsets below are decoded-character offsets; assert every 33-character window is disjoint across roles.

| Role | Seeds | Eight training starts or sixteen validation starts |
|---|---|---|
| Calibration trajectories/policies | 28151, 28152 | `4096*j`, j=0..7 |
| Later development assessment | 28153, 28154 | `262144+4096*j`, j=0..7 |
| Calibration validation text | none | `524288+4096*j`, j=0..15 |
| Later assessment validation text | none | `655360+4096*j`, j=0..15 |

The original final 10% validation interval and seeds 29000--29999 remain untouched. The assessment seeds/text are fresh development evidence, not a permanent confirmatory set. Do not evaluate their outcomes until the predictor, selected policy settings and analysis code have been committed. Vocabulary presence is disclosed; these are disjoint target windows, not an unseen-character test.

## Causal features and prediction test

At state `theta_t`, compute training CE `L_t` and gradient `g_t`; issue the score before applying the next update. Only current/past information is allowed. For plain GD define `u_t=-g_t/||g_t||`. Retain recent unit directions and actual steps `s_t=theta_(t+1)-theta_t`.

A single angle `acos(u_t dot u_(t-1))` contains exactly the same information as gradient cosine. It cannot establish incremental geometry value. Test the following nested predictors, all logistic ridge models with mean cross-entropy plus squared-coefficient penalty, the same four penalty weights `{0.01,0.1,1,10}`, and calibration-only standardization:

- B0: CE level; causal 32-step slope, median/MAD; rolling AR(1)/AR(2) forecasts and past forecast errors.
- B1: B0 plus gradient norm/change, consecutive cosine, parameter-norm drift, and secant curvature `(g_t-g_(t-1)) dot s_(t-1) / ||s_(t-1)||^2`.
- B2: B1 plus the complete Gram matrix of the last four unit directions and recent step-norm ratios. This is the matched-history baseline.
- B3: B2 plus adjacent-plane coherence: for `a=u_(t-2), b=u_(t-1), c=u_t`, use `((a dot b)*(b dot c)-a dot c)/sqrt((1-(a dot b)^2)*(1-(b dot c)^2))`, and its causal 8-step mean/variation. Also give B2 the corresponding eight-step raw-history summaries.

The last feature is a coordinate-dependent nonlinear summary of existing dot products, not independent information. Near parallel/antiparallel directions make the denominator ill-conditioned: mark unavailable when either factor is below `1e-4`; retain the mask, and never turn an undefined angle into zero rotation. All baselines get the same masks. Validate formulas on synthetic straight, alternating, drifting and rotating directions, including near-degeneracy.

Label a future burst at time t when any CE in the next 16 updates exceeds the preceding 32-step median (including t) by `max(0.10 nats, 3*1.4826*MAD)`. Record numerical guard stops separately and count a guard inside the horizon as an adverse event; otherwise label right-censored horizons unavailable. This is an operational prediction target, not a dynamical-instability definition. Record event counts and temporal coverage.

Stage A: 800 updates per constant-rate trajectory, factors `{0.8,1.6,2.5}` times `2/initial_max_Ritz`, with the same bounded Lanczos settings as the pilot. The Ritz value supplies a rate scale only. First run the six calibration trajectories. Fit on steps 32--383, tune on 400--583, and assess on 600--783, so label windows do not cross boundaries. Use leave-one-calibration-seed-out checks as well. Report Brier score, precision-recall area, event lead time, false alarms per 100 steps, and runtime by trajectory; no step-wise independence assumption or significance claim.

Proceed to the six later-assessment trajectories and Stage B only if B3 reduces time-held Brier score by at least 5% against B2 on both calibration seeds without reducing precision-recall area, with at least five distinct burst episodes per seed across its three rates (episodes separated by 16 quiet steps). If events are too sparse or prediction adds nothing, retain the negative/uninformative result and stop. Any revised challenge is a new disclosed exploratory attempt. If the gate passes, freeze the predictor, finish the calibration-policy sweep, and commit selected settings before opening any assessment outcomes.

## Closed-loop utility, separately from forecast accuracy

Only after the gate, compare five policy families: constant rate; 50-step linear warmup followed by cosine decay to 0.1 times the peak; generic current-loss backoff; B2-score backoff; B3-score backoff. All backoffs multiply the current rate by 0.5, use a 16-step cooldown, and have floor 0.05 times the original rate; no recovery or rollback. Generic backoff uses the same past median/MAD threshold as the burst definition. Score thresholds are fixed at the calibration 95th percentile, giving equal nominal alarm budgets; disclose realized interventions. Predictive actions affect only future updates.

Each family gets exactly four peak-rate candidates `{0.8,1.6,2.5,4}` on the two calibration seeds, 800 updates each. Select one factor per family by mean final calibration-validation CE, with guard failure ranked worst and lower rate breaking ties. Then run each selected policy once on each fresh assessment seed/text pairing. The only tuning difference between B2 and B3 is the feature set. Report all candidates and failed attempts.

Primary control endpoint: mean token CE on the assessment validation windows after 800 gradient evaluations, paired by seed/initialization/text. Also report train CE, guard failures, interventions, completed updates, validation CE at steps 0/200/400/600/800, and wall time. Validation metrics never trigger online actions. Equal gradient budgets and equally sized tuning searches do not guarantee equal runtime: measure feature and validation overhead, and report both. A better forecast with unchanged/worse validation is a failed utility bridge.

## Resources, evidence and later extensions

Upper bound if all stages proceed: 12 diagnostic trajectories + 40 calibration-policy trajectories + 10 assessment-policy trajectories = 49,600 GD updates. Prior throughput was roughly 2--5 ms/update, suggesting 100--250 seconds for training kernels alone. Profile 100 updates with all features/validation first; use at most 30 minutes owned local GPU wall time, one CPU thread, nice 10, and 5 GiB new retained outputs. Stop if profiling makes the cap infeasible. Recheck shared GPU ownership before any run; no paid compute is proposed.

Store every state (roughly 3.2 GB before compression at this maximum), scalar/feature trace, causal predictions, policy actions, checkpoints and all guards in a new data-store attempt, with exact source/data/config hashes and runtime/precision metadata. Replay selected features by recomputing gradients at saved states, and verify decisions cannot read future losses. No new full Hessian or Floquet sweep is needed. Representative matched-state float64 checks assess arithmetic sensitivity, not trajectory equivalence.

This unit isolates deterministic full-batch drift. Constant-rate GD supplies the autonomous control; scheduled and history-dependent rate policies are separate arms. Log their counter, recent-data buffers, cooldown and current rate as controller state. A later minibatch unit must use paired batch-index streams, fixed evaluation anchors, and distinguish sampling noise from secant curvature. A later Adam unit must save `(theta,m,v,step)`, schedule position, and RNG/sampler state; any derivative or rollback must include the relevant state. Neither extension launches automatically, and pooled results across these regimes would obscure the scientific question.
