# Broader fixed data: a new tiny-LM development challenge

**Status: design for independent review; no numerical run authorized by this file.**
Dated amendment, 8 October 2026, 07:20 UTC. Base integration:
`5e716f6a191e58022e17052277d4de90c44091c7`. Exploratory and adapted after seeing
[the negative calibration](../explore-tiny-lm-calibration.md) and its
[independent outcome review](../reviews/tiny-lm-calibration-outcome-review.md).

**Plain-language question:** does giving the same tiny language model a broader
fixed training set improve learning on a separate development slice, and can
recent update geometry forecast learning progress better than ordinary loss
history? This concerns deterministic full-batch training. It is not a claim
about shrimp regimes, verified attractors, inference chaos or a useful control
policy. There is no new result yet.

## Why this is a different, bounded question

The preceding six-case unit stays stopped. Five of its trajectories memorized
256 repeated target tokens to training CE near 0.009–0.011. Its late test windows
contained zero and three burst episodes across the two seeds, below five per
seed; B3 also lost to B2 on the eventful seed's Brier and pooled AP. B0 loss/AR
features were stronger on the sole eventful trajectory. These outcomes shaped
this amendment and remain immutable; no old gate, rate or threshold is relaxed.

There is **one controlled training change**: 8 versus 256 distinct fixed training
windows. The arms share initialization, model, optimizer, arithmetic, exact
learning rate and update budget. We deliberately retain deterministic full-batch
GD rather than adding minibatch noise, a schedule or Adam state. The separate
**endpoint change** is explicit: predict continuous 16-update CE progress instead
of conditioning usefulness on rare binary bursts. Quiet, smooth learning is a
valid outcome. Burst labels remain descriptive and cannot open a later stage.

All model settings stay fixed: 15,776 parameters, context/width 32, one causal
attention block, tied embedding counted once, ridge 0.001, ordinary existing
initialization, no dropout/momentum/scheduler, float32 with TF32 disabled.

## Exact development data and cases

Reuse the already archived tinyShakespeare source, SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`,
1,115,394 bytes. Source/license attribution stays in its existing dataset
receipt. Use the existing vocabulary from the first 90% plus the unknown token.
This vocabulary exposure is disclosed; probe outcomes, not its alphabet, are
new. No external data or network download is needed.

Each start `s` names the 33-byte/character interval `[s,s+33)`: inputs `[s,s+32)`,
next-character targets `[s+1,s+33)`. ASCII equality of byte/character offsets must
be asserted for the selected slices before conversion. Token IDs must use the
unchanged vocabulary order. Retain the full start/byte-interval/token-ID lists,
source SHA, selected-byte SHA, and integer-token-array SHA in the new manifest.
Do not print text into Git or read protected outcomes for this bookkeeping.

| Role | Exact starts | Envelope, exclusive upper end | Targets/evaluation |
|---|---|---|---:|
| Expanded fixed training pool | `65536 + 256*j`, `j=0..255` | `[65536,130849)` | 8192 |
| Nested 8-window control | first eight expanded starts | `[65536,67361)` | 256 |
| New development probe | `196608 + 256*j`, `j=0..31` | `[196608,204577)` | 1024 |

All 33-character windows are mutually disjoint except intentional training-arm
nesting. Arithmetic interval checks pass against the prior training windows,
old assessment starts `262144+4096*j`, old validation starts `524288+4096*j` and
`655360+4096*j`, and the final 10% interval `[1003854,1115394)`. Executable code
must repeat these checks. The probe is explicitly development-only and cannot
be reused as untouched confirmatory evidence. Its CE is never a feature,
training objective, rate/penalty/checkpoint selector or stopping criterion.

Target cases: seeds **28161 and 28162**, each crossed with 8 and 256 windows:
exactly four trajectories, 800 updates each. A separate engineering profile uses
seed **28160**, at most 100 updates per arm, and is excluded from predictive
fitting/scoring. Assessment seeds 28153–28154, reserved seeds 29000–29999 and all
old protected text outcomes remain untouched. New seeds are ordinary exploratory
initializations, not a fresh confirmation of a preexisting headline claim.

For each seed compute the initial maximum Ritz estimate on the **256-window
objective only**, using the existing 32-step Lanczos and direction seed 28165.
Require a finite positive value and finite residual in `[0,0.02]`, or stop that
seed as unresolved without changing Lanczos length. Use the identical rate
`eta = 1.6 * 2 / initial_max_Ritz` in both arms. The factor 1.6 is fixed because
the earlier unit's corresponding cases learned without late bursts; it is not
chosen to maximize oscillation. The Ritz value is a scale, not an exact spectral
bound or stability certificate. No extra rate, arm or initialization is allowed.

## Learning endpoints and continuous prediction

The main learning endpoint is paired development-probe CE difference
`CE_256(theta_800) - CE_8(theta_800)` for each seed; negative favors broader data.
Report both train CEs, the train–probe gaps, processed target-token counts,
gradient calls, wall time and memory. The arms match update count, **not** token
or runtime cost. A predeclared equal-training-token comparison is probe CE of
the 256-window arm at update 25 versus the 8-window arm at update 800: both have
processed 204,800 training targets. This does not equate optimizer steps or wall
time. Evaluate probe CE only at saved steps `0,25,100,200,...,800`, including
step 25 in both arms for a uniform schedule; every scheduled outcome is retained.
No checkpoint is selected. Report the two seed differences individually and
their mean/range; two independent seeds do not support a precise population CI.

At each training state `theta_t`, features may use only information through the
current loss/gradient. The continuous target is
`y_t = CE(theta_(t+16)) - CE(theta_t)` on that arm's fixed training objective.
Negative means progress, positive means worsening. It is not a stability or
chaos definition. Complete horizons only; guards and right-censoring remain
explicit and are not imputed as numeric labels.

Fit on `t=32..383`, tune on `400..583`, score on `600..783`; target observations
end at 399, 599 and 799 respectively. Keep the 16-update embargo. Fit each seed's
predictor separately, pooling its two arms only in the early fitting/tuning
windows. This tests time-held prediction within a development seed, not unseen-
seed generalization. Standardization and missing-value imputation use fitting
rows only. Each arm's score scale is the fitting-window RMS of `y`, floored at
0.01 nats; it is fixed before tuning and used only for metric normalization.
Predictor regression fits raw progress in nats, with an unpenalized intercept,
mean squared error plus ridge penalty, and the same four penalties
`{0.01,0.1,1,10}` for every fitted family. Select by mean normalized tuning MSE
across arms; higher penalty breaks exact ties. No late outcome affects fitting,
penalties, transformations, normalization or comparator selection.

Comparators are fixed before outcomes:

- Persistence: predicted progress zero.
- Direct AR(1) and AR(2): intercept-inclusive least-squares models on the latest
  32 observed CEs, recursively forecast 16 steps without future teacher forcing.
- B0: the existing causal loss level/trend/median/MAD and AR/error features.
- B1: B0 plus existing gradient norm/change, cosine, secant curvature and norm
  drift.
- B2: B1 plus the full eight-history Gram representation and step ratios.
- B3: B2 plus the current/eight-history plane-coherence summaries.

B0–B3 receive identical geometric-availability masks and the known log number of
training targets as a common feature; no seed ID or future information. B2 must
retain enough raw history to reconstruct B3 wherever defined. Keep the reviewed
near-collinearity abstention floor unchanged. B3 is nonlinear featurization of
matched information, never new information or a verified NS signal.

Main forecast endpoint: normalized late-window MSE on the **256-window arm**, per
seed; also report MAE, signed bias, every comparator and the nested-control
scores. Select the ordinary comparator from persistence, AR1, AR2, B0 and B1 by
mean normalized tuning MSE across arms (ties in that listed order). A narrow
positive forecasting result requires at least 5% lower B3 MSE than both this
selected ordinary comparator and B2 on each expanded-arm seed, with no worse
MAE. If either comparator's normalized MSE is below `1e-6`, label incremental
forecasting uninformative instead of dividing by a near-zero error. Always
report the full continuous differences; this rule does not authorize a follow-
on stage. Overlapping horizons are correlated; no window-level p-values,
bootstrap or independent-event claim. Probe quality and forecasting are separate
endpoints: a forecasting gain alone establishes no useful training intervention.

## Prospective engineering profile and hard stops

This is a proposed resource envelope, not an allocation or executable launch
card. After source and local controls are committed/pushed and independently
reviewed, coordinate the short GPU reservation with the root agent (CIFAR may
also need it), then recheck CPU/RAM/GPU/process inventory. CPU/BLAS threads 1,
nice 10; no paid resource or consultation. Do not obtain capacity by disturbing
another process.

Profile both arms for at most 100 updates at seed 28160, using the same data,
features, retention and probe schedule that collection would use. Include one
matched-rounded-state CPU-float64/GPU-float32 gradient/HVP check for the expanded
objective, with the same limited implementation-parity interpretation as before.
Internal profile cap 60s, external 80s plus 5s cleanup grace. Preserve profile
learning outputs but **gate feasibility only on runtime, memory and storage**;
no rate/data/endpoint choice may depend on favorable loss or burst behavior.

Let `T8` and `T256` be the measured 100-update arm times including scheduled
probe evaluations, features and durable writes, and `R` the measured expanded
initial Ritz time. Permit proposing collection only if
`1.5 * (16*(T8+T256) + 2*R + 20s) <= 420s`, CUDA peak allocated memory <=6 GiB,
and projected total retained unit data <=512 MiB. Report the allocation/context
memory distinction. If a bound fails, retain the result and stop; do not reduce
precision, examples, controls or retention silently. A changed implementation
or envelope needs a new dated amendment/review before a new attempt.

Prospective collection ceilings: shared internal 450s, external 480s plus 5s
cleanup; one CPU-only analysis under a 30s cap. Including the profile, these
external ceilings total at most 600s. The 3,204 target states require exactly
202,185,216 raw float32 bytes before feature/manifest overhead. Store every
state, causal feature row, loss, guard, partial attempt, probe result, fitted
coefficient, candidate tuning score and prediction in new exclusive directories
under `/data2/neural-shrimp-data/explore/tiny-lm-broader-data/`, with compact
source/plan/data/review/hash-bound receipts in Git. Never overwrite old LM data.

A numerical guard ends that case, leaving the other fixed cases eligible only
within the shared cap; missing/censored cases cannot satisfy the positive
forecast interpretation. No retries or substituted seeds/rates. SIGTERM must
retain the last accepted state and partial evidence through the reviewed cleanup
path; SIGKILL/storage failure remains a disclosed limitation. Refuse dirty
executable source and exact-inventory/hash mismatches. Stop after the four-case
learning/prediction report whether favorable, negative, quiet or inconclusive.
No controller, old assessment, inference side experiment or paper edit follows
automatically. Source review and a concrete run card must precede launch.
