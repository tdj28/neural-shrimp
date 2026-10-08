# Tiny-LM drift prediction: independent source review

**TLDR.** The reviewed machinery can test whether nonlinear summaries of
recent gradient geometry improve prediction during drifting tiny-transformer
training. It has not established predictive gain, controller utility,
language-model validation, recurrence, or an NS-specific signal.

**Verdict:** no remaining source blocker for the exact six-case exploratory
calibration card at `b9e5b8c3973e8bd776402b0cb3d8d61609bb71f1`.
All 23 base, producer and independent tests passed in 2.29s, CPU1/nice10.
This verdict does not clear assessment, validation outcomes or controller runs.

## Source and resolved findings

Initial frozen source: `5c477cca9445d507f1e1c58f8c93154ef35097e3`.
Corrected source: `b9e5b8c3973e8bd776402b0cb3d8d61609bb71f1`.
Reviewed `tiny_lm_prediction.py`, both prediction drivers, producer tests,
`docs/plans/explore-tiny-lm-drift-controls.md` and the prospective run card.
The unchanged causal transformer, tied flat parameters and objective derivatives
were covered by the earlier review and retested here. No producer files were
edited; no calibration, assessment, validation, GPU or paid run was launched
by this reviewer. Review branch: `codex/tiny-lm-prediction-review`.

Precommit review identified that Gram-history means/std do not determine
nonlinear coherence-history means/std. Before source freeze, B2 gained all
eight complete Gram vectors and a reconstruction control. Exact analysis
input inventory validation, committed-source checks, and imported-source plus
NumPy/SciPy analysis identity were also added before freeze.

Two findings on the frozen snapshot were corrected in `b9e5b8c3`:

- The Ritz scale gate admitted NaN through false comparisons, and infinity
  could produce a zero rate. It now requires finite positive scale and finite
  residual in `[0,0.02]` before constructing eta; invalid and boundary controls pass.
- A reproduced exception at `progress-000050.json` reported 50 accepted updates
  but saved only theta_0 through theta_49, omitting terminal theta_50. Cleanup
  now retains the accepted state exactly once; the same independent control
  passes with 51 rows. SIGTERM now raises `CollectionInterrupted` so ordinary
  failure/result/manifest cleanup can run before the external kill-after grace.

Forcible SIGKILL, storage/device failure, or failure during cleanup can still
prevent final flushing. Fsynced 50-state chunks and launcher evidence provide
partial durability; an incomplete attempt is not silently accepted.

## Scientific contract checked

Features at t use only current/past CE, objective gradients, parameter norms,
previous actual parameter steps and past forecast errors. AR forecasts use the
last 32 available losses. Prefix perturbations leave earlier features unchanged.
Secant curvature uses `(g_t-g_(t-1))` and the observed prior parameter step.

B2 retains the current four-direction Gram vector and seven preceding vectors.
B3's current/mean/std plane coherence is a deterministic nonlinear function of
that retained information. Independent random-direction controls reconstructed
all three B3 values across 58 rows. Parallel/antiparallel degeneracy produces
unavailable geometry, not zero rotation. Every family receives identical
availability masks. Only the geometric feature abstains: the logistic model may
still issue a score using training-imputed values and shared masks. B0 thus
includes that shared availability metadata as well as its loss features.

Imputation and standardization use fit rows only, t=32..383; penalties are
selected at t=400..583; scores are tested at t=600..783. Sixteen-step labels
finish at399/599 for the preceding fit/tune blocks, before the next split.
Independent controls changed CE at400+ without changing fit labels, and at600+
without changing tuning labels; changing the final included observations399/599
changed the expected boundary labels. Mutating all test features/outcomes left
models, scalers, penalty selection and thresholds unchanged. Feature-history
overlap is legitimate past information, not independent time-step replication.

All families use the same four penalty candidates and training-score 95th-
percentile thresholds. AP groups tied scores; it is the declared PR summary.
The development gate uses mean per-trajectory Brier and pooled AP across three
rates, separately for each calibration seed, plus the declared episode count.
Leave-one-seed-out output is diagnostic, not another selection opportunity.
A positive result could support useful nonlinear featurization for this model
class, not new information, an NS signature, or geometry-specific causal utility.

## Protected scope, guards and resources

The collector exposes only profile/calibration. Calibration fixes seeds28151/
28152, factors0.8/1.6/2.5 and800 updates. Assessment seeds28153/28154,
validation losses, final-10%-text outcomes and reserved seeds29000..29999 are
not evaluated. Corpus hash/offsets are asserted; vocabulary uses the disclosed
first90% prefix. Analyzer requires exactly the six calibration identities
before fitting. Protected modes and invalid/nonfinite budgets were rejected.

A burst label means a future CE threshold crossing or numerical guard within16
steps. Other incomplete horizons are censored. Wall stops are not numerical
guards. Synthetic controls verified the offending guard state/index and future
labels, plus initial-state retention with an already exhausted budget.
These are operational prediction events, not mathematical divergence labels.

Exact manifest inventory, byte sizes and hashes are checked; extra files,
modified payloads and unsafe symlinks were independently rejected. Analysis
outputs go outside the immutable collection. Source bytes must match committed
HEAD; collection and analysis bind their imported implementation files.

The final card uses internal240s/external260s plus5s grace for collection,
CPU analysis30s, CPU/BLAS1 and nice10. Including the already measured3.61s
profile, these execution caps fit the five-minute stage envelope. The internal
check operates between updates/HVPs; the external timeout supplies the process
cap. GPU ownership still requires a producer preflight. The older profile
summary's285s sentence was superseded by this tighter card.

## Independent controls and existing-profile audit

Reviewer-owned `tests/test_tiny_lm_prediction_independent.py` adds nine controls:
label embargo endpoints; training-only statistics/test-outcome mutation;
degenerate directions/prefix invariance; numerical-guard state/labels;
expired-budget initial state; invalid modes/budgets; full-history reconstruction;
inventory tampering; and terminal-state retention after a checkpoint exception.
The last test reproduced the original defect and passed after correction.
The final combined suite passed23 tests in2.29s, with one existing Torch
`jit.script` deprecation warning.

The existing engineering profile remains bound to source `5c477cca` and
manifest SHA256 `43d33d6de3fcfee2d066d4800841e9f70e5e7c0822ec9657264b7c068838f127`.
Independent checks validated all13 files/6,467,760 bytes, historical source
hashes, launcher/log hashes and quoted case fields. The saved101x15,776 states
are finite; burst labels replay exactly (53 uncensored, zero positive).
All68 eligible saved feature rows reconstruct B3 from B2. No model was fitted
from that profile, and the reviewer did not recompute GPU gradients. Its
matched-state precision check does not prove trajectory precision equivalence.
