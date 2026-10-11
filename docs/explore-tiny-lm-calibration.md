# Tiny-LM development calibration: gate failed

Exploratory, 2026-10-08 UTC. The six fixed calibration cases completed, and the committed continuation gate failed on both seeds. **No assessment, validation-outcome evaluation or policy runs followed.** Sparse late events and worse B3 performance provide no evidence of useful nonlinear geometry features in this challenge; they do not establish that such features can never help.

Source `b9e5b8c3973e8bd776402b0cb3d8d61609bb71f1` was committed with the prospective card and independently reviewed at `39ba17219075c92cd33038b032501bb60da6fd25` before launch. See the [plan](plans/explore-tiny-lm-drift-controls.md), [engineering profile](explore-tiny-lm-prediction.md) and [compact receipt](explore-tiny-lm-calibration-receipt.json). Earlier evidence is unchanged.

## Fixed result

Each seed 28151/28152 used rates 0.8/1.6/2.5 times 2/its initial maximum Ritz estimate, 800 full-batch GD updates and the original fixed eight context 32 text windows. All 4800 updates completed without numerical guards. All 4806 states are finite and retained, along with 4800 rows of 81 causal features, labels, predictions, fitted models and all tuning candidates. There are 753 complete 16-step future labels per case; the fitting/tuning/test windows contain 352/184/184 labels, respectively. No label crosses a split embargo.

| Seed | Late positive horizons / episodes | B2 mean Brier | B3 mean Brier | B2 pooled AP | B3 pooled AP | Gate |
|---|---:|---:|---:|---:|---:|---|
| 28151 | 0 / 0 | 3.846e-8 | 7.639e-8 | undefined | undefined | fail |
| 28152 | 49 / 3 | 0.063011 | 0.063782 | 0.685126 | 0.647679 | fail |

The fixed gate required at least five episodes on each seed, at least 5% lower B3 Brier and no lower pooled AP. None of those conditions supports continuation. Positive horizons overlap heavily; 49 is not 49 independent events. The sole late eventful trajectory is seed 28152 at factor 2.5 (eta 0.723665). On that trajectory, the ordinary loss/AR baseline B0 achieves Brier 0.173557 and AP 0.911133, versus B2 0.189033/0.693410 and B3 0.191347/0.694245. Per-trajectory AP differs from the pooled seed AP because the latter also ranks predictions from quiet rates. Every family selects penalty 0.01 from the same four candidates using only tuning windows. Leave-one-seed-out results are retained; they do not rescue the gate.

Five trajectories finish near training CE 0.009–0.011; the eventful highest-rate case ends at CE 1.483. These are memorization/training measurements, not validation performance. Geometric features often abstain late as update directions become nearly collinear: the exact availability fractions and masks are retained. B2 contains every raw Gram history needed to reconstruct B3, so an eventual B3 benefit would mean useful nonlinear featurization of matched information, not new information, NS, Floquet or recurrence evidence.

## Resources and evidence

Calibration process 19.931s; offline CPU analysis 0.711s. Including the earlier profile 3.610s, launched process time totals 24.252s, comfortably within the five-minute ceiling. The six cases' shared internal 240s/external 260s(+5s grace) cap and analysis 30s cap were unchanged. Hardware was the locally owned idle RTX 2000 Ada with one CPU/BLAS thread and nice 10; no paid call occurred. Raw collection: 221 hash-verified files, 307,259,570 bytes, plus separately hashed analysis/launcher evidence. All retained raw files are read-only. Source review ran 23 focused/independent controls; the result inventory and state completeness were verified after collection.

This unit stops at its negative/sparse calibration outcome. Fresh assessment text/seeds 28153–28154, all validation outcomes, the final 10% target interval, reserved seeds 29000–29999 and controller policies remain unopened for outcomes. A future revised challenge would require a new dated design and fresh exploratory attempt; no such run is proposed or launched here.
