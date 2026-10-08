# Broader-data LM profile: resource feasibility passed

Exploratory engineering result, 8 October 2026. **The larger fixed-data challenge
fits the measured local resource envelope. No new learning-utility conclusion is
claimed, and the four target development trajectories remain unrun.** The
previous failed calibration remains stopped; this profile supplies no shrimp,
attractor, inference or controller result.

Execution source `ca9e96b362189d3220370d798f649eb24fbfdc00` was committed and
independently reviewed at `e0c09a21be4ef2cd370e19511d71e664754f0dd7` before the
root-authorized GPU reservation. See the [design](plans/explore-tiny-lm-broader-data.md)
and [compact receipt](explore-tiny-lm-broader-profile-receipt.json).

Seed 28160 completed exactly 100 updates on each nested 8/expanded 256-window arm.
Their initial state hash and learning rate are identical; target counts are 256
and 8192. The process exited successfully in 4.409s, and both full 101-state paths
are finite and retained. Each arm has 100x82 feature rows and 84 complete
16-update progress targets. Probe observations at steps 0/25/100 are retained
but do not enter feasibility, predictor fitting or design selection. Profile
outcomes are excluded from the four target cases' fitting/scoring.

| Engineering measurement | Value | Declared limit |
|---|---:|---:|
| 8-window 100-update collection | 0.3497s | profile shared 60s internal |
| 256-window 100-update collection | 0.3293s | same shared cap |
| Expanded initial 32-step Ritz computation | 0.7975s | included in profile |
| Conservative projected four-case collection | 48.6892s | 420s |
| Peak CUDA allocated memory | 125,251,584 bytes | 6 GiB |
| Projected retained unit data | 249,025,879 bytes | 512 MiB |

The precommitted projection is
`1.5*(16*(T8+T256)+2*R+20s)`. Memory excludes driver/context allocation; the
fresh prelaunch inventory showed RTX 2000 Ada at 0% use, 2 MiB allocated and no
foreign compute process, driver 580.178.04. CPU/BLAS threads 1, nice 10,
float32/TF32-off, CUDA 13.0. This is one short profile, not a throughput scaling
law or a proof that wider batches are generally cheaper.

At one rounded initial state, expanded-objective CPU float64/GPU float32 relative
discrepancies were 7.10e-8 for the gradient, 8.95e-8 for the HVP and 1.93e-8 for
the update. These support a matched-state implementation check, not trajectory
precision equivalence. The maximum initial Ritz estimate 7.36499 had residual
5.116e-5 and provides the stipulated rate scale only; common eta 0.434488.

The exact 29-file inventory (13,286,641 bytes), token/data identities, source and
launcher bindings, state completeness and progress-label replay pass. Raw files
and logs are retained read-only under
`/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-001-profile`.
All learning/probe/guard outcomes are preserved in those files. The decision to
propose collection uses only implementation validity and the declared resource
bounds; no favorable CE, burst or probe result is used to pass it.

Next action is coordinator authorization of the exact four-case collection and
one offline analysis in the appended prospective run card. No rate adjustment,
extra seed, retry, policy or old protected-text evaluation is proposed.
