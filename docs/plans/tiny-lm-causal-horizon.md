# Causal validity horizon of a frozen local response model

11 October 2026 UTC. Exploratory development amendment, fixed before computing
new rolling-prefix outcomes. The original response and two-snapshot audits are
consumed evidence: frozen curvature failed in the expanded arm, while the
two-snapshot forecast improved but missed its fixed accuracy gates. This asks
whether a measurement available after one local update warns that the frozen
response will become inaccurate over the next three updates. It is not a new
training intervention, a chaos test, or evidence of generalization.

Use only the four existing float64 reference trajectories: seeds 28161/28162,
8/256 training-window arms, original update600 checkpoints and objective. Fix
origins t=0,4,8,12 and both original directions, giving32 correlated development
cases. No new text, rate, seed, protected holdout, trajectory or perturbation.
The response at origin is the continuing original perturbation normalized by
the transported tangent norm at t; this is not a fresh perturbation experiment.
Retain failures of that interpretation under the finite-difference checks below.

At each origin, predictor inputs stop at local theta1=theta[t+1]: past tangent
u=v[t]/||v[t]||, v1=J[t]u, a2=J[t+1]v1, g[t], g[t+1], loss[t], loss[t+1].
Although a2 is stored at tangent index t+2, it requires only the derivative at
theta[t+1]; no theta[t+2] value or later Jacobian enters the predictor. This
restriction must be represented by a separate input object and tested by
mutating all later evaluation arrays. Frozen f2=J[t]v1 and f3,f4 use theta[t]
only. Reuse the original frozen forecast at t=0 and acquire exactly three new
HVPs per later origin/direction:72 target HVPs total. Exact training losses at
the two early states require32 target CE evaluations; add the original ridge
term explicitly. Original gradients/tangents are authenticated retained data.

## Fixed forecast and comparisons

Let e2=||a2-f2||/max(||a2||,1e-12). The proposed warning extrapolates the leading
triangular accumulation of derivative drift: predicted frozen error at horizon
h is h(h-1)e2/2 for h=1,2,3,4. This is a heuristic, not an error bound. Its
predicted first crossing of20% is the forecast validity horizon; report right
censoring at4. It adds a specified growth assumption to the early defect and
must not be advertised as information beyond that defect.

All comparators use only the same early information budget, with fixed alarms
at20% and no fitting: persistent defect e2 at all later horizons; gradient
change h||g1-g0||/max(||g0||,1e-12); relative loss change
h|L1-L0|/max(|L0|,1e-12); static directional curvature
h||v1-u||/max(||v1||,1e-12); and the maximum of those four baseline scores.
Report raw scores, full confusion counts, missed crossings and false warnings,
plus whether the drift warning resolves any failure missed by the composite.
Also retain ordinary response persistence, scalar one-step gain and the frozen
vector prediction as response-error comparators. These are not extra labels
used to tune any warning. Information availability and acquisition costs must
remain explicit; gradients/losses are not substitutes for an extra HVP.

## Evaluation, separated from predictor construction

Only after serializing all32 predictor records, compute later responses from
the two saved central perturbation amplitudes at h=1,2,3,4. Normalize by the
same initial transported norm. At each required horizon, two-amplitude vector
agreement and actual ordered-transport agreement must each be within1%, with
denominator max(response norm,1e-12). Unqualified cases remain unresolved.
First frozen-vector error above20% is the observed crossing. The primary
future-warning table uses cases still accurate at h=2; cases already failing
then are a separate detection table, not successful advance warnings. Report
every origin/seed/arm/direction and dependence, not pooled p-values or claims
from32 independent samples. Four original checkpoints cannot establish general
LLM behavior, useful training control, or shrimp-specific effects.

## Bounded implementation and provenance

Bind the original response outer fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5,
independent response outer296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948,
and two-snapshot independent outer7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d.
Retain exact native/source/data/coordinate/ridge/rate identities and all attempted
inputs before calls. Implement separate source-reviewed profile and target
units; this plan alone does not launch a target. Profile at the old consumed
profile seed28160/update50, two HVP+CE per arm (four each), using maximum
per-operation/arm cost. Target72HVP+32CE, no gradient call. Forecast
2*(36*sum_arm(max HVP)+16*sum_arm(max CE))+20s must fit180s; historical measured
costs forecast about44s. Fixed CPU/BLAS1nice10,4GiB RSS/8GiB AS/128MiB raw;
profile60s and target180s with80/200+5 owned launchers. Preserve immutable
failures, exact source archive and all raw predictions/evaluations. New source,
controls, committed card and independent peer review precede either unit.

### Narrow legacy-container amendment before any native operation

Profile001 rejected the original broader-data directory775 despite complete matching read-only files (29profile/191target). Its immutable failure and exact source are mandatory inputs to fresh002. Do not alter old modes/evidence. Recognize only the two literal B.PARENTS root+manifest pairs as historical writable containers; still require complete fresh inventory, exact content hashes and read-only manifest/every file. Every new input/attempt remains a fully closed directory. No scientific selection, numerical operation, prediction rule or resource cap changes. Fresh002 profile must pass before separately bound002 target.
