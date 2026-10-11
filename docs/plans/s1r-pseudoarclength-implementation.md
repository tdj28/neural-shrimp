# Local PAL implementation and prospective run card — 8 October 2026 UTC

Source and synthetic tests only. Root accepted design9f472f175a48fcfb13680343072fe4353264b911
with the per-leg predictor radius and explicit actual-native spectral adapter.
No target, model-data profile or outcome has been run/read by this implementation
unit. Target execution still requires an independent producer source review,
a frozen/reviewed prospective saved-outcome auditor and separate root approval.
Stop preparation at the root's14:30UTC decision if those are not ready.

The new core is `src/neural_shrimp/s1r_pseudoarclength.py`, with driver and bounded
launcher `tools/{explore,launch}_s1r_pseudoarclength.py`. All historical helpers,
raw data and exact-real evidence remain unchanged. Runtime source inventory is
`SOURCES` plus every `bound_repository_files` entry in the design input JSON:
17 distinct files, including indirect package init/controls/models imports,
old branch/Store/rank helpers, the independent analytic primitive, new runtime
files, both design files, this card and the committed evidence/input companions.
Every file must match clean execution HEAD; old bound files must also equal the
design's byte identities. Tests are separately committed, not runtime imports.

## Disclosed implementation amendments before target work

Add an explicit **256 assembly cap**, independently of256 analytic derivative
calls and256 map/Jacobian constructions. Cached trials can otherwise emit more
assemblies than derivative calls. This tightens the work/storage envelope;
it changes no acceptance gate. Record each attempted assembly before its phases.
All102 SVDs retain complete U,s,Vh, including the68 norm/condition decompositions,
so the saved-data auditor can check matrix residuals without a fresh SVD.

The corrector's `orientation_reference` is separate from the parent plane normal:
direct/midpoint use the parent tangent; half-endpoint uses the midpoint tangent;
reverse uses the direct tangent. The reverse predictor has negative travel length
but the carried forward orientation remains unchanged. For each leg,
ell=||predictor−seed||_2>0 and correction distance<=ell/2. Native tangent transport
uses `(DF*t_theta + partial_h_F*t_h, t_h)`. Anchor decomposition is the anchor
qualifier's cached result; no extra initial SVD is hidden outside the34 allowance.

Closure uses the retained **actual second image**, not theta+R reconstructed with
another floating addition. Each native monodromy uses DF(F(theta_i))DF(theta_i).
The new spectral adapter takes these matrices separately, uses one full norm SVD
and one full condition SVD per phase, then the existing left/right eigen-residual
and cross-phase multiset equations/gates. It never calls the legacy spectral
helper, a hidden norm(matrix,2), or cond in addition. Its full decomposition
checks use Frobenius aggregate residuals<=1e-10; this is at least as strict as
per-column residual gates. No scientific tolerance is loosened.

At convergence the already evaluated state is retained; the tiny final Newton
step is not applied without evaluation. Small correction, parameter/plane gates
and original full physical closure/primitive qualification must all hold.
Every attempted trial is emitted, including an out-of-local-box trial; that
boundary rejects the edge immediately rather than silently backtracking.

The imported historical HiddenBridge stored an aligned parent pair, so its
`parent_match.phase` directly identifies the child's carried native phase;
`parent_phase XOR child_carried` is the native-pair exchange. The new PAL nodes
keep native pairs unflipped, so new edge carry obeys the usual parent-carry XOR
native-match rule. These conventions must not be conflated by the outcome audit.

## Record and partial-result contract

Use unchanged Store JSON/NPZ descriptors and complete manifest semantics. The
journal is append-only. Each expensive operation emits `operation_start` with a
global integer id, kind, complete inputs and cumulative attempted counters;
`operation_return` retains the complete return and increments returned counters
before subsequent checks. Raised calls emit `operation_exception`. Kinds/caps:
analytic256, map256, solve64, svd102, eig34. An attempted-counter increment can
precede a failed start emission; a returned increment can precede a failed return
emission. Preserve those actual boundaries, never invent completed operations.

Analytic derivatives cache by exact float64 theta bytes: the loss is independent
of learning rate. Map/Jacobian/parameter-partial results cache by theta+h bytes;
`cache_hit` names the original operation id and inputs. There are no repeated
analytic calls for identical keys. Map starts name their analytic id. Map returns
include image,J,c. Assembly starts include id,purpose,theta,h; assembly returns
include map1,map2,R,M,A,b and both map ids. Attempted assemblies count even if a
phase fails. No inferred derivative or free trajectory is added.

Each SVD start retains its input matrix and purpose; returns retain U,s,Vh.
`svd_checks` retains the residuals/orthogonality/rank classification. Eigensystem
starts retain the actual native matrix; returns retain all72 values and full
left/right vectors. Spectrum checks reference norm/condition SVD ids; qualifier
ends preserve the complete spectral assignment rows/columns. Tangent records
retain the SVD id, reference, orientation and residual; phase-transport records
retain both tangents, parameter-aware transported vector and checks.

`corrector_start/end` retain ids, seed,predictor,ell,plane origin/normal/offset,
orientation reference and latest evaluated/accepted state. Solve starts retain
z,G,J; returns retain the full step. Linear checks retain true residual, forcing,
nearzero label, Frobenius backward error, slope and correction sizes. Every
`armijo_trial` records factor,state and either full G/merit/decision or the actual
locality/nonfinite failure. Qualifier start/end retain full native pairs and
accepted status. Every started common-h check has an end even on failure, with
canonical_h and candidate_original_h distinct; there is no fixed-h Newton solve.

Every edge starts before its direct/half/endpoint/reverse work. Accepted edge ends
are durable before `last_witnessed` and accepted-edge lists advance. If that write
fails, the previously witnessed pair remains authoritative even if an in-memory
candidate was qualified. Failed and unfinished candidates remain separate;
node/native indices are never silently reordered. Terminal result includes all
attempted/returned/stage counters, last witnessed id, accepted edges and actual
status/error. Missing records after a hard kill retain only their saved scope.

## Resource, storage and synthetic screen

Limits remain CPU/BLAS1,nice>=10,60s internal; GNU timeout75s TERM+5s KILL;
owned-group supervisor80s; monitored2GiB group RSS (not kernel-hard). Caps include
16 arclength correctors,64 solves,8 iterations/corrector,13 trial factors,
17 orbit qualifiers,8 common-h state-only requalifiers,4 attempted/2 accepted
edges,256 analytic calls,256 maps/Jacobians,256 assemblies,102 SVDs and34 eigensystems.
No GPU, paid compute, protected dataset, bootstrap or free trajectory is used.

Total128MiB:127MiB Store (95.25MiB ordinary,31.75MiB finalization reserve),1MiB
launcher, log<=128KiB. A synthetic serialization screen with72-state maximum
payload shapes projects77,725,156 bytes against99,876,864 ordinary bytes. It uses
256 analytic,256 map,256 assembly,64 solve,102 worst-shape SVD and34 eigensystem
payloads plus8MiB allowance. That allowance covers split operation start/return
headers, bounded cache-hit descriptors, rank/linear/Armijo/tangent/qualification
checks, node/edge/common records, input/source bindings, journal/checkpoints and
inventory metadata. Launcher storage is separate. This is a conservative payload
screen, not permission to exceed Store's enforced limit or omit any failed data.

The unchanged historical timing heuristic in the design passes its60s screen.
No workload benchmark was added. Caps may stop the pilot before two edges; such
a stop remains unresolved. Source-only tests include the analytic one-coordinate
polynomial loss L=1.5x^2−.5x^4+x^6/6: a local simple period-two fold and a segment
that stays on one side, both native carry conventions, full witnesses, separate
plane/orientation references, actual native matrices and closure images, cache
accounting, zero-plane reverse, rank/finite/operation-cap failures, failed common
checks, out-of-box trial retention and failed durable edge rollback. No research
model or target file is imported by these tests.19 producer controls pass; together
with root's20 separately owned controls,39 pass in0.53s. Root retains its controls
and independent review separately. No further scientific conclusion is implied.

## Exact prospective command, only after separate authorization

Both exclusive paths must be absent. Record a fresh host-resource check and leave
foreign jobs untouched. Bind full execution and independent-review SHAs:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nice -n 10 \
/data2/neural-shrimp/.venv/bin/python tools/launch_s1r_pseudoarclength.py \
  --source-commit SOURCE_FULL_SHA --review-commit REVIEW_FULL_SHA \
  --out /data2/neural-shrimp-data/explore/s1r-pseudoarclength/attempt-001 \
  --launcher /data2/neural-shrimp-data/explore/s1r-pseudoarclength/attempt-001-launcher
```

The wrapper saves its source, exact argv, source inventory, resources, owned PIDs,
exit/time/RSS/log and complete remaining raw inventory. It signals only its own
process group, bounds inherited pipes, checks manifests and makes files read-only.
`two_edges_complete`, `unresolved_gap` and an explicitly labelled cooperative
`stopped` outcome are completed diagnostic outcomes; process/source/filesystem
failure remains separate. Scientific stop is not evidence for nonexistence.
No fold, historical-family association, lower-bootstrap match or utility claim is
emitted. The unit ends after this one attempt and independent saved-data audit;
no retry or method expansion is authorized by this card.
