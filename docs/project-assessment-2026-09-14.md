# Research assessment, 14 September 2026

Updated through EXP-011, executed on 15 September UTC. The project has a
verified small-system dynamical result and a useful negative/conditional
control result. It does not yet show that shrimp geometry improves modern
neural-network training or saves end-to-end compute.

## What is established

| Work | Completed evidence | Limit |
|---|---|---|
| EXP-001–003: initial instruments and training pilots | Cycle/stability tools and exploratory training/control pilots are implemented and preserved. The training map found no verified cycles and the recurrence-control pilot had no triggers. | No demonstrated trained-network or recurrence-control benefit. |
| EXP-004–005: fixed-weight recurrent network | A known three-neuron tanh shrimp-shaped window is numerically reproduced, with full-state primitive cycles and stability checks at selected locations. | A fixed-weight small system; complete boundary organization and a modern trained-network analogue are unfinished. |
| EXP-006–008: geometry and control | Geometry helps some observed prediction/error means, but prospective pooled action-utility evidence is unresolved. In EXP-008 direct minimum-preview selection is better than the geometry learner with the same eighteen previews. | Prediction improvement is distinct from action utility and from extra information conditional on coordinates. |
| Server arithmetic and regeneration audits | Every original aggregate score and selected action survives complete saved-trajectory input regeneration. Directional cancellation and ridge conditioning are diagnosed. | Original strict feature/coefficient failures are retained; no general portability certification. |
| EXP-009: longer-target refinement | All 2,776 traces complete; every one of the 220 original selected actions meets the declared native scalar rule. | 953/1,369 selected cases remain scalar-unresolved; 952 exceed the finer-step difference threshold. The selected cohort is not representative. |
| EXP-010: preview qualification | All 6,944 traces complete; all 3,456 75-unit preview labels pass native scalar/waveform checks, sixteen independent sentinels agree, and all 192 rankings stay unchanged. | Agreement at tested resolutions, not a proof of exact numerical error or acceptable task loss. |
| EXP-011: four-preview feasibility | All 24 origin folds and 96 models complete and pass numerical checks/replay. Signed axes have lower pooled error than either learner; geometry improves the shared-feature mean with uncertainty spanning zero. | Retrospective development on 24 known origins; no fresh-origin confirmation, long-target utility or net speedup. |

The four-preview result is especially informative. Mean preview RMSE is
0.00617504 for the full-eighteen minimum-error reference, 0.00856899 for four signed axes,
0.0107899 for geometry's four, and 0.0162661 for the shared-feature learner's
four. Geometry's observed mean improvement over shared features is 33.67%,
but it does not beat the inexpensive signed-axis control in the pooled mean.
The paired geometry increment is 0.00547615 with descriptive bootstrap bounds
[-0.00265276, 0.01561846]. All resolution comparisons preserve the same lists
and actions. Performance varies by radius, so a pooled ranking should not be
treated as universal dominance.

## What remains, in priority order

1. **Specify useful quality and measure complete cost.** Decide the intended
   waveform endpoint and acceptable degradation before fresh outcomes. Measure
   the reference/sensitivity/atlas and preview costs instead of equating four
   calls with a measured speedup. Keep simple controls prominent.
2. **Freeze a fresh-origin evaluation or a clearly exploratory tradeoff study.**
   Keep all origins disjoint from the known cohort, preserve numerical and
   missingness rules, and compare against the full-eighteen reference. Any
   radius-dependent or redesigned learned policy now requires a new freeze.
3. **Resolve the longer-target numerical question if pursuing 300-unit
   control.** Stable old selected actions do not qualify every candidate as
   a learning target or every exhaustive oracle. A changed endpoint would be
   a new scientific design, not a repair to old evidence.
4. **Finish the broader scientific extensions separately.** Boundary
   organization, trained-network cycles, a topology-specific mechanism and
   dynamical-memory hypotheses each require their own evidence. Do not infer
   them from an attractive map or loss oscillations.
5. **Finish repository review and data resilience.** The original strict
   EXP-008 gate remains failed. Review PR creation requires GitHub API access;
   private SSH pushes already work. External raw data need an independently
   verified backup if loss of this server must be covered.

## Server and preservation changes completed

[Skills.md](../Skills.md) records the shared i9-14900K, approximately 91.75 GiB
RAM and RTX 2000 Ada with approximately 16 GiB VRAM. It authorizes suitable
local work, requires checking shared capacity and limits, and retains RunPod
as a costed fallback. This research used four CPU workers and one numerical
thread per worker; no GPU port or paid compute was necessary.

The private checkpoint cadence is after coherent work units, about every
30 minutes during sustained work, and before long runs, handoffs or stopping.
Each push is checked against the remote SHA. New bulky raw evidence is under
`/data2/neural-shrimp-data/`; Git keeps code, plans and compact hash-bound
results, with accidental local raw directories ignored. A sibling directory
is not an independent backup, and existing historical evidence stays intact.

The complete suite passed 385 tests before EXP-011 execution. Exact execution
and independent-audit receipts are in the [verification ledger](verification-ledger.md)
and [EXP-011 audit](EXP-011-results-audit.md). The
[next-step decision](next-steps.md) preserves prior decisions and identifies
what the completed results actually justify.
