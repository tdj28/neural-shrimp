# Overnight research report — 8 October 2026

Interim update at 13:28 UTC / 06:28 PDT. New work stops at 08:00 PDT / 15:00 UTC.
Private human reading draft; no submission or public release.

**The strongest advance is a proof that a 72-parameter neural network can settle
into an attracting two-update cycle.** Nearby weights converge locally toward
that repeating pattern under the exact-real training map. This extends the
rigorous examples beyond our 13-parameter network. It does not yet demonstrate
a useful intervention or a benefit for LLM training, inference or nanoGPT speed.

## What we learned

- **Proofs:** two 13D period-ten orbits and one 72D period-two orbit have
  existence, primitive-period and attraction certificates. The 72D proof and
  its independent audit pass; a full monodromy power has norm below 0.556.
  A separate scalar example has a local supercritical Neimark–Sacker proof.
  These are local results, not proofs of the entire window census or chaos.
- **Branch association:** twelve witnessed links connect the S1R lineage to
  a newly reproduced upper bootstrap at identical controls. The attempted
  lower connection still has a gap after 54 of 75 macro steps. Instability
  of the last accepted orbit does not itself terminate its branch.
- **New gap diagnosis:** the stored Newton direction solves its local equation
  accurately: relative residual about 3.14e-9. Nevertheless, all thirteen saved
  finite steps increase the original residual; even the smallest raises it
  about 34%. Independent audit passes. This points toward testing a different
  continuation method. A nearly singular derivative at the failed, nonclosed
  state is not a fold or disconnection result. Its tangent orientation also
  depends on the declared metric. No new correction was applied.
- **CIFAR:** the 98,666-parameter periodic-orbit candidate remains unresolved.
  Keeping 200 GMRES iterations in one cycle improves true linear forcing from
  0.8843 to 0.6620, but misses the 0.5 gate and all four nonlinear trials fail.
  The input-span reduction leaves 16,746 nontrivial coordinates, yet eliminated
  directions explain only 1.75e-8 of the original squared residual. Both
  diagnostics are independently audited. We stopped this correction sequence;
  no CIFAR job is running and no orbit or solver speedup is claimed.
- **Utility:** the proposed output-only pulse cannot match the chosen uniform
  pulse within the fixed cap in any of four saved cases. No pulse was applied.
  A reviewed algebra note also shows that matching both displacement from sham
  and total step length to a uniform backoff forces the same direction. A fair
  alternative comparison and distant periodic/chaos-qualified controls are
  needed before testing a geometry-specific benefit.
- **LLM:** both completed forecasting designs remain negative. In the broader
  data pilot, nonlinear features underperform ordinary comparators. Better
  development loss from broader data is not evidence for our geometry. The
  next proposal tests an explanatory mechanism first, without another LM sweep.

## Current paper

[Read the current 31-page paper](../paper/gd-route/gd-route-working-draft.pdf).
It has 292 checked source-derived values and nine figures. Compilation and
visual checks pass. The current paper is privately pushed and independently
reviewed at `2868e790`; its PDF SHA-256 is
`c9f9e2081f8ab95e152ce7cd1254e71a029b9f8e2e570990031cab46ac0e2e07`.

The accuracy pass distinguishes independently initialized S2 slices from an
established continuous branch, coefficient-sign candidates from proved
generalized bifurcations, and near-+1 digit-network boundaries from confirmed
folds. It includes the gap diagnosis and negative utility/CIFAR/LM evidence.
The shared-input limits of numerical audits remain explicit. Jensen1983 was
read in full; Arnold and Aronson checks remain partial. Research, bibliography
and human-authorship items remain open: this is a reading draft, not a release.

## Work in preparation

1. **Local continuation:** implement and independently review a two-edge
   pseudo-arclength pilot starting only from the last qualified S1R orbit.
   Direct, half-step and reverse witnesses must agree at common controls.
   The fixed metric, failure history and operation/time/storage caps are
   declared in `docs/plans/s1r-pseudoarclength-next.md`. Source and synthetic
   preparation is authorized; **no target is launched yet**. A prospective
   independent auditor is being prepared. If not ready by 14:30 UTC, leave it
   prepared rather than start it near the overnight cutoff.
2. **A possible bridge to larger models:** a fixed six-reference panel will
   test whether two changing sharp-curvature directions approximate the full
   complex dynamics better than matched-strength commuting controls. The
   selected phase derivatives are not retained, so acquisition needs its own
   reviewed source and measured budget. The design is under static review;
   no acquisition or fit has run. Elementary product algebra is not a novelty
   claim, and success would still not establish LLM utility.
3. **Practical value:** choose a meaningful intervention comparison and qualify
   its controls. Frozen-weight inference stops the optimizer map; prediction
   sensitivity and generation feedback require separate evidence.

## Preservation and resources

Authoritative worktree:
`/home/d7082791602/.codex/worktrees/post-edge-next-steps/neural-shrimp`, branch
`codex/post-edge-next-steps`. Coherent units are committed, privately pushed and
remote hashes checked. Main and Claude's pre-existing dirty log are untouched.
PR authentication is unavailable; no merge or external message occurred.

Raw evidence has additional verified copies on the separate physical archive
disk, supplementing the earlier 18GB snapshot. The latest increment contains
297 files / 7,067,996 bytes of gap, audit and reference evidence at
`/archive/neural-shrimp-backups/20261008T130606Z-gap-increment`.
Complete inventories, every file hash and a second source hash pass agree.
See [backup receipt](backups/2026-10-08-gap-increment.json). This is local
point-in-time protection, not offsite or continuous backup.

All completed experiment/audit processes exited. This unit used local CPU only;
no paid compute or consultation. Foreign workloads were left untouched and
protected holdouts remain sealed. Automatic approval review previously rejected
the private Astra payload transmission pending payload-specific human approval;
it has not been retried.
