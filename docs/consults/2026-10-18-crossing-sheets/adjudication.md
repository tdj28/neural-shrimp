# Adjudication of consult 13 (crossing of the two window families), 18 October 2026

Review: `review.md` (gpt-6-astra, Pro, xhigh). The advice is advisory. Each point is judged against the repository and
the running data.

## Adopted

1. **Separate existence from selection, and test it at specific switches with microtransects.** Each microtransect
   has 17 points across one visible switch of the fixed-start map. It runs paired starts (x0 and one frozen
   non-special token embedding) and continuation of both families from both ends. Each switch is classified as
   stable-overlap selection, intrinsic branch change, mixed, or unresolved. This is the decisive experiment and comes
   first.
   - **Development transect:** the crescent switch at log10 T -0.148.
   - **Held-out transects B and S:** the nearest band/stripe switches below (0.0708, -0.1396) in T and along
     log10 T -0.1396 in alpha. They are selected from the fixed-start map alone, under a rule fixed before any seeded
     map is inspected.
   - **Endpoint:** the number of held-out switches, out of two, where x0 changes family inside a refined bracket while
     both periodic attractors persist and attract on both sides. Unresolved cases are reported as such.
2. **The proximity explanation is withdrawn.** x0's nearness to the crescent's blank-line phase describes its
   trajectory but does not explain capture: Task 28's nearest-phase classifier did worse than the majority baseline.
   No text will use proximity as the reason.
3. **"Sheets" is an illustration, not a mechanism.** The paper will say that there is no vertical order of
   attractors in the state space, and that the visible one is the one whose basin holds x0.
4. **A newborn sink need not have a globally tiny basin.** Near a fold the local node-saddle separation scales as a
   square root, but global returns can feed a large basin. No basin-volume or switch-offset law is claimed without a
   test.
5. **The main figure** is a branch diagram along each transect: all continued attractors with their stability, and
   separate fate ribbons for x0 and the alternative start, with settle times and censored runs marked. 3D renderings
   are optional.
6. **Labels.** The family is identified by full-state recurrence and continuation; token fingerprints serve as
   observables. Tolerances are calibrated on known cycles. Unresolved outcomes are kept, never discarded.
7. **Boundary organizer** (Codex Task 29). Measure the residence time near a candidate saddle and compare the escape
   slope with its independently computed unstable multiplier: tau(s) = -(p / log|Lambda_u|) log|s - s*| + C. Use
   individual brackets, not one global threshold.

## Deferred or declined

- **Chaotic transients against attracting chaos** (survival curves). Run only where the maps or transects show
  horizon-sensitive labels.
- **Testing the fold-offset law** (seven distances; pseudo-arclength continuation of the saddle). This is after
  submission, unless the crescent transect makes it cheap.
- **Fractal or Wada claims.** None will be made. Multiple switches along a segment are reported as multiple switches.
- **Compute.** The consult's GPU estimate for the microtransects (0.3-1 million evaluations) is high. One fixed-start
  trajectory costs about 1,800 map evaluations, and the paired transects need about 300 trajectories, so local CPUs
  suffice (about 15-30 minutes per transect). The running six-GPU maps are reused for selection and context, not
  repeated.

## Cost

One request: 29,153 input and 24,279 output tokens. The amount is in the private cost ledger.
