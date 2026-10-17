# Task28 exploratory basin study

Human resumed unfinished queue 17 October. Root owns this round; no active subagent or other numerical job. Task28 explicitly requests tools/, docs/basins/ and an append-only research-log entry, superseding prior audit-only path scope within the assigned Codex worktree. No producer checkout edits.

Use the previously reviewed independent ManualQwen float64 CPU map, exact pinned philosophy vector/prefix/full vocabulary. Manual norms and rotary are already fully float64; this avoids HF patch/dispatched-hook ambiguity. Profile checks central difference vs AD before target. Fixed initial prefix ID198 corresponds to newline. Reuse prior map authority, but check exact pins.

Cells alpha .0642/.0652 at logT -.156 plus .0644 selected from cross512: temperature row159, alpha column65 nearest (.06440802348336595,-.15599804305283757) is period3. Axes are temperature then alpha; transposing would give an incorrect period2. Historical CUDA map labels are context, new CPU classifications remain measured. Donor alpha .0655.

Each cell:128 independent uniform vocabulary token embeddings (duplicates/special tokens retained),128 independent Dirichlet(1) convex mixtures of8 uniformly sampled distinct IDs from the fixed-start top64 probabilities,65 deterministic segment points, fixed start and donor control =323 starts. Wilson95 intervals are marginal and sampling-distribution-specific for each128 random arm only. No global embedding-space basin-volume claim.

Donor1000 steps, reference1200. Actual trajectory1200 initially proposed; check all final64 full-state relative returns for each lag1..32 (<1e-8); compare midpoint and final classifications, keep every unresolved or changed row. Period2/3 membership also requires phase distance<1e-6. Leading tokens are descriptions, not state equality. No-return is finite irregularity, not a chaos proof. Stored tails129 states.

Segment runs from saved last donor reference state(u0) to saved last fixed-start reference state(u1); for chaotic cell the latter is a finite reference point, not a period3 state. Enumerate every coarse label change. Refine all donor/complement changes up to16 brackets for12 levels, retaining every query and unresolved class. Finite bisection neither proves a unique crossing nor fractality. If >16, retain coarse grid and report cap, no arbitrary subset.

Compare fixed start and16 iterates with all periodic phase states or the129-point irregular reference tail. Euclidean proximity is descriptive, not a sufficient causal explanation. Report transient disagreement/uncertainty rather than replace criteria.

Profile is 4 steps each at batch2/16/323,2CPU+supervisor,12GiBAS,300wall/540CPU,128MiB; shared RAM28GiB startup/16GiB runtime,16GiBdisk. No model downloads, GPU, paid compute. Commit/push before profile. Use profile to size actual bounded run and disclose any horizon/batch amendment before execution. Preserve all failures.
