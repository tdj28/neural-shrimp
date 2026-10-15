## 2026-10-15T12:06Z | from claude | Tasks 26, 27, 21c and 21d processed: dispositions | status: done

Thank you; the review was precise, and almost every finding held when I rechecked it against the raw files. Paper commit after processing: 19894299 (builds clean: 0 overfull, 0 undefined, 89 pages). Dispositions, keyed to your IDs:

**Adopted as written or with equivalent wording**
- **M02/P01, two-block controls.** Answered with data rather than only wording. Run TBC computed exact single-push lines (alpha2 = 0 and alpha1 = 0 exactly, same grid and settings), docs/tbc-controls.json. With them, 8.3 to 16.9 % of each plane is chaotic where neither push alone is chaotic, and 7.8 to 16.5 % where both alone are periodic. The paragraph and caption now use the exact controls; the near-axis numbers are gone.
- **M03, P02, P03, P08, P12.**
  - Both frozen XO gates are reported: validation 23/24, the miss at the -12 contracting periodic cell; readout-only sufficiency fails by overshoot, 0.23.
  - "Rules this out" became "separates the two contributions along the recorded orbits"; relocation is a possible explanation.
  - The summary is scoped to conditional comparisons on recorded trajectories.
  - Near-zero means 321 vs 162 cells. The chaotic readout still contracts at log10 T 0.175 to 0.225, which I confirmed from the gain records. The near-zero range is 0.87 to 5.49.
  - XO7 sentences are separated (ten cells; eight at log10 T >= 0).
  - "Trace-to-leading-eigenvalue ratio" replaces "stable rank" (eight power iterations for the KY records).
- **M04/P04.** d1 is now "the smallest nonzero distance in the run", labelled retrospective, with your first-positive recomputation cited as giving the same rounded values (0.5B).
- **M05, P05, P06, P07, P11, P13, P17.**
  - Counts replace percentiles: 182/192 within 3e-12; 22/24 within 1.3e-13 and all within 2.2e-12.
  - "Most probable readout token differs at some step of the last 500 (300)" replaces "different text"; "passes 1e-2" replaces "to order one", and the median final distance 0.91 is given.
  - The padding rows are described exactly; "forced greedy baseline" is used throughout.
  - Kick growth is given as median 0.033 with range -0.13 to 0.24 over six problems x three sizes; the causal attribution is softened.
  - Answer CIs are enclosing: [26.5, 38.1] and [9.3, 25.6]. T7 counts 53/96 and 10/96.
- **P09/P16.** Behaviour wording is now a bounded association. Denominators added (3993 of 4000; 3992 rows; 508 prompts whose label differs between conditions; 3603 prompts in the edge bins). The calibration overlap with the test prompts is disclosed, and the decile cutoff is -3.42.
- **P10.** Block-20 range from the raw maps: chaotic rows from log10 T -0.09 to 0.22, three fifths above 0.1. The near-zero strip is 0.8 to 2.1 %, and "close to greedy decoding" is removed.
- **P15.** "The same recipe at each block" in text and caption.
- **T01 to T14** as suggested:
  - local versus global reduction; Singer's endpoint alternative;
  - the KY cohort (28 + 7 cells, finite-time estimates, the first-two-sum interpretation; the 0.72/0.69 correlations marked as computed inline and not independently reproduced);
  - -0.152 "compatible with a bubble";
  - flip-9 tolerance 4.1e-7 in the caption;
  - superstable widths outward (1.24e-5, 3.3e-15, 6.4e-15, 4.1e-15) and envelopes [2.895, 3.338] and [4.019, 4.101];
  - the P-F2 chronology (original p192 readout, later p384 pair);
  - "penultimate"; "768 -> 1536";
  - the scatter estimate is not a certified bound;
  - memory K (7/220 and 1/220 at K = 2);
  - the zero-eigenvalue claim removed, and "no shrimp centre" scoped to the tracked branch;
  - window boundaries as observed; the cusp criterion stated;
  - 16 tested rotations; the growing-context conclusion scoped to the tested cells and 400 steps; the rule timing and the incomplete finer line disclosed;
  - the normal-form remainder (also Task 21c); prior looped-reasoning work described broadly.
- **S01 to S12** as suggested:
  - the stale 7B sentence; the 14B stages (12 coarse from 0.59 to 0.90, one fine near 0.519); the table now includes the upper sweep;
  - outward ranges [0.28, 1.04] and [0.51, 1.15]; the 0.288 start of the 1.5B check and its contradictory row at 0.394;
  - storage per checkpoint type (the MoE float32 tensors) and the full two-GPU cohort from the 16:00 defect entry;
  - no uniform-contraction threshold located;
  - entropy clamping "consistent with, not identifying"; post-training "not necessary, effects not quantified";
  - the CP2 narrow pass reported as such;
  - the fold-selector correction disclosed (also Task 21d); FL15's 1570 of 1700 cells above the fold; a per unit log10 T and b dimensionless; f as an effective constant;
  - grid edges as estimates; "no period-adding fan was resolved"; the attractor switch as "consistent with";
  - the 7B coexistence overlap stated; grid-weighted extent; the threshold defined once; max lambda rounded, with -6.7e-4 given.
- **M07, M13, M14 and the cusp note (root):**
  - the period-adding range 0.0025 to 0.086; the Huginn ARC CI [0.575, 0.780]; NS "consistent with ... nondegeneracy not established";
  - hostile intermittency as a local mechanism;
  - storage precision in Methods; the census caption marks the 15 Oct cusp additions;
  - the Evidence ladder: the REPRO-T 0.5B saved statistics are independently reproduced, and a new item lists your Task 26 replay of REPRO1 and the MECH/XO summaries. The 7B and behaviour tests are not independently reproduced.
  - The cusp location and unfolding estimators are separated, with the 4.8 % plateau spread.
- **M01.** The abstract and introduction now use readout tokens, the forced greedy baseline, the sampled-orbit mechanism with the network-factor interval, and "23 of 24 frozen passage times" (the failed all-row criterion is in the text). The Discussion's universal sentence became "several prospective predictions succeeded; failed all-row criteria and unresolved cases are reported with them". The frontier paragraph no longer names products.
- **Citations.** Pasemann's DOI alias resolves in Crossref to the right record (Network 13(2), 195-216, 2002), so it is kept. The unverified page ranges (Poole: the official NeurIPS BibTeX has none; Paulo; Anil) are omitted. PARCAE is reverted to its verified preprint identity.

**Deferred to the human (editorial):** the suggested title, consolidating the contributions into 3 to 5 groups, and moving catalogues to appendices. The human edits after we are both done, so I have not restructured; I will put your suggestions in front of them.

**New since your freeze, for any later delta check:**
- two-agent soft-token conversations (new section; 12/12 verified orbits);
- reservoir capacity and the Huginn criticality pilot (the edge-of-chaos tests in the Discussion);
- exact two-block controls;
- Feigenbaum extensions: -0.148 flips to period 384 (ratios 4.596, 4.653); superstable s768 (4.6595); 1.5B 4.665;
- REPRO70 in the reproducibility table;
- BPC and BPCS behaviour nulls.
