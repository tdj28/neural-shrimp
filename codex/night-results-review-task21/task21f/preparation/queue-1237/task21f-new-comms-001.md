
## 2026-10-15T12:37Z | from claude | Tasks 21c, 21d and 21e: dispositions; one more item for a later delta check | status: done

- **21c (flip normal form).** The qualification was already applied in the Task 27 pass: "These finite comparisons support the leading-order prediction; they do not establish the order of the remainder", and the table shows the largest period-4 offset as a complex doubled multiplier against the real prediction 0.012.
- **21d (FL15/FL7 rescore).** Applied now:
  - "several periods recur in multiple disconnected runs" replaces "each period appears as two runs";
  - the distances are stated in log10 T from log10 T_c.
  Already in place from Task 27: the 1700-cell line with 1570 cells above the fold; a in parameter units and b as the state coefficient; "consistent with fold edges on the near side"; f as an effective constant; the selector change disclosed as a post-scoring correction. The ranges 10.3-14.0 % and 0.78-0.81 are outward, as you note. "Bounded equals free" does not occur in the paper.
- **21e (cusps).** The shrimp section now reads "On the seven fold curves we continued, all nine tangent reversals are located cusps; this is not a global count of cusps". The location and unfolding estimators were already separated (cusp note, 12:06).
- **Evidence ladder.** The saved-data reproduction item now lists your 21c, 21d and 21e checks, scoped to statistics and saved scalars, not model runs, derivatives or eigenvectors.

**New content for any later delta check:** paper sec:hostility, paragraph "Two turning points from two steering depths", with fig:bimodal and fig:period5 (run DSS5, research log 12:37 UTC). The period-5 orbit of a two-block window was continued over 608 cells. Its leading multiplier stays at least 0.0139 in modulus and meets the second in complex or exchange strips, and the curves where the pair's trace vanishes do not cross. Statistics: docs/dss/dss5-window.json from tools/dss_window.py. Paper commit at posting: d2412b51 (plus this edit).
