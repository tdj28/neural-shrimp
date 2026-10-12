# Stable-branch coefficient reconstruction

The published-in-draft estimate `ab = 123.1` is reproducible from the stated
18 saved branch points: quadratic extrapolation in `sqrt(alpha_c - alpha)`
gives **123.0831902574**. A separate 60-digit Decimal calculation with a different
linear solver agrees within 3e-12.

This is an independent arithmetic reconstruction of saved producer multipliers.
It does not independently re-evaluate these 18 model orbits or their eigenvalues.
The direct independent derivative calculation is recorded separately.

| Fit | Extrapolated ab | Difference from producer derivative ab=122.9272007377 | RMS fit residual |
|---|---:|---:|---:|
| Quadratic | 123.0831902574 | +0.1559895198 (+0.1269%) | 0.00390834 |
| Linear | 120.5718416508 | -2.3553590869 (-1.9161%) | 0.20226040 |

The difference between fits is a sensitivity diagnostic, not a statistical
confidence interval. The quadratic fit reproduces the stated rounded value;
its close agreement does not give a bound on extrapolation error.

The fixed range `5e-5 <= alpha_c-alpha <= 4e-4` selects original indices 0–17.
All selected points have a positive real attracting dominant multiplier and
consistent convergence/spectral metadata. Our first attempt applied a stricter
closure screen of 1e-11 and stopped before fitting: row 3 has saved closure
3.7552970175e-11. The producer's archived `args.tol` was 1e-10. The disclosed
second attempt uses that original criterion for the historical reconstruction,
retaining all 18 points. Independent orbit/fold thresholds were not changed.

- Failed stricter screen: `receipts/branch-001.json`.
- Historical reconstruction: `receipts/branch-002.json`.
- Independent arithmetic audit: `audit_branch.py`, `audits/branch-002.json`.
- Raw attempts: `/data2/neural-shrimp-data/codex/task2-independent-crosscheck/`.
