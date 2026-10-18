# Run F7: a held-out fold family in the unsteered Qwen2.5-7B, with passage times predicted before they are measured

Frozen 18 Oct 2026, before any F7 data; requested by the external review (a detailed bifurcation family outside the
small Qwen models) and approved by the human ("Do all of them").

**System.** Qwen2.5-7B-Instruct (revision a09a3545, untied embeddings), the paper's prefix, unsteered (alpha = 0; the
steering vector file only fixes the width), float64, the soft-token map with temperature as the parameter.

**Candidates** (from the existing wide sweep big/qwen7b-alpha0-wide, 160 temperatures):
- a period-3 window at log10 T = 0.2176 (exponent -2.74) between chaotic cells at 0.2075 and 0.2277;
- a period-10 window from -0.044 to 0.016 that ends in chaos at 0.026.

**Stage A (locate; job set eslseq).** tline continuation of each window's orbit in log10 T from inside the window,
both directions for the period-3 window and towards higher T for the period-10 window (ds 5e-4 to 2e-3). At each
turn (fold), run normal-form --nearest-fold --param temperature: the fold location T_c and the coefficients a and b
of u' = u + a eps + b u^2, with eps in units of log10 T. The normal-form tool refuses to give coefficients if the
multiplier nearest +1 is not real and within 0.01 of 1.

**Stage B (predict; local, no model).** For each located fold with ab > 0, choose six eps values before any passage is
measured, giving predicted passages of about 6 to 60 G-iterates; G is the p-fold map. For each eps, three
entrances: full (u_in = -U), midpoint (u_in = 0) and full + transverse, with U = 0.1 / b. The predicted passage
counts come from the discrete quadratic normal form (bifurcation_curves.passage_count). The prediction table is
committed and pushed before stage C.

**Stage C (measure; passage tool, temperature mode).** On the soft map at log10 T = T_c + eps, the same entrances,
counting G-iterates until u exceeds U.

**Criteria** (the paper's earlier folds).
- Primary: every forecast within one G-iterate of the measured count (the all-row criterion).
- Secondary: at least 75 % of the forecasts within one iterate.
- Also reported: the relative errors, and the arctangent formula.

Failures are reported as failures. A fold not found, or a coefficient refused, is reported and not replaced by
another window after the fact; the second candidate was named here in advance.
