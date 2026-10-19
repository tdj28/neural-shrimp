# Adjudication of consult 21 (mathematics and references of "Why the loop can be chaotic")

Review: `review.md` (gpt-6-astra, Pro, high effort; completed 19 Oct 2026 21:12 UTC; artifacts in
neural-shrimp-data/consults/2026-10-19-theory-math-refs/execute-001). The reviewer saw only Sec. III, the factorization
paragraph of Sec. II and three bibliography entries; findings were checked against the full article, the report, the
tools and the cited sources before acting. Advice is advisory.

| Finding | Decision | What changed or why not |
|---|---|---|
| Verdict: factorization, high-temperature contraction and the tied network-free energy argument sound | noted | no change |
| B01 block bounds as finite-record diagnostics | accepted | "the bound tightens as k grows" was too strong: now "cannot increase when the block length doubles" (nested blocks, exact norms); blocks named as disjoint and tiling the record; exponent bound stated in the spectral norm with upper limits where averages do not converge; Kingman's theorem now with its assumptions (orbits typical of an ergodic invariant measure, log-norm integrable). The 1/k rate was already stated as sample-specific. Article and report. |
| B02 chaos needs an operational definition | accepted (pointer) | The label is defined in Sec. IV ("Scans and chaos labels": exponent above 1e-3 and no period found); Sec. III now points to it. Fig. 6(b) caption now gives both parts of the mechanism grids' rule (both twin exponents above 1e-3 and no period up to 64, tools/steer_gain.py), which it had stated only in part. Energy argument: "cycles of period above one". |
| B03 tied/untied scope at low temperature | accepted | C is now the norm of the sum of (e_i - e_j)(w_i - w_j)^T with w the output vectors (equal to e when tied); convergence "pointwise and in general not uniform". The margin condition "bounded away from zero" and the finiteness of L(alpha) on the compact K were already stated. |
| S01 Fisher pullback lacks 1/T^2 and is tied-only | accepted | Now "for tied embeddings, ... divided by T^2, the same covariance is the Fisher information ...; for untied embeddings the covariance of the output vectors takes its place". Article and report. |
| R02 delete the Fisher digression and its two citations | declined | The sentence positions the factorization against verified prior work (Jacobian Scopes, FishBack; checked on arXiv earlier the same day, entries in the bibliography) that the human asked to find; corrected per S01 instead. |
| S03 contraction certificate is not the chaos boundary | accepted | "rigorous but far from sharp" became "rigorous, but it does not locate the measured chaos boundary"; Banach's fixed-point theorem named. The numerical T* values were already stated as lower estimates (power iteration). |
| R01 verify and narrow Wang (2026) | accepted | Checked on arXiv (2605.15651, T. Wang, 15 May 2026): affine logit systems with softmax feedback, threshold beta times the norm of the projected matrix below 2 for uniqueness and global stability of the fixed point, sharp. Sentence now: "a recent preprint proves a sharp spectral threshold for the uniqueness and global stability of the fixed point". |
| Gao and Pavel (2017) only supports a displayed identity | declined | Kept as the attribution of the softmax Jacobian, a standard identity; no claim depends on it. |
| B04 intervention grids not matched | accepted in part | The text now says the two grids share their temperature rows but not a matched strength of intervention, so the shares are counts per grid, not a comparison of doses; crossover products described as tangent products assembled along recorded states, not the dynamics of a rerun hybrid model. The full factor-by-factor design table belongs to the report's methods, not the article. |
| S02 estimands of the intervals | accepted | Accounting interval: bootstrap over the 18 temperature rows (tools/plot_mechanism.py); crossover intervals: bootstrap over the 14 held-out cells; the 7B readout effect +0.23 [-0.11, 0.63] described as an interval that includes zero. |
| F01 caption blurs directional gains and operator norms | accepted | Fig. 6(a) now plots "the relative gains g_read and g_net of the text (gains along the tangent direction, not operator norms)". |
| Reference table: retain Kingman, Ramsauer | noted | retained |

Builds after the edits: article 27 pages, report 175 pages, no overfull boxes or undefined references.
