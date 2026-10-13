# Independent proposal-budget review

2026-10-13, independent Codex cascades subagent. No actual seed arrays,
checkpoint, or model were loaded; no existing implementation was edited.

Reviewed source `proposal_budget.py`:
`63893ec0c91739b0cd5668e1257163110c1846a3ea0d40389dce92e246deb8c2`.
Reviewed tests:
`80b2205ac62283c257d40382a9485b7ecd3fd024c0d8d6540c4f8b341c4932c5`.
The referenced factored shooting gate is
`9c967c0dcb9783df53f33c05eb897050f4e8971f66767abbc8a01a8b1b85cf1f`;
the reused input loader is
`f24f9cdecec243abd25671a684aec03b8b105288526ff7fa4f60a963af745b02`.

**Verdict: no unresolved blocker for a separately planned bounded floating
budget calculation. This is not a proof-gate or model-run approval.**

The signed products have the correct order:
P_i=A_(i-1)...A_0, Q_j=A_(p-1)...A_(j+1), and
D=I-B(I-P_p). They are formed before induced-infinity norms. For the implicit
inverse block, the expression is
R_ij=-P_i B Q_j minus the forward segment when j<i. Its factor estimate is
||P_i|| ||B|| ||Q_j|| plus the product of the intervening A norms; the empty
segment has norm one. The P_i D defect appears only in column zero and thus
once in each complete row sum, not once per phase pair.

For equal phase radii and one common discrepancy epsilon, the resulting
planning expression is q_i approximately d_i+epsilon*c_i. The row threshold
(1-d_i)/c_i and strict minimum threshold are appropriate for budgeting q<1.
Epsilon means an **induced-infinity matrix discrepancy**, not a largest
individual entry error: individual error radii must be converted to full row
sums. Unequal phase radii require the missing radius ratios. Even a rigorously
positive threshold would leave the residual/inclusion, primitive-period, and
attraction obligations separate.

All arithmetic here is ordinary binary64, including the signed matrix
products and norm estimates. The source and reports correctly state that
these are not outward enclosures. `certificate_passed` and
`rigorous_enclosures` remain false. The tool does not evaluate derivatives or
claim that the unknown phase-box Jacobians meet its hypothetical epsilon.

Nonpositive or nonfinite threshold results retain all completed phase-pair
rows, all row thresholds and errors, and the full pair count before failure.
An earlier matrix/scalar arithmetic failure retains a clearly failed partial
report; it does not fill missing rows or claim completion. A zero coefficient
is conservatively rejected instead of inventing a finite positive threshold.
NaN, wrong shapes/dtypes, and overflow are refused.

Plan admission requires the exact CPU1/nice10/18GiB/120-second envelope,
explicit source hashes, and the existing full source-bound prefix/seed plan.
Actual loading delegates to the previously reviewed complete-receipt and
seven-array loader; no model constructor is called. The driver checks fresh
owned output, captured-plan location, and launcher source identity. The
external launcher remains responsible for the hard resource cap and private
pushed snapshot. No actual input admission was performed in this review.

Independent verification: **70 synthetic budget/shooting-gate tests passed
in 0.47 s**, CPU1, bytecode and pytest cache disabled. A separate exact
`Fraction` implementation additionally checked **51 phase-pair factors**,
propagated defects, row sums, and thresholds for noncommuting examples with
periods1,3,4,5 and dimensions2,3. Factors and signed defect norms agreed with
the exact dyadic values; threshold quotients agreed to expected binary64
rounding. The synthetic suite also exercises defective inverses, signed
nonnormal cancellation, complete failed-row retention, source/plan refusals,
mocked success/failure persistence, and isolated `-I` help.
