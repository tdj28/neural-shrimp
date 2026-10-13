# Task 5 prospective fold-passage design review

12 October 2026 local date / 13 October UTC. Independent design review, before Task 5 model passages. This file records recommendations; the final pushed machine plan is authoritative. No model was loaded or run for this review, and no Task 5 producer trajectories were inspected. Read: current handoff and Task 5 assignment/correction; AGENTS.md, Skills.md, adaptive protocol; integrity playbook sections 1–5 and certified-computation playbook; Task 2 map, numerical, passage and audit implementation.

## Scope and the prospective boundary

A is a fresh point on the same period-2 orbit family, fixed log10(T)=-0.140, physical alpha near 0.0670. The earlier 0.0646 guess is superseded; the 0.0419 branch is not the assigned target. B is a fresh period-8 orbit fold at fixed alpha=0.0717, log10(T) near -0.1033. Both use the same model, prefix, position, steering vector and one-token map as Task 2. This is fresh parameter/orbit prediction, not new-prompt, new-model or ordinary-decoding generalization.

Allowed preparation: solve the fold and periodic state independently, classify its spectrum, compute derivatives, numerical sensitivity checks, and scalar predictions. Record starting-guess provenance and exactly which producer fields were consumed. Do not inspect producer passages or tune entrances/epsilons using full-model passage outcomes. A fold solver's map evaluations and declared derivative checks are setup, not covert passage pilots. Freeze both targets' eligible plans before the first controlled test trajectory; a setup failure remains a target-level outcome, never triggers a substitute target.

Before outcomes, privately push the human-readable protocol, source, machine trial table, coefficients/vectors/initial states or hash-bound immutable artifacts, prediction table, validity and failure rules, and independent audit. Runtime records the exact remote freeze SHA and every source/input hash. This project requires private pushes; no public registry or release is authorized. Call it prospectively frozen under the local adaptive protocol, not formally preregistered.

## Coordinates, derivatives, signs and clock

For target period p, define G=F^p. At the fold use M=D_x G, M r=r, l^T M=l^T, ||r||_2=1, l^T r=1. Compute raw nested-AD coefficients

- a_raw = l^T partial_parameter G;
- b_raw = (1/2) l^T D_x^2 G[r,r].

The parameter is physical alpha for A and log10(T) for B. B's derivative is not per unit T: d/dlog10(T)=(ln 10)T d/dT. A includes alpha's effects on the entire steered prefix and its KV cache. A detached cache loses this derivative. Hold x, r and l fixed while differentiating these coefficient formulas; do not differentiate the orbit solver or eigenvector algorithm.

A general positive-oriented convention is sigma=sign(b_raw), r_o=sigma*r, l_o=sigma*l, s_parameter=sign(a_raw*b_raw). Run parameter=parameter_c+s_parameter*epsilon. Then u=l_o^T(x-x_c) has quadratic model u_next=u+|a_raw|*epsilon+|b_raw|*u^2. Store raw coefficients and signs as well as oriented ones. Simply taking absolute values while always adding epsilon to the physical parameter is wrong when a_raw*b_raw<0.

One reported iterate is one G application: two F calls for A, eight for B. Also report F-step cost, but do not compare A and B counts on an undeclared clock. Fix the cycle phase by a deterministic rule before computing vectors and starts; different representatives have the same multipliers but different local coordinates and coefficients.

Existing-code hazards:

- `fold_normal_form` in Task 2 `numerics.py` returns finite-difference means in `a,b`. Its AD callbacks populate separate `exact_callback_a,b` fields. Task 5 must freeze AD values explicitly; FD tables are checks, not an undisclosed replacement.
- Task 2 `fixed_entrances` assumes a*b>0, and its passage driver always uses alpha_c+epsilon and period 2. It must not be reused unchanged for B.
- Compose derivatives in chronological order along every phase; inspect the derivative of G itself, not just one F evaluation.

## Clean-fold admission gates

Proposed concrete gates to settle before setup outcomes are used for admission:

1. Finite periodic state with relative closure ||G(x)-x||/max(1,||x||) <= 1e-11. Record absolute residual too. Proper-divisor returns must exceed 1e-6 on the same relative scale (A divisor 1; B divisors 1,2,4). Otherwise this is not the assigned primitive family.
2. Exactly one critical multiplier within 1e-7 of real +1, imaginary part <=1e-9, critical left/right residuals <=1e-8 in their declared norm. Retain the full computed spectrum or complete method of bounding/computing its complement. Another crossing type fails fold admission; never select a different nearby eigenvalue to force the type.
3. Freeze what “next modulus near one” means. The agreed admission gate is rho_complement <0.95, plus critical/noncritical separation. The initial <=0.9 review suggestion is superseded before outcomes; it may be reported as a sensitivity flag only and must not create a second admission rule. Retain the complex value, modulus and angle of the next multiplier. This numerical threshold is a design choice, not a theorem.
4. Nonzero a,b with signs stable to declared derivative checks. Use nested AD as primary and fixed central-difference step ladders as independent checks; report all values and the measured error trend. A proposed agreement target is relative 1e-4 for at least two consecutive admissible FD scales. Do not choose a plateau using later passages. If sign or numerical stability is unresolved, stop that target.
5. Tightened independent fold refinement should change the parameter by no more than 1e-4*epsilon_min=3e-10, and coefficients by the declared tolerance. This is numerical sensitivity, not a rigorous enclosure. Retain unsuccessful refinement histories, GMRES counts and actual linear residuals.

The exact thresholds in the final protocol can differ if fixed and justified before affected outcomes. No coefficient field may quietly average estimates from an unclassified or nonconverged point. Failures get explicit statuses such as `no_clean_fold`, `primitive_gate_failed`, `derivative_unresolved` or `resource_not_run`.

## Entrances and the meaning of transverse

For each target let U=0.1/|b_raw|; use the exact unrounded frozen value. Entrances are x_c-U*r_o (full), x_c (midpoint), and x_c-U*r_o+U*w (full plus transverse).

Generate a fixed-seed Gaussian z with a specified RNG/version. Form w0=z-r_o*(l_o^T z), then normalize w=w0/||w0||. Numerically check l_o^T w approximately zero. Thus the transverse kick is in the critical left eigenvector's kernel, preserving initial u=-U. Euclidean orthogonality to r alone is insufficient in a non-normal map.

“Same norm” means ||U*w||=U, equal to the axial displacement's norm. It does not mean the combined state perturbation has norm U; r_o and w need not be orthogonal. Record both component norms, their inner product and total displacement. One preselected seed per target gives one transverse direction, not an isotropic robustness result. The seed/vector is shared across all four epsilons for that target. Do not sweep random directions and select a favorable one. Seeds 29000–29999 remain forbidden.

These starts are fixed linear-coordinate entrances, not points projected onto a numerically fitted center manifold. Do not burn in, reproject, recenter at the perturbed parameter, or reset the passage clock after the transverse kick. Those changes would alter the tested prediction.

## Predictions, endpoint and complete reporting

Exactly four epsilons, in declared order: 3e-6, 1e-5, 3e-5, 1e-4. Two targets times four epsilons times three entrances yields 24 expected trial identities, including blocked/missing rows.

Let A=|a_raw| and B=|b_raw|. For each entrance freeze:

- Integer discrete prediction: iterate v_next=v+A*epsilon+B*v^2 from v0=-U or 0. N_quad is the first n>=1 for which v_n>U.
- Continuous approximation, unrounded: N_arc = [atan(U*sqrt(B/(A*epsilon)))-atan(v0*sqrt(B/(A*epsilon)))]/sqrt(A*B*epsilon).

The full-model count is the first n>=1 with l_o^T(G^n(x0)-x_c)>U, using a strict inequality. Save initial values and every pre-exit and first-exit coordinate. The crossing iterate counts; no interpolation, nearest-integer rounding or off-by-one convention change is permitted. A return below -U or a large transverse displacement is retained, not silently restarted. Log these as diagnostics unless a specific validity/termination gate was frozen.

Recommended primary endpoint: every eligible planned count satisfies |N_model-N_quad|<=1. Also give signed and absolute raw N_model-N_arc errors and a separate all-row within-one verdict for the arctangent approximation. If claiming that both predictors meet the one-iterate criterion, require both conjunctions. Do not choose whichever predictor looks better per row. A missing, censored, nonfinite or setup-blocked row cannot count as success. Report target-specific and overall denominators, maximum error and every row; successful A never cancels failed B.

Fix the iteration/resource cap before outcomes (it may be calculated mechanically from frozen scalar predictions, with a predeclared overall ceiling). No exit by the cap is right-censored, not a count equal to the cap. Very short predictions still remain rows; explain that one-iterate tolerance is weak relative accuracy there. This is a deterministic numerical falsification test, not 24 independent random samples or a population confidence interval.

Independent audit must reconstruct predictions and first crossings from retained traces, check both margins U-u_(N-1) and u_N-U, match all trial IDs and hashes, and verify that runtime started after remote freeze. A second implementation should compute scalar first-exit counts without importing the producer function. Any post-outcome correction preserves originals, receives a dated amendment, and downgrades affected rows as required.

## F8 and non-normality risks

Small noncritical multiplier moduli do not imply a small transverse one-step norm. Report ||l|| (with ||r||=1), ||Q|| for Q=I-r*l^T, and, where already available from the independently computed dense matrix, ||Q*M^k*Q|| for fixed k=1,2,4,8. These reveal finite-time amplification; do not use them to choose a favorable transverse vector. The same tangent and initial state precision must be checked on the eight-step composition. All arithmetic, norms and rotary tables must really be float64.

Record transverse distances ||(I-r_o*l_o^T)(x_n-x_c)|| through each first exit. The U-sized kick can produce a nonlinear center displacement even though its initial linear center coordinate is zero. A discrepancy is informative about the tested linear entrance/reduced model; it must not be erased by shrinking the kick or dropping a transient. Likewise, the straight full entrance can depart from the center manifold at O(U^2); the local normal form is asymptotic and is not a global accuracy guarantee.

The linear functional l can amplify state roundoff. Retain projection/crossing margins and show they exceed observed numerical discrepancies in any independent replay. Such checks support numerical validation/cross-checking, not certified bounds. All results remain rung 1 or 2, never a computer-assisted proof merely because the prospective test passes.


## Independent saved-result audit implementation

The separate `audit.py` imports neither the model nor the producer modules. It reconstructs all planned trial identities, AD-vector contractions, sign conventions, physical parameter coordinates, RNG-defined entrances, scalar predictions, every retained full-state projection and transverse norm, strict first crossings, bracketing states and crossing margins. It checks artifact/source hashes, plan and source bytes in the named Git freeze, and retained prospective timestamps. All 24 rows remain visible, including missing, censored and setup-blocked rows. Integrity and the two scientific all-row verdicts are separate; a missing/censored row cannot pass.

Synthetic tests cover both periods and physical coordinates, either ghost-side sign, strict exit versus equality, nonfinite/tail failures, source-artifact corruption, changed count/clock/prediction/parameter metadata, duplicated or missing trial IDs, missing groups, corrupted interior/crossing states, right-censoring and outcome-before-freeze or mismatching-Git-tree rejection. Git-tree calls are mocked in tests; no Git mutation, checkpoint/model load or Task 5 outcome inspection occurs in this review. This is an offline evidence audit, not an independent model replay or interval certificate.
