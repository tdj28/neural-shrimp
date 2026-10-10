# Paired comparator result and first-order limitation — 10 October2026

All four saved forks admit the fixed paired contrast under the unchanged25%blockcap, with both pulse and update norms matched within2e−13relative tolerance. No model, gradient, pulse or trajectory was acquired. The angular differences between actual update directions are only0.284–0.535degrees. The contrast vector has normalized gradient alignment−0.975to−0.986; its predicted first-order loss advantage is already strong relative to its direction. Feasibility does not make this a clean test of shrimp geometry. Full outcomes and inputs are in utility-paired-feasibility.json.

There is a stronger algebraic obstruction in the original two-rate plane. Let s_h=eta_h*g_h and s_o=eta_o*g_o, with positive normsH,O. A two-block contrast v=a*s_h+b*s_o satisfying equal-norm pairing has

    s·v = a H² + b O² = 0.

To also remove the instantaneous first-order loss contrast requires

    g·v = a H²/eta_h + b O²/eta_o = 0.

If eta_h differs from eta_o and both blocks are nonzero, these equations force a=b=0. Thus no distinct two-rate paired contrast is simultaneously orthogonal to the actual GD step and the raw gradient. This is elementary linear algebra, not a novelty claim. A nonzero test must either model/subtract the first-order prediction, change the intervention family, or work where rates coincide; none supplies missing distant controls automatically.

Next concrete method direction: measure directional response conditional on its known first-order gradient prediction, with sham and the equal-update uniform comparator kept distinct, and qualify distant ordinary-periodic and explicitly diagnosed chaotic controls before any geometry-specific outcome. Do not extend the old impossible matched-uniform design or interpret paired asymmetry as a geometry effect. Existing all-four failures of the original5%pulse and zero-qualified-distant-control result remain part of the record.
