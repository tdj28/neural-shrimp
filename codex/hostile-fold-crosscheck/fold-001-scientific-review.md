# Task7 completed fold: independent scientific review — 2026-10-13

**Accepted at evidence rung 2:** the separately implemented float64 Qwen forward/AD computation reproduces the local fold of a primitive period-two orbit for the hostility-versus-kindness layer12 steering vector, fixed alpha=0.083, fixed37-token prefix/window1 and full896-dimensional state. The original attempt and numerical audit have completed and been reaped. Every saved gate passes; the independent full-spectrum/array audit reports no metadata disagreement. This is a numerical cross-check, not an interval proof.

The source commit is `6271ca3bb70f859a916abf2e17ba3c2a9ae748e9`, plan SHA256 `53fcb608cd4dcaf8aa3437fffe13c5fe38e3873e6a5d31a8071ea8b91bfa4cd7`. Original `fold-001/receipt.json` is `f562321e4170b396a5cd7b4e70a28df5843039a1dbfaa206dba2b96aa923a92a`. The owner authenticated all74 original artifacts and11 Git blobs and verified both original processes/groups absent. The independent saved-array audit is `fold-001-saved-audit-001/audit.json`, SHA256 `01582d1fa67d26060c59beb339eaabecbf828f02c689435d9e78b2cf97dbccfa`, using auditor source `63d68c5b4ff2e83e9b666c1e42a8b69847309f9bab67bc864ee8bc404efd5598`. No model was replayed in that audit or this interpretation.

## Numerical gates

The independently refined parameter is **log10T=-0.04624653567444522**. One full Newton correction reduced the augmented residual from4.56099e-9 to4.63078e-14. The refined critical multiplier is1.0000000000000457, real, with argument0. The next modulus is0.05001573791281647 and isolation from the other eigenvalues is0.9499842620872293. The full896-mode spectrum agrees between the two saved-array eigendecompositions to1.70e-17; the largest newly recomputed right/left eigenpair residuals are2.63e-15 and8.53e-15. The critical eigenvector residual is4.68e-14. The neutral fold multiplier is kept distinct from attraction.

Both phase returns close: relative F² residuals are7.22161e-15 and1.09824e-14. The one-step return distance is1.1562088753 relative, excluding period1 at the numerical tolerance. The state/matrix/parameter/gauge records agree with the solver and both saved phase returns. The canonical right mode has Euclidean norm1, and left-right pairing is1. The left/right cosine≈0.0479799202 records substantial non-normality; the composed derivative precision checks are therefore material.

For the critical and deterministic directions, dense-F²/JVP relative discrepancies are3.63249e-13 and4.26289e-13. Critical-direction central-difference errors at relative state steps3e-4,1e-4,3e-5 are1.12476e-4,1.24975e-5,1.12479e-6; the deterministic-direction errors are1.22654e-6,1.36240e-7,1.22468e-8. These scale consistently with h² and pass the frozen finest-error/convergence gates. The critical multiplier from the independent composed JVP differs by3.61489e-13. Every row, including the coarser rows above the finest-error tolerance, is retained; no coarse row is represented as passing that finer threshold.

Nested AD gives, in **p=log10T** units and `u=l·(x-x_c)`, `||r||₂=1`, `l·r=1`, orientation b>0:

- a=-1.1683279225482621;
- b=44.52540973728357;
- ab=-52.02027945897067;
- sqrt(|ab|)/pi=2.295812772895059.

Parameter finite-difference contractions at steps7e-6,2e-6,7e-7 are[-1.168328025075126,-1.1683279301972862,-1.1683279207586272], with relative AD discrepancies[8.78e-8,6.55e-9,1.53e-9]. State quadratic contractions at relative steps3e-4,1e-4,3e-5 are[44.5286766038013,44.52577276283858,44.52544241093541], with relative discrepancies[7.34e-5,8.15e-6,7.34e-7]. Both finest rows pass separately, without averaging opposing errors; all recorded coefficient signs agree with AD.

## Comparison with the producer result

The comparison uses only the completed producer JSON `/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hnormal/normal-form-p2/normal-form.json`, SHA256 `df5c263fc64f17c6e942b98ff3778d863963f4a1b45682769e8beaf3ed6444d1`. The original producer state was disclosed as a seed; its forward/Jacobian/eigenvector/coefficient code was not imported into this independent computation.

| Quantity | Independent minus producer | Relative difference |
|---|---:|---:|
| log10T fold location | +2.9989198624e-11 | +6.48464e-10 |
| a | -5.3363744534e-9 | -4.56753e-9 |
| b | -3.6964943018e-4 | -8.30192e-6 |
| ab | +4.3163414460e-4 | +8.29735e-6 using abs(producer ab) |
| sqrt(abs(ab))/pi | -9.5246424774e-6 | -4.14868e-6 |

The b discrepancy is retained: these are not identical estimates. The producer's b=44.52577938671375 is exactly the mean of its final two finite-difference values,44.52607523906492 at3e-4 and44.52548353436257 at1e-4. A post-hoc h² Richardson comparison using those saved values gives44.52540957127477, only1.66009e-7 (3.73e-9 relative) from the new nested-AD result. This reconciles the small discrepancy with leading finite-difference truncation; it is an estimate comparison under an h² assumption, not an exact cause proof or an additional model evaluation. The gauge-invariant coefficient product agrees to8.30ppm, and its passage-slope coefficient to4.15ppm.

## Allowed paper claim and scope

A suitable claim is: “An independent float64 forward and automatic-differentiation implementation reproduces a primitive period-two saddle-node of the hostility-steered Qwen soft-token map at alpha=0.083 and log10T≈-0.0462465356744. Its simple real+1 multiplier, strongly contracting complementary spectrum and nonzero unfolding/quadratic coefficients agree with the original computation.” The coefficient table should state log10T units and the gauge above; use the nested-AD values for this cross-check.

Since a<0 and b>0, the ghost side is p-p_c<0, and local motion crosses from negative to positive u. The local asymptotic inverse-passage coefficient is2.29581277 for full passages; one return-map iterate means **two** forward F steps. This attempt measured neither controlled passage trajectories nor natural reinjection in the period16–20 windows. It verifies the local fold and coefficients numerically, not the global window organization, behavioral utility, or an interval-certified attracting cycle. A separately frozen controlled-passage test can be considered prospectively.

The full attempt used CPU2/nice10/12GiB admission and completed in295.070652s with peak6,455,968KiB RSS. The separate CPU1 saved-array audit took1.268924s with190,940KiB RSS. Both are complete, all failures/iterations were retained, and no model or passage run was added during this review. The compact full gate/comparison report is `fold-001-scientific-review.json`, SHA256 `c240e4c632bb8e462822b4243abf56e70bebdca6d283f24a7f9be44600954c04`.
