# Novelty check for Track A claims, 6 October 2026

Delegated literature check using generic queries only (titles, authors and
technical phrases; no private project text). The eight closest works were read
in full on arXiv; other items are marked by verification status. This is a
starting map for positioning, not an exhaustive certification; every citation
must be rechecked at its primary source and exact version before use.

## Verdicts

| Claim | Verdict | Positioning |
|---|---|---|
| C1. Verified attracting cycles of the full gradient-descent map, organized as shrimp domains with continued fold/flip curves, in the per-layer learning-rate plane | **Novel** | Do not claim the first period-colored training map (Sveleba et al. 2026 color Adam beta1-beta2 planes by detected period, without full-state verification or shrimps) or the first bifurcation continuation for gradient descent (Hofmann 2026 continues flip and catapult thresholds of one-dimensional reductions). Periodic windows in one-dimensional learning-rate sweeps date back at least to van der Maas, Verschure and Molenaar (1990) |
| C2. Window organization explains the fractal trainability boundary | **Novel, under-evidenced; competing mechanisms** | Roughness (Liu 2024), covering-map folding in initialization space (Liang and Montufar 2026), a boundary crisis in one-dimensional reductions (Hofmann 2026). Windows sit inside the bounded set; the boundary itself plausibly comes from crises |
| C3. Reproducible islands inside chaotic large-rate training | **Partly anticipated** | Ly and Gong (2025) establish learning-rate-dependent riddling and a performance/reproducibility trade-off. Reproducibility inside attracting windows is nearly definitional; value lies in quantification and caveats (coexisting attractors, transient chaos near edges, riddling) |
| C4. Averaging over periodic orbits recovers low loss | **Largely anticipated** | Chen et al. (2024) find trajectory averaging helps in periodic and chaotic phases; central flows (Cohen, Damian et al. 2025) explain averaged iterates. Not a headline claim |

## Closest works and what they lack (summary)

- **Hofmann, arXiv:2607.04993v1.** Exact theory for single-step-size gradient
  descent in solvable models; one-dimensional cubic and Ricker-type maps with
  Feigenbaum cascades and periodic windows; boundary crisis; analytic
  continuation of thresholds in a depth-learning-rate plane for the reductions.
  No two-dimensional period maps, per-layer rates, Newton refinement on
  nonlinear networks, fractal-boundary link or reproducibility analysis.
- **Liang and Montufar, ICLR 2026, arXiv:2509.25351v3.** Critical step size for
  scalar factorization; fractal convergence region in initialization space;
  existence of periodic orbits of all periods (unstable, via Li-Yorke). No
  hyperparameter-plane windows or verified attracting cycles.
- **Liu, arXiv:2406.13971v1.** Fractal bounded/divergent boundaries in
  one-dimensional rate scans of perturbed quadratics, controlled by roughness.
- **Torkamandi, arXiv:2501.04286v2.** Two-dimensional convergence heatmaps for
  a small transformer under Adam; boundary dimension near 1.97; no dynamics.
- **Ly and Gong, arXiv:2510.05606v2.** Riddled basins and near-zero
  uncertainty exponents in parameter and function space; riddling varies with
  learning rate.
- **Chen, Balasubramanian, Ghosal, Agrawalla, TMLR 2024, arXiv:2310.01687v1.**
  Five-phase cubic-map reduction; trajectory averaging; best test accuracy in
  periodic and chaotic phases.
- **Jimenez-Gonzalez, Soriano, Lacasa, Phys. Rev. Research 8, L022032 (2026).**
  Training time minimized near the onset of positive maximal Lyapunov exponent;
  toy cubic map with self-affine windows.
- **Danovski, Soriano, Lacasa, Front. Complex Syst. 2024.** Intermittency and
  finite-time Lyapunov exponents at three learning rates.
- **Sveleba et al., Electron. Inf. Technol. 34 (2026), doi:10.30970/eli.34.2.**
  Period-colored maps over Adam beta1-beta2 from a two-output projection; no
  full-state verification, stability analysis or shrimps.

## Experiments reviewers are likely to demand

- Verification coverage and residual statistics across maps; Lyapunov-spectrum
  chaos diagnostics; distant ordinary-periodic and chaotic controls.
- Robustness to float32 versus float64, horizon, seed, width, activation, data
  and minibatch noise; at least one larger network and real data.
- Feigenbaum ratios along flip sequences; window size and measure
  distributions; the MSS ordering test.
- Crisis-curve continuation, and whether the trainability boundary's fractal
  dimension changes when windows are masked.
- Uncertainty exponents and function-space disagreement inside versus outside
  windows (comparison with Ly and Gong).
- What shrimp geometry adds beyond cycle detection.

Further related items flagged by the check (verification status as reported):
Kalra, He, Barkeshli (ICLR 2025); Ghosh et al. (ICLR 2025); Meng et al.
(arXiv:2406.05033v3, coexisting cycles and fixed points); Gan
(arXiv:2606.15551v1, flip-bifurcation theory at the edge-of-stability
threshold); Chen and Bruna (ICML 2023); Nar and Sastry (NeurIPS 2018); Song
and Yun (NeurIPS 2023); Altintas, Kwok, Raffel, Rolnick (ICML 2025); Summers
and Dinneen (ICML 2021); Cohen, Damian et al., Central Flows (ICLR 2025,
arXiv:2410.24206v2); van der Maas, Verschure, Molenaar, Neural Networks 3:119
(1990), doi:10.1016/0893-6080(90)90050-U; Lai and Grebogi, PRL 83:2926 (1999).
