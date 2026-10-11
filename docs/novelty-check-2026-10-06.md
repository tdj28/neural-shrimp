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

## Addendum, 7 October 2026: quasi-periodicity and phase locking in optimizer maps

A second delegated check (generic queries only) asked whether quasi-periodic
training, Neimark-Sacker bifurcations, Arnold tongues, devil's staircases or
torus-breakdown routes had been reported for optimizers acting on losses.

| Item | Verdict |
|---|---|
| Neimark-Sacker of the edge-of-stability 2-cycle | Partly anticipated: Hofmann (arXiv:2607.04993, Prop. 16, "rotational edge") derives complex unstable multipliers of an aligned 2-cycle in a two-mode depth-2 linear network and a persistently misaligned oscillatory state; no invariant-circle check or rotation number. Bock and Weiss (ICANN 2019) informally label an Adam "Hopf" bifurcation |
| Quasi-periodic training (invariant circles, rotation numbers) | Novel for optimizer-on-loss maps; only loose uses of "quasiperiodic" in ML papers |
| Phase locking / Arnold tongues in hyperparameter planes | Novel for gradient descent, heavy-ball and Adam; documented in learning-in-games maps (Metaxas and Sapsis, arXiv:2606.29967; Umezuki 2018; Goll, Barfuss and Heitzig 2026) |
| Devil's staircase / Farey ordering | Novel. Goujaud, Taylor and Dieuleveut (Math. Program. 2025; open problem arXiv:2502.19916) find heavy-ball cycles rotating by 1/K and ask about fractional K, a natural rotation-number question |
| Torus-breakdown routes in training | Novel; the ML literature reports period doubling, Li-Yorke or fractal chaos, and intermittency hints |

Structural notes from the check (the agent's own algebra; to be verified
before use): at a gradient-descent fixed point the Jacobian `I - eta H` is
symmetric, so multipliers are real and Neimark-Sacker is impossible there; it
needs an orbit of period at least 2 in dimension at least 2. One-dimensional
heavy-ball has constant Jacobian determinant below one, which excludes
invariant closed curves.

Classical references to verify for the mechanism: Arnold 1961/1965 (circle
maps); Aronson, Chory, Hall and McGehee, CMP 83:303 (1982); Jensen, Bak and
Bohr, PRL 50:1637 (1983) and PRA 30:1960/1970 (1984); Kaneko, PTP 69:1427
(1983) and 71:1112 (1984); Curry and Yorke, LNM 668:48 (1978); Afraimovich and
Shilnikov (AMS Transl. 1991); Ruelle and Takens (1971); Newhouse, Ruelle and
Takens (1978); Ostlund, Rand, Sethna and Siggia, Physica D 8:303 (1983);
Gallas, Physica A 202:196 (1994); Schilder and Peckham, J. Comput. Phys.
220:932 (2007); Freire, Poschel and Gallas, EPL 100:48002 (2012); Kuznetsov
and Meijer, Numerical Bifurcation Analysis of Maps (2019).
