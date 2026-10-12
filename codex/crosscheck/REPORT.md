# Task 2 completed: independent cross-check of the LLM bottleneck mechanism

12 October 2026. The independent transformer implementation reproduces the
primitive P2 and P8 cycles, the P2 fold, its normal-form coefficients, and all
18 controlled full-model passage counts. The local two-step normal form
predicts these passages within one iteration without fitting the passage times.
This supplies independent numerical support for the LLM paper's local mechanism.

The complete manual forward implementation imports no producer research tool
or Transformers model. It uses the same pinned pretrained weights and PyTorch
tensor/autodifferentiation primitives. This is a known-outcome numerical
cross-check (evidence rung 2), with independent implementation and solvers.

## Full-state cycles and spectra

Here `t = log10(T)` and alpha is physical steering strength. Each dense return
Jacobian is 896 by 896; all 896 eigenvalues and their residuals are retained.

| Case | t | alpha | Relative full-state closure | Leading multiplier | Second multiplier |
|---|---:|---:|---:|---:|---:|
| Stable P2 near the fold | -0.115 | 0.0624 | 7.89275e-15 | +0.605399529486875 | -0.021597682820953 |
| Historical manuscript P2 coordinates, current map | -0.17263 | 0.061579 | 2.41672e-14 | -0.270620082117908 | -0.058458058661950 |
| P8, current map | -0.1205 | 0.0717 | 4.53111e-12 | +0.004089399188953 | -7.019692625525e-6 |

All three pass the predeclared closure threshold of 1e-11, proper-divisor
separation, spectral contraction, eigenpair residuals, all phase closure checks
(<1e-10), and dense/composed versus phase-chain tangent consistency. The P8
relative returns at steps 1, 2 and 4 are respectively 2.02549, 1.44592 and
1.40533, establishing its primitive period numerically. The P2 one-step
separations are 1.03701 and 1.05872. The P8 maximum phase closure is 4.53111e-12;
its small residual is not being presented as an interval enclosure.

The two P2 rows are different parameter points. The second multiplier of the
manuscript P2 row is about -0.0585, not of order 1e-6. Claude has corrected the
manuscript's blanket multiplier sentence; the small complementary spectrum
is specific to the measured higher-period interiors.

### Discrepancies and historical map changes

The stable P2 reference already uses the current CPU float64 map. All 12 saved
reference multipliers match within 5.115e-13; its leading difference is
-5.11480e-13 and its second differs by +2.75543e-14.

The historical table P2 and P8 seeds/spectra precede the rotary-table precision
fix. Comparing them with the current all-float64 forward arithmetic includes a
map-version change; these differences must not be described as replication
roundoff or as an isolated measurement of the rotary fix alone.

| Comparison with historical result | Table P2 | P8 |
|---|---:|---:|
| Historical seed closure under current map | 4.07426e-7 | 2.11359e-6 |
| Relative seed displacement after refinement | 3.12236e-7 | 2.13583e-6 |
| Historical leading multiplier | -0.270620565825334 | +0.004086210916518 |
| Absolute leading-multiplier difference | 4.83707e-7 | 3.18827e-6 |
| Relative leading-multiplier difference | 1.78740e-6 (1.79 ppm) | 7.80252e-4 (0.0780%) |
| Historical second multiplier | -0.058457931314424 | -7.026043527925e-6 |
| Absolute second-multiplier difference | 1.27348e-7 | 6.35090e-9 |

Every available reference multiplier comparison (12 stable-P2, 8 table-P2,
8 P8), including absolute and relative differences, is retained in
[`audits/orbits-001.json`](audits/orbits-001.json) and the raw results. Relative
comparisons of eigenvalues near floating-point resolution should not be
interpreted as meaningful relative precision. No discrepancy is dropped.

## Fold, coefficients and passage mechanism

At t=-0.115, independent augmented spectral Newton yields
`alpha_c = 0.06279827217666506`, only -5.80092e-15 from the producer value.
Its full-state closure is 6.55145e-15, augmented residual 5.43531e-14,
critical multiplier 0.999999999999946, and complementary spectral radius
0.0369846745279201. The fold critical mode is neutral to numerical precision.

Nested automatic differentiation of the independent forward map gives:

- `a = -4.834227003689755` (producer difference +2.598e-12).
- `b = -25.4284561647936` (difference +5.61173e-5, or 2.21 ppm,
  from the producer's averaged finite-difference estimate).
- `ab = 122.92692945398646` (difference -0.000271284).

Repeating the producer's five finite-difference step sizes reproduces each b
estimate within 5.99e-10; its small discrepancy from nested AD is consistent
with quadratic finite-step bias. Parameter differentiation includes the effect
of steering on the prefix cache. See [`FOLD.md`](FOLD.md) for the full comparison,
neutral-mode labeling correction, and independent saved-array audit.

All **18/18** full-transformer passage counts match exactly, across six epsilon
values and three fixed entrance conditions, including transverse perturbations.
The principal four full-channel passages are 144, 71, 34 and 14 two-step
iterations. The quadratic model matches seven counts and overestimates eleven
by one iteration. This includes a short development passage of 3 versus 4
iterations: 25% relative to the quadratic prediction, or 33.3% relative to
the measured count. The one-iteration statement is absolute.
The maximum pre-exit coordinate discrepancy between transformer implementations
is 1.66088e-11, against a minimum pre-exit/crossing margin of 8.88528e-6.

Substituting our new AD coefficients while keeping the original width and
entrances fixed preserves all 18 discrete quadratic predictions. Continuous
predictions shift by at most 0.00013743 two-step iterations. This scalar recheck
uses the existing full-model trajectories at the original centre; it is not a
second full-model passage experiment at the newly refined centre.
[`PASSAGES.md`](PASSAGES.md) records every count and the independent first-exit
and trajectory audit.

An additional arithmetic reconstruction of the 18 saved stable-branch points
gives quadratic extrapolation `ab = 123.0831902574`, reproducing the quoted
123.1. The linear extrapolation is 120.5718416508. The fit-order sensitivity,
one failed stricter closure screen, and disclosed restoration of the archived
producer criterion are retained in [`BRANCH.md`](BRANCH.md). This checks the
historical fitting recipe, not an independently recomputed 18-point branch.

## What this adds to the LLM paper

The main contribution supported here is a quantitatively predictive local
bifurcation mechanism in a transformer inference feedback map: derivative
information at a two-cycle fold predicts controlled bottleneck residence times,
and the result survives an independent forward implementation and numerical
solver. The orbit results also support the paper's measured P2/P8 examples.
These checks strengthen the numerical mechanism claim; priority against prior
literature remains the separately maintained literature assessment.

The state is one 896-dimensional continuous expected embedding, with a fixed
37-token prefix and frozen Qwen2.5-0.5B-Instruct weights. This is neither a
training cycle in 0.5 billion parameters nor an ordinary sampled-generation
experiment. It does not establish a practical training/inference benefit,
global shrimp organization, or a ball/interval certificate. Those are distinct
research questions, not outcomes of Task 2.

## Reproducibility and audit trail

- [`README.md`](README.md): original staged plan, gates and dated/disclosed
  amendments; [`reference-inputs.json`](reference-inputs.json): pinned inputs.
- Raw root: `/data2/neural-shrimp-data/codex/task2-independent-crosscheck/`.
- Orbit source commits: `9a609a7c9b0750bdc1f85ced63c355d4d9fa483d`
  (stable P2/P8) and `0b156204bd014f4f1821f3cc37d5beecec882053`
  (table P2); hash-checked inventories in `receipts/`.
- All finished raw attempts, including the failed stricter saved-branch screen,
  are preserved read-only with complete source/output hashes and exit receipts.
- Orbit offline audit independently recomputes all three dense eigensystems,
  saved trajectory closures/divisor returns, seed shifts and tangent matrix
  actions, and checks every source byte against its Git commit and every
  reference input hash. Per-phase return vectors and Newton linear-system
  vectors were not persisted: their scalar diagnostics are consistency-checked
  against stage records, not independently re-evaluated by that audit.
- The 31 Task 2 regression tests pass (manual-forward controls, numerical
  solvers, analysis gates, passages and branch reconstruction).
- Separate audits cover passages, fold/coefficients and branch arithmetic.
  Small floating-point residuals remain numerical evidence, not rigorous bounds.
- CPU only, four compute threads per model worker, at most two concurrently,
  nice 10, 16 GiB cap per worker. Stable P2/P8 completed in 1176.32 seconds;
  table P2 in 248.03 seconds. No GPU, RunPod or paid consultation was used.

Task 2 is complete. Claude owns the LLM manuscript and has the component
reports, corrected table coordinates, and coefficient-method update request
through the agreed coordination channel. Paper edits and other handoff tasks
remain with their respective owners.
