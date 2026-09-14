# EXP-008 independent arithmetic results audit

14 September 2026. The completed diagnostic identifies numerical mechanisms
behind the known replay fragility. **It does not repair the original frozen
gate or establish a new geometry-utility result.** The original evidence,
coefficients, features and tolerances remain unchanged.

The diagnostic ran from private-pushed source
`b3cf51bb88df75eccec4b19e0f7a15976b3a9b21` and completed in 5.7033665 seconds
on the server, with no integrations or new targets. Its
[method](EXP-008-arithmetic-method.md) and executable were frozen before this
diagnostic execution, after the original EXP-008 results and failures were
already known. This is post-run numerical development evidence.

## Evidence and independent checks

The full report remains at
`/data2/neural-shrimp-data/audits/EXP-008-arithmetic/attempt-001/report.json`:
**6,794,062 bytes**, SHA-256
`56618ec929f8851cecd722cbeb279d66254ffdecb9cf020ccd1180d3302deba0`.
The [compact summary](EXP-008-arithmetic-summary.json) retains its provenance,
complete failure counts and the numerical aggregates below. The full report
contains every directional row and all solver coefficients; it is not copied
into Git.

The report records successful original receipt/artifact bindings for all
10,650 artifacts, 48 sensitivity contexts and 3,456 directional feature rows.
It evaluates three learners, twelve development folds plus the final fit,
five penalties and three solvers: **195 common ridge objectives and 585
coefficient vectors**, with 540 fold/solver/penalty validation losses. All
original development/test records remain bound; only development labels enter
fitting. Test-action utility is not re-evaluated by this diagnostic.

This separate audit checked the full-report hash and size, the executable and
method hashes against their exact Git freeze, and original manifest/receipt
hashes. Direct reconstruction from the report reproduces all 585 pairwise
coefficient maxima, all 195 normal/augmented condition-number relationships,
and all nine solver/learner CV grids and selected indices. The independent
60-digit calculation in the appendix additionally checks the known failing
direction from the original saved sensitivity arrays, without importing the
diagnostic implementation.

The recorded runtime is Linux x86-64, Python 3.12.14, NumPy 2.5.3 and SciPy
1.18.1. OpenBLAS, OMP, MKL, BLIS and NUMEXPR thread settings were each one.
Binary64 has 53 mantissa bits and the measured long-double type has 64. This
is additional precision on already rounded input arrays, not exact integration
or a certified real-arithmetic reference.

## Directional cancellation explains the two local-feature failures

The unchanged formula fails the same two rows:
`development-10-p675-r0.016-c16` and `...-c17`, the two opposite local
directions at one context. All 48 saved/recomputed direction comparisons pass
their original arithmetic rule; their largest component difference is
2.4980e-16. No frozen quadratic form is negative in the 3,456 rows, so no
observed negative value is hidden by the original nonnegative clamp.

For the failing pair, the sum of absolute quadratic-form terms is
**20,500.77 times** the reference quadratic value. The positive and negative
terms nearly cancel. The original projection already forms residual columns;
this is not a defect from subtracting two large Gram matrices.

At candidate 16, the values below distinguish the **original saved feature**
from its **Linux recomputation using the frozen Gram formula**. Candidate 17
has the same squared response and RMS values because its delta has the
opposite sign.

| Evaluation | Local predicted RMSE | Absolute error against Decimal reference |
|---|---:|---:|
| Original saved feature | 0.0036248908761150602 | 5.38126e-15 |
| Linux frozen-form replay | 0.0036248908761290386 | 8.59714e-15 |
| Direct norm of projected response | 0.0036248908761204466 | 5.05520e-18 |
| Independent Decimal reference | 0.003624890876120441505141… | — |

The original saved-to-replayed difference is **1.39784017694e-14**, above the
unchanged combined bound **1.36248908761e-14** by a factor of 1.02595. It
remains a failed comparison. Neither the original saved value nor the Linux
replay is treated as the arithmetic ground truth.

The diagnostic separates squared-response errors at this fixed delta:
rounded Gram formation contributes an absolute discrepancy of **5.85450e-17**,
quadratic-form evaluation **3.74389e-18**, and projection **3.78898e-20**.
These quantities are in squared-error units, distinct from the RMSE errors in
the table. Rounded Gram formation dominates this example. Evaluating the
directional response first and then its squared norm avoids that measured
source of cancellation; its RMSE error is roughly 1,700 times smaller here.

The full census limits the claim. Against the diagnostic's extended reference,
direct evaluation is closer in **1,784** rows, equal in **1,396**, and farther
in **276**. Its maximum RMSE error is **3.66460e-16**, versus **8.59714e-15**
for the frozen-form replay. It is not uniformly better on every row, and
neither calculation checks whether the saved sensitivities accurately follow
the differential equation.

## Ridge coefficients are sensitive; changing the solver does not repair them

Every solver uses the same centering, scaling and intercept reconstructed on
this host by the original implementation, with the same saved development
data. All nine solver/learner combinations retain
their original CV-selected penalty. At those penalties:

| Learner | Selected penalty | Normal-matrix condition | Augmented-design condition | Failed coefficients versus original: normal / SVD / QR |
|---|---:|---:|---:|---:|
| Baseline | 1e-6 | 13,494,907 | 3,673.54 | 8 / 8 / 8 |
| Geometry | 0.01 | 1,511.89 | 38.883 | 0 / 0 / 0 |
| Shuffled geometry | 1e-6 | 14,779,432 | 3,844.40 | 8 / 8 / 8 |

The largest coefficient discrepancies against the original saved vectors
are **7.81991e-8**, **8.72488e-14** and **7.79794e-8**, respectively, across
the three solvers at the original selected penalties. The existing
`rtol=1e-7, atol=1e-9` failures remain failures for SVD and QR as well as
normal equations. Solving the present objective more consistently cannot
retroactively make the original coefficient vectors pass.

Across all 195 objectives, normal versus SVD differs by at most **3.01537e-8**
in a coefficient and **5.89432e-11** in a training prediction. SVD versus
pivoted augmented QR agrees more closely: **9.46235e-11** and **1.96486e-13**,
respectively. These are same-data, same-standardizer solver comparisons;
they do not isolate every source of the historical cross-platform difference
and are not test-action comparisons.

All three solvers' objective values are identical **at the report's exported
binary64 precision** for every objective. Their relative stationarity
diagnostics remain below 9e-16. QR has the smallest measured stationarity in
154/195 objectives, normal equations in 27, and SVD in 14. Normal equations
are not uniformly worse by this check. The forward sensitivity of correlated,
weakly regularized coefficient coordinates is compatible with small backward
errors and almost identical fitted predictions. The estimated
stationarity/lambda error bounds are floating-point diagnostics, not certified
intervals or identified physical feature effects.

## Consequences for the next numerical method

The evidence supports prospectively specifying **direct norms of projected
responses** for future local-error features and **weighted SVD or augmented
QR** for future ridge fitting. Preserve explicit conditioning diagnostics,
the mathematical objective, group weights, fold separation and unclipped
ranking semantics. No solver was selected using held-out utility in this
diagnostic.

Redundant-column treatment also needs an explicit future specification.
Removing or reparameterizing correlated columns can change the ridge penalty
and hence the fitted model; it is not automatically an equivalent numerical
repair. The current comparison holds standardization fixed and does not test
such a redesign.

These arithmetic results leave integration convergence, fully regenerated
inputs and any revised action policy to their separate assessments. They
provide no new trained-model result, controller benefit, memory-capacity
result or justification to replace the old failure records. The broad
geometry-utility conclusion is unchanged by this diagnostic alone.

## Appendix: independent post-run Decimal check

This appendix was added after inspecting the arithmetic report. It checks the
already identified failing case; it is not a prospectively sampled accuracy
study. The source sensitivity file is
`evidence/EXP-008/attempt-001/sensitivity/development-10-p675.npz`, SHA-256
`ca484b26f759c10057d4d5a5e5cb6eee3838309a9eed33af832218d0dac1741f`.

The following code, run from the repository root, reconstructs the projection
and response with 60 decimal digits from exact binary64 inputs. It produces
`0.00362489087612044150514103547841289857998083973208100474807409`.
Precision is finite, and the original input tensors are still rounded.

```python
from decimal import Decimal, localcontext
import numpy as np

path = "evidence/EXP-008/attempt-001/sensitivity/development-10-p675.npz"
with np.load(path, allow_pickle=False) as raw:
    tail = raw["times"] > 75
    S = raw["activation_sensitivity"][tail].reshape(-1, 2)
    v = raw["activation_velocity"][tail].ravel()

with localcontext() as arithmetic:
    arithmetic.prec = 60
    dec = lambda x: Decimal.from_float(float(x))
    columns = [[dec(x) for x in row] for row in S]
    phase = [dec(x) for x in v]
    denominator = sum(x*x for x in phase)
    coefficients = [
        sum(x*row[j] for x, row in zip(phase, columns)) / denominator
        for j in range(2)
    ]
    delta = [dec(0.0010572155364542508), dec(-0.01596503352046214)]
    response = [
        sum((row[j] - x*coefficients[j])*delta[j] for j in range(2))
        for row, x in zip(columns, phase)
    ]
    quadratic = sum(x*x for x in response) / Decimal(len(response))
    print(quadratic.sqrt())
```

The [prefreeze design audit](EXP-008-arithmetic-design-audit.md) records
**18 passing arithmetic tests in 0.33 seconds**, covering analytic ridge
solutions, deliberate cancellation, correlated designs, the CV path, invalid
inputs, evidence/source gates and failed-output retention. This results
audit added arithmetic reconstruction checks without changing those tests,
source, original data or endpoints. No paid computation or review was used.
