# Neural shrimp: do periodic windows help us manage neural networks?

Private exploratory research project. The first three local pilots are complete;
no new neural-network shrimp finding or practical advantage is established.
The question is whether information about
periodic windows improves a training or inference decision after paying for its
measurement, compared with ordinary loss, recurrence, and stability diagnostics.

We separate three claims: periodic orbit existence, utility of detecting cycles,
and incremental utility of shrimp geometry. They require different evidence.
The first implementation uses small deterministic CPU models in float64.

The initial instrument verified cycles in the Hénon control, but found no
verified cycles in the coarse neural scan. The recurrence policy never
triggered in the separate utility pilot and matched the fixed-rate trajectories
exactly. These finite pilot results are preserved in [the evidence](evidence/README.md)
and do not establish absence of neural shrimps or general futility of dynamical
control. [Next experiments](docs/next-steps.md) start with a neural periodic
positive control before extending the search.

## What is implemented

- Exact full-batch tanh training maps, gradients, Hessians, and Jacobians.
- Hénon/logistic controls; conservative full-state recurrence screening;
  numerical primitive-orbit closure and Floquet stability checks.
- Three frozen-input exploratory pilots: Hénon parameter scan, two-learning-rate
  neural training scan, and paired learning-rate control policies.
- Fixed-rate, scheduled-decay, scalar-stagnation and recurrence-triggered decay
  controls. Equal update budgets do **not** imply equal total compute costs.
- Immutable raw trials, exact source/input hashes, complete failure accounting,
  receipt verification, and descriptive reports/parameter maps.
- Offline RunPod archives and cost/watchdog preparation. This is a CPU
  implementation; GPU acceleration and provider lifecycle automation are not
  implemented or validated.

Shrimp boundary continuation, geometry-specific prediction/control, trained
latent-reasoner experiments, and practical speedup claims remain future work.
The [research plan](docs/research-plan.md) makes these gates explicit; the
[claim ledger](docs/claim-ledger.md) distinguishes each level of evidence.

## Reproduce locally

Python 3.11+ is required. Dependencies are pinned by `uv.lock` and a hashed
`requirements.lock` export. From a clean clone:

```sh
uv sync --frozen --extra dev
uv run --frozen pytest -q
uv run --frozen python -m neural_shrimp.cli validate experiments/manifests/*.json
uv run --frozen python -m neural_shrimp.cli plan experiments/manifests/EXP-002-training-pilot.json
```

The `.npz` input files already contain the exact training, validation and test
arrays, teacher parameters, and initial weights. No training outcomes are used
to prepare them. A changed design needs a new manifest/experiment ID; do not
overwrite a consumed manifest or attempt directory.

Execution requires clean source and a remote tracking reference containing the
source commit. Run only after the full code and manifest are committed/pushed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python -m neural_shrimp.cli run \
  --manifest experiments/manifests/EXP-002-training-pilot.json \
  --output runs/EXP-002/attempt-001
uv run --frozen python -m neural_shrimp.cli verify runs/EXP-002/attempt-001
uv run --frozen python -m neural_shrimp.cli analyze runs/EXP-002/attempt-001 \
  --destination artifacts/EXP-002/attempt-001
```

Each output directory is exclusively created. Incomplete attempts remain
inspectable and cannot be silently replaced. Resume is not implemented; spot
compute is therefore not qualified. The utility pilot takes no test-feedback
decisions: policies inspect only training loss or full training state.

## Paper and resources

The [working manuscript PDF](paper/neural-shrimp-working-draft.pdf)
([LaTeX source](paper/main.tex)) is a private research draft,
not a submission-ready paper. Build from `paper/` with `pdflatex main`,
`bibtex main`, then `pdflatex main` twice. References resolve to primary sources.

See the [compute plan](docs/compute-plan.md) before considering a paid run.
Paid compute and paid reviews are disabled by default. The archive watchdog
limits the workload process; an external owner must still retrieve artifacts
and verify provider termination to stop billing.

This repository stays private until the owner explicitly authorizes publication.
