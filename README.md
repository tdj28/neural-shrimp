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

The visual revision contains ten figures: six evidence-derived plates, three
explanatory diagrams, and one exact analytic neural-cycle example. Each figure
has vector PDF/SVG, a 300-dpi PNG, and a machine-readable receipt recording its
source, construction, scope, and hashes. The [figure guide](docs/visual-revision.md)
connects each plate to the question it answers.

An exact tied-weight toy makes a concrete output consequence visible:
an attracting training cycle alternates predictions, and prediction averaging
reduces phase-dependent error while retaining bias. The [derivation](docs/analytic-cycle.md)
does not establish shrimp geometry or an advantage over a convergent solution.
The [observable-consequences plan](docs/observable-consequences.md) and
[next steps](docs/next-steps.md) specify how to test robustness, directional
control, stopping, and averaging against simpler diagnostics at matched cost.

Verify the released evidence and figures without running new experiments:

```sh
uv run --frozen python tools/summarize_pilots.py
uv run --frozen python tools/build_evidence_figures.py --check
uv run --frozen python tools/build_concept_figures.py --check
```

Omitting `--check` rebuilds figures from the immutable evidence or stated
algebra. Conceptual plots are explicitly labeled; they are not pilot outcomes.

See the [compute plan](docs/compute-plan.md) before considering a paid run.
Paid compute and paid reviews are disabled by default. The archive watchdog
limits the workload process; an external owner must still retrieve artifacts
and verify provider termination to stop billing.

This repository stays private until the owner explicitly authorizes publication.
