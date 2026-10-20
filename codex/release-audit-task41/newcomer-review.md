# Task 41 newcomer test

**B: WORKS.** An anonymous HTTPS clone of the public release, a fresh virtual
environment, a fresh model download and the README example completed. Eight
selected tests passed. This is a Linux-host smoke test of the released code,
not a new operating-system installation or a repetition of the paper's
scientific validation.

## Source and isolation

The clone is under
`/data2/neural-shrimp-data/scratch/audit-codex-1020b/newcomer/neural-shrimp`.
Both its HEAD and peeled `arxiv-v1` tag are
`1314528118ac1300c2cffd34fb60cfbfc67d2c82`. Tracked files remained unchanged
after execution.

The adjacent `newcomer/neural-shrimp-data` directory was new and empty before
setup. It was explicitly selected with `NEURAL_SHRIMP_DATA`, so its location
also agrees with the README's relative copy command. Both the dependency
cache and `HF_HOME` were new directories. The model loader explicitly uses
the selected data directory for its Hugging Face cache. Implicit Hub-token
use was disabled; the first download logged an unauthenticated-request
warning and succeeded.

The downloaded Qwen2.5-0.5B-Instruct snapshot is exactly the listed revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Seven downloaded model/tokenizer
files, their literal cache links and their complete file hashes are in
`newcomer-evidence.json`. The copied steering file matches its released
source hash.

## Steps and measured outcomes

| Step | Outcome | Elapsed seconds |
|---|---|---:|
| Anonymous public HTTPS clone at `arxiv-v1` | Exit 0; exact tagged source | 145.43 |
| `uv sync --frozen --extra dl --extra llm` | Exit 0; new virtual environment and dependency cache | 52.56 |
| README directory creation and steering-vector copy | Succeeded; copy later hash-checked | Not separately timed |
| README CPU float64 8×8 scan, with `--threads 6` | Exit 0; all 64 cells and five output files | 505.01 |
| Add locked `dev` extra for the requested tests | Exit 0 | 0.80 |
| Eight explicitly selected tests | Exit 0; eight passed, none skipped | 8.31 |

The scan's own loop timer reports 471.7 seconds. The enclosing 505.01 seconds
includes the first model download, load and final output/plot work. Thus this
run is consistent with the README's approximate eight-minute CPU estimate.
Maximum reported resident memory was 6,028,436 KiB for the scan. The selected
tests reported 6.54 seconds internally and 8.31 seconds for their enclosing
invocation.

Existing host Python was 3.11.5, satisfying the stated minimum. Runs used
CPU, `nice -n 10`, and explicit numerical-thread limits. The local GPU was
not used. The example has a 90-minute outer limit and was run only once.
The tests have a 300-second outer limit and were run only once.

## Commands

The public clone used credential helpers and interactive authentication
disabled:

```sh
git -c credential.helper= -c core.askPass=/bin/false clone \
  --branch arxiv-v1 --single-branch \
  https://github.com/tdj28/neural-shrimp.git \
  /data2/neural-shrimp-data/scratch/audit-codex-1020b/newcomer/neural-shrimp
```

Inside that clone, the README commands were:

```sh
uv sync --frozen --extra dl --extra llm
mkdir -p ../neural-shrimp-data/explore/llm-phase-map
cp -r docs/steering-vectors/vectors-l12 ../neural-shrimp-data/explore/llm-phase-map/
HF_HUB_OFFLINE=0 uv run python tools/explore_llm_soft_loop.py scan \
  --device cpu --dtype float64 --resolution 8 8 \
  --alpha 0.0625 0.0775 --log-temperature -0.17 -0.125 \
  --out llm-soft-loop/example --threads 6
```

The explicit `--threads 6` is the sole scientific-command deviation: the
assignment caps CPU threads at six whereas the released CLI defaults to
eight. It changes the resource setting, not the model, precision, mesh,
transient length, sampling length, tolerances or other scientific options.
The command was wrapped by `nice`, environment limits, timeout and
`/usr/bin/time` as recorded in `plan.md` and the compact receipt.

Pytest is absent from the README's two extras. For the separately requested
test subset, the declared `dev` extra was installed with:

```sh
uv sync --frozen --extra dl --extra llm --extra dev
HF_HUB_OFFLINE=1 uv run --no-sync python -m pytest -q -p no:cacheprovider \
  --basetemp /data2/neural-shrimp-data/scratch/audit-codex-1020b/test-temp-001 \
  tests/test_model_revisions.py tests/test_softmax_gain.py \
  tests/test_llm_soft_loop.py::test_cached_step_matches_full_recompute \
  tests/test_llm_soft_loop.py::test_low_temperature_feeds_back_the_greedy_token \
  tests/test_llm_soft_loop.py::test_float64_model_keeps_float64_norms \
  tests/test_llm_soft_loop.py::test_float64_model_builds_float64_rotary_tables
```

These cover one model-revision consistency test, three toy
covariance/factorization/rank tests and four cached-model tests. The model
fixture requests four PyTorch threads. No dense-Jacobian, subspace solver
or full-suite test was run.

## Outputs and interpretation

The example's output directory is the fresh data directory's
`explore/llm-soft-loop/example/`:

| File | Bytes |
|---|---:|
| `scan.npz` | 3,032 |
| `scan-partial.npz` | 3,858 |
| `scan.json` | 7,659 |
| `scan.png` | 82,887 |
| `progress.jsonl` | 425 |

All numerical arrays in `scan.npz` are finite; the exponent, period and entropy
arrays have shape 8×8. The saved parameters match the invocation.
The plot was visually inspected and is readable. It shows the expected
three panels with axes, labels and color bars.

There are 17 positive finite-time exponent labels at the configured
threshold; 46 cells receive a detected period and 18 do not. These short-run
labels are not rigorous attractor classifications, primitive-orbit proofs
or an independent implementation of the model. Task 41 asks whether a
newcomer can run this example, and that test succeeds.

The anonymous-Hub warning and a Transformers cache-method deprecation warning
did not prevent execution. No source patch, existing model cache or existing
steering-data copy was needed. If a newcomer chooses a different custom
`NEURAL_SHRIMP_DATA` path, the README's literal vector-copy destination must
also be changed to that path; that alternate layout was not tested here.

## License and link checks

The tag's README states MIT for code and CC BY 4.0 for manuscripts, figures,
documentation and research records. `LICENSE` contains the MIT terms and
`LICENSE-docs.md` states CC BY 4.0 and preserves third-party model/dataset
terms. These declarations agree. They do not independently establish
ownership of every third-party item; that is considered in Part C.

The README's three relative clickable targets resolve at `arxiv-v1`:
`paper/llm/main.pdf`, `LICENSE`, and `LICENSE-docs.md`. Its
[uv documentation link](https://docs.astral.sh/uv/) and the documentation
license's [CC BY 4.0 link](https://creativecommons.org/licenses/by/4.0/)
resolved when checked. The GitHub repository and tagged raw README returned
anonymous HTTP success. The arXiv identifier is explicitly pending, not
presented as a working preprint link.

The tagged README's approximate full-report page count is stale; the
post-cleanup README corrects it to 182. The tag was not modified.

## Receipt and limits

`newcomer-evidence.json` pins all five step logs/time records, output files,
downloaded model files and steering copies. `newcomer_receipt.py` ran only
as a read-only receipt checker, with no model calls. Its successful invocation
reported exit 0, 1.20 seconds and 37,252 KiB maximum RSS.

Two host-launcher setup limitations are preserved in
`receipt-execution-amendment.md`. In particular, a receipt-only invocation
failed with an undersized launcher address-space limit; the subsequent
disclosed invocation used 4 GiB. Neither event repeated or invalidated the
completed example scan or tests.

This does not establish installation success on another operating system,
GPU reproducibility, bitwise agreement with the original paper's large
maps, full test-suite success or availability of all external native
certificate evidence.
