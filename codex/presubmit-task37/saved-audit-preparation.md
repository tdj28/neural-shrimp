# Task37 saved-numerical checker preparation

Prepared 2026-10-19 UTC. **No new neural-network behavior has been measured.**
This is source preparation for independently checking a future Task37 R-pair
attempt. No target arrays, model calls, producer imports, paid calls, GPU work,
existing six/22 controls, or Git mutations were used. The original packet is
unchanged. Root owns review, checkpoint, capacity and eventual execution.

The checker is deliberately limited to the actual `p6_kite.py` R-pair schema:
two seed directories, six phases, dimension 896, vocabulary 151936, and dense
JVP chunks of eight. The test-only Python API supplies dimension six and
vocabulary seven; the command-line interface has no such overrides. This is
not a checker for an unspecified future body-point schema.

## Source reviewed

| File | SHA-256 |
| --- | --- |
| `p6_kite.py` (read only) | `370c05f86334814848fdeb64ba323ab4d89bd50e7a549ac10e83ca3d322d9d45` |
| `periodic_audit.py` (read only) | `667ea2d366e6f49f48793d769a8c231079bce359315d4499a84a60b316b0d245` |
| `audit_saved_numerics.py` (new) | `e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215` |
| `test_saved_numerics.py` (new) | `c9ba170cebe556d21a4e3afa3f1c034736465e4fac4662a8a3cdfe5003ee1f24` |

The checker imports NumPy and SciPy's assignment solver, but neither the runner,
its numerical helper, the independent model map, nor producer code. It reads
arrays with pickle disabled, verifies finite float64/complex128 storage and
expected shapes, and hashes every input it actually reads. Inputs are read
only; a requested report must have a fresh path. Failure at either seed prevents
a positive saved-arithmetic result while the other seed is still checked.

## Independently recomputed from retained values

- Terminal Newton residual at a retained evaluation of its saved final state;
  the separate six-step phase chain and closure; composed-return residuals at
  all six phases; proper-divisor separations for 1, 2 and 3.
- Every dense matrix column against the corresponding retained standard-basis
  batch JVP and saved chunk. Every saved eigenpair residual, a fresh complete
  dense eigenspectrum with one-to-one eigenvalue matching, and the largest
  eight modes' retained scalar-JVP residuals. The nearest multiplier to -1,
  spectral radius, next-largest modulus and conditional stability label are
  checked. R remains explicitly unclassified as a bisected crossing.
- The three seeded random dense/scalar JVP discrepancies. Every FD row is
  reconstructed from its retained AD direction and eight plus/minus outputs;
  exact base/offset association, error norms, cancellation floors, error
  ratios, quadratic intervals and the final gate are checked. At least one
  random direction must have a quadratic interval, as the runner requires.
- Full-state cyclic alignment across all six shifts, using maximum phase
  distance and the mean of the two mean orbit norms. Root and seed summary
  copies are compared to these recalculations, including body-point
  eligibility as a numerical prerequisite only.
- Each phase's finite full-vocabulary logits, argmax token ID, top-two logit
  gap and probability at `model-ready.json`'s saved binary64 temperature.
  Composed-map/JVP invocation and column counters are replayed from the event
  sequence. These counts do not include independent authentication of the
  model calls behind the journal.

## Admission still required

`scientific_verdict_admitted` and `certificate_passed` are always false. There
is no switch to override this. A successful exit means only that the supplied
saved values and the checked summaries pass the specified numerical checks.
It is not a final admitted scientific verdict or an interval certificate.

Before scientific use, a separate review must bind the original supervisor's
exit and attempt identity, source commit and source-file pins, run card and
admission hashes, seed/input/model/tokenizer inventory, resource admission,
and the complete immutable raw-file inventory. The checker inventories only
files it reads. Those newly computed hashes cannot authenticate an original
run, prove completeness, or replace the original supervisor record. Recorded
temperature, random-generator behavior and expected vocabulary also remain
conditional on that source/runtime/inventory binding.

Exact unchecked items: left-eigenvector condition estimates (left vectors are
not retained); decoded token text (needs the pinned tokenizer); expected
embedding readouts (needs authenticated model embeddings); original
operator/model authenticity; effective internal arithmetic precision; and
Newton/GMRES iteration history, linear-solves and unconverged-GMRES counters.
The final Newton residual and operator counters are checked, but do not verify
every solver decision. FD arithmetic from saved outputs is checked, but is
not proof that those outputs came from the intended float64 composed map.
No target attempt exists for this preparation to audit.

## New synthetic controls and preserved attempts

Only the existing `/data2/neural-shrimp/.venv/bin/python` was used. Root had
already authorized this fallback after the preserved snap `uv` refusal;
`uv` was not retried, installed or modified here. Each invocation used
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `nice -n 10`, and Python `-B`.

- `/tmp/task37-saved-controls-attempt001`: eight tests passed, 0.851 seconds,
  explicit exit 0. Both source snapshots, every synthetic array/journal and
  checker report, stdout, stderr and exit-status file are retained. This
  earlier version preceded terminal-Newton/nearest-minus-one checks.
- `/tmp/task37-saved-controls-attempt002`: final changed-source run, nine
  tests passed, 0.963 seconds, explicit exit 0; the same complete retention.
  No control invocation failed. Deliberately corrupted and missing-evidence
  fixture cases remain saved with their rejected audit reports.

The controls cover valid complex eigensystems, two distinct phase arrays,
same-orbit shift three, altered dense matrix, missing operator output,
broken one-step phase chain, a false FD summary, altered retained FD output,
repeated/omitted saved multiplier, and false token ID. Every case asserts that
scientific admission stays false. These are manufactured saved-schema fixtures
with a small local cubic operator; they are not evidence about a model or an
actual composed six-step neural-network map.

Final control source snapshots equal the new sources above. Final stderr
SHA-256: `77185241df53639b37197a79b9444f9c57c2f0ba5fc5c826d3fee05ee03c536a`;
exit-status file (`0` and newline) SHA-256:
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

The final invocation was:

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TASK37_SAVED_CONTROL_ROOT=/tmp/task37-saved-controls-attempt002/fixtures \
nice -n 10 /data2/neural-shrimp/.venv/bin/python -B \
  /tmp/task37-saved-controls-attempt002/test_saved_numerics.py
```

For a future authenticated original attempt, root may run the new checker
with the same bounded environment, `--root ORIGINAL_ATTEMPT` and
`--report NEW_REPORT.json`. That will read the attempt and perform a fresh
dense eigenspectrum calculation; it will not evaluate a model or admit the
science. Runtime/source admission remains separate and mandatory.
