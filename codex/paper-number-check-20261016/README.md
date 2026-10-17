# Paper number-to-data candidates

This tool finds possible data sources for the paper's numeric literals. A match
does **not** verify a scientific claim: identical numbers occur in unrelated
experiments, metadata, parameters and formulas. Review the candidate key, cohort,
units and original receipt before asserting support.

Rerun on a clean, committed submission checkout using one CPU:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -I codex/paper-number-check-20261016/check.py \
  --root "$PWD" --output /absolute/path/to/a/new/report-directory --include-log
```

The output must be a new directory. Every selected input must match HEAD; a
dirty paper or receipt refuses execution. `--inventory` accepts an explicit
hash-bound source snapshot (the published run uses this mode). The report
records the exact main.tex SHA, source commit and every used file's SHA/size,
and rechecks the bytes after indexing. It does not edit the paper or receipts.

Outputs:

- `numbers.json`: every extracted literal, original line/column, section,
  sentence, precision, status, candidate count and five ranked previews.
- `report.md`: review list by section, covering unmatched, log-only, bound and
  ambiguous values. A design-parameter tag means an args/config candidate exists.
- `candidate-groups.json.gz` and `receipt-index.json.gz`: **all** candidates,
  without truncation, stored through shared IDs to avoid repeated giant lists.
- `coverage.json`: file dispositions, invalid/nonfinite inputs, excluded TeX
  references and explicitly excluded secret/monetary fields.
- `run-summary.json` and `manifest.json`: counts, timing, scope and output hashes.

To expand every candidate for a number ID from the report:

```python
import gzip, json
from pathlib import Path
p = Path('/absolute/path/to/report-directory')
number_id = 0
n = json.loads((p/'numbers.json').read_text())['numbers'][number_id]
groups = json.loads(gzip.decompress((p/'candidate-groups.json.gz').read_bytes()))
index = json.loads(gzip.decompress((p/'receipt-index.json.gz').read_bytes()))
print(n['file'], n['line'], n['sentence'])
for gid in n['candidate_groups']:
    g = groups[gid]
    for vid in g['value_ids']:
        v = index['values'][vid]
        for rid in v['refs']:
            print(g['mode'], g['receipt_multiplier'], v['value'], index['refs'][rid])
```

Matching uses Decimal half-open precision intervals, with native and
fraction-to-percentage alternatives. Bound candidates use one outward rounding
quantum and are explicitly flagged; this does not prove they are the correct
extremum. Lexical overlap ranks previews but never drops a candidate. Word-form
numbers, macro expansion, drawing geometry and plot-image values require other
checks. Literal `\input` captions are included; unresolved inputs are reported.

Large generated indices/fixtures belong in the external evidence store with
compact hash-bound receipts here. Preserve failed attempts and use a fresh output
directory for any reviewed rerun. No model execution or accelerator is involved.
