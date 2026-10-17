# Task26: frozen paper number-to-data candidates

17 October 2026. The checker and its corrected frozen scan are complete. This
accepts a source-finding tool and its saved candidate index, **not the scientific
support for every paper claim**. Numeric coincidence cannot identify the correct
cohort, statistic, unit, source or extremum. All candidate matches remain visible
for that contextual review.

The paper snapshot is commit `36713c413505cf814dd114e5618bd58212019619`, with
`paper/llm/main.tex` SHA256
`910c8e37a201c769937b3e524417d3adcbdb653bfd8bc1bbde808ad6c7c929e7`.
Its input manifest is
`6f99963fb87d01124bf47bbe1ce9f8bf00bb37195249c184a94e93a76792645a`.
The scan binds 1,361 committed JSON files (28,563,249 bytes), the main source and
five literal input-caption files (394,198 bytes), and the optional research log
(3,070,896 bytes). Those 1,368 selected files total 32,028,343 bytes; the inventory
adds 543,395 bytes. Input hashes were checked before and after execution.

| Category | Count |
|---|---:|
| Extracted numeric tokens | 6,513 |
| Numeric data candidates | 5,905 |
| Outward-bound candidates | 373 |
| Log-only candidates | 226 |
| Unmatched literals | 9 |
| Ambiguous data-candidate sets | 6,215 |
| With an args/config candidate | 3,938 |

The index contains 541,163 numeric references, 154,427 distinct values and
1,862 shared candidate groups; no selected JSON file failed parsing.

The ambiguity and design-candidate counts overlap the match categories. A
design tag means that an args/config path was among the candidates; it does not
verify the intended experimental setting. A log-only match supplies a log lead,
not a missing raw-data receipt. The unmatched list below identifies unlinked
literals under this index and matching rule, **not nine established errors**.

| Number ID | Frozen main.tex line | Literal | Context |
|---:|---:|---|---|
| 1573 | 1054 | `0.072940` | Period adding written in text |
| 1579 | 1057 | `0.069060` | Period adding written in text |
| 1903 | 1233 | `-72.695` | Mechanism: a saddle-node bottleneck |
| 1953 | 1252 | `42.64` | Mechanism: a saddle-node bottleneck |
| 2032 | 1288 | `2.895` | A period-doubling cascade to chaos; lower bound |
| 2035 | 1289 | `4.101` | A period-doubling cascade to chaos; upper bound |
| 2556 | 1525 | `0.18921` | Steering toward hostility |
| 5367 | 3459 | `4.648` | Replication and scale |
| 6322 | 3924 | `893\,868` | Evidence and reproducibility |

The external report preserves every extracted number's file, line, column,
context, quoted precision, category and candidate count. Five ranked previews
are convenient entry points; they are not the complete set. The lossless
`numbers.json → candidate-groups.json.gz → receipt-index.json.gz` chain retains
every candidate reference. The README contains an expansion example.

Matching uses Decimal intervals at the literal's quoted precision, including
fraction-to-percentage candidates. Bound candidates search one outward rounding
quantum and remain explicitly flagged; this does not prove an inequality or
check the true extremum of an experiment. Citation/reference keys, rendering
commands, explicit dates and contextually identified years are masked with
recorded dispositions. Main prose, equations, tables and literal input captions
are considered. This is a literal TeX heuristic, not macro expansion or a TeX
engine: number words, plot-image values, generated labels, arbitrary malformed
markup and text semantics nested in math remain outside its scope. Some literal
rendering quantities can remain as candidates. Current manuscript revisions
after the frozen hash have not been scanned here.

Run001 completed but misread prose hyphens as negative signs. Run002 corrected
that rule, but saved-output review found that `\\[2pt]` at main line606 falsely
opened a display-math span through the remaining document. Run003 corrects only
literal delimiter handling, with escape parity and adjacent-dollar state. All
original sources, outputs, receipts, failures of interpretation and dated
amendments remain unchanged. Nineteen original manufactured groups/profile,
eight sign groups and seven delimiter groups each ran once; no broad suite or
profile was repeated. The two narrow corrections received independent source
review. No scientific threshold or input was changed to improve agreement.

The accepted scan003 used checker SHA
`d8c4c278b77c1507cbaa308913eb24c2e3bfe57f6085fa94dc627e864d31f403`
at executed source commit `90785df2617172ede10c543caf914f64057f3acd`.
It completed exit 0 under CPU1 plus supervisor, nice10, 2GiB AS, 140 CPU seconds,
150 wall seconds and a sampled 128MiB output cap. The supervisor measured
118.634965 s externally; the checker recorded
113.432211 s before final output serialization. Child RSS recorded
at reaping was 536,832 KiB. Seven output files total
22,524,890 bytes. Inference observed the original supervisor/worker
termination and authenticated all three source blobs and seven output files.
Root read the final summary, receipt/closure fields, every unmatched row and the
specific corrected prose-sign cases; a saved comparison retains all 362 token
value changes from scan002. The narrow search found no remaining negative
period/model/block/top/batch regression cases. This is targeted output review,
not manual verification of every candidate or a universal TeX parsing claim.

Canonical external outputs:

- [Complete report](/data2/neural-shrimp-data/codex/paper-number-check-20261016/run-003/report.md), SHA `c0e1a1c4c9e8bdf2cb73a03d2d50ea21d44568ff2ed12b69dd4481ce202705f0`.
- [Output manifest](/data2/neural-shrimp-data/codex/paper-number-check-20261016/run-003/manifest.json), SHA `1294e3793551d115fa63ddd6cd166a0f3020cc66519e8a71cace5979054ad8a8`.
- [Run summary](/data2/neural-shrimp-data/codex/paper-number-check-20261016/run-003/run-summary.json), SHA `ff499b7e33eee0b9a2e7e5bf5284cce8eeacc114764647e11afe95e57c4d6f1a`.
- [Original receipt](/data2/neural-shrimp-data/codex/paper-number-check-20261016/run-003-supervision/receipt.json), SHA `776b880dda9db1abc8aaa770efa267088439e020d903aeca35414ce515736d5b`.
- [Original closure](/data2/neural-shrimp-data/codex/paper-number-check-20261016/run-003-original-closure.json), SHA `cafbdb0a67d61f7f74ce3648bcb916e6fc7a351be3791d2d69c0a0ab37f6c7da`.

The saved token-review record is SHA
`aa25c483704584f012f1cd067d164333bf04be29b9c5ffa2f2a51ed0c7dd6b09`.
Its compact provenance pointer is retained with this report; the original
output manifests bind all full candidate/index files. No second scientific
analysis or model execution was used to validate this utility.

Run the tool on a clean committed submission checkout with a fresh output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -I codex/paper-number-check-20261016/check.py \
  --root "$PWD" --output /absolute/path/to/a/new/report-directory --include-log
```

Default mode checks selected files against Git HEAD and records the exact new
main hash; dirty selected files are refused. A future scan requires its own
resource admission. The command is supplied for submission reuse, not scheduled
or launched automatically. The paper, producer tools and shared research log
were not edited. Bulky indices remain outside Git, with compact hash-bound
receipts and this report in the owned task folder.
