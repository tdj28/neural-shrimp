# Task 41 audit plan

Prepared 20 October 2026 before the fresh-clone test.

The assignment is a read-only audit of the article changes from `e300c8e`
through `b15a593`, the research repository at `arxiv-v1`, and main after its
release cleanup. Only this report directory is changed in the research
checkout. The article and existing evidence remain unchanged.

## Newcomer test

1. Clone the anonymous HTTPS GitHub URL at `arxiv-v1` into a new directory
   under `/data2/neural-shrimp-data/scratch/audit-codex-1020b/`. If anonymous
   access fails, preserve that failure and disclose a local-clone fallback.
2. Read the cloned README and verify the tag's peeled commit.
3. Run `uv sync --frozen --extra dl --extra llm`. Use a new dependency cache
   and record the environment, command, elapsed time, and exit status.
4. Use a new, initially empty `../neural-shrimp-data` beside the clone as
   `NEURAL_SHRIMP_DATA`. Execute the README's directory creation and vector
   copy. Run its CPU float64 8-by-8 scan with the same scientific options.
   The assignment's six-thread limit requires the explicit administrative
   addition `--threads 6`: the released CLI defaults to eight. This
   deviation is disclosed rather than silently exceeding the limit.
5. The scan has a 90-minute wall limit including model download and load.
   Dependency installation has a 20-minute limit. These are resource caps,
   not completion predictions. No automatic numerical retry or changes to
   precision, resolution, transient, recording length, or tolerances.
6. Use `nice -n 10`, restrict numerical thread pools to six or fewer, and
   use CPU only. Recheck shared host capacity before execution. Stop only
   this audit's process group if the wall limit is reached.
7. Run a small explicitly selected unit-test subset, with its temporary
   directory inside the audit scratch area. Record all failures, including
   setup failures. Retain logs and outputs there and commit compact hashes
   and receipts only.

## Article and release checks

Inventory every final changed hunk and verify the changed claims against
the relevant primary records. Check the submitted archive against the
article commit, changed figure, figure/table references, changed
bibliographic metadata, and named certificate paths at the research tag.

Inspect tracked files and relevant history for release risks, excluding
all `.env` file contents. Sensitive findings are recorded by path and line
or commit, category, current presence, and recommendation, without quoting
the sensitive value. No off-limits project tree is opened or searched.

Reports are public. They contain no credentials, monetary amounts, or
private personal remarks. A passed check is scoped to the evidence examined;
unavailable data and incomplete checks remain explicit.
