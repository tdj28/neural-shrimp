# Task 41 exact PDF inventory amendment

Prepared 20 October 2026 before PDF payload screening. This is a separate
read-only public-release check, not a scientific rerun or a repeat of the
article build. Root must checkpoint and approve the exact source before
one invocation. No execution is claimed by this preparation.

The pinned compact `public-findings.json` is 138,425 bytes with SHA-256
`277813a8b2911a825ca50a1febd654370f3fb9b57d497d032a2cf32be7ebdabd`.
Its 54 PDF inventory entries identify exactly 27 unique path/blob pairs:
25 project figure paths, the project LLM report path, and the working-draft
path. These are provenance groups based on the known repository paths,
not an independent authorship, ownership, or license determination.

The 27 objects total 30,848,634 bytes and are identical at the two frozen
targets: `arxiv-v1` commit `1314528118ac1300c2cffd34fb60cfbfc67d2c82`
and reviewed post-cleanup commit `21272a7830d74fade1f6592af2a7b8189370ffad`.
The new script fixes every path, object ID and byte length literally and
checks only those exact paths at those commits. It does not enumerate
another tree or inspect general history. No environment-file or prohibited
project payload is requested.

`public_pdf_inventory.py` retrieves each exact Git object once, verifies
its Git object hash, computes its complete SHA-256, and writes it exclusively
to a numbered file in the fresh scratch allocation
`/data2/neural-shrimp-data/scratch/audit-codex-1020b/pdf-inventory-001/`.
No PDF or decoded text is written into Git. Existing scratch or result
allocations are refused; incomplete attempts and partial files are retained.
There is no automatic retry, deletion, overwrite, or resume.

The exact system `pdfinfo` and `pdftotext` executables are byte-pinned in
the source. Metadata and full text are captured through bounded pipes,
with child stderr discarded rather than risking content in logs. Metadata
fields produce only presence, byte length and hashes. Titles, authors,
matched values, snippets and exception messages are never emitted.
Full text is split by Poppler page separators; page-count disagreement
fails closed. The first-page classification is explicitly a subset of the
full-text classification.

The existing hash-pinned heuristic category definitions are selected by
AST from `public_blob_followup.py`; its imports and `main()` do not run.
PDF text uses conservative dollar-delimiter bookkeeping. Only closed
math-shaped expressions without monetary-context keywords retain that
suppression; the suppressed-line counts remain recorded. Other candidate
categories retain the frozen semantics. Candidates are locations for
review, not confirmed disclosures. All public output is value-free.

Limits are one serial worker, one numerical thread, nice 10, CPU only,
`HF_HUB_OFFLINE=1`, and empty `CUDA_VISIBLE_DEVICES`. Worker address space
is bounded at 2 GiB, CPU at 120 seconds, internal wall time at 170 seconds
and the external wall envelope at 180 seconds. Each external tool has
a 20-second limit within that overall deadline. Git payload is bounded
at 32 MiB per object and 32 MiB in aggregate. Extracted metadata is limited
to 64 KiB per PDF; full text to 2 MiB per PDF and 16 MiB in aggregate;
candidate rows to 10,000. Output caps refuse overflow rather than silently
truncate. These are resource limits, not measured completion forecasts.
The host launcher needs its separately reviewed envelope; the Python
worker sets the stated limits after launch.

The exclusive result is
`codex/release-audit-task41/public-pdf-inventory-evidence.json`.
Ordinary failures record a stage and exception category without content.
A hard resource kill can leave an empty or partial exclusive result;
the outer execution receipt must preserve that failed attempt and its
exit status. No result is called complete unless all 27 object, metadata,
page and text checks complete.

This does not render pages, perform OCR, inspect embedded attachments or
JavaScript, authenticate a native computation, or establish third-party
ownership. Empty extracted pages are explicitly listed. Successful text
extraction and absence of heuristic candidates are not exhaustive privacy,
credential or license clearance.
