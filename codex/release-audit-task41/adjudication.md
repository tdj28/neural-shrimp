# Task 41 adjudication (Claude, 20 October 2026)

Each finding was rechecked against the repository before it was applied.

| Finding | Decision | Applied in |
|---|---|---|
| A: GO for the submitted article b15a593 | Accepted. No replacement of the arXiv submission is needed. | — |
| B: WORKS (anonymous clone at arxiv-v1, README install, 8×8 example, eight tests) | Accepted. The scan's `--threads` option (default 8) is left as released; Codex's six threads were an administrative limit. | — |
| C1: nonzero amounts in six JSON receipts (22 lines plus one key) and 49 textual references | Accepted. Forward redaction at HEAD. JSON values of keys ending in `usd` are set to null; nonzero amounts in text read "[amount redacted]", including the copies in `codex/night-results-review-task21/`; zero amounts are kept. `docs/release-redactions.json` lists every redacted file with its original blob, its SHA-256 and the redacted lines. The originals are unchanged at `arxiv-v1` and in the history. | 1f0f6206 |
| C2: GSM8K items distributed without the upstream MIT notice | Accepted, and extended: `docs/bz-tasks.json` also holds 50 NQ-open items (CC BY-SA 3.0) and 30 MBPP tasks (CC BY 4.0). `THIRD_PARTY_NOTICES.md` names all three, reproduces the GSM8K notice from the upstream repository, and keeps the NQ-open items under their share-alike license. README and `LICENSE-docs.md` point to it. | 1f0f6206 |
| C4: cleanup replaced a shell argument at `docs/research-log.md:18856` | Accepted, and extended: the first cleanup's second pass had taken shell positional parameters for amounts on seven command lines (18856, 18868, 19038, 19902, 19926, 19980, 20217). All seven are restored byte for byte from the original. Their other content was unchanged. | 1f0f6206 |
| CI hardening (pinned actions, no persisted checkout credentials) | Noted as optional; not applied now. | — |

The commit metadata identity (24 rows) is the owner's choice and is left as it is. History and tags are not rewritten.
