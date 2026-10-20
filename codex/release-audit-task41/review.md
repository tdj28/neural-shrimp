# Task 41 independent release audit

Audit date: 20 October 2026.

**A: GO.**

**B: WORKS.**

**C: ACTION-NEEDED.**

## Findings

No article replacement is required by the changed-claim checks in this
audit, and no blocking newcomer-run failure was found. The public
repository needs the following corrections. Sensitive values and private
remarks are deliberately omitted.

| Priority | Finding and evidence | Present at frozen HEAD? | Recommendation |
|---|---|---|---|
| **MUST-FIX — C1** | Operational monetary fields remain: `docs/EXP-013-local-review.json:85`, `docs/EXP-016-local-review.json:54`, and four receipts in `docs/consults/`. The confirmed subset contains 22 lines across six files; [the release review](public-review.md) lists every location and blob. | Yes; also at the release tag. | Make disclosed forward redactions of public-facing records, retaining immutable original evidence and its provenance under the project policy. A HEAD correction cannot remove copies in the existing tag/history. Do not rewrite history or move tags. |
| **SHOULD-FIX — C2** | `docs/pi-gsm8k-test.json:1` distributes 1,319 GSM8K records. The inspected file and license boundary lack the upstream MIT copyright/permission notice. | Yes; also at the tag. | Add explicit upstream attribution and retain the upstream notice, clearly separating the dataset's terms from the repository's own documentation license. Do not alter benchmark questions to redact their fictional arithmetic. |
| **SHOULD-FIX — C4** | Cleanup altered the nonmonetary shell argument after `--ridge` in `docs/research-log.md:18856`, replacing it with an amount-redaction marker. The preserved pre-cleanup/tag version supplies the original argument. | Yes; introduced by cleanup. | Restore the nonmonetary argument through a dated forward correction; preserve the original record and cleanup history. |

The CI review found no embedded credential, secret reference, untrusted
pull-request execution or environment dump in the inspected workflow.
Commit-pinning actions and disabling persisted checkout credentials are
optional hardening, not release-blocking findings.

C1's 22-line receipt subset is not an exhaustive count. The completed
semantic supplement identifies another receipt field and 49 textual project
monetary references, including duplicate copies and zero-spend statements.
Their value-free locations are included for the correction pass.

## A. Submitted article

The reviewed article is
`b15a593676fa5ff3614b6bc1922fee514c08109b`, compared with the last reviewed
article `e300c8eda07361fa5ad7054bf540a3951226c6b4`.
Every final changed TeX hunk was read. The difference comprises 83 added
and 87 removed TeX lines, one changed bibliography field, one changed
figure image and the rebuilt PDF.

The independent saved-data checker completed ten sections against 76
hash-bound inputs. It confirms the revised sampled-cell count, local
trace geometry, overlap and sign counts, cusp numerical context,
return-map turning description, cascade/caption indexing, selected
spectral statements, context-window fractions and sampled-embedding
bridge statistics. Notable results:

- 7,936 distinct sampled cells across the two maps, not 7,936 per map.
- 852 solved tracked-block cells; 38 overlaps below 0.9; trace/leading-sign
  agreement in 849 cells. Independent tracking and local quadratic fitting
  reproduce the positive trace saddle near
  `(0.07318804, -0.15208332)`, with trace `0.04488448`.
- The revised local/fitted wording preserves the failed prospective
  determinant-saddle prediction and does not claim a global two-dimensional
  reduction.
- The selected Figure 26(b) record has multiple turning points under
  both recorded estimators; the broader wording is supported.
- The three-cell finite-time Kaplan–Yorke estimates and the long-context
  fractions agree with the primary saved records at their stated scope.
- The reproducibility counts 173/192 and 7/192 use the explicitly stated
  final 500 steps. Different counts over all 1,500 steps are a different
  reporting window, not a contradiction.

The concluding factorization has the appropriate fixed-prompt,
expected-embedding, smooth-network and linear-softmax-head scope. For input
embedding matrix `E` and output-head matrix `W`, the derivative is
`Eᵀ(diag(p) − ppᵀ)W Dh / T`; tied embeddings give the covariance form.
This algebraic scope does not imply that every such network has the
observed bifurcation organization. The submitted conclusion correctly
limits the cross-family observation to the families scanned.

The archive was checked against the submitted commit and built in scratch.
All 34 rendered pages of the submitted PDF, rebuilt canonical source and
rebuilt archive agree byte for byte. PDF container bytes differ, as
explicitly recorded. All 21 pages whose render changed from the last
review were inspected visually. The source has 27 figures and eight tables,
with no missing/duplicate reference targets and no detected LaTeX errors
or overfull boxes. The changed citation category matches its
[primary arXiv record](https://arxiv.org/abs/2606.30571).
The named certificate paths exist at the research release tag.

Detailed changed-claim dispositions, original residual/precision/provenance
limitations and artifact checks are in:

- [Article raw review](article-raw-review.md)
- [Raw evidence and input hashes](article-raw-evidence.json)
- [Reproducibility-window supplement](article-tail-supplement.json)
- [Bundle, layout, citation and path review](article-bundle-review.md)
- [Bundle build evidence](article-bundle-evidence.json)

## B. Newcomer execution

An anonymous HTTPS clone at `arxiv-v1` resolved to
`1314528118ac1300c2cffd34fb60cfbfc67d2c82`. It used a new virtual environment,
new dependency cache, empty data directory and a fresh unauthenticated
download of the pinned Qwen2.5-0.5B-Instruct revision. No producer source
patch or existing model cache was needed.

| Step | Result | Elapsed seconds |
|---|---|---:|
| Anonymous clone | Exit 0 | 145.43 |
| README locked dependency installation | Exit 0 | 52.56 |
| README vector copy | Hash-identical copy | Not separately timed |
| README 8×8 CPU float64 scan | Exit 0; 64 cells | 505.01 |
| Install locked test extra | Exit 0 | 0.80 |
| Eight selected tests | Eight passed; none skipped | 8.31 |

The example was run once, with the required administrative addition
`--threads 6`; the released default is eight. All other scientific options
were unchanged. It produced `scan.npz`, `scan-partial.npz`, `scan.json`,
`scan.png` and `progress.jsonl`. All saved numerical arrays are finite.
The figure is readable.

The tests cover model-revision pins, toy covariance/factorization/rank,
cached versus full forward computation, the low-temperature limit,
float64 normalization and float64 rotary tables. They use a scratch
`--basetemp`; the full test suite was not run.

The MIT/CC BY 4.0 declarations agree with the README. The three relative
README links resolve at the tag; the external uv and CC BY links resolved.
The arXiv identifier is explicitly pending. The tag's stale approximate
full-report page count is already corrected at post-cleanup main.

See [Newcomer review](newcomer-review.md) for exact commands, setup caveats,
resource limits and output counts, and
[Newcomer evidence](newcomer-evidence.json) for log, file, model and
steering hashes. Two host-wrapper setup limitations, including a failed
receipt-only launcher invocation, are preserved in
[the execution amendment](receipt-execution-amendment.md).
The scan and tests were not repeated.

## C. Public repository

The release audit freezes the research tag at
`1314528118ac1300c2cffd34fb60cfbfc67d2c82` and post-cleanup main at
`21272a7830d74fade1f6592af2a7b8189370ffad`, following cleanup
`e359312e339c8f2fda59f280e6d7f92d88c0552d`.
Later commits made by this audit are confined to this report directory.

The first bounded screen reached its candidate cap, mainly because of
mathematical dollar notation. It is explicitly incomplete and preserved
by [its receipt](public-audit-attempt001-receipt.json).
A separately reviewed, checkpointed
[follow-up design](public-audit-amendment-002.md) prioritizes the two exact
target trees and cleanup file versions, with value-free detailed candidates
in scratch and compact evidence in Git. No general all-history clearance
is inferred from either screen.

The completed follow-up assigned all 26,236 requested unique blobs a
disposition, screening 5,539 distinct text blobs and 215,768,591 bytes.
Each frozen tree has 28,684 entries. The supplement separately screened
the seven oversized JSON/JSONL blobs, another 114,608,770 bytes, with no
additional candidates. It completed once with exit 0 in 38.75 seconds.
The original 362 candidate rows were triaged without publishing matched
values. Credential-shaped matches that were inspected resolved to Git
commit references; no live credential was confirmed or tested.
All 41 initially unresolved current candidates received a final semantic
disposition. All 26 flagged current personal-context rows were read; none
was a private personal remark. This does not clear unflagged content or
the entire repository history.

The separate PDF supplement verified all 27 inventoried PDF objects and
screened their metadata and extracted text: 241 pages, 819,973 text bytes,
no empty extracted pages or new candidates. It completed once with exit 0
in 2.60 seconds. This is text screening, not an ownership determination or
inspection of image pixels and embedded attachments.

The release review records sensitive finding locations, cleanup effects,
third-party notice checks, CI observations and the limits of historical
coverage. Its evidence is:

- [Public-release review](public-review.md)
- [Frozen tree screen and confirmed subset](public-findings.json)
- [Seven-file text result](public-large-text-result.md) and
  [evidence](public-large-text-gap-evidence.json)
- [Candidate features](public-candidate-triage.json) and
  [initial category dispositions](public-semantic-triage.json)
- [Completed semantic review](public-semantic-supplement.json)
- [PDF text result](public-pdf-inventory-result.md) and
  [evidence](public-pdf-inventory-evidence.json)

## What this audit does not establish

- It is not a new interval proof or fresh replay of all unchanged scientific
  experiments. Older failures, missing final diagnostics and effective
  precision limits remain explicit in the numerical review.
- Primary exploratory records without an original source/launcher receipt
  are hash-bound here, but their historical execution is not thereby
  independently authenticated.
- Newcomer success on this Linux host does not establish success on every
  operating system, bitwise agreement with the paper's large maps or
  certified attractor classifications from the short example.
- Certificate code and release records exist in Git; complete bulky native
  proof evidence and original trajectories are not all distributed there.
  No Zenodo package was audited in this assignment.
- Public screening excludes every `.env` payload and the prohibited
  project directories. Binary/image payloads, arbitrary encodings and
  unreviewed historical content prevent an exhaustive absence claim.
- No credential was tested against a service, no paid compute was used,
  and no article, producer file, tag or existing evidence was modified.

Only `codex/release-audit-task41/` is authored and committed by this audit.
Bulky execution logs, model downloads, clone and raw screening output are
under `../neural-shrimp-data/scratch/audit-codex-1020b/`.
