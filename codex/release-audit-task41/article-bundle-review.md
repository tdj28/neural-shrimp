# Submitted article: source, bundle, layout and citation checks

Checked on 20 October 2026, read-only. This note covers the submitted article
`b15a593676fa5ff3614b6bc1922fee514c08109b` relative to the previous reviewed
article `e300c8eda07361fa5ad7054bf540a3951226c6b4`. Numerical checks are recorded
separately in `article-raw-evidence.json` and its review.

## Source and submitted archive

The final Git difference consists of `main.tex`, `references.bib`,
`figures/fig-p6-return-maps-compact.png`, and the compiled `main.pdf`.
The TeX difference is 83 added and 87 removed lines; the bibliography changes
one field. No article file was edited by this audit.

The submitted source archive is 10,729,832 bytes, SHA-256
`2e7ee04f8e776b481000170493fb6511e3409c55a05279b9b22d34822a3859aa`.
The checked committed source hashes are:

| File | SHA-256 |
|---|---|
| `main.tex` | `c86823950d9c3464a3808ed5ff3244af7c74fbf5b18c0e00c66b92bf46fe7fcf` |
| `references.bib` | `da8a883749086af928b5aaff87e32610b0334d1ece65c99b9a195e5325c24914` |
| `main.pdf` | `fa59c605c9c99e23020c80e9b37c26ec695896ba9a2499f2cc436ed33886cc52` |

The independently checkpointed `bundle_audit.py` extracted the archive into
new scratch space and built both the bundle and the committed article source.
It compared archive graphics with their committed bytes and checked the
declared TeX comment removal. Both builds succeeded. Their generated
bibliographies match exactly. Both logs have no detected undefined
references, undefined citations, missing files, LaTeX errors or overfull boxes.

The bundle build, canonical source build and submitted PDF have exactly
matching extracted layout text and byte-identical rendered images for all
34 pages. The PDF container bytes differ; this check does not claim that
the PDF files themselves are byte-identical or identify every container-level
difference.

The source has 27 figure environments and eight table environments, with
no duplicate labels or missing reference targets. The result and command
receipts are in `article-bundle-evidence.json`. The bounded check completed
once, exit zero, with a reported internal interval of 61.157 seconds;
the enclosing timed invocation reported 61.39 seconds.

## Visual checks

Compared with the previous review's rendered-page hashes, 21 pages changed:
1, 2, 3, 5, 9, 15, 16, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
and 34. Each of those pages was inspected as a full-page image. The other
13 pages have identical rendered-image hashes to the earlier review; they
were not presented as a new visual inspection.

No clipping, overlapping captions, missing glyphs, missing panels or
unreadable table layout was found on the changed pages. Specific checks:

- The revised abstract, sampled-map count and scope qualifications are
  visible in the built PDF.
- The normal-form references resolve to Figures 14(c) and 15(b).
- Figure 17's caption states its level indexing and omitted uncertainty
  bars; Table VI and the period-384 discussion are legible.
- Figure 23 retains the numerical-evidence qualification.
- Figure 25's rectangle refers to Figure 24(a), which is the correct
  numbered figure in this submitted article.
- Figure 26(b)'s title says “several turning points”; the surrounding
  discussion says “more than one.” This avoids the earlier single-turn
  description. Numerical interpretation is checked separately.
- The concluding factorization scope, long-context observations and
  finite-sample Kaplan–Yorke qualifications are readable.
- Table VIII is on page 32 in the submitted PDF, followed by the complete
  reference list through page 34.

## Changed citation

The only bibliography metadata change is the primary arXiv category of
Ko and Geiping's *Attractor States Emerge in Multi-Turn LLM Conversations*,
from `cs.CL` to `cs.LG`. The primary
[arXiv record](https://arxiv.org/abs/2606.30571) confirms the title, authors,
identifier, submission date and primary `cs.LG` category, with `cs.CL` as
a secondary category. This edit is correct. No new publisher citation or
new DOI was introduced by the final difference.

## Data Availability paths

Using the research repository's peeled `arxiv-v1` commit
`1314528118ac1300c2cffd34fb60cfbfc67d2c82`, Git resolves:

| Named path | Git object |
|---|---|
| `certificates/llm-p8-certificate-final-001.json` | blob `17a3037d3baaa409e7a42361bbe41dad71ebff86` |
| `codex/certify-llm/` | tree `98cd3c2698544294903ee643a81ecaf13dcb8557` |
| `codex/coexistence-task33/` | tree `178fd21ec4c1493a341013d5e4b4f8658e7436c4` |

These are public repository paths, not evidence that all native interval
enclosures and original raw files are included in Git. The README explicitly
places bulky native evidence outside the repository. This audit did not
download a complete native proof package or perform a new interval proof.

## Limits

This lane checks the submitted artifact and changed bibliography metadata.
It does not independently regenerate unchanged scientific figures, certify
the model or backend, or infer correctness of a scientific claim from a
successful LaTeX build. The numerical lane and public-release lane have
their own evidence and limitations.
