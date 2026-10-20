# Task40 frozen arXiv bundle and PDF review

Build/content PASS at article `bb42a82189da91c37f50804c077832df9e2237ae`; two editorial/layout should-fix items below. This does not override the citation, raw-evidence or publication-state findings.

The only executed build attempt was the separately authorized host `bundle-002`, after private checkpoint `52eb95f1add11586ccc5ccd53244f12827bc4098`. It used frozen `bundle_pdf_audit.py`, SHA256 `a85b9a0a0442ad788f556b714cf1a8dc959cc8dd93d53268d6fcdbd119a6fdbf`, through Root's bounded runner: one serial child, one numerical thread, nice increment 10, 4 GiB address space, 300 CPU seconds per child, 600-second inner deadline and 650-second outer wall guard. Exit 0; child reaped; no timeout. Parent launch-through-reap was 61.443587634246796 seconds; the audit body recorded 61.326962533872575 seconds. These are execution durations, not CPU usage or model-run timings.

The first sandbox invocation, `bundle-001`, failed in the outer Snap uv launcher before Python or any child audit started. Its exact refusal and the absence of all output allocations are preserved in `bundle-001-launch-refusal.json`. The host invocation was a separately disclosed, authorized action, not an automatic retry or interpreter/source change. The unused fixed scratch directory retains the name `bundle-001`.

## Source and build evidence

- Input archive: `/data2/neural-shrimp-data/arxiv/arxiv-source-bb42a82.tar.gz`, 10,727,428 bytes, SHA256 `80d8273c8980555afd7d6e5b0ac3cbcad3ad648be8feb9bc5cf609a71218ea60`.
- All 33 archive members are regular files. No links, duplicate names, escaping paths or other entry types were accepted. The 27 raster figures match exact frozen article Git bytes. All four TeX files match an independently implemented escaped-percent-aware comment stripping of the frozen Git sources. The compiler/top-level declaration is the expected `pdflatex`/`main.tex`.
- A separate canonical scratch build regenerated the bibliography from the exact frozen `references.bib`; its `main.bbl` matches the bundled bbl byte-for-byte. The bundle built without BibTeX from its own bbl. No producer build function was imported or executed. The article checkout was not modified.
- Both final build logs contain zero undefined/missing references or citations, rerun requests, missing-file messages, emergency stops or overfull boxes. Source scanning found 27 figure environments, eight table environments, zero missing labels and zero duplicate labels. The generated aux confirms figures 1–27 and tables I–VIII.
- Frozen, freshly built canonical and bundle PDFs each contain 34 letter-size pages and are each 9,242,363 bytes. Extracted layout text is exactly equal; normalized text is equal; all 34 page PNGs rendered at 90 dpi are byte-identical across the three PDFs.
- Raw PDF SHA256s differ: frozen `b564dd5da877c8190c665287c6f7060407a159498b77b7bfefddbd23fef54a05`; canonical `79c3f56ac93b3553ccf8361eb70ce479762964bb100f2c255282fd0452dc648d`; bundle `5289a54e3fce2568b4719a70e99296a159b906b6e5d0cdd40cea8fd46315cd1c`. A subsequent bounded, read-only metadata comparison replaced exactly two `/CreationDate` or `/ModDate` values and exactly one trailer `/ID` array in each file. The resulting complete byte strings are equal, with SHA256 `86ebc266a440b442fc0396028cb3e3aca516eccc858d11513830740983716118`. Thus the byte differences are confined to these timestamp/identifier fields; this is stronger than merely observing equal extracted text.
- Receipt: `bundle-002.receipt.json`, SHA256 `fd4a377dcdd6daf8731c9faab7e3807082f383f8ce2764c992e24bfd2aaa0ecd`. Script evidence: `bundle-evidence.json`, SHA256 `1111572071e8f855ae02dd704a225fa89bb551d616cc28f689d23b28f1c4de39`. Command outputs, build logs, aux, exact snapshot inputs, three PDFs and 102 page renders remain under `/data2/neural-shrimp-data/scratch/audit-codex-1020/bundle-001/`.

## Visual inspection and findings

All 34 bundle page renders were visually inspected, in order, at their saved 90-dpi resolution. No missing graphics, visible clipping, overlapping text, broken equations, missing-glyph boxes, detached captions or table overflow were observed. Figure 1 remains on page 1. Figures' panel labels, axes, legends and captions describe the displayed qualitative content. This visual check does not re-estimate their scientific values; Root and the raw-evidence lane cover those. Small multipanel legends, particularly Figures 2, 9 and 10, benefit from PDF zoom, but no newly introduced raster truncation or broken layout was found.

### B1. Remove the stray “Near” from the cusp paragraph [should-fix]

Where: `main.tex`, Section VII, cusp unfolding paragraph, lines 1382–1383; PDF page 24, right column.

Claim: “Near four of these five come in two pairs, near the ends of the upper-left and lower-left legs”.

Evidence: The literal text is present in both the frozen source and the visually inspected bundle PDF. It is not a text-extraction artifact.

Correct wording: “Four of these five come in two pairs, near the ends of the upper-left and lower-left legs”.

### B2. Figure 22 falls two pages after its first body mention [should-fix]

Where: `main.tex`, Section VII, “A shrimp in the plane of Fig. 1”, lines 1344–1348, and `fig:p6shrimp` at lines 1269–1283; PDF pages 22 and 24.

Claim/check: The requested placement criterion is that no figure lies more than a page from its first mention without a reason.

Evidence: Figure 22 is first referenced in the body on page 22, including the continued-attractor discussion, but is displayed on page 24. The intervening page 23 contains the large Figure 21 and its detailed caption. Every other figure is within one page of its first body mention. Earlier forward pointers from captions create two-page gaps for Figure 2 (page-1 caption to page 3) and Figure 23 (page-23 caption to page 25), while their body discussions are within one page.

Correct action: Bring Figure 22 forward if the final revised layout permits, or explicitly retain the two-page delay as an intentional consequence of putting the canonical anatomy figure first. Recheck pagination after the must-fix prose changes. This is an editorial placement issue, not a build or scientific failure.

## Figure placement ledger

The body-mention page excludes captions and does not require a figure to follow its first mention. Earlier caption pointers are listed above. Source/aux and full-page inspection were used to resolve plural references and extraction line breaks.

| Figure | First body mention | Display page |
|---|---:|---:|
| 1 | 2 | 1 |
| 2 | 2 | 3 |
| 3 | 4 | 4 |
| 4 | 5 | 5 |
| 5 | 5 | 6 |
| 6 | 8 | 8 |
| 7 | 9 | 10 |
| 8 | 11 | 11 |
| 9 | 11 | 11 |
| 10 | 11 | 12 |
| 11 | 11 | 12 |
| 12 | 13 | 13 |
| 13 | 13 | 14 |
| 14 | 16 | 15 |
| 15 | 16 | 17 |
| 16 | 17 | 17 |
| 17 | 19 | 19 |
| 18 | 19 | 20 |
| 19 | 20 | 21 |
| 20 | 22 | 21 |
| 21 | 22 | 23 |
| 22 | 22 | 24 |
| 23 | 24 | 25 |
| 24 | 25 | 26 |
| 25 | 26 | 27 |
| 26 | 27 | 28 |
| 27 | 29 | 28 |

Tables I–VIII are displayed on pages 15, 16, 16, 17, 18, 19, 29 and 31 respectively and were visually checked. Table VIII's scientific status claims remain subject to the separate claims/raw-evidence review.

## Limits

No article edits, model calls, numerical orbit runs, original experiment reruns, paid services or publication actions were performed. The brief's instruction to defer the Zenodo content audit was respected. The bundle finding applies to the frozen snapshot only; any final author corrections require a fresh bundle/build check. The audit script's `completed:true` means its inspection completed, not a scientific or submission authorization. The manual visual review is recorded here, separately from the immutable script evidence's intentionally false `visual_inspection_completed_by_script` field.
