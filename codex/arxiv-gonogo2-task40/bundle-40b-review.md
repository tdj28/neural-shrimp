# Revised arXiv bundle and PDF review

**Build/content/layout PASS for article `e300c8eda07361fa5ad7054bf540a3951226c6b4`.** One minor bibliography metadata correction is visible in reference 17: `[cs.CL]` should be `[cs.LG]` according to Root's primary-record check. This is not a complete scientific or publication-release verdict.

The revised bundle and exact canonical article source produce the same 34-page content as the committed PDF. All 34 pages were visually inspected through an explicit split: this reviewer viewed every full-page PNG on pages **17–34**; Root separately viewed every PNG on pages **1–16**. Root's inspected-page hashes and observations are preserved in `visual-40b-root.json`; this review's page hashes and full placement ledger are in `visual-review-40b.json`. No page is marked inspected solely because rendered hashes match.

## Build identity and outcome

- Article: `e300c8eda07361fa5ad7054bf540a3951226c6b4`.
- Source script: `bundle_40b_audit.py`, SHA256 `6a28c6fc833c50e969392ebda82b977c85440ff7ce139183af37dfdb41257c33`.
- Private execution source recorded by Root: `a4d1969c9503c61aac7d67f7805009e03a6d6fe4`.
- Bundle: `/data2/neural-shrimp-data/arxiv/arxiv-source-e300c8e.tar.gz`, 10,728,088 bytes, SHA256 `ea7fd829935a5612ee0a26419099c091a0780e23b5f7f2d16bded3cf6d8fa38e`.
- Build evidence: `bundle-40b-evidence.json`, SHA256 `d093b2cf652b1672190e4326d6ef94e1c1f35a7d6ae24d4f95b28aef31f0fcd9`.
- Parent receipt: `bundle-40b-001.receipt.json`, SHA256 `1cbcadaaf3cbaccc52328a4a90a93da964748130b04ea6e2b0b07a7bf7073c60`.
- Receipt: exit 0, no timeout, child 1678578 reaped, 61.897043 seconds from parent launch through reap. Script internal elapsed time is 61.747998 seconds. One numerical thread, nice increment 10, 4 GiB address-space limit, 300 CPU seconds, 650-second parent wall limit; child build commands were serial with the script's shared 600-second deadline.
- Output is exclusively under `/data2/neural-shrimp-data/scratch/audit-codex-1020/bundle-40b-001`. Previous Task40 script, scratch and evidence were preserved. This reviewer did not repeat the build.

All 33 archive members meet the script's safe regular-file extraction checks. Frozen graphics match exact Git bytes; TeX matches the declared lexical comment normalization. The arXiv top-level/compiler JSON matches the required declaration. The canonical build independently regenerates `main.bbl`, which matches the bundled bibliography byte for byte.

There are 27 figure environments and 8 table environments, no duplicate or missing source labels, no unresolved final references or bibliography, no final rerun warnings, and no overfull-box warnings. Source references, auxiliary labels and extracted page text agree on the numbered displays.

## PDF equivalence

All three PDFs have 34 pages and 9,244,681 bytes:

| PDF | SHA256 |
|---|---|
| Committed article | `306ab78ba1ac08c1295103a3ca9650139129309631369f3961cac9f718940b1f` |
| Independent canonical build | `d7383ab34b0c42f49e3e73d416bc93580d7b7757c7e965dab32a4efdfa57b3b6` |
| Independent bundle build | `3746345d804a765d96fd752a759980467bea57570a89fc7b4cab03be0587822a` |

The layout text is exactly identical, SHA256 `e34328e698be501cb24f3f414d8471ac41e7c093ba959b02103c232449fcf806`, and every corresponding 90-dpi page PNG is byte-identical. Their original PDF bytes differ.

Root separately normalized exactly two PDF date fields and one document-ID array in each PDF. Full normalized bytes then match, SHA256 `bef49aa3cada06226dc41f3933c9ba1167e19370e11ebd595e68cdba61a24f78`. That distinct check is recorded in `bundle-40b-pdf-metadata.json`, SHA256 `31429287061155d3ef1921239a649d8b526933b137ad071466b530cb20c9c389`; this reviewer read the result and did not rerun it. Thus the reported original PDF differences are confined to those metadata fields.

## Visual and placement findings

No new clipped graphics, overlapping text, missing glyph boxes, overflowing tables, or panel/caption mismatches were observed on pages 17–34. Figures 15–27, Tables III–VIII and references 1–84 are legible at normal PDF zoom. Small multipanel legends benefit from zoom, as in the original layout. Root's separate pages 1–16 review reports no such defects and confirms **Figure 1 remains on page 1**.

Task40 F15's stray “Near” is fixed: article `main.tex:1407` and printed page 25 read **“Four of these five”**. Figure 22 is now one page after its first body mention, and Figure 25 is one page after its first body mention.

Figure 15 still has a two-page body-mention gap (15 → 17), despite its source move; the preceding full-width Figure 14 occupies page 16, and Figure 15 shares page 17 with Figure 16 and Table III. Figure 19 also retains a two-page gap (20 → 22), after the large full-width Figure 18 on page 21, with Figures 19 and 20 stacked on page 22. Both exceptions are disclosed in the adjudication and have visible float-layout reasons. This review does not claim every figure is within one page.

The ledger separates first body mentions from earlier forward references inside other captions. Plural and wrapped references were resolved against the source; external references such as “Fig. 7 of Ref. 32” are not counted as mentions of this article's Figure 7.

| Figure | First body mention | Display page |
|---|---:|---:|
| 1 | 2 | 1 |
| 2 | 2 | 3 |
| 3 | 4 | 4 |
| 4 | 5 | 5 |
| 5 | 6 | 6 |
| 6 | 8 | 9 |
| 7 | 9 | 10 |
| 8 | 11 | 11 |
| 9 | 12 | 11 |
| 10 | 12 | 12 |
| 11 | 12 | 12 |
| 12 | 13 | 13 |
| 13 | 13 | 14 |
| 14 | 15 | 16 |
| 15 | 15 | 17 |
| 16 | 17 | 17 |
| 17 | 19 | 19 |
| 18 | 20 | 21 |
| 19 | 20 | 22 |
| 20 | 21 | 22 |
| 21 | 22 | 23 |
| 22 | 23 | 24 |
| 23 | 24 | 25 |
| 24 | 26 | 26 |
| 25 | 26 | 27 |
| 26 | 28 | 28 |
| 27 | 29 | 29 |

Two additional two-page forward-caption references remain: Figure 1's caption on page 1 refers to Figure 2 on page 3; Figure 21's caption on page 23 refers to Figure 23 on page 25. Their first body mentions are each one page before display. Tables I–VIII display on pages 15, 16, 17, 18, 19, 19, 29 and 31.

## Remaining minor metadata issue

**Where:** `references.bib:185` at `e300c8e`; printed reference 17 on page 33.

**Visible text:** “arXiv:2606.30571 [cs.CL]”.

The revised entry no longer claims the unverified COLM venue. Root independently verified the primary arXiv category as `cs.LG`; this reviewer confirms that `cs.CL` is visibly printed in the exact revised PDF. Change `primaryClass` to `cs.LG` (or omit the category). Primary record: <https://arxiv.org/abs/2606.30571>. The primary-record verification belongs to Root; this visual pass did not repeat it.

## Scope limits

This checks the frozen revised bundle, build equivalence, source cross-references and displayed layout. It does not independently recompute plotted quantities, repeat numerical or certificate audits, establish that the public repository or Zenodo record is available, or authorize publication. Scientific corrections, the staged 42/48 wording, citation metadata and the human availability release gate belong to Root's separate Task40b integration review. No files in the article checkout, shared research log or previous audit outputs were changed.
