GO — revised article e300c8e, within the Task40b review scope.
No remaining scientific or build must-fix findings; two minor article recommendations below.
The revised bundle matches the canonical 34-page article; every page was visually inspected.

# Task40b final revised-article and bundle check

Article reviewed: **`e300c8eda07361fa5ad7054bf540a3951226c6b4`**, compared
with `bb42a82189da91c37f50804c077832df9e2237ae`.
The revised `main.tex` has SHA-256
`65fad928b20e812daa973193abba57c64e23e75926ec142902c25a871d2a2824`.
All article locations below refer to this exact revision in
`/data2/chaos-journal-neural-shrimp`, not a later author edit.

The Task40 scientific and wording must-fix corrections are implemented.
Changed numerical statements agree with the frozen independent Task40
evidence. The primary/secondary forecast distinction is now explicit:
the continuous forecast succeeds in 24/24 first-stage cases and 18/24
second-stage cases; it was secondary in the first stage, whose discrete
primary forecast succeeded in 23/24. This does not turn 42/48 into a pooled
primary-endpoint success rate.

I accept the F8 adjudication as the author's release gate and do not reopen
it as a finding. No public-release check or publication action was performed.
The GO verdict retains the scientific limits documented in Task40; it is
not a fresh validation of unchanged experiments or an approval of an unseen
later article.

## Remaining article recommendations

### F1. Use the arXiv record's primary category for reference 17 [should-fix, minor]

**Where:** `references.bib`, `ko2026attractor`, line 185; reference 17 on
PDF page 33.

**Claim:** `primaryClass = {cs.CL}`, rendered as
“arXiv:2606.30571 [cs.CL]”.

**Evidence:** The primary [arXiv record](https://arxiv.org/abs/2606.30571),
read directly on 20 October 2026, confirms the authors, title, year and
identifier. Its primary subject and prescribed citation use **cs.LG**;
cs.CL is a secondary subject. The rebuilt PDF visibly prints cs.CL.
The unverified COLM venue has correctly been removed.

**Correct value / wording:** Set `primaryClass = {cs.LG}`, or omit the
category. This is a small citation-metadata correction, not a scientific
or submission blocker. See [citation-40b.md](citation-40b.md).

### F2. Match the Fig. 25 caption to the corrected tracking description [should-fix, minor]

**Where:** `main.tex`, caption of `fig:delta2d`, lines 1343–1345; PDF page 27.

**Claim:** “The two leading directions of the monodromy were followed from
cell to cell over 852 continued orbits”.

**Evidence:** Task40's independent `delta-evidence.json` records 723
selections of the locally leading pair `(0,1)`, 63 of `(0,2)` and 66 of
`(1,2)`. At 129 cells the tracked pair is therefore not the two locally
largest-modulus directions. The revised body, lines 1437–1439, correctly
describes initialization from leading Ritz directions followed by overlap
tracking; Table VIII correctly limits the result to the sampled patch.

**Correct wording:** “A two-dimensional block, initialized from leading
Ritz directions and tracked by subspace overlap, was followed over 852
continued orbits.”

This aligns the caption with the correct body. The current “followed”
wording can refer to initialization, so this ambiguity is not a new
scientific blocker or a rejection of the trace result.

## Adjudication and additional-change coverage

The complete frozen `main.tex` and `references.bib` diff was read.
The detailed line-by-line disposition and exact reused evidence hashes are
in [revisions-40b.md](revisions-40b.md).

| Prior finding or new change | Outcome |
|---|---|
| F1 natural passages | Closed: complete-phase/channel definition, signed medians, single-phase 71% exception and Table IV agree with the independent evidence. |
| F2 cusp bracket range | Closed: all twelve points, extrema 0.1704770969 and 39.1082399960, correctly reported as 0.17–39.11. |
| F3 Soft Thinking | Closed: now explicitly a smooth full-distribution variant. |
| F4 synchronization summary | Closed: approximate growth prediction separated from start-, duration-, tolerance- and attractor-dependent locking. |
| F5 cascade reductions | Closed: 3.52, 6.54 and 3.93 from n=4, with the uncertain 5.26 separately qualified. |
| F6 fixed-start identity | Closed: nearest scan cells and two unresolved cells disclosed; no blanket same-orbit identity assertion. |
| F7 bounds/localization | Closed: 3.1e−13 and 2.3e−6 enclose the measured discrepancies; localization is explicitly model-based estimation. |
| F8 availability | Adjudication accepted; no new finding. |
| F9 “most” pairs | Closed: “some”/“include pairs” does not invent a denominator. |
| F10 basin/bifurcation logic | Closed: map properties distinguished from attractors visible to a chosen start. |
| F11 tracked block | Body/table closed; minor caption recommendation above. |
| F12 Backour assumptions | Closed: unbiased two-token reduction, linear logit transport and positive constant coupling stated. |
| F13 descent | Closed: identifies what the descent estimate alone establishes without claiming model-specific nonconvergence. |
| F14 Fujisaka–Yamada | Closed: framework attribution separated from the displayed derivation for this coupling. |
| F15 wording/layout | Stray “Near” removed; Fig. 22 and Fig. 25 now within one page of their first body mentions. Disclosed float exceptions are described below. |
| F17 sampled passages | Closed: 7.8 and 10.5 at the smallest offset; single-sample phases become rare, with 1/1/3 retained phases across offsets. |
| New 42/48 stage split | Correct in abstract/introduction; first-stage primary/secondary roles and failed all-rows primary criteria retained in the detailed account. |
| New breadth/depth sentence | Correctly narrows detailed analysis to 0.5B with a second cascade in 1.5B; adds no numerical result. |
| Ko–Geiping preprint citation | Correct preprint identity and removal of unverified venue; minor category correction above. |
| Figure source moves | Verified in diff and actual rebuilt pagination; source moves do not guarantee every display moves. |

## Revised bundle and all-page inspection

**PASS.** The exact revised source archive is
`/data2/neural-shrimp-data/arxiv/arxiv-source-e300c8e.tar.gz`,
10,728,088 bytes, SHA-256
`ea7fd829935a5612ee0a26419099c091a0780e23b5f7f2d16bded3cf6d8fa38e`.
All 33 members pass their content checks: frozen graphics/source or the
declared lexical comment normalization, the required compiler/top-level
declaration, and the bundled bibliography. The canonical build regenerates
the bibliography; it exactly matches the bundled bibliography.

The committed PDF, independent canonical build and independent bundle build
each have 34 pages and 9,244,681 bytes. All extracted layout text and all
34 rendered page images match exactly. Full PDF bytes also match after
normalizing exactly two date fields and one document-ID array; normalized
SHA-256 is
`bef49aa3cada06226dc41f3933c9ba1167e19370e11ebd595e68cdba61a24f78`.
No unresolved references/citations, missing or duplicate source labels,
final rerun requests or overfull boxes were found. There are 27 figures
and eight tables.

Root viewed every full-page render on pages 1–16; the independent layout
reviewer viewed every page on 17–34. No page was counted as inspected merely
because its hash matched. No new clipping, overlapping text, missing
graphics/glyph boxes or table overflow was observed. Figure 1 remains on
page 1.

Fig. 15 retains a two-page first-body-mention gap, page 15 to 17, around
the full-width Fig. 14. Fig. 19 retains page 20 to 22 around full-width
Fig. 18 and the stacked Fig. 19/20 display. These exceptions were disclosed
in the adjudication and have visible layout reasons. Fig. 22 is page 23
to 24 and Fig. 25 is page 26 to 27. The report does not claim every figure
is within one page.

Build source was reviewed and privately checkpointed at `a4d1969c` before
the single `bundle-40b-001` execution. It exited 0 without timeout;
child 1678578 was reaped. Parent launch-through-reap time was
61.897043 seconds, with one numerical thread, nice increment 10, 4 GiB
address space, 300 CPU seconds per child and a 650-second parent wall limit.
Details, page/figure ledger, exact receipts and output hashes:
[bundle-40b-review.md](bundle-40b-review.md).

## Zenodo follow-up, performed last

The specified three-file follow-up addresses the prior two deposit
must-fix items. The README now disclaims portable replay, distinguishes
the two checker routes and separate token/basin releases, correctly states
the coexistence attraction powers, and declares the external tokenizer.
The added supervisor's bytes exactly match the missing historical SHA-256.

The new checksum listing has 35 entries: one added supervisor, one changed
existing README hash, and unchanged declarations for all 16 archives.
The two small content-file hashes were freshly checked. Archive bytes were
not restreamed; their previous authenticated hashes were reused.

One early README sentence still says “every evidence hash” is in the
release records, whereas its new detailed-inventory explanation correctly
points onward to receipts/manifests. This is a minor package-only
consistency recommendation, **Z1**, with replacement wording in
[package-40b-review.md](package-40b-review.md). It does not alter the article
verdict. Existing privacy and transitive-resolution limits remain explicit.

## Completed and limited checks

**Completed:** all requested revision dispositions, changed numbers against
the existing independent evidence, stage/endpoint counts, scope correction,
primary-record verification of the changed citation, exact revised bundle
build/content equivalence, all-page visual inspection and figure placement,
and the three specified Zenodo small-file changes.

**Reused rather than rerun:** Task40's raw-data reductions, original
source/execution authentication, numerical and certificate evidence,
and unchanged primary-literature checks. The earlier partial coverage,
including nonlinear free-exponent fits and complete same-parameter
fixed-start orbit identity, remains partial.

**Not performed:** new model/derivative/trajectory/certificate calculations,
large Zenodo archive rescans, complete recursive dependency resolution,
exhaustive privacy or license clearance, public landing-page validation,
article edits, public release or submission. No paid compute or GPU work
was used. All new review files are confined to the authorized Task40 folder;
build scratch is in its authorized external scratch tree.
