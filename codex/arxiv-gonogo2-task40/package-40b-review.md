# Task40b — revised Zenodo small-file follow-up

**The two prior deposit must-fix items are addressed. One minor README
consistency edit remains.** This check followed the article-source review
and successful revised-bundle build. It read only the three specified
small files; the original archive audit was not repeated.

At 2026-10-20T04:21:46Z, all three files were regular no-follow opens with
stable size/inode/mtime during their reads:

| File | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 7,825 | `e131c4d3f207fdb3cbac0668cea5043a3e75a2b4febd1d9c29586008d12d73b0` |
| `SHA256SUMS.txt` | 3,561 | `9e1fe1a58f2aed8676d694a241bf836567a9035e604a63f8319916204f39c206` |
| `codex-certify-chaos-supervise.py` | 8,727 | `306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44` |

The checksum listing now contains 35 entries. Its sole added entry is
the supervisor; its sole changed existing digest is README. Both freshly
read file digests match the new listing. All 16 archive digest declarations
are identical to Task40's independently computed digests. This last
comparison authenticates the declarations, not a fresh read of archive bytes.
Exact machine-readable evidence: [package-40b-evidence.json](package-40b-evidence.json).

## Disposition of the prior package findings

- **P1 addressed:** the README distinguishes the P8 API/wrapper from the
  coexistence power checker and explicitly says the preserved programs do
  not yet offer a portable replay procedure. It describes their historical
  paths, hashes and exclusive old output. No portability claim remains.
- **P2 addressed:** the added supervisor exactly matches the previously
  absent historical digest. The external tokenizer is now identified with
  checkpoint revision, exact size and digest. Including a separately named
  supervisor supplies the missing bytes; it does not make historical paths
  work automatically.
- **P3 remains an author publication decision:** local paths are expressly
  acknowledged and preserved. Task40's 4,666 candidates were not rescanned
  or granted blanket privacy clearance.
- **P4 addressed:** fourth powers are correctly stated to suffice for both
  coexistence orbits, with successful eighth powers also retained.
- **P5 substantively addressed:** separate token-label and inner-basin
  releases are named with correct hashes and scope; the detailed-inventory
  distinction is correct in “How to check.” One earlier line still conflicts
  with that distinction, as below.

## Z1. Align the early inventory sentence with the later explanation [should-fix, minor]

**Where:** revised package `README.md`, “The map,” third bullet.

**Claim:** “Prefix token IDs, exact parameters and every evidence hash:
in the release records (code archive).”

**Evidence:** Task40's direct-reference and deeper-reference audit separates
the principal release references from the inventories reached through
receipts/manifests. The new “How to check,” item 3, makes this distinction
correctly. The earlier “every” sentence was left unchanged.

**Correct wording:** “The release records give prefix token IDs, exact
parameters and principal evidence hashes; referenced receipts and manifests
give the detailed inventories.”

This is an internal documentation consistency edit, not a missing direct
proof record or a reason to delay the article review.

## Limits

No archive was rehashed, extracted or executed in this follow-up. No proof
arithmetic, full recursive reference-resolution check, license assessment,
exhaustive privacy screen, public landing-page validation or publication was
performed. Original Task40 evidence and its stated limits remain intact.
