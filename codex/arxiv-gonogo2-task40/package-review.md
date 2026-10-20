# Task40 Zenodo addendum — audited after the article report

**Checksums and direct release closure PASS. Deposit documentation/dependency
fixes remain.** Five package findings follow: two must-fix before describing
the deposit as a complete executable reproduction package, three should-fix.
These are separate from the nine article corrections already delivered.
The human expressly prioritized arXiv over Zenodo; no package repair or
publication was performed.

Package:
`/data2/neural-shrimp-data/zenodo/neural-shrimp-proofs-v1/`.
The article report was delivered at private commit
`ecc660eac7dd45eb9416bb278553592554b5278a`, before this scan.

## What passed

- **34/34 listed file checksums**, totaling **26,737,110,374 bytes**. No
  missing or unlisted top-level files, apart from the checksum file itself.
  All eight P8 and seven coexistence parts, the code archive, their listings,
  README and steering assets are present.
- **73,945 regular archive members**, totaling **26,669,204,772 bytes**,
  individually SHA-256 hashed. No duplicate canonical names, unsafe names,
  links or other nonregular file members. Directories were allowed.
- The two packaged release records match their independently pinned
  identities: P8 `539ead0d…`, coexistence `eb1e2238…`.
  **All 34 direct path/hash references and five named inherited P8 reviews
  match**—39 reference entries covering 37 unique packaged files, including
  sizes wherever the reference specifies one.
- **738/738 code-archive files match Git commit
  `fcf2d462`** byte-for-byte, checked by hashing their exact Git blobs.
- The additional token-label and inner-basin release records are packaged
  with the expected hashes. All **12/12 inner-basin direct references**
  match. For the token-label release, **34/35 direct references** match;
  the remaining reference is an external tokenizer asset described under
  P2 below.
- Text screening covered **22,591,778,055 bytes**. There were **zero
  matches** for the tested private-key headers, GitHub/service-token
  shapes, long credential assignments, or email-address patterns.
  **4,666 files** match the local home-directory-path pattern.
  This is a bounded pattern screen, not exhaustive secrecy/privacy clearance.

## P1. The advertised saved-witness replay route is not portable [must-fix before deposit]

The README's instruction to extract anywhere and run the named independent
checker is not an executable recipe. The P8 wrapper hardcodes original
evidence and historical worktree paths, requires a script hash and one
literal output allocation, and creates that output exclusively. The
complete historical evidence already contains the old output. The
`witness_audit.py` file supplies an API rather than a portable command.
The coexistence higher-power check uses a separate `power_witness.py`
checker, not the P8 wrapper.

Correct the README to identify the two checker routes and their exact
dependencies/path/output requirements. Either supply a separately reviewed
portable saved-only adapter later, or say that the deposit preserves
historical programs and authenticated results but does not yet provide
a portable replay procedure. Do not relax or overwrite historical bindings
just to make the old programs run.

Exact source findings and proposed immediate wording are in
[package-readme-review.md](package-readme-review.md), R1. No checker or
archived code was executed in this audit.

## P2. Declare and supply the missing shared supervisor; identify external tokenizer data [must-fix before deposit]

The direct release checks pass, but a reachable dependency investigation
finds eight references to one absent code artifact:

`codex/certify-chaos/supervise.py`, SHA-256
`306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44`.

That digest is absent from **all** individually hashed regular package
members. The references occur in the packaged P2/P3 refinement cards/plans,
prefix plan, and higher-power plan. Include the exact historical source
as an additional versioned package artifact, or expressly state that the
deposit is not a complete original execution environment. This does not
invalidate the unchanged original proof or its direct receipt hashes.

The packaged token-label release additionally pins external `tokenizer.json`,
7,031,645 bytes, SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
It is not packaged. The README already declares external model weights and
configuration, but should also identify the tokenizer dependency when
claiming coverage of token labels. Document the checkpoint revision and
required hashes; no model/license/publication action is authorized here.

The supplementary metadata check did not read the original tokenizer or
supervisor bytes; it compared their declared hashes with the complete
package-member hash inventory.

## P3. Review local account-path metadata before public deposit [should-fix]

The **4,666 candidate filenames** are listed in
[package-risk-files.json](package-risk-files.json), with categories only.
Matching path contents are not reproduced in this report. These paths
occur in historical source/provenance records and may expose local account
or machine-layout information.

The author should decide whether this metadata is appropriate for release.
If sanitized publication copies are desired, preserve immutable originals
and clearly version/hash the derived copies and mapping; silently editing
hash-bound evidence would break its provenance.

No listed key/token/email pattern matched. This neither proves that
every possible credential format was covered nor screens arbitrary names
and free-form personal information exhaustively. No general publication
clearance is granted.

## P4. Clarify attraction powers for the coexistence pair [should-fix]

Both P2 and P3 orbit releases use **power four** as their attraction
witness; both also retain successful power-eight bounds. The README's
“fourth and eighth powers for the coexistence pair” can be read as
assigning a different power to each orbit.

Use: “For each coexistence orbit the fourth power already supplies a
strict attraction witness; successful eighth-power bounds are also
retained.” This is a documentation clarification based on the frozen
release records, not new proof arithmetic.

## P5. Separate orbit releases, later token/basin releases and detailed inventories [should-fix]

The two orbit releases do not themselves certify token labels or the
explicit inner-basin neighborhoods. Separate packaged records do:

- `codex/coexistence-task33/token-labels-release.json`, SHA-256
  `e586c0e4cefb599186d3cd5a717efaefd78d009488cbecd87e12f9e792b6822f`.
- `codex/coexistence-task33/inner-basin-release.json`, SHA-256
  `ae5e66a2078535b0dd1817df949d8958614e128d4886a4dbd180aa5bffc8db9e`.

Name their distinct scopes in the README if the deposit is presented as
supporting those article claims. None establishes the designated fixed
start's reachability or a large basin.

Replace “the release records give every evidence file” with: “The release
records pin principal evidence and authority records; referenced receipts
and manifests contain the detailed inventories.” Explain mapping code-tree
and historical worktree references as well as raw-data roots.

## Deeper reference analysis: explicit limits

The original scan's limited path classifier followed 1,020 matching
reference edges and visited 831 files/small JSON objects. It recorded 18
unavailable mapped-path edges, two mapped-path hash mismatches and 1,356
unmapped edges. These counts are **not** counts of missing scientific data.

A separate metadata-only join against all member digests found:

| Initial category | Edges | Edges with no matching digest anywhere |
|---|---:|---:|
| Mapped path absent | 18 | 8, all the single supervisor artifact in P2 |
| Mapped current path differs from historical hash | 2 | 0 |
| Unmapped by limited path classifier | 1,356 | 0 |

The two historical README/status versions exist elsewhere in archived
source snapshots. Many unmapped references are simple owner-relative
manifest names. A matching digest somewhere establishes availability of
those bytes; it does **not** establish correct path resolution, replay
semantics, or full transitive closure.

The scan parsed 37,777 small JSON files with no parse errors, while
recording path/reference schema exceptions in 78 files. Another 371
larger JSON files were fully hashed and text-screened but not structurally
parsed for nested references. No claim is made to have reconstructed
every nested dependency, compared every human-readable listing row, or
rerun the mathematical proof.

## Execution and exact evidence

Reviewed source `package_audit.py`: 19,202 bytes, SHA-256
`2ca798998528c597f767bcd58219f76852cd45cce5f98fe6d32fdb831c5ed07a`,
privately frozen at `fe0b376db4d4ab11a21ba6c1fedda99916d6b049`.
Both static reviews preceded execution; their identified source gaps were
fixed before the first scan.
Independent final metadata consistency review passed without rereading the
archives or repeating proof arithmetic.

The single `package-001` run began at
2026-10-20T03:47:37.531511Z, exited 0 without timeout, and its direct
child 1646210 was reaped. Parent launch-through-reap time was
273.34316611336544 seconds; internal scan time 273.2290985058062 seconds.
Limits were one CPU thread, nice increment 10, 2 GiB address space,
1200 CPU / 1800 parent-wall seconds and 256 MiB per output. No extraction,
GPU/model work, paid compute, archived-program execution or publication.

| Evidence | SHA-256 |
|---|---|
| `package-evidence.json` | `d082766cd43d5d0bca3d97bca3e1d6ef8404ddf2684ad84e3fed1c03064e6af0` |
| `package-metadata-joins.json` | `94db41290b083e49dc5ca07d4761e02d414b16aceb542603bc4f10ad3e3e3048` |
| `package-risk-files.json` | `af70f90879e8ea1554899b2b0fec9ccd6b355537a1dcc7f438fa1b51e8079141` |
| Scratch `package-member-manifest.json` | `e8d3475d0e0b95e43deb599ac92be54f811fb223e0f15065cbb99081461684b1` |

The 28,059,254-byte member manifest is kept at
`/data2/neural-shrimp-data/scratch/audit-codex-1020/package-member-manifest.json`.
The parent receipt and stdout/stderr logs are in the Task40 directory.
Original archives and prior certificates remain unchanged.

**Completed:** outer/member hashes, exact frozen direct releases/reviews,
code-export identity, additional direct token/basin joins, README semantic
review and bounded text-pattern screening. **Partial:** recursive dependency
resolution and exhaustive privacy assessment. **Not performed:** proof
replay, original execution recreation, license/publication authorization,
repository visibility changes, Zenodo upload or arXiv submission.
