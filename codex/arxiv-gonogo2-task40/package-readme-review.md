# Package README semantic review

Verdict: **README corrections required before presenting this as an executable reproduction guide.** The stated local orbit conclusions agree with the two named release records. This review did not scan archives, authenticate packaged members, replay witnesses, evaluate a model, or re-establish any certificate.

Reviewed README: `/data2/neural-shrimp-data/zenodo/neural-shrimp-proofs-v1/README.md`, 5,104 bytes, SHA256 `75eedfdeca4240957d0cfabf616ddb39205aaf3c24af5059df9459453b7f05d8`.

Exact current records, whose hashes match the README:

- `certificates/llm-p8-certificate-final-001.json`: `539ead0d344f0eb3e0b91f55999da1c6cd76f801d62e274b4537241018949f97`.
- `codex/coexistence-task33/pair-certificate-release.json`: `eb1e2238f2607408ddcbb31383c6a7e4c1c25f0803f5d9a5af90769a196fcf97`.

## R1 — must-fix: the described reproduction route is not portable or complete

**README lines 64–71:** “Extract each set of parts into its own directory”; “Run the independent checker from the code archive on the extracted evidence”.

The named P8 wrapper, `codex/certify-llm/final-assembly-002-witness-audit.py`, hardcodes the original `/data2/neural-shrimp-data/codex/llm-p8-certificate` evidence root and a historical `/home/d7082791602/.codex/worktrees/llm-p8-certificate/neural-shrimp` checker path (lines 7–10). Its CLI requires `--output` and `--script-sha256`, requires one literal original output path, and creates that output exclusively (lines 158–165). If the complete evidence directory contains the original audit output, that output is already occupied. Merely extracting to an arbitrary local directory and conceptually mapping the paths does not make this wrapper runnable.

`witness_audit.py` is a Python API, `audit(...)` at line 94, not a command-line reader. Its attraction checker requires saved powers `[1,2]` (line 190). The separate coexistence higher-power check uses `codex/coexistence-task33/power_witness.py`; `higher_power.py:47` invokes it after producing the saved power enclosures. The P8 wrapper alone therefore does not reproduce the higher-power coexistence check. `release_pair.py` also has historical absolute paths and writes release records; it should not be presented as a read-only portable verifier.

**Proposed correction:** identify the P8 scalar audit and coexistence higher-power scalar audit separately. Supply an actually supported saved-only command/API example, required input records and path mappings, dependencies/resources, and a fresh output allocation, or state candidly that this deposit contains the historical execution code and saved audit results but does not yet provide a portable one-command replay. Any later portability adapter must preserve original evidence and identity checks. This review does not authorize or execute such an adapter.

Suggested immediate wording:

> The archive preserves the historical verification programs and their original path bindings. The P8 witness wrapper is not directly relocatable: it requires the recorded evidence layout, checker path, script hash and an unused fixed output allocation. The coexistence power witnesses use a separate scalar checker. The instructions above verify downloaded bytes; a portable saved-witness replay procedure is not yet supplied here.

## R2 — should-fix: distinguish the two powers from the two orbits

**README lines 39–40:** “the fourth and eighth powers for the coexistence pair”.

Both entries of `pair-certificate-release.json` have `witness_power: 4`; both also record successful eighth-power bounds. The released rounded-up fourth/eighth bounds are respectively `0.000750081` / `0.000000019` for P2 and `0.341675995` / `0.001090443` for P3. `power-pair-001-review.md` explicitly says powers 4 and 8 were evaluated for **both** orbits. The README is compatible with that reading but can be mistaken for a fourth-power P2 / eighth-power P3 pairing.

**Replacement:**

> For each coexistence orbit, the fourth power already supplies a strict attraction witness; successful eighth-power bounds are also retained. The monodromies are strongly nonnormal.

This is clarification, not a failed attraction gate or a new numerical check.

## R3 — should-fix: distinguish these two releases from the article's separate token/basin releases

**README lines 3–4, 21–31 and 46–47:** broad reference to “the computer-assisted proofs” in the article, followed by two orbit releases and a “token labels” exclusion.

The token-label exclusion is correct for the two named orbit releases. It should not be read as an exclusion from all subsequently released results. Two separate current authorities exist:

- `codex/coexistence-task33/token-labels-release.json`, SHA256 `e586c0e4cefb599186d3cd5a717efaefd78d009488cbecd87e12f9e792b6822f`: strict all-vocabulary argmax margins over containing receiver boxes for the P2, P3 and P8 phases; no fixed-start reachability claim.
- `codex/coexistence-task33/inner-basin-release.json`, SHA256 `ae5e66a2078535b0dd1817df949d8958614e128d4886a4dbd180aa5bffc8db9e`: every real state in the stated phase-zero infinity-norm ball converges to the corresponding coexistence orbit, with radii `2^-681` and `2^-700`. Its scope excludes fixed-start reachability, large basin volume, whole original shooting-box invariance, global geometry and ordinary generation.

**Proposed correction:** explicitly say “The two orbit-release records summarized here do not themselves certify token labels or fixed-start reachability.” If the deposit is intended to support the article's separate token-label and explicit inner-basin claims, list these additional release records and their hashes and include them in the package-reference coverage check. Otherwise state that the README/package validation scope is limited to the two orbit releases. Do not infer that the extra records or all their dependencies are packaged: that remains the archive auditor's task.

## R4 — should-fix: direct release references are not the complete evidence inventory

**README line 18 and lines 67–69:** “every evidence hash” / “The release records give every evidence file with its SHA-256”.

The release records directly pin selected definitions, original receipts, reports, audits and review authorities. The detailed derivative and source inventories are reached through those records. For example, the P8 release reports 112 original shards but does not individually enumerate all 112 shard files in its top-level direct references. The coexistence record also includes an absolute historical worktree review-binding path, outside the two raw roots shown in the extraction paragraph.

**Replacement:**

> The release records pin the principal evidence and authority records. Their referenced receipts and manifests contain the detailed inventories. Resolve raw-evidence paths against the corresponding extracted evidence tree and repository/worktree-relative review paths against the extracted code tree; preserve all recorded hashes.

Package completeness requires the intended dependency coverage to be specified. Checking only direct release references must not be described as checking every transitive evidence file.

## Checked-consistent wording and remaining limits

- Dimension 896, vocabulary 151,936, fixed 37 prefix IDs, steering layer 12, and the three model/config/vector hashes agree with both map records. The stated exact-binary64 parameter convention agrees with the recorded rational interpretation. For mathematical typography, “the binary64 value denoted by 0.071674” is clearer than treating the short decimal itself as exactly equal to the displayed dyadic rational.
- The P8 statement correctly restricts uniqueness to the ordered shooting root in the product of eight phase boxes, with local attraction and radius `2^-640`; it does not assert uniqueness of all `F^8` fixed points in the first box.
- The pair release states coexisting primitive P2/P3 ordered shooting roots and local attraction at the same parameter pair. Its map ratios agree with the README's binary64 decimal convention. The pair's `2^-640` domain-radius statement is not independently rederived here from its raw domain definitions.
- Trusted Python/FLINT/Arb execution, scalar recombination rather than a second dense-product/model replay, and preservation of failed/superseded attempts are appropriately distinguished. The README should name the separate coexistence checker as in R1.
- The README's archive-completeness, archived member identity, license-file presence, extraction-root layout, and checksum-list completeness assertions remain for the package scan. No large archive was opened in this review. No repository publicity, DOI publication, release action or model-license analysis is established by this review.

Read-only inspection limitations: one overly broad display was truncated, so relevant scope fields and the short power review were subsequently read in focused form. An attempted scoped search named nonexistent `power_pair.py` and returned an error; the actual `higher_power.py` / `power_witness.py` names were then resolved with `rg --files`. A later radius search returned no matches; the corresponding raw-domain check remains explicitly unperformed. No checker, test, source import, proof arithmetic, target array or model execution occurred.
