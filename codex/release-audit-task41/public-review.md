# Task41 Part C: action needed

This review covers `arxiv-v1` at `1314528118ac1300c2cffd34fb60cfbfc67d2c82`, frozen postcleanup main `21272a7830d74fade1f6592af2a7b8189370ffad`, and both versions of the 32 paths changed by cleanup `e359312e339c8f2fda59f280e6d7f92d88c0552d`. It does not clear later commits or the entire reachable history. No secret, monetary value, private remark or benchmark question is reproduced here.

## C1 — must-fix: monetary fields remain in both public target trees

**Confirmed category:** nonzero monetary literals in operational records, not mathematical notation. **HEAD status:** present; the listed blobs are also in the tag. The numeric-key check and exact positive-location inspection identify 22 such lines across six documentation/receipt files, excluding synthetic test code:

| Path | Lines | Blob |
| --- | --- | --- |
| `docs/EXP-013-local-review.json` | 85 | `df90b6628e3cdc93c13c23fe4a483af2d147fcce` |
| `docs/EXP-016-local-review.json` | 54 | `066d60a5edf37e5cf73fbd1ecc9c78ab2869507c` |
| `docs/consults/2026-09-17-pro-receipt.json` | 61, 63, 64, 86, 87, 88, 97, 136 | `65b9db95f600e8a0f6bd7367ef566e751248adbc` |
| `docs/consults/2026-09-17-receipt.json` | 10, 45 | `f7d9c67af8d573e533904d63ef5308eceecf5356` |
| `docs/consults/2026-09-22-pro-receipt.json` | 33, 46, 49, 52, 64 | `0f77dbf5827f4d27a207adea59d7d73835035146` |
| `docs/consults/2026-09-22-receipt.json` | 14, 15, 20, 21, 23 | `62c5995a204e7ba6ccfe267d8c5e67c8755dbd06` |

The completed masked semantic read additionally confirms 49 current textual project monetary references, including allocations, estimates, rates and zero-spend statements, across prose, source docstrings and copied transcripts. These are location rows, not 49 distinct spending incidents. Representative current locations are `comms/claude-to-codex.md:329` (blob `9312df7135c0d647d9b9e549b6fac79debaebde6`) and `docs/EXP-013-operational-amendment.md:63` (blob `a38855022f781b5c22701dcac37985f0d296d634`). A further nonzero currency-key field is confirmed at `docs/consults/2026-09-17-pro-receipt.json:68`, beyond the original 22-line subset. The original compact scan receipt is frozen; the additional adjudications are in `public-semantic-supplement.json`. The 22-line subset remains sufficient to establish the policy breach and is not an exhaustive count.

**Recommendation:** prepare a disclosed forward correction of public-facing records and a curated release policy for immutable evidence. Preserve the originals and their provenance under the research policy. A new HEAD edit does not remove the original tag/history copies. Do not rewrite history, move the tag or claim historical erasure. This review made no repository-content corrections.

## C2 — should-fix: explicit GSM8K attribution/notice is missing from the inspected distribution boundary

**Category:** third-party dataset provenance. **HEAD/tag location:** `docs/pi-gsm8k-test.json:1`, blob `ceea115650543e36ee98004e30a5ede6f6a8e7d5`. The file contains 1,319 `question`/`gold`/`index` records. Its note identifies GSM8K/OpenAI, but the file does not contain the MIT notice, the upstream copyright notice or permission notice. The tracked license/notice filename inventory contains only `LICENSE` and `LICENSE-docs.md`; those provide the repository's own MIT notice and a general third-party exception, respectively. This is a checked omission at those specific boundaries, not a full legal determination about every text in the repository.

The [primary GSM8K repository](https://github.com/openai/grade-school-math) distributes its [MIT license and upstream notice](https://github.com/openai/grade-school-math/blob/master/LICENSE), which calls for retaining the copyright and permission notice with copies or substantial portions.

**Recommendation:** add an explicit dataset provenance entry, upstream URL and retained upstream license/notice, with the third-party exception clearly applied to this file. Benchmark word-problem amounts are dataset content, not evidence about the author's finances. If the release policy excludes all literal monetary examples, resolve that through a disclosed upstream-fetch/data-packaging boundary; do not silently rewrite benchmark prompts or frozen evidence.

## C3 — note: CI has a restrained trigger/permission configuration; optional hardening remains

**Location:** `.github/workflows/tests.yml`, same blob `304d0a272a8c010bf81e639100e7de7b3d1939c8` in both target trees. Direct source inspection confirmed manual dispatch at line 3, `contents: read` at lines 4–5, hash-checked dependency installation, and a specific temporary diagnostic artifact upload. No secret reference, embedded credential literal, pull-request trigger, untrusted-head checkout or environment dump was identified in that workflow. This is a static configuration review; no hosted workflow or repository settings were exercised.

**Recommendation:** optionally pin the checkout/setup/upload actions to reviewed commit IDs and explicitly disable checkout credential persistence. Their current major-version references and omitted persistence override are hardening opportunities, not a demonstrated exploit or a must-fix finding.

## C4 — should-fix: cleanup changed a nonmonetary reproduction-command argument

**Location:** `docs/research-log.md:18856`. **HEAD status:** present. The tag/cleanup-before blob `44bcec2f90b4072c3e4fb6ffa50e16c2aa155fc1` passes a shell positional argument to the ridge option. The frozen HEAD blob `c8c357056690cfc0a43d86bc7a357da19bd309b3` substitutes an amount-redaction marker in that position. This is shell syntax, not a monetary amount; the substitution damages the recorded reproduction command.

**Recommendation:** restore only that original nonmonetary shell argument in a disclosed forward correction using the preserved original blob. Retain genuine monetary redactions and all original history/tag identities. No command was executed to check this source-text discrepancy.

## Screening, triage and preserved limits

The follow-up ran once after public checkpoint `f2988f7f`, included in remotely verified main `79e5f774fe94ac49da4bcabf4d528abec85d6d25`. It exited 0 after 87.003438 seconds of script time under the declared one-thread/nice-10/2-GiB/300-CPU-second/600-wall-second bounds. The parent reaped the launcher; the three exact owned process identities and their dedicated group were absent afterward. The host `ps` implementation rejected one group-query option; a supported exact-group query then confirmed absence. This inspection failure is retained. Shared-session closure is not claimed.

Both frozen trees contain 28,684 entries. The combined target/cleanup pass assigned every one of 26,236 unique requested blobs a disposition: 5,539 distinct text blobs screened, 20,689 binary blobs inventoried, and eight oversized blobs skipped. Each target has 5,806 screened text paths, 22,870 binary paths and eight oversized paths. Reads totaled 215,768,591 bytes. Separately, Root's closed seven-large-text supplement screened all 114,608,770 bytes of the seven oversized JSON/JSONL blobs with strict UTF-8 and exact Git/hash checks, reporting no additional category candidates. Its exit-0 result and timing/observation limits are in `public-large-text-result.md`; its 5,084-byte evidence SHA256 is `7a4fa1691ea1e43e5992870601845c86bcd46afd5730382e7984f92f787f872d`. This separate result closes the seven text gaps without changing the frozen first receipt; the remaining oversized PDF belongs to the binary scope.

The scanner emitted 362 candidate location rows in 184 groups. The first triage assigned context features, Git object types and Python AST classifications; that pass alone was not a completed semantic read. The separate completion pass then read the exact requested current contexts with values masked. The six bearer-pattern rows across the tag/cleanup/HEAD versions resolve to Git commit references. Twelve apparent email rows are matrix multiplication syntax; 24 are retained Git identity metadata. The remaining contact candidate is a configured identity in a temporary test repository, not a disclosed private contact. The two off-limits project mentions are access-restriction instructions, not evidence of copied project contents. No named forbidden project directory or env-family payload was opened.

All 41 previously unresolved current rows now have a semantic disposition: 29 shell-parameter rows, eight project monetary references, and one each concerning retired research counts, a mathematical variable, numeric-parser syntax and fictional benchmark arithmetic. None remains ambiguous after that exact-context read. All 26 current personal-context rows were also read: four bibliographic, one process-family, eight source-code, six research/results, six generated-model-output, and one benchmark-content row. None of these reviewed current candidates is a confirmed private personal remark. This is a statement about those locations, not a whole-repository privacy guarantee. Thirty-five zero-valued currency-key metadata rows and synthetic test examples remain separately classified. The earlier feature-only records and their unresolved labels are preserved, with the completion explicitly recorded in the new supplement.

The separate PDF supplement subsequently checked all 27 exact objects: 25 figure files and two project manuscript paths. It extracted metadata and text from all 241 pages, with no empty extracted pages or new heuristic candidates. The one invocation exited 0 in 2.60 seconds; its scope and hashes are in `public-pdf-inventory-result.md`. These path groups and successful text checks do not prove ownership or exclude third-party copied images. Model-weight extensions were not identified in the targeted filename inventory, but this is not an audit of every binary/archive payload or model license. No plagiarism, complete redistribution-license clearance, arbitrary encoded-secret detection, credential validity test or complete historical-secret absence is claimed. The public source and release JSONs are also distinct from complete external proof assets.

The first scan remains failed and immutable: it hit its 100,000-candidate cap, predominantly on mathematical notation, before its final coverage/cleanup/commit-message stages. Its exact output is preserved under scratch and bound by `public-audit-attempt001-receipt.json`. It is not counted as a completed history audit.

## Evidence

- `public-findings.json`: compact successful follow-up receipt, grouped candidate locations and exact oversized-path list.
- `public-candidate-triage.json` and `public-semantic-triage.json`: value-free positive-location features and classification counts; unresolved categories remain explicit.
- `public-semantic-supplement.json`: completed exact-context adjudication of the requested current candidates, additional confirmed monetary locations, and C4's original/current source join. It supersedes the earlier unresolved current labels without changing the frozen receipt.
- `public-pdf-inventory-result.md` and `public-pdf-inventory-evidence.json`: the separate completed metadata/text check of all 27 PDFs; no OCR, embedded-content or ownership clearance.
- Scratch `public-findings-followup002-full.json`: 6,734,537 bytes, SHA256 `d25d6fd7b1be34479becf64fd61f56a9304b2a624356639d51b461f7377f0b2e`.
- Scratch `public-candidates-followup002.jsonl`: 126,011 bytes, SHA256 `fc943a4d6243447f58e6698e693b134dc10cc82a60c07d6213cdc9e77d4af78b`.
- Frozen follow-up source SHA256 `7ebd63c331f1f76a888a1b627b8228dc2fb6c998756b34c2cc75f3f810de1819`.

Part C is **ACTION NEEDED** because C1 is confirmed. C2 needs a concrete provenance improvement, and C4 needs a narrow forward repair. Unflagged content, historical coverage and the binary/provenance limits still prevent a blanket clean-release assurance.
