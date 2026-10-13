# Task8 Part B: independent frozen saved-row reproduction

13 October 2026. The independently implemented statistical replay reproduced all
81 compared frozen-producer result leaves exactly, using all 800 retained records
(200 GSM8K questions × four arms). No schema discrepancy or nonzero numerical
difference occurred, including within the declared absolute tolerance of 1e-12.
Admission, operational completion, field agreement and scientific gates are
separately recorded. The frozen joint B1/B3 criterion passes. The final independent
saved-evidence review is recorded separately; the original completion receipt is
immutable and retains its initial pending-review field.

## Scientific results

Flexible accuracy at λ=1.0,1.05,1.15,1.20 is 33.0%,39.5%,25.5%,9.5%.
Strict accuracy is 27.5%,34.0%,25.0%,9.0%. Differences and paired percentile
95% intervals below are percentage points relative to λ=1.0; all contrasts and
all diagnostics are retained.

| Scoring | Arm | Accuracy | Difference | 95% interval | Lost / gained |
|---|---:|---:|---:|---:|---:|
| Flexible | 1.05 | 39.5% | +6.5 | [1.5,12.0] | 8 / 21 |
| Flexible | 1.15 | 25.5% | −7.5 | [−13.0,−1.5] | 26 / 11 |
| Flexible | 1.20 | 9.5% | −23.5 | [−30.0,−17.5] | 50 / 3 |
| Strict | 1.05 | 34.0% | +6.5 | [1.5,12.0] | 8 / 21 |
| Strict | 1.15 | 25.0% | −2.5 | [−8.5,3.5] | 23 / 18 |
| Strict | 1.20 | 9.0% | −18.5 | [−25.0,−12.5] | 43 / 6 |

B1 is the predeclared flexible-score decline at 1.20: paired one-sided exact
binomial p=2.759792394613214e-12. B3 is **noninferiority**, with a tolerated loss
margin of 4 percentage points at 1.05: the one-sided 95% bootstrap lower bound is
+2.5 points, above −4 points. Both pass, so their frozen intersection-union
criterion passes. The observed +6.5-point improvement and its two-sided interval
are reported, without changing the primary noninferiority estimand. B2 at 1.15
is descriptive; its strict-score interval crosses zero. The paired exact
binomial decline p-values at flexible 1.15 and strict 1.15/1.20 are respectively
0.010036925959866494,0.26635462761805684,2.8638826421456542e-8.

All arms used the frozen finite generation setup: 32 recurrent steps, cot8,
maximum 256 new tokens, question index as seed, and all indices 0–199. Bootstrap
sampling follows the frozen 10,000-draw PCG64/seed0 contract with the original
per-contrast resets and linear percentile interpolation. The fixed generation
budget is part of this result. This establishes neither a compute-saving claim
nor a causal explanation of the dynamical mechanism.

| Arm | Token-cap rate | Median new tokens | Strict-format share |
|---|---:|---:|---:|
| 1.0 | 4.5% | 87.5 | 81.5% |
| 1.05 | 5.0% | 87.0 | 86.5% |
| 1.15 | 2.0% | 75.0 | 95.5% |
| 1.20 | 7.5% | 70.0 | 88.0% |

## Execution and authority

Four scientific source files are bound to commit
`06ef341522537c93bd0afb30dbcf922d27299bfa`; frozen producer design/analysis is
`9805ed32055e7ef790df0cc67316357c1067aef0`. Private prelaunch checkpoint is
`ab43131ca1f483b91e2f517b614dd0fd316d10c9`, card SHA256
`1e02f39850ac3dfb13ded83f772b784774ddbe598e74b108aac1b4b8b2775e7e`.
One CPU worker plus supervisor ran under nice10, 4GiB AS and the external
300-second watchdog, with 20/16GiB startup/runtime available-memory floors.
The command returned exit0 in 0.770594625s; driver work took 0.597997678s,
peak72,472KiB, with no warnings. Original worker817440/start186419809 and whole
process group817439 are absent; the supervisor returned and was reaped. No
retry occurred and the auxiliary reservation is released.

Input002 SHA256
`4de42764f539de11ec087b16d75b51c4eff826aeab3b9f5df3a324d69924dc60`
binds 51 files/1,625,387 bytes. Authentication checks 94 original receipt
assertions, 11 members across the three original archives, all input hashes,
nine exact Git blobs, 800 original/retained diagnostic records and 81 saved
comparison leaves. No producer analysis, model, bootstrap or statistical reducer
was executed by the saved-evidence audit.

The finite provenance basis is authenticated local retrieval/recorded termination,
reviewed pinned source, and the producer's explicit exact-command/data-path
attestation. Current dataset export hashes and retrospective archive-size
reconstruction support that basis. **Historical upload-content hashes and remote
model-coefficient byte attestations are absent.** The producer's 800-gold-answer
comparison was not independently replayed. Intended model/tokenizer revision is
`bb6621b65e90b6a4b9b29ef88dc83866d450470c`; the current GSM8K test export hash is
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

The original producer0–95 attempt's 130-minute job status124 remains intact;
its retrieval wrapper's status0 is distinct. The separate96–99 completion and
100–199 partition complete the frozen 200-question cohort. Producer summary
exposure before this independent replay is disclosed (17:58 announcement and
earlier committed summary); no cohort, threshold or estimator was retuned.
This reproduces saved-row statistics under the stated trust basis, not an
independent model-generation rerun or cryptographic historical asset attestation.
Part A is still pending; no partial Part A outcomes or pods were accessed.

## Immutable evidence and audit entrypoint

All paths below are under
`/data2/neural-shrimp-data/codex/huginn-independent-replay/`:

- `part-B-001/{independent,producer,comparison,admission,status}.json`: complete
  results, all 800 diagnostic rows, all comparison leaves and admission/stages.
- `part-B-001-supervision/`: original card/source-bound scripts, preflight,
  stdout/stderr, worker identity and `closure.json`.
- `part-B-001-manifest.json`, SHA256
  `10f326c561ba799399df78a737672cde02c47edc2712299552770eea553081c9`:
  all 28 read-only execution/supervision/audit artifacts. Compact receipt
  `part-B-001-receipt.json` in this repository has SHA256
  `51709904f5d2d1da8c10cb442ce97b84e3f1d3f83145d2a3564be20e6ac7c68d`.
- `part-B-001-audit-002/report.json`, SHA256
  `d852f3184ac16a3263c3de29215f058c39ca98e293cc484e88c3b2d014956e17`:
  saved-evidence check passed in0.090321990s, peak25,204KiB.
  Existing `audit.py` there has SHA256
  `2e280d0e1965572baf34c80a05daa92c1759ab040fd078950c174a9837c524c1`;
  it authenticates saved files/fields only, using the unchanged admission reader.
  Its CLI is `audit.py --output <new owned audit directory>` under CPU1/nice10,
  4GiB and an external120-second timeout. No further execution is required to
  read the saved report.

Audit001 failed only because the verifier treated the synthetic comparison path
`/n_vs_frozen_count` as a literal JSON pointer. Its source/failure are preserved.
Audit002 maps that declared check to producer n and frozen200; the separate `/n`
check binds independent n and all 800 row keys are checked. The corrected audit
ran before the request for a narrow independent delta review arrived. The saved
exact delta was subsequently read and accepted independently, before sealing.
No scientific source, plan, original result or statistical run changed. One
incidental admission-reader pycache is explicitly retained in the inventory.
