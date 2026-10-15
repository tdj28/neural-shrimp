# Task21a — original tool ancestry supplement

This compact supplement extends the accepted source-only review `/tmp/task21a-source-math-review.md` (SHA256 `daa90f2ec9035d1ecd70df8700a73e1971beec423c5cb33bdac02e82aa3025be`), which remains unchanged. I compared the complete text deltas of the six original `llm_feigenbaum.py` snapshots against reviewed commit `44b22d3b43ac661df3a609abbeb49a86b3b1e3a2` (source SHA256 `884aa83ac14157453ad56f8fa724038ed59097d3cc6098efad0c49ee0801d82c`). No test, producer import, model call or saved numerical arithmetic was run.

The accepted metadata basis is `/tmp/task21a-input-authority-001/inputs.json`, SHA256 `52b0acd5d65e1201d6e283193872fc8543f27840ada830313535090e312b4bdf`; `original-authority.json`, SHA256 `172604af36c67db26f2137f3445ddc79260c2b01b46f068b2163a7bf0c82f729`; manifest SHA256 `7da862b31dec5b773e59b8923c54cb325d98d5014409f7cfd60cbd5c02a643b8`. The six sources are under that packet’s `source/<commit>/llm_feigenbaum.py`.

| Original source commit | Tool SHA256 | Metadata-bound attempts |
|---|---|---|
| `bb5bd4181d1e1debf7e1bef5cd04061f29f05c5d` | `3d1e3e37d9654d61be7f9bb98b2bae5fe7ded777d44b48c5268ecef90c342977` | FB152 and FB156 initial attempts |
| `5ad048f255732a210098fd7c7af724ccde3cdf0b` | `fbb0dc468bd340084825f71403a46ab1156a2b026bedbbbea3f943176e14f944` | FB152b and FB156b |
| `65aeba3fa0b2e9e97988920bf4318766ddfed551` | same `fbb0dc46…` | FB152d |
| `c8af6434d911b881656fbb0bf201fd6831beb30e` | same `fbb0dc46…` | FB152e |
| `c3da24dcea14187209c37f1f9c1afc3a4b8459ef` | same `fbb0dc46…` | FBS156 |
| `b5af43a7b3259b00e2e7cd573c097f999f6587f9` | same `fbb0dc46…` | FBD and FBS192 |

**Exact differences.** There are only two unique older tool byte sequences. Relative to the reviewed source, `fbb0dc46…` differs only by the explicit `g_accept` argument/plumbing/documentation and the CLI exposure/plumbing of `lam_rel_tol`. It retains the literal final acceptance test `abs(best.g)>1e−6`; its internal relative bracket tolerance defaults to 1e−13. It has no `--g-accept` or `--lam-rel-tol` CLI argument. All nine metadata-bound pod attempts therefore used the pre-change acceptance semantics; the FBD refusal cannot be recast as success under the later option.

The earlier `3d1e3e37…` source additionally predates the map-guided switching options. It fixes the initial ratio guess at 4.0, the later switch fraction at 0.5 and the continuation maximum at 16 substeps. `fbb0dc46…` exposes those three values while preserving their defaults. They alter search placement and reach, not the formula for the located flip ratio. Their actual nondefault choices must remain attached to each attempt’s recorded arguments and amendment history.

**Saved fields and gates unchanged.** The complete text comparison found no differences in `tol_for`, `chain`, low-rank `newton`, `ritz`, `Evaluation` and its `record`, `evaluate`, half-period-collapse checks, `switch`, `interval_ratios`, the superstable locator/serializer, or the successful cascade record fields. Thus the accepted review’s finite semantics apply unchanged: return residual uses the period-scaled tolerance; the reported next modulus concerns other computed Ritz values; progress logs contain newly evaluated locator points, not every continuation-walk multiplier; the best accepted value need not be a final-bracket midpoint; the superstable estimate is the secant of the clean signed endpoints and its `record` belongs to the positive endpoint. Neither older source measures multiplier scatter or returns a calibrated root-error interval.

**Local and pending limits.** `local-FB156c` is separately admitted from terminal `done:true` records plus the disclosed producer/source-card basis. It has no independently captured historical loaded-source commit, process births or exit receipt. Do not silently assign either older hash to it merely because its fields are compatible. The local `crescent-t-0.156-d` and `crescent-t-0.152-c` attempts remain excluded/pending for lack of terminal-completion authority; no numerical prefix was read in this review. Later attempts are not brought into scope by this source comparison.

Verdict: original saved-field and gate semantics are resolved at the accepted finite provenance level. The existing source/math caveats stand; no additional implementation review, test or model execution is required before the separately authorized saved arithmetic.
