# Task8 Part A merge and ordering review

13 October 2026. Scope: the merge source at `0a3376cd22a0630028d7deef71d0ae3c4767746e`, its relationship to frozen analysis `9805ed32055e7ef790df0cc67316357c1067aef0`, and exported provenance metadata. This is not a numerical replay. No numerical spectrum/readout artifact payloads, question text, outcome arrays or model/cache files were opened; no producer analysis, tests, model or pod was executed/accessed. The authorized 20:56 announcement disclosed producer results; those are exposure, not independently reproduced evidence.

**Final source/metadata conclusion: no remaining blocker in this narrow review.** The disclosed selection rule, original archive-header metadata and exported 64-question manifest are consistent. Part A followed an explicitly amended generation order; it must not be described as having satisfied the original predictor-before-readout-generation rule. The original generation order was explicitly amended; numerical reproduction remains pending.

## Selection and frozen analysis

`merge.py` lines 39–54 hashes each original archive and checks its receipt's local hash. Lines 60–73 sort the run-folder UTC stamps, identify a completed copy by the presence of `fixed-spectrum.json`, choose the first such run and record every unused complete copy. Lines 75–84 copy the selected question's files byte for byte from that archive and hash the copied files. The rule never inspects a spectrum's residual, Ritz values or downstream outcome when selecting between copies. In particular, it does not prefer a later copy because that copy passes the scientific gate.

"Complete" in the merge is a file-presence rule, not convergence, all-ARPACK-success or scientific qualification. The unchanged frozen `items` path independently assigns a disposition to each requested question, checks the fixed-point residual and computed finite nonempty Ritz union, requires the spectrum token hash to match **every** requested readout arm, and uses the same admitted cohort in all four arms. It refuses extra spectrum indices and malformed/duplicate/missing readout cells. The merge manifest's 64 assignments therefore do not independently establish 64 `ok` dispositions; that remains the pending authorized replay's job. Partial ARPACK sets are governed by the frozen finite/nonempty-union criterion, not retrospectively used to choose a backup.

An independent metadata-only check of manifest SHA `cd892c102bcdd3b6f9728d740aca68a38f57bd42649088e62c23ec9eae09fa22` found exactly q0000–q0063, no missing assignments, eight unique receipt-matched runs, a selected `fixed-spectrum.json` hash for each, and precisely these selections:

| Run stamp | Selected questions | Unused complete copies |
|---|---|---|
| 151945 | 0–15 | none |
| 175506 | 16–23 | none |
| 175521 | 24–31 | none |
| 175536 | 32–36 | none |
| 184229 backup | 37–39 | its copies of 35 and 36 |
| 175551 | 40–47 | none |
| 175606 | 48–55 | none |
| 175621 | 56–63 | none |

All stamps are 2026-10-13 UTC run names. The sealed original header metadata confirms incomplete q0016 in the original run and q0037 in the slow completion run; neither incomplete copy is selected. The slow run's selected copies of 35/36 precede the backup in the frozen stamp order. The reported local extraction clash does not determine selection: the code reads the original archives, not whichever loose files happened to occupy the shared extraction directory.

## Ordering: generation differs from inspection

Frozen design lines 68 and 84 require the predictor to be computed first and the outcome second; its Rules section says spectra are saved before the readout run starts. Amendment 1, lines 201–215 of the amended design, explicitly records that the initial predictor timed out after 16 complete spectra, the readout run then generated all 64×4 cells, and spectra 16–63 were completed afterward. It attests that no readout array or residual had been opened and preserves the analysis, cohort rule, statistic, bootstrap and criterion. Amendment 1a, lines 217–226, fixes the earliest-launched completed-copy rule before reported outcome inspection. The 19:33 communication repeats that no Part A outcome or prefix had been read and promises analysis only after merging.

The amended design is the unchanged frozen document with these amendments appended, as independently checked from its bytes. Thus the original generation-order condition was changed transparently, not met retroactively. The stronger statement that no outcome was inspected is a producer attestation: source, hashes and directory stamps do not independently prove absence of prior access. The code can establish that the merge selection itself does not consult outcome values; it cannot establish everything a person or other process did. This is not evidence of leakage, and it does not require a rerun merely because raw outcomes existed earlier. The appropriate audit status is “original generation order amended; reported outcome blinding retained,” not an unconditional original-order-pass Boolean.

Suggested concise methods wording: “Following a disclosed timeout amendment, missing fixed-point spectra were completed after readout generation and before reported outcome inspection. Duplicate completed spectra were selected by a fixed launch-order rule, without inspecting their numerical values. The frozen numerical admission criteria and analysis were unchanged.”

## Source/metadata identities and limits

- Original export manifest: `dec6ee10d1542350e260de1ccd27e5203553e83651512eb3ecb0358c778f0543`.
- Merge source: `54604d8828502012e8294c4b60bd0fb2dac6f3c09894d2cad4deeab72cf19649`.
- Frozen analysis source: `a38586151f9acf6e60c8d355df22d44d8b8ca7aec78270a6f6e73cfc6b838dc2`.
- Frozen design: `4384d02509a696b978c3c79ac1b80eac5cc3320bece17342e73c94ce6feb3a3d`.
- Amended design: `2b5bc95263fff1f291c00fe457d0eef0850c5287609c6120c780eb53295a90dc`.
- 19:33 declaration: `044f0a485e8648eeae7db9e789eadc64a23b074e18f8be3e8a29998c0d7380d7`.
- Original receipt metadata export: `44e03584b4a48e09f289e53aa5ddb439c80da808b82f4ce61c01d6b54337814f`.
- Metadata checker: `/tmp/task8-partA-merge-metadata-check.py`, SHA `c46c27e4b6088939e0725f52c6965c98e530fbb751ae80051897ba09a7311304`.
- Checker report: `/tmp/task8-partA-merge-metadata-check.json`, SHA `b1dd8ba3887e013687f07fe91e8e16e8d2653e81ce1bd5c697439c3fc7e3fabc`.

The initial supplement mislabeled the development-only `huginn_item_level.py` as the spectrum source. It is not the Part A fixed-spectrum producer; no inference about test outcome access is drawn from that development script. The corrected predictor export is `tools/explore_latent_loop.py`, SHA `904e37f3409c1c8718258f23f76e98825e54a2971b2abe09f9a537c8005a32d3`, under correction-manifest SHA `f5b6091b7bb32793c02ffbd93ca88ea80a770acdfb2a021fe00452ebcb63f7d5`. Its bytes and the evaluator's bytes are identical at 9805ed32, cb4b7e10 and 041e9822, independently verified from all three exported copies. The only change to `explore_llm_attractors.py` is an unrelated OLMo model alias. The runner adds a spectra-only `huginnS` branch with the same fixed Huginn flags/defaults; it is not byte-identical in its entirety. `fixed_spectrum`/`spectrum_at` (lines 881–984) read model and gold token inputs, not saved Part A residual/readout artifacts. Their own fixed-point token readout is a generated diagnostic, not an input taken from the outcome run.

The sealed metadata authority is `/data2/neural-shrimp-data/codex/huginn-independent-replay/part-A-input-authority-001`, manifest SHA `304742d9c440f515575aea60a170f7f72ffb81674b8376eefa5e5035b37ac356`. The coordinator's report states that all eight original archive hashes, 64 selected spectrum byte copies and the single original readout member match. This reviewer authenticated the permitted metadata files against that manifest, independently derived every selection from their original archive headers, found no duplicate member names, and matched all 64 assignments and both unused backups. No selected numerical payload was opened and raw archives were not rehashed a second time by this reviewer.

The recorded `started` order agrees with the run-stamp order. Safe ledger fields record original retrieval at 17:53:00Z, before completion pods were created from 17:55:07Z; the slow pod was created at 17:55:40Z and the backup at 18:42:30Z. This corroborates the selection chronology and that the original readout artifact existed before the completion jobs. It does not timestamp a person's first inspection. Recorded termination events cover all eight pods, with the last at 20:53:50Z. This is retained lifecycle evidence, not a fresh remote status check.

The metadata authority explicitly does not attest historical upload-content hashes or remote model bytes. Those limits remain disclosed; no completeness or no-access claim is inferred from hashes alone. Its raw ledger metadata must stay outside Git because it contains monetary fields. The review checker emits only a strict time/event/pod/source allowlist. The initial source-only export and mislabeled development file are preserved, not rewritten. Actual numerical field reproduction remains the separate authorized replay obligation.
