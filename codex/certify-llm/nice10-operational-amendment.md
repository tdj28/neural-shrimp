# 13 October 2026: exact nice10 operational amendment

Source-only prospective correction, requested by Root before any remaining-111 derivative campaign. The task handoff (`docs/handoff-codex-2026-10-12.md`, lines45–47) requires own CPU processes to run at nice10. The earlier dispatcher design and independent source review described a nice0 controller to compensate for the launcher's additive `os.nice(10)`. That exception was not justified by the task policy. Those historical documents and their hashes are preserved; this dated amendment supersedes only their priority exception.

## Exact change

`launch.py` now has a small `ensure_nice10` helper. At launcher entry, before Git/snapshot subprocesses, it reads the inherited nice value. A value below10 is increased to10,10 is unchanged, and a value above10 is refused rather than attempting a privileged priority increase or accepting a different exact envelope. It rereads and requires exactly10. The worker's existing pre-exec callback calls the same helper before the unchanged18GiB AS limit, so inherited nice10 remains10 rather than becoming19. The campaign controller must itself be launched at exact nice10; it refuses other priorities before creating output or performing admission. All other source, resource, snapshot, ownership, timeout and proof logic is unchanged by this amendment.

This corrects the control-plane priority and does not alter the mathematical evaluator, derivative equations, arithmetic precision, cache, domain or native input. No active process is signalled or reniced. In particular Root/inference retain ownership of first worker454658 and its immutable captured1f1de536 source; that run and original source must remain untouched. Its eventual independent audit must use the original captured Git blobs. Prospective launcher hashes are not substituted into old evidence. The first result may serve as the measured campaign authority only under the existing exact evaluator/domain/prefix equality and separate independent audit obligations.

## Source identities

| File | Before | Prospective SHA-256 |
|---|---|---|
| `codex/certify-llm/launch.py` | `07b9afff4236d71cca476b52896406710865344f7566208742d165a28fa40827` | `7c6fd77462e856f8f05753f0eb8b8c5c831a4a2ba43ac2daa692de3558371e81` |
| `codex/certify-llm/derivative_campaign.py` | `0cc8a192737508b449472eb7d20562c6d84abf84b07565de7e0108d712cea161` | `5266776c8618a27d6c97772e4aa7018983b5bb1407872fbb315b9f7fc17542a9` |
| `tests/test_codex_llm_nice_policy.py` | new | `aec6bf48968780d46a5e0f723fa85718430979e80167cf97bf626d3e41baea68` |

Unchanged derivative driver SHA: `3fe44a583176e0551bb209093fec08cd059d055fcb1a57702ac8799c7cb7cc50`.

Preserved historical design SHA: `d6f3d75848f0cdc3783685901d8e40be8b7d07ad92ff77ae24d3dffbcce11179`. Preserved historical independent review SHA: `84ef3816c432fb3703f26414bd4698e6cb5b881dd55b985649dd83dccfcced05`. Existing dispatcher tests remain `194115a938806b6e53be0a4c9e94ae870cdff48c6861e5cc01cad09702e509a3`.

## Checks and remaining issue disclosed

Nine new focused controls cover inherited nice0/10/12, idempotent parent/worker behavior, failed establishment, controller refusal and exact priority before mocked Git and the real launcher's captured pre-exec function. These do not change any live process priority. Together with existing launcher and dispatcher controls: **59 passed in6.44 seconds**, CPU1/nice10, no model, target artifact, Git network or target launch.

The preceding combined run had58 passes and one safe lifecycle failure: an immediate `/proc` command read after synthetic `Popen` transiently differed. The dispatcher stopped, later recovered the exact launcher command and cleaned up its owned children; it did not dispatch further or falsely certify completion. The complete tiny-process failure remains at `/tmp/pytest-of-d7082791602/pytest-1190/test_real_owned_overlap_sequen0/campaign`. The identical source passed on the next run. This startup-admission race is disclosed to Root separately; no grace/identity relaxation is hidden in this priority amendment. Root may request a separate bounded registration fix.

Inference remains the sole Git owner. These prospective files are unstaged pending Root and independent review. No campaign card or actual first-shard audit was invented, and this correction does not authorize the111-shard campaign.
