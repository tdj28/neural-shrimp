# Independent continuation and completed-capsule review

2026-10-13 UTC. Scope: source review and CPU1 synthetic tests only, with no model/checkpoint loading, target evaluation, real-parent admission, Git mutation, or launch by this reviewer. No unresolved correctness blocker was found for the bounded one-candidate proposal workflow under its explicit external-audit trust boundary. This is not a periodic-orbit certificate or an authorization to launch an unfrozen plan.

## Reviewed source bindings

| File | SHA-256 |
|---|---|
| `center_continue.py` | `705e3c020c1a7c77f50a5e738d9d519fb428369f7f4865df00ae5f338e58f6ed` |
| `secant_parent.py` | `77c8ca3db2ee6f618eeb53d1991aaac5c921738e6c24e6dfbeb4813acf628704` |
| `secant_restore.py` | `c1daae4ad7ccc4e633701db3b4f308465fe519050faa3c8c90f538eafecf92ba` |
| `inverse_secant_fast.py` | `50a74a620b7744eb6ef4226664199969a2b84d64d1a607b63a1944195e59b026` |
| `parallel_phase_pass.py` | `2cdc3afd9e73d95857941034420d279db7f44b5177b82131cf5f1a44de6d2c43` |
| `phase_point.py` | `e87f1efdffd17925d8ec971f7ce164cfd5c1eaa1f0ad046cbc02916994ec1b4a` |
| `launch.py` | `d991f4f58ffe6072e183bc8506079342c84506f48ae080d139537c573e2b9ee1` |
| `test_codex_llm_center_continue.py` | `3a66b858da8b1ab40fa54f1d4708d4095e82b26b92a46f79fbabd7caeefe5eb8` |
| independent `test_codex_llm_capsule_admission.py` | `b942a527fe6f87afbbfb31978da9098ac4a2bfda9bb5cc530bac708b2a1c0084` |
| `center-continue-design.md` | `d98d383f6307b8e932e2434554bb2f87bf92ca388439fd6c262e13f292bde090` |

The authorized restore change explicitly imports the reviewed equivalent fast kernel. `install_factors` constructs separate matrices and exact factor copies; the controller checks the installed class and unchanged original A/B before and after update/proposal. This binds the small integration change that was still pending in the earlier fast-kernel review.

## Mathematical and evidence flow

The controller checks the admitted consecutive pair before constructing a kernel, consumes it once, retains the update decision, and proposes once. A rejected update clears history; a full four-factor history causes the original discard/restart behavior without re-admitting that pair. A proposal fallback clears the post-update factors. Separate before-update, after-update, and after-proposal counts and witness references prevent accidentally resuming the pre-fallback state. The new capsule's pending pair is the previous completed candidate and the newly evaluated candidate, not the pair already consumed by its producer.

Centers remain exact dyadics. Full original image balls are retained, their radii are checked before decoding, and residuals use the correct cyclic successor. Imported images are decoded outward and residuals freshly recomputed; the saved interval residuals and norm bounds are separately checked against original serialized endpoints. No image midpoint is substituted for its enclosure. The reader validates proposal residual binding, selected correction, midpoint candidate construction, clear/reason counts, and recorded step-limit decision. A complete regression is retained as a chronological result rather than silently replaced by a better earlier state.

Immediate evidence admission checks the entire parent receipt inventory, captured source inventory and current source hashes, one exact captured plan, assets, prefix/seed lineage, precision and arithmetic backend. The new point pass is independently bound to the expected map, backend, source commit and complete worker bundle, prefix/cache pins and exact center tuple. Original live input records are compared before any decoding. Every phase requires a successful canonical receipt and termination row, exact worker output inventory, all ordered decoder events, full finite output and zero tangents. Scheduler hash and frozen resource/scope fields are checked as well.

The future capsule is an explicit externally audited trust anchor. The reader authenticates all immediate evidence again and checks the immediate archive link against its producer's plan. It intentionally does not reread older archive bytes or replay historical dense inverse arithmetic. Older full images, prior factor provenance and audit conclusions are therefore trusted through the independently committed capsule/audit pins; the code reports `older_ancestry_rehashed:false`. The presence or hash of a Markdown audit cannot itself establish that its scientific review was performed. The human/controlling agent must only supply this pin after actual review. Restored factors remain arbitrary exact numerical proposals, never validated inverses.

The shared 1800-second deadline starts before non-stdlib imports and includes admission and proposal preparation. The launcher bounds the complete controller lifetime; the scheduler does not reset that deadline and requires the full 600-second worker reserve. Four independent CPU1 workers, per-worker address-space caps, sampled aggregate RSS/headroom checks, owned process groups, parent-death behavior, complete termination records and failure retention are covered by the separately reviewed scheduler/launcher. The aggregate memory check remains sampled, not a hard cgroup bound. No target timing or speedup is asserted by synthetic checks.

## Independent integration checks

The new independent fixture runs the actual proposal writer, frozen-request writer and final capsule writer, then calls the complete `load_capsule` reader. It uses all eight phase positions, one synthetic coordinate per phase, two synthetic decoder-event layers, nonzero tiny image radii, real file hashes, the actual full captured source inventory and actual worker/receipt schemas. No loader, source authenticator, request admission or phase collector is mocked. The upstream base-plan contents and older audit/source records are intentionally synthetic; this fixture does not claim to audit a real original prefix or checkpoint, nor does it exercise the driver's separate full base-plan importer.

Seventeen independent integration cases passed in 1.15 seconds. Coherently resealed corruptions test pending-pair identity, map/backend, exact factor radii and counts, immediate archive linkage, proposal residuals/correction/clear state, request center digest, inherited image narrowing, cumulative phase count, changed captured source and incomplete status. Separate cases reject a wrong external capsule pin and an unlisted extra artifact. The valid case retains all image/residual radii, post-proposal factors and the next unconsumed pair, and explicitly reports its unread older ancestry.

The final combined suite passed **240 tests in 6.76 seconds**, process exit 0, across the independent capsule fixture, continuation, restore, original/fast kernel equivalence, scheduler, phase worker and launcher tests. This includes tiny real-process scheduler tests but no model evaluation. The root's separate real initial-parent admission is complementary evidence reported by the root, not repeated or independently claimed here.

## Refinements resolved during review

The implementation now checks the scheduler digest/envelope, strict archive-link schema and its immediate-parent match, complete proposal residual binding, and request-center binding. These were review requests and are present at the hashes above. The prior gap in future-resume coverage is closed by the complete eight-phase synthetic writer-to-reader round trip; real future continuation evidence still requires its own completed-result audit before reuse.
