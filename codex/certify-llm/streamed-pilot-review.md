# Independent 64-column streamed-pilot review

Reviewed 2026-10-13. No checkpoint load, model evaluation or launch.

Reviewed exact sources and plan:

- `streamed_pilot.py`: `3400b36477168ffbbc2ba9d922641e1627c18bd264d63f700ecff9f811de82fc`.
- `tests/test_codex_llm_streamed_pilot.py`: `c6b99a7f35c732f6d1e9452c45984ab33e917eef3820acd909cb7db34002915a`.
- `streamed_head.py`: `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a` (mathematical review in `streamed-head-review.md`).
- `streamed-derivative-001-plan.json`: `04f56aa29bf71c94a21e3cc75acc9845bb20ec096760bf366b022a61a6d6a98f`.
- `launch.py`: `6f18aade862735ecfbaa1fd2729b2cbeb7823263863c8b27aa754e3039b09259`.

No blocking defect found for this bounded engineering scope. The new driver remains separate from all original evaluator, cache and refinement sources. The launcher delta adds only the new driver choice and its mandatory committed-plan requirement; its existing one-thread rule, private source snapshot, process ownership, 18 GiB cap and one-hour watchdog remain active.

The concrete plan pins the completed 32-column baseline receipt, full step, actual input file and actual-ball data hash. Its seven explicit common-source pins must match both the baseline snapshot and current code. Admission rechecks the entire successful baseline receipt and manifest, its exact captured plan, runtime/backend metadata and map assets, then reauthenticates the original prefix/proposal/eight-column point lineage. It reconstructs the full nominal domain from the pinned prototype and requires equality of the resulting actual balls, witnesses and hashes to the original baseline. It checks all 32 basis vectors, all 32 full baseline columns and every original eight-column point overlap. This is stronger than accepting an equal nominal radius while silently changing decoded receiver balls.

The actual baseline admission was independently executed at 1024 bits with one Arb thread, using only imported metadata and saved artifacts. It passed all 51 baseline artifacts plus 22 prefix-pilot and 27 signed-point artifacts; all 896 actual state coordinates and all 32 baseline rows were retained. The exact reproduced actual-ball SHA-256 is `e12d0a97b12bf2c8f04856e780990380b28716e079ff9a3b57e740cb8ff3a0fe`. No checkpoint was loaded and no map was called during that admission.

The new directions are exactly e0 through e63 in all 896 state coordinates. The full input domain and direction matrix are persisted before model construction. The reviewed streamed subclass processes all 151936 vocabulary entries with 1024-row head chunks; no original state coordinate or vocabulary tail is dropped. The plan's chunk size is bounded and typed explicitly. The full 896-dimensional primal and 64 complete derivative columns are persisted before all 32 shared-column overlap comparisons. Failure retains the saved result and comparison. The other 32 columns receive complete finite/radius records, but do not acquire a prior comparison they did not undergo.

The local argmax diagnostic is bound to this actual prototype box and explicitly separated from any orbit statement. A non-strict label gap does not falsely become a rigorous label or a certificate. Timing records separate decoder time from the final RMSNorm, streamed head and return overhead; the latter is accurately described as post-decoder time. Actual chunk events and peak RSS remain available for later audit.

Independent verification: all 89 pilot/head synthetic cases passed, including incomplete and tampered baseline refusals, last-row comparison failure, full-output retention, immutable parent files, bad resource types and isolated `-I --help`. A valid streamed launcher request was intercepted before its first Git operation; seven invalid variants were refused before Git. No launch was performed. The approximately 1977-second estimate is simply twice the measured baseline forward time; changed operation ordering, chunking and contention may change runtime, memory and interval widths.

Even a successful run would concern one tiny prototype box and 64 of 896 derivative directions, leaving 832 directions and seven phases absent. The box remains centred on the original floating prototype and is not proposed as an isolating periodic-orbit box. No existence, uniqueness, primitive-period or attraction conclusion follows from this engineering comparison.
