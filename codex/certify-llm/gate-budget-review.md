# Independent synthetic gate-budget and launcher review

2026-10-13 UTC. No blocker found in the released implementation, concrete plan or minimal launcher extension. Independently ran **58 synthetic gate-budget and launcher tests: all passed in 0.51 seconds**, explicit process exit zero. No actual seed admission, large target-sized arithmetic, model evaluation, checkpoint load or launch was performed by this reviewer.

Reviewed source bindings:

- `codex/certify-llm/gate_budget.py`: `9a6d2a7036793c4113bb55b4b3fd5ff4d326b91ebe027b0ee93178fead358424`.
- `tests/test_codex_llm_gate_budget.py`: `eb45fd94af4e392078a365eb8389f7158bba8e8a5f395940b021ce9596b49568`.
- `codex/certify-llm/gate-budget-001-plan.json`: `5534931c4beec14d0d5c2b2a33a84373138639cdcf528aaccbdbf95bb36798f4`.
- `codex/certify-llm/gate-budget-design.md`: `cb6758b11d406b2d26e81c381abb5feed36cd400e66b40f50dac83240926d245`.
- `codex/certify-llm/launch.py`: `d0961aa8d624423e19e00fe8c3644c1b97bf78b6805482fb64621be536f4680b`.

The concrete plan passed strict schema/recipe/envelope validation and every one of its 14 current source pins matched. The existing continuation-004 plan was also revalidated with unchanged runtime pins. Per the parent's pre-freeze instruction, the inherited zero-dollar metadata was removed from the two unfrozen plans; no frozen file was changed.

Only the numerical seed's A/B arrays are decoded after full inventory authentication. Unused NPZ fields in the tests require pickle, so the successful allow_pickle=False path checks that target centers/images and other arrays are not accessed. Original A/B arrays stay read-only and are copied with hashes before synthetic construction. The seed source commit is cross-bound to the receipt and captured source bytes; this importer explicitly does not verify Git blobs. Seed numerical matrices are arbitrary proposal constants here, not target derivative evidence.

The dyadic perturbation is formed through exact rational arithmetic, rejects nonfinite inputs and mantissas longer than 1024 bits, and verifies zero radius and exact value before and after matrix assignment. Exact binary64 B conversion and synthetic J enclosure/radius checks are consistent. The synthetic local affine maps on disjoint boxes have the retained exact next-center images and constant A derivatives; the J boxes are deliberately wider. The unchanged conditional wrapper receives a full synthetic domain/enclosure/proposal contract, factored bounds and powers (1,2). Synthetic conditional failure is retained as a completed engineering outcome. It cannot become a target certificate: output/status/events are explicitly synthetic, certificate_passed=false, and model_evaluations=0.

Completed stages, original proposal copy, exact deterministic recipe, phase hashes and matrix precision statistics are retained before the full gate. Compact gate results precede streamed full-report serialization. Interrupted partial JSON is retained. Owned SIGTERM and ordinary failures attempt a failed stage/status record; a native operation that postpones signal delivery is still bounded by the outer launcher and its immutable timeout receipt. Existing stricter address-space limits are preserved. Test controls cover unstable synthetic gates, exactness refusal, malformed plans/input hashes, retained intermediate failures and partial serialization.

The launcher change only admits gate_budget with an explicit committed plan, CPU1, and exactly 1800 seconds. Its existing nice10, 18-GiB process cap, private-source snapshot and receipt workflow remain intact. Existing center_continue behavior is unchanged; its regression tests pass. The driver starts its timer before numeric imports, disables GPU visibility, sets CPU thread limits before numerical work and preserves a tighter inherited address-space cap.

This is a bounded resource measurement, not an upper bound on eventual proof cost. Full-width mantissa construction, signed dense products, all gate outputs and report serialization may consume substantial time or memory; timeout or allocation failure is an informative retained outcome, not evidence against the scientific claim. Shared resource admission remains the launcher's owner's responsibility. No launch is authorized by this review alone.
