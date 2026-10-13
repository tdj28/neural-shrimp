# Final assembly001: preserved admission setup failure

The final P8 proof computation did not reach any mathematical gate. The original
attempt from12998ba9/plan58b842f1 ended exit1 without timeout after136.625s,
peak523,092KiB. Point-image admission completed in29.293s; derivative admission
then refused with `ValueError('exactly one captured plan required')`.

Four explicit recovery shard identities, phase1 columns256/320/384/448, correctly
retain both001 and002 captured plan filenames with identical bytes. The original
command names the002 plan, but the adapter searches all captured JSON files by
hash and incorrectly requires uniqueness. The first affected ordered shard is
final-derivative-p1-c256-002. The source/metadata diagnosis does not decode or
recompute derivatives and establishes no mathematical failure of the target.

Original receipt8f681af49df216f645bde66079f44c2acda59bd05fe7dde2efc8ee169d9409e4
and every original artifact remain unchanged/read-only. Separate bounded audit
checked all301 artifacts and291 source files against original Git blobs, with
no native numerical decode. Manifest28f6a4ec08f580bd032932f11d55328ec869e9187192ce2eff8a195b8b537002,
report82c4f0030924de20ec9f4859acd28478a03689334b484a8902b4cfcc165350d6 and
closure6acb0374a6708562ed22d627ee6f8e287aac98d9d1e95d4e4c4e06bfe5223311 live in
/data2/neural-shrimp-data/codex/llm-p8-certificate/audits-final-assembly-001-failure.

Inference reaped tool session64516 with exit1 at22:03UTC13October2026 and verified
original worker934982/launcher934661 and all members of both former PGID/SID are
absent. The recorded attempt completed earlier; the delayed scheduled check is
not a longer compute duration. No retry or extension occurred. A prospective
command-bound plan-selection correction requires source review, explicit new
attempt002 plan/card/private checkpoint and fresh ownership/resources. All
numerical enclosures, domain/asset/evaluator pins and frozen gates stay unchanged.
No certificate or evidence against the orbit is claimed.
