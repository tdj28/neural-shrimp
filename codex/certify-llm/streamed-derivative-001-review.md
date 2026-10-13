# Independent result audit: streamed-derivative-001

13 October 2026. Accepted as a completed prototype-box engineering measurement.
This audit read only completed saved evidence and source; it did not load model
weights or replay a target map. It used one CPU thread, an 18 GiB address-space
limit and a 120-second alarm. The main audit finished in 9.353 seconds after
imports, peak 623,860 KiB.

## Immutable evidence

- Receipt SHA256: `2770cc1aa1e46739ec8ebc85c8e5ea814a45eab1a0810f0d66692bc9e00cb81d`.
- Source commit: `d5b712381a0a3c612c00ea3b693d0ba7a456c092`.
- Driver SHA256: `3400b36477168ffbbc2ba9d922641e1627c18bd264d63f700ecff9f811de82fc`.
- Streamed head SHA256: `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a`.
- Plan SHA256: `04f56aa29bf71c94a21e3cc75acc9845bb20ec096760bf366b022a61a6d6a98f`.
- Full step SHA256: `d5782830d1cb5737ec7cde44fef9978d7caca17670d3f51bc252b0712c812154`.
- Original actual-input-ball hash: `e12d0a97b12bf2c8f04856e780990380b28716e079ff9a3b57e740cb8ff3a0fe`.

All 67 artifacts match receipt sizes/hashes, with no missing or unaccounted file.
All 52 captured source files match both the source manifest and their bytes in
Git at the stated commit. The committed plan, producer source hashes, runtime
versions, asset pins, exact prefix IDs, 1024-bit/CPU1 context, restored-prefix
source/hash/receiver binding, original proposal lineage, and resource envelope
agree. The receipt reports exit 0, no timeout, within CPU1/18 GiB/3600 seconds.

The unchanged saved-data baseline loader was independently rerun: all 51 baseline
artifacts, 22 original-prefix artifacts and 27 signed-point artifacts passed
admission. All original eight point-column comparisons were rechecked there.
The new and baseline original actual-domain records are exactly equal, not just
overlapping reconstructed boxes. Every original input endpoint contains the
specified exact phase-0 centre plus/minus the nominal radius 2^-800. Source
phase, plan, proposal, actual-input file and direction hashes are bound correctly.

## Every retained result and comparison

Every one of the 64 declared positive basis vectors was checked entry by entry:
896 exact values, one unit entry and all other entries zero. All 57,344 signed
derivative entries and all 896 primal entries have valid finite dyadic midpoint/
nonnegative-radius records. No column or output coordinate is omitted from its
declared scope.

Independent exact-Fraction endpoint comparisons passed for every shared entry:
**32 x 896 = 28,672 derivative overlaps, and 896 primal overlaps**. These used
original saved intervals; outward decoding cannot repair a failed comparison.
All 64 per-column finite/radius summaries were checked, including the 32 newly
measured columns without a baseline counterpart. The complete saved comparison
record also exactly matches recomputation. The maximum derivative radius is
approximately `6.115284398459191e-75` (column 53); maximum primal radius is
`5.255700445185481e-79`. These decimal summaries are diagnostics; the raw dyadic
balls remain the evidence.

Events contain exactly 24 decoder starts/ends, in the correct interleaved layer
order. Each head pass has exactly 149 ordered, nonoverlapping vocabulary chunks:
all 151,936 rows once in the scalar-logit pass and once in the signed-derivative
pass, including the final 384-row chunk. No token truncation, gap or duplicate
chunk appears. This checks execution evidence against the independently reviewed
head's full-vocabulary loops; it is not a fresh evaluation of the checkpoint.

## Measured time, memory and label scope

| Quantity | Measured value |
|---|---:|
| Forward calculation | 1787.968361 s |
| Decoder layers | 1133.412820 s |
| Post-decoder stage | 654.533618 s |
| Complete driver | 1809.941724 s |
| Complete launcher receipt | 1810.667844 s |
| Peak RSS | 5,393,488 KiB = 5.143631 GiB |

The post-decoder stage includes final RMSNorm, streamed head and Python return
overhead; it is not a separately isolated head-only timing. Compared with the
previous 32-column run, this 64-column forward used 90.43% of twice its forward
time and 64.86% of its peak RSS. This is a single measured comparison, not a
throughput law or guarantee for 128 columns, other phases, or different boxes.

The local all-vocabulary argmax diagnostic proposes token ID 374. Its retained
candidate lower bound minus maximum competitor upper bound has a strictly
positive margin, approximately 5.807867468749899. The record includes all
151,936 vocabulary rows and 151,935 competitors; its exact margin arithmetic,
strict sign and original actual-domain binding check out. The complete scalar
logit vector is not separately retained, so this audit verifies the saved margin
against its saved extrema and the reviewed all-token evaluator, not by rerunning
or sorting all logits. No tokenizer-string inference was made.

This label applies only to the original **prototype input box**, conditional on
the reviewed map/evaluator contract. It is not a label certificate for the
unknown exact periodic orbit. Likewise, 64 columns at phase 0 leave 832 state
directions and seven other phases unevaluated. No nominal final-orbit boxes,
complete Jacobian, root inclusion, primitive period or attraction gate is supplied
by this attempt; `certificate_passed` correctly remains false.
