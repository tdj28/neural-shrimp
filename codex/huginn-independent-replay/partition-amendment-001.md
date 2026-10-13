# Task8 operational partition amendment 001

13 October 2026, after Claude's 17:58 UTC completion announcement. This changes
how Part B's externally pinned row files may be packaged. It does not change
any scientific input, cohort, arm, statistic, threshold or comparison rule.

The announcement at commit `27f1a49a33ba83de32b7f5cc4a161690bd5aff1e`
says the original `partB-0` job reached its 130-minute limit after problems
0–95. It identifies `partB-96` as completion of problems 96–99 and `partB-100`
as problems 100–199. These are the three declared disjoint partitions of the
same 200-problem, four-arm panel. The original limited attempt remains part of
the evidence; this amendment does not relabel it a completed 100-problem job.

The reviewed source-only announcement export is
`/tmp/task8-partB-completion-authority-announcement-20261013`:

- Manifest SHA256: `23b58e854c4bb88ead2ae2a6f48043c109cb13b902513ae97a385aa6b48f0a3f`.
- Assignment SHA256: `25a73cbb65e70731071560f9a341a4ee0a6151d91de2e8d8297c5eeb5842e442`.
- The exported completion evaluator at `705e8dc9dd8e6ced6bc4a1a8b01b8d6ddf3da34e`
  and frozen evaluator at `9805ed32055e7ef790df0cc67316357c1067aef0` both hash to
  `fc86d5a68987b77013c5b285eeddf0cee41213215cadf8f447e61bd886f398c6`.
  This verifies those exported file bytes, not all historical execution.

The assignment includes a producer B1/B3/B2 numerical summary. Root and the
coordinator had already read it; this amendment's author also read it as part
of the exact assignment. Those numbers are prior exposure, not independently
reproduced findings. No target archive, row, result file, model/cache file or
pod was accessed, and no producer code was executed for this amendment.

## Narrow source change

`admission.py` previously required exactly two B row-file roles. It now requires
an externally pinned, nonempty list of unique nonempty string role IDs. Part A
still requires exactly one row file. Existing unique physical-path, role-file,
archive-member, byte-hash and externally reviewed receipt checks are unchanged.
All listed partitions are read; the reader does not discover, choose, omit,
retry or reslice partitions based on their outcomes.

The unchanged reducer still requires every index 0–199 in each of arms
1.0, 1.05, 1.15 and 1.20 exactly once, with the frozen metadata. Missing cells,
duplicate cells, extra cells and incomplete prefixes fail. A byte-authentic
partition list is not by itself evidence of complete scientific coverage.
The actual manifest must enumerate and pin all three declared partitions and
their original retrieval/termination/source authorities. The original timeout,
completion attempt and aggregate coverage must be described honestly in that
external authority; no actual input manifest is admitted here.

This prospectively supersedes only the sentence “Part B takes exactly two
disjoint full producer partitions” in `execution-design.md`. That source-bound
historical review remains intact. It does not alter the complete-input or
scientific-contract requirements in that note or `plan.md`.

## Verification and source identities

All **154** admission, statistics, comparison and runtime synthetic tests passed
in **3.04 seconds** with CPU1/nice10 and one BLAS/OpenMP thread. The 13 new
controls include a real synthetic-tar three-part admission followed by the
unchanged 200×4 reducer, missing-partition/missing-cell/duplicate-cell/extra-cell
refusals, empty/duplicate/non-string partition roles, Part A's exact-one rule,
and duplicate physical files. The synthetic reducer uses eight bootstrap draws
for these coverage controls; the actual replay's 10,000-draw default is unchanged.
No test failure occurred in this amendment's first run. Earlier fixture failures
are preserved in `execution-design.md` and were not repeated or overwritten.

- Prior admission SHA256: `18f94bbd73f9d20fb50b6d12d9d0d81b25ddb827821c2b7156e57f5b4d047081`.
- Amended admission SHA256: `7ed4cbc119fa42e286653de0831d91d705df2e2dbe02c21ba064e3db9054f4d4`.
- Amended admission-test SHA256: `32c8405bd26eeba3c2488440c9fe774c6cfeae872edc821b79d55ce7ce0bbe73`.
- Unchanged statistics SHA256: `0a76f6ad2949d2234aa560ec4e9267f4a3fe429332b1c33c1b85d3551498581c`.
- Unchanged replay SHA256: `8df093ddb55fcc58bf25dc4e20327fae6e4f031ec4c90aeb126bb3ddf0662469`.

Original source/test snapshots are retained under
`/tmp/task8-partition-amendment-001-before/`, and the exact source/test delta is
`/tmp/task8-partition-amendment-001.patch`. Root's review, a private checkpoint,
and actual complete input/resource admission still precede any target replay.
