# EXP-008 preflight and verification ledger

Date: 13 September 2026. Status: **unit-tested and outcome-free runtime
rehearsal; no EXP-008 neural trajectories generated at this entry**.

- All 249 repository tests passed in 12.15 seconds. New tests cover the
  old-map selector, regression/scaling, full action census, failed development
  coarse targets and shams, paired-step primary gating, and immutable seal
  ordering. The whole-driver test mocks integration and model fitting; its
  synthetic receipt is not experimental evidence.
- The exact 45-file bound source allowlist was copied to an otherwise empty
  temporary tree. Both new CLI entry points passed `--help` with the same
  Python interpreter under `-I -B`, without PYTHONPATH/PYTHONHOME. No host
  checkout imports, missing/extra source files, or bytecode writes occurred.
  `tools/check_information_bundle.py` repeats this source-only check.
- The plan validator confirms 24 origins, 3456 preview records and 7008
  targets. Independent selection, model, census and runtime reviews accompany
  the protocol. Final plan regeneration binds the actual completed sources.
- A synthetic-only 1728-row, 51+6-column fit took 0.03669 seconds on the local
  interpreter. A correlated synthetic ridge system (condition number
  2.47e6) agreed between NumPy's general solve and SciPy's positive-definite
  solve to maximum coefficient difference 5.49e-13, within the prospectively
  fixed model-replay tolerance. This benchmarks algebra and fit overhead;
  it does not measure new neural throughput or prove arbitrary conditioning.
- NumPy 2.5.3 and SciPy 1.18.1 are pinned and checked before execution. Native
  compiler/library hashes and the actual Python/platform versions are recorded
  by the runtime. The previous measured EXP-007 workload supplies the neural
  runtime estimate; no future target was used as a smoke test.

The complete source/plan must be committed and pushed before the exclusive
attempt starts. The execution receipt will supply that exact commit and plan
hash, costs, census and outcome status. No paid review or RunPod resource was
requested or used. Private exploratory status remains unchanged.
