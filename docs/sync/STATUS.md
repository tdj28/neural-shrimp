# SYNC status (coupled soft-token loops)

Exploratory run, requested on 17 Oct 2026. It is complete as of 08:10 UTC. The run card, amendment 1 and the results entry are in docs/research-log.md (search for "SYNC").

- Code: tools/sync_coupled_loops.py (check, sweep, transverse, periodic, analyze), tools/plot_sync.py, and tests/test_sync_coupled_loops.py (8 tests).
- Raw data: neural-shrimp-data/explore/sync/<run>/. Logs are in neural-shrimp-data/explore/sync/logs/.
- Receipt: docs/sync/sync-summary.json. Figure: paper/llm/figures/fig-sync.png.

## Runs

| run | command (all with nice -n 10, CUDA_VISIBLE_DEVICES=, OMP_NUM_THREADS=4) | status |
|---|---|---|
| check | `check --out check` | done |
| transverse | `transverse --c 0 0.05 0.1 0.15 0.2 0.25 0.3 --record 2000 --record-jvp 3000 --out transverse` | done |
| sweep-near | `sweep --start near --c-min 0 --c-max 0.30 --c-step 0.01 --record 3000 --out sweep-near` | done |
| sweep-refine | `sweep --start near --c-per-cell '{...}' --record 10000 --save-every 500 --out sweep-refine` (grids in amendment 1) | done |
| sweep-far | `sweep --start far --c-min 0 --c-max 0.30 --c-step 0.02 --record 2000 --out sweep-far` | done |
| periodic, periodic-synced | `periodic --pairs sweep:cell:c ...` | done |

Rebuild the receipt and the figure:
`python tools/sync_coupled_loops.py analyze --sweeps sweep-near sweep-refine sweep-far`, then
`python tools/plot_sync.py --refine sweep-refine --ts sweep-refine:1:0.165 sweep-refine:2:0.115`.

## Open items (not run)

- Exponents of embedded periodic orbits, to show the bubbling mechanism in cell 2.
- Multipliers of the periodic unsynchronized orbits and of cell 0's period-2 orbit.
- Several starts per (cell, c); added mismatch or noise, to control the roundoff-driven burst floor.
