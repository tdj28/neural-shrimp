# C12-3 status (coupled agents; exploratory)

Run card: docs/research-log.md "Run card C12-3" (commit 190045db). Tool: tools/c12_coupled_agents.py; tests:
tests/test_c12_coupled_agents.py. Raw outputs: /data2/neural-shrimp-data/explore/pi/c12-3/<run>/ (run.json,
records.jsonl, msgs/), logs in .../c12-3/logs/.

## Done
- dev-0.5b-1, dev-1.5b-1 (phase 1, 16 dev problems): model rule chose qwen1.5b (0.547 vs 0.227 independent accuracy);
  docs/pi/c12-3-choices.json committed 080af302 before the held-out run.
- smoke-1.5b-1 (48-row layout, before the c = 0.5 arms were added) and smoke-1.5b-2 (final 64-row layout): gates pass.

- test-1 (qwen1.5b, 64 held-out problems, both phases) complete; analyses written for test-1, dev-0.5b-1, dev-1.5b-1,
  smoke-1.5b-2; receipt docs/pi/c12-3-summary.json; results in docs/research-log.md (C12-3 results, 04:45 UTC 17 Oct).

## State: COMPLETE. Possible follow-ups: asymmetric or delayed coupling at strong c (so locked agents do not receive
identical inputs), the text-only arm, the equal-compute single solver, the readout-24 vs 48 accuracy reversal.

## Plan / to resume (6 threads, CPU only; every run resumes per problem and phase)
    cd /data2/neural-shrimp
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    T=tools/c12_coupled_agents.py
    nice -n 10 uv run --no-sync python $T run --model qwen0.5b --split dev --phases 1 --out dev-0.5b-1 --threads 6
    nice -n 10 uv run --no-sync python $T run --model qwen1.5b --split dev --phases 1 --out dev-1.5b-1 --threads 6
    uv run --no-sync python $T choose dev-0.5b-1 dev-1.5b-1        # -> docs/pi/c12-3-choices.json; commit before held-out
    nice -n 10 uv run --no-sync python $T run --model <chosen> --split test --out test-1 --threads 6
    uv run --no-sync python $T analyze test-1 ; uv run --no-sync python $T analyze dev-0.5b-1 ; ... dev-1.5b-1
    uv run --no-sync python $T receipt smoke-1.5b-2 dev-0.5b-1 dev-1.5b-1 test-1 --template TEMPLATE.json
