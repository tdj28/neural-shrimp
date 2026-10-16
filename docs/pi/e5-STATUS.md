# E5 status (2026-10-16 21:25 UTC, written when the session had to stop)

Run card: docs/research-log.md "Run card E5" (aecac5d5). Tool: tools/e5_repetition_onset.py; tests:
tests/test_e5_repetition_onset.py (36 pass). Raw outputs: /data2/neural-shrimp-data/explore/pi/e5/.

## Done
- Tool and tests committed before any development run (425716a8); decision rubric committed before any development
  outcome (docs/pi/e5-decision-rubric.md, 1372b309); later analysis-only and receipt code, buffered sampler, compare.
- precision-1: float64 effective precision confirmed (central differences fall about 100-fold per decade of h to about
  1e-8 at log10 T = 0.11 and 0.37; batch-position repeats bitwise identical).
- dev-loop-1 (8 dev prompts x 24 T, full procedure, virtual retirement): 177 periodic, 15 chaotic, 0 irregular; all
  gates pass; 175 of 192 cells virtually retired, 0 label mismatches.
- dev-sample-1 (16 samples x 256 tokens per cell); dev-sample-buffered-1: buffered sampler re-run of prompt 0, all 384
  samples bitwise identical (also the sampling repeat control).
- analysis-dev-1 (card rule), analysis-dev-2 (chosen rule).
- Choices committed before any held-out run (docs/pi/e5-choices.json, 3a667908): --retire for the held-out loop;
  repetition rule period <= 48 (card's 32 degenerate: 4 of 8 dev prompts censored, 1 of 8 with 48); 2 nats kept;
  12 samples per held-out cell (projected 7.8 h > 7 h); buffered sampler.
- heldout-loop-1 (24 prompts, --retire): 529 periodic, 43 chaotic, 4 irregular of 576 cells; all gates pass; no
  supplementary cells. Never chaotic (censored): prompts 16, 25, 32, 41, 47, 48.

## Running
- heldout-sample-1 (script logs/run_heldout.sh, log logs/heldout-20261016T1935.log): 24 prompts x 24 T x 12 samples,
  about 466 s per prompt, 3 of 24 done at 21:23 UTC; files samples-pNN.npz. The run resumes per prompt (existing files
  are skipped), so if it was stopped, rerun the command below with the same --out.

## To finish (6 threads, CPU only)
    cd /data2/neural-shrimp
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    T=tools/e5_repetition_onset.py
    nice -n 10 uv run --no-sync python $T sample --split heldout --samples 12 --sampler-impl buffered --out heldout-sample-1 --threads 6
    nice -n 10 uv run --no-sync python $T analyze --split heldout --loop-runs heldout-loop-1 --sample-runs heldout-sample-1 --choices docs/pi/e5-choices.json --out analysis-heldout-1
    nice -n 10 uv run --no-sync python $T analyze --split heldout --loop-runs heldout-loop-1 --sample-runs heldout-sample-1 --out analysis-heldout-cardrule-1   # card's rule, sensitivity
    nice -n 10 uv run --no-sync python $T summarize --runs precision-1 dev-loop-1 dev-sample-1 dev-sample-buffered-1 analysis-dev-1 analysis-dev-2 heldout-loop-1 heldout-sample-1 analysis-heldout-1 analysis-heldout-cardrule-1 design-probe-eos logs smoke1-loop smoke1-loop-retire smoke1-sample smoke1-analysis --dev-analysis analysis-dev-2/analysis-dev.json --heldout-analysis analysis-heldout-1/analysis-heldout.json --template TEMPLATE.json
  TEMPLATE.json: text fields (card, label, deviations, caveats, choices); drafts of the deviation list are in the
  session report. Primary endpoint: analysis-heldout-1 primary.spearman with primary.bootstrap.

## Development findings (provisional, 8 prompts, exploratory)
- With EOS unmasked the model ends its reply within 9-40 tokens and writes unrelated documents (design probe), hence
  end-of-turn masking. Low-T loops are mostly verbatim sentence loops of 33-48 tokens; list-like replies (prompt 2)
  never loop within 256 tokens.
- Dev, chosen rule: Spearman(T_chaos, T_rep) = 0.29 (95% bootstrap -0.75 to 0.91); partial given T_H -0.12;
  Spearman(T_H, T_rep) = 0.71; matched-temperature chaotic minus periodic repetition fraction -0.038 (-0.11 to 0.04).
  Too few prompts to conclude anything.
