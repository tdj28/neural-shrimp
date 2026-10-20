# Task 41 seven-file text supplement, 20 October 2026

The completed focused tree screen deliberately did not read seven JSON/JSONL
blobs above its 8 MiB per-blob limit. This separate supplement closes only
those named text gaps. It does not rerun that screen or any scientific work.
The seven identities and paths are literal in `public_large_text_gap.py`;
they total 114,608,770 bytes and are identical in the two frozen target trees.

Root statically read the complete new source before checkpointing. Before any
execution, the compact receipt gained the completed manual adjudication;
its size/hash pin was updated to that final version. The seven target rows
and coverage fields were unchanged. It pins
the compact prior receipt and category source, selects only the frozen
category definitions through the AST, and checks the seven exact path/object
joins before reading payloads. Output contains hashes, categories and line
locations, never matched values. It does not open `.env` files, inspect the
forbidden project directories, execute the former scanner, load models,
validate credentials against a service or repeat the general history scan.

One invocation is allocated after a verified remote source checkpoint:
offline CPU, nice 10, one numerical thread, 2 GiB worker address space,
120 CPU seconds, 170 seconds internal wall time and 180 seconds outer wall
time. Each blob is bounded at 32 MiB and their aggregate at 128 MiB. Output
is exclusive, limited to 16 MiB, and preserves ordinary failures. A hard
CPU or outer-wall termination could leave a partial file; such an attempt
must be retained and reported, not silently replaced.

The host `uv` launcher runs outside the worker address-space limit; the
worker applies its limits before payload reads. This follows the previously
documented host launcher constraint. Invocation:

```sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
nice -n 10 timeout 180 uv run --no-sync python \
  codex/release-audit-task41/public_large_text_gap.py
```

This remains heuristic screening. Binary/image text, arbitrary encodings,
PDF contents and third-party ownership are outside this supplement.
