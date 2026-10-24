# Qwen 0.5B maximum-context feasibility pilot

Prepared and approved 2026-10-24; launch pending validation. The human explicitly approved the quoted pilot ceiling on 2026-10-24; one rental only.

Purpose: measure whether the existing soft-token map can be evaluated at the pinned Qwen2.5-0.5B-Instruct context limit, and obtain a measured quote for a later 16-by-16 temperature/steering map. This pilot is not a chaos classification or approval for the full map.

Model revision: 7ae557604adf67be50417f59c2c2f167def9a775. Fixed prefix plus one feedback position must total at most 32768 tokens. Archive exact prefix text, token IDs and hashes; count formatting tokens and assert the final position, with no silent truncation. Use deterministic prose and the same final instructions for a short baseline and the long context. Repeated filler, if used, must be disclosed as a content confound; do not claim a pure prompt-length effect.

One A100 80GB, one-hour absolute resource lifetime including setup and retrieval; at most 40 minutes numerical work, at least 15 minutes reserved for retrieval/termination. No automatic rental retry. Cost and ceiling: see private ledger and conversation.

Before rental, implement and CPU-test bounded/chunked prefix prefill as needed: the current runner prefills the entire sequence and duplicates prefix caches by batch. Compare chunked versus existing prefix results on short inputs, verify attention masks/positions/cache reset, and retain failures. A large memory model is not evidence that the long prompt will fit. Begin remote work with one trajectory and qualify two-offset batch memory before expanding.

Remote sequence: validate short-input numerical agreement and deterministic repeatability; measure prefill time, peak memory and 128 feedback updates at the maximum context. If feasible within the remaining deadline, collect up to four parameter settings, at most 256 settling and 512 measured steps per setting, with two float32 offsets safely above the rounding floor. Coordinates will be bound to the existing parent-map axes before launch. Save states and growth histories incrementally with remote/local hashes. Do not shorten steps silently to claim a completed scientific map. Failure to fit or finish is a feasibility outcome.

Deliver measured throughput, memory, verified artifacts, exact termination and an extrapolated quote for a separately approved long-context map and short-context control. No stronger chaos claim from this short pilot. A later precision study must qualify effective normalization/rotary arithmetic rather than relying on dtype labels.

Implementation: float32, chunk size 256, 128 timed updates at short and maximum context; four optional cells at log10 temperature ±0.3 and steering 0/0.1, original philosophy direction. Numerical deadline 30 minutes, external tool timeout 35 minutes, controller runtime 40 minutes, absolute resource lifetime 60 minutes. Exact token-ID filler construction is disclosed in the output; it does not isolate length from content. Outputs are small (base embedding histories only), with per-stage atomic files and a final per-file manifest.
