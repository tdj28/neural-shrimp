# Original loader supplement closure

The separately read original initializer supplement is /tmp/task26a-loader-supplement-001/manifest.json, SHA fcfd7216de9cbb86e4346c2746dd1452b10f2442e34e2e53f8e3edfb638dcfc1. It binds original source a9c411ff1bf992a20f595039000c55be15c6d12c: SoftLoop lines 38–66, SteeredModel lines 80–97 and Engine lines 136–194. The complete original keep_float64_norms implementation remains in the first packet.

SoftLoop sets state_dtype to float64 only for that request and float32 otherwise; the expected-embedding table is retained separately from the later model bfloat16 cast in repro_task.load. The float64 loader calls the norms/rotary patch and checks supported attention implementations. This chain is consistent with the declared implementation and pre-input-cast distance meaning. It is not a new numerical precision test or historical loaded-weight byte attestation.

The supplement confirms that no separate saved inline statistical implementation was located and that the source-bound smoke statement does not identify its held-out index. The accepted review and prospective RNG disclosure remain unchanged. No target records, smoke outcomes, model, producer execution or new precision experiment were accessed for this note.
